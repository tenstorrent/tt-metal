# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chunk replay time of the traced chunked prefill vs position, through the model's own ``prefill_interleaved`` (4 users = 4 mesh rows x 1 slot, as the B=4 demo), on a LAYER SUBSET.

    PROBE_LAYERS=0,1,2,3,14,20,21,24 PROBE_LENS=16384,32768,65536 PROBE_WMAX=1048576 PROBE_MODE=matched|single pytest tests/test_chunk_probe.py -s

PROBE_MODE=matched: every prompt length gets its own S_pad trace (``s_pad_max`` None: the demo's behaviour); single: ``s_pad_max`` = PROBE_WMAX (one trace for every length; with
DSV41_PF_SINGLE=1 the new single-trace chunk, else the old per-S_pad layout captured once at the max context).
Logs ``PROBE`` lines: per length the replay count, mean ms per replay and the ms of the replays at chosen positions, plus the number of captured chunk traces."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "0,1,2,3,14,20,21,24").split(",")]
LENS = [int(x) for x in os.environ.get("PROBE_LENS", "16384,32768,65536").split(",")]
WMAX = int(os.environ.get("PROBE_WMAX", "1048576"))
MODE = os.environ.get("PROBE_MODE", "matched")
CHUNK = int(os.environ.get("PROBE_CHUNK", "512"))
REPS = int(os.environ.get("PROBE_REPS", "2"))
B = 4


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "650000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_chunk_probe(mesh_device):
    md = mesh_device
    log = lambda m: print(m, flush=True)
    max_ctx = WMAX if MODE == "single" else max(max(LENS) + 512, WMAX)
    pp = default_page_params(max_ctx, 1)
    args, m, _, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=LAYERS,
        log=log,
    )
    assert m.Up == 1 and m.U == 1, (m.Up, m.U)
    pm = m.prefill_model
    if os.environ.get(
        "PROBE_APC_GB"
    ):  # prefix snapshots as the server (DSV41_APC): the snapshot cost is part of the call timing
        m.enable_apc(int(float(os.environ["PROBE_APC_GB"]) * 2**30))
    if os.environ.get("PROBE_PROF") == "1":
        import cProfile

        pm.build_prof = cProfile.Profile()
    rec = []
    orig_rw = pm.replay_window

    def rw(tokens, hashes, s0, C, active, vis=None, **kw):  # (prepped / overlap / ops of the host-overlap path)
        th, td = orig_rw(tokens, hashes, s0, C, active, vis=vis, **kw)
        rec.append((s0, th, td))
        return th, td

    pm.replay_window = rw
    ncap = [0]
    orig_cap = pm.capture_dyn_trace

    def cap(C):
        ncap[0] += 1
        return orig_cap(C)

    pm.capture_dyn_trace = cap
    if (
        os.environ.get("PROBE_WARM") == "1"
    ):  # the server's warm-up (allocate / compile everything, then capture the chunk trace and the decode traces) before any prefill
        m.warm_serving(CHUNK, WMAX if MODE == "single" else 16384, [], [])
        log("PROBE warm_serving done")
    g = torch.Generator().manual_seed(1234)
    toks = torch.randint(1000, 100000, (B, max(LENS)), generator=g, dtype=torch.int64)
    if os.environ.get("PROBE_PROMPT"):  # real text (json list with a "prompt" field), the same for every user
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash"))
        import json

        ids = tk(json.load(open(os.environ["PROBE_PROMPT"]))[0]["prompt"])["input_ids"]
        assert len(ids) >= max(LENS), len(ids)
        toks = torch.tensor(ids[: max(LENS)]).long().unsqueeze(0).repeat(B, 1)
        os.environ["PROBE_SAME"] = "1"
    if os.environ.get(
        "PROBE_IDS"
    ):  # a saved 1-D token tensor, the same prompt for every user (LENS must equal its length)
        toks = torch.load(os.environ["PROBE_IDS"]).long().unsqueeze(0).repeat(B, 1)
        os.environ["PROBE_SAME"] = "1"
    SAME = os.environ.get("PROBE_SAME") == "1"  # every user gets the SAME prompt: their logits must agree
    if SAME:
        toks = toks[:1].expand(B, -1).contiguous()
    if os.environ.get(
        "PROBE_RAGGED"
    ):  # users at DIFFERENT chunk positions in one call (per-row chunk starts): every user's last-token logits must equal those of its own lone prefill
        E0 = int(os.environ["PROBE_RAGGED"])
        Es = [E0 - 512 * u - 37 for u in range(B)]
        cuts = [512 * (u + 1) for u in range(B)]
        assert all(c < e for c, e in zip(cuts, Es)), (cuts, Es)
        sp = WMAX if MODE == "single" else None
        refs = {}
        for u in range(B):
            for b_ in range(B):
                m.release_user(b_)
            refs[u] = (
                m.prefill_interleaved([(u, toks[u], 0, Es[u])], CHUNK, s_pad_max=sp, want_logits=True)[u][1]
                .float()
                .flatten()
            )
        for b_ in range(B):
            m.release_user(b_)
        t_ = time.perf_counter()
        m.prefill_interleaved([(u, toks[u], 0, cuts[u]) for u in range(B)], CHUNK, s_pad_max=sp, want_logits=False)
        n1 = pm.timing.get("replays")
        r2 = m.prefill_interleaved(
            [(u, toks[u], cuts[u], Es[u]) for u in range(B)], CHUNK, s_pad_max=sp, want_logits=True
        )
        log(
            f"PROBE RAGGED call timing: replays {m.timing.get('replays')} (per-row: max chunks of one user; grouped: sum over distinct starts) {time.perf_counter() - t_:.1f} s"
        )
        lg_r = {u: r2[u][1].float().flatten() for u in range(B)}
        for u in range(B):
            d = float((lg_r[u] - refs[u]).abs().max())
            pc = float(torch.corrcoef(torch.stack([lg_r[u], refs[u]]))[0, 1])
            log(
                f"PROBE RAGGED user {u} (cut {cuts[u]}, end {Es[u]}): max |logit - lone| {d:.4f} pcc {pc:.6f} top {int(lg_r[u].argmax())} vs {int(refs[u].argmax())}"
            )
        torch.save(
            {"ragged": lg_r, "lone": refs},
            os.environ.get("PROBE_RAGGED_SAVE", "/tmp/ragged.pt") + f".perrow{os.environ.get('DSV41_PF_PERROW', '1')}",
        )
        log("PROBE RAGGED DONE")
        return
    for E in LENS:
        USERS = [
            int(x) for x in os.environ.get("PROBE_USERS", ",".join(str(b) for b in range(B))).split(",")
        ]  # users in the call (the others stay idle: a lone request of the server)
        items = [(b, toks[b], 0, E) for b in USERS]
        dev_map = {}  # window start(s) -> device seconds of a window (per-window td of the synchronous run)
        cmp_snaps = {}
        cmp_ = (
            {}
        )  # PROBE_CMP=1: decode logits + decode index keys after every rep (reps differ in PROBE_ENV_REPS settings) must be identical
        for rep in range(REPS):
            for spec_ in filter(
                None, os.environ.get("PROBE_ENV_REPS", "").split(";")
            ):  # "VAR=a,b,c" -> rep i sets VAR to the i-th value
                var_, vals_ = spec_.split("=", 1)
                os.environ[var_] = vals_.split(",")[rep % len(vals_.split(","))]
                log(f"PROBE rep {rep}: {var_}={os.environ[var_]}")
            if os.environ.get("PROBE_APC_GB"):
                m.reset_prefix_cache()  # (every rep is a cold prompt)
            if os.environ.get("PROBE_RELEASE") == "1":
                for b_ in range(B):
                    m.release_user(b_)
            rec.clear()
            calls_t = []  # m.timing of every call of the rep
            if os.environ.get("PROBE_WL") == "1":
                pm.winlog_stop()
                pm.winlog_start()
            elif pm.__dict__.get("winlog") is not None:
                pm.winlog_stop()
            c0 = ncap[0]
            t0 = time.perf_counter()
            n_pieces = int(
                os.environ.get("PROBE_PIECES", "1")
            )  # the prompt in several calls (server chunk steps): every piece ends on a chunk multiple
            cuts_ = [E * (i + 1) // n_pieces // CHUNK * CHUNK for i in range(n_pieces - 1)]
            cuts_ = [c for i, c in enumerate(cuts_) if c > (cuts_[i - 1] if i else 0)]
            st_ = 0
            for (
                c_
            ) in cuts_:  # non-final pieces (the first token of a piece that does not end the prompt is meaningless)
                m.prefill_interleaved(
                    [(b, toks[b], st_, c_) for b in USERS],
                    CHUNK,
                    s_pad_max=WMAX if MODE == "single" else None,
                    want_logits=False,
                )
                st_ = c_
                calls_t.append(dict(m.timing))
                if n_pieces <= 32:
                    log(
                        f"PROBE call [{st_ - 0}) timing: { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.timing.items()} }"
                    )
            items = [(b, toks[b], st_, E) for b in USERS]
            res = m.prefill_interleaved(
                items,
                CHUNK,
                s_pad_max=WMAX if MODE == "single" else None,
                want_logits=(bool(os.environ.get("PROBE_SAVE")) or SAME) and os.environ.get("PROBE_NOLOGITS") != "1",
            )
            wall = time.perf_counter() - t0
            calls_t.append(dict(m.timing))
            log(
                f"PROBE last call timing: { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.timing.items()} }"
            )
            n = len(rec)
            td = [r[2] for r in rec]
            pos = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
            log(
                f"PROBE mode={MODE} E={E} rep={rep} S_pad={pm.dyn.S_pad} captures_this_call={ncap[0] - c0} captures_total={ncap[0]} replays={n} wall={wall:.2f}s "
                f"replay_mean={sum(td) / n * 1e3:.1f}ms host_mean={sum(r[1] for r in rec) / n * 1e3:.1f}ms "
                + " ".join(f"s0={rec[i][0]}:{td[i] * 1e3:.0f}ms" for i in pos)
                + f" first={[int(res[b][0]) for b in USERS]}"
            )
            # ---- account of the time outside the device (PROBE_WL=1 adds the per-window timeline)
            sync_run = os.environ.get("DSV41_PF_ASYNC_SERVER") == "0"
            key_ = lambda s0_: tuple(s0_) if isinstance(s0_, (list, tuple)) else s0_  # noqa: E731
            if sync_run:
                for s0_, th_, td_ in rec:
                    dev_map[key_(s0_)] = td_
            tot = {}
            for d_ in calls_t:
                for k_, v_ in d_.items():
                    if isinstance(v_, (int, float)):
                        tot[k_] = tot.get(k_, 0.0) + v_
            ncalls = len(calls_t)
            dev_sum = sum(dev_map.get(key_(r_[0]), float("nan")) for r_ in rec)
            log(
                f"PROBE ACCOUNT rep {rep} async={not sync_run} pieces={n_pieces} calls={ncalls} replays={len(rec)} wall={wall:.2f}s device(sync-run per-window td)={dev_sum:.2f}s outside-device={wall - dev_sum:.2f}s ({(wall - dev_sum) / wall * 100:.1f}% of wall, {(wall - dev_sum) / ncalls * 1e3:.0f} ms/call)"
            )
            log(
                "PROBE ACCOUNT components per prompt (s): "
                + " ".join(
                    f"{k_}={tot.get(k_, 0.0):.2f}"
                    for k_ in (
                        "pre_s",
                        "first_build_s",
                        "host_overlapped",
                        "host_s",
                        "hash_s",
                        "host_rows_gather_dequant",
                        "w_dynupd",
                        "w_prehooks",
                        "post_hooks",
                        "apc_snapshot_s",
                        "apc_overlapped_s",
                        "dyn_prebuild_s",
                        "export_s",
                        "replay_s",
                    )
                )
            )
            wl_ = pm.__dict__.get("winlog")
            if wl_ is not None:
                pm.winlog_drain()
                W_, A_, B_, C_, B0_, nB_ = wl_["windows"], 0.0, 0.0, 0.0, 0.0, 0
                A_parts = [0.0, 0.0, 0.0]
                gaps = []
                Cp_ = [0.0, 0.0, 0.0]
                Ck_ = [0.0, 0.0]
                for c_ in wl_["calls"]:
                    ws_ = [w_ for w_ in W_ if c_["t_start"] <= w_["t_in"] <= c_["t_end"]]
                    if not ws_:
                        continue
                    A_ += ws_[0]["t_enq"] - c_["t_start"]
                    A_parts[0] += c_["t_pre_end"] - c_["t_start"]
                    A_parts[1] += c_["first_build_s"]
                    A_parts[2] += ws_[0]["t_enq"] - c_["t_pre_end"] - c_["first_build_s"]
                    B0_ += ws_[0]["t_done"] - ws_[0]["t_enq"] - dev_map.get(key_(ws_[0]["s0"]), 0.0)
                    for a_, b_ in zip(ws_, ws_[1:]):
                        g_ = b_["t_done"] - a_["t_done"] - dev_map.get(key_(b_["s0"]), 0.0)
                        B_ += g_
                        gaps.append(g_)
                    C_ += c_["t_end"] - ws_[-1]["t_done"]
                    Cp_[0] += (
                        c_["t_loop_end"] - ws_[-1]["t_done"]
                    )  # after the last replay: hooks, bookkeeping, snapshot starts
                    Ck_[0] += (
                        ws_[-1].get("t_ret", 0.0) - ws_[-1]["t_done"]
                    )  # replay_window return (watcher latency, sync wait, post hooks)
                    Ck_[1] += ws_[-1].get("t_bk", 0.0) - ws_[-1].get(
                        "t_ret", 0.0
                    )  # export record, resume, snapshot starts
                    Cp_[1] += c_["t_apc_end"] - c_["t_loop_end"]  # snapshot completions not overlapped
                    Cp_[2] += c_["t_end"] - c_["t_apc_end"]  # trailing sync / stamps
                gs_ = sorted(gaps) or [0.0]
                log(
                    f"PROBE TIMELINE rep {rep}: start-of-call {A_:.2f}s (pre {A_parts[0]:.2f}, first window build {A_parts[1]:.2f}, first copies/record {A_parts[2]:.2f}) + first-window launch {B0_:.2f}s + gaps between windows {B_:.2f}s ({len(gaps)} gaps; median {gs_[len(gs_) // 2] * 1e3:.1f} ms, p90 {gs_[int(len(gs_) * 0.9)] * 1e3:.1f} ms, max {gs_[-1] * 1e3:.0f} ms) + end-of-call {C_:.2f}s (after last replay: hooks/bookkeeping/snapshot starts {Cp_[0]:.2f} [replay_window return {Ck_[0]:.2f} + bookkeeping/snapshot starts {Ck_[1]:.2f}], snapshot completion {Cp_[1]:.2f}, trailing {Cp_[2]:.2f}) = {A_ + B0_ + B_ + C_:.2f}s exposed; wall {wall:.2f}s - device {dev_sum:.2f}s = {wall - dev_sum:.2f}s"
                )
                torch.save(wl_["windows"], f"/tmp/ph_winlog_rep{rep}.pt")
            if os.environ.get("PROBE_CMP") == "1":
                tok_ = torch.tensor([int(res[b][0]) for b in range(B)], dtype=torch.long)
                lgs_ = []
                for k_ in range(
                    int(os.environ.get("PROBE_DECODE", "2"))
                ):  # (the first decode step runs the pending key hand-off)
                    out_ = m.decode_forward(
                        tok_, torch.full((B,), E + k_, dtype=torch.long), enable_trace=False, reload_inputs=True
                    )
                    lgs_.append(m.read_logits().float().clone())
                    tok_ = out_.clone().long()
                from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as _PC

                keys_ = {}
                for L_, d_ in list(m.index_owner.items())[:3]:
                    rr_ = m.attns[L_].ratio
                    keys_[L_] = (
                        _PC.read_shard(d_.k_cache, 0).reshape(m.U, d_.k_cache.shape[2], -1)[0, : E // rr_].clone()
                    )
                if getattr(m, "apc", None) is not None:
                    m.apc.finish_pending()  # (snapshots of the last windows stay pending until the next call / lookup)
                cmp_snaps[rep] = (
                    {k_[2:]: (v_.carried, v_.pool) for k_, v_ in m.apc.store.snaps.items()}
                    if getattr(m, "apc", None) is not None
                    else {}
                )
                cmp_[rep] = (
                    [int(res[b][0]) for b in USERS],
                    lgs_,
                    keys_,
                    (res[USERS[0]][1].float().clone() if res[USERS[0]][1] is not None else None),
                )
        if cmp_:
            ref_ = cmp_[0]
            for rep, c_ in cmp_.items():
                eq_first = c_[0] == ref_[0]
                eq_dec = all(bool((a == b).all()) for a, b in zip(c_[1], ref_[1]))
                eq_keys = all(bool((c_[2][L_] == ref_[2][L_]).all()) for L_ in ref_[2])
                eq_lg = c_[3] is None or bool((c_[3] == ref_[3]).all())
                sn_, sr_ = cmp_snaps[rep], cmp_snaps[0]
                eq_snap = sn_.keys() == sr_.keys() and all(
                    all(all(torch.equal(x, y) for x, y in zip(sn_[k_][0][n_], sr_[k_][0][n_])) for n_ in sr_[k_][0])
                    and all(torch.equal(sn_[k_][1][n_], sr_[k_][1][n_]) for n_ in sr_[k_][1])
                    for k_ in sr_
                )
                log(f"PROBE CMP E={E} rep {rep} vs rep 0: {len(sn_)} prefix snapshots, bit-equal {eq_snap}")
                assert eq_snap, f"prefix snapshots of rep {rep} differ from rep 0"
                log(
                    f"PROBE CMP E={E} rep {rep} vs rep 0: first tokens equal {eq_first}, prefill logits bit-equal {eq_lg}, decode logits bit-equal {eq_dec}, decode index keys bit-equal {eq_keys}"
                )
                assert eq_first and eq_dec and eq_keys and eq_lg, f"rep {rep} differs from rep 0"
        if os.environ.get(
            "PROBE_SAVE_IDS"
        ):  # translated pool rows of the last chunk (layer -> [C, 512]) of mesh device 0
            out_ids = {}
            for lid, pl in pm.layers:
                sp_ = getattr(pl.pa, "sparse", None)
                if sp_ is not None and getattr(sp_, "pids", None) is not None:
                    out_ids[lid] = (
                        ttnn.to_torch(ttnn.get_device_tensors(sp_.pids)[0]).reshape(-1, 512).long() & 0xFFFFFFFF
                    )
            torch.save(
                out_ids, f"{os.environ['PROBE_SAVE_IDS']}.E{E}.{MODE}.single{os.environ.get('DSV41_PF_SINGLE', '0')}"
            )
        if os.environ.get(
            "PROBE_SAVE_POOL"
        ):  # pool latents of every kv-source layer present, user 0, entries [0, E / ratio)
            from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as _P
            from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as _PC

            pool_d = {src: _PC.read_pool_latents(m, 0, src, E // _P.SRC_RATIO[src]) for src in _PC._sources(m)}
            torch.save(
                pool_d, f"{os.environ['PROBE_SAVE_POOL']}.E{E}.{MODE}.single{os.environ.get('DSV41_PF_SINGLE', '0')}"
            )
        if os.environ.get(
            "PROBE_SAVE_DEC"
        ):  # decode state of user 0 after the prefill (hand-off): pool latents, ring rows, prev_cs, decode index keys
            from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as _P
            from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as _PC

            m.flush_key_exports(
                why="probe read"
            )  # (the key hand-off is lazy: it runs before the first decode step or on demand)
            st_, cols_, U_ = {}, m.cols, m.U
            full_ = _PC.read_shard(m.pool.pool, 0)[0, 0]
            for src in _PC._sources(m):
                st_[f"pool.lat{src}"] = full_[
                    m.pool.phys_rows(0, src, torch.arange(E // _P.SRC_RATIO[src])).long()
                ].clone()
            for L_, d_ in m.index_owner.items():
                rr_ = m.attns[L_].ratio
                st_[f"idxkeys{L_}"] = (
                    _PC.read_shard(d_.k_cache, 0).reshape(U_, d_.k_cache.shape[2], -1)[0, : E // rr_].clone()
                )
            for b_ in range(
                1, B
            ):  # the other users (mesh row b_): their decode index keys / pool latents must equal user 0's for identical prompts
                full_b = _PC.read_shard(m.pool.pool, b_ * m.cols)[0, 0]
                for src in _PC._sources(m):
                    st_[f"u{b_}.pool.lat{src}"] = full_b[
                        m.pool.phys_rows(b_, src, torch.arange(E // _P.SRC_RATIO[src])).long()
                    ].clone()
                for L_, d_ in m.index_owner.items():
                    rr_ = m.attns[L_].ratio
                    st_[f"u{b_}.idxkeys{L_}"] = (
                        _PC.read_shard(d_.k_cache, b_ * m.cols)
                        .reshape(U_, d_.k_cache.shape[2], -1)[0, : E // rr_]
                        .clone()
                    )
            torch.save(
                st_, f"{os.environ['PROBE_SAVE_DEC']}.E{E}.{MODE}.single{os.environ.get('DSV41_PF_SINGLE', '0')}"
            )
        if os.environ.get(
            "PROBE_DECODE"
        ):  # plain eager decode steps after the prefill: tokens / logits of the 4 users must agree (identical prompts)
            nstep = int(os.environ["PROBE_DECODE"])
            tok_ = torch.tensor([int(res[b][0]) for b in range(B)], dtype=torch.long)
            dec_lg = []
            for k_ in range(nstep):
                pos_ = torch.full((B,), E + k_, dtype=torch.long)
                out_ = m.decode_forward(tok_, pos_, enable_trace=False, reload_inputs=True)
                lg_ = m.read_logits().float()
                log(
                    f"PROBE DECODE step {k_}: tokens {[int(x) for x in out_]} max |logit(user b) - logit(user 0)| {[float((lg_[b] - lg_[0]).abs().max()) for b in range(1, B)]}"
                )
                tok_ = out_.clone().long()
                dec_lg.append(lg_.clone())
            if os.environ.get("PROBE_DECODE_SAVE"):
                torch.save(
                    {"logits": dec_lg, "first": [int(res[b][0]) for b in USERS]},
                    f"{os.environ['PROBE_DECODE_SAVE']}.E{E}.lazy{os.environ.get('DSV41_PF_LAZY_EXPORT', '1')}",
                )
        if SAME and os.environ.get("PROBE_NOLOGITS") != "1":
            lg = torch.stack([res[b][1] for b in USERS]).float()
            log(
                f"PROBE TOP5 user0: {lg[0].topk(5).indices.tolist()} values {[round(v, 2) for v in lg[0].topk(5).values.tolist()]}"
            )
            log(
                f"PROBE SAME E={E}: max |logit(user b) - logit(user 0)| = {[float((lg[b] - lg[0]).abs().max()) for b in range(1, len(USERS))]}"
            )
        if os.environ.get("PROBE_SAVE") and rep == REPS - 1:
            torch.save(
                torch.stack([res[b][1] for b in USERS]).float(),
                f"{os.environ['PROBE_SAVE']}.E{E}.{MODE}.single{os.environ.get('DSV41_PF_SINGLE', '0')}",
            )
    if os.environ.get(
        "PROBE_RAW"
    ):  # calibration: n replays of the chunk trace back to back, no host work in between (the pure device time per replay at the last position)
        n_raw = int(os.environ["PROBE_RAW"])
        ttnn.synchronize_device(md)
        t_ = time.perf_counter()
        for _ in range(n_raw):
            ttnn.execute_trace(md, pm.dyn_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        log(f"PROBE RAW {n_raw} back-to-back replays: {(time.perf_counter() - t_) / n_raw * 1e3:.1f} ms each")
    if pm.__dict__.get("build_prof") is not None:
        import io
        import pstats

        buf_ = io.StringIO()
        pstats.Stats(pm.build_prof, stream=buf_).strip_dirs().sort_stats("cumulative").print_stats(45)
        log("PROBE BUILD PROFILE (host work of the window builds, cumulative):\n" + buf_.getvalue())
    log("PROBE DONE")
