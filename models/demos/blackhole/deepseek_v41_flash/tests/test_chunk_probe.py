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
        for rep in range(REPS):
            if os.environ.get("PROBE_RELEASE") == "1":
                for b_ in range(B):
                    m.release_user(b_)
            rec.clear()
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
            items = [(b, toks[b], st_, E) for b in USERS]
            res = m.prefill_interleaved(
                items,
                CHUNK,
                s_pad_max=WMAX if MODE == "single" else None,
                want_logits=(bool(os.environ.get("PROBE_SAVE")) or SAME) and os.environ.get("PROBE_NOLOGITS") != "1",
            )
            wall = time.perf_counter() - t0
            n = len(rec)
            td = [r[2] for r in rec]
            pos = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
            log(
                f"PROBE mode={MODE} E={E} rep={rep} S_pad={pm.dyn.S_pad} captures_this_call={ncap[0] - c0} captures_total={ncap[0]} replays={n} wall={wall:.2f}s "
                f"replay_mean={sum(td) / n * 1e3:.1f}ms host_mean={sum(r[1] for r in rec) / n * 1e3:.1f}ms "
                + " ".join(f"s0={rec[i][0]}:{td[i] * 1e3:.0f}ms" for i in pos)
                + f" first={[int(res[b][0]) for b in USERS]}"
            )
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
    log("PROBE DONE")
