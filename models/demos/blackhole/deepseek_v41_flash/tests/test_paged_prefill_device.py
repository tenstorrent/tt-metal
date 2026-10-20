# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged prefill (tt/prefill_paged.py, DSV41_PF_PAGED) against the dense prefill, same model object, same prompts.

For each mode in PP_MODES (default "0,write,read"; "0" = dense release path, "write" = PF1 dual write, "read" = PF2 paged sparse_sdpa without kv tables) the chunk trace is torn down and
re-captured and the same users are prefilled through ``prefill_interleaved`` (production path: Up=1 slot per mesh row, chunks of PP_CHUNK, ragged lengths). Compared against the dense run:
first token and last-token logits (bit equality, max abs diff), the whole pool (pages + rings, everything below the scratch region: bit equality), and per mode the replay time per chunk and the
DRAM per bank after the capture.

Env: DSV41_LAYERS (default 0-24), PP_PAGES (pages per mesh row of the pool), PP_U (users per row, 4), PP_ISL (4096), PP_CHUNK (512), PP_MODES, PP_REPS (timed repetitions, 2), DSV41_POOL_DTYPE must stay unset (bf16 pool).
    DSV41_PF_PAGED=write DSV41_PREFILL_UP=1 DSV41_PF_UMASK=1 pytest tests/test_paged_prefill_device.py -s
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_paged as PG
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

U = int(os.environ.get("PP_U", "4"))
ISL = int(os.environ.get("PP_ISL", "4096"))
CHUNK = int(os.environ.get("PP_CHUNK", "512"))
MODES = os.environ.get("PP_MODES", "0,write,read").split(",")
REPS = int(os.environ.get("PP_REPS", "2"))
LAYERS = os.environ.get("DSV41_LAYERS", "0-24")
NEEDLE = (
    os.environ.get("PP_NEEDLE") == "1"
)  # needle-in-haystack prompts (depths PP_DEPTHS, one per user), exact length ISL, answers decoded
DEPTHS = [float(x) for x in os.environ.get("PP_DEPTHS", "0.1,0.4,0.7,0.95").split(",")]
DECODE = int(os.environ.get("PP_DECODE", "0"))  # plain greedy decode steps after every prefill (end-to-end check)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("PP_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_paged_prefill(mesh_device):
    try:
        _run(mesh_device)
    except BaseException:
        import traceback

        traceback.print_exc()
        print("PP FAILED with the traceback above", flush=True)
        raise


def _run(mesh_device):
    os.environ.setdefault("DSV41_PF_UMASK", "1")
    os.environ.setdefault("DSV41_PREFILL_UP", "1")
    md = mesh_device
    log = lambda s: print(s, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    S_pad = -(-ISL // CHUNK) * CHUNK
    max_ctx = max(ISL + 64, S_pad)
    pp = default_page_params(max_ctx, U)
    if os.environ.get(
        "PP_PAGES"
    ):  # pages per mesh row (default: all users x max_ctx): a smaller pool lets the bf16 pool + dense tables fit at 1M
        pp["page_max_num_blocks_per_dp"] = int(os.environ["PP_PAGES"])
    args, m, pool, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    assert pool.scratch_rows > 0, "build with DSV41_PF_PAGED=write|read"
    assert (
        pool.dtype == ttnn.bfloat16 or set(MODES) == {"0"} or os.environ.get("DSV41_PF_PAGED_FP8") == "1"
    ), "write / read need a bf16 pool (P1)"
    g = torch.Generator().manual_seed(4321)
    prompts = torch.randint(1000, 100000, (B, ISL), generator=g)
    users = [0, 5, 10, 15] if U >= 4 else [0, U]
    lens_pick = [ISL, ISL - 197, ISL - 1024 - 3, ISL - 512]  # aligned, unaligned, short unaligned, chunk-aligned
    lens = {u: lens_pick[i % 4] for i, u in enumerate(users)}
    if NEEDLE:
        tk = args.tokenizer
        rng = torch.Generator().manual_seed(7)
        words = "alpha beta gamma delta river stone cloud forest engine window garden bridge market".split()
        filler = tk.encode(
            " ".join(words[i] for i in torch.randint(0, len(words), (ISL,), generator=rng).tolist()),
            add_special_tokens=False,
        )
        pre = tk.encode("The following is a long text with a hidden fact.\n\n", add_special_tokens=False)
        post = tk.encode(
            "\n\nQuestion: What is the secret code mentioned in the text above? Answer with just the number.\nAnswer:",
            add_special_tokens=False,
        )
        codes = {}
        for i, u in enumerate(users):
            code = 4817 + 1111 * i
            codes[u] = code
            nd = tk.encode(f" The secret code is {code}. ", add_special_tokens=False)
            room = ISL - len(pre) - len(post) - len(nd)
            n1 = int(DEPTHS[i % len(DEPTHS)] * room)
            ids = pre + filler[:n1] + nd + filler[n1:room] + post
            assert len(ids) == ISL and len(filler) >= room
            prompts[u] = torch.tensor(ids)
        lens = {u: ISL for u in users}
    log(f"PP model: B={B} U={U} Up={m.Up} layers {LAYERS} chunk {CHUNK} S_pad {S_pad} lens {lens}")

    def snap():
        if os.environ.get("PP_NOPOOL"):  # 1M: the pool is 2.5 GiB per mesh row
            return []
        return [ttnn.to_torch(ttnn.get_device_tensors(pool.pool)[r * m.cols]).clone() for r in range(m.rows)]

    def snapk():
        """decode index-key slabs (bfp8 / bf16 tile [U,1,n_alloc,128]) of the key owners, per mesh row"""
        if os.environ.get("PP_NOPOOL") or not os.environ.get("PP_KEYS"):
            return {}
        return {
            L: [ttnn.to_torch(ttnn.get_device_tensors(d.k_cache)[r * m.cols]).clone() for r in range(m.rows)]
            for L, d in m.index_owner.items()
        }

    results = {}
    for mode in MODES:
        PG.set_mode(mode.split("#")[0])  # "0#b": a second dense run (run-to-run noise baseline)
        m.prefill_model.teardown_dyn()
        m.release_trace()
        m.pf_resume = {}
        m.__dict__.pop("pf_slot_of", None)
        m.__dict__.pop("pf_slot_owner", None)
        if os.environ.get("PP_ZERO_IDLE", "1") == "1":
            # the decode key slabs of the users that are not prefilled hold leftovers of whatever ran before; their decode rows share the batch with the real users' (batch-composition noise in
            # the decode logits, dense vs dense), so start every mode from zeroed slabs (the prefill writes the real users' slabs)
            for d in m.index_owner.values():
                z = ttnn.zeros_like(d.k_cache)
                ttnn.copy(z, d.k_cache)
                ttnn.deallocate(z)
            ttnn.synchronize_device(md)
        times, res = [], None
        for rep in range(REPS):
            for u in range(B):
                m.release_user(u)
            m.pf_resume = {}
            t0 = time.perf_counter()
            res = m.prefill_interleaved(
                [(u, prompts[u], 0, lens[u]) for u in users], CHUNK, s_pad_max=S_pad, want_logits=True
            )
            wall = time.perf_counter() - t0
            tm = dict(m.timing)
            times.append((wall, tm.get("replay_s", 0.0), tm.get("replays", 0)))
            log(
                f"PP mode {mode} rep {rep}: wall {wall:.2f}s replay_s {tm.get('replay_s', 0):.3f} replays {tm.get('replays')}"
            )
        pre = dict(pool=snap(), keys=snapk())  # state right after the prefill, before any decode step
        dec = None
        if DECODE:
            tok = torch.zeros(B, dtype=torch.long)
            pos = torch.zeros(B, dtype=torch.long)
            for u in users:
                tok[u], pos[u] = int(res[u][0]), lens[u]
            dec = {"tok": [], "logits": []}
            step_t = []
            t0 = time.perf_counter()
            for step in range(DECODE):
                ts = time.perf_counter()
                out = m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True)
                step_t.append(time.perf_counter() - ts)
                lg = m.read_logits().clone()
                dec["tok"].append(torch.stack([out[u].clone() for u in users]))
                dec["logits"].append(lg[users].float().clone())
                for u in users:
                    tok[u], pos[u] = int(out[u]), pos[u] + 1
            if NEEDLE:
                for i, u in enumerate(users):
                    ids = [int(res[u][0])] + [int(t[i]) for t in dec["tok"]]
                    txt = tk.decode(ids)
                    log(
                        f"PP NEEDLE mode {mode} depth {DEPTHS[i % len(DEPTHS)]} user {u} code {codes[u]}: found {str(codes[u]) in txt} answer {txt[:60]!r}"
                    )
            st_ = sorted(step_t[1:])
            log(
                f"PP DECODE TPOT mode {mode} pool {pool.dtype}: median step {st_[len(st_) // 2] * 1e3:.1f} ms, min {st_[0] * 1e3:.1f} ms, max {st_[-1] * 1e3:.1f} ms (steps 1..{DECODE - 1}, step 0 includes the trace capture)"
            )
            log(f"PP mode {mode}: {DECODE} decode steps in {time.perf_counter() - t0:.1f}s")
        free = m.dram_snapshot()
        log(
            f"PP mode {mode}: DRAM per bank after capture+run: allocated {free[0]:.1f} MiB free {free[1]:.1f} MiB largest block {free[2]:.1f} MiB"
        )
        st = snap()
        results[mode] = dict(res=res, pool=st, times=times, dram=free, dec=dec, pre=pre)

    if os.environ.get(
        "PP_SAVE"
    ):  # logits of every mode (dense logits do not depend on the pool dtype: prefill reads the bf16 tables)
        torch.save({mode: {u: results[mode]["res"][u] for u in users} for mode in MODES}, os.environ["PP_SAVE"])
    ref = torch.load(os.environ["PP_REF"]) if os.environ.get("PP_REF") else None
    if ref is not None:
        for mode in MODES:
            for u in users:
                t0_, l0 = ref["0"][u]
                t1_, l1 = results[mode]["res"][u]
                eq = bool(torch.equal(l0, l1))
                log(
                    f"PP {mode} vs saved dense user {u}: token {t0_} / {t1_} logits equal {eq} maxabs {(l0 - l1).abs().max().item():.3e}"
                )
                assert t0_ == t1_ and eq, "differs from the saved dense run"
    base = results[MODES[0]]
    ok = True
    for mode in MODES[1:] if ref is None else []:
        r = results[mode]
        for u in users:
            t0_, l0 = base["res"][u]
            t1_, l1 = r["res"][u]
            d = (l0 - l1).abs().max().item()
            eq = bool(torch.equal(l0, l1))
            log(
                f"PP {mode} vs {MODES[0]} user {u} len {lens[u]}: token {t0_} / {t1_} logits equal {eq} maxabs {d:.3e} (|l|max {l0.abs().max():.2f})"
            )
            a0, a1 = l0.float() - l0.float().mean(), l1.float() - l1.float().mean()
            pcc = float((a0 @ a1) / (a0.norm() * a1.norm() + 1e-12))
            top5 = len(set(l0.topk(5).indices.tolist()) & set(l1.topk(5).indices.tolist()))
            log(f"PP {mode} vs {MODES[0]} user {u}: logits PCC {pcc:.6f} top-5 overlap {top5}/5")
            ok &= t0_ == t1_ and (eq or os.environ.get("PP_SOFT") == "1")
        for tag, a_, b_ in (("pre-decode", base["pre"], r["pre"]), ("post-decode", base, r)):
            if a_["pool"]:
                bad = []
                for row in range(m.rows):
                    x = a_["pool"][row].reshape(-1, 512)[: pool.scratch_origin]
                    y = b_["pool"][row].reshape(-1, 512)[: pool.scratch_origin]
                    ne = (
                        (~((x.float() == y.float()) | (x.float().isnan() & y.float().isnan())))
                        .any(dim=1)
                        .nonzero()
                        .reshape(-1)
                    )
                    if len(ne):
                        pg = sorted({int(i) // 320 for i in ne.tolist()})
                        bad.append(
                            f"mesh row {row}: {len(ne)} rows differ (pages {pg[:4]}..{pg[-2:]}; first rows {ne[:4].tolist()})"
                        )
                log(f"PP {mode} vs {MODES[0]} POOL {tag}: " + ("bit-identical" if not bad else "; ".join(bad)))
            for L in a_.get("keys", {}):
                for row in range(m.rows):
                    x, y = a_["keys"][L][row].float(), b_["keys"][L][row].float()
                    du = [
                        u for u in range(x.shape[0]) if not bool(((x[u] == y[u]) | (x[u].isnan() & y[u].isnan())).all())
                    ]
                    if du:
                        det = []
                        for u in [u_ for u_ in du if row * U + u_ in lens]:  # the in-row users that were prefilled
                            ne = (
                                ((x[u] != y[u]) & ~(x[u].isnan() & y[u].isnan())).any(dim=(0, 2)).nonzero().reshape(-1)
                                if x[u].dim() == 3
                                else ((x[u] != y[u]) & ~(x[u].isnan() & y[u].isnan())).any(dim=-1).nonzero().reshape(-1)
                            )
                            ratio = 1 if L == 20 else 2
                            b_user = row * U + u
                            nv = (lens.get(b_user, 0)) // ratio
                            det.append(
                                f"user {u}: {len(ne)} entries differ, range [{int(ne.min())}, {int(ne.max())}], valid entries {nv}, differ below valid {int((ne < nv).sum())}"
                            )
                        log(
                            f"PP {mode} vs {MODES[0]} KEYSLAB {tag} layer {L} mesh row {row}: users differ {du}; "
                            + "; ".join(det)
                        )
        if base["pre"][
            "pool"
        ]:  # the pool right after the prefill must be bit-identical (decode steps are compared by ids / logits below)
            ok &= all(
                torch.equal(
                    base["pre"]["pool"][row].reshape(-1, 512)[: pool.scratch_origin].float().nan_to_num(7.0),
                    r["pre"]["pool"][row].reshape(-1, 512)[: pool.scratch_origin].float().nan_to_num(7.0),
                )
                for row in range(m.rows)
            )
    if DECODE:
        ref_dec = results[MODES[0]]["dec"]
        for mode in MODES[1:]:
            d = results[mode]["dec"]
            toks_eq = all(torch.equal(a, b) for a, b in zip(ref_dec["tok"], d["tok"]))
            lg_eq = sum(int(torch.equal(a, b)) for a, b in zip(ref_dec["logits"], d["logits"]))
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(ref_dec["tok"], d["tok"])) if not torch.equal(a, b)), None
            )
            log(
                f"PP DECODE {mode} vs {MODES[0]}: {DECODE} steps x {len(users)} users, generated ids identical {toks_eq} (first differing step {first_diff}), logits bit-identical in {lg_eq}/{DECODE} steps"
            )
            ok &= toks_eq
        log("PP DECODE sample ids user " + str(users[0]) + ": " + str([int(t[0]) for t in ref_dec["tok"][:16]]))
    for mode in MODES:
        w = results[mode]["times"][-1]
        log(
            f"PP SUMMARY mode {mode}: replay {w[1]:.3f}s over {w[2]} replays = {w[1] / max(w[2], 1) * 1e3:.1f} ms/replay, wall {w[0]:.2f}s, DRAM allocated {results[mode]['dram'][0]:.1f} MiB/bank free {results[mode]['dram'][1]:.1f}"
        )
    assert ok, "paged prefill differs from dense"
