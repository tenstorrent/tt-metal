# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged prefill (tt/prefill_paged.py, DSV41_PF_PAGED) against the dense prefill, same model object, same prompts.

For each mode in PP_MODES (default "0,write,read"; "0" = dense release path, "write" = PF1 dual write, "read" = PF2 paged sparse_sdpa without kv tables) the chunk trace is torn down and
re-captured and the same users are prefilled through ``prefill_interleaved`` (production path: Up=1 slot per mesh row, chunks of PP_CHUNK, ragged lengths). Compared against the dense run:
first token and last-token logits (bit equality, max abs diff), the whole pool (pages + rings, everything below the scratch region: bit equality), and per mode the replay time per chunk and the
DRAM per bank after the capture.

Env: DSV41_LAYERS (default 0-24), PP_U (users per row, 4), PP_ISL (4096), PP_CHUNK (512), PP_MODES, PP_REPS (timed repetitions, 2), DSV41_POOL_DTYPE must stay unset (bf16 pool).
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
    args, m, pool, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    assert pool.dtype == ttnn.bfloat16 and pool.scratch_rows > 0, "build with DSV41_PF_PAGED=write|read and a bf16 pool"
    g = torch.Generator().manual_seed(4321)
    prompts = torch.randint(1000, 100000, (B, ISL), generator=g)
    users = [0, 5, 10, 15] if U >= 4 else [0, U]
    lens_pick = [ISL, ISL - 197, ISL - 1024 - 3, ISL - 512]  # aligned, unaligned, short unaligned, chunk-aligned
    lens = {u: lens_pick[i % 4] for i, u in enumerate(users)}
    log(f"PP model: B={B} U={U} Up={m.Up} layers {LAYERS} chunk {CHUNK} S_pad {S_pad} lens {lens}")

    def snap():
        return [ttnn.to_torch(ttnn.get_device_tensors(pool.pool)[r * m.cols]).clone() for r in range(m.rows)]

    results = {}
    for mode in MODES:
        PG.set_mode(mode)
        m.prefill_model.teardown_dyn()
        m.release_trace()
        m.pf_resume = {}
        m.__dict__.pop("pf_slot_of", None)
        m.__dict__.pop("pf_slot_owner", None)
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
        free = m.dram_snapshot()
        log(
            f"PP mode {mode}: DRAM per bank after capture+run: allocated {free[0]:.1f} MiB free {free[1]:.1f} MiB largest block {free[2]:.1f} MiB"
        )
        st = snap()
        results[mode] = dict(res=res, pool=st, times=times, dram=free)

    base = results[MODES[0]]
    ok = True
    for mode in MODES[1:]:
        r = results[mode]
        for u in users:
            t0_, l0 = base["res"][u]
            t1_, l1 = r["res"][u]
            d = (l0 - l1).abs().max().item()
            eq = bool(torch.equal(l0, l1))
            log(
                f"PP {mode} vs {MODES[0]} user {u} len {lens[u]}: token {t0_} / {t1_} logits equal {eq} maxabs {d:.3e} (|l|max {l0.abs().max():.2f})"
            )
            ok &= t0_ == t1_ and eq
        bad = []
        for row in range(m.rows):
            x = base["pool"][row].reshape(-1, 512)[: pool.scratch_origin]
            y = r["pool"][row].reshape(-1, 512)[: pool.scratch_origin]
            ne = (
                (~((x.float() == y.float()) | (x.float().isnan() & y.float().isnan()))).any(dim=1).nonzero().reshape(-1)
            )
            if len(ne):
                bad.append(f"mesh row {row}: {len(ne)} pool rows differ (first {ne[:6].tolist()})")
        log(f"PP {mode} vs {MODES[0]} pool (pages + rings): " + ("bit-identical" if not bad else "; ".join(bad)))
        ok &= not bad
    for mode in MODES:
        w = results[mode]["times"][-1]
        log(
            f"PP SUMMARY mode {mode}: replay {w[1]:.3f}s over {w[2]} replays = {w[1] / max(w[2], 1) * 1e3:.1f} ms/replay, wall {w[0]:.2f}s, DRAM allocated {results[mode]['dram'][0]:.1f} MiB/bank free {results[mode]['dram'][1]:.1f}"
        )
    assert ok, "paged prefill differs from dense"
