# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R2-4: second tiling sweep of the B and C projections at paper (M = 4096 x 64 rows), one job.

    FLOWSTATE_SWEEP_JSON=out.jsonl pytest -q -s models/experimental/flowstate_forecast/tests/perf/sweep_bc.py

B and C are the same in both presets (tt/common.py), so this runs the accuracy policy only. Per config:
synchronized wall ms (median of 5 after 2 warm-ups) and the max abs difference to the current config (R1-1:
in0_block_w 2, out_subblock 1x4, out_block_h 5, out_block_w = N). Configs: the 1D in1-mcast config with
out_block_w < N and other subblocks (at most 4 tiles with fp32 dest), and minimal_matmul blocks."""

import itertools
import json
import os

import pytest
import torch
import ttnn

from models.experimental.flowstate_forecast.tests.perf.sweep_matmul import GRID, M, ROLES, _time
from models.experimental.flowstate_forecast.tt.common import policy


def one_d(Kt, Nt, kw, sh, sw, bh, bw):
    per_core_M = -(-M // 32 // (GRID[0] * GRID[1]))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=kw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=bh,
        out_block_w=bw,
        per_core_M=per_core_M,
        per_core_N=Nt,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


def configs(K, N):
    Kt, Nt = K // 32, N // 32
    per_core_M = -(-M // 32 // (GRID[0] * GRID[1]))
    yield "current", "linear", one_d(Kt, Nt, 2, 1, 4, 5, Nt)
    for kw, (sh, sw), bh, bw in itertools.product(
        (1, 2, 4), ((1, 4), (2, 2), (4, 1), (1, 2), (1, 1)), (1, 3, 5, 15), (Nt, Nt // 2, Nt // 4, Nt // 8)
    ):
        if Kt % kw or per_core_M % bh or bh % sh or bw % sw or Nt % bw or (kw, sh, sw, bh, bw) == (2, 1, 4, 5, Nt):
            continue
        if bw == Nt and (sh, sw) == (1, 4):
            continue  # the R1-1 sweep covered these
        yield f"1d_kw{kw}_sb{sh}x{sw}_bh{bh}_bw{bw}", "linear", one_d(Kt, Nt, kw, sh, sw, bh, bw)
    for mb, kb, nb, (sh, sw) in itertools.product((4, 8, 16), (4, 8), (4, 8, 16), ((1, 4), (2, 2), (4, 1), (2, 1))):
        if Kt % kb or Nt % nb or mb % sh or nb % sw:
            continue
        yield f"mm_m{mb}_k{kb}_n{nb}_sb{sh}x{sw}", "minimal", ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=GRID,
        )


@pytest.mark.parametrize("role", ["B", "C"])
def test_sweep_bc(device, role):
    K, N, in_dt, wkey, out_dt, fkey, _, _ = ROLES[role]
    p = policy(device, "accuracy")
    g = torch.Generator().manual_seed(0)
    a = ttnn.from_torch(torch.randn(1, 1, M, K, generator=g), dtype=in_dt, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(
        torch.randn(K, N, generator=g) / K**0.5, dtype=getattr(p, wkey), layout=ttnn.TILE_LAYOUT, device=device
    )
    ref, rows = None, []
    for name, op, cfg in configs(K, N):

        def fn():
            if op == "minimal":
                return ttnn.experimental.minimal_matmul(
                    a, w, config=cfg, compute_kernel_config=getattr(p, fkey), dtype=out_dt
                )
            return ttnn.linear(
                a,
                w,
                dtype=out_dt,
                program_config=cfg,
                compute_kernel_config=getattr(p, fkey),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        fn.device = device
        try:
            out = ttnn.to_torch(fn()).float().reshape(M, N)
            ms = _time(fn)
        except Exception as e:  # L1 overflow, invalid config, unsupported dtype
            rows.append({"config": name, "error": str(e).splitlines()[0][:160]})
            continue
        if ref is None:
            ref = out
        diff = float((out - ref).abs().max())
        rows.append({"config": name, "ms": ms, "max_abs_diff": diff})
        print(f"{role} {name}: {ms:.3f} ms, diff {diff:.3g}")
    ok = [r for r in rows if "ms" in r]
    best = min(ok, key=lambda r: r["ms"])
    print(f"BEST {role} {best['config']} {best['ms']:.3f} ms vs current {ok[0]['ms']:.3f} ({len(ok)} ran)")
    if path := os.environ.get("FLOWSTATE_SWEEP_JSON"):
        with open(path, "a") as f:
            f.write(json.dumps({"role": role, "rows": rows}) + "\n")
