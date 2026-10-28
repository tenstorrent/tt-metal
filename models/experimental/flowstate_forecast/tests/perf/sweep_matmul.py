# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""G2-r1 item 1: sweep the 1D-mcast program config of every matmul role at paper (M = 4096 x 64 rows), one job.

    FLOWSTATE_SWEEP_JSON=out.json pytest -q -s models/experimental/flowstate_forecast/tests/perf/sweep_matmul.py

Each role runs with the model's dtypes and fidelity (tt/common.py), random inputs. Per config: synchronized wall
ms (median of 5 after 2 warm-ups) and the max abs difference to the default config's output (a changed in0_block_w
changes the accumulation order, so 0 is not expected). The default is the round-0 auto config
(in0_block_w 2, out_subblock 1x4, out_block_h 15, per_core_M 75)."""

import itertools
import json
import os
import time

import pytest
import torch
import ttnn

from models.experimental.flowstate_forecast.tt.common import policy

M = 4096 * 64
GRID = (11, 10)
# role: (K, N, in dtype, weight dtype key, out dtype, fidelity key, bias, activation)
ROLES = {
    "B": (512, 1024, ttnn.bfloat16, "b_weight_dtype", ttnn.float32, "b_proj", False, None),
    "C": (1024, 512, ttnn.float32, "c_weight_dtype", ttnn.bfloat16, "c_proj", False, None),
    "gate": (512, 512, ttnn.bfloat16, "dense_weight_dtype", ttnn.bfloat16, "dense", True, None),
    "w1": (512, 1024, ttnn.bfloat16, "dense_weight_dtype", ttnn.bfloat16, "dense", True, "silu"),
    "w3": (1024, 512, ttnn.bfloat16, "dense_weight_dtype", ttnn.bfloat16, "dense", True, None),
}


def configs(K, N, act=None):
    Kt, Nt, cores = K // 32, N // 32, GRID[0] * GRID[1]
    per_core_M = -(-M // 32 // cores)
    yield "default", None
    for kw, (sh, sw), bh in itertools.product((1, 2, 4, 8), ((1, 4), (2, 2), (4, 1), (1, 8), (2, 4)), (5, 15, 25, 75)):
        if Kt % kw or per_core_M % bh or bh % sh or Nt % sw:
            continue
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=GRID,
            in0_block_w=kw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            out_block_h=bh,
            out_block_w=Nt,
            per_core_M=per_core_M,
            per_core_N=Nt,
            fuse_batch=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if act else None,
            mcast_in0=False,
        )
        yield f"kw{kw}_sb{sh}x{sw}_bh{bh}", cfg


def _time(fn, reps=5):
    for _ in range(2):
        ttnn.deallocate(fn())
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(fn.device)
        t = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(fn.device)
        ts.append((time.perf_counter() - t) * 1e3)
        ttnn.deallocate(out)
    return sorted(ts)[len(ts) // 2]


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
@pytest.mark.parametrize("role", list(ROLES))
def test_sweep(device, role, precision):
    K, N, in_dt, wkey, out_dt, fkey, bias, act = ROLES[role]
    p = policy(device, precision)
    if precision == "performance" and fkey != "dense":
        pytest.skip("B / C are the same in both presets")
    g = torch.Generator().manual_seed(0)
    a = ttnn.from_torch(torch.randn(1, 1, M, K, generator=g), dtype=in_dt, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(torch.randn(K, N, generator=g) / K**0.5, dtype=getattr(p, wkey), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, N, generator=g), dtype=getattr(p, wkey), layout=ttnn.TILE_LAYOUT, device=device) if bias else None
    ref, rows = None, []
    for name, cfg in configs(K, N, act):

        def fn():
            return ttnn.linear(
                a, w, bias=b, activation=act if cfg is None else None, dtype=out_dt, program_config=cfg,
                compute_kernel_config=getattr(p, fkey), memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        fn.device = device
        try:
            out = ttnn.to_torch(fn()).float()
            ms = _time(fn)
        except Exception as e:  # L1 overflow, invalid config
            rows.append({"config": name, "error": str(e).splitlines()[0][:160]})
            continue
        if ref is None:
            ref = out
        diff = float((out - ref).abs().max())
        rows.append({"config": name, "ms": ms, "max_abs_diff": diff})
        print(f"{role} {precision} {name}: {ms:.3f} ms, diff {diff:.3g}")
    ok = [r for r in rows if "ms" in r]
    best = min(ok, key=lambda r: r["ms"])
    print(f"BEST {role} {precision} {best['config']} {best['ms']:.3f} ms vs default {ok[0]['ms']:.3f}")
    if path := os.environ.get("FLOWSTATE_SWEEP_JSON"):
        with open(path, "a") as f:
            f.write(json.dumps({"role": role, "precision": precision, "rows": rows}) + "\n")
