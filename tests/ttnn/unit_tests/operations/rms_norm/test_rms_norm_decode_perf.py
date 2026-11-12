# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5: decode / few-row _PERF_BASE cases at their EXACT config, for --profile runs.

bf16 TILE x, bf16 TILE gamma, fp32_dest_acc_en=False, HiFi2. Interleaved DRAM cases + the pinned
WIDTH_SHARDED / BLOCK_SHARDED geometries (shard_shape, core_grid). Correctness is checked at the perf
cases' soft PCC gate (0.9995); perf is read from the profiler CSV.
"""

import pytest
import torch
import ttnn

from eval.metrics import check_output
from eval.golden_tests.rms_norm.helpers import TOLERANCES
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

ML = ttnn.TensorMemoryLayout

CASES = [
    # (rows, W, memory_layout, shard_shape, core_grid)
    (32, 1024, ML.INTERLEAVED, None, None),
    (32, 2304, ML.INTERLEAVED, None, None),
    (32, 5120, ML.INTERLEAVED, None, None),
    (32, 7168, ML.INTERLEAVED, None, None),
    (640, 6144, ML.INTERLEAVED, None, None),
    (640, 7168, ML.INTERLEAVED, None, None),
    (32, 1024, ML.WIDTH_SHARDED, [32, 128], (8, 1)),
    (32, 2304, ML.WIDTH_SHARDED, [32, 256], (9, 1)),
    (32, 5120, ML.WIDTH_SHARDED, [32, 160], (8, 4)),
    (32, 7168, ML.WIDTH_SHARDED, [32, 256], (7, 4)),
    (640, 6144, ML.BLOCK_SHARDED, [64, 704], (9, 10)),
    (640, 7168, ML.BLOCK_SHARDED, [64, 1024], (7, 10)),
]


def _id(c):
    rows, W, ml, _, _ = c
    return f"{rows}x{W}-{str(ml).split('.')[-1]}"


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_decode_perf(device, case):
    rows, W, ml, shard, grid = case
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, W, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    mc = ttnn.DRAM_MEMORY_CONFIG
    if shard is not None:
        mc = shard_config(shard, grid, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    kwargs = {"memory_config": tx.memory_config()} if shard is not None else {}
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, **kwargs)
    xf = x.float()
    exp = (xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * g.float().reshape(-1)).to(torch.bfloat16)
    check_output(
        out,
        exp,
        shape=[1, 1, rows, W],
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=(0.9995, TOLERANCES[ttnn.bfloat16][1]),
    )
    for t in (tx, tg, out):
        ttnn.deallocate(t)


# Regime guard: shapes near the interleaved R1/R3 occupancy boundary, pinned to each regime (for --profile).
GUARD = [(928, 1696), (992, 3000), (2208, 5120), (1504, 544), (224, 3072), (2560, 4096), (3200, 2048), (3200, 7168)]


@pytest.mark.parametrize("regime", ["r1", "r3"])
@pytest.mark.parametrize("rows,W", GUARD, ids=lambda v: str(v))
def test_regime_guard(device, monkeypatch, rows, W, regime):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    if regime == "r1":
        monkeypatch.setattr(pd, "R3_OCCUPANCY_RATIO", 1e9)
    else:
        monkeypatch.setattr(pd, "FORCE_W_SPLIT", True)
    test_decode_perf(device, (rows, W, ML.INTERLEAVED, None, None))


@pytest.mark.parametrize(
    "rows,W,want_r3",
    [
        (32, 7168, True),  # decode: R1 would run 1 core
        (640, 7168, True),  # 20 tile-rows: R1 runs 20 cores, R3 fills the grid
        (8192, 1024, False),  # prefill: R1 already fills the grid -> must not flip
        (1504, 544, False),  # Rt=47, Wt=17: R3 would run fewer cores than R1
    ],
)
def test_r3_occupancy_predicate(device, rows, W, want_r3):
    """Refinement 5's pinned interleaved predicate: R3 iff R3 occupies >= R3_OCCUPANCY_RATIO x R1's cores."""
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    x = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    gm = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    try:
        g = pd._Geometry(x, gm, x, False)
        grid = device.compute_with_storage_grid_size()
        assert pd._take_r3_interleaved(g, grid, pd._cb_budget(device)) == want_r3
    finally:
        ttnn.deallocate(x)
        ttnn.deallocate(gm)
