# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 6: BLOCK_SHARDED prefill _PERF_BASE cases at their EXACT config, for --profile runs.

bf16 TILE x, bf16 TILE gamma, fp32_dest_acc_en=False, HiFi2, pinned (shard_shape, core_grid). Correctness is
checked at the perf cases' soft PCC gate (0.9995); perf is read from the profiler CSV.
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
    # (rows, W, shard_shape, core_grid)
    (7168, 1024, [736, 128], (8, 10)),
    (7168, 2048, [736, 192], (11, 10)),
    (8192, 1024, [1024, 128], (8, 8)),
    (640, 6144, [64, 704], (9, 10)),
    (640, 7168, [64, 1024], (7, 10)),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}x{c[1]}" for c in CASES])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "nogamma"])
def test_block_prefill_perf(device, case, with_gamma):
    rows, W, shard, grid = case
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, W, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    mc = shard_config(shard, grid, ML.BLOCK_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if with_gamma else None
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=tx.memory_config())
    xf = x.float()
    exp = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    if with_gamma:
        exp = exp * g.float().reshape(-1)
    check_output(
        out,
        exp.to(torch.bfloat16),
        shape=[1, 1, rows, W],
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=(0.9995, TOLERANCES[ttnn.bfloat16][1]),
    )
    for t in (tx, tg, out):
        if t is not None:
            ttnn.deallocate(t)
