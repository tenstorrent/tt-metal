# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3: interleaved _PERF_BASE prefill cases at their EXACT config, for --profile runs.

bf16 TILE x, bf16 TILE gamma, fp32_dest_acc_en=False, HiFi2, DRAM interleaved. Correctness is checked
at the perf cases' soft PCC gate (0.9995); perf is read from the profiler CSV.
"""

import pytest
import torch
import ttnn

from eval.metrics import check_output
from eval.golden_tests.rms_norm.helpers import TOLERANCES
from ttnn.operations.rms_norm import rms_norm

PREFILL = [(8192, 1024), (8192, 2304), (8192, 5120), (8192, 7168), (7168, 1024), (7168, 2048)]


@pytest.mark.parametrize("rows,W", PREFILL, ids=lambda v: str(v))
def test_prefill_perf(device, rows, W):
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, W, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
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
