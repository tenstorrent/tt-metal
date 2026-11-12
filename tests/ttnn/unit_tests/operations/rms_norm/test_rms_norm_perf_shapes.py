# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Representative shapes for on-device profiling (scripts/run_safe_pytest.sh --profile).

Correctness is checked; perf is read from the profiler CSV (DEVICE KERNEL DURATION).
"""

import pytest
import torch
import ttnn

from eval.metrics import check_output
from eval.golden_tests.rms_norm.helpers import TOLERANCES
from ttnn.operations.rms_norm import rms_norm


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("rows,W", [(8192, 1024), (8192, 4096), (32, 4096)], ids=lambda v: str(v))
def test_perf_shape(device, rows, W, layout):
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, W, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = rms_norm(tx, gamma=tg)
    xf = x.float()
    exp = (xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * g.float()).to(torch.bfloat16)
    check_output(
        out,
        exp,
        shape=[1, 1, rows, W],
        dtype=ttnn.bfloat16,
        expected_layout=layout,
        tolerance=TOLERANCES[ttnn.bfloat16],
    )
