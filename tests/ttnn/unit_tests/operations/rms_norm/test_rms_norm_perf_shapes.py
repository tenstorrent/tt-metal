# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""LLM-representative shapes from eval/golden_tests/rms_norm/feature_spec.py's perf cases, run at the
Phase 0 precision corner (HiFi4, fp32 DEST accumulation). Correctness-checked here; the point of the
file is to be profiled by node id:

    scripts/run_safe_pytest.sh --profile \
        "tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf_shapes.py::test_rms_norm_perf_shape[decode_7168]"

The device fixture comes from this directory's conftest (module-scoped device).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

SHAPES = [
    pytest.param((1, 1, 32, 1024), id="decode_1024"),
    pytest.param((1, 1, 32, 7168), id="decode_7168"),
    pytest.param((1, 1, 8192, 1024), id="prefill_1024"),
    pytest.param((1, 1, 8192, 7168), id="prefill_7168"),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_perf_shape(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.float32).to(torch.bfloat16)
    xf, gf = x.float(), g.float()
    expected = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * gf

    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_g = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(ttnn_x, gamma=ttnn_g)
    assert_with_pcc(expected, ttnn.to_torch(out).float(), 0.995)
