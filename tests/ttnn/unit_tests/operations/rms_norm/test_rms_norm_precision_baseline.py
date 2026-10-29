# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm (Phase 0: TILE, interleaved, fp32_dest_acc_en=True, HiFi4).

Prints one PRECISION_BASELINE line per case (run with -s). The got/true ratio spread is the
scale-bug detector: a tight cluster around a constant != 1.0 is a structural error, not rounding.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.comparison import ulp
from ttnn.operations.rms_norm import rms_norm

PCC_BY_DTYPE = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
EPSILON = 1e-6


def reference(x, gamma):
    x32 = x.to(torch.float32)
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPSILON) * gamma.to(torch.float32).reshape(-1)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 64), (2, 1, 100, 47), (2, 4, 128, 512), (1, 1, 128, 4096)],
    ids=["small", "non_aligned", "medium", "wide"],
)
def test_rms_norm_precision_baseline(device, shape, dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    torch_gamma = torch.randn(shape[-1], dtype=torch.float32).to(TORCH_DTYPE[dtype])
    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    actual = ttnn.to_torch(rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=EPSILON)).to(torch.float32)
    expected = reference(torch_input, torch_gamma)

    abs_err = (actual - expected).abs()
    rel_rms = (abs_err.pow(2).mean().sqrt() / expected.std()).item()
    ulp_err = abs_err / ulp(expected.to(TORCH_DTYPE[dtype])).to(torch.float32)
    nonzero = expected.abs() > 1e-3
    ratio = actual[nonzero] / expected[nonzero]
    _, allclose_msg = comp_allclose(actual, expected)
    _, pcc_msg = assert_with_pcc(expected, actual, PCC_BY_DTYPE[dtype])
    print(
        f"PRECISION_BASELINE shape={shape} dtype={dtype} pcc={pcc_msg} "
        f"max_abs={abs_err.max().item():.3e} mean_abs={abs_err.mean().item():.3e} rel_rms={rel_rms:.3e} "
        f"max_ulp={ulp_err.max().item():.1f} mean_ulp={ulp_err.mean().item():.3f} "
        f"ratio_median={ratio.median().item():.5f} ratio_p5={ratio.quantile(0.05).item():.5f} "
        f"ratio_p95={ratio.quantile(0.95).item():.5f} allclose=[{allclose_msg}]"
    )
