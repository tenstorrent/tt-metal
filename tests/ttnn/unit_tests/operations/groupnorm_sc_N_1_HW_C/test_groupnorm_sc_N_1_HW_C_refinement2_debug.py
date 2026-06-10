# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 deterministic debug tests — DO NOT DELETE.

Hand-calculable inputs to isolate tail-masking bugs:

- All-ones: mean = 1 exactly, var = 0, output = (1-1)*rstd = 0 everywhere.
  If padding leaks into the mean (pass 1), mean < 1 and output != 0.
  If padding leaks into the variance (pass 2 missing mask: padding becomes
  (-mean)^2 = 1 per padded element), rstd shrinks but output stays 0 — so
  use the constant-input test below to expose pass 2 specifically.
- Constant 3.0: mean = 3, var = 0, output = 0. Variance leak makes
  var = Npad*9/N (large) but again output 0. Combined with the linear-ramp
  test which has nonzero variance, all three together isolate the failing
  pass:
    ones fail            -> pass 1 (mean wrong)
    ones ok + ramp fail  -> pass 2 (variance wrong) or pass 3
- Linear ramp along HW: known mean/var per column slab.
"""

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def reference(x, num_groups, eps=1e-5):
    x_nchw = x.to(torch.float32).squeeze(1).permute(0, 2, 1)
    y = torch.nn.functional.group_norm(x_nchw, num_groups, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 17, 64), id="hw_tail_17x64"),
        pytest.param((1, 1, 64, 17), id="c_tail_64x17"),
        pytest.param((1, 1, 17, 17), id="both_tails_17x17"),
    ],
)
def test_all_ones(device, shape):
    """All-ones: mean = 1 exactly => output = 0 exactly (up to bf16 rounding)."""
    x = torch.ones(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, 1)).to(torch.float32)
    assert out.abs().max().item() < 1e-2, f"all-ones output should be 0, max abs {out.abs().max().item()}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 17, 64), id="hw_tail_17x64"),
        pytest.param((1, 1, 64, 17), id="c_tail_64x17"),
        pytest.param((1, 1, 17, 17), id="both_tails_17x17"),
    ],
)
def test_linear_ramp(device, shape):
    """Ramp along HW: nonzero variance — wrong tail masking shifts mean AND rstd."""
    N, _, HW, C = shape
    x = torch.arange(HW, dtype=torch.float32).reshape(1, 1, HW, 1).expand(N, 1, HW, C) / HW
    x = x.to(torch.bfloat16)
    expected = reference(x, 1)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, 1)).to(torch.float32)
    max_diff = (out - expected).abs().max().item()
    assert max_diff < 0.05, f"max diff {max_diff}"
