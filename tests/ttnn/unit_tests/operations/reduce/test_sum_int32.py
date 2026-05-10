# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Tests for INT32 ttnn.sum (issue #26724, plan in #43736).
#
# Background: the FPU GMPOOL primitive that ttnn's reduce ops historically
# relied on silently produces zeros for INT32 inputs.  The SUM extension of
# the SFPU reduce path (compute_kernel_lib::reduce_sfpu, compiled into
# reduce_sfpu.cpp) handles INT32 SUM along H or W; full-tensor (HW) reductions
# are decomposed at the prim layer into a W reduce followed by an H reduce
# (see reduce_op.cpp's use_two_step_hw_reduce_for_int32 flag).

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device


# Value range for inputs.  We pick a small absolute bound so that any
# accumulation across the largest test tensor (2 * 4 * 64 * 60 = 30720
# elements) cannot overflow int32 (|sum| < 30720 * 1000 ~= 3e7 << 2^31).
# This lets us compare exactly against torch.sum without worrying about
# wraparound semantics.
_VALUE_BOUND = 1000


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),  # single tile (issue #26724-style smallest case)
        (1, 1, 64, 60),  # 2x2 tiles with partial tile in W
        (1, 1, 100, 120),  # 4x4 tiles with partial tile in both H and W
        (1, 1, 30, 96),  # Ht=1, Wt=3 (no H accumulation needed)
        (1, 1, 90, 32),  # Ht=3, Wt=1 (no W accumulation needed)
        (2, 3, 64, 64),  # multi-batch (NC>1) with whole tiles
        (1, 3, 17, 19),  # exact #26724 repro shape (non-tile-aligned, NC=3)
        (2, 4, 64, 60),  # multi-batch + partial W tile
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        # Single-axis cases
        -1,  # W axis  -> MULTI_CORE_W
        -2,  # H axis  -> MULTI_CORE_H
        # Non-H/W axes drive the front-end's transpose-then-reduce path
        # (call_fast_nc rejects INT32 today, so we exercise the slow path).
        0,
        1,
        # Multi-axis / full-tensor cases drive the W-then-H decomposition
        # in reduce_op.cpp (use_two_step_hw_reduce_for_int32).
        (-1, -2),
        None,
    ],
)
def test_sum_int32(device, input_shape, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.randint(
        -_VALUE_BOUND,
        _VALUE_BOUND + 1,
        input_shape,
        dtype=torch.int32,
    )

    # torch.sum on int32 promotes to int64; we keep both sides in int64 for
    # the value comparison and verify the on-device dtype separately.
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    ttnn_output_tensor = ttnn.sum(input_tensor, dim=dim)
    ttnn_output_torch = ttnn.to_torch(ttnn_output_tensor)

    assert ttnn_output_torch.dtype == torch.int32, f"Expected int32 output, got {ttnn_output_torch.dtype}"

    # Align dtypes (torch.sum returns int64) and shapes (ttnn keeps trailing
    # singleton dims for tile alignment; reshape to torch's logical output
    # shape) before exact comparison.
    actual = ttnn_output_torch.to(torch.int64).reshape(torch_output_tensor.shape)
    assert_equal(actual, torch_output_tensor)


# Direct repro from issue #26724 (the all-zeros symptom).  Kept as its own
# test so a future regression on this exact failure mode is easy to spot.
def test_sum_int32_issue_26724_repro(device):
    torch.manual_seed(0)
    x = torch.randint(-_VALUE_BOUND, _VALUE_BOUND + 1, (1, 3, 17, 19), dtype=torch.int32)
    x_ttnn = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)

    # (A) Reduce over all dims -> scalar.
    torch_sum_all = torch.sum(x)
    ttnn_sum_all = ttnn.to_torch(ttnn.sum(x_ttnn))
    assert ttnn_sum_all.dtype == torch.int32
    assert_equal(ttnn_sum_all.to(torch.int64).reshape(torch_sum_all.shape), torch_sum_all)

    # (B) Reduce over a specific dim.
    torch_sum_dim = torch.sum(x, dim=2, keepdim=False)
    ttnn_sum_dim = ttnn.to_torch(ttnn.sum(x_ttnn, dim=2, keepdim=False))
    assert ttnn_sum_dim.dtype == torch.int32
    assert_equal(ttnn_sum_dim.to(torch.int64).reshape(torch_sum_dim.shape), torch_sum_dim)
