# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_sum import moreh_sum_backward

pytestmark = pytest.mark.use_module_device

# input_grad is output_grad broadcast back over the reduced dims. Reducing W or H sets the wt_need_bcast /
# ht_need_bcast compile-time args; the other dims broadcast through reader runtime args. 63 x 63 leaves the last tile
# in H and W partly filled.


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "dim",
    [3, 2, [2, 3], 1, None],
    ids=["w", "h", "hw", "c", "all_dims"],
)
def test_moreh_sum_backward(dim, device):
    torch.manual_seed(0)
    assert moreh_sum_backward([2, 3, 63, 63], dim, True, True, False, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, dim, keepdim, use_provide_output, fp32_dest_acc_en",
    [
        ([2, 3, 63, 63], [0, 1], False, True, False),
        ([2, 3, 63, 63], 3, True, True, True),
        # No input_grad passed: the op allocates it with the input's shape.
        ([2, 3, 63, 63], [1, 3], True, False, False),
        # 149 tiles: a prime above any device's core count, so the work split leaves a second core group and the
        # factory builds its second compute kernel.
        ([1, 1, 32, 149 * 32], 2, True, True, False),
    ],
    ids=["keepdim_false", "fp32_dest_acc", "allocated_input_grad", "core_group_2"],
)
def test_moreh_sum_backward_corner_cases(input_shape, dim, keepdim, use_provide_output, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    assert moreh_sum_backward(input_shape, dim, keepdim, use_provide_output, fp32_dest_acc_en, device)
