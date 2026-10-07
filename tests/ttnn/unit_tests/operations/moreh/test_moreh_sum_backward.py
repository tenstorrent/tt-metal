# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_sum import (
    get_backward_tensors,
    get_tensors,
    moreh_sum_backward,
)
from tests.ttnn.unit_tests.operations.test_utils import get_compute_kernel_options

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


def run_moreh_sum_backward_test(input_shape, dim, device):
    # Not nightly moreh_sum_backward: it reseeds to 2023 on every call, so a program-cache test would see the same
    # data on both runs and miss a cache hit that reads a stale address. Same steps with nightly's tensor builders.
    tt_input, _, tt_output_shape, torch_output_shape, torch_input = get_tensors(input_shape, dim, device, keepdim=True)
    tt_output_grad, tt_input_grad, torch_output_grad = get_backward_tensors(
        tt_output_shape, torch_output_shape, input_shape, device
    )
    torch.sum(torch_input, dim, True).backward(torch_output_grad)

    tt_input_grad = ttnn.moreh_sum_backward(
        tt_output_grad,
        input=tt_input,
        dim=dim,
        keepdim=True,
        input_grad=tt_input_grad,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, output_pcc = comp_allclose_and_pcc(
        torch_input.grad, ttnn.to_torch(tt_input_grad), pcc=0.999, rtol=0.1, atol=0.1
    )
    assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_sum_backward_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_sum_backward_test([2, 3, 63, 63], 3, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses. Row-major,
    # so creating it runs no device program of its own.
    tt_placeholder = ttnn.from_torch(torch.zeros([2, 3, 63, 63]), dtype=ttnn.bfloat16, device=device)
    run_moreh_sum_backward_test([2, 3, 63, 63], 3, device)
    assert device.num_program_cache_entries() == num_program_cache_entries
