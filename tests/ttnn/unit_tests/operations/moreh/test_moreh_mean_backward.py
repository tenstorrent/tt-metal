# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_mean import run_moreh_mean_backward

pytestmark = pytest.mark.use_module_device

# input_grad is output_grad broadcast back over the reduced dims, divided by how many values were averaged. Reducing
# W or H sets the wt_need_bcast / ht_need_bcast compile-time args; the other dims broadcast through reader runtime
# args. 63 x 63 leaves the last tile in H and W partly filled.


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "dim",
    [[3], [2], [2, 3], [1], None],
    ids=["w", "h", "hw", "c", "all_dims"],
)
def test_moreh_mean_backward(dim, device):
    torch.manual_seed(0)
    run_moreh_mean_backward([[2, 3, 63, 63], dim], device, keepdim=True, compute_kernel_options=False)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_dim, keepdim, fp32_dest_acc_en, create_input_grad",
    [
        # The nightly helper skips keepdim=False on the last two dims, so this drops N and C.
        ([[2, 3, 63, 63], [0, 1]], False, False, False),
        ([[3, 4, 5, 17, 22], [2]], True, False, False),
        ([[2, 3, 63, 63], [3]], True, True, False),
        # No input_grad passed: the op allocates it from input_grad_shape.
        ([[2, 3, 63, 63], [1, 3]], True, False, True),
        # 149 tiles: a prime above any device's core count, so the work split leaves a second core group and the
        # factory builds its second compute kernel.
        ([[1, 1, 32, 149 * 32], [2]], True, False, False),
    ],
    ids=["keepdim_false", "rank_5", "fp32_dest_acc", "allocated_input_grad", "core_group_2"],
)
def test_moreh_mean_backward_corner_cases(input_shape_dim, keepdim, fp32_dest_acc_en, create_input_grad, device):
    torch.manual_seed(0)
    run_moreh_mean_backward(
        input_shape_dim,
        device,
        keepdim=keepdim,
        compute_kernel_options=fp32_dest_acc_en,
        create_input_grad=create_input_grad,
    )


@pytest.mark.merge_gate
def test_moreh_mean_backward_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_mean_backward([[2, 3, 63, 63], [3]], device, keepdim=True, compute_kernel_options=False)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses. Row-major,
    # so creating it runs no device program of its own.
    tt_placeholder = ttnn.from_torch(torch.zeros([2, 3, 63, 63]), dtype=ttnn.bfloat16, device=device)
    run_moreh_mean_backward([[2, 3, 63, 63], [3]], device, keepdim=True, compute_kernel_options=False)
    assert device.num_program_cache_entries() == num_program_cache_entries
