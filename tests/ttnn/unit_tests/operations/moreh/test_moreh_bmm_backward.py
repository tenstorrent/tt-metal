# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_bmm import run_moreh_bmm_backward

pytestmark = pytest.mark.use_module_device

# The op has no kernels of its own: input_grad = moreh_matmul(output_grad, mat2^T) and mat2_grad =
# moreh_matmul(input^T, output_grad), each only when requested. compute_kernel_options=True: with fp32 accumulation the
# nightly helper checks pcc >= 0.998 instead of 0.97.


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "requires_grad",
    [(True, True), (True, False), (False, True)],
    ids=["both_grads", "input_grad_only", "mat2_grad_only"],
)
def test_moreh_bmm_backward(requires_grad, device):
    torch.manual_seed(0)
    # [batch, m, k, n]
    run_moreh_bmm_backward([2, 32, 64, 96], requires_grad, True, device)


@pytest.mark.merge_gate
def test_moreh_bmm_backward_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_bmm_backward([2, 32, 64, 96], (True, True), True, device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses. Row-major,
    # so creating it runs no device program of its own.
    tt_placeholder = ttnn.from_torch(torch.zeros([2, 32, 96]), dtype=ttnn.bfloat16, device=device)
    run_moreh_bmm_backward([2, 32, 64, 96], (True, True), True, device)
    assert device.num_program_cache_entries() == num_program_cache_entries
