# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_matmul import moreh_matmul_backward

pytestmark = pytest.mark.use_module_device

# The op has no kernels of its own: it calls moreh_matmul for each requested grad, then moreh_sum when that grad's
# batch dims were broadcast. The dot route (scalar output_grad, 1-D inputs) only forwards to moreh_dot_backward, which
# #58420 deprecates, so it isn't covered here.


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "params, requires_grad",
    [
        # (input, other, output)
        (([32, 64], [64, 96], [32, 96]), (True, True)),
        (([32, 64], [64, 96], [32, 96]), (True, False)),
        (([32, 64], [64, 96], [32, 96]), (False, True)),
        # Both inputs broadcast over batch dims, so each grad goes through moreh_matmul and then moreh_sum.
        (([2, 32, 64], [3, 1, 64, 96], [3, 2, 32, 96]), (True, True)),
    ],
    ids=["both_grads", "input_grad_only", "other_grad_only", "broadcast_batch"],
)
def test_moreh_matmul_backward(params, requires_grad, device):
    torch.manual_seed(0)
    moreh_matmul_backward(params, requires_grad, device)


@pytest.mark.merge_gate
def test_moreh_matmul_backward_program_cache(device):
    torch.manual_seed(0)
    params = ([32, 64], [64, 96], [32, 96])
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    moreh_matmul_backward(params, (True, True), device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses. Row-major,
    # so creating it runs no device program of its own.
    tt_placeholder = ttnn.from_torch(torch.zeros([32, 96]), dtype=ttnn.bfloat16, device=device)
    moreh_matmul_backward(params, (True, True), device)
    assert device.num_program_cache_entries() == num_program_cache_entries
