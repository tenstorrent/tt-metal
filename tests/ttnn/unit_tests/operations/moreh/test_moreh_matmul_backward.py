# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
