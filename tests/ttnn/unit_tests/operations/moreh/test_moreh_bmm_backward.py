# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
