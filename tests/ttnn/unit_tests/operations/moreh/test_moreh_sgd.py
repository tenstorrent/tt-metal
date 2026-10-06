# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device


def run_moreh_sgd_test(
    shape, lr, momentum, dampening, weight_decay, nesterov, device, momentum_initialized=True, has_param_out=True
):
    x = torch.rand(shape).to(torch.bfloat16)
    y = torch.rand(shape).to(torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(shape).to(torch.bfloat16))
    torch.nn.functional.l1_loss(x * weight, y).backward()
    optimizer = torch.optim.SGD(
        [weight], lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov
    )
    if momentum_initialized:
        # The first step creates the momentum buffer, so the device op runs the second one.
        optimizer.step()

    state = optimizer.state[weight]
    tt_param_in = create_ttnn_tilized_tensor(weight.detach(), device, ttnn.bfloat16)
    tt_grad = create_ttnn_tilized_tensor(weight.grad, device, ttnn.bfloat16)
    tt_momentum_buffer_in = (
        create_ttnn_tilized_tensor(state["momentum_buffer"], device, ttnn.bfloat16) if momentum_initialized else None
    )
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    tt_param_out = create_ttnn_tilized_tensor(zeros, device, ttnn.bfloat16) if has_param_out else None
    tt_momentum_buffer_out = create_ttnn_tilized_tensor(zeros, device, ttnn.bfloat16) if momentum != 0 else None
    optimizer.step()

    tt_outputs = ttnn.moreh_sgd(
        tt_param_in,
        tt_grad,
        tt_momentum_buffer_in,
        param_out=tt_param_out,
        momentum_buffer_out=tt_momentum_buffer_out,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        momentum_initialized=momentum_initialized,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    torch_outputs = [weight.detach()]
    if momentum != 0:
        torch_outputs.append(state["momentum_buffer"])
    else:
        assert tt_outputs[1] is None
    for expected, actual in zip(torch_outputs, tt_outputs):
        passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(actual), pcc=0.99, rtol=0.05, atol=0.05)
        assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_sgd(device):
    torch.manual_seed(0)
    # nesterov=True needs dampening=0 (torch rejects other values); dampening is a scalar, not a kernel branch.
    run_moreh_sgd_test([32, 32], lr=3.0, momentum=7.7, dampening=0.0, weight_decay=2.2, nesterov=True, device=device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, momentum, dampening, weight_decay, nesterov, momentum_initialized, has_param_out",
    [
        # momentum != 0, momentum_initialized, nesterov and weight_decay != 0 each set a separate kernel define.
        ([32, 32], 0.0, 0.0, 2.2, False, False, True),
        ([32, 32], 7.7, 0.0, 2.2, True, False, True),
        ([32, 32], 7.7, 0.5, 2.2, False, True, True),
        ([32, 32], 7.7, 0.0, 0.0, True, True, True),
        ([32, 32], 7.7, 0.0, 2.2, True, True, False),
        ([1, 1, 30, 32], 7.7, 0.0, 2.2, True, True, True),
        ([1, 1, 32, 40], 7.7, 0.0, 2.2, True, True, True),
    ],
    ids=["no_momentum", "first_step", "dampening", "no_weight_decay", "no_param_out", "h_partial", "w_partial"],
)
def test_moreh_sgd_corner_cases(
    shape, momentum, dampening, weight_decay, nesterov, momentum_initialized, has_param_out, device
):
    torch.manual_seed(0)
    run_moreh_sgd_test(
        shape,
        lr=3.0,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        device=device,
        momentum_initialized=momentum_initialized,
        has_param_out=has_param_out,
    )


@pytest.mark.merge_gate
def test_moreh_sgd_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_sgd_test([32, 32], lr=3.0, momentum=7.7, dampening=0.0, weight_decay=2.2, nesterov=True, device=device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros([32, 32]), device, ttnn.bfloat16)
    run_moreh_sgd_test([32, 32], lr=3.0, momentum=7.7, dampening=0.0, weight_decay=2.2, nesterov=True, device=device)
    assert device.num_program_cache_entries() == num_program_cache_entries
