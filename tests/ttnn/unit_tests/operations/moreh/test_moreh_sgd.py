# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

# Nightly has no moreh_sgd helper (its logic is inline in test_* functions), so this file has its own.


def run_moreh_sgd_test(
    shape,
    momentum,
    dampening,
    weight_decay,
    nesterov,
    device,
    momentum_initialized=True,
    provide_outputs=True,
    fp32_dest_acc_en=False,
):
    x = torch.rand(shape).to(torch.bfloat16)
    y = torch.rand(shape).to(torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(shape).to(torch.bfloat16))
    # Summed, so the gradient stays around 1, well above the tolerance. lr, momentum and weight_decay stay near 1:
    # nightly's larger values push intermediates into the hundreds, where bfloat16 loses the result.
    torch.nn.functional.l1_loss(x * weight, y, reduction="sum").backward()
    optimizer = torch.optim.SGD(
        [weight], lr=1.0, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov
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
    # NaN, so an output the op never writes fails.
    nans = torch.full(shape, float("nan"), dtype=torch.bfloat16)
    tt_param_out = create_ttnn_tilized_tensor(nans, device, ttnn.bfloat16) if provide_outputs else None
    tt_momentum_buffer_out = (
        create_ttnn_tilized_tensor(nans, device, ttnn.bfloat16) if provide_outputs and momentum != 0 else None
    )
    optimizer.step()

    result = ttnn.moreh_sgd(
        tt_param_in,
        tt_grad,
        tt_momentum_buffer_in,
        param_out=tt_param_out,
        momentum_buffer_out=tt_momentum_buffer_out,
        lr=1.0,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        momentum_initialized=momentum_initialized,
        compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
    )
    # Check the provided buffers themselves, not the return value.
    actual_outputs = [tt_param_out, tt_momentum_buffer_out] if provide_outputs else result

    expected_outputs = [weight.detach()]
    if momentum != 0:
        expected_outputs.append(state["momentum_buffer"])
    else:
        # Without momentum there is no buffer to return.
        assert result[1] is None
    for expected, actual in zip(expected_outputs, actual_outputs):
        passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(actual), pcc=0.99, rtol=0.05, atol=0.05)
        assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "momentum, dampening, weight_decay, nesterov, momentum_initialized",
    [
        (0.0, 0.0, 0.0, False, False),
        (0.0, 0.0, 0.5, False, False),
        (0.9, 0.0, 0.0, False, False),
        # nesterov needs dampening = 0 (torch rejects other values).
        (0.9, 0.0, 0.5, True, False),
        (0.9, 0.5, 0.5, False, True),
        (0.9, 0.0, 0.5, True, True),
        (0.9, 0.0, 0.0, True, True),
    ],
    ids=[
        "plain",
        "weight_decay_only",
        "momentum_first_step",
        "nesterov_first_step",
        "dampening",
        "nesterov",
        "no_weight_decay",
    ],
)
def test_moreh_sgd(momentum, dampening, weight_decay, nesterov, momentum_initialized, device):
    torch.manual_seed(0)
    run_moreh_sgd_test(
        [32, 32], momentum, dampening, weight_decay, nesterov, device, momentum_initialized=momentum_initialized
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, provide_outputs, fp32_dest_acc_en",
    [
        ([32, 32], True, True),
        ([32, 32], False, False),
        # Regression for #51278: the tile count came from the logical shape and skipped partial tiles.
        ([1, 1, 30, 32], True, False),
        ([1, 1, 32, 40], True, False),
        ([32, 149 * 32], True, False),
    ],
    ids=["fp32_dest_acc", "allocated_outputs", "h_partial", "w_partial", "core_group_2"],
)
def test_moreh_sgd_corner_cases(shape, provide_outputs, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_sgd_test(
        shape, 0.9, 0.0, 0.5, True, device, provide_outputs=provide_outputs, fp32_dest_acc_en=fp32_dest_acc_en
    )
