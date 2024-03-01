# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import tt_lib as ttl
import pytest
from models.utility_functions import (
    comp_allclose_and_pcc,
)
from loguru import logger


def create_tt_tensor(tensor, device):
    ret = (
        ttl.tensor.Tensor(
            tensor,
            ttl.tensor.DataType.BFLOAT16,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    return ret


@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 32, 32),  # single
        (12, 6, 64, 64),  # multiple tiles
    ),
)
@pytest.mark.parametrize("lr", [3.0])
@pytest.mark.parametrize("momentum", [0.0, 7.7])
@pytest.mark.parametrize("dampening", [0.0, 0.5])
@pytest.mark.parametrize("weight_decay", [0.0, 2.2])
@pytest.mark.parametrize("nesterov", [True, False], ids=["NESTEROV_TRUE", "NESTEROV_FALSE"])
@pytest.mark.parametrize(
    "momentum_initialized", [True, False], ids=["MOMENTUM_INITIALIZED", "MOMENTUM_NOT_INITIALIZED"]
)
def test_moreh_sgd(shape, lr, momentum, dampening, weight_decay, nesterov, momentum_initialized, device):
    if nesterov and (momentum <= 0 or dampening != 0):
        pytest.skip()

    torch.manual_seed(0)

    # make model and compute grad
    N, C, H, W = shape

    x_data = torch.rand((N, C, H, W)).to(torch.bfloat16)
    y_data = torch.rand((N, C, H, W)).to(torch.bfloat16)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(N, C, H, W).to(torch.bfloat16)).to(torch.bfloat16)

        def forward(self, x):
            return torch.mul(x, self.weight)

    model = SimpleModel()

    criterion = nn.L1Loss()
    optimizer = optim.SGD(
        {model.weight}, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov
    )
    optimizer.zero_grad()

    outputs = model(x_data)
    loss = criterion(outputs, y_data)
    loss.backward()

    # do step for momentum_initialized test
    step_cnt = 2 if momentum_initialized else 1

    cpu_momentum_in = None
    cpu_momentum_out = None
    for i in range(0, step_cnt):
        cpu_param_in = model.weight.clone()
        dev_param_in = create_tt_tensor(cpu_param_in, device)

        optimizer_state_dict = optimizer.state_dict()
        if momentum != 0:
            if 0 in optimizer_state_dict["state"]:
                cpu_momentum_in = optimizer_state_dict["state"][0]["momentum_buffer"].clone()

        optimizer.step()

        optimizer_state_dict = optimizer.state_dict()
        if momentum != 0:
            if 0 in optimizer_state_dict["state"]:
                cpu_momentum_out = optimizer_state_dict["state"][0]["momentum_buffer"].clone()

    # create other dev tensors
    dev_param_out = create_tt_tensor(cpu_param_in, device)

    cpu_grad = model.weight.grad
    dev_grad = create_tt_tensor(cpu_grad, device)

    dev_momentum_buffer_in = None
    dev_momentum_buffer_out = None
    if momentum != 0:
        if momentum_initialized:
            if cpu_momentum_in is not None:
                dev_momentum_buffer_in = create_tt_tensor(cpu_momentum_in, device)
            else:
                dev_momentum_buffer_in = create_tt_tensor(cpu_param_in, device)

        dev_momentum_buffer_out = create_tt_tensor(cpu_param_in, device)

    ttl.operations.primary.moreh_sgd(
        dev_param_in,
        dev_grad,
        dev_momentum_buffer_in,
        dev_param_out,
        dev_momentum_buffer_out,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        momentum_initialized,
    )

    assert dev_param_in.get_legacy_shape() == list(model.weight.shape)

    # check param_out
    param_result = dev_param_out.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=0.99, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing

    # check momentum_out
    if momentum != 0:
        momentum_buffer_result = (
            dev_momentum_buffer_out.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
        )

        passing, out = comp_allclose_and_pcc(cpu_momentum_out, momentum_buffer_result, pcc=0.99, rtol=rtol, atol=atol)
        logger.debug(f"Momentum_out passing (param)={passing}")
        logger.debug(f"Momentum_out pcc={out}")

        assert passing
