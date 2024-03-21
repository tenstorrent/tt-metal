# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import tt_lib as ttl
import pytest
from models.utility_functions import (
    skip_for_wormhole_b0,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
)
from loguru import logger


@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 32, 32),  # single
        (12, 6, 64, 64),  # multi tile
    ),
)
@pytest.mark.parametrize("lr", [0.0, 1e-2])
@pytest.mark.parametrize("betas", ((0.9, 0.999), (0.5, 0.555)))
@pytest.mark.parametrize("eps", [1e-06, 1e-08])
@pytest.mark.parametrize("weight_decay", [0.0, 0.3])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("step", [1, 2, 8])
def test_moreh_adamw(shape, lr, betas, eps, weight_decay, amsgrad, step, device):
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

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
    optimizer = optim.AdamW({model.weight}, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    # until step-1
    for _ in range(step - 1):
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()

    cpu_weight = model.weight.clone()
    if step == 1:
        cpu_exp_avg = torch.zeros_like(model.weight)
        cpu_exp_avg_sq = torch.zeros_like(model.weight)
        cpu_max_exp_avg_sq = torch.zeros_like(model.weight)
    else:
        optimizer_state_dict = optimizer.state_dict()
        cpu_exp_avg = optimizer_state_dict["state"][0]["exp_avg"].clone()
        cpu_exp_avg_sq = optimizer_state_dict["state"][0]["exp_avg_sq"].clone()
        if amsgrad:
            cpu_max_exp_avg_sq = optimizer_state_dict["state"][0]["max_exp_avg_sq"].clone()
        else:
            cpu_max_exp_avg_sq = None

    # last step
    optimizer.zero_grad()
    outputs = model(x_data)
    loss = criterion(outputs, y_data)
    loss.backward()
    optimizer.step()

    cpu_grad = model.weight.grad.clone()
    optimizer_state_dict = optimizer.state_dict()
    cpu_exp_avg_result = optimizer_state_dict["state"][0]["exp_avg"].clone()
    cpu_exp_avg_sq_result = optimizer_state_dict["state"][0]["exp_avg_sq"].clone()
    if amsgrad:
        cpu_max_exp_avg_sq_result = optimizer_state_dict["state"][0]["max_exp_avg_sq"].clone()
    else:
        cpu_max_exp_avg_sq_result = None

    dev_grad = (
        ttl.tensor.Tensor(
            cpu_grad,
            ttl.tensor.DataType.BFLOAT16,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    dev_param = (
        ttl.tensor.Tensor(
            cpu_weight,
            ttl.tensor.DataType.BFLOAT16,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    dev_exp_avg = (
        ttl.tensor.Tensor(
            cpu_exp_avg,
            ttl.tensor.DataType.BFLOAT16,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    dev_exp_avg_sq = (
        ttl.tensor.Tensor(
            cpu_exp_avg_sq,
            ttl.tensor.DataType.BFLOAT16,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    if amsgrad:
        dev_max_exp_avg_sq = (
            ttl.tensor.Tensor(
                cpu_max_exp_avg_sq,
                ttl.tensor.DataType.BFLOAT16,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )

    if amsgrad:
        ret_list_ = ttl.operations.primary.moreh_adamw(
            dev_param,
            dev_grad,
            dev_exp_avg,
            dev_exp_avg_sq,
            lr,
            betas[0],
            betas[1],
            eps,
            weight_decay,
            step,
            amsgrad,
            dev_max_exp_avg_sq,
        )
    else:
        ret_list_ = ttl.operations.primary.moreh_adamw(
            dev_param,
            dev_grad,
            dev_exp_avg,
            dev_exp_avg_sq,
            lr,
            betas[0],
            betas[1],
            eps,
            weight_decay,
            step,
            amsgrad,
        )

    assert dev_param.get_legacy_shape() == list(model.weight.shape)

    param_result = dev_param.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    exp_avg_result = dev_exp_avg.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    exp_avg_sq_result = dev_exp_avg_sq.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    if amsgrad:
        max_exp_avg_sq_result = dev_max_exp_avg_sq.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    else:
        max_exp_avg_sq_result = None

    whole_passing = True

    rtol = atol = 0.1
    pcc = 0.99
    passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=pcc, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")
    whole_passing &= passing

    passing, out = comp_allclose_and_pcc(cpu_exp_avg_result, exp_avg_result, pcc=pcc, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (exp_avg)={passing}")
    logger.debug(f"Output pcc={out}")
    whole_passing &= passing

    passing, out = comp_allclose_and_pcc(cpu_exp_avg_sq_result, exp_avg_sq_result, pcc=pcc, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (exp_avg_sq)={passing}")
    logger.debug(f"Output pcc={out}")
    whole_passing &= passing

    if amsgrad:
        passing, out = comp_allclose_and_pcc(
            cpu_max_exp_avg_sq_result, max_exp_avg_sq_result, pcc=pcc, rtol=rtol, atol=atol
        )
        logger.debug(f"Out passing (max_exp_avg_sq)={passing}")
        logger.debug(f"Output pcc={out}")
        whole_passing &= passing

    assert whole_passing
