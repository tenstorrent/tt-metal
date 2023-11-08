# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import tt_lib as ttl
import pytest
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from models.utility_functions import comp_pcc
from loguru import logger


@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 32 * 2, 32 * 2),  # single tile

    ),
)
@skip_for_wormhole_b0
def test_moreh_adam(shape, device):
    torch.manual_seed(0)

    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]

    x_data = torch.randn(N * C * H * W).reshape((N, C, H, W)).to(torch.bfloat16)
    y_data = 2 * x_data + 1

    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(64, 64).to(torch.bfloat16)  # 1 입력 피처, 1 출력

        def forward(self, x):
            return self.linear(x)

    model = LinearRegression()
    print("weigth:", model.linear.weight)
    cpu_exp_avg = torch.zeros_like(model.linear.weight.unsqueeze(0).unsqueeze(0))
    cpu_exp_avg_sq = torch.zeros_like(model.linear.weight.unsqueeze(0).unsqueeze(0))

    dev_param = ttl.tensor.Tensor(
        model.linear.weight.reshape(-1).tolist(),
        (1, 1, *model.linear.weight.shape),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)
    print("dev_param:", model.linear.weight)

    criterion = nn.L1Loss()

    optimizer = optim.Adam({model.linear.weight}, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    optimizer.zero_grad()

    outputs = model(x_data)

    loss = criterion(outputs, y_data)

    loss.backward()
    cpu_grad = model.linear.weight.grad.clone().reshape((N, C, H, W))
    print("cpu_grad:", cpu_grad.shape)

    dev_grad = ttl.tensor.Tensor(
        cpu_grad.reshape(-1).tolist(),
        cpu_grad.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    dev_exp_avg = ttl.tensor.Tensor(
        cpu_exp_avg.reshape(-1).tolist(),
        cpu_exp_avg.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    dev_exp_avg_sq = ttl.tensor.Tensor(
        cpu_exp_avg_sq.reshape(-1).tolist(),
        cpu_exp_avg_sq.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(device)

    optimizer.step()

    print("Updated weigth:", model.linear.weight)

    print("dev_param:", dev_param.shape())
    print("dev_grad:", dev_grad.shape())
    print("dev_exp_avg:", dev_exp_avg.shape())
    print("dev_exp_avg_sq:", dev_exp_avg_sq.shape())

    # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
    ret_list_ = ttl.operations.primary.moreh_adam(dev_param, dev_grad, dev_exp_avg, dev_exp_avg_sq
        , 0.001, 0.9, 0.999, 1e-08, 0.0, 0, False)

    assert dev_param.shape() == list((1, 1, *model.linear.weight.shape))
    print("Updated param end2")
    result = dev_param.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)
    print("Updated param end3")

    print("result:", result)

    passing, out = comp_pcc(model.linear.weight, result)
    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={out}")
    assert passing
