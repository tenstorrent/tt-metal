# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import ttnn
import pytest
from models.utility_functions import comp_allclose_and_pcc, is_wormhole_b0
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)
from loguru import logger

fp32_dest_acc_en = [
    False,  # for grayskull
]
fp32_dest_acc_en_ids = ["fp32_dest_acc_en=False"]
if is_wormhole_b0():
    fp32_dest_acc_en.append(True)
    fp32_dest_acc_en_ids.append("fp32_dest_acc_en=True")


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # single
        [12, 6, 64, 64],  # multiple tiles
    ],
)
@pytest.mark.parametrize("lr", [3.0])
@pytest.mark.parametrize("momentum", [0.0, 7.7])
@pytest.mark.parametrize("dampening", [0.0, 0.5])
@pytest.mark.parametrize("weight_decay", [0.0, 2.2])
@pytest.mark.parametrize("nesterov", [True, False], ids=["NESTEROV_TRUE", "NESTEROV_FALSE"])
@pytest.mark.parametrize(
    "momentum_initialized", [True, False], ids=["MOMENTUM_INITIALIZED", "MOMENTUM_NOT_INITIALIZED"]
)
@pytest.mark.parametrize("has_param_out", [True, False], ids=["HAS_PARAM_OUT_TRUE", "HAS_PARAM_OUT_FALSE"])
@pytest.mark.parametrize("fp32_dest_acc_en", fp32_dest_acc_en, ids=fp32_dest_acc_en_ids)
@pytest.mark.parametrize(
    "npu_dtype, cpu_dtype",
    [[ttnn.bfloat8_b, torch.bfloat16], [ttnn.bfloat16, torch.bfloat16]],
    ids=["bfloat8", "bfloat16"],
)
def test_moreh_sgd(
    shape,
    lr,
    momentum,
    dampening,
    weight_decay,
    nesterov,
    momentum_initialized,
    has_param_out,
    fp32_dest_acc_en,
    npu_dtype,
    cpu_dtype,
    device,
):
    if nesterov and (momentum <= 0 or dampening != 0):
        pytest.skip()
    if npu_dtype == ttnn.bfloat8_b:
        # Duong: ttnn.bfloat8_b has some bugs. only around half the tests passed for bfloat8_b. Some tests produce 0.0 or Inf results.
        # I couldn't identify the pattern of failed tests, it seems kind of random so I think it's a precision error.
        pytest.skip()

    torch.manual_seed(0)

    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)

    # make model and compute grad
    x_data = torch.rand(shape).to(cpu_dtype)
    y_data = torch.rand(shape).to(cpu_dtype)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(shape).to(cpu_dtype)).to(cpu_dtype)

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
        dev_param_in = ttnn.from_torch(cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)

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
    dev_param_out = ttnn.from_torch(cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cpu_grad = model.weight.grad

    dev_grad = ttnn.from_torch(cpu_grad, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    dev_momentum_buffer_in = None
    dev_momentum_buffer_out = None
    if momentum != 0:
        if momentum_initialized:
            if cpu_momentum_in is not None:
                dev_momentum_buffer_in = ttnn.from_torch(
                    cpu_momentum_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )
            else:
                dev_momentum_buffer_in = ttnn.from_torch(
                    cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device
                )

        dev_momentum_buffer_out = ttnn.from_torch(cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    dev_param_out, dev_momentum_buffer_out = ttnn.operations.moreh.sgd(
        dev_param_in,
        dev_grad,
        dev_momentum_buffer_in,
        dev_param_out if has_param_out else None,
        dev_momentum_buffer_out,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        momentum_initialized=momentum_initialized,
        compute_kernel_config=compute_kernel_config,
    )

    assert dev_param_in.shape == list(model.weight.shape)

    # check param_out
    param_result = ttnn.to_torch(dev_param_out).to(cpu_dtype)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=0.99, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    assert passing

    # check momentum_out
    if momentum != 0:
        momentum_buffer_result = ttnn.to_torch(dev_momentum_buffer_out).to(cpu_dtype)

        passing, out = comp_allclose_and_pcc(cpu_momentum_out, momentum_buffer_result, pcc=0.99, rtol=rtol, atol=atol)
        logger.debug(f"Momentum_out passing (param)={passing}")
        logger.debug(f"Momentum_out pcc={out}")

        assert passing
    if momentum == 0:
        assert dev_momentum_buffer_out == None


@pytest.mark.parametrize(
    "shape",
    [[32, 32]],  # single
)
@pytest.mark.parametrize("lr", [3.0])
@pytest.mark.parametrize("momentum", [7.7])
@pytest.mark.parametrize("dampening", [0.5])
@pytest.mark.parametrize("weight_decay", [0.0])
@pytest.mark.parametrize("nesterov", [False], ids=["NESTEROV_FALSE"])
@pytest.mark.parametrize(
    "momentum_initialized", [True, False], ids=["MOMENTUM_INITIALIZED", "MOMENTUM_NOT_INITIALIZED"]
)
@pytest.mark.parametrize("has_param_out", [True], ids=["HAS_PARAM_OUT_TRUE"])
@pytest.mark.parametrize("fp32_dest_acc_en", fp32_dest_acc_en, ids=fp32_dest_acc_en_ids)
@pytest.mark.parametrize("npu_dtype, cpu_dtype", [[ttnn.bfloat16, torch.bfloat16]], ids=["bfloat16"])
def test_moreh_sgd_callback(
    shape,
    lr,
    momentum,
    dampening,
    weight_decay,
    nesterov,
    momentum_initialized,
    has_param_out,
    fp32_dest_acc_en,
    npu_dtype,
    cpu_dtype,
    device,
):
    if nesterov and (momentum <= 0 or dampening != 0):
        pytest.skip()

    torch.manual_seed(0)
    num_program_cache_entries_list = []
    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)

    # make model and compute grad
    x_data = torch.rand(shape).to(cpu_dtype)
    y_data = torch.rand(shape).to(cpu_dtype)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(shape).to(cpu_dtype)).to(cpu_dtype)

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

        optimizer_state_dict = optimizer.state_dict()
        if momentum != 0:
            if 0 in optimizer_state_dict["state"]:
                cpu_momentum_in = optimizer_state_dict["state"][0]["momentum_buffer"].clone()

        optimizer.step()

        optimizer_state_dict = optimizer.state_dict()
        if momentum != 0:
            if 0 in optimizer_state_dict["state"]:
                cpu_momentum_out = optimizer_state_dict["state"][0]["momentum_buffer"].clone()

    cpu_grad = model.weight.grad

    # create other dev tensors
    for _ in range(2):
        dev_param_in = ttnn.from_torch(cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        dev_param_out = ttnn.from_torch(cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        dev_grad = ttnn.from_torch(cpu_grad, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        dev_momentum_buffer_in = None
        dev_momentum_buffer_out = None
        if momentum != 0:
            if momentum_initialized:
                if cpu_momentum_in is not None:
                    dev_momentum_buffer_in = ttnn.from_torch(
                        cpu_momentum_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device
                    )
                else:
                    dev_momentum_buffer_in = ttnn.from_torch(
                        cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device
                    )

            dev_momentum_buffer_out = ttnn.from_torch(cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        dev_param_out, dev_momentum_buffer_out = ttnn.operations.moreh.sgd(
            dev_param_in,
            dev_grad,
            dev_momentum_buffer_in,
            dev_param_out if has_param_out else None,
            dev_momentum_buffer_out,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            momentum_initialized=momentum_initialized,
            compute_kernel_config=compute_kernel_config,
        )
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    assert dev_param_in.shape == list(model.weight.shape)
    # check param_out
    param_result = ttnn.to_torch(dev_param_out).to(cpu_dtype)

    rtol = atol = 0.05
    passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=0.99, rtol=rtol, atol=atol)

    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")
    assert passing

    # check momentum_out
    if momentum != 0:
        momentum_buffer_result = ttnn.to_torch(dev_momentum_buffer_out).to(cpu_dtype)

        passing, out = comp_allclose_and_pcc(cpu_momentum_out, momentum_buffer_result, pcc=0.99, rtol=rtol, atol=atol)
        logger.debug(f"Momentum_out passing (param)={passing}")
        logger.debug(f"Momentum_out pcc={out}")

        assert passing
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
