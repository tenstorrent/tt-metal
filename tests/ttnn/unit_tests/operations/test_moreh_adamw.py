# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import ttnn

import pytest
from models.utility_functions import (
    comp_allclose_and_pcc,
)
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_ttnn,
)
from loguru import logger


def create_tt_tensors(cpu_grad, cpu_weight, cpu_exp_avg, cpu_exp_avg_sq, cpu_max_exp_avg_sq, amsgrad, dtype, device):
    def create_tt_tensor(tensor: torch.Tensor, dtype, device):
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # input tensors
    param_in = create_tt_tensor(cpu_weight, dtype, device)
    grad = create_tt_tensor(cpu_grad, dtype, device)
    exp_avg_in = create_tt_tensor(cpu_exp_avg, dtype, device)
    exp_avg_sq_in = create_tt_tensor(cpu_exp_avg_sq, dtype, device)
    max_exp_avg_sq_in = create_tt_tensor(cpu_max_exp_avg_sq, dtype, device) if amsgrad else None

    # output tensors
    param_out = create_tt_tensor(cpu_weight, dtype, device)
    exp_avg_out = create_tt_tensor(cpu_exp_avg, dtype, device)
    exp_avg_sq_out = create_tt_tensor(cpu_exp_avg_sq, dtype, device)
    max_exp_avg_sq_out = create_tt_tensor(cpu_max_exp_avg_sq, dtype, device) if amsgrad else None

    return (
        (param_in, grad, exp_avg_in, exp_avg_sq_in, max_exp_avg_sq_in),
        (param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out),
    )


def run_moreh_adamw(
    shape,
    lr,
    betas,
    eps,
    weight_decay,
    amsgrad,
    step,
    device,
    *,
    ttnn_dtype=ttnn.bfloat16,
    torch_dtype=torch.bfloat16,
    compute_kernel_options=None,
):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)

    x_data = torch.rand(shape).to(torch_dtype)
    y_data = torch.rand(shape).to(torch_dtype)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(shape).to(torch_dtype)).to(torch_dtype)

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

    tt_input_tensors, tt_output_tensors = create_tt_tensors(
        cpu_grad,
        cpu_weight,
        cpu_exp_avg,
        cpu_exp_avg_sq,
        cpu_max_exp_avg_sq,
        amsgrad,
        ttnn_dtype,
        device,
    )

    tt_param_in, tt_grad, tt_exp_avg_in, tt_exp_avg_sq_in, tt_max_exp_avg_sq_in = tt_input_tensors
    tt_param_out, tt_exp_avg_out, tt_exp_avg_sq_out, tt_max_exp_avg_sq_out = tt_output_tensors

    ret_list_ = ttnn.operations.moreh.adamw(
        tt_param_in,
        tt_grad,
        tt_exp_avg_in,
        tt_exp_avg_sq_in,
        lr,
        betas[0],
        betas[1],
        eps,
        weight_decay,
        step,
        amsgrad,
        max_exp_avg_sq_in=tt_max_exp_avg_sq_in,
        param_out=tt_param_out,
        exp_avg_out=tt_exp_avg_out,
        exp_avg_sq_out=tt_exp_avg_sq_out,
        max_exp_avg_sq_out=tt_max_exp_avg_sq_out,
        compute_kernel_config=compute_kernel_config,
    )

    param_result = ttnn.to_torch(tt_param_out).reshape(shape)
    exp_avg_result = ttnn.to_torch(tt_exp_avg_out).reshape(shape)
    exp_avg_sq_result = ttnn.to_torch(tt_exp_avg_sq_out).reshape(shape)
    print(param_result.shape, model.weight.shape)
    if amsgrad:
        max_exp_avg_sq_result = ttnn.to_torch(tt_max_exp_avg_sq_out).reshape(shape)
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


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # single
        [4, 3, 2, 6, 64, 64],  # multi tile
    ],
)
@pytest.mark.parametrize("lr", [1e-2])
@pytest.mark.parametrize("betas", [[0.5, 0.555]])
@pytest.mark.parametrize("eps", [1e-08])
@pytest.mark.parametrize("weight_decay", [0.3])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("step", [8])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_moreh_adamw(shape, lr, betas, eps, weight_decay, amsgrad, step, ttnn_dtype, device):
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("Bfloat8_b is only supported with fp32_dest_acc set to True")
    torch.manual_seed(0)
    run_moreh_adamw(shape, lr, betas, eps, weight_decay, amsgrad, step, device, ttnn_dtype=ttnn_dtype)


@pytest.mark.parametrize(
    "shape",
    [[32, 32]],  # single
)
@pytest.mark.parametrize("lr", [1e-2])
@pytest.mark.parametrize("betas", [[0.5, 0.555]])
@pytest.mark.parametrize("eps", [1e-08])
@pytest.mark.parametrize("weight_decay", [0.3])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("step", [8])
def test_moreh_adamw_callback(shape, lr, betas, eps, weight_decay, amsgrad, step, device):
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_adamw(shape, lr, betas, eps, weight_decay, amsgrad, step, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize(
    "shape",
    [[5, 3], [32, 32]],
)
@pytest.mark.parametrize("lr", [1e-2])
@pytest.mark.parametrize("betas", [[0.5, 0.555]])
@pytest.mark.parametrize("eps", [1e-08])
@pytest.mark.parametrize("weight_decay", [0.3])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("step", [8])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_moreh_adamw_compute_kernel_options(
    shape, lr, betas, eps, weight_decay, amsgrad, step, ttnn_dtype, compute_kernel_options, device
):
    if ttnn_dtype == ttnn.bfloat8_b and not compute_kernel_options:
        pytest.skip("Bfloat8_b is only supported with fp32_dest_acc set to True")

    torch.manual_seed(0)
    run_moreh_adamw(
        shape,
        lr,
        betas,
        eps,
        weight_decay,
        amsgrad,
        step,
        device,
        ttnn_dtype=ttnn_dtype,
        compute_kernel_options=compute_kernel_options,
    )


@pytest.mark.parametrize(
    "shape",
    [[32, 32]],  # single
)
@pytest.mark.parametrize("lr", [1e-2])
@pytest.mark.parametrize("betas", [[0.5, 0.555]])
@pytest.mark.parametrize("eps", [1e-08])
@pytest.mark.parametrize("weight_decay", [0.3])
@pytest.mark.parametrize("amsgrad", [True])
def test_moreh_adamw_cache(shape, lr, betas, eps, weight_decay, amsgrad, device):
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for step in range(1, 5):
        run_moreh_adamw(shape, lr, betas, eps, weight_decay, amsgrad, step, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    for i in range(1, 4):
        assert num_program_cache_entries_list[0] == num_program_cache_entries_list[i]

    num_program_cache_entries_list = []
    for _ in range(4):
        # generate a new lr between (0, 1)
        lr = torch.rand(1).item()

        run_moreh_adamw(shape, lr, betas, eps, weight_decay, amsgrad, 8, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    for i in range(1, 4):
        assert num_program_cache_entries_list[0] == num_program_cache_entries_list[i]
