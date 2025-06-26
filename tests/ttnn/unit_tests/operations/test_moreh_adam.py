# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim

import ttnn
import pytest
from models.utility_functions import (
    comp_allclose_and_pcc,
)
from loguru import logger
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    to_ttnn,
)


def create_tt_tensor(tensor: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)


def run_moreh_adam(shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device, dtype=ttnn.bfloat16, step=1):
    x_data = torch.rand(shape).to(torch.bfloat16)
    y_data = torch.rand(shape).to(torch.bfloat16)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.weight = nn.Parameter(torch.randn(shape).to(torch.bfloat16)).to(torch.bfloat16)

        def forward(self, x):
            return torch.mul(x, self.weight)

    model = SimpleModel()

    cpu_exp_avg = torch.zeros_like(model.weight)
    cpu_exp_avg_sq = torch.zeros_like(model.weight)
    cpu_max_exp_avg_sq = torch.zeros_like(model.weight)

    dev_param = create_tt_tensor(model.weight, device, dtype=dtype)
    dev_exp_avg = create_tt_tensor(cpu_exp_avg, device, dtype=dtype)
    dev_exp_avg_sq = create_tt_tensor(cpu_exp_avg_sq, device, dtype=dtype)
    dev_max_exp_avg_sq = create_tt_tensor(cpu_max_exp_avg_sq, device, dtype=dtype)

    dev_param_out = create_tt_tensor(model.weight, device, dtype=dtype)
    dev_exp_avg_out = create_tt_tensor(cpu_exp_avg, device, dtype=dtype)
    dev_exp_avg_sq_out = create_tt_tensor(cpu_exp_avg_sq, device, dtype=dtype)
    dev_max_exp_avg_sq_out = create_tt_tensor(cpu_max_exp_avg_sq, device, dtype=dtype) if amsgrad else None

    criterion = nn.L1Loss()
    optimizer = optim.Adam({model.weight}, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    optimizer.zero_grad()
    optimizer_state_dict = optimizer.state_dict()
    outputs = model(x_data)
    loss = criterion(outputs, y_data)
    loss.backward()

    cpu_grad = model.weight.grad.clone()
    dev_grad = create_tt_tensor(cpu_grad, device, dtype=dtype)

    for _ in range(step):
        optimizer.step()

    optimizer_state_dict = optimizer.state_dict()

    cpu_exp_avg_result = optimizer_state_dict["state"][0]["exp_avg"]
    cpu_exp_avg_sq_result = optimizer_state_dict["state"][0]["exp_avg_sq"]
    if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
        cpu_max_exp_avg_sq_result = optimizer_state_dict["state"][0]["max_exp_avg_sq"]
    else:
        cpu_max_exp_avg_sq_result = None

    compute_kernel_config = get_compute_kernel_options(fp32_dest_acc_en)

    (
        dev_param_out,
        dev_exp_avg_out,
        dev_exp_avg_sq_out,
        dev_max_exp_avg_sq_out,
    ) = ttnn.operations.moreh.adam(
        dev_param,
        dev_grad,
        dev_exp_avg,
        dev_exp_avg_sq,
        lr=lr,
        beta1=betas[0],
        beta2=betas[1],
        eps=eps,
        weight_decay=weight_decay,
        step=step,
        amsgrad=amsgrad,
        max_exp_avg_sq_in=dev_max_exp_avg_sq,
        param_out=dev_param_out,
        exp_avg_out=dev_exp_avg_out,
        exp_avg_sq_out=dev_exp_avg_sq_out,
        max_exp_avg_sq_out=dev_max_exp_avg_sq_out,
        compute_kernel_config=compute_kernel_config,
    )

    assert dev_param.padded_shape == ttnn.Shape(model.weight.shape)

    param_result = dev_param_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    exp_avg_result = dev_exp_avg_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    exp_avg_sq_result = dev_exp_avg_sq_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
        max_exp_avg_sq_result = dev_max_exp_avg_sq_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().to(torch.bfloat16)
    else:
        max_exp_avg_sq_result = None

    rtol = atol = 0.01
    passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (param)={passing}")
    logger.debug(f"Output pcc={out}")

    passing, out = comp_allclose_and_pcc(cpu_exp_avg_result, exp_avg_result, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (exp_avg)={passing}")
    logger.debug(f"Output pcc={out}")

    passing, out = comp_allclose_and_pcc(cpu_exp_avg_sq_result, exp_avg_sq_result, pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing (exp_avg_sq)={passing}")
    logger.debug(f"Output pcc={out}")

    if "max_exp_avg_sq" in optimizer_state_dict["state"][0]:
        passing, out = comp_allclose_and_pcc(
            cpu_max_exp_avg_sq_result, max_exp_avg_sq_result, pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"Out passing (max_exp_avg_sq)={passing}")
        logger.debug(f"Output pcc={out}")
    assert passing


@pytest.mark.parametrize(
    "shape",
    [[32, 32], [2, 2, 2, 2, 2, 2, 64, 64]],
)
@pytest.mark.parametrize("lr", [0.0, 1e-1])
@pytest.mark.parametrize("betas", ((0.9, 0.999), (0.5, 0.555)))
@pytest.mark.parametrize("eps", [1e-06, 1e-08])
@pytest.mark.parametrize("weight_decay", [0.0, 0.3])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("fp32_dest_acc_en", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_moreh_adam(shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device, dtype, step=1):
    torch.manual_seed(0)
    if dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported")
    return run_moreh_adam(
        shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device, dtype=dtype, step=step
    )


@pytest.mark.parametrize(
    "params",
    (
        # shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en
        ([32, 32], 0.0, (0.9, 0.999), 1e-06, 0.0, True, True),
        ([2, 2, 2, 2, 2, 2, 64, 64], 0.0, (0.9, 0.999), 1e-06, 0.0, False, False),
    ),
)
def test_moreh_adam_callback(params, device):
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en = params
        run_moreh_adam(shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize(
    "params",
    (
        # shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en
        ([32, 32], 0.0, (0.9, 0.999), 1e-06, 0.0, True, True),
        ([2, 2, 2, 2, 2, 2, 64, 64], 0.0, (0.9, 0.999), 1e-06, 0.0, False, False),
    ),
)
def test_moreh_adam_caching(params, device):
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(1, 5):
        shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en = params
        run_moreh_adam(shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device, step=i)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    for i in range(1, 4):
        assert num_program_cache_entries_list[0] == num_program_cache_entries_list[i]

    num_program_cache_entries_list = []
    for i in range(4):
        shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en = params

        # generate a random lr between (0, 1)
        lr = torch.rand(1).item()

        run_moreh_adam(shape, lr, betas, eps, weight_decay, amsgrad, fp32_dest_acc_en, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    for i in range(1, 4):
        assert num_program_cache_entries_list[0] == num_program_cache_entries_list[i]
