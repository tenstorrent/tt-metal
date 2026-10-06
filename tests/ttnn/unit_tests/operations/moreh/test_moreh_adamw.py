# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import (
    TILE_HEIGHT,
    TILE_WIDTH,
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
)

pytestmark = pytest.mark.use_module_device

INPUT_SHAPE = [TILE_HEIGHT, TILE_WIDTH]


def run_moreh_adamw_test(shape, lr, betas, eps, weight_decay, step, device, amsgrad=True, fp32_dest_acc_en=False):
    x = torch.rand(shape).to(torch.bfloat16)
    y = torch.rand(shape).to(torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(shape).to(torch.bfloat16))
    optimizer = torch.optim.AdamW([weight], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    for _ in range(step - 1):
        optimizer.zero_grad()
        torch.nn.functional.l1_loss(x * weight, y).backward()
        optimizer.step()
    optimizer.zero_grad()
    torch.nn.functional.l1_loss(x * weight, y).backward()

    state = optimizer.state[weight]
    tt_param = create_ttnn_tilized_tensor(weight.detach(), device, ttnn.bfloat16)
    tt_grad = create_ttnn_tilized_tensor(weight.grad, device, ttnn.bfloat16)
    tt_exp_avg = create_ttnn_tilized_tensor(state["exp_avg"], device, ttnn.bfloat16)
    tt_exp_avg_sq = create_ttnn_tilized_tensor(state["exp_avg_sq"], device, ttnn.bfloat16)
    tt_max_exp_avg_sq = create_ttnn_tilized_tensor(state["max_exp_avg_sq"], device, ttnn.bfloat16) if amsgrad else None
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    tt_outputs = [create_ttnn_tilized_tensor(zeros, device, ttnn.bfloat16) for _ in range(4 if amsgrad else 3)]
    optimizer.step()

    ttnn.moreh_adamw(
        tt_param,
        tt_grad,
        tt_exp_avg,
        tt_exp_avg_sq,
        lr=lr,
        beta1=betas[0],
        beta2=betas[1],
        eps=eps,
        weight_decay=weight_decay,
        step=step,
        amsgrad=amsgrad,
        max_exp_avg_sq_in=tt_max_exp_avg_sq,
        param_out=tt_outputs[0],
        exp_avg_out=tt_outputs[1],
        exp_avg_sq_out=tt_outputs[2],
        max_exp_avg_sq_out=tt_outputs[3] if amsgrad else None,
        compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
    )

    torch_outputs = [weight.detach(), state["exp_avg"], state["exp_avg_sq"]]
    if amsgrad:
        torch_outputs.append(state["max_exp_avg_sq"])
    for expected, actual in zip(torch_outputs, tt_outputs):
        passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(actual), pcc=0.99, rtol=0.1, atol=0.1)
        assert passing, output_pcc


def run_moreh_adamw_inplace_test(lr, step, device):
    # Random moments move by ~0.5 in one step, so a write that misses the aliased outputs fails the check.
    weight = torch.nn.Parameter(torch.rand(INPUT_SHAPE, dtype=torch.bfloat16))
    weight.grad = torch.rand(INPUT_SHAPE, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW([weight], lr=lr, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, amsgrad=True)
    state = optimizer.state[weight]
    state["step"] = torch.tensor(float(step - 1))
    for name in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
        state[name] = torch.rand(INPUT_SHAPE, dtype=torch.bfloat16)

    tt_grad = create_ttnn_tilized_tensor(weight.grad, device, ttnn.bfloat16)
    tt_tensors = [
        create_ttnn_tilized_tensor(tensor, device, ttnn.bfloat16)
        for tensor in [weight.detach(), state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"]]
    ]
    optimizer.step()

    # Every output aliases its input, the way the tt-train optimizer calls the op.
    ttnn.moreh_adamw(
        tt_tensors[0],
        tt_grad,
        tt_tensors[1],
        tt_tensors[2],
        lr=lr,
        beta1=0.5,
        beta2=0.555,
        eps=1e-8,
        weight_decay=0.3,
        step=step,
        amsgrad=True,
        max_exp_avg_sq_in=tt_tensors[3],
        param_out=tt_tensors[0],
        exp_avg_out=tt_tensors[1],
        exp_avg_sq_out=tt_tensors[2],
        max_exp_avg_sq_out=tt_tensors[3],
        compute_kernel_config=get_compute_kernel_options(False),
    )

    torch_outputs = [weight.detach(), state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"]]
    for expected, actual in zip(torch_outputs, tt_tensors):
        passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(actual), pcc=0.99, rtol=0.1, atol=0.1)
        assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_adamw(device):
    torch.manual_seed(0)
    run_moreh_adamw_test(
        [TILE_HEIGHT, TILE_WIDTH], lr=1e-2, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, step=8, device=device
    )


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, amsgrad, fp32_dest_acc_en",
    [
        (INPUT_SHAPE, False, False),
        # Smaller than one tile in H and W, so the single tile is mostly padding.
        ([5, 3], True, False),
        (INPUT_SHAPE, True, True),
    ],
    ids=["no_amsgrad", "hw_unaligned", "fp32_dest_acc"],
)
def test_moreh_adamw_corner_cases(shape, amsgrad, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_adamw_test(
        shape,
        lr=1e-2,
        betas=(0.5, 0.555),
        eps=1e-8,
        weight_decay=0.3,
        step=8,
        device=device,
        amsgrad=amsgrad,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )


@pytest.mark.merge_gate
def test_moreh_adamw_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_adamw_test(INPUT_SHAPE, lr=1e-2, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, step=8, device=device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    run_moreh_adamw_test(INPUT_SHAPE, lr=1e-2, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, step=8, device=device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_adamw_inplace_program_cache(device):
    torch.manual_seed(0)
    # Regression for #48928: an in-place call on a program-cache hit wrote to the previous call's buffers.
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_adamw_inplace_test(lr=1e-2, step=1, device=device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros(INPUT_SHAPE), device, ttnn.bfloat16)
    # lr and step are not in the program hash, so this is still a cache hit that must also patch them.
    run_moreh_adamw_inplace_test(lr=2e-2, step=2, device=device)
    assert device.num_program_cache_entries() == num_program_cache_entries
