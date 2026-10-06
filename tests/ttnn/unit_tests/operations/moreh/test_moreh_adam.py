# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device


def run_moreh_adam_test(
    shape, lr, betas, eps, weight_decay, device, amsgrad=True, step=1, fp32_dest_acc_en=False, param_atol=0.01
):
    x = torch.rand(shape).to(torch.bfloat16)
    y = torch.rand(shape).to(torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(shape).to(torch.bfloat16))
    torch.nn.functional.l1_loss(x * weight, y).backward()

    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    tt_param = create_ttnn_tilized_tensor(weight.detach(), device, ttnn.bfloat16)
    tt_grad = create_ttnn_tilized_tensor(weight.grad, device, ttnn.bfloat16)
    tt_exp_avg = create_ttnn_tilized_tensor(zeros, device, ttnn.bfloat16)
    tt_exp_avg_sq = create_ttnn_tilized_tensor(zeros, device, ttnn.bfloat16)
    tt_max_exp_avg_sq = create_ttnn_tilized_tensor(zeros, device, ttnn.bfloat16) if amsgrad else None
    tt_outputs = [create_ttnn_tilized_tensor(zeros, device, ttnn.bfloat16) for _ in range(4 if amsgrad else 3)]

    optimizer = torch.optim.Adam([weight], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    # The kernel raises beta to `step` in one shot: seed the torch state at step - 1 with the same zero moments.
    if step > 1:
        state = optimizer.state[weight]
        state["step"] = torch.tensor(float(step - 1))
        state["exp_avg"] = torch.zeros_like(weight)
        state["exp_avg_sq"] = torch.zeros_like(weight)
        if amsgrad:
            state["max_exp_avg_sq"] = torch.zeros_like(weight)
    optimizer.step()
    state = optimizer.state[weight]

    ttnn.moreh_adam(
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
    atols = [param_atol, 0.01, 0.01, 0.01]
    for expected, actual, atol in zip(torch_outputs, tt_outputs, atols):
        passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(actual), pcc=0.999, rtol=0.01, atol=atol)
        assert passing, output_pcc


@pytest.mark.merge_gate
def test_moreh_adam(device):
    torch.manual_seed(0)
    run_moreh_adam_test([32, 32], lr=1e-1, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, device=device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "lr, betas, weight_decay, amsgrad, step, fp32_dest_acc_en, param_atol",
    [
        (1e-1, (0.5, 0.555), 0.3, False, 1, False, 0.01),
        (1e-1, (0.5, 0.555), 0.0, True, 1, False, 0.01),
        # lr=1 so a kernel that ignores `step` misses by >= 0.26; beta2=0.999 rounds to 0.99609375 in bfloat16,
        # which shifts the update by up to ~0.05. step and lr are not in the program hash, so step_10 reuses
        # the step_2 program and also checks that a cache hit patches the new step.
        (1.0, (0.9, 0.999), 0.0, False, 2, False, 0.05),
        (1.0, (0.9, 0.999), 0.0, False, 10, False, 0.05),
        (1e-1, (0.5, 0.555), 0.3, True, 1, True, 0.01),
    ],
    ids=["no_amsgrad", "no_weight_decay", "step_2", "step_10", "fp32_dest_acc"],
)
def test_moreh_adam_corner_cases(lr, betas, weight_decay, amsgrad, step, fp32_dest_acc_en, param_atol, device):
    torch.manual_seed(0)
    run_moreh_adam_test(
        [32, 32],
        lr=lr,
        betas=betas,
        eps=1e-8,
        weight_decay=weight_decay,
        device=device,
        amsgrad=amsgrad,
        step=step,
        fp32_dest_acc_en=fp32_dest_acc_en,
        param_atol=param_atol,
    )


@pytest.mark.merge_gate
def test_moreh_adam_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_adam_test([32, 32], lr=1e-1, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, device=device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses.
    tt_placeholder = create_ttnn_tilized_tensor(torch.zeros([32, 32]), device, ttnn.bfloat16)
    run_moreh_adam_test([32, 32], lr=1e-1, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, device=device)
    assert device.num_program_cache_entries() == num_program_cache_entries
