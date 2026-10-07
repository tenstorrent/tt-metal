# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

# Not the nightly helper: it pre-fills every output with its input, and one step at lr=1e-2 moves the values by less
# than the 0.1 tolerance, so an op that writes nothing still passes.


def run_moreh_adamw_test(shape, device, amsgrad=True, fp32_dest_acc_en=False, step=8, provide_outputs=True):
    x = torch.rand(shape).to(torch.bfloat16)
    y = torch.rand(shape).to(torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(shape).to(torch.bfloat16))
    optimizer = torch.optim.AdamW([weight], lr=1e-2, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, amsgrad=amsgrad)

    def backward():
        optimizer.zero_grad()
        # Summed, not averaged: the gradients, and so the moments, stay around 1 instead of 1e-3, well above the 0.1
        # tolerance, so a moment the op never writes fails.
        torch.nn.functional.l1_loss(x * weight, y, reduction="sum").backward()

    for _ in range(step - 1):
        backward()
        optimizer.step()
    backward()

    # Before the first step the optimizer has no state yet; the op then starts from zero moments.
    state = optimizer.state[weight]
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    tt_param = create_ttnn_tilized_tensor(weight.detach(), device, ttnn.bfloat16)
    tt_grad = create_ttnn_tilized_tensor(weight.grad, device, ttnn.bfloat16)
    tt_exp_avg = create_ttnn_tilized_tensor(state.get("exp_avg", zeros), device, ttnn.bfloat16)
    tt_exp_avg_sq = create_ttnn_tilized_tensor(state.get("exp_avg_sq", zeros), device, ttnn.bfloat16)
    tt_max_exp_avg_sq = (
        create_ttnn_tilized_tensor(state.get("max_exp_avg_sq", zeros), device, ttnn.bfloat16) if amsgrad else None
    )
    # Zeros, so an output the op never writes fails.
    num_outputs = 4 if amsgrad else 3
    tt_outputs = [create_ttnn_tilized_tensor(zeros, device, ttnn.bfloat16) for _ in range(num_outputs)]
    if not provide_outputs:
        tt_outputs = [None] * num_outputs
    optimizer.step()

    result = ttnn.moreh_adamw(
        tt_param,
        tt_grad,
        tt_exp_avg,
        tt_exp_avg_sq,
        lr=1e-2,
        beta1=0.5,
        beta2=0.555,
        eps=1e-8,
        weight_decay=0.3,
        step=step,
        amsgrad=amsgrad,
        max_exp_avg_sq_in=tt_max_exp_avg_sq,
        param_out=tt_outputs[0],
        exp_avg_out=tt_outputs[1],
        exp_avg_sq_out=tt_outputs[2],
        max_exp_avg_sq_out=tt_outputs[3] if amsgrad else None,
        compute_kernel_config=get_compute_kernel_options(fp32_dest_acc_en),
    )
    # With provided outputs, check those buffers themselves: the op must write into them.
    actual_outputs = tt_outputs if provide_outputs else result[:num_outputs]

    expected_outputs = [weight.detach(), state["exp_avg"], state["exp_avg_sq"]]
    if amsgrad:
        expected_outputs.append(state["max_exp_avg_sq"])
    for expected, actual in zip(expected_outputs, actual_outputs):
        passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(actual), pcc=0.99, rtol=0.1, atol=0.1)
        assert passing, output_pcc


def run_moreh_adamw_inplace_test(lr, step, device):
    # Random moments move by ~0.5 in one step, so a write that misses the aliased outputs fails the check.
    weight = torch.nn.Parameter(torch.rand([32, 32], dtype=torch.bfloat16))
    weight.grad = torch.rand([32, 32], dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW([weight], lr=lr, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, amsgrad=True)
    state = optimizer.state[weight]
    state["step"] = torch.tensor(float(step - 1))
    for name in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
        state[name] = torch.rand([32, 32], dtype=torch.bfloat16)

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

    expected_outputs = [weight.detach(), state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"]]
    for expected, actual in zip(expected_outputs, tt_tensors):
        passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(actual), pcc=0.99, rtol=0.1, atol=0.1)
        assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["bf16_acc", "fp32_dest_acc"])
@pytest.mark.parametrize("amsgrad", [True, False], ids=["amsgrad", "no_amsgrad"])
def test_moreh_adamw(amsgrad, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    # AMSGRAD and FP32_DEST_ACC_EN are the two defines: four compiled variants.
    run_moreh_adamw_test([32, 32], device, amsgrad=amsgrad, fp32_dest_acc_en=fp32_dest_acc_en)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, step, provide_outputs",
    [
        # 149 tiles: a prime above any device's core count, so the work split leaves a second core group and the
        # factory builds its second compute kernel.
        ([32, 149 * 32], 8, True),
        # Smaller than one tile in H and W, so the single tile is mostly padding.
        ([5, 3], 8, True),
        # The first optimizer step: bias corrections 1 - beta^1, starting from zero moments.
        ([32, 32], 1, True),
        # No outputs passed: the op allocates them.
        ([32, 32], 8, False),
    ],
    ids=["core_group_2", "hw_unaligned", "step_1", "allocated_outputs"],
)
def test_moreh_adamw_corner_cases(shape, step, provide_outputs, device):
    torch.manual_seed(0)
    run_moreh_adamw_test(shape, device, step=step, provide_outputs=provide_outputs)


@pytest.mark.merge_gate
def test_moreh_adamw_program_cache(device):
    torch.manual_seed(0)
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_adamw_test([32, 32], device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses. Row-major,
    # so creating it runs no device program of its own.
    tt_placeholder = ttnn.from_torch(torch.zeros([32, 32]), dtype=ttnn.bfloat16, device=device)
    run_moreh_adamw_test([32, 32], device)
    assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.merge_gate
def test_moreh_adamw_inplace_program_cache(device):
    torch.manual_seed(0)
    # Regression for #48928: an in-place call on a program-cache hit wrote to the previous call's buffers.
    # Start from an empty cache: the module-scoped device carries entries over from earlier tests.
    device.clear_program_cache()
    run_moreh_adamw_inplace_test(lr=1e-2, step=1, device=device)
    num_program_cache_entries = device.num_program_cache_entries()
    # Without this, the equality below would also pass for an op that never caches a program.
    assert num_program_cache_entries > 0
    # Holding this tensor moves the next allocations, so the cache hit must update the buffer addresses. Row-major,
    # so creating it runs no device program of its own.
    tt_placeholder = ttnn.from_torch(torch.zeros([32, 32]), dtype=ttnn.bfloat16, device=device)
    # lr and step are not in the program hash, so this is still a cache hit that must also patch them.
    run_moreh_adamw_inplace_test(lr=2e-2, step=2, device=device)
    assert device.num_program_cache_entries() == num_program_cache_entries
