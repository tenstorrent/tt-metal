# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import create_ttnn_tilized_tensor, get_compute_kernel_options

pytestmark = pytest.mark.use_module_device

# Not the nightly helper: its outputs start as the inputs, which stay within tolerance after one step.


def run_moreh_adamw_test(shape, device, amsgrad=True, fp32_dest_acc_en=False, step=8, provide_outputs=True):
    x = torch.rand(shape).to(torch.bfloat16)
    y = torch.rand(shape).to(torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(shape).to(torch.bfloat16))
    optimizer = torch.optim.AdamW([weight], lr=1e-2, betas=(0.5, 0.555), eps=1e-8, weight_decay=0.3, amsgrad=amsgrad)

    def backward():
        optimizer.zero_grad()
        # Summed, so the moments stay around 1, well above the tolerance.
        torch.nn.functional.l1_loss(x * weight, y, reduction="sum").backward()

    for _ in range(step - 1):
        backward()
        optimizer.step()
    backward()

    state = optimizer.state[weight]
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    tt_param = create_ttnn_tilized_tensor(weight.detach(), device, ttnn.bfloat16)
    tt_grad = create_ttnn_tilized_tensor(weight.grad, device, ttnn.bfloat16)
    tt_exp_avg = create_ttnn_tilized_tensor(state.get("exp_avg", zeros), device, ttnn.bfloat16)
    tt_exp_avg_sq = create_ttnn_tilized_tensor(state.get("exp_avg_sq", zeros), device, ttnn.bfloat16)
    tt_max_exp_avg_sq = (
        create_ttnn_tilized_tensor(state.get("max_exp_avg_sq", zeros), device, ttnn.bfloat16) if amsgrad else None
    )
    # NaN, so an output the op never writes fails.
    num_outputs = 4 if amsgrad else 3
    nans = torch.full(shape, float("nan"), dtype=torch.bfloat16)
    tt_outputs = [create_ttnn_tilized_tensor(nans, device, ttnn.bfloat16) for _ in range(num_outputs)]
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
    # Check the provided buffers themselves, not the return value.
    actual_outputs = tt_outputs if provide_outputs else result[:num_outputs]

    expected_outputs = [weight.detach(), state["exp_avg"], state["exp_avg_sq"]]
    if amsgrad:
        expected_outputs.append(state["max_exp_avg_sq"])
    for expected, actual in zip(expected_outputs, actual_outputs):
        passing, output_pcc = comp_allclose_and_pcc(expected, ttnn.to_torch(actual), pcc=0.99, rtol=0.1, atol=0.1)
        assert passing, output_pcc


@pytest.mark.merge_gate
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["bf16_acc", "fp32_dest_acc"])
@pytest.mark.parametrize("amsgrad", [True, False], ids=["amsgrad", "no_amsgrad"])
def test_moreh_adamw(amsgrad, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_adamw_test([32, 32], device, amsgrad=amsgrad, fp32_dest_acc_en=fp32_dest_acc_en)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, step, provide_outputs",
    [
        ([32, 149 * 32], 8, True),
        ([5, 3], 8, True),
        ([32, 32], 1, True),
        ([32, 32], 8, False),
    ],
    ids=["core_group_2", "hw_unaligned", "step_1", "allocated_outputs"],
)
def test_moreh_adamw_corner_cases(shape, step, provide_outputs, device):
    torch.manual_seed(0)
    run_moreh_adamw_test(shape, device, step=step, provide_outputs=provide_outputs)
