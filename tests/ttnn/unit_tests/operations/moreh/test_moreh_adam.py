# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_adam import run_moreh_adam

pytestmark = pytest.mark.use_module_device


@pytest.mark.merge_gate
@pytest.mark.parametrize("shape", [[32, 32], [32, 149 * 32]], ids=["single_tile", "core_group_2"])
def test_moreh_adam(shape, device):
    torch.manual_seed(0)
    run_moreh_adam(shape, 0.1, (0.5, 0.555), 1e-8, 0.3, True, False, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "lr, betas, weight_decay, amsgrad, fp32_dest_acc_en, step, param_atol",
    [
        (0.1, (0.5, 0.555), 0.3, False, False, 1, 0.01),
        (0.1, (0.5, 0.555), 0.3, True, True, 1, 0.01),
        (0.1, (0.5, 0.555), 0.3, False, True, 1, 0.01),
        # lr=1 so ignoring `step` misses by ~0.26; beta2=0.999 truncated to bfloat16 shifts the update by up to ~0.05.
        (1.0, (0.9, 0.999), 0.0, False, False, 2, 0.05),
        (1.0, (0.9, 0.999), 0.0, False, False, 10, 0.05),
    ],
    ids=["no_amsgrad", "fp32_dest_acc", "no_amsgrad_fp32_dest_acc", "step_2", "step_10"],
)
def test_moreh_adam_corner_cases(lr, betas, weight_decay, amsgrad, fp32_dest_acc_en, step, param_atol, device):
    torch.manual_seed(0)
    run_moreh_adam(
        [32, 32], lr, betas, 1e-8, weight_decay, amsgrad, fp32_dest_acc_en, device, step=step, param_atol=param_atol
    )


@pytest.mark.parametrize("mismatched", ["grad", "exp_avg_in", "param_out"])
def test_moreh_adam_rejects_shape_other_than_param(mismatched, device, expect_error):
    # The program walks param_in's tiles in every buffer, so a smaller input is read past its end and a smaller
    # output is written past its end, into whatever DRAM follows it.
    shape, small = (1, 1, 64, 64), (1, 1, 32, 32)
    tt = lambda s: ttnn.from_torch(torch.randn(s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    param = tt(shape)
    grad = tt(small if mismatched == "grad" else shape)
    exp_avg = tt(small if mismatched == "exp_avg_in" else shape)
    exp_avg_sq = tt(shape)
    param_out = tt(small) if mismatched == "param_out" else None
    with expect_error(RuntimeError, f"{mismatched} shape .* must match the param shape"):
        ttnn.moreh_adam(param, grad, exp_avg, exp_avg_sq, lr=0.1, step=1, param_out=param_out)
