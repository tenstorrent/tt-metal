# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_adam import run_moreh_adam

pytestmark = pytest.mark.use_module_device


@pytest.mark.merge_gate
# 149 tiles: a prime above any device's core count, so the work split leaves a second core group
# and the factory builds its second compute kernel.
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
        # lr=1 so a kernel that ignores `step` misses by >= 0.26; beta2=0.999 rounds to 0.99609375 in bfloat16,
        # which shifts the update by up to ~0.05.
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
