# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.batchnorm2d import BatchNorm2d
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_batchnorm2d(device):
    torch.manual_seed(0)

    batch_size = 1
    height = 8
    width = 8
    channels = 16

    torch_batchnorm = torch.nn.BatchNorm2d(
        num_features=channels,
        eps=1e-5,
        momentum=0.01,
        affine=True,
        track_running_stats=True,
    ).eval()

    torch_input = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    torch_output = torch_batchnorm(torch_input)

    tt_batchnorm = BatchNorm2d(
        device=device,
        num_features=channels,
        momentum=0.01,
        eps=1e-5,
    )
    state_dict = {
        "encoder.bn.weight": torch_batchnorm.weight,
        "encoder.bn.bias": torch_batchnorm.bias,
        "encoder.bn.running_mean": torch_batchnorm.running_mean,
        "encoder.bn.running_var": torch_batchnorm.running_var,
    }
    tt_batchnorm.load_state_dict(state_dict=state_dict, key="bn", module_prefix="encoder.")

    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_output = tt_batchnorm(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).permute(0, 3, 1, 2)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
