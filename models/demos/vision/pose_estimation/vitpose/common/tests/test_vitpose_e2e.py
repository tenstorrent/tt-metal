# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.vision.pose_estimation.vitpose.common.common import load_torch_model
from models.demos.vision.pose_estimation.vitpose.common.reference.vitpose_reference import (
    extract_reference_parameters,
    vitpose_forward,
)
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose import VitPose
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_vitpose_e2e(device, batch_size):
    torch.manual_seed(0)

    model = load_torch_model()
    state_dict = model.state_dict()
    ref_params = extract_reference_parameters(model)

    pixel_values = torch.randn(batch_size, 3, 256, 192, dtype=torch.bfloat16)
    torch_output = vitpose_forward(pixel_values, parameters=ref_params)

    tt_model = VitPose(state_dict, device, batch_size=batch_size)
    tt_input = VitPose.prepare_input(pixel_values, device)
    tt_output = tt_model(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    tt_output_reshaped = tt_output.reshape(batch_size, 64, 48, 17).permute(0, 3, 1, 2)
    assert_with_pcc(torch_output.float(), tt_output_reshaped.float(), 0.999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_vitpose_e2e_vs_hf(device, batch_size):
    """Compare TTNN output against HuggingFace model output directly."""
    torch.manual_seed(0)

    model = load_torch_model()
    state_dict = model.state_dict()

    pixel_values = torch.randn(batch_size, 3, 256, 192)
    with torch.no_grad():
        hf_output = model(pixel_values)
    hf_heatmaps = hf_output.heatmaps

    tt_model = VitPose(state_dict, device, batch_size=batch_size)
    pixel_values_bf16 = pixel_values.to(torch.bfloat16)
    tt_input = VitPose.prepare_input(pixel_values_bf16, device)
    tt_output = tt_model(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    tt_output_reshaped = tt_output.reshape(batch_size, 64, 48, 17).permute(0, 3, 1, 2)
    assert_with_pcc(hf_heatmaps.float(), tt_output_reshaped.float(), 0.998)
