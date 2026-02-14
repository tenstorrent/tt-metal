# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForDepthEstimation

import ttnn
from models.demos.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor
from models.utility_functions import comp_pcc


@pytest.mark.parametrize("device_params", [{"batch_size": 1}], indirect=True)
def test_depth_anything_v2_pcc(device):
    # This test compares the ttnn model output against the torch reference
    # Note: Requires hardware to run real ttnn operations.

    model_id = "depth-anything/Depth-Anything-V2-Large-hf"
    torch_model = AutoModelForDepthEstimation.from_pretrained(model_id, trust_remote_code=True)
    torch_model.eval()

    # 1. Create dummy input
    batch_size = 1
    input_shape = (batch_size, 3, 518, 518)
    pixel_values = torch.randn(input_shape)

    # 2. Run Torch Reference
    with torch.no_grad():
        torch_output = torch_model(pixel_values).predicted_depth

    # 3. Run TTNN Model
    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)

    # Convert input to ttnn
    # Use ROW_MAJOR_LAYOUT for raw pixels
    tt_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Inference
    tt_output_tensor = tt_model(tt_pixel_values)

    # Convert back to torch for comparison
    tt_output = ttnn.to_torch(tt_output_tensor)

    # 4. Compare using PCC
    # For now, we print shapes as a basic check if pcc is not imported
    print(f"Torch output shape: {torch_output.shape}")
    print(f"TTNN output shape: {tt_output.shape}")

    pcc_result = comp_pcc(torch_output, tt_output)
    print(f"PCC Result: {pcc_result}")
    assert pcc_result[1] > 0.99

    assert tt_output.shape == torch_output.shape
    print("Shape verification successful.")
