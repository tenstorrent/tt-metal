# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForDepthEstimation

import ttnn
from models.experimental.depth_anything_v2.tt.model_def import TtDepthAnythingV2, custom_preprocessor
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_depth_anything_v2_pcc(device):
    """Compare ttnn model output against the torch reference.

    Requires Wormhole B0 hardware to run real ttnn operations.
    Asserts PCC > 0.99 between torch and ttnn depth maps.
    """
    model_id = "depth-anything/Depth-Anything-V2-Large-hf"
    torch_model = AutoModelForDepthEstimation.from_pretrained(model_id, trust_remote_code=True)
    torch_model.eval()

    # 1. Create deterministic input
    batch_size = 1
    input_shape = (batch_size, 3, 518, 518)
    torch.manual_seed(42)
    pixel_values = torch.randn(input_shape)

    # 2. Run Torch Reference
    with torch.no_grad():
        torch_output = torch_model(pixel_values).predicted_depth

    # 3. Run TTNN Model
    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)

    tt_pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output_tensor = tt_model(tt_pixel_values)

    # Convert back to torch for comparison
    tt_output = ttnn.to_torch(tt_output_tensor).float()

    # 4. Shape alignment — ttnn may produce a different spatial resolution
    #    due to tile padding; interpolate to match torch reference shape.
    if tt_output.shape != torch_output.shape:
        tt_output = torch.nn.functional.interpolate(
            tt_output.unsqueeze(0) if tt_output.dim() == 3 else tt_output,
            size=torch_output.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    print(f"Torch output shape: {torch_output.shape}")
    print(f"TTNN output shape:  {tt_output.shape}")

    # 5. PCC comparison
    passing, pcc_value = comp_pcc(torch_output, tt_output)
    print(f"PCC Result: passing={passing}, pcc={pcc_value}")
    assert passing, f"PCC {pcc_value} < 0.99 threshold"
