# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForDepthEstimation, DepthAnythingConfig
from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingDepthEstimationHead

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.depth_anything_v2.tt.model_def import (
    DictNamespace,
    TtDepthAnythingV2,
    TtDPTHead,
    custom_preprocessor,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
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
    passing, pcc_value = comp_pcc(torch_output, tt_output, pcc=0.99)
    print(f"PCC Result: passing={passing}, pcc={pcc_value}")
    assert passing, f"PCC {pcc_value} < 0.99 threshold"


def _conv_parameters(device, **convs):
    return DictNamespace(
        {
            name: DictNamespace(
                {
                    "weight": ttnn.from_torch(
                        conv.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(
                        conv.bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                    ),
                }
            )
            for name, conv in convs.items()
        }
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dpt_head_resize(device):
    """The head resizes conv1's output from 8x to 14x the patch grid. Random weights on a 4x4 patch grid (32 -> 56,
    the same 1.75 ratio as 296 -> 518), against the HF head."""
    torch.manual_seed(0)
    config = DepthAnythingConfig(fusion_hidden_size=256, head_hidden_size=32)
    torch_head = DepthAnythingDepthEstimationHead(config).eval()
    hidden = torch.randn(1, 256, 32, 32)
    with torch.no_grad():
        torch_output = torch_head([hidden], patch_height=4, patch_width=4)

    parameters = _conv_parameters(device, conv1=torch_head.conv1, conv2=torch_head.conv2, conv3=torch_head.conv3)
    tt_head = TtDPTHead(parameters, device, max_depth=config.max_depth)
    tt_hidden = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.to_torch(tt_head(tt_hidden, patch_height=4, patch_width=4)).float().reshape(torch_output.shape)

    passing, pcc_value = comp_pcc(torch_output, tt_output, pcc=0.99)
    assert passing, f"PCC {pcc_value} < 0.99 threshold"
