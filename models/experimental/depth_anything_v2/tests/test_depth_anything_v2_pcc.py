# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForDepthEstimation, DepthAnythingConfig
from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingReassembleLayer

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.depth_anything_v2.tt.model_def import (
    DictNamespace,
    TtDepthAnythingV2,
    TtDPTReassembleLayer,
    conv_transpose_as_linear,
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("read_idx, factor", [(0, 4), (1, 2)])
def test_dpt_reassemble_learned_upsampling(device, read_idx, factor):
    """Reassembly layers 0 and 1 upsample with the checkpoint's ConvTranspose2d (kernel_size == stride == factor).
    Random weights on the 37x37 patch grid with a small hidden size, against the HF layer."""
    torch.manual_seed(0)
    hidden_size, channels = 64, 32
    config = DepthAnythingConfig(reassemble_hidden_size=hidden_size)
    torch_layer = DepthAnythingReassembleLayer(config, channels=channels, factor=factor).eval()
    hidden = torch.randn(1, hidden_size, 37, 37)
    with torch.no_grad():
        torch_output = torch_layer(hidden)

    # The model's token layout: 32 CLS slots, 1369 patch tokens, padding to 1568.
    tokens = torch.zeros(1, 1568, hidden_size)
    tokens[0, 32:1401] = hidden.flatten(2).transpose(1, 2)[0]
    resize_weight, resize_bias = conv_transpose_as_linear(torch_layer.resize)
    projection_weight = torch_layer.projection.weight.permute(1, 2, 3, 0).reshape(-1, channels)
    parameters = DictNamespace(
        {
            "projection": DictNamespace(
                {
                    "weight": ttnn.from_torch(
                        projection_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(
                        torch_layer.projection.bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
                    ),
                }
            ),
            "resize": DictNamespace(
                {
                    "weight": ttnn.from_torch(
                        resize_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                    "bias": ttnn.from_torch(resize_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                }
            ),
        }
    )
    tt_layer = TtDPTReassembleLayer(parameters, read_idx=read_idx, config=None, device=device, feature_size=channels)
    tt_tokens = ttnn.from_torch(tokens, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.to_torch(tt_layer(tt_tokens)).float()

    passing, pcc_value = comp_pcc(torch_output, tt_output, pcc=0.99)
    assert passing, f"PCC {pcc_value} < 0.99 threshold"
