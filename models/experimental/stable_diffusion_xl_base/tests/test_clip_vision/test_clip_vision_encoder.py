# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import utils
from model_pt import CLIPVisionEncoderAndResamplerPT, get_input, run_pytorch_model
from model_ttnn import CLIPVisionEncoderAndResamplerTTNN
from tracy import signpost
from weight_loader import load_weights_from_pytorch

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_clip_vision_encoder(mesh_device):
    """
    CLIP Vision Encoder (laion/CLIP-ViT-H-14-laion2B-s32B-b79K) + IP-Adapter Plus Resampler.

    Weights: ip-adapter-plus_sdxl_vit-h.bin from h94/IP-Adapter (subfolder: sdxl_models)

    Pipeline:
        1. CLIP Vision Encoder processes input image [batch, 3, 224, 224]
        2. Extracts penultimate hidden layer [batch, 257, 1280]
        3. Resampler produces IP-Adapter tokens [batch, 16, 2048]
    """
    # Load input tensor
    input_torch = get_input()

    # Calculate torch output
    output_torch = run_pytorch_model(input_torch)

    # Convert torch input to host TTNN tensor
    input_ttnn_host = ttnn.from_torch(input_torch["pixel_values"])
    input_ttnn_host = ttnn.to_layout(input_ttnn_host, ttnn.Layout.TILE)
    input_ttnn_host = ttnn.to_dtype(input_ttnn_host, ttnn.DataType.BFLOAT16)

    # Load PyTorch and TTNN models
    # Use PyTorch model weights for TTNN model
    model_pt = CLIPVisionEncoderAndResamplerPT()
    model_ttnn = CLIPVisionEncoderAndResamplerTTNN(
        load_weights_from_pytorch(model_pt.state_dict(), mesh_device), mesh_device
    )

    # Run ttnn model
    for i in range(3):
        start_time = time.time()
        signpost(f"ttnn_model_start_{i}")

        # Run ttnn model
        out_ttnn_device = model_ttnn(input_ttnn_host)[0]

        # Get outputs
        out_ttnn_host = ttnn.from_device(out_ttnn_device, blocking=True)
        ttnn.synchronize_device(mesh_device)

        signpost(f"ttnn_model_end_{i}")
        end_time = time.time()

        # Calculate FPS and PCC
        duration = (end_time - start_time) * 1000
        fps = 1.0 / (end_time - start_time)  # batch size is 1
        pcc = utils.calculate_pcc(output_torch, ttnn.to_torch(out_ttnn_host))

        # Print results
        print(f"Iteration {i}")
        print(f"\tDuration: {duration:.1f}ms")
        print(f"\tFPS: {fps:.2f}")
        print(f"\tPCC: {pcc:.6f}")
