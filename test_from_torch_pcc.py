# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test case for ttnn.from_torch() PCC verification.
Tests conversion of transformer block norm2 weights from the SDXL refiner model
using both bfloat16 and float32 dtypes.
"""

import torch
import ttnn
from diffusers import UNet2DConditionModel
from models.common.utility_functions import comp_pcc


def test_from_torch_pcc():
    """
    Test PCC of ttnn.from_torch/to_torch roundtrip conversion
    for the norm2 weights that fail in the refiner UNet model.
    """
    # Load the refiner UNet model
    model_name = "stabilityai/stable-diffusion-xl-refiner-1.0"
    print(f"Loading model: {model_name}")
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="unet",
    )
    unet.eval()
    state_dict = unet.state_dict()

    # The failing weight path: mid_block.attentions.0.transformer_blocks.1.norm2.weight
    # This corresponds to "Creating attention layer 0..." -> "Creating transformer layer 1..."
    weight_path = "mid_block.attentions.0.transformer_blocks.1.norm2.weight"

    print(f"\n{'='*60}")
    print(f"Testing weight: {weight_path}")
    print(f"{'='*60}")

    # Get the weight tensor
    torch_weight = state_dict[weight_path]
    print(f"Original torch tensor shape: {torch_weight.shape}")
    print(f"Original torch tensor dtype: {torch_weight.dtype}")
    print(f"Original torch tensor min: {torch_weight.min():.6f}, max: {torch_weight.max():.6f}")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Test 1: bfloat16 conversion
        print(f"\n{'='*60}")
        print("Test 1: bfloat16 conversion")
        print(f"{'='*60}")

        ttnn_tensor_bf16 = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        print(f"TTNN tensor (bfloat16) shape: {ttnn_tensor_bf16.shape}")
        print(f"TTNN tensor (bfloat16) dtype: {ttnn_tensor_bf16.dtype}")

        # Convert back to torch
        torch_weight_bf16_back = ttnn.to_torch(ttnn_tensor_bf16)
        print(f"Converted back torch tensor shape: {torch_weight_bf16_back.shape}")
        print(f"Converted back torch tensor dtype: {torch_weight_bf16_back.dtype}")

        # Compute PCC
        matches_bf16, pcc_bf16 = comp_pcc(torch_weight, torch_weight_bf16_back, pcc=0.99)
        print(f"\n>>> bfloat16 PCC: {pcc_bf16}")
        print(f">>> bfloat16 PCC >= 0.99: {matches_bf16}")
        print(f">>> bfloat16 PCC == 1.0: {pcc_bf16 == 1.0}")

        ttnn.deallocate(ttnn_tensor_bf16)

        # Test 2: float32 conversion
        print(f"\n{'='*60}")
        print("Test 2: float32 conversion")
        print(f"{'='*60}")

        ttnn_tensor_fp32 = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.float32,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        print(f"TTNN tensor (float32) shape: {ttnn_tensor_fp32.shape}")
        print(f"TTNN tensor (float32) dtype: {ttnn_tensor_fp32.dtype}")

        # Convert back to torch
        torch_weight_fp32_back = ttnn.to_torch(ttnn_tensor_fp32)
        print(f"Converted back torch tensor shape: {torch_weight_fp32_back.shape}")
        print(f"Converted back torch tensor dtype: {torch_weight_fp32_back.dtype}")

        # Compute PCC
        matches_fp32, pcc_fp32 = comp_pcc(torch_weight, torch_weight_fp32_back, pcc=0.99)
        print(f"\n>>> float32 PCC: {pcc_fp32}")
        print(f">>> float32 PCC >= 0.99: {matches_fp32}")
        print(f">>> float32 PCC == 1.0: {pcc_fp32 == 1.0}")

        ttnn.deallocate(ttnn_tensor_fp32)

        # Test 3: Pre-convert to bfloat16 in torch, then send to ttnn
        print(f"\n{'='*60}")
        print("Test 3: torch bfloat16 -> ttnn -> torch (compare vs original fp32)")
        print(f"{'='*60}")

        # First convert to bfloat16 in torch
        torch_weight_bf16 = torch_weight.to(torch.bfloat16)
        print(f"Torch bfloat16 tensor shape: {torch_weight_bf16.shape}")
        print(f"Torch bfloat16 tensor dtype: {torch_weight_bf16.dtype}")

        # Check PCC of just the torch fp32 -> bf16 conversion
        matches_torch_bf16, pcc_torch_bf16 = comp_pcc(torch_weight, torch_weight_bf16, pcc=0.99)
        print(f">>> torch fp32 -> bf16 PCC (before ttnn): {pcc_torch_bf16}")

        # Now send to ttnn as bfloat16
        ttnn_tensor_from_bf16 = ttnn.from_torch(
            torch_weight_bf16,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        print(f"TTNN tensor shape: {ttnn_tensor_from_bf16.shape}")
        print(f"TTNN tensor dtype: {ttnn_tensor_from_bf16.dtype}")

        # Convert back to torch
        torch_weight_from_bf16_back = ttnn.to_torch(ttnn_tensor_from_bf16)
        print(f"Converted back torch tensor shape: {torch_weight_from_bf16_back.shape}")
        print(f"Converted back torch tensor dtype: {torch_weight_from_bf16_back.dtype}")

        # Compare against ORIGINAL fp32 tensor
        matches_bf16_vs_fp32, pcc_bf16_vs_fp32 = comp_pcc(torch_weight, torch_weight_from_bf16_back, pcc=0.99)
        print(f"\n>>> PCC (ttnn output vs original fp32): {pcc_bf16_vs_fp32}")
        print(f">>> PCC >= 0.99: {matches_bf16_vs_fp32}")

        # Also compare against the torch bfloat16 intermediate (should be ~1.0 if ttnn preserves bf16)
        matches_bf16_vs_bf16, pcc_bf16_vs_bf16 = comp_pcc(torch_weight_bf16, torch_weight_from_bf16_back, pcc=0.99)
        print(f">>> PCC (ttnn output vs torch bf16 intermediate): {pcc_bf16_vs_bf16}")

        ttnn.deallocate(ttnn_tensor_from_bf16)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Weight path: {weight_path}")
        print(f"Weight shape: {torch_weight.shape}")
        print(f"Test 1 - ttnn.from_torch(fp32, bf16) roundtrip PCC: {pcc_bf16}")
        print(f"Test 2 - ttnn.from_torch(fp32, fp32) roundtrip PCC: {pcc_fp32}")
        print(f"Test 3 - torch fp32->bf16 conversion PCC:           {pcc_torch_bf16}")
        print(f"Test 3 - torch bf16->ttnn->torch vs original fp32:  {pcc_bf16_vs_fp32}")
        print(f"Test 3 - torch bf16->ttnn->torch vs torch bf16:     {pcc_bf16_vs_bf16}")
        print(f"{'='*60}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_from_torch_pcc()
