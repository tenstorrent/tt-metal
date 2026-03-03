# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: VitModel - TTNN vs PyTorch

Tests the VitModel (CLIP-L) from DeepSeek-OCR with pretrained weights.

Usage:
    pytest test_pcc_vit_model.py -v
    python test_pcc_vit_model.py  # Standalone
"""

import sys
import time
from pathlib import Path

import pytest
import torch
import ttnn

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
sys.path.insert(
    0,
    "/home/ubuntu/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR/9f30c71f441d010e5429c532364a86705536c53a",
)

# Import PyTorch model
from deepencoder import build_clip_l, vit_model_cfg

# Import TTNN model
from models.experimental.tt_symbiote.tests.deepseek_ocr_vision_model.ttnn_vit_model import (
    build_clip_l_ttnn,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42
PCC_THRESHOLD = 0.90
BATCH_SIZE = 1
IMAGE_SIZE = 224


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()
    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)
    if std1 < 1e-6 or std2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    covariance = torch.mean((t1 - mean1) * (t2 - mean2))
    return (covariance / (std1 * std2)).item()


def convert_pytorch_weights_to_ttnn_format(pytorch_model, device: ttnn.Device) -> dict:
    """
    Convert PyTorch model weights to TTNN format.

    Args:
        pytorch_model: PyTorch VitModel instance
        device: TTNN device

    Returns:
        Dictionary of weights in TTNN format matching TTNN model structure
    """
    weights = {}

    # Embeddings
    weights["embeddings"] = {}
    weights["embeddings"]["class_embedding"] = pytorch_model.embeddings.class_embedding.data
    weights["embeddings"]["patch_embedding.weight"] = pytorch_model.embeddings.patch_embedding.weight.data
    # Note: patch_embedding has bias=False, so no bias to extract
    if pytorch_model.embeddings.patch_embedding.bias is not None:
        weights["embeddings"]["patch_embedding.bias"] = pytorch_model.embeddings.patch_embedding.bias.data
    weights["embeddings"]["position_embedding.weight"] = pytorch_model.embeddings.position_embedding.weight.data

    # Pre-layer norm
    weights["pre_layrnorm.weight"] = pytorch_model.pre_layrnorm.weight.data
    weights["pre_layrnorm.bias"] = pytorch_model.pre_layrnorm.bias.data

    # Transformer blocks - structure: transformer.layers.{i}.{component}
    weights["transformer"] = {}
    for i, layer in enumerate(pytorch_model.transformer.layers):
        layer_key = f"layers.{i}"
        weights["transformer"][layer_key] = {}

        # Layer norms
        weights["transformer"][layer_key]["layer_norm1.weight"] = layer.layer_norm1.weight.data
        weights["transformer"][layer_key]["layer_norm1.bias"] = layer.layer_norm1.bias.data
        weights["transformer"][layer_key]["layer_norm2.weight"] = layer.layer_norm2.weight.data
        weights["transformer"][layer_key]["layer_norm2.bias"] = layer.layer_norm2.bias.data

        # Attention
        weights["transformer"][layer_key]["self_attn"] = {}
        weights["transformer"][layer_key]["self_attn"]["qkv_proj.weight"] = layer.self_attn.qkv_proj.weight.data
        if layer.self_attn.qkv_proj.bias is not None:
            weights["transformer"][layer_key]["self_attn"]["qkv_proj.bias"] = layer.self_attn.qkv_proj.bias.data
        weights["transformer"][layer_key]["self_attn"]["out_proj.weight"] = layer.self_attn.out_proj.weight.data
        if layer.self_attn.out_proj.bias is not None:
            weights["transformer"][layer_key]["self_attn"]["out_proj.bias"] = layer.self_attn.out_proj.bias.data

        # Feedforward
        weights["transformer"][layer_key]["mlp"] = {}
        weights["transformer"][layer_key]["mlp"]["fc1.weight"] = layer.mlp.fc1.weight.data
        if layer.mlp.fc1.bias is not None:
            weights["transformer"][layer_key]["mlp"]["fc1.bias"] = layer.mlp.fc1.bias.data
        weights["transformer"][layer_key]["mlp"]["fc2.weight"] = layer.mlp.fc2.weight.data
        if layer.mlp.fc2.bias is not None:
            weights["transformer"][layer_key]["mlp"]["fc2.bias"] = layer.mlp.fc2.bias.data

    return weights


# =============================================================================
# PYTEST TEST FUNCTION
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pcc_vit_model(device):
    """Test VitModel: TTNN vs PyTorch with pretrained weights."""
    torch.manual_seed(SEED)

    # Create config
    cfg = vit_model_cfg

    # Load PyTorch model with pretrained weights
    model_torch = build_clip_l()
    model_torch.eval()

    # Convert weights to TTNN format
    weights = convert_pytorch_weights_to_ttnn_format(model_torch, device)

    # Create input
    pixel_values = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    patch_embeds = None

    # PyTorch forward
    with torch.no_grad():
        out_torch = model_torch(pixel_values, patch_embeds)

    # TTNN forward
    model_ttnn = build_clip_l_ttnn(weights=weights, device=device)

    # Convert input to TTNN
    pixel_values_ttnn = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TTNN forward
    out_ttnn = model_ttnn.forward(pixel_values_ttnn, patch_embeds=None)

    # Convert to torch
    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    # Compute PCC
    pcc = compute_pcc(out_torch, out_ttnn)

    print(f"\n✅ VitModel PCC: {pcc:.6f}")
    print(f"   PyTorch output shape: {out_torch.shape}")
    print(f"   TTNN output shape: {out_ttnn.shape}")
    print(f"   PyTorch output mean: {out_torch.mean().item():.6f}, std: {out_torch.std().item():.6f}")
    print(f"   TTNN output mean: {out_ttnn.mean().item():.6f}, std: {out_ttnn.std().item():.6f}")

    # Debug: Print max difference and where it occurs
    diff = torch.abs(out_torch - out_ttnn)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"   Max absolute difference: {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")

    # Pretrained weights should achieve high PCC (debug test shows ~0.9998)
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


# =============================================================================
# STANDALONE RUNNER
# =============================================================================


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  VitModel PCC Test (TTNN vs PyTorch)")
    print("=" * 70)

    torch.manual_seed(SEED)
    cfg = vit_model_cfg

    print(f"\n📋 Config:")
    print(f"   Image size: {cfg.image_size}x{cfg.image_size}")
    print(f"   Patch size: {cfg.patch_size}")
    print(f"   Hidden size: {cfg.hidden_size}")
    print(f"   Num layers: {cfg.num_layers}")
    print(f"   Num heads: {cfg.num_attention_heads}")

    # Open device
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        # Load PyTorch model with pretrained weights
        print("\n1. Loading pretrained weights...")
        model_torch = build_clip_l()
        model_torch.eval()

        # Convert weights to TTNN format
        weights = convert_pytorch_weights_to_ttnn_format(model_torch, device)
        print(f"   ✅ Loaded weights for {cfg.num_layers} transformer layers")

        # Create input
        pixel_values = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
        print(f"   Input: {pixel_values.shape}")

        # PyTorch forward
        print("\n2. Running PyTorch forward...")
        t0 = time.time()
        with torch.no_grad():
            out_torch = model_torch(pixel_values, None)
        torch_time = (time.time() - t0) * 1000
        print(f"   Output: {out_torch.shape}, Time: {torch_time:.2f}ms")

        # TTNN forward
        print("\n3. Running TTNN forward...")
        model_ttnn = build_clip_l_ttnn(weights=weights, device=device)

        pixel_values_ttnn = ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        t0 = time.time()
        out_ttnn = model_ttnn.forward(pixel_values_ttnn, patch_embeds=None)
        ttnn.synchronize_device(device)
        ttnn_time = (time.time() - t0) * 1000

        if isinstance(out_ttnn, ttnn.Tensor):
            out_ttnn = ttnn.to_torch(out_ttnn)
        print(f"   Output: {out_ttnn.shape}, Time: {ttnn_time:.2f}ms")

        # PCC
        pcc = compute_pcc(out_torch, out_ttnn)
        passed = pcc >= PCC_THRESHOLD

        print("\n" + "=" * 70)
        print("  RESULTS (pretrained weights)")
        print("=" * 70)
        print(f"   PCC:       {pcc:.6f}")
        print(f"   Threshold: {PCC_THRESHOLD}")
        print(f"   Status:    {'✅ PASS' if passed else '❌ FAIL'}")
        if ttnn_time > 0 and ttnn_time < torch_time:
            print(f"   Speedup:   {torch_time / ttnn_time:.2f}x")
        print("=" * 70)

        return 0 if passed else 1

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
