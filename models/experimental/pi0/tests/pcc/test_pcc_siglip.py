# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: SigLIP Vision Tower - TTNN vs PyTorch

Tests the full SigLIP vision tower with both random and real checkpoint weights.

Usage:
    pytest test_pcc_siglip_full.py -v
    pytest test_pcc_siglip_full.py -v -k "pretrained_weight_true"   # Only real weights
    pytest test_pcc_siglip_full.py -v -k "pretrained_weight_false"  # Only random weights (fast)
    python test_pcc_siglip_full.py  # Standalone
"""

import sys
import os
import time
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.reference.torch_siglip import SigLIPVisionTower as SigLIPVisionTowerTorch
from models.experimental.pi0.tt.ttnn_siglip import SigLIPVisionTowerTTNN
from models.experimental.pi0.common.configs import SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
SEED = 42
PCC_THRESHOLD = 0.90


def create_siglip_config() -> SigLIPConfig:
    """Create SigLIP config matching checkpoint."""
    return SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )


def create_small_siglip_config() -> SigLIPConfig:
    """Create smaller SigLIP config for fast random testing."""
    return SigLIPConfig(
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=4,
        num_attention_heads=6,
        image_size=224,
        patch_size=14,
    )


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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_siglip_vision_tower(device, use_pretrained):
    """Test SigLIP Vision Tower: TTNN vs PyTorch."""
    torch.manual_seed(SEED)

    if use_pretrained:
        checkpoint_path = Path(CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")

        config = create_siglip_config()
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        vision_weights = weight_loader.get_vlm_vision_weights()
    else:
        # Use smaller config for random tests (faster)
        config = create_small_siglip_config()
        vision_weights = create_random_siglip_weights(config)

    # Create input
    pixel_values = torch.randn(1, 3, config.image_size, config.image_size)

    # PyTorch forward
    model_torch = SigLIPVisionTowerTorch(config, vision_weights)
    out_torch = model_torch.forward(pixel_values)

    # TTNN forward
    model_ttnn = SigLIPVisionTowerTTNN(config, vision_weights, device)
    out_ttnn = model_ttnn.forward(pixel_values)

    # Convert to torch
    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    # Compute PCC
    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ SigLIP Vision Tower PCC ({weight_type}): {pcc:.6f}")
    print(f"   Output shape: {out_ttnn.shape}")

    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


def create_random_siglip_weights(config: SigLIPConfig) -> dict:
    """Create random weights for fast testing."""
    weights = {}
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    patch_size = config.patch_size
    num_patches = (config.image_size // patch_size) ** 2

    # Patch embedding (SigLIP doesn't use class token, so position_embedding is exactly num_patches)
    weights["vision_model.embeddings.patch_embedding.weight"] = torch.randn(hidden, 3, patch_size, patch_size)
    weights["vision_model.embeddings.patch_embedding.bias"] = torch.randn(hidden)
    weights["vision_model.embeddings.position_embedding.weight"] = torch.randn(num_patches, hidden)

    # Encoder blocks
    for i in range(config.num_hidden_layers):
        prefix = f"vision_model.encoder.layers.{i}."
        weights[f"{prefix}layer_norm1.weight"] = torch.randn(hidden)
        weights[f"{prefix}layer_norm1.bias"] = torch.randn(hidden)
        weights[f"{prefix}layer_norm2.weight"] = torch.randn(hidden)
        weights[f"{prefix}layer_norm2.bias"] = torch.randn(hidden)
        weights[f"{prefix}self_attn.q_proj.weight"] = torch.randn(hidden, hidden)
        weights[f"{prefix}self_attn.q_proj.bias"] = torch.randn(hidden)
        weights[f"{prefix}self_attn.k_proj.weight"] = torch.randn(hidden, hidden)
        weights[f"{prefix}self_attn.k_proj.bias"] = torch.randn(hidden)
        weights[f"{prefix}self_attn.v_proj.weight"] = torch.randn(hidden, hidden)
        weights[f"{prefix}self_attn.v_proj.bias"] = torch.randn(hidden)
        weights[f"{prefix}self_attn.out_proj.weight"] = torch.randn(hidden, hidden)
        weights[f"{prefix}self_attn.out_proj.bias"] = torch.randn(hidden)
        weights[f"{prefix}mlp.fc1.weight"] = torch.randn(intermediate, hidden)
        weights[f"{prefix}mlp.fc1.bias"] = torch.randn(intermediate)
        weights[f"{prefix}mlp.fc2.weight"] = torch.randn(hidden, intermediate)
        weights[f"{prefix}mlp.fc2.bias"] = torch.randn(hidden)

    # Final layer norm
    weights["vision_model.encoder.final_layer_norm.weight"] = torch.randn(hidden)
    weights["vision_model.encoder.final_layer_norm.bias"] = torch.randn(hidden)

    return weights


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  SigLIP Vision Tower PCC Test (Checkpoint Weights)")
    print("=" * 70)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    torch.manual_seed(SEED)
    config = create_siglip_config()

    print(f"\n📁 Checkpoint: {checkpoint_path}")
    print(f"📋 Config: {config.image_size}x{config.image_size}, {config.num_hidden_layers} layers")

    # Open device
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        # Load weights
        print("\n1. Loading checkpoint weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        vision_weights = weight_loader.get_vlm_vision_weights()
        print(f"   ✅ Loaded {len(vision_weights)} vision weight tensors")

        # Create input
        print("\n2. Creating test input...")
        pixel_values = torch.randn(1, 3, config.image_size, config.image_size)
        print(f"   Input: {pixel_values.shape}")

        # PyTorch
        print("\n3. Running PyTorch forward...")
        model_torch = SigLIPVisionTowerTorch(config, vision_weights)
        t0 = time.time()
        with torch.no_grad():
            out_torch = model_torch.forward(pixel_values)
        torch_time = (time.time() - t0) * 1000
        print(f"   Output: {out_torch.shape}, Time: {torch_time:.2f}ms")

        # TTNN
        print("\n4. Running TTNN forward...")
        model_ttnn = SigLIPVisionTowerTTNN(config, vision_weights, device)
        t0 = time.time()
        with torch.no_grad():
            out_ttnn = model_ttnn.forward(pixel_values)
        ttnn.synchronize_device(device)
        ttnn_time = (time.time() - t0) * 1000

        if isinstance(out_ttnn, ttnn.Tensor):
            out_ttnn = ttnn.to_torch(out_ttnn)
        print(f"   Output: {out_ttnn.shape}, Time: {ttnn_time:.2f}ms")

        # PCC
        pcc = compute_pcc(out_torch, out_ttnn)
        passed = pcc >= PCC_THRESHOLD

        print("\n" + "=" * 70)
        print("  RESULTS (pretrained)")
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
