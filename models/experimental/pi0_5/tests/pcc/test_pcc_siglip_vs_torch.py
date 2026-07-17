# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: SigLIP Vision Tower - TTNN vs PyTorch

Tests the full SigLIP vision tower with real (pretrained upstream libero) checkpoint weights.

Usage:
    pytest test_pcc_siglip_full.py -v
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

from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower as SigLIPVisionTowerTorch
from models.experimental.pi0_5.tt.ttnn_siglip import SigLIPVisionTowerTTNN
from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.environ.get(
    "PI05_CHECKPOINT_DIR",
    str(Path(__file__).resolve().parents[2] / "weights" / "pi05_base"),
)
SEED = 42
PCC_THRESHOLD = 0.99


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
def test_pcc_siglip_vision_tower(device):
    """Test SigLIP Vision Tower: TTNN vs PyTorch."""
    torch.manual_seed(SEED)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.is_absolute() and not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    config = create_siglip_config()
    weight_loader = PI0WeightLoader(str(checkpoint_path))
    vision_weights = weight_loader.get_vlm_vision_weights()

    # Production pi0.5 LIBERO uses 3 image slots (base + wrist + zero placeholder).
    # See [[pi05-siglip-bs3-production]]. bs=2 was the prior assumption — kept as
    # a backward-compat A/B target via PI0_NUM_CAMERAS=2. bs=1 hides a class of
    # bug where the BS path flattens batch into seq and SDPA computes cross-image
    # attention.
    bs = int(os.environ.get("PI0_NUM_CAMERAS", "2"))
    pixel_values = torch.randn(bs, 3, config.image_size, config.image_size)

    # PyTorch forward
    model_torch = SigLIPVisionTowerTorch(config, vision_weights)
    out_torch = model_torch.forward(pixel_values)

    # Convert to TTNN tensor for TTNN model
    pixel_values_ttnn = ttnn.from_torch(
        pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN forward
    model_ttnn = SigLIPVisionTowerTTNN(config, vision_weights, device)
    out_ttnn = model_ttnn.forward(pixel_values_ttnn)

    # Convert to torch
    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    # Compute PCC
    pcc = compute_pcc(out_torch, out_ttnn)

    print(f"\n✅ SigLIP Vision Tower PCC (pretrained): {pcc:.6f}")
    print(f"   Output shape: {out_ttnn.shape}")

    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  SigLIP Vision Tower PCC Test (Checkpoint Weights)")
    print("=" * 70)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.is_absolute() and not checkpoint_path.exists():
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

        # Production pi0.5 LIBERO is bs=3. See [[pi05-siglip-bs3-production]].
        print("\n2. Creating test input...")
        bs = int(os.environ.get("PI0_NUM_CAMERAS", "2"))
        pixel_values = torch.randn(bs, 3, config.image_size, config.image_size)
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
        # forward() consumes a device tensor (BCHW); production callers (e2e /
        # LIBERO / perf) upload bf16 TILE before the call. Match that convention.
        pixel_values_ttnn = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t0 = time.time()
        with torch.no_grad():
            out_ttnn = model_ttnn.forward(pixel_values_ttnn)
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
