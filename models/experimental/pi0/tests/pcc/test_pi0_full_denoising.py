# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Full Denoising Test - TTNN vs PyTorch

This script runs the complete denoising loop (sample_actions) and compares
TTNN vs PyTorch outputs using PCC.

Hardcoded config:
- Checkpoint: /home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base
- Full denoising: Enabled (10 denoising steps)
- Batch size: 1

Usage:
    python test_pi0_full_denoising.py           # Standalone
    pytest test_pi0_full_denoising.py -v        # With pytest
"""

import sys
import time
from pathlib import Path

import pytest
import torch
import ttnn

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.ttnn_pi0_reference.ttnn_pi0 import (
    PI0ModelTorch,
    PI0ModelTTNN,
    PI0ModelConfig,
)
from models.experimental.pi0.ttnn_pi0_reference.weight_loader import PI0WeightLoader
from models.experimental.pi0.ttnn_pi0_reference.ttnn_siglip import SigLIPConfig


# =============================================================================
# CONFIGURATION (hardcoded)
# =============================================================================
CHECKPOINT_PATH = "/home/ubuntu/work/sdawle_pi0/torch_checkpoint/pi0_base"
BATCH_SIZE = 1
SEED = 42
PCC_THRESHOLD = 0.85


def create_config() -> PI0ModelConfig:
    """Create PI0ModelConfig with correct dimensions."""
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
    )

    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )

    return config


def create_test_inputs(config: PI0ModelConfig, batch_size: int = 1):
    """Create test inputs."""
    image_size = config.siglip_config.image_size

    images = [torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32) for _ in range(2)]
    img_masks = [torch.ones(batch_size, dtype=torch.bool) for _ in range(2)]

    lang_tokens = torch.randint(0, 256000, (batch_size, 32))
    lang_masks = torch.ones(batch_size, 32, dtype=torch.bool)

    state = torch.randn(batch_size, config.state_dim, dtype=torch.float32)

    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
    }


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    mean1, mean2 = torch.mean(t1), torch.mean(t2)
    std1, std2 = torch.std(t1), torch.std(t2)

    if std1 == 0 or std2 == 0:
        return 1.0 if torch.allclose(t1, t2) else 0.0

    covariance = torch.mean((t1 - mean1) * (t2 - mean2))
    return (covariance / (std1 * std2)).item()


# =============================================================================
# PYTEST TEST FUNCTION
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_full_denoising(device):
    """
    Pytest: PI0 full denoising (sample_actions) TTNN vs PyTorch.

    This test:
    1. Loads PI0 model (PyTorch and TTNN)
    2. Runs full denoising loop (10 steps)
    3. Compares outputs with PCC >= 0.85
    """
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    # Create config and inputs
    config = create_config()
    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)

    # Load weights
    weight_loader = PI0WeightLoader(str(checkpoint_path))

    # Initialize models
    model_torch = PI0ModelTorch(config, weight_loader)
    model_ttnn = PI0ModelTTNN(config, weight_loader, device)

    # Run PyTorch
    torch.manual_seed(SEED)
    with torch.no_grad():
        torch_actions = model_torch.sample_actions(
            images=inputs["images"],
            img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            state=inputs["state"],
        )

    # Run TTNN
    torch.manual_seed(SEED)
    with torch.no_grad():
        ttnn_actions = model_ttnn.sample_actions(
            images=inputs["images"],
            img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            state=inputs["state"],
        )

    # Compute PCC
    pcc = compute_pcc(torch_actions, ttnn_actions)

    print(f"\n✅ Full Denoising PCC: {pcc:.6f}")
    print(f"   Output shape: {ttnn_actions.shape}")

    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


# =============================================================================
# STANDALONE RUNNER
# =============================================================================


def main():
    """Run full denoising test."""
    print("=" * 80)
    print("  PI0 FULL DENOISING TEST")
    print("  TTNN vs PyTorch (sample_actions)")
    print("=" * 80)

    # Verify checkpoint
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"\n📁 Checkpoint: {checkpoint_path}")
    print(f"📊 Batch size: {BATCH_SIZE}")
    print(f"🎲 Seed: {SEED}")

    # Open device
    print("\n🔌 Opening TTNN device...")
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    print(f"✅ Device opened (grid: {grid.x}x{grid.y})")

    try:
        # Create config
        config = create_config()
        print(f"\n📋 Config: {config.siglip_config.image_size}x{config.siglip_config.image_size} images")

        # Load weights
        print("\n1. Loading weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        print("   ✅ Weights loaded")

        # Initialize models
        print("\n2. Initializing models...")
        model_torch = PI0ModelTorch(config, weight_loader)
        print("   ✅ PyTorch model initialized")

        model_ttnn = PI0ModelTTNN(config, weight_loader, device)
        print("   ✅ TTNN model initialized")

        # Create inputs
        print("\n3. Creating test inputs...")
        inputs = create_test_inputs(config, batch_size=BATCH_SIZE)
        print(f"   ✅ Inputs: batch_size={BATCH_SIZE}, images={inputs['images'][0].shape}")

        # Run full denoising
        print("\n4. Running FULL DENOISING (sample_actions)...")
        print("   This runs the complete denoising loop (10 steps)")

        # PyTorch
        print("\n   Running PyTorch sample_actions()...")
        torch.manual_seed(SEED)
        start = time.time()
        with torch.no_grad():
            torch_actions = model_torch.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        torch_time = (time.time() - start) * 1000
        print(f"      PyTorch: {torch_actions.shape}, time: {torch_time:.2f}ms")

        # TTNN
        print("\n   Running TTNN sample_actions()...")
        torch.manual_seed(SEED)  # Same seed
        start = time.time()
        with torch.no_grad():
            ttnn_actions = model_ttnn.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
        ttnn_time = (time.time() - start) * 1000
        print(f"      TTNN:    {ttnn_actions.shape}, time: {ttnn_time:.2f}ms")

        # Compute PCC
        print("\n5. Computing PCC (PyTorch vs TTNN)...")
        pcc = compute_pcc(torch_actions, ttnn_actions)
        passed = pcc >= PCC_THRESHOLD

        # Final summary
        print("\n" + "=" * 80)
        print("  RESULTS")
        print("=" * 80)
        print(f"\n   PyTorch Time:    {torch_time:.2f}ms")
        print(f"   TTNN Time:       {ttnn_time:.2f}ms")

        if ttnn_time > 0 and torch_time > 0:
            if ttnn_time < torch_time:
                speedup = torch_time / ttnn_time
                print(f"   Speedup:         {speedup:.2f}x faster (TTNN) 🚀")
            else:
                slowdown = ttnn_time / torch_time
                print(f"   Slowdown:        {slowdown:.2f}x slower (TTNN)")

        print(f"\n   Output Shape:    {ttnn_actions.shape}")
        print(f"   PCC:             {pcc:.6f}")
        print(f"   Threshold:       {PCC_THRESHOLD}")
        print(f"\n   Status:          {'✅ PASS' if passed else '❌ FAIL'}")
        print("=" * 80)

        return 0 if passed else 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        print("\n🔌 Closing device...")
        ttnn.close_device(device)
        print("✅ Device closed")


if __name__ == "__main__":
    sys.exit(main())
