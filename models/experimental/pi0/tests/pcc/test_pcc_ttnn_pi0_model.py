# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 PCC Test - TTNN (tt/) vs PyTorch (reference/)

This test compares the TTNN implementation from tt/ against the
PyTorch reference implementation from reference/.

Config:
    - Checkpoint: $TT_METAL_HOME/models/experimental/pi0/weights/pi0_base
    - Full denoising: 10 steps
    - Batch size: 1

Usage:
    pytest test_pcc_ttnn_pi0_model.py -v
"""

import sys
import os
import time
from pathlib import Path

import pytest
import torch
import ttnn

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# PyTorch reference implementation
from models.experimental.pi0.reference.torch_pi0_model import PI0Model as PI0ModelTorch

# TTNN implementation
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN

# Shared configs and weight loader
from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


# =============================================================================
# CONFIGURATION
# =============================================================================
TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
BATCH_SIZE = 1
SEED = 42
PCC_THRESHOLD = 0.93  # Account for BFloat16 precision and hardware non-determinism


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
def test_pcc_pi0_ttnn(device):
    """
    Pytest: PI0 TTNN vs PyTorch.

    This test:
    1. Loads PI0 model from reference/ (PyTorch) and tt/ (TTNN)
    2. Runs full denoising loop (10 steps)
    3. Compares outputs with PCC >= threshold
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

    print(f"\n✅ PI0 PCC: {pcc:.6f}")
    print(f"   Output shape: {ttnn_actions.shape}")

    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


# =============================================================================
# STANDALONE RUNNER
# =============================================================================


def main():
    """Run PCC test."""
    print("=" * 80)
    print("  PI0 TTNN PCC TEST")
    print("  TTNN (tt/) vs PyTorch (reference/)")
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
        print("   ✅ PyTorch model initialized (reference/)")

        model_ttnn = PI0ModelTTNN(config, weight_loader, device)
        print("   ✅ TTNN model initialized (tt/)")

        # Create inputs
        print("\n3. Creating test inputs...")
        inputs = create_test_inputs(config, batch_size=BATCH_SIZE)
        print(f"   ✅ Inputs: batch_size={BATCH_SIZE}")

        # Run full denoising
        print("\n4. Running sample_actions()...")

        # PyTorch
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
        print(f"   PyTorch: {torch_actions.shape}, {torch_time:.2f}ms")

        # TTNN
        torch.manual_seed(SEED)
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
        print(f"   TTNN:    {ttnn_actions.shape}, {ttnn_time:.2f}ms")

        # Compute PCC
        pcc = compute_pcc(torch_actions, ttnn_actions)
        passed = pcc >= PCC_THRESHOLD

        # Results
        print("\n" + "=" * 80)
        print("  RESULTS")
        print("=" * 80)
        print(f"\n   PyTorch Time: {torch_time:.2f}ms")
        print(f"   TTNN Time:    {ttnn_time:.2f}ms")
        if ttnn_time > 0 and torch_time > 0 and ttnn_time < torch_time:
            print(f"   Speedup:      {torch_time / ttnn_time:.2f}x 🚀")
        print(f"\n   PCC:          {pcc:.6f}")
        print(f"   Threshold:    {PCC_THRESHOLD}")
        print(f"\n   Status:       {'✅ PASS' if passed else '❌ FAIL'}")
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
