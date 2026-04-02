# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 PCC Test - TTNN (tt/) vs PyTorch (reference/)

Tests Pi0.5 model with adaRMS conditioning.

Config:
    - Checkpoint: $TT_METAL_HOME/models/experimental/pi0/weights/pi05_base
    - Full denoising: 10 steps
    - Batch size: 1

Usage:
    python test_pcc_pi05_model.py
"""

import sys
import os
import time
from pathlib import Path

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


TT_METAL_HOME = os.environ.get("TT_METAL_HOME", "/home/ttuser/experiments/pi0_5/tt-metal")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi05_base")
BATCH_SIZE = 1
SEED = 42
PCC_THRESHOLD = 0.93


def create_pi05_config() -> PI0ModelConfig:
    """Create PI0.5 model config with adaRMS enabled."""
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,  # Enable Pi0.5 mode
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


def main():
    """Run Pi0.5 PCC test."""
    print("=" * 80)
    print("  PI0.5 TTNN PCC TEST (adaRMS)")
    print("  TTNN (tt/) vs PyTorch (reference/)")
    print("=" * 80)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"\n📁 Checkpoint: {checkpoint_path}")

    # Open device
    print("\n🔌 Opening TTNN device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    grid = device.compute_with_storage_grid_size()
    print(f"✅ Device opened (grid: {grid.x}x{grid.y})")

    try:
        config = create_pi05_config()
        print(f"\n📋 Pi0.5 mode: pi05={config.pi05}, adaRMS={config.expert_config.use_adarms}")

        # Load weights
        print("\n1. Loading weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        print("   ✅ Weights loaded")

        # Check available weight keys
        pi0_proj = weight_loader.get_pi0_projections()
        print(f"   Pi0 projection keys: {sorted(pi0_proj.keys())}")

        # Initialize PyTorch reference model
        print("\n2. Initializing PyTorch reference model...")
        model_torch = PI0ModelTorch(config, weight_loader)
        print("   ✅ PyTorch model initialized")

        # Initialize TTNN model
        print("\n3. Initializing TTNN model...")
        torch.manual_seed(SEED)
        model_ttnn = PI0ModelTTNN(config, weight_loader, device)
        print("   ✅ TTNN model initialized")

        # Create inputs
        print("\n4. Creating test inputs...")
        inputs = create_test_inputs(config, batch_size=BATCH_SIZE)

        # Run PyTorch reference
        print("\n5. Running PyTorch reference...")
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
        print(f"   PyTorch: shape={torch_actions.shape}, time={torch_time:.1f}ms")

        # Run TTNN
        print("\n6. Running TTNN model...")
        start = time.time()
        with torch.no_grad():
            # Convert inputs to TTNN
            images_ttnn = [
                ttnn.from_torch(
                    img,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for img in inputs["images"]
            ]
            lang_tokens_ttnn = ttnn.from_torch(
                inputs["lang_tokens"],
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            lang_masks_ttnn = ttnn.from_torch(
                inputs["lang_masks"].float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            state_ttnn = ttnn.from_torch(
                inputs["state"],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

            ttnn_actions = model_ttnn.sample_actions(
                images=images_ttnn,
                img_masks=inputs["img_masks"],
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
            )
        ttnn_time = (time.time() - start) * 1000
        print(f"   TTNN: time={ttnn_time:.1f}ms")

        # Convert TTNN output to torch
        if isinstance(ttnn_actions, ttnn.Tensor):
            ttnn_actions = ttnn.to_torch(ttnn_actions)
        print(f"   TTNN output shape: {ttnn_actions.shape}")

        # Compute PCC
        pcc = compute_pcc(torch_actions, ttnn_actions)
        passed = pcc >= PCC_THRESHOLD

        # Results
        print("\n" + "=" * 80)
        print("  RESULTS")
        print("=" * 80)
        print(f"\n   PyTorch Time: {torch_time:.1f}ms")
        print(f"   TTNN Time:    {ttnn_time:.1f}ms")
        if ttnn_time > 0 and torch_time > 0 and ttnn_time < torch_time:
            print(f"   Speedup:      {torch_time / ttnn_time:.2f}x")
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


if __name__ == "__main__":
    sys.exit(main())
