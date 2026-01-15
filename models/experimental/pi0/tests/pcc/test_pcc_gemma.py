# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: Gemma Blocks - TTNN vs PyTorch

Tests Gemma transformer blocks with both random and real checkpoint weights.

Usage:
    pytest test_pcc_gemma_full.py -v
    pytest test_pcc_gemma_full.py -v -k "pretrained_weight_true"   # Only real weights
    pytest test_pcc_gemma_full.py -v -k "pretrained_weight_false"  # Only random weights (fast)
    python test_pcc_gemma_full.py  # Standalone
"""

import sys
import os
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.reference.torch_gemma import (
    GemmaMLP as GemmaMLPTorch,
)
from models.experimental.pi0.tt.ttnn_gemma import (
    GemmaMLPTTNN,
)
from models.experimental.pi0.common.configs import GemmaConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
SEED = 42
PCC_THRESHOLD = 0.90


def create_vlm_config() -> GemmaConfig:
    """Create VLM Gemma config (2B)."""
    return GemmaConfig(
        width=2048,
        depth=18,
        mlp_dim=16384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )


def create_expert_config() -> GemmaConfig:
    """Create Expert Gemma config (300M)."""
    return GemmaConfig(
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
    )


def create_random_mlp_weights(config: GemmaConfig) -> dict:
    """Create random weights for fast testing."""
    return {
        "mlp.gate_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.up_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.down_proj.weight": torch.randn(config.width, config.mlp_dim),
    }


def get_block_weights(all_weights: dict, layer_idx: int) -> dict:
    """Extract weights for a specific layer."""
    prefix = f"model.layers.{layer_idx}."
    block_weights = {}
    for key, value in all_weights.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            block_weights[new_key] = value
    return block_weights


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


def get_mlp_weights(use_pretrained: bool, config: GemmaConfig, component: str = "action_expert"):
    """Get MLP weights - either from checkpoint or random."""
    if use_pretrained:
        checkpoint_path = Path(CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        if component == "action_expert":
            all_weights = weight_loader.get_action_expert_weights()
        else:
            all_weights = weight_loader.get_vlm_language_weights()
        block_weights = get_block_weights(all_weights, layer_idx=0)
        return {k: v for k, v in block_weights.items() if "mlp" in k}
    else:
        return create_random_mlp_weights(config)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_gemma_vlm_mlp(device, use_pretrained):
    """Test Gemma VLM MLP: TTNN vs PyTorch."""
    torch.manual_seed(SEED)
    config = create_vlm_config()
    mlp_weights = get_mlp_weights(use_pretrained, config, "vlm_language")

    # Create input
    seq_len = 64
    hidden = torch.randn(1, seq_len, config.width)

    # PyTorch
    mlp_torch = GemmaMLPTorch(config, mlp_weights)
    out_torch = mlp_torch.forward(hidden)

    # TTNN - convert weights
    mlp_weights_ttnn = {}
    for key in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
        if key in mlp_weights:
            mlp_weights_ttnn[key] = ttnn.from_torch(
                mlp_weights[key].T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    mlp_ttnn = GemmaMLPTTNN(config, mlp_weights_ttnn, device)
    hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_ttnn = mlp_ttnn.forward(hidden_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Gemma VLM MLP PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_gemma_expert_mlp(device, use_pretrained):
    """Test Gemma Expert MLP: TTNN vs PyTorch."""
    torch.manual_seed(SEED)
    config = create_expert_config()
    mlp_weights = get_mlp_weights(use_pretrained, config, "action_expert")

    # Create input
    seq_len = 64
    hidden = torch.randn(1, seq_len, config.width)

    # PyTorch
    mlp_torch = GemmaMLPTorch(config, mlp_weights)
    out_torch = mlp_torch.forward(hidden)

    # TTNN
    mlp_weights_ttnn = {}
    for key in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
        if key in mlp_weights:
            mlp_weights_ttnn[key] = ttnn.from_torch(
                mlp_weights[key].T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    mlp_ttnn = GemmaMLPTTNN(config, mlp_weights_ttnn, device)
    hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_ttnn = mlp_ttnn.forward(hidden_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Gemma Expert MLP PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  Gemma Blocks PCC Test (Checkpoint Weights)")
    print("=" * 70)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    torch.manual_seed(SEED)

    print(f"\n📁 Checkpoint: {checkpoint_path}")

    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        print("\n1. Loading checkpoint weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))

        # Test Expert MLP (smaller, faster)
        print("\n2. Testing Expert MLP (300M)...")
        config = create_expert_config()
        expert_weights = weight_loader.get_action_expert_weights()
        block_weights = get_block_weights(expert_weights, layer_idx=0)
        mlp_weights = {k: v for k, v in block_weights.items() if "mlp" in k}

        seq_len = 64
        hidden = torch.randn(1, seq_len, config.width)

        mlp_torch = GemmaMLPTorch(config, mlp_weights)
        out_torch = mlp_torch.forward(hidden)

        mlp_weights_ttnn = {}
        for key in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
            if key in mlp_weights:
                mlp_weights_ttnn[key] = ttnn.from_torch(
                    mlp_weights[key].T.contiguous(),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

        mlp_ttnn = GemmaMLPTTNN(config, mlp_weights_ttnn, device)
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = mlp_ttnn.forward(hidden_ttnn)

        if isinstance(out_ttnn, ttnn.Tensor):
            out_ttnn = ttnn.to_torch(out_ttnn)

        pcc = compute_pcc(out_torch, out_ttnn)
        passed = pcc >= PCC_THRESHOLD

        print("\n" + "=" * 70)
        print("  RESULTS - Expert MLP (pretrained)")
        print("=" * 70)
        print(f"   PCC:       {pcc:.6f}")
        print(f"   Threshold: {PCC_THRESHOLD}")
        print(f"   Status:    {'✅ PASS' if passed else '❌ FAIL'}")
        print("=" * 70)

        return 0 if passed else 1

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
