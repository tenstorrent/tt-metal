# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: Suffix Embedding - TTNN vs PyTorch

Tests the suffix embedding module with both random and real checkpoint weights.

Usage:
    pytest test_pcc_suffix_full.py -v
    pytest test_pcc_suffix_full.py -v -k "pretrained_weight_true"   # Only real weights
    pytest test_pcc_suffix_full.py -v -k "pretrained_weight_false"  # Only random weights (fast)
    python test_pcc_suffix_full.py  # Standalone
"""

import sys
import os
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.reference.torch_suffix import SuffixEmbedding as SuffixEmbeddingTorch
from models.experimental.pi0.tt.ttnn_suffix import SuffixEmbeddingTTNN, convert_suffix_weights_to_ttnn
from models.experimental.pi0.common.configs import SuffixConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
SEED = 42
PCC_THRESHOLD = 0.93


def create_suffix_config() -> SuffixConfig:
    """Create SuffixConfig matching checkpoint."""
    return SuffixConfig(
        action_dim=32,
        action_horizon=50,
        expert_width=1024,
        state_dim=32,
        time_emb_dim=1024,
        pi05=False,
    )


def create_random_suffix_weights(config: SuffixConfig) -> dict:
    """Create random weights for fast testing."""
    return {
        "action_in_proj.weight": torch.randn(config.expert_width, config.action_dim),
        "action_in_proj.bias": torch.randn(config.expert_width),
        "action_out_proj.weight": torch.randn(config.action_dim, config.expert_width),
        "action_out_proj.bias": torch.randn(config.action_dim),
        "state_proj.weight": torch.randn(config.expert_width, config.state_dim),
        "state_proj.bias": torch.randn(config.expert_width),
        "action_time_mlp_in.weight": torch.randn(config.time_emb_dim, config.expert_width * 2),
        "action_time_mlp_in.bias": torch.randn(config.time_emb_dim),
        "action_time_mlp_out.weight": torch.randn(config.expert_width, config.time_emb_dim),
        "action_time_mlp_out.bias": torch.randn(config.expert_width),
    }


def get_suffix_weights(use_pretrained: bool, config: SuffixConfig):
    """Get weights - either from checkpoint or random."""
    if use_pretrained:
        checkpoint_path = Path(CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        return weight_loader.get_pi0_projections()
    else:
        return create_random_suffix_weights(config)


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
def test_pcc_suffix_embed_actions(device, use_pretrained):
    """Test suffix action embedding: TTNN vs PyTorch."""
    torch.manual_seed(SEED)
    config = create_suffix_config()
    suffix_weights = get_suffix_weights(use_pretrained, config)

    # Create input
    noisy_actions = torch.randn(1, config.action_horizon, config.action_dim)

    # PyTorch forward
    model_torch = SuffixEmbeddingTorch(config, suffix_weights)
    out_torch = model_torch.embed_actions(noisy_actions)

    # TTNN forward
    weights_ttnn = convert_suffix_weights_to_ttnn(suffix_weights, device)
    model_ttnn = SuffixEmbeddingTTNN(config, weights_ttnn, device)
    actions_ttnn = ttnn.from_torch(noisy_actions, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_ttnn = model_ttnn.embed_actions(actions_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Suffix embed_actions PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_suffix_embed_state(device, use_pretrained):
    """Test suffix state embedding: TTNN vs PyTorch."""
    torch.manual_seed(SEED)
    config = create_suffix_config()
    suffix_weights = get_suffix_weights(use_pretrained, config)

    # Create input
    state = torch.randn(1, config.state_dim)

    # PyTorch
    model_torch = SuffixEmbeddingTorch(config, suffix_weights)
    out_torch = model_torch.embed_state(state)

    # TTNN
    weights_ttnn = convert_suffix_weights_to_ttnn(suffix_weights, device)
    model_ttnn = SuffixEmbeddingTTNN(config, weights_ttnn, device)
    state_ttnn = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_ttnn = model_ttnn.embed_state(state_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Suffix embed_state PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_suffix_project_output(device, use_pretrained):
    """Test suffix output projection: TTNN vs PyTorch."""
    torch.manual_seed(SEED)
    config = create_suffix_config()
    suffix_weights = get_suffix_weights(use_pretrained, config)

    # Create input (simulated expert output)
    expert_output = torch.randn(1, config.action_horizon, config.expert_width)

    # PyTorch
    model_torch = SuffixEmbeddingTorch(config, suffix_weights)
    out_torch = model_torch.project_output(expert_output)

    # TTNN
    weights_ttnn = convert_suffix_weights_to_ttnn(suffix_weights, device)
    model_ttnn = SuffixEmbeddingTTNN(config, weights_ttnn, device)
    expert_ttnn = ttnn.from_torch(expert_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_ttnn = model_ttnn.project_output(expert_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Suffix project_output PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_suffix_full_embed(device, use_pretrained):
    """Test full suffix embedding: TTNN vs PyTorch."""
    torch.manual_seed(SEED)
    config = create_suffix_config()
    suffix_weights = get_suffix_weights(use_pretrained, config)

    # Create inputs
    state = torch.randn(1, config.state_dim)
    noisy_actions = torch.randn(1, config.action_horizon, config.action_dim)
    timestep = torch.tensor([0.5])

    # PyTorch
    model_torch = SuffixEmbeddingTorch(config, suffix_weights)
    embs_torch, _, _, _ = model_torch.embed_suffix(state, noisy_actions, timestep)

    # TTNN
    weights_ttnn = convert_suffix_weights_to_ttnn(suffix_weights, device)
    model_ttnn = SuffixEmbeddingTTNN(config, weights_ttnn, device)

    state_ttnn = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actions_ttnn = ttnn.from_torch(noisy_actions, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    timestep_ttnn = ttnn.from_torch(timestep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    embs_ttnn, _, _, _ = model_ttnn.embed_suffix(state_ttnn, actions_ttnn, timestep_ttnn)

    if isinstance(embs_ttnn, ttnn.Tensor):
        embs_ttnn = ttnn.to_torch(embs_ttnn)

    pcc = compute_pcc(embs_torch, embs_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Suffix full embed_suffix PCC ({weight_type}): {pcc:.6f}")
    print(f"   Output shape: {embs_ttnn.shape}")
    assert pcc >= 0.90, f"PCC {pcc:.6f} < threshold 0.90"


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  Suffix Embedding PCC Test (Checkpoint Weights)")
    print("=" * 70)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    torch.manual_seed(SEED)
    config = create_suffix_config()

    print(f"\n📁 Checkpoint: {checkpoint_path}")
    print(f"📋 Config: action_dim={config.action_dim}, horizon={config.action_horizon}")

    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        print("\n1. Loading checkpoint weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        suffix_weights = weight_loader.get_pi0_projections()
        print(f"   ✅ Loaded {len(suffix_weights)} suffix weight tensors")

        print("\n2. Testing embed_actions...")
        noisy_actions = torch.randn(1, config.action_horizon, config.action_dim)

        model_torch = SuffixEmbeddingTorch(config, suffix_weights)
        out_torch = model_torch.embed_actions(noisy_actions)

        weights_ttnn = convert_suffix_weights_to_ttnn(suffix_weights, device)
        model_ttnn = SuffixEmbeddingTTNN(config, weights_ttnn, device)
        actions_ttnn = ttnn.from_torch(noisy_actions, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = model_ttnn.embed_actions(actions_ttnn)

        if isinstance(out_ttnn, ttnn.Tensor):
            out_ttnn = ttnn.to_torch(out_ttnn)

        pcc = compute_pcc(out_torch, out_ttnn)
        passed = pcc >= PCC_THRESHOLD

        print("\n" + "=" * 70)
        print("  RESULTS - embed_actions (pretrained)")
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
