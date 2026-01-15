# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: PaliGemma Backbone - TTNN vs PyTorch

Tests the full PaliGemma backbone (VLM + Expert) with both random and real checkpoint weights.

Usage:
    pytest test_pcc_paligemma_full.py -v
    pytest test_pcc_paligemma_full.py -v -k "pretrained_weight_true"   # Only real weights
    pytest test_pcc_paligemma_full.py -v -k "pretrained_weight_false"  # Only random weights (fast)
    python test_pcc_paligemma_full.py  # Standalone
"""

import sys
import os
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.reference.torch_paligemma import PaliGemmaBackbone as PaliGemmaBackboneTorch
from models.experimental.pi0.tt.ttnn_paligemma import PaliGemmaBackboneTTNN
from models.experimental.pi0.common.configs import SigLIPConfig, GemmaConfig, PaliGemmaConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
SEED = 42
PCC_THRESHOLD = 0.90


def create_config() -> PaliGemmaConfig:
    """Create PaliGemma config matching checkpoint."""
    siglip = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    vlm = GemmaConfig(
        width=2048,
        depth=18,
        mlp_dim=16384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )
    expert = GemmaConfig(
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
    )
    return PaliGemmaConfig(
        siglip_config=siglip,
        vlm_config=vlm,
        expert_config=expert,
        max_seq_len=1024,
    )


def create_small_config() -> PaliGemmaConfig:
    """Create smaller PaliGemma config for fast random testing."""
    siglip = SigLIPConfig(
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=4,
        num_attention_heads=6,
        image_size=224,
        patch_size=14,
    )
    vlm = GemmaConfig(
        width=512,
        depth=2,
        mlp_dim=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=64,
    )
    expert = GemmaConfig(
        width=256,
        depth=2,
        mlp_dim=1024,
        num_heads=4,
        num_kv_heads=1,
        head_dim=64,
    )
    return PaliGemmaConfig(
        siglip_config=siglip,
        vlm_config=vlm,
        expert_config=expert,
        max_seq_len=256,
    )


def create_random_siglip_weights(config: SigLIPConfig) -> dict:
    """Create random SigLIP weights."""
    weights = {}
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    patch_size = config.patch_size
    num_patches = (config.image_size // patch_size) ** 2

    # SigLIP doesn't use class token, so position_embedding is exactly num_patches
    weights["vision_model.embeddings.patch_embedding.weight"] = torch.randn(hidden, 3, patch_size, patch_size)
    weights["vision_model.embeddings.patch_embedding.bias"] = torch.randn(hidden)
    weights["vision_model.embeddings.position_embedding.weight"] = torch.randn(num_patches, hidden)

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

    weights["vision_model.encoder.final_layer_norm.weight"] = torch.randn(hidden)
    weights["vision_model.encoder.final_layer_norm.bias"] = torch.randn(hidden)

    return weights


def create_random_gemma_weights(config: GemmaConfig, prefix: str = "") -> dict:
    """Create random Gemma weights."""
    weights = {}
    width = config.width
    mlp_dim = config.mlp_dim
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim

    # Embedding
    weights[f"{prefix}model.embed_tokens.weight"] = torch.randn(257152, width)
    weights[f"{prefix}model.norm.weight"] = torch.randn(width)

    for i in range(config.depth):
        layer_prefix = f"{prefix}model.layers.{i}."
        weights[f"{layer_prefix}input_layernorm.weight"] = torch.randn(width)
        weights[f"{layer_prefix}post_attention_layernorm.weight"] = torch.randn(width)
        weights[f"{layer_prefix}self_attn.q_proj.weight"] = torch.randn(num_heads * head_dim, width)
        weights[f"{layer_prefix}self_attn.k_proj.weight"] = torch.randn(num_kv_heads * head_dim, width)
        weights[f"{layer_prefix}self_attn.v_proj.weight"] = torch.randn(num_kv_heads * head_dim, width)
        weights[f"{layer_prefix}self_attn.o_proj.weight"] = torch.randn(width, num_heads * head_dim)
        weights[f"{layer_prefix}mlp.gate_proj.weight"] = torch.randn(mlp_dim, width)
        weights[f"{layer_prefix}mlp.up_proj.weight"] = torch.randn(mlp_dim, width)
        weights[f"{layer_prefix}mlp.down_proj.weight"] = torch.randn(width, mlp_dim)

    return weights


def create_random_projector_weights(in_size: int, out_size: int) -> dict:
    """Create random multi-modal projector weights (single linear layer)."""
    return {
        "linear.weight": torch.randn(out_size, in_size),
        "linear.bias": torch.randn(out_size),
    }


def create_random_paligemma_weights(config: PaliGemmaConfig) -> dict:
    """Create random weights for PaliGemma backbone."""
    weights = {
        "vlm_vision": create_random_siglip_weights(config.siglip_config),
        "vlm_language": create_random_gemma_weights(config.vlm_config),
        "vlm_projector": create_random_projector_weights(config.siglip_config.hidden_size, config.vlm_config.width),
        "action_expert": create_random_gemma_weights(config.expert_config),
    }
    return weights


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


def get_paligemma_weights(use_pretrained: bool, config: PaliGemmaConfig):
    """Get weights - either from checkpoint or random."""
    if use_pretrained:
        checkpoint_path = Path(CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        return weight_loader.categorized_weights
    else:
        return create_random_paligemma_weights(config)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_paligemma_embed_image(device, use_pretrained):
    """Test PaliGemma image embedding: TTNN vs PyTorch."""
    torch.manual_seed(SEED)

    # Use smaller config for random tests (much faster)
    config = create_config() if use_pretrained else create_small_config()
    weights = get_paligemma_weights(use_pretrained, config)

    # Create input
    pixel_values = torch.randn(1, 3, config.siglip_config.image_size, config.siglip_config.image_size)

    # PyTorch forward
    model_torch = PaliGemmaBackboneTorch(config, weights)
    out_torch = model_torch.embed_image(pixel_values)

    # TTNN forward
    model_ttnn = PaliGemmaBackboneTTNN(config, weights, device)
    out_ttnn = model_ttnn.embed_image(pixel_values)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ PaliGemma embed_image PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_paligemma_vlm_block(device, use_pretrained):
    """Test single PaliGemma VLM block: TTNN vs PyTorch."""
    torch.manual_seed(SEED)

    # Use smaller config for random tests
    config = create_config() if use_pretrained else create_small_config()
    weights = get_paligemma_weights(use_pretrained, config)

    # Create models
    model_torch = PaliGemmaBackboneTorch(config, weights)
    model_ttnn = PaliGemmaBackboneTTNN(config, weights, device)

    # Create input (simulated prefix embeddings)
    batch_size = 1
    seq_len = 64
    hidden = torch.randn(batch_size, seq_len, config.vlm_config.width)

    # Run single VLM block
    block_torch = model_torch.vlm_blocks[0]
    cos_torch, sin_torch = model_torch.cos[:seq_len], model_torch.sin[:seq_len]
    out_torch, _ = block_torch.forward(hidden, cos_torch, sin_torch)

    # TTNN
    hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    block_ttnn = model_ttnn.vlm_blocks[0]
    out_ttnn, _ = block_ttnn.forward(hidden_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ PaliGemma VLM Block[0] PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= 0.85, f"PCC {pcc:.6f} < threshold 0.85"


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  PaliGemma Backbone PCC Test (Checkpoint Weights)")
    print("=" * 70)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    torch.manual_seed(SEED)
    config = create_config()

    print(f"\n📁 Checkpoint: {checkpoint_path}")
    print(f"📋 VLM: {config.vlm_config.depth} layers, width={config.vlm_config.width}")
    print(f"📋 Expert: {config.expert_config.depth} layers, width={config.expert_config.width}")

    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        print("\n1. Loading checkpoint weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        weights = weight_loader.categorized_weights
        print(f"   ✅ Loaded weights")

        print("\n2. Testing embed_image...")
        pixel_values = torch.randn(1, 3, config.siglip_config.image_size, config.siglip_config.image_size)

        model_torch = PaliGemmaBackboneTorch(config, weights)
        out_torch = model_torch.embed_image(pixel_values)

        model_ttnn = PaliGemmaBackboneTTNN(config, weights, device)
        out_ttnn = model_ttnn.embed_image(pixel_values)

        if isinstance(out_ttnn, ttnn.Tensor):
            out_ttnn = ttnn.to_torch(out_ttnn)

        pcc = compute_pcc(out_torch, out_ttnn)
        passed = pcc >= PCC_THRESHOLD

        print("\n" + "=" * 70)
        print("  RESULTS - embed_image (pretrained)")
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
