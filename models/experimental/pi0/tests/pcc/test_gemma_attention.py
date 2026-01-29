# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: Gemma Attention - TTNN vs PyTorch

Tests Gemma attention blocks with both random and real checkpoint weights.

Usage:
    pytest test_pcc_gemma_attention.py -v
    pytest test_pcc_gemma_attention.py -v -k "pretrained_weight_true"   # Only real weights
    pytest test_pcc_gemma_attention.py -v -k "pretrained_weight_false"  # Only random weights (fast)
    python test_pcc_gemma_attention.py  # Standalone
"""

import sys
import os
from pathlib import Path

import pytest
import torch
import ttnn
from models.tt_transformers.tt.load_checkpoints import split_hf_keys


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.reference.torch_gemma import (
    GemmaAttention as GemmaAttentionTorch,
)
from models.experimental.pi0.tt.ttnn_gemma import (
    GemmaAttentionTTNN,
    precompute_freqs_cis_meta_format,
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


def create_random_attention_weights(config: GemmaConfig) -> dict:
    """Create random weights for fast testing."""
    # Fused QKV weight: [hidden_dim, (num_heads + 2*num_kv_heads) * head_dim]
    qkv_dim = (config.num_heads + 2 * config.num_kv_heads) * config.head_dim
    return {
        "self_attn.qkv_proj.weight": torch.randn(config.width, qkv_dim).T,
        "self_attn.o_proj.weight": torch.randn(config.width, config.width),
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


def get_attention_weights(use_pretrained: bool, config: GemmaConfig, component: str = "action_expert"):
    """Get attention weights - either from checkpoint or random."""
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
        return {k: v for k, v in block_weights.items() if "self_attn" in k}
    else:
        return create_random_attention_weights(config)


def split_qkv_weights(weights: dict, config: GemmaConfig) -> dict:
    """Split fused QKV weights into separate Q, K, V for PyTorch reference."""
    if "self_attn.wqkv" in weights:
        wqkv = weights["self_attn.wqkv"]  # Shape: (2048, 2560)

        # Calculate dimensions
        q_dim = config.num_heads * config.head_dim  # 8 * 256 = 2048
        kv_dim = config.num_kv_heads * config.head_dim  # 1 * 256 = 256

        # Slice along COLUMNS (dimension 1)
        q_proj = wqkv[:, :q_dim]  # (2048, 2048)
        k_proj = wqkv[:, q_dim : q_dim + kv_dim]  # (2048, 256)
        v_proj = wqkv[:, q_dim + kv_dim : q_dim + 2 * kv_dim]  # (2048, 256)

        split_weights = weights.copy()
        # Store weights in PyTorch linear format: (in_features, out_features)
        split_weights["self_attn.q_proj.weight"] = q_proj
        split_weights["self_attn.k_proj.weight"] = k_proj
        split_weights["self_attn.v_proj.weight"] = v_proj

        return split_weights
    return weights


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [False],
    ids=["pretrained_weight_false"],
)
def test_pcc_gemma_vlm_attention(device, use_pretrained):
    """Test Gemma VLM Attention: TTNN vs PyTorch."""
    torch.manual_seed(SEED)
    config = create_vlm_config()
    attn_weights = get_attention_weights(use_pretrained, config, "vlm_language")

    # Create input
    seq_len = 32
    batch_size = 1
    hidden = torch.randn(batch_size, seq_len, config.width)

    # Create RoPE embeddings for TTNN
    cos_meta, sin_meta = precompute_freqs_cis_meta_format(
        config.head_dim,
        seq_len,
        device,
        base=10000.0,
    )

    attn_weights_torch = split_hf_keys(attn_weights.copy(), config.num_heads, config.num_kv_heads)

    # TTNN
    attn_weights_ttnn = {}
    for key in ["self_attn.qkv_proj.weight", "self_attn.o_proj.weight"]:
        if key in attn_weights:
            # Convert qkv_proj.weight to wqkv for TTNN
            ttnn_key = "self_attn.wqkv" if key == "self_attn.qkv_proj.weight" else key
            attn_weights_ttnn[ttnn_key] = ttnn.from_torch(
                attn_weights[key].T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    # PyTorch reference
    attn_torch = GemmaAttentionTorch(config, attn_weights_torch, layer_idx=0)
    # Create position IDs and cos/sin for PyTorch
    position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]

    # Precompute cos/sin for PyTorch
    inv_freq = 1.0 / (config.rope_base ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_pt = torch.cos(freqs)  # [seq_len, head_dim//2]
    sin_pt = torch.sin(freqs)  # [seq_len, head_dim//2]

    out_torch = attn_torch.forward(hidden, cos_pt, sin_pt)

    attn_ttnn = GemmaAttentionTTNN(
        config,
        attn_weights_ttnn,
        layer_idx=0,
        device=device,
        cos_meta=cos_meta,
        sin_meta=sin_meta,
    )

    hidden_ttnn = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Create dummy cos/sin for API compatibility (not used with native RoPE)
    cos_dummy = ttnn.zeros((1, 1, seq_len, config.head_dim), device=device)
    sin_dummy = ttnn.zeros((1, 1, seq_len, config.head_dim), device=device)

    print("INPUT SHAPE: ", hidden_ttnn.shape)
    out_ttnn, _ = attn_ttnn.forward(
        hidden_ttnn,
        cos_dummy,
        sin_dummy,
        use_cache=False,
    )
    print("OUTPUT SHAPE: ", out_ttnn.shape)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch[0], out_ttnn)  # Extract first element

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Gemma VLM Attention PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_gemma_expert_attention(device, use_pretrained):
    """Test Gemma Expert Attention: TTNN vs PyTorch."""
    torch.manual_seed(SEED)
    config = create_expert_config()
    attn_weights = get_attention_weights(use_pretrained, config, "action_expert")

    # Create input
    seq_len = 64
    batch_size = 1
    hidden = torch.randn(batch_size, seq_len, config.width)

    # Create RoPE embeddings
    cos_meta, sin_meta = precompute_freqs_cis_meta_format(
        config.head_dim,
        seq_len,
        device,
        base=10000.0,
    )

    # PyTorch reference
    attn_torch = GemmaAttentionTorch(config, attn_weights)
    out_torch = attn_torch.forward(hidden)

    # TTNN
    attn_weights_ttnn = {}
    for key in ["self_attn.wqkv", "self_attn.o_proj.weight"]:
        if key in attn_weights:
            attn_weights_ttnn[key] = ttnn.from_torch(
                attn_weights[key].T.contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    attn_ttnn = GemmaAttentionTTNN(
        config,
        attn_weights_ttnn,
        layer_idx=0,
        device=device,
        cos_meta=cos_meta,
        sin_meta=sin_meta,
    )

    hidden_ttnn = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Create dummy cos/sin for API compatibility (not used with native RoPE)
    cos_dummy = ttnn.zeros((1, 1, seq_len, config.head_dim), device=device)
    sin_dummy = ttnn.zeros((1, 1, seq_len, config.head_dim), device=device)

    out_ttnn, _ = attn_ttnn.forward(
        hidden_ttnn,
        cos_dummy,
        sin_dummy,
        use_cache=False,
    )

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Gemma Expert Attention PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  Gemma Attention PCC Test (Checkpoint Weights)")
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

        # Test Expert Attention (smaller, faster)
        print("\n2. Testing Expert Attention (300M)...")
        config = create_expert_config()
        expert_weights = weight_loader.get_action_expert_weights()
        block_weights = get_block_weights(expert_weights, layer_idx=0)
        attn_weights = {k: v for k, v in block_weights.items() if "self_attn" in k}

        seq_len = 64
        batch_size = 1
        hidden = torch.randn(batch_size, seq_len, config.width)

        # Create RoPE embeddings
        cos_meta, sin_meta = precompute_freqs_cis_meta_format(
            config.head_dim,
            seq_len,
            device,
            base=10000.0,
        )

        attn_torch = GemmaAttentionTorch(config, attn_weights)
        out_torch = attn_torch.forward(hidden)

        attn_weights_ttnn = {}
        for key in ["self_attn.wqkv", "self_attn.o_proj.weight"]:
            if key in attn_weights:
                attn_weights_ttnn[key] = ttnn.from_torch(
                    attn_weights[key].T.contiguous(),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

        attn_ttnn = GemmaAttentionTTNN(
            config,
            attn_weights_ttnn,
            layer_idx=0,
            device=device,
            cos_meta=cos_meta,
            sin_meta=sin_meta,
        )

        hidden_ttnn = ttnn.from_torch(
            hidden,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        cos_dummy = ttnn.zeros((1, 1, seq_len, config.head_dim), device=device)
        sin_dummy = ttnn.zeros((1, 1, seq_len, config.head_dim), device=device)

        out_ttnn, _ = attn_ttnn.forward(
            hidden_ttnn,
            cos_dummy,
            sin_dummy,
            use_cache=False,
        )

        if isinstance(out_ttnn, ttnn.Tensor):
            out_ttnn = ttnn.to_torch(out_ttnn)

        pcc = compute_pcc(out_torch, out_ttnn)
        passed = pcc >= PCC_THRESHOLD

        print("\n" + "=" * 70)
        print("  RESULTS - Expert Attention (pretrained)")
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
