# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC Test: Prefix Embedding - TTNN vs PyTorch

Tests the prefix embedding module with both random and real checkpoint weights.

Usage:
    pytest test_pcc_prefix_full.py -v
    pytest test_pcc_prefix_full.py -v -k "pretrained_weight_true"   # Only real weights
    pytest test_pcc_prefix_full.py -v -k "pretrained_weight_false"  # Only random weights (fast)
    python test_pcc_prefix_full.py  # Standalone
"""

import sys
import os
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.reference.torch_prefix import PrefixEmbedding as PrefixEmbeddingTorch
from models.experimental.pi0.tt.ttnn_prefix import PrefixEmbeddingTTNN
from models.experimental.pi0.common.configs import PrefixConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base")
SEED = 42
PCC_THRESHOLD = 0.95


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


def get_embed_weights(use_pretrained: bool, hidden_dim: int = 2048, vocab_size: int = 257152):
    """Get embedding weights - either from checkpoint or random."""
    if use_pretrained:
        checkpoint_path = Path(CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        vlm_weights = weight_loader.get_vlm_language_weights()
        embed_weights = vlm_weights.get("model.embed_tokens.weight") or vlm_weights.get("lm_head.weight")
        if embed_weights is None:
            pytest.skip("Could not find embedding weights in checkpoint")
        return embed_weights
    else:
        return torch.randn(vocab_size, hidden_dim)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained",
    [True, False],
    ids=["pretrained_weight_true", "pretrained_weight_false"],
)
def test_pcc_prefix_language_embedding(device, use_pretrained):
    """Test prefix language embedding: TTNN vs PyTorch."""
    torch.manual_seed(SEED)

    hidden_dim = 2048
    vocab_size = 257152 if use_pretrained else 10000  # Smaller for random
    seq_len = 32

    embed_weights = get_embed_weights(use_pretrained, hidden_dim, vocab_size)
    actual_vocab_size = embed_weights.shape[0]
    actual_hidden_dim = embed_weights.shape[1]

    def embed_fn_torch(tokens):
        return torch.nn.functional.embedding(tokens, embed_weights)

    def embed_fn_ttnn(tokens):
        if isinstance(tokens, ttnn.Tensor):
            tokens = ttnn.to_torch(tokens)
        emb = torch.nn.functional.embedding(tokens, embed_weights)
        return ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    config = PrefixConfig(vlm_hidden_size=actual_hidden_dim)

    # Create inputs with valid token IDs
    lang_tokens = torch.randint(0, min(actual_vocab_size, 10000), (1, seq_len))
    lang_masks = torch.ones(1, seq_len, dtype=torch.bool)

    lang_masks_ttnn = ttnn.from_torch(lang_masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # PyTorch
    prefix_torch = PrefixEmbeddingTorch(config, embed_language_fn=embed_fn_torch)
    out_torch = prefix_torch.embed_language(lang_tokens, lang_masks)

    # TTNN
    prefix_ttnn = PrefixEmbeddingTTNN(config, device, embed_language_fn=embed_fn_ttnn)
    lang_tokens_ttnn = ttnn.from_torch(lang_tokens, device=device)
    out_ttnn = prefix_ttnn.embed_language(lang_tokens_ttnn, lang_masks_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)

    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Prefix language embedding PCC ({weight_type}): {pcc:.6f}")
    print(f"   Embedding shape: {embed_weights.shape}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD}"


def main():
    """Standalone runner."""
    print("=" * 70)
    print("  Prefix Embedding PCC Test (Checkpoint Weights)")
    print("=" * 70)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    torch.manual_seed(SEED)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        print("\n1. Loading checkpoint weights...")
        weight_loader = PI0WeightLoader(str(checkpoint_path))
        vlm_weights = weight_loader.get_vlm_language_weights()

        embed_weights = vlm_weights.get("model.embed_tokens.weight") or vlm_weights.get("lm_head.weight")
        if embed_weights is None:
            print("❌ Could not find embedding weights")
            return 1

        hidden_dim = embed_weights.shape[1]
        vocab_size = embed_weights.shape[0]
        seq_len = 32

        print(f"   ✅ Loaded embed_tokens: {embed_weights.shape}")
        print(f"   Vocab size: {vocab_size}, Hidden dim: {hidden_dim}")

        def embed_fn_torch(tokens):
            return torch.nn.functional.embedding(tokens, embed_weights)

        def embed_fn_ttnn(tokens):
            if isinstance(tokens, ttnn.Tensor):
                tokens = ttnn.to_torch(tokens)
            emb = torch.nn.functional.embedding(tokens, embed_weights)
            return ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        config = PrefixConfig(vlm_hidden_size=hidden_dim)

        print("\n2. Testing embed_language with checkpoint weights...")
        lang_tokens = torch.randint(0, min(vocab_size, 10000), (1, seq_len))
        lang_masks = torch.ones(1, seq_len, dtype=torch.bool)

        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # PyTorch
        prefix_torch = PrefixEmbeddingTorch(config, embed_language_fn=embed_fn_torch)
        out_torch = prefix_torch.embed_language(lang_tokens, lang_masks)

        # TTNN
        prefix_ttnn = PrefixEmbeddingTTNN(config, device, embed_language_fn=embed_fn_ttnn)
        lang_tokens_ttnn = ttnn.from_torch(lang_tokens, device=device)
        out_ttnn = prefix_ttnn.embed_language(lang_tokens_ttnn, lang_masks_ttnn)

        if isinstance(out_ttnn, ttnn.Tensor):
            out_ttnn = ttnn.to_torch(out_ttnn)

        pcc = compute_pcc(out_torch, out_ttnn)
        passed = pcc >= PCC_THRESHOLD

        print("\n" + "=" * 70)
        print("  RESULTS - embed_language (Checkpoint Weights)")
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
