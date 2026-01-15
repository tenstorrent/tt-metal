#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Run all PCC tests with both pretrained_weight_true and pretrained_weight_false options.

Usage:
    python run_all_pcc_tests.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


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


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_suffix(device, use_pretrained: bool) -> float:
    """Test Suffix Embedding."""
    from models.experimental.pi0.tests.pcc.test_pcc_suffix import (
        create_suffix_config,
        get_suffix_weights,
    )
    from models.experimental.pi0.reference.torch_suffix import (
        SuffixEmbedding as SuffixTorch,
    )
    from models.experimental.pi0.tt.ttnn_suffix import (
        SuffixEmbeddingTTNN,
        convert_suffix_weights_to_ttnn,
    )

    config = create_suffix_config()
    weights = get_suffix_weights(use_pretrained, config)

    torch.manual_seed(42)
    noisy_actions = torch.randn(1, config.action_horizon, config.action_dim)

    # PyTorch
    model_torch = SuffixTorch(config, weights)
    out_torch = model_torch.embed_actions(noisy_actions)

    # TTNN
    weights_ttnn = convert_suffix_weights_to_ttnn(weights, device)
    model_ttnn = SuffixEmbeddingTTNN(config, weights_ttnn, device)
    actions_ttnn = ttnn.from_torch(noisy_actions, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_ttnn = model_ttnn.embed_actions(actions_ttnn)
    out_ttnn = ttnn.to_torch(out_ttnn)

    return compute_pcc(out_torch, out_ttnn)


def test_prefix(device, use_pretrained: bool) -> float:
    """Test Prefix Language Embedding."""
    from models.experimental.pi0.tests.pcc.test_pcc_prefix import get_embed_weights
    from models.experimental.pi0.common.configs import PrefixConfig
    from models.experimental.pi0.reference.torch_prefix import (
        PrefixEmbedding as PrefixTorch,
    )
    from models.experimental.pi0.tt.ttnn_prefix import PrefixEmbeddingTTNN

    hidden_dim = 2048
    vocab_size = 257152
    seq_len = 32

    embed_weights = get_embed_weights(use_pretrained, hidden_dim, vocab_size)
    actual_hidden_dim = embed_weights.shape[1]
    actual_vocab_size = embed_weights.shape[0]

    def embed_fn_torch(tokens):
        return torch.nn.functional.embedding(tokens, embed_weights)

    def embed_fn_ttnn(tokens):
        if isinstance(tokens, ttnn.Tensor):
            tokens = ttnn.to_torch(tokens)
        emb = torch.nn.functional.embedding(tokens, embed_weights)
        return ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    config = PrefixConfig(vlm_hidden_size=actual_hidden_dim)

    torch.manual_seed(42)
    lang_tokens = torch.randint(0, min(actual_vocab_size, 10000), (1, seq_len))
    lang_masks = torch.ones(1, seq_len, dtype=torch.bool)
    lang_masks_ttnn = ttnn.from_torch(lang_masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # PyTorch
    prefix_torch = PrefixTorch(config, embed_language_fn=embed_fn_torch)
    out_torch = prefix_torch.embed_language(lang_tokens, lang_masks)

    # TTNN
    prefix_ttnn = PrefixEmbeddingTTNN(config, device, embed_language_fn=embed_fn_ttnn)
    lang_tokens_ttnn = ttnn.from_torch(lang_tokens, device=device)
    out_ttnn = prefix_ttnn.embed_language(lang_tokens_ttnn, lang_masks_ttnn)
    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    return compute_pcc(out_torch, out_ttnn)


def test_gemma_mlp(device, use_pretrained: bool) -> float:
    """Test Gemma MLP."""
    from models.experimental.pi0.tests.pcc.test_pcc_gemma import (
        create_expert_config,
        get_mlp_weights,
    )
    from models.experimental.pi0.reference.torch_gemma import GemmaMLP as GemmaMLPTorch
    from models.experimental.pi0.tt.ttnn_gemma import GemmaMLPTTNN

    config = create_expert_config()
    mlp_weights = get_mlp_weights(use_pretrained, config, "action_expert")

    torch.manual_seed(42)
    x = torch.randn(1, 32, config.width)  # GemmaConfig uses 'width' not 'hidden_size'

    # PyTorch - call forward() directly (not nn.Module)
    model_torch = GemmaMLPTorch(config, mlp_weights)
    out_torch = model_torch.forward(x)

    # TTNN - call forward() directly
    model_ttnn = GemmaMLPTTNN(config, mlp_weights, device)
    out_ttnn = model_ttnn.forward(x)

    return compute_pcc(out_torch, out_ttnn)


def test_siglip(device, use_pretrained: bool) -> float:
    """Test SigLIP Vision Tower."""
    from models.experimental.pi0.tests.pcc.test_pcc_siglip import (
        create_siglip_config,
        create_small_siglip_config,
        create_random_siglip_weights,
    )
    from models.experimental.pi0.common.weight_loader import PI0WeightLoader
    from models.experimental.pi0.reference.torch_siglip import (
        SigLIPVisionTower as SigLIPTorch,
    )
    from models.experimental.pi0.tt.ttnn_siglip import SigLIPVisionTowerTTNN

    torch.manual_seed(42)

    if use_pretrained:
        config = create_siglip_config()
        tt_metal_home = os.environ.get("TT_METAL_HOME")
        if not tt_metal_home:
            raise EnvironmentError("TT_METAL_HOME environment variable is not set")
        checkpoint_path = os.path.join(tt_metal_home, "models/experimental/pi0/weights/pi0_base")
        weight_loader = PI0WeightLoader(checkpoint_path)
        weights = weight_loader.get_vlm_vision_weights()
    else:
        config = create_small_siglip_config()
        weights = create_random_siglip_weights(config)

    x = torch.randn(1, 3, config.image_size, config.image_size)

    # PyTorch - call forward() directly (not nn.Module)
    model_torch = SigLIPTorch(config, weights)
    out_torch = model_torch.forward(x)

    # TTNN - call forward() directly
    model_ttnn = SigLIPVisionTowerTTNN(config, weights, device)
    out_ttnn = model_ttnn.forward(x)

    # Convert to torch if needed
    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    return compute_pcc(out_torch, out_ttnn)


def test_paligemma(device, use_pretrained: bool) -> float:
    """Test PaliGemma Backbone (image embedding only)."""
    from models.experimental.pi0.tests.pcc.test_pcc_paligemma import (
        create_config,
        create_small_config,
        get_paligemma_weights,
    )
    from models.experimental.pi0.reference.torch_paligemma import (
        PaliGemmaBackbone as PaliGemmaTorch,
    )
    from models.experimental.pi0.tt.ttnn_paligemma import PaliGemmaBackboneTTNN

    torch.manual_seed(42)

    # Use smaller config for random tests (much faster)
    config = create_config() if use_pretrained else create_small_config()
    weights = get_paligemma_weights(use_pretrained, config)

    x = torch.randn(1, 3, config.siglip_config.image_size, config.siglip_config.image_size)

    # PyTorch
    model_torch = PaliGemmaTorch(config, weights)
    out_torch = model_torch.embed_image(x)

    # TTNN
    model_ttnn = PaliGemmaBackboneTTNN(config, weights, device)
    out_ttnn = model_ttnn.embed_image(x)

    # Convert TTNN tensor to torch if needed
    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    return compute_pcc(out_torch, out_ttnn)


def main():
    print("=" * 80)
    print("  PI0 PCC Tests - All Components")
    print("  pretrained_weight_true vs pretrained_weight_false")
    print("=" * 80)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    grid = device.compute_with_storage_grid_size()
    print(f"\n✅ Device opened (grid: {grid.x}x{grid.y})")

    results = []
    tests = [
        ("Suffix", test_suffix),
        ("Prefix", test_prefix),
        ("Gemma MLP", test_gemma_mlp),
        ("SigLIP", test_siglip),
        ("PaliGemma", test_paligemma),
    ]

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"  {test_name}")
        print("=" * 60)

        for use_pretrained in [True, False]:
            option = "pretrained_weight_true" if use_pretrained else "pretrained_weight_false"
            try:
                start = time.time()
                pcc = test_func(device, use_pretrained)
                elapsed = time.time() - start
                status = "✅ PASS" if pcc >= 0.93 else "❌ FAIL"
                print(f"  {option:25s} PCC: {pcc:.6f} {status} ({elapsed:.1f}s)")
                results.append((test_name, option, pcc, status))
            except Exception as e:
                print(f"  {option:25s} ❌ ERROR: {e}")
                import traceback

                traceback.print_exc()
                results.append((test_name, option, 0.0, "❌ ERROR"))

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"\n{'Component':<15} {'Option':<30} {'PCC':<12} {'Status'}")
    print("-" * 70)
    for test_name, option, pcc, status in results:
        print(f"{test_name:<15} {option:<30} {pcc:.6f}     {status}")

    ttnn.close_device(device)
    print("\n✅ All tests complete")

    all_passed = all(s == "✅ PASS" for _, _, _, s in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
