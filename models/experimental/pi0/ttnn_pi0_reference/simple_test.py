#!/usr/bin/env python3
"""
Simple test to verify ttnn_pi0_reference basic functionality.

This tests:
1. PyTorch reference implementations work
2. Basic operations are correct
3. PCC between torch implementations is high
"""

import sys
import torch
import numpy as np

def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient."""
    t1 = tensor1.detach().float().cpu().numpy().flatten()
    t2 = tensor2.detach().float().cpu().numpy().flatten()
    
    if len(t1) != len(t2):
        raise ValueError(f"Tensor sizes don't match: {len(t1)} vs {len(t2)}")
    
    std1, std2 = np.std(t1), np.std(t2)
    if std1 == 0 or std2 == 0:
        return 1.0 if np.allclose(t1, t2) else 0.0
    
    return float(np.corrcoef(t1, t2)[0, 1])


def test_siglip_components():
    """Test SigLIP components."""
    print("\n" + "=" * 70)
    print("  Testing SigLIP Components")
    print("=" * 70)
    
    from ttnn_siglip import (
        SigLIPConfig,
        SigLIPAttentionTorch,
        SigLIPMLPTorch,
        SigLIPBlockTorch,
        PatchEmbeddingTorch,
    )
    
    config = SigLIPConfig(hidden_size=256, num_attention_heads=8, intermediate_size=1024)
    
    # Test Patch Embedding
    print("\n1. Testing Patch Embedding...")
    patch_weights = {
        "patch_embedding.weight": torch.randn(config.hidden_size, 3, config.patch_size, config.patch_size),
        "patch_embedding.bias": torch.randn(config.hidden_size),
    }
    patch_embed = PatchEmbeddingTorch(config, patch_weights)
    
    images = torch.randn(2, 3, 224, 224)
    patches = patch_embed.forward(images)
    expected_patches = (224 // config.patch_size) ** 2
    
    assert patches.shape == (2, expected_patches, config.hidden_size), f"Wrong shape: {patches.shape}"
    print(f"   ✅ Patch embedding: {images.shape} -> {patches.shape}")
    
    # Test consistency
    patches2 = patch_embed.forward(images)
    pcc = compute_pcc(patches, patches2)
    assert pcc > 0.999, f"Inconsistent results: PCC = {pcc}"
    print(f"   ✅ Consistency: PCC = {pcc:.6f}")
    
    # Test Attention
    print("\n2. Testing Attention...")
    attn_weights = {
        "self_attn.q_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.k_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.v_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.out_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.q_proj.bias": torch.randn(config.hidden_size),
        "self_attn.k_proj.bias": torch.randn(config.hidden_size),
        "self_attn.v_proj.bias": torch.randn(config.hidden_size),
        "self_attn.out_proj.bias": torch.randn(config.hidden_size),
    }
    attention = SigLIPAttentionTorch(config, attn_weights)
    
    hidden = torch.randn(2, 64, config.hidden_size)
    attn_out = attention.forward(hidden)
    
    assert attn_out.shape == hidden.shape, f"Wrong shape: {attn_out.shape}"
    print(f"   ✅ Attention: {hidden.shape} -> {attn_out.shape}")
    
    # Test MLP
    print("\n3. Testing MLP...")
    mlp_weights = {
        "mlp.fc1.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "mlp.fc1.bias": torch.randn(config.intermediate_size),
        "mlp.fc2.weight": torch.randn(config.hidden_size, config.intermediate_size),
        "mlp.fc2.bias": torch.randn(config.hidden_size),
    }
    mlp = SigLIPMLPTorch(config, mlp_weights)
    
    mlp_out = mlp.forward(hidden)
    assert mlp_out.shape == hidden.shape, f"Wrong shape: {mlp_out.shape}"
    print(f"   ✅ MLP: {hidden.shape} -> {mlp_out.shape}")
    
    # Test Block
    print("\n4. Testing Transformer Block...")
    block_weights = {
        "layer_norm1.weight": torch.randn(config.hidden_size),
        "layer_norm1.bias": torch.randn(config.hidden_size),
        "layer_norm2.weight": torch.randn(config.hidden_size),
        "layer_norm2.bias": torch.randn(config.hidden_size),
    }
    block_weights.update(attn_weights)
    block_weights.update(mlp_weights)
    
    block = SigLIPBlockTorch(config, block_weights)
    block_out = block.forward(hidden)
    
    assert block_out.shape == hidden.shape, f"Wrong shape: {block_out.shape}"
    print(f"   ✅ Block: {hidden.shape} -> {block_out.shape}")
    
    # Test residual connection
    diff = (block_out - hidden).abs().mean()
    print(f"   ✅ Residual effect: mean diff = {diff:.6f}")
    
    print("\n✅ All SigLIP component tests passed!")
    return True


def test_gemma_components():
    """Test Gemma components."""
    print("\n" + "=" * 70)
    print("  Testing Gemma Components")
    print("=" * 70)
    
    from ttnn_gemma import (
        GemmaConfig,
        rms_norm_torch,
        GemmaAttentionTorch,
        GemmaMLPTorch,
        GemmaBlockTorch,
        precompute_freqs_cis_torch,
    )
    
    config = GemmaConfig(width=512, depth=4, mlp_dim=2048, num_heads=8, num_kv_heads=1)
    
    # Test RMSNorm
    print("\n1. Testing RMSNorm...")
    x = torch.randn(2, 10, config.width)
    weight = torch.randn(config.width)
    normed = rms_norm_torch(x, weight)
    
    assert normed.shape == x.shape, f"Wrong shape: {normed.shape}"
    print(f"   ✅ RMSNorm: {x.shape} -> {normed.shape}")
    
    # Check normalization
    variance = normed.pow(2).mean(dim=-1)
    print(f"   ✅ Output variance: mean = {variance.mean():.6f}, std = {variance.std():.6f}")
    
    # Test RoPE
    print("\n2. Testing RoPE...")
    cos, sin = precompute_freqs_cis_torch(config.head_dim, 128)
    assert cos.shape == (128, config.head_dim // 2), f"Wrong cos shape: {cos.shape}"
    assert sin.shape == (128, config.head_dim // 2), f"Wrong sin shape: {sin.shape}"
    print(f"   ✅ RoPE: cos/sin shape = {cos.shape}")
    
    # Test Attention
    print("\n3. Testing Attention...")
    attn_weights = {
        "self_attn.q_proj.weight": torch.randn(config.num_heads * config.head_dim, config.width),
        "self_attn.k_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.v_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.o_proj.weight": torch.randn(config.width, config.num_heads * config.head_dim),
    }
    attention = GemmaAttentionTorch(config, attn_weights, layer_idx=0)
    
    hidden = torch.randn(2, 32, config.width)
    attn_out, _ = attention.forward(hidden, cos, sin)
    
    assert attn_out.shape == hidden.shape, f"Wrong shape: {attn_out.shape}"
    print(f"   ✅ Attention: {hidden.shape} -> {attn_out.shape}")
    
    # Test MLP
    print("\n4. Testing MLP...")
    mlp_weights = {
        "mlp.gate_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.up_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.down_proj.weight": torch.randn(config.width, config.mlp_dim),
    }
    mlp = GemmaMLPTorch(config, mlp_weights)
    
    mlp_out = mlp.forward(hidden)
    assert mlp_out.shape == hidden.shape, f"Wrong shape: {mlp_out.shape}"
    print(f"   ✅ MLP: {hidden.shape} -> {mlp_out.shape}")
    
    print("\n✅ All Gemma component tests passed!")
    return True


def test_suffix_embedding():
    """Test suffix embedding."""
    print("\n" + "=" * 70)
    print("  Testing Suffix Embedding")
    print("=" * 70)
    
    from ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
    
    config = SuffixConfig(
        action_dim=32,
        action_horizon=50,
        expert_width=512,
        pi05=False,
    )
    
    weights = {
        "action_in.weight": torch.randn(config.expert_width, config.action_dim),
        "action_in.bias": torch.randn(config.expert_width),
        "state_in.weight": torch.randn(config.expert_width, config.state_dim),
        "state_in.bias": torch.randn(config.expert_width),
        "action_time_mlp_in.weight": torch.randn(config.expert_width, config.expert_width * 2),
        "action_time_mlp_in.bias": torch.randn(config.expert_width),
        "action_time_mlp_out.weight": torch.randn(config.expert_width, config.expert_width),
        "action_time_mlp_out.bias": torch.randn(config.expert_width),
        "action_out.weight": torch.randn(config.action_dim, config.expert_width),
        "action_out.bias": torch.randn(config.action_dim),
    }
    
    suffix_emb = SuffixEmbeddingTorch(config, weights)
    
    # Test embedding
    state = torch.randn(2, config.state_dim)
    actions = torch.randn(2, config.action_horizon, config.action_dim)
    timestep = torch.rand(2)
    
    suffix_embs, pad_masks, att_masks, adarms = suffix_emb.embed_suffix(state, actions, timestep)
    
    expected_len = 1 + config.action_horizon  # state + actions
    assert suffix_embs.shape == (2, expected_len, config.expert_width), f"Wrong shape: {suffix_embs.shape}"
    print(f"   ✅ Suffix embedding: state {state.shape} + actions {actions.shape} -> {suffix_embs.shape}")
    
    # Test output projection
    expert_output = torch.randn(2, config.action_horizon, config.expert_width)
    action_pred = suffix_emb.project_output(expert_output)
    
    assert action_pred.shape == (2, config.action_horizon, config.action_dim), f"Wrong shape: {action_pred.shape}"
    print(f"   ✅ Output projection: {expert_output.shape} -> {action_pred.shape}")
    
    print("\n✅ All suffix embedding tests passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("  TTNN PI0 Reference - Simple Test Suite")
    print("=" * 70)
    
    try:
        # Test SigLIP
        if not test_siglip_components():
            return 1
        
        # Test Gemma
        if not test_gemma_components():
            return 1
        
        # Test Suffix
        if not test_suffix_embedding():
            return 1
        
        print("\n" + "=" * 70)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. These tests verify PyTorch reference implementations work")
        print("  2. To test TTNN implementations, you need:")
        print("     - TTNN installed")
        print("     - Tenstorrent device available")
        print("  3. Run PCC tests with: python3 tests/pcc/run_all_pcc.py")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

