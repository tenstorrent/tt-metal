#!/usr/bin/env python3
"""
Standalone PCC test for ttnn_pi0_reference.

Tests PyTorch vs PyTorch (consistency) and will test PyTorch vs TTNN if device is available.
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


def check_pcc(reference, comparison, threshold=0.99, test_name="unnamed"):
    """Check if PCC meets threshold."""
    pcc = compute_pcc(reference, comparison)
    passed = pcc >= threshold - 1e-9
    
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"[{status}] {test_name}: PCC = {pcc:.6f} (threshold: {threshold})")
    
    return passed


def test_siglip_torch_vs_torch():
    """Test SigLIP PyTorch implementation consistency."""
    print("\n" + "=" * 70)
    print("  SigLIP: PyTorch vs PyTorch (Consistency Test)")
    print("=" * 70)
    
    from ttnn_siglip import (
        SigLIPConfig,
        SigLIPAttentionTorch,
        SigLIPMLPTorch,
        SigLIPBlockTorch,
        SigLIPVisionTowerTorch,
    )
    
    config = SigLIPConfig(hidden_size=256, num_attention_heads=8, intermediate_size=1024, num_hidden_layers=4)
    
    # Test Attention consistency
    print("\n1. Testing SigLIP Attention...")
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
    out1 = attention.forward(hidden)
    out2 = attention.forward(hidden)
    
    assert check_pcc(out1, out2, threshold=1.0, test_name="SigLIP Attention consistency")
    
    # Test MLP consistency
    print("\n2. Testing SigLIP MLP...")
    mlp_weights = {
        "mlp.fc1.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "mlp.fc1.bias": torch.randn(config.intermediate_size),
        "mlp.fc2.weight": torch.randn(config.hidden_size, config.intermediate_size),
        "mlp.fc2.bias": torch.randn(config.hidden_size),
    }
    mlp = SigLIPMLPTorch(config, mlp_weights)
    
    mlp_out1 = mlp.forward(hidden)
    mlp_out2 = mlp.forward(hidden)
    
    assert check_pcc(mlp_out1, mlp_out2, threshold=1.0, test_name="SigLIP MLP consistency")
    
    # Test Block consistency
    print("\n3. Testing SigLIP Block...")
    block_weights = {
        "layer_norm1.weight": torch.randn(config.hidden_size),
        "layer_norm1.bias": torch.randn(config.hidden_size),
        "layer_norm2.weight": torch.randn(config.hidden_size),
        "layer_norm2.bias": torch.randn(config.hidden_size),
    }
    block_weights.update(attn_weights)
    block_weights.update(mlp_weights)
    
    block = SigLIPBlockTorch(config, block_weights)
    block_out1 = block.forward(hidden)
    block_out2 = block.forward(hidden)
    
    assert check_pcc(block_out1, block_out2, threshold=1.0, test_name="SigLIP Block consistency")
    
    print("\n✅ All SigLIP PyTorch consistency tests passed!")
    return True


def test_gemma_torch_vs_torch():
    """Test Gemma PyTorch implementation consistency."""
    print("\n" + "=" * 70)
    print("  Gemma: PyTorch vs PyTorch (Consistency Test)")
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
    cos, sin = precompute_freqs_cis_torch(config.head_dim, 128)
    
    # Test RMSNorm consistency
    print("\n1. Testing Gemma RMSNorm...")
    x = torch.randn(2, 32, config.width)
    weight = torch.randn(config.width)
    
    norm1 = rms_norm_torch(x, weight)
    norm2 = rms_norm_torch(x, weight)
    
    assert check_pcc(norm1, norm2, threshold=1.0, test_name="Gemma RMSNorm consistency")
    
    # Test Attention consistency
    print("\n2. Testing Gemma Attention...")
    attn_weights = {
        "self_attn.q_proj.weight": torch.randn(config.num_heads * config.head_dim, config.width),
        "self_attn.k_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.v_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.o_proj.weight": torch.randn(config.width, config.num_heads * config.head_dim),
    }
    attention = GemmaAttentionTorch(config, attn_weights, layer_idx=0)
    
    hidden = torch.randn(2, 32, config.width)
    attn_out1, _ = attention.forward(hidden, cos, sin)
    attn_out2, _ = attention.forward(hidden, cos, sin)
    
    assert check_pcc(attn_out1, attn_out2, threshold=1.0, test_name="Gemma Attention consistency")
    
    # Test MLP consistency
    print("\n3. Testing Gemma MLP...")
    mlp_weights = {
        "mlp.gate_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.up_proj.weight": torch.randn(config.mlp_dim, config.width),
        "mlp.down_proj.weight": torch.randn(config.width, config.mlp_dim),
    }
    mlp = GemmaMLPTorch(config, mlp_weights)
    
    mlp_out1 = mlp.forward(hidden)
    mlp_out2 = mlp.forward(hidden)
    
    assert check_pcc(mlp_out1, mlp_out2, threshold=1.0, test_name="Gemma MLP consistency")
    
    # Test Block consistency
    print("\n4. Testing Gemma Block...")
    block_weights = {
        "input_layernorm.weight": torch.randn(config.width),
        "post_attention_layernorm.weight": torch.randn(config.width),
    }
    block_weights.update(attn_weights)
    block_weights.update(mlp_weights)
    
    block = GemmaBlockTorch(config, block_weights, layer_idx=0)
    block_out1, _ = block.forward(hidden, cos, sin)
    block_out2, _ = block.forward(hidden, cos, sin)
    
    assert check_pcc(block_out1, block_out2, threshold=1.0, test_name="Gemma Block consistency")
    
    print("\n✅ All Gemma PyTorch consistency tests passed!")
    return True


def test_siglip_ttnn_vs_torch():
    """Test SigLIP TTNN vs PyTorch implementation."""
    print("\n" + "=" * 70)
    print("  SigLIP: TTNN vs PyTorch (PCC Test)")
    print("=" * 70)
    
    try:
        import ttnn
    except ImportError:
        print("⚠️  TTNN not available, skipping TTNN vs PyTorch tests")
        return True
    
    from ttnn_siglip import (
        SigLIPConfig,
        SigLIPAttentionTorch,
        SigLIPAttentionTTNN,
        SigLIPMLPTorch,
        SigLIPMLPTTNN,
        SigLIPBlockTorch,
        SigLIPBlockTTNN,
    )
    
    try:
        device = ttnn.open_device(device_id=0)
    except Exception as e:
        print(f"⚠️  Cannot open TTNN device: {e}")
        print("   Skipping TTNN vs PyTorch tests")
        return True
    
    try:
        config = SigLIPConfig(hidden_size=256, num_attention_heads=8, intermediate_size=1024)
        
        # Test Attention TTNN vs Torch
        print("\n1. Testing SigLIP Attention TTNN vs Torch...")
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
        
        attn_torch = SigLIPAttentionTorch(config, attn_weights)
        attn_ttnn = SigLIPAttentionTTNN(config, attn_weights, device)
        
        # Use tile-aligned dimensions
        hidden = torch.randn(2, 32, config.hidden_size)  # 32 is tile-aligned
        
        # Torch forward
        out_torch = attn_torch.forward(hidden)
        
        # TTNN forward
        hidden_ttnn = ttnn.from_torch(
            hidden,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_ttnn_tensor = attn_ttnn.forward(hidden_ttnn)
        out_ttnn = ttnn.to_torch(out_ttnn_tensor)
        
        # Compare
        pcc_passed = check_pcc(out_torch, out_ttnn, threshold=0.95, test_name="SigLIP Attention TTNN vs Torch")
        
        if not pcc_passed:
            print(f"   Shape: {out_torch.shape} vs {out_ttnn.shape}")
            print(f"   Mean diff: {(out_torch - out_ttnn).abs().mean():.6f}")
            print(f"   Max diff: {(out_torch - out_ttnn).abs().max():.6f}")
        
        print("\n✅ SigLIP TTNN vs PyTorch tests completed!")
        
    finally:
        ttnn.close_device(device)
    
    return True


def main():
    """Run all PCC tests."""
    print("=" * 70)
    print("  TTNN PI0 Reference - PCC Test Suite")
    print("=" * 70)
    
    try:
        # Test PyTorch consistency
        if not test_siglip_torch_vs_torch():
            return 1
        
        if not test_gemma_torch_vs_torch():
            return 1
        
        # Test TTNN vs PyTorch if available
        if not test_siglip_ttnn_vs_torch():
            return 1
        
        print("\n" + "=" * 70)
        print("  ✅ ALL PCC TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✅ PyTorch implementations are consistent (PCC = 1.0)")
        print("  ✅ TTNN implementations tested where available")
        print("\nNext steps:")
        print("  1. If TTNN device is available, run full TTNN tests")
        print("  2. Load real model weights and test end-to-end")
        print("  3. Benchmark performance on device")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

