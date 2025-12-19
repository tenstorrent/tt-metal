#!/usr/bin/env python3
"""
Test TTNN PI0 Reference on device with PCC validation.

This script:
1. Opens TTNN device
2. Tests TTNN components on device
3. Compares with PyTorch reference (PCC)
4. Reports results
"""

import sys
import torch
import numpy as np

# Check TTNN
try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    print("❌ TTNN not available")
    sys.exit(1)

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


def check_pcc(reference, comparison, threshold=0.95, test_name="unnamed"):
    """Check if PCC meets threshold."""
    pcc = compute_pcc(reference, comparison)
    passed = pcc >= threshold - 1e-9
    
    status = "✓ PASSED" if passed else "✗ FAILED"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}[{status}]{reset} {test_name}: PCC = {pcc:.6f} (threshold: {threshold})")
    
    return passed, pcc


def test_siglip_attention_on_device(device):
    """Test SigLIP Attention TTNN vs PyTorch on device."""
    print("\n" + "=" * 70)
    print("  Testing SigLIP Attention on Device")
    print("=" * 70)
    
    from ttnn_siglip import (
        SigLIPConfig,
        SigLIPAttentionTorch,
        SigLIPAttentionTTNN,
    )
    
    # Use smaller dimensions for faster testing
    config = SigLIPConfig(hidden_size=256, num_attention_heads=8, intermediate_size=1024)
    
    # Create weights
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
    
    # Create models
    attn_torch = SigLIPAttentionTorch(config, attn_weights)
    attn_ttnn = SigLIPAttentionTTNN(config, attn_weights, device)
    
    # Test input (tile-aligned: 32 is divisible by 32)
    hidden = torch.randn(2, 32, config.hidden_size)
    
    # PyTorch forward
    print("Running PyTorch forward pass...")
    out_torch = attn_torch.forward(hidden)
    print(f"  Output shape: {out_torch.shape}")
    
    # TTNN forward
    print("Running TTNN forward pass on device...")
    hidden_ttnn = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_ttnn_tensor = attn_ttnn.forward(hidden_ttnn)
    out_ttnn = ttnn.to_torch(out_ttnn_tensor)
    print(f"  Output shape: {out_ttnn.shape}")
    
    # Compare
    passed, pcc = check_pcc(out_torch, out_ttnn, threshold=0.95, test_name="SigLIP Attention")
    
    if not passed:
        print(f"\n  Debugging info:")
        print(f"    Mean: {out_torch.mean():.6f} vs {out_ttnn.mean():.6f}")
        print(f"    Std: {out_torch.std():.6f} vs {out_ttnn.std():.6f}")
        print(f"    Mean abs diff: {(out_torch - out_ttnn).abs().mean():.6f}")
        print(f"    Max abs diff: {(out_torch - out_ttnn).abs().max():.6f}")
    
    return passed, pcc


def test_siglip_mlp_on_device(device):
    """Test SigLIP MLP TTNN vs PyTorch on device."""
    print("\n" + "=" * 70)
    print("  Testing SigLIP MLP on Device")
    print("=" * 70)
    
    from ttnn_siglip import (
        SigLIPConfig,
        SigLIPMLPTorch,
        SigLIPMLPTTNN,
    )
    
    config = SigLIPConfig(hidden_size=256, intermediate_size=1024)
    
    # Create weights
    mlp_weights = {
        "mlp.fc1.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "mlp.fc1.bias": torch.randn(config.intermediate_size),
        "mlp.fc2.weight": torch.randn(config.hidden_size, config.intermediate_size),
        "mlp.fc2.bias": torch.randn(config.hidden_size),
    }
    
    # Create models
    mlp_torch = SigLIPMLPTorch(config, mlp_weights)
    mlp_ttnn = SigLIPMLPTTNN(config, mlp_weights, device)
    
    # Test input
    hidden = torch.randn(2, 32, config.hidden_size)
    
    # PyTorch forward
    print("Running PyTorch forward pass...")
    out_torch = mlp_torch.forward(hidden)
    print(f"  Output shape: {out_torch.shape}")
    
    # TTNN forward
    print("Running TTNN forward pass on device...")
    hidden_ttnn = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_ttnn_tensor = mlp_ttnn.forward(hidden_ttnn)
    out_ttnn = ttnn.to_torch(out_ttnn_tensor)
    print(f"  Output shape: {out_ttnn.shape}")
    
    # Compare
    passed, pcc = check_pcc(out_torch, out_ttnn, threshold=0.97, test_name="SigLIP MLP")
    
    if not passed:
        print(f"\n  Debugging info:")
        print(f"    Mean: {out_torch.mean():.6f} vs {out_ttnn.mean():.6f}")
        print(f"    Std: {out_torch.std():.6f} vs {out_ttnn.std():.6f}")
        print(f"    Mean abs diff: {(out_torch - out_ttnn).abs().mean():.6f}")
        print(f"    Max abs diff: {(out_torch - out_ttnn).abs().max():.6f}")
    
    return passed, pcc


def test_siglip_block_on_device(device):
    """Test SigLIP Block TTNN vs PyTorch on device."""
    print("\n" + "=" * 70)
    print("  Testing SigLIP Block on Device")
    print("=" * 70)
    
    from ttnn_siglip import (
        SigLIPConfig,
        SigLIPBlockTorch,
        SigLIPBlockTTNN,
    )
    
    config = SigLIPConfig(hidden_size=256, num_attention_heads=8, intermediate_size=1024)
    
    # Create weights
    block_weights = {
        "layer_norm1.weight": torch.randn(config.hidden_size),
        "layer_norm1.bias": torch.randn(config.hidden_size),
        "layer_norm2.weight": torch.randn(config.hidden_size),
        "layer_norm2.bias": torch.randn(config.hidden_size),
        "self_attn.q_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.k_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.v_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.out_proj.weight": torch.randn(config.hidden_size, config.hidden_size),
        "self_attn.q_proj.bias": torch.randn(config.hidden_size),
        "self_attn.k_proj.bias": torch.randn(config.hidden_size),
        "self_attn.v_proj.bias": torch.randn(config.hidden_size),
        "self_attn.out_proj.bias": torch.randn(config.hidden_size),
        "mlp.fc1.weight": torch.randn(config.intermediate_size, config.hidden_size),
        "mlp.fc1.bias": torch.randn(config.intermediate_size),
        "mlp.fc2.weight": torch.randn(config.hidden_size, config.intermediate_size),
        "mlp.fc2.bias": torch.randn(config.hidden_size),
    }
    
    # Create models
    block_torch = SigLIPBlockTorch(config, block_weights)
    block_ttnn = SigLIPBlockTTNN(config, block_weights, device)
    
    # Test input
    hidden = torch.randn(2, 32, config.hidden_size)
    
    # PyTorch forward
    print("Running PyTorch forward pass...")
    out_torch = block_torch.forward(hidden)
    print(f"  Output shape: {out_torch.shape}")
    
    # TTNN forward
    print("Running TTNN forward pass on device...")
    hidden_ttnn = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_ttnn_tensor = block_ttnn.forward(hidden_ttnn)
    out_ttnn = ttnn.to_torch(out_ttnn_tensor)
    print(f"  Output shape: {out_ttnn.shape}")
    
    # Compare
    passed, pcc = check_pcc(out_torch, out_ttnn, threshold=0.95, test_name="SigLIP Block")
    
    if not passed:
        print(f"\n  Debugging info:")
        print(f"    Mean: {out_torch.mean():.6f} vs {out_ttnn.mean():.6f}")
        print(f"    Std: {out_torch.std():.6f} vs {out_ttnn.std():.6f}")
        print(f"    Mean abs diff: {(out_torch - out_ttnn).abs().mean():.6f}")
        print(f"    Max abs diff: {(out_torch - out_ttnn).abs().max():.6f}")
    
    return passed, pcc


def test_gemma_attention_on_device(device):
    """Test Gemma Attention TTNN vs PyTorch on device."""
    print("\n" + "=" * 70)
    print("  Testing Gemma Attention on Device")
    print("=" * 70)
    
    from ttnn_gemma import (
        GemmaConfig,
        GemmaAttentionTorch,
        GemmaAttentionTTNN,
        precompute_freqs_cis_torch,
    )
    
    config = GemmaConfig(width=512, depth=4, mlp_dim=2048, num_heads=8, num_kv_heads=1)
    cos, sin = precompute_freqs_cis_torch(config.head_dim, 128)
    
    # Create weights
    attn_weights_torch = {
        "self_attn.q_proj.weight": torch.randn(config.num_heads * config.head_dim, config.width),
        "self_attn.k_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.v_proj.weight": torch.randn(config.num_kv_heads * config.head_dim, config.width),
        "self_attn.o_proj.weight": torch.randn(config.width, config.num_heads * config.head_dim),
    }
    
    # Convert weights to TTNN for TTNN model
    # Note: TTNN linear expects transposed weights compared to PyTorch
    attn_weights_ttnn = {}
    for key, weight in attn_weights_torch.items():
        attn_weights_ttnn[key] = ttnn.from_torch(
            weight.T,  # Transpose for TTNN
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    
    # Create models
    attn_torch = GemmaAttentionTorch(config, attn_weights_torch, layer_idx=0)
    attn_ttnn = GemmaAttentionTTNN(config, attn_weights_ttnn, layer_idx=0, device=device)
    
    # Test input
    hidden = torch.randn(2, 32, config.width)
    
    # PyTorch forward
    print("Running PyTorch forward pass...")
    out_torch, _ = attn_torch.forward(hidden, cos, sin)
    print(f"  Output shape: {out_torch.shape}")
    
    # TTNN forward
    print("Running TTNN forward pass on device...")
    hidden_ttnn = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_ttnn = ttnn.from_torch(
        cos,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_ttnn = ttnn.from_torch(
        sin,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_ttnn_tensor, _ = attn_ttnn.forward(hidden_ttnn, cos_ttnn, sin_ttnn)
    out_ttnn = ttnn.to_torch(out_ttnn_tensor)
    print(f"  Output shape: {out_ttnn.shape}")
    
    # Compare
    passed, pcc = check_pcc(out_torch, out_ttnn, threshold=0.95, test_name="Gemma Attention")
    
    if not passed:
        print(f"\n  Debugging info:")
        print(f"    Mean: {out_torch.mean():.6f} vs {out_ttnn.mean():.6f}")
        print(f"    Std: {out_torch.std():.6f} vs {out_ttnn.std():.6f}")
        print(f"    Mean abs diff: {(out_torch - out_ttnn).abs().mean():.6f}")
        print(f"    Max abs diff: {(out_torch - out_ttnn).abs().max():.6f}")
    
    return passed, pcc


def main():
    """Main test runner."""
    print("=" * 70)
    print("  TTNN PI0 Reference - On-Device Testing")
    print("=" * 70)
    
    # Open device
    print("\n🔌 Opening TTNN device...")
    try:
        device = ttnn.open_device(device_id=0)
        print(f"✅ Device opened: {device}")
    except Exception as e:
        print(f"❌ Failed to open device: {e}")
        return 1
    
    try:
        results = {}
        
        # Test SigLIP Attention
        passed, pcc = test_siglip_attention_on_device(device)
        results["SigLIP Attention"] = {"passed": passed, "pcc": pcc}
        
        # Test SigLIP MLP
        passed, pcc = test_siglip_mlp_on_device(device)
        results["SigLIP MLP"] = {"passed": passed, "pcc": pcc}
        
        # Test SigLIP Block
        passed, pcc = test_siglip_block_on_device(device)
        results["SigLIP Block"] = {"passed": passed, "pcc": pcc}
        
        # Test Gemma Attention
        passed, pcc = test_gemma_attention_on_device(device)
        results["Gemma Attention"] = {"passed": passed, "pcc": pcc}
        
        # Summary
        print("\n" + "=" * 70)
        print("  Test Summary")
        print("=" * 70)
        
        total = len(results)
        passed_count = sum(1 for r in results.values() if r["passed"])
        
        print(f"\nResults:")
        for name, result in results.items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"  {status} {name}: PCC = {result['pcc']:.6f}")
        
        print(f"\nTotal: {passed_count}/{total} tests passed")
        
        if passed_count == total:
            print("\n✅ ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n⚠️  {total - passed_count} test(s) failed")
            return 1
        
    finally:
        # Close device
        print("\n🔌 Closing device...")
        ttnn.close_device(device)
        print("✅ Device closed")


if __name__ == "__main__":
    sys.exit(main())

