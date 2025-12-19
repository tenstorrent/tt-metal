#!/usr/bin/env python3
"""
Simple test runner for ttnn_pi0_reference.

This script:
1. Checks if TTNN is available
2. Runs basic sanity tests
3. Runs PCC tests if requested
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_environment():
    """Check if TTNN is available."""
    print("=" * 70)
    print("  Environment Check")
    print("=" * 70)
    
    # Check TTNN
    try:
        import ttnn
        print("✅ TTNN is available")
        print(f"   Version: {getattr(ttnn, '__version__', 'unknown')}")
        
        # Try to open device
        try:
            device = ttnn.open_device(device_id=0)
            print("✅ TTNN device opened successfully")
            ttnn.close_device(device)
            return True, True
        except Exception as e:
            print(f"⚠️  TTNN device not available: {e}")
            print("   Tests will run in CPU-only mode")
            return True, False
            
    except ImportError as e:
        print(f"❌ TTNN not available: {e}")
        print("   Only PyTorch tests will run")
        return False, False


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 70)
    print("  Import Tests")
    print("=" * 70)
    
    modules = [
        "ttnn_common",
        "ttnn_attention",
        "ttnn_gemma",
        "ttnn_siglip",
        "ttnn_suffix",
        "ttnn_prefix",
        "ttnn_paligemma",
        "ttnn_denoise",
        "ttnn_pi0",
        "weight_loader",
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n⚠️  {len(failed)} module(s) failed to import")
        return False
    else:
        print(f"\n✅ All {len(modules)} modules imported successfully")
        return True


def test_basic_functionality():
    """Test basic functionality without TTNN device."""
    print("\n" + "=" * 70)
    print("  Basic Functionality Tests")
    print("=" * 70)
    
    import torch
    from ttnn_siglip import SigLIPConfig, SigLIPAttentionTorch, SigLIPMLPTorch
    from ttnn_gemma import GemmaConfig, rms_norm_torch
    from ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
    
    try:
        # Test SigLIP config
        config = SigLIPConfig()
        print(f"✅ SigLIPConfig created: {config.hidden_size}d, {config.num_hidden_layers} layers")
        
        # Test Gemma config
        gemma_config = GemmaConfig.gemma_2b()
        print(f"✅ GemmaConfig created: {gemma_config.width}d, {gemma_config.depth} layers")
        
        # Test RMSNorm
        x = torch.randn(2, 10, 256)
        weight = torch.randn(256)
        normed = rms_norm_torch(x, weight)
        print(f"✅ RMSNorm works: input {x.shape} -> output {normed.shape}")
        
        # Test SigLIP Attention (Torch)
        attn_weights = {
            "self_attn.q_proj.weight": torch.randn(1152, 1152),
            "self_attn.k_proj.weight": torch.randn(1152, 1152),
            "self_attn.v_proj.weight": torch.randn(1152, 1152),
            "self_attn.out_proj.weight": torch.randn(1152, 1152),
            "self_attn.q_proj.bias": torch.randn(1152),
            "self_attn.k_proj.bias": torch.randn(1152),
            "self_attn.v_proj.bias": torch.randn(1152),
            "self_attn.out_proj.bias": torch.randn(1152),
        }
        attn = SigLIPAttentionTorch(config, attn_weights)
        hidden = torch.randn(2, 256, 1152)
        attn_out = attn.forward(hidden)
        print(f"✅ SigLIPAttention works: {hidden.shape} -> {attn_out.shape}")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ttnn_functionality(device_available=False):
    """Test TTNN functionality if available."""
    if not device_available:
        print("\n⚠️  Skipping TTNN functionality tests (no device)")
        return True
    
    print("\n" + "=" * 70)
    print("  TTNN Functionality Tests")
    print("=" * 70)
    
    import torch
    import ttnn
    from ttnn_siglip import SigLIPConfig, SigLIPAttentionTTNN
    
    try:
        device = ttnn.open_device(device_id=0)
        
        # Test SigLIP Attention TTNN
        config = SigLIPConfig(hidden_size=256, num_attention_heads=8)
        attn_weights = {
            "self_attn.q_proj.weight": torch.randn(256, 256),
            "self_attn.k_proj.weight": torch.randn(256, 256),
            "self_attn.v_proj.weight": torch.randn(256, 256),
            "self_attn.out_proj.weight": torch.randn(256, 256),
            "self_attn.q_proj.bias": torch.randn(256),
            "self_attn.k_proj.bias": torch.randn(256),
            "self_attn.v_proj.bias": torch.randn(256),
            "self_attn.out_proj.bias": torch.randn(256),
        }
        
        attn_ttnn = SigLIPAttentionTTNN(config, attn_weights, device)
        
        # Create input and convert to TTNN
        hidden = torch.randn(2, 16, 256)
        hidden_ttnn = ttnn.from_torch(
            hidden,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        
        # Forward pass
        output_ttnn = attn_ttnn.forward(hidden_ttnn)
        output = ttnn.to_torch(output_ttnn)
        
        print(f"✅ SigLIPAttentionTTNN works: {hidden.shape} -> {output.shape}")
        
        ttnn.close_device(device)
        print("\n✅ TTNN functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ TTNN functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            ttnn.close_device(device)
        except:
            pass
        return False


def run_pcc_tests(module=None):
    """Run PCC tests."""
    print("\n" + "=" * 70)
    print("  PCC Tests")
    print("=" * 70)
    
    os.chdir(Path(__file__).parent)
    
    if module:
        cmd = f"python tests/pcc/run_all_pcc.py --module {module}"
    else:
        cmd = "python tests/pcc/run_all_pcc.py"
    
    print(f"Running: {cmd}")
    print()
    
    result = os.system(cmd)
    return result == 0


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ttnn_pi0_reference")
    parser.add_argument("--pcc", action="store_true", help="Run PCC tests")
    parser.add_argument("--module", type=str, help="Run PCC tests for specific module")
    parser.add_argument("--full", action="store_true", help="Run all tests including PCC")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  TTNN PI0 Reference - Test Runner")
    print("=" * 70)
    
    # Check environment
    ttnn_available, device_available = check_environment()
    
    # Import tests
    if not test_imports():
        print("\n❌ Import tests failed. Fix imports before running other tests.")
        return 1
    
    # Basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed.")
        return 1
    
    # TTNN functionality
    if ttnn_available:
        if not test_ttnn_functionality(device_available):
            print("\n❌ TTNN functionality tests failed.")
            return 1
    
    # PCC tests if requested
    if args.pcc or args.full:
        if not run_pcc_tests(args.module):
            print("\n❌ PCC tests failed.")
            return 1
    else:
        print("\n" + "=" * 70)
        print("  Skipping PCC Tests")
        print("=" * 70)
        print("  Run with --pcc to execute PCC tests")
        print("  Run with --full to execute all tests including PCC")
    
    print("\n" + "=" * 70)
    print("  ✅ All Tests Passed!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

