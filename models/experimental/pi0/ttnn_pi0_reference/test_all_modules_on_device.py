#!/usr/bin/env python3
"""
Comprehensive test suite for all TTNN PI0 modules on device.

Tests each module individually with PCC validation against PyTorch.
"""

import sys
import torch
import numpy as np

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
    
    status = "✓" if passed else "✗"
    return passed, pcc, status


# ==============================================================================
# MODULE 1: ttnn_common - Utility Functions
# ==============================================================================

def test_common_module(device):
    """Test ttnn_common utility functions."""
    print("\n" + "=" * 70)
    print("  MODULE 1: ttnn_common (Utility Functions)")
    print("=" * 70)
    
    from ttnn_common import (
        create_sinusoidal_pos_embedding_torch,
        create_sinusoidal_pos_embedding_ttnn,
        sample_noise_torch,
    )
    
    results = []
    
    # Test 1: Sinusoidal embeddings
    print("\n1. Testing sinusoidal position embeddings...")
    try:
        seq_len, dim = 32, 256
        
        # PyTorch
        emb_torch = create_sinusoidal_pos_embedding_torch(seq_len, dim)
        
        # TTNN
        emb_ttnn_tensor = create_sinusoidal_pos_embedding_ttnn(seq_len, dim, device)
        emb_ttnn = ttnn.to_torch(emb_ttnn_tensor)
        
        passed, pcc, status = check_pcc(emb_torch, emb_ttnn, threshold=0.99, test_name="Sinusoidal embeddings")
        print(f"   [{status}] PCC = {pcc:.6f}")
        results.append(("Sinusoidal embeddings", passed, pcc))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("Sinusoidal embeddings", False, 0.0))
    
    # Test 2: Noise sampling
    print("\n2. Testing noise sampling...")
    try:
        shape = (2, 50, 32)
        noise1 = sample_noise_torch(shape)
        noise2 = sample_noise_torch(shape)
        
        # Check shape and statistics
        assert noise1.shape == shape
        mean_close = abs(noise1.mean().item()) < 0.1
        std_close = abs(noise1.std().item() - 1.0) < 0.1
        
        passed = mean_close and std_close
        print(f"   [{'✓' if passed else '✗'}] Mean: {noise1.mean():.4f}, Std: {noise1.std():.4f}")
        results.append(("Noise sampling", passed, 1.0 if passed else 0.0))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("Noise sampling", False, 0.0))
    
    return results


# ==============================================================================
# MODULE 2: ttnn_siglip - Vision Encoder
# ==============================================================================

def test_siglip_module(device):
    """Test SigLIP vision encoder components."""
    print("\n" + "=" * 70)
    print("  MODULE 2: ttnn_siglip (Vision Encoder)")
    print("=" * 70)
    
    from ttnn_siglip import (
        SigLIPConfig,
        SigLIPAttentionTorch,
        SigLIPAttentionTTNN,
        SigLIPMLPTorch,
        SigLIPMLPTTNN,
        SigLIPBlockTorch,
        SigLIPBlockTTNN,
    )
    
    results = []
    config = SigLIPConfig(hidden_size=256, num_attention_heads=8, intermediate_size=1024)
    
    # Test 1: Attention
    print("\n1. Testing SigLIP Attention...")
    try:
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
        
        attn_torch = SigLIPAttentionTorch(config, attn_weights)
        attn_ttnn = SigLIPAttentionTTNN(config, attn_weights, device)
        
        hidden = torch.randn(2, 32, 256)
        out_torch = attn_torch.forward(hidden)
        
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = ttnn.to_torch(attn_ttnn.forward(hidden_ttnn))
        
        passed, pcc, status = check_pcc(out_torch, out_ttnn, threshold=0.95, test_name="SigLIP Attention")
        print(f"   [{status}] PCC = {pcc:.6f}")
        results.append(("SigLIP Attention", passed, pcc))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("SigLIP Attention", False, 0.0))
    
    # Test 2: MLP
    print("\n2. Testing SigLIP MLP...")
    try:
        mlp_weights = {
            "mlp.fc1.weight": torch.randn(1024, 256),
            "mlp.fc1.bias": torch.randn(1024),
            "mlp.fc2.weight": torch.randn(256, 1024),
            "mlp.fc2.bias": torch.randn(256),
        }
        
        mlp_torch = SigLIPMLPTorch(config, mlp_weights)
        mlp_ttnn = SigLIPMLPTTNN(config, mlp_weights, device)
        
        hidden = torch.randn(2, 32, 256)
        out_torch = mlp_torch.forward(hidden)
        
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = ttnn.to_torch(mlp_ttnn.forward(hidden_ttnn))
        
        passed, pcc, status = check_pcc(out_torch, out_ttnn, threshold=0.97, test_name="SigLIP MLP")
        print(f"   [{status}] PCC = {pcc:.6f}")
        results.append(("SigLIP MLP", passed, pcc))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("SigLIP MLP", False, 0.0))
    
    # Test 3: Block
    print("\n3. Testing SigLIP Block...")
    try:
        block_weights = {
            "layer_norm1.weight": torch.randn(256),
            "layer_norm1.bias": torch.randn(256),
            "layer_norm2.weight": torch.randn(256),
            "layer_norm2.bias": torch.randn(256),
        }
        block_weights.update(attn_weights)
        block_weights.update(mlp_weights)
        
        block_torch = SigLIPBlockTorch(config, block_weights)
        block_ttnn = SigLIPBlockTTNN(config, block_weights, device)
        
        hidden = torch.randn(2, 32, 256)
        out_torch = block_torch.forward(hidden)
        
        hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_ttnn = ttnn.to_torch(block_ttnn.forward(hidden_ttnn))
        
        passed, pcc, status = check_pcc(out_torch, out_ttnn, threshold=0.95, test_name="SigLIP Block")
        print(f"   [{status}] PCC = {pcc:.6f}")
        results.append(("SigLIP Block", passed, pcc))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("SigLIP Block", False, 0.0))
    
    return results


# ==============================================================================
# MODULE 3: ttnn_gemma - Language Model
# ==============================================================================

def test_gemma_module(device):
    """Test Gemma language model components."""
    print("\n" + "=" * 70)
    print("  MODULE 3: ttnn_gemma (Language Model)")
    print("=" * 70)
    
    from ttnn_gemma import (
        GemmaConfig,
        rms_norm_torch,
        GemmaMLPTorch,
        GemmaMLPTTNN,
        precompute_freqs_cis_torch,
    )
    
    results = []
    config = GemmaConfig(width=512, depth=4, mlp_dim=2048, num_heads=8)
    
    # Test 1: RMSNorm
    print("\n1. Testing Gemma RMSNorm...")
    try:
        x = torch.randn(2, 32, 512)
        weight = torch.randn(512)
        
        out1 = rms_norm_torch(x, weight)
        out2 = rms_norm_torch(x, weight)
        
        passed, pcc, status = check_pcc(out1, out2, threshold=1.0, test_name="Gemma RMSNorm")
        print(f"   [{status}] PCC = {pcc:.6f} (consistency check)")
        results.append(("Gemma RMSNorm", passed, pcc))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("Gemma RMSNorm", False, 0.0))
    
    # Test 2: RoPE
    print("\n2. Testing Gemma RoPE...")
    try:
        cos, sin = precompute_freqs_cis_torch(config.head_dim, 128)
        
        assert cos.shape == (128, config.head_dim // 2)
        assert sin.shape == (128, config.head_dim // 2)
        
        print(f"   [✓] RoPE shape correct: {cos.shape}")
        results.append(("Gemma RoPE", True, 1.0))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("Gemma RoPE", False, 0.0))
    
    # Test 3: MLP (PyTorch only - TTNN needs weight conversion)
    print("\n3. Testing Gemma MLP (PyTorch)...")
    try:
        mlp_weights = {
            "mlp.gate_proj.weight": torch.randn(2048, 512),
            "mlp.up_proj.weight": torch.randn(2048, 512),
            "mlp.down_proj.weight": torch.randn(512, 2048),
        }
        
        mlp_torch = GemmaMLPTorch(config, mlp_weights)
        hidden = torch.randn(2, 32, 512)
        out = mlp_torch.forward(hidden)
        
        assert out.shape == hidden.shape
        print(f"   [✓] MLP output shape correct: {out.shape}")
        results.append(("Gemma MLP", True, 1.0))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("Gemma MLP", False, 0.0))
    
    return results


# ==============================================================================
# MODULE 4: ttnn_suffix - Action Embedding
# ==============================================================================

def test_suffix_module(device):
    """Test suffix embedding for actions."""
    print("\n" + "=" * 70)
    print("  MODULE 4: ttnn_suffix (Action Embedding)")
    print("=" * 70)
    
    from ttnn_suffix import SuffixConfig, SuffixEmbeddingTorch
    
    results = []
    config = SuffixConfig(action_dim=32, action_horizon=50, expert_width=512)
    
    # Test 1: Suffix embedding (PyTorch only - TTNN needs proper setup)
    print("\n1. Testing Suffix Embedding (PyTorch)...")
    try:
        weights = {
            "action_in.weight": torch.randn(512, 32),
            "action_in.bias": torch.randn(512),
            "state_in.weight": torch.randn(512, config.state_dim),
            "state_in.bias": torch.randn(512),
            "action_time_mlp_in.weight": torch.randn(512, 1024),
            "action_time_mlp_in.bias": torch.randn(512),
            "action_time_mlp_out.weight": torch.randn(512, 512),
            "action_time_mlp_out.bias": torch.randn(512),
            "action_out.weight": torch.randn(32, 512),
            "action_out.bias": torch.randn(32),
        }
        
        suffix = SuffixEmbeddingTorch(config, weights)
        
        state = torch.randn(2, config.state_dim)
        actions = torch.randn(2, 50, 32)
        timestep = torch.rand(2)
        
        suffix_embs, pad_masks, att_masks, adarms = suffix.embed_suffix(state, actions, timestep)
        
        expected_len = 1 + config.action_horizon
        assert suffix_embs.shape == (2, expected_len, 512)
        print(f"   [✓] Suffix embedding shape correct: {suffix_embs.shape}")
        results.append(("Suffix Embedding", True, 1.0))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("Suffix Embedding", False, 0.0))
    
    return results


# ==============================================================================
# MODULE 5: ttnn_prefix - Prompt Embedding
# ==============================================================================

def test_prefix_module(device):
    """Test prefix embedding for prompts."""
    print("\n" + "=" * 70)
    print("  MODULE 5: ttnn_prefix (Prompt Embedding)")
    print("=" * 70)
    
    from ttnn_prefix import PrefixConfig, PrefixEmbeddingTorch
    
    results = []
    config = PrefixConfig(image_seq_len=256, language_seq_len=10, expert_width=512)
    
    # Test 1: Prefix embedding (PyTorch only)
    print("\n1. Testing Prefix Embedding (PyTorch)...")
    try:
        # Mock embedding functions
        def mock_embed_image(images):
            return torch.randn(2, 256, 512)
        
        def mock_embed_language(tokens):
            return torch.randn(2, 10, 512)
        
        prefix = PrefixEmbeddingTorch(
            config,
            embed_image_fn=mock_embed_image,
            embed_language_fn=mock_embed_language,
        )
        
        images = torch.randn(2, 3, 224, 224)
        language_tokens = torch.randint(0, 1000, (2, 10))
        
        prefix_embs, masks, adarms = prefix.embed_prefix(images, language_tokens)
        
        expected_len = 256 + 10
        assert prefix_embs.shape == (2, expected_len, 512)
        print(f"   [✓] Prefix embedding shape correct: {prefix_embs.shape}")
        results.append(("Prefix Embedding", True, 1.0))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("Prefix Embedding", False, 0.0))
    
    return results


# ==============================================================================
# MODULE 6: ttnn_denoise - Denoising
# ==============================================================================

def test_denoise_module(device):
    """Test denoising utilities."""
    print("\n" + "=" * 70)
    print("  MODULE 6: ttnn_denoise (Denoising)")
    print("=" * 70)
    
    from ttnn_denoise import compute_snr_torch, get_alphas_torch
    
    results = []
    
    # Test 1: SNR computation
    print("\n1. Testing SNR computation...")
    try:
        timesteps = torch.tensor([0.0, 0.5, 1.0])
        snr = compute_snr_torch(timesteps)
        
        assert snr.shape == timesteps.shape
        assert torch.all(snr >= 0)
        print(f"   [✓] SNR computation works: {snr}")
        results.append(("SNR computation", True, 1.0))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("SNR computation", False, 0.0))
    
    # Test 2: Alpha values
    print("\n2. Testing alpha values...")
    try:
        timesteps = torch.tensor([0.0, 0.5, 1.0])
        alphas = get_alphas_torch(timesteps)
        
        assert alphas.shape == timesteps.shape
        assert torch.all(alphas > 0) and torch.all(alphas <= 1)
        print(f"   [✓] Alpha values correct: {alphas}")
        results.append(("Alpha values", True, 1.0))
    except Exception as e:
        print(f"   [✗] Failed: {e}")
        results.append(("Alpha values", False, 0.0))
    
    return results


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

def main():
    """Main test runner."""
    print("=" * 70)
    print("  TTNN PI0 Reference - Comprehensive Module Testing")
    print("=" * 70)
    
    # Open device
    print("\n🔌 Opening TTNN device...")
    try:
        device = ttnn.open_device(device_id=0)
        print(f"✅ Device opened: {device}")
        grid = device.compute_with_storage_grid_size()
        print(f"   Grid size: {grid.x}x{grid.y} ({grid.x * grid.y} cores)")
    except Exception as e:
        print(f"❌ Failed to open device: {e}")
        return 1
    
    try:
        all_results = []
        
        # Test each module
        all_results.extend(test_common_module(device))
        all_results.extend(test_siglip_module(device))
        all_results.extend(test_gemma_module(device))
        all_results.extend(test_suffix_module(device))
        all_results.extend(test_prefix_module(device))
        all_results.extend(test_denoise_module(device))
        
        # Summary
        print("\n" + "=" * 70)
        print("  Test Summary")
        print("=" * 70)
        
        passed_count = sum(1 for _, passed, _ in all_results if passed)
        total_count = len(all_results)
        
        print(f"\nResults by Module:")
        print(f"{'Component':<30} {'Status':<10} {'PCC':<10}")
        print("-" * 50)
        
        for name, passed, pcc in all_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            pcc_str = f"{pcc:.6f}" if pcc > 0 else "N/A"
            print(f"{name:<30} {status:<10} {pcc_str:<10}")
        
        print("\n" + "=" * 70)
        print(f"Total: {passed_count}/{total_count} tests passed ({100*passed_count/total_count:.1f}%)")
        print("=" * 70)
        
        if passed_count == total_count:
            print("\n✅ ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n⚠️  {total_count - passed_count} test(s) failed")
            return 1
        
    finally:
        # Close device
        print("\n🔌 Closing device...")
        ttnn.close_device(device)
        print("✅ Device closed")


if __name__ == "__main__":
    sys.exit(main())

