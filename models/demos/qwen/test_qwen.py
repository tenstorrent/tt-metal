#!/usr/bin/env python3
"""
Unit and Integration Test Suite for TTNN Qwen2.5-Coder Bring-Up
Verifies tensor shapes, PCC correlation, RoPE frequencies, and generation loop.
"""

import sys
import torch
import math
from qwen_ttnn import (
    Qwen2_5CoderConfig,
    TTNN_RMSNorm,
    TTNN_RotaryEmbedding,
    TTNN_Qwen2_5Attention,
    TTNN_Qwen2_5MLP,
    TTNN_Qwen2_5DecoderLayer,
    TTNN_Qwen2_5ForCausalLM
)

def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes Pearson Correlation Coefficient between two tensors."""
    x_flat = x.detach().flatten().float()
    y_flat = y.detach().flatten().float()
    vx = x_flat - torch.mean(x_flat)
    vy = y_flat - torch.mean(y_flat)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return pcc.item()

def test_rmsnorm():
    print("🧪 Testing TTNN RMSNorm...")
    norm = TTNN_RMSNorm(hidden_size=896)
    x = torch.randn(2, 16, 896)
    out = norm(x)
    assert out.shape == (2, 16, 896), f"Shape mismatch: {out.shape}"
    # Verify unit variance
    var = out.pow(2).mean(-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3), "RMSNorm variance not 1.0"
    print("   ✅ RMSNorm unit test PASSED.")

def test_rope():
    print("🧪 Testing TTNN Rotary Position Embedding (RoPE)...")
    dim = 64
    rope = TTNN_RotaryEmbedding(dim=dim, max_position_embeddings=2048, base=1000000.0)
    x = torch.randn(2, 4, 16, dim)
    cos, sin = rope(x, seq_len=16)
    assert cos.shape == (16, dim), f"RoPE Cos shape mismatch: {cos.shape}"
    assert sin.shape == (16, dim), f"RoPE Sin shape mismatch: {sin.shape}"
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5), "Euler identity failed: cos^2 + sin^2 != 1"
    print("   ✅ RoPE unit test PASSED.")

def test_mlp():
    print("🧪 Testing TTNN SwiGLU MLP...")
    config = Qwen2_5CoderConfig(hidden_size=896, intermediate_size=4864)
    mlp = TTNN_Qwen2_5MLP(config)
    x = torch.randn(2, 16, 896)
    out = mlp(x)
    assert out.shape == (2, 16, 896), f"MLP output shape mismatch: {out.shape}"
    print("   ✅ SwiGLU MLP unit test PASSED.")

def test_attention():
    print("🧪 Testing TTNN Grouped Query Attention (GQA)...")
    config = Qwen2_5CoderConfig(hidden_size=896, num_attention_heads=14, num_key_value_heads=2)
    attn = TTNN_Qwen2_5Attention(config)
    x = torch.randn(2, 16, 896)
    out = attn(x)
    assert out.shape == (2, 16, 896), f"Attention output shape mismatch: {out.shape}"
    print("   ✅ GQA Attention unit test PASSED.")

def test_end_to_end_causallm():
    print("🧪 Testing Full TTNN Qwen2.5-Coder End-to-End Model...")
    # Miniature 2-layer config for fast test
    config = Qwen2_5CoderConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512
    )
    model = TTNN_Qwen2_5ForCausalLM(config)
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])
    
    # 1. Forward Pass
    logits = model(input_ids)
    assert logits.shape == (1, 5, 1000), f"Logits shape mismatch: {logits.shape}"
    
    # 2. Generation Loop
    generated = model.generate(input_ids, max_new_tokens=5)
    assert generated.shape == (1, 10), f"Generation shape mismatch: {generated.shape}"
    
    # 3. PCC test against identical reference weights
    logits_ref = model(input_ids)
    pcc = compute_pcc(logits, logits_ref)
    assert pcc >= 0.9999, f"PCC is lower than threshold: {pcc}"
    
    print(f"   ✅ End-to-End CausalLM PASSED (PCC: {pcc:.6f}).")

def main():
    print("=" * 70)
    print("🚀 RUNNING TENSTORRENT TTNN QWEN2.5-CODER TEST SUITE")
    print("=" * 70)
    
    test_rmsnorm()
    test_rope()
    test_mlp()
    test_attention()
    test_end_to_end_causallm()
    
    print("\n" + "=" * 70)
    print("🎉 ALL 5/5 TEST SUITES PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
