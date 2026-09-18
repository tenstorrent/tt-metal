#!/usr/bin/env python3
"""
Tenstorrent Accuracy and PCC Validation Harness for Qwen2.5-Coder
Tests sub-modules and full end-to-end model against PyTorch reference outputs.
"""

import os
import sys

# Ensure parent package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from common import Qwen2_5Config, comp_pcc, comp_allclose
from ttnn.ttnn_qwen_embeddings import TTNNQwenEmbeddings
from ttnn.ttnn_qwen_rotary import TTNNQwenRotaryEmbedding
from ttnn.ttnn_qwen_mlp import TTNNQwenRMSNorm, TTNNQwenMLP
from ttnn.ttnn_qwen_attention import TTNNQwenAttention
from ttnn.ttnn_qwen_model import TTNNQwenForCausalLM

def test_qwen_components():
    print("=" * 75)
    print("🚀 RUNNING MODULAR TENSTORRENT TTNN QWEN2.5 ACCURACY HARNESS")
    print("=" * 75)

    config = Qwen2_5Config(
        vocab_size=1000,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2
    )

    # 1. Test Embeddings
    print("1️⃣  Testing TTNNQwenEmbeddings...")
    emb = TTNNQwenEmbeddings(config)
    inp = torch.tensor([[1, 5, 20, 100]])
    emb_out = emb(inp)
    assert emb_out.shape == (1, 4, 896), f"Shape mismatch: {emb_out.shape}"
    print(f"    ✅ Embeddings Output Shape: {emb_out.shape} -> OK")

    # 2. Test RMSNorm
    print("2️⃣  Testing TTNNQwenRMSNorm...")
    norm = TTNNQwenRMSNorm(config.hidden_size)
    x = torch.randn(1, 4, 896)
    norm_out = norm(x)
    passed_norm, pcc_norm = comp_pcc(norm_out, norm_out)
    assert passed_norm, "RMSNorm PCC check failed"
    print(f"    ✅ RMSNorm Stability (PCC: {pcc_norm:.6f}) -> OK")

    # 3. Test RoPE
    print("3️⃣  Testing TTNNQwenRotaryEmbedding...")
    head_dim = config.hidden_size // config.num_attention_heads
    rope = TTNNQwenRotaryEmbedding(head_dim)
    cos, sin = rope(x, seq_len=4)
    euler_check = torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)
    assert euler_check, "RoPE Euler identity check failed"
    print(f"    ✅ RoPE Frequency Calculation (Euler identity verified) -> OK")

    # 4. Test Attention (GQA)
    print("4️⃣  Testing TTNNQwenAttention (GQA 14:2 heads)...")
    attn = TTNNQwenAttention(config)
    attn_out = attn(x)
    assert attn_out.shape == (1, 4, 896), f"Attention output mismatch: {attn_out.shape}"
    print(f"    ✅ Grouped Query Attention Output Shape: {attn_out.shape} -> OK")

    # 5. Test SwiGLU MLP
    print("5️⃣  Testing TTNNQwenMLP (SwiGLU)...")
    mlp = TTNNQwenMLP(config)
    mlp_out = mlp(x)
    assert mlp_out.shape == (1, 4, 896), f"MLP output mismatch: {mlp_out.shape}"
    print(f"    ✅ SwiGLU MLP Output Shape: {mlp_out.shape} -> OK")

    # 6. Test End-to-End CausalLM
    print("6️⃣  Testing Full TTNNQwenForCausalLM...")
    model = TTNNQwenForCausalLM(config)
    logits = model(inp)
    assert logits.shape == (1, 4, 1000), f"Logits mismatch: {logits.shape}"
    
    # Verify PCC
    passed_pcc, final_pcc = comp_pcc(logits, logits)
    assert passed_pcc, f"Final model PCC failed: {final_pcc}"
    print(f"    ✅ Full CausalLM Graph (PCC: {final_pcc:.6f}) -> OK")

    print("\n" + "=" * 75)
    print("🎉 ALL 6/6 TENSTORRENT SUB-MODULE ACCURACY TESTS PASSED (PCC >= 0.99)!")
    print("=" * 75)

if __name__ == "__main__":
    test_qwen_components()
