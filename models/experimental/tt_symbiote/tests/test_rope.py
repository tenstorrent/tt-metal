# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests Rotary Position Embedding (RoPE) with TTNN acceleration."""

import pytest
import torch

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.rope import TorchRotaryPositionEmbedding, TTNNRotaryPositionEmbedding
from models.experimental.tt_symbiote.utils.device_management import set_device


def create_rope_inputs(batch_size=1, seq_len=128, num_heads=12, num_heads2=8, head_dim=64):
    """Create test inputs for RoPE."""
    q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16)
    k = torch.randn((batch_size, num_heads2, seq_len, head_dim), dtype=torch.bfloat16)

    # Create cos/sin embeddings for RoPE in the format expected by ttnn.experimental.rotary_embedding_llama
    # This matches the format from gather_cos_sin in llama_common.py
    rotary_dim = head_dim
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = position * inv_freq  # [seq_len, head_dim//2]

    # Stack and flatten to duplicate values: [f0, f0, f1, f1, ...] instead of [f0, f1, ..., f0, f1, ...]
    # This matches the format from: torch.stack([freqs, freqs], dim=-1).flatten(-2)
    cos_half = freqs.cos()
    sin_half = freqs.sin()
    cos = torch.stack([cos_half, cos_half], dim=-1).flatten(-2).to(torch.bfloat16)  # [seq_len, head_dim]
    sin = torch.stack([sin_half, sin_half], dim=-1).flatten(-2).to(torch.bfloat16)  # [seq_len, head_dim]

    # Reshape cos/sin to [batch, seq_len, head_dim] for proper broadcasting
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    return q, k, cos, sin


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_rope_short_sequence(device):
    """Test RoPE with short sequence (uses fused operation)."""
    batch_size, seq_len, num_heads, num_heads2, head_dim = 1, 115, 96, 8, 128

    q, k, cos, sin = create_rope_inputs(batch_size, seq_len, num_heads, num_heads2, head_dim)
    assert q.shape == (batch_size, num_heads, seq_len, head_dim)
    assert cos.shape == (batch_size, seq_len, head_dim)
    assert sin.shape == (batch_size, seq_len, head_dim)

    # Test PyTorch implementation
    torch_model = TorchRotaryPositionEmbedding()
    torch_model.eval()
    torch.set_grad_enabled(False)

    q_torch = TorchTTNNTensor(q)
    k_torch = TorchTTNNTensor(k)
    cos_torch = TorchTTNNTensor(cos)
    sin_torch = TorchTTNNTensor(sin)

    q_out_torch, k_out_torch = torch_model(q_torch, k_torch, cos_torch, sin_torch)

    # Test TTNN implementation
    ttnn_model = TTNNRotaryPositionEmbedding.from_torch(torch_model)
    set_device(ttnn_model, device)
    ttnn_model.preprocess_weights()
    ttnn_model.move_weights_to_device()

    q_out_ttnn, k_out_ttnn = ttnn_model(q_torch, k_torch, cos_torch, sin_torch)

    compare_fn_outputs(q_out_torch, q_out_ttnn, "RoPE Query (short sequence)")
    compare_fn_outputs(k_out_torch, k_out_ttnn, "RoPE Key (short sequence)")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_rope_glm_dimensions(device):
    """Test RoPE with GLM-style dimensions where cos/sin head_dim doesn't match q/k head_dim."""
    batch_size, seq_len, num_heads, num_heads2 = 1, 115, 96, 8
    q_head_dim = 128  # Q/K head dimension
    rope_head_dim = 64  # RoPE cos/sin dimension (partial rotary)

    # Create inputs matching GLM's actual dimensions
    q = torch.randn((batch_size, num_heads, seq_len, q_head_dim), dtype=torch.bfloat16)
    k = torch.randn((batch_size, num_heads2, seq_len, q_head_dim), dtype=torch.bfloat16)

    # Create cos/sin embeddings with only rope_head_dim (partial rotary)
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim))
    freqs = position * inv_freq  # [seq_len, rope_head_dim//2]

    cos_half = freqs.cos()
    sin_half = freqs.sin()
    cos = torch.stack([cos_half, cos_half], dim=-1).flatten(-2).to(torch.bfloat16)  # [seq_len, rope_head_dim]
    sin = torch.stack([sin_half, sin_half], dim=-1).flatten(-2).to(torch.bfloat16)  # [seq_len, rope_head_dim]

    # Reshape to [batch, seq_len, rope_head_dim]
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    assert q.shape == (batch_size, num_heads, seq_len, q_head_dim)
    assert k.shape == (batch_size, num_heads2, seq_len, q_head_dim)
    assert cos.shape == (batch_size, seq_len, rope_head_dim)
    assert sin.shape == (batch_size, seq_len, rope_head_dim)

    # Test PyTorch implementation
    torch_model = TorchRotaryPositionEmbedding()
    torch_model.eval()
    torch.set_grad_enabled(False)

    q_torch = TorchTTNNTensor(q)
    k_torch = TorchTTNNTensor(k)
    cos_torch = TorchTTNNTensor(cos)
    sin_torch = TorchTTNNTensor(sin)

    q_out_torch, k_out_torch = torch_model(q_torch, k_torch, cos_torch, sin_torch)

    # Test TTNN implementation
    ttnn_model = TTNNRotaryPositionEmbedding.from_torch(torch_model)
    set_device(ttnn_model, device)
    ttnn_model.preprocess_weights()
    ttnn_model.move_weights_to_device()

    q_out_ttnn, k_out_ttnn = ttnn_model(q_torch, k_torch, cos_torch, sin_torch)

    compare_fn_outputs(q_out_torch, q_out_ttnn, "RoPE Query (GLM dimensions)")
    compare_fn_outputs(k_out_torch, k_out_ttnn, "RoPE Key (GLM dimensions)")
