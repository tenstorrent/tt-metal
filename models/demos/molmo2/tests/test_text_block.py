# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Molmo2 Text Model Block.

Validates a single decoder block against PyTorch reference.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc


def get_block_weights(model_id: str = "allenai/Molmo2-8B", layer_num: int = 0):
    """Load weights for a single text block from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    prefix = f"model.transformer.blocks.{layer_num}"
    keys = [
        # Attention norm
        f"{prefix}.attn_norm.weight",
        # Self-attention
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
        f"{prefix}.self_attn.att_proj.weight",
        f"{prefix}.self_attn.attn_out.weight",
        # FF norm
        f"{prefix}.ff_norm.weight",
        # MLP
        f"{prefix}.mlp.ff_proj.weight",
        f"{prefix}.mlp.ff_out.weight",
    ]
    return load_state_dict_from_safetensors(model_id, keys)


class ReferenceRMSNorm(nn.Module):
    """PyTorch reference RMSNorm."""

    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def rotate_half(x):
    """Rotates half the hidden dims (half-span pairing)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding (half-span). Matches ttnn.experimental.rotary_embedding."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ReferenceTextBlock(nn.Module):
    """PyTorch reference implementation of a text decoder block."""

    def __init__(
        self,
        hidden_dim=4096,
        intermediate_dim=12288,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        eps=1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim**-0.5

        # Norms
        self.attn_norm = ReferenceRMSNorm(hidden_dim, eps)
        self.ff_norm = ReferenceRMSNorm(hidden_dim, eps)

        # Attention
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # QK norm
        self.q_norm = ReferenceRMSNorm(head_dim, eps)
        self.k_norm = ReferenceRMSNorm(head_dim, eps)

        # MLP
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape

        # Attention block
        residual = x
        x = self.attn_norm(x)

        # QKV projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV for GQA
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.to(attn.device), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)

        # Reshape and output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_out = self.o_proj(attn_out)

        x = residual + attn_out

        # MLP block
        residual = x
        x = self.ff_norm(x)

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_out = self.down_proj(gate * up)

        x = residual + mlp_out

        return x


import pytest


@pytest.mark.parametrize("layer_num", [0, 17, 35], ids=["layer0", "layer17", "layer35"])
def test_text_block_parametrized(device, layer_num):
    """
    Test TextBlock against PyTorch reference for first, middle, and last layers.

    All layers must meet PCC >= 0.99 (CLAUDE.md requirement for individual blocks).
    """
    _run_text_block_test(device, layer_num)


def test_text_block(device):
    """
    Test TextBlock against PyTorch reference (layer 0 for backwards compatibility).
    """
    _run_text_block_test(device, layer_num=0)


def _run_text_block_test(device, layer_num: int):
    """Shared implementation for text block tests."""
    from models.demos.molmo2.tt.text_block import TextBlock

    model_id = "allenai/Molmo2-8B"
    hidden_dim = 4096
    intermediate_dim = 12288
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_len = 32

    # Load weights
    state_dict = get_block_weights(model_id, layer_num)
    prefix = f"model.transformer.blocks.{layer_num}"

    # Create reference model
    ref_model = ReferenceTextBlock(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # Load weights into reference model
    # Split fused QKV projection
    att_proj = state_dict[f"{prefix}.self_attn.att_proj.weight"]
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    ref_model.q_proj.weight.data = att_proj[:q_dim, :]
    ref_model.k_proj.weight.data = att_proj[q_dim : q_dim + kv_dim, :]
    ref_model.v_proj.weight.data = att_proj[q_dim + kv_dim :, :]

    ref_model.o_proj.weight.data = state_dict[f"{prefix}.self_attn.attn_out.weight"]

    # QK-norm weights
    ref_model.q_norm.weight.data = state_dict[f"{prefix}.self_attn.q_norm.weight"]
    ref_model.k_norm.weight.data = state_dict[f"{prefix}.self_attn.k_norm.weight"]

    # Norms
    ref_model.attn_norm.weight.data = state_dict[f"{prefix}.attn_norm.weight"]
    ref_model.ff_norm.weight.data = state_dict[f"{prefix}.ff_norm.weight"]

    # Split fused MLP projection
    ff_proj = state_dict[f"{prefix}.mlp.ff_proj.weight"]
    ref_model.gate_proj.weight.data = ff_proj[:intermediate_dim, :]
    ref_model.up_proj.weight.data = ff_proj[intermediate_dim:, :]

    ref_model.down_proj.weight.data = state_dict[f"{prefix}.mlp.ff_out.weight"]
    ref_model.eval()

    # Create random input
    torch.manual_seed(42)
    x_torch = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)

    # Compute RoPE for reference
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_ref = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin_ref = torch.sin(freqs).repeat_interleave(2, dim=-1)
    cos_ref = cos_ref.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin_ref = sin_ref.unsqueeze(0).unsqueeze(0)

    # Reference forward
    with torch.no_grad():
        ref_output = ref_model(x_torch, cos_ref, sin_ref)

    # Create TTNN block
    tt_block = TextBlock(
        mesh_device=device,
        state_dict=state_dict,
        layer_num=layer_num,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=ttnn.bfloat16,
    )

    # Convert input to TTNN
    x_ttnn = ttnn.from_torch(
        x_torch.unsqueeze(0),  # [1, 1, seq_len, hidden_dim]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert RoPE to TTNN (use same cos/sin as reference)
    cos_ttnn = ttnn.from_torch(
        cos_ref,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_ttnn = ttnn.from_torch(
        sin_ref,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create rotary setup for transformation matrices
    from models.demos.molmo2.tt.text_rotary_setup import TextRotarySetup

    rotary_setup = TextRotarySetup(
        mesh_device=device,
        head_dim=head_dim,
        max_seq_len=8192,
        rope_theta=1000000.0,
        batch_size=1,
        datatype=ttnn.bfloat16,
    )
    transformation_mats = rotary_setup.get_transformation_mats()

    # Prepare rot_mats as a list [cos, sin]
    rot_mats = [cos_ttnn, sin_ttnn]

    # TTNN forward
    tt_output, _ = tt_block(x_ttnn, rot_mats, transformation_mats, None, 0, None)

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0).squeeze(0)

    # Compare with PCC — CLAUDE.md requires >= 0.99 for all individual blocks
    passing, pcc_msg = comp_pcc(ref_output, tt_output_torch, pcc=0.99)
    print(f"TextBlock (layer={layer_num}) PCC: {pcc_msg}")

    assert passing, f"TextBlock failed PCC check: {pcc_msg}"


if __name__ == "__main__":
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        test_text_block(device)
        print("Test passed!")
    finally:
        ttnn.close_device(device)
