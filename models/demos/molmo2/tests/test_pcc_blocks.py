# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC comparison test for text model blocks.

Compares TTNN implementation against PyTorch reference layer by layer.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoTokenizer

import ttnn


def calculate_pcc(ref, out):
    """Calculate Pearson Correlation Coefficient."""
    if ref.shape != out.shape:
        return -1.0, f"Shape mismatch: {ref.shape} vs {out.shape}"
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()
    ref_mean = ref_flat.mean()
    out_mean = out_flat.mean()
    numerator = ((ref_flat - ref_mean) * (out_flat - out_mean)).sum()
    denominator = torch.sqrt(((ref_flat - ref_mean) ** 2).sum() * ((out_flat - out_mean) ** 2).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0, "zero std"
    pcc = numerator / denominator
    return pcc.item(), "ok"


class RefRMSNorm(nn.Module):
    """Reference RMSNorm implementation."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings."""
    # x: [batch, num_heads, seq_len, head_dim]
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rotated * sin


def test_pcc_blocks():
    """Test PCC for each text model block."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    # Config
    hidden_dim = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    intermediate_dim = 12288
    rms_norm_eps = 1e-5
    rope_theta = 1000000.0

    # Load
    tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    logger.info("Loaded weights")

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    logger.info(f"Input: {prompt} (seq_len={seq_len})")

    # Reference embedding
    wte = state_dict["model.transformer.wte.embedding"]
    ref_hidden = wte[input_ids[0]].unsqueeze(0)  # [1, seq_len, hidden_dim]
    logger.info(f"Ref hidden shape: {ref_hidden.shape}")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Test Block 0
        layer_num = 0
        prefix = f"model.transformer.blocks.{layer_num}"

        # Load weights for block 0
        attn_norm_weight = state_dict[f"{prefix}.attn_norm.weight"]
        ff_norm_weight = state_dict[f"{prefix}.ff_norm.weight"]

        # Attention weights
        att_proj = state_dict[f"{prefix}.self_attn.att_proj.weight"]
        attn_out = state_dict[f"{prefix}.self_attn.attn_out.weight"]
        q_norm = state_dict[f"{prefix}.self_attn.q_norm.weight"]
        k_norm = state_dict[f"{prefix}.self_attn.k_norm.weight"]

        # MLP weights
        ff_proj = state_dict[f"{prefix}.mlp.ff_proj.weight"]
        ff_out = state_dict[f"{prefix}.mlp.ff_out.weight"]

        logger.info("Loaded block 0 weights")

        # Step 1: Test attn_norm
        ref_attn_norm = RefRMSNorm(hidden_dim, rms_norm_eps)
        ref_attn_norm.weight.data = attn_norm_weight
        ref_attn_norm_out = ref_attn_norm(ref_hidden)

        # TTNN attn_norm
        from models.demos.molmo2.tt.text_rmsnorm import TextRMSNorm

        ttnn_attn_norm = TextRMSNorm(
            mesh_device=device,
            state_dict=state_dict,
            hidden_dim=hidden_dim,
            eps=rms_norm_eps,
            state_dict_prefix=f"{prefix}.attn_norm",
        )
        hidden_ttnn = ttnn.from_torch(
            ref_hidden.unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_attn_norm_out = ttnn_attn_norm(hidden_ttnn)
        ttnn_attn_norm_torch = ttnn.to_torch(ttnn_attn_norm_out).squeeze(0)

        pcc, status = calculate_pcc(ref_attn_norm_out, ttnn_attn_norm_torch)
        logger.info(f"attn_norm PCC: {pcc:.6f} ({status})")

        # Step 2: Test Q, K, V projections
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        wq = att_proj[:q_dim, :]
        wk = att_proj[q_dim : q_dim + kv_dim, :]
        wv = att_proj[q_dim + kv_dim :, :]

        ref_q = F.linear(ref_attn_norm_out, wq)
        ref_k = F.linear(ref_attn_norm_out, wk)
        ref_v = F.linear(ref_attn_norm_out, wv)

        logger.info(f"Ref Q shape: {ref_q.shape}, K: {ref_k.shape}, V: {ref_v.shape}")

        # Step 3: Reshape and QK-norm
        ref_q = ref_q.reshape(1, seq_len, num_heads, head_dim).transpose(1, 2)
        ref_k = ref_k.reshape(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        ref_v = ref_v.reshape(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # QK-norm
        ref_q_norm = RefRMSNorm(head_dim, 1e-5)
        ref_q_norm.weight.data = q_norm
        ref_k_norm = RefRMSNorm(head_dim, 1e-5)
        ref_k_norm.weight.data = k_norm

        ref_q = ref_q_norm(ref_q)
        ref_k = ref_k_norm(ref_k)

        # Step 4: RoPE
        positions = torch.arange(seq_len)
        freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = positions[:, None] * freqs[None, :]
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

        # Expand to full head_dim
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        ref_q = apply_rotary_emb(ref_q, cos, sin)
        ref_k = apply_rotary_emb(ref_k, cos, sin)

        # Step 5: GQA expansion
        ref_k_expanded = ref_k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        ref_v_expanded = ref_v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        # Step 6: Attention
        scale = head_dim**-0.5
        attn_weights = torch.matmul(ref_q, ref_k_expanded.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        attn_weights = attn_weights + causal_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        ref_attn_out = torch.matmul(attn_probs, ref_v_expanded)

        # Step 7: Output projection
        ref_attn_out = ref_attn_out.transpose(1, 2).reshape(1, seq_len, hidden_dim)
        ref_attn_out = F.linear(ref_attn_out, attn_out)

        # Step 8: Residual
        ref_after_attn = ref_hidden + ref_attn_out

        logger.info(
            f"Ref after attention: min={ref_after_attn.min():.4f}, mean={ref_after_attn.mean():.4f}, max={ref_after_attn.max():.4f}"
        )

        # Step 9: MLP
        ref_ff_norm = RefRMSNorm(hidden_dim, rms_norm_eps)
        ref_ff_norm.weight.data = ff_norm_weight
        ref_ff_norm_out = ref_ff_norm(ref_after_attn)

        # SwiGLU (HuggingFace order: first half is UP, second half is GATE)
        ref_ff_proj_out = F.linear(ref_ff_norm_out, ff_proj)
        up = ref_ff_proj_out[..., :intermediate_dim]
        gate = ref_ff_proj_out[..., intermediate_dim:]
        ref_swiglu = F.silu(gate) * up
        ref_mlp_out = F.linear(ref_swiglu, ff_out)

        ref_block_out = ref_after_attn + ref_mlp_out

        logger.info(
            f"Ref block 0 output: min={ref_block_out.min():.4f}, mean={ref_block_out.mean():.4f}, max={ref_block_out.max():.4f}"
        )

        # Now test TTNN block 0
        from models.demos.molmo2.tt.text_block import TextBlock
        from models.demos.molmo2.tt.text_rotary_setup import TextRotarySetup

        # Create rotary setup
        rotary_setup = TextRotarySetup(
            mesh_device=device,
            head_dim=head_dim,
            max_seq_len=8192,
            rope_theta=rope_theta,
            batch_size=1,
            datatype=ttnn.bfloat16,
        )
        transformation_mats = rotary_setup.get_transformation_mats()
        rot_mats = rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)

        ttnn_block = TextBlock(
            mesh_device=device,
            state_dict=state_dict,
            layer_num=0,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=8192,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            state_dict_prefix="model.transformer.blocks",
            dtype=ttnn.bfloat8_b,
        )

        # Run TTNN block
        ttnn_hidden = ttnn.from_torch(
            ref_hidden.unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_block_out, _ = ttnn_block(
            ttnn_hidden,
            rot_mats,
            transformation_mats,
            attn_mask=None,
            start_pos=0,
            kv_cache=None,
        )

        ttnn_block_torch = ttnn.to_torch(ttnn_block_out).squeeze(0)
        pcc, status = calculate_pcc(ref_block_out, ttnn_block_torch)
        logger.info(f"Block 0 PCC: {pcc:.6f} ({status})")

        logger.info(
            f"TTNN block 0 output: min={ttnn_block_torch.min():.4f}, mean={ttnn_block_torch.mean():.4f}, max={ttnn_block_torch.max():.4f}"
        )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_pcc_blocks()
