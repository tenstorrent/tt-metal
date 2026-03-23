# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Full PCC comparison test for all text model layers.

Compares TTNN text path with ``bfloat8_b`` **weights** (``TextBlock``, ``TextRMSNorm`` /
``ln_f``, ``lm_head``) against a CPU reference. RoPE cos/sin tables stay ``bfloat16`` because
``TextRotarySetup`` uses ``ROW_MAJOR`` for decode embedding tables, and TTNN requires
``TILE`` layout for ``bfloat8_b``. Activations stay ``bfloat16`` on device. The reference runs
under ``autocast(bfloat16)`` so precision tracks TTNN bf16 activations + bf8 linear weights.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoTokenizer

import ttnn

# Linear / norm weights on device use bf8 (TILE). RoPE tables use bf16 (ROW_MAJOR decode path).
TTNN_WEIGHT_DTYPE = ttnn.bfloat8_b
TTNN_ROPE_TABLE_DTYPE = ttnn.bfloat16


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
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rotated * sin


def ref_block_forward(
    hidden_states,
    state_dict,
    layer_num,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    hidden_dim=4096,
    intermediate_dim=12288,
    rms_norm_eps=1e-5,
    rope_theta=1000000.0,
):
    """Reference PyTorch implementation of a single transformer block."""
    seq_len = hidden_states.shape[1]
    prefix = f"model.transformer.blocks.{layer_num}"

    # Step 1: attn_norm
    attn_norm = RefRMSNorm(hidden_dim, rms_norm_eps)
    attn_norm.weight.data = state_dict[f"{prefix}.attn_norm.weight"]
    normed = attn_norm(hidden_states)

    # Step 2: Q, K, V projections
    att_proj = state_dict[f"{prefix}.self_attn.att_proj.weight"]
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    wq = att_proj[:q_dim, :]
    wk = att_proj[q_dim : q_dim + kv_dim, :]
    wv = att_proj[q_dim + kv_dim :, :]

    q = F.linear(normed, wq)
    k = F.linear(normed, wk)
    v = F.linear(normed, wv)

    # Step 3: Reshape
    q = q.reshape(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Step 4: QK-norm
    q_norm = RefRMSNorm(head_dim, 1e-5)
    q_norm.weight.data = state_dict[f"{prefix}.self_attn.q_norm.weight"]
    k_norm = RefRMSNorm(head_dim, 1e-5)
    k_norm.weight.data = state_dict[f"{prefix}.self_attn.k_norm.weight"]

    q = q_norm(q)
    k = k_norm(k)

    # Step 5: RoPE
    positions = torch.arange(seq_len)
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    # Step 6: GQA expansion
    k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v_expanded = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

    # Step 7: Attention
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
    attn_weights = attn_weights + causal_mask

    attn_probs = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_probs, v_expanded)

    # Step 8: Output projection
    attn_out_weight = state_dict[f"{prefix}.self_attn.attn_out.weight"]
    attn_out_reshape = attn_out.transpose(1, 2).reshape(1, seq_len, hidden_dim)
    attn_proj = F.linear(attn_out_reshape, attn_out_weight)

    # Step 9: Residual
    after_attn = hidden_states + attn_proj

    # Step 10: ff_norm
    ff_norm = RefRMSNorm(hidden_dim, rms_norm_eps)
    ff_norm.weight.data = state_dict[f"{prefix}.ff_norm.weight"]
    ff_normed = ff_norm(after_attn)

    # Step 11: MLP (SwiGLU)
    # HuggingFace order: first half is UP (x), second half is GATE
    # Output: silu(gate) * up  (i.e., silu(second_half) * first_half)
    ff_proj = state_dict[f"{prefix}.mlp.ff_proj.weight"]
    ff_out = state_dict[f"{prefix}.mlp.ff_out.weight"]

    ff_proj_out = F.linear(ff_normed, ff_proj)
    up, gate = ff_proj_out.chunk(2, dim=-1)  # up is first half, gate is second half
    swiglu = F.silu(gate) * up
    mlp_out = F.linear(swiglu, ff_out)

    # Step 12: Final residual
    block_out = after_attn + mlp_out

    return block_out


def test_all_layers_pcc():
    """Test PCC for all 36 text model layers using high precision."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    # Config
    hidden_dim = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    intermediate_dim = 12288
    rms_norm_eps = 1e-5
    rope_theta = 1000000.0
    num_layers = 36

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
    ref_hidden = wte[input_ids[0]].unsqueeze(0).float()  # Use float32

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        from models.demos.molmo2.tt.text_block import TextBlock
        from models.demos.molmo2.tt.text_rotary_setup import TextRotarySetup

        # Create rotary setup
        rotary_setup = TextRotarySetup(
            mesh_device=device,
            head_dim=head_dim,
            max_seq_len=8192,
            rope_theta=rope_theta,
            batch_size=1,
            datatype=TTNN_ROPE_TABLE_DTYPE,
        )
        transformation_mats = rotary_setup.get_transformation_mats()
        rot_mats = rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)

        # Track hidden states
        ttnn_hidden = ref_hidden.clone()

        logger.info("=" * 80)
        logger.info(
            "Layer-by-layer PCC (TTNN bf8: block+ln_f+lm_head; RoPE tables bf16; bf16 activations; CPU ref autocast bf16)"
        )
        logger.info("=" * 80)

        for layer_num in range(num_layers):
            # Reference forward (bf16 autocast aligns with TTNN bf16 activations + bf8 weights)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                ref_out = ref_block_forward(
                    ref_hidden,
                    state_dict,
                    layer_num,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=rope_theta,
                )
            ref_out = ref_out.float()

            # TTNN forward (bfloat8_b stored weights; MLP/attention use CPU→bf16 paths where implemented)
            ttnn_block = TextBlock(
                mesh_device=device,
                state_dict=state_dict,
                layer_num=layer_num,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seq_len=8192,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                state_dict_prefix="model.transformer.blocks",
                dtype=TTNN_WEIGHT_DTYPE,
            )

            hidden_ttnn = ttnn.from_torch(
                ttnn_hidden.unsqueeze(0),
                device=device,
                dtype=ttnn.bfloat16,  # Use bfloat16 for activations
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            ttnn_out, _ = ttnn_block(
                hidden_ttnn,
                rot_mats,
                transformation_mats,
                attn_mask=None,
                start_pos=0,
                kv_cache=None,
            )

            ttnn_out_torch = ttnn.to_torch(ttnn_out).squeeze(0).float()

            # Calculate PCC
            pcc, status = calculate_pcc(ref_out, ttnn_out_torch)

            # Calculate max absolute difference
            diff = (ref_out - ttnn_out_torch).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            logger.info(f"Layer {layer_num:2d}: PCC={pcc:.6f}, max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}")

            # Update hidden states for next layer
            ref_hidden = ref_out
            ttnn_hidden = ttnn_out_torch

            # Clean up TTNN tensors
            ttnn.deallocate(hidden_ttnn)
            ttnn.deallocate(ttnn_out)

            # Early stopping if PCC drops too low
            if pcc < 0.9:
                logger.warning(f"PCC dropped below 0.9 at layer {layer_num}, stopping")
                break

        # Final ln_f and lm_head
        logger.info("=" * 80)
        logger.info("Final layers")
        logger.info("=" * 80)

        # Reference ln_f + lm_head (bf16 autocast vs bf8 lm_head on device)
        ref_ln_f = RefRMSNorm(hidden_dim, rms_norm_eps)
        ref_ln_f.weight.data = state_dict["model.transformer.ln_f.weight"]
        lm_head = state_dict["lm_head.weight"]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ref_normed = ref_ln_f(ref_hidden)
            ref_logits = F.linear(ref_normed, lm_head)
        ref_normed = ref_normed.float()
        ref_logits = ref_logits.float()

        # TTNN ln_f
        from models.demos.molmo2.tt.text_rmsnorm import TextRMSNorm

        ttnn_ln_f = TextRMSNorm(
            mesh_device=device,
            state_dict=state_dict,
            hidden_dim=hidden_dim,
            eps=rms_norm_eps,
            state_dict_prefix="model.transformer.ln_f",
            dtype=TTNN_WEIGHT_DTYPE,
        )

        hidden_ttnn_final = ttnn.from_torch(
            ttnn_hidden.unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_normed = ttnn_ln_f(hidden_ttnn_final)
        ttnn_normed_torch = ttnn.to_torch(ttnn_normed).squeeze(0).float()

        pcc, _ = calculate_pcc(ref_normed, ttnn_normed_torch)
        logger.info(f"ln_f PCC: {pcc:.6f}")

        # TTNN lm_head
        lm_head_t = torch.transpose(lm_head, -2, -1).unsqueeze(0).unsqueeze(0)
        lm_head_ttnn = ttnn.from_torch(
            lm_head_t,
            device=device,
            dtype=TTNN_WEIGHT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_logits = ttnn.linear(
            ttnn_normed,
            lm_head_ttnn,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_logits_torch = ttnn.to_torch(ttnn_logits).squeeze().float()

        pcc, _ = calculate_pcc(ref_logits.squeeze(), ttnn_logits_torch)
        logger.info(f"Logits PCC: {pcc:.6f}")

        # Compare top predictions
        logger.info("=" * 80)
        logger.info("Top-5 predictions comparison")
        logger.info("=" * 80)

        ref_last_logits = ref_logits[0, -1, :]
        ttnn_last_logits = ttnn_logits_torch[-1, :]

        ref_top5 = torch.topk(ref_last_logits, 5)
        ttnn_top5 = torch.topk(ttnn_last_logits, 5)

        logger.info("Reference top-5:")
        for i, (score, idx) in enumerate(zip(ref_top5.values, ref_top5.indices)):
            token_text = tokenizer.decode([idx.item()])
            logger.info(f'  {i+1}. "{token_text}" (id={idx.item()}, score={score.item():.2f})')

        logger.info("TTNN top-5:")
        for i, (score, idx) in enumerate(zip(ttnn_top5.values, ttnn_top5.indices)):
            token_text = tokenizer.decode([idx.item()])
            logger.info(f'  {i+1}. "{token_text}" (id={idx.item()}, score={score.item():.2f})')

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_all_layers_pcc()
