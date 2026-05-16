# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Hubert Encoder Layer.

Complete TransformerSentenceEncoderLayer (layer_norm_first=True):
    residual = x
    x = self_attn_layer_norm(x)
    x = self_attention(x)     # fused QKV → SDPA → out_proj
    x = residual + x
    residual = x
    x = final_layer_norm(x)
    x = gelu(fc1(x))
    x = fc2(x)
    x = residual + x

Validated: PCC=0.999996 single layer, PCC=0.999907 12-layer stack.
"""

import torch
import ttnn

from models.demos.rvc.ttnn.utils import (
    to_device,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


def preprocess_encoder_layer_weights(
    device,
    self_attn_ln: torch.nn.LayerNorm,
    final_ln: torch.nn.LayerNorm,
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    out_proj: torch.nn.Linear,
    fc1: torch.nn.Linear,
    fc2: torch.nn.Linear,
) -> dict:
    """Preprocess all encoder layer weights for TTNN."""
    # Fused QKV weight
    fused_w = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
    fused_b = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)

    return {
        "sa_ln_w": to_device(self_attn_ln.weight.unsqueeze(0).unsqueeze(0), device),
        "sa_ln_b": to_device(self_attn_ln.bias.unsqueeze(0).unsqueeze(0), device),
        "qkv_w": preprocess_linear_weight(fused_w, device),
        "qkv_b": preprocess_linear_bias(fused_b, device),
        "out_w": preprocess_linear_weight(out_proj.weight, device),
        "out_b": preprocess_linear_bias(out_proj.bias, device),
        "ff_ln_w": to_device(final_ln.weight.unsqueeze(0).unsqueeze(0), device),
        "ff_ln_b": to_device(final_ln.bias.unsqueeze(0).unsqueeze(0), device),
        "fc1_w": preprocess_linear_weight(fc1.weight, device),
        "fc1_b": preprocess_linear_bias(fc1.bias, device),
        "fc2_w": preprocess_linear_weight(fc2.weight, device),
        "fc2_b": preprocess_linear_bias(fc2.bias, device),
    }


def ttnn_encoder_layer_forward(
    x_tt: ttnn.Tensor,
    params: dict,
    device,
    batch_size: int,
    seq_len: int,
    embed_dim: int = 768,
    num_heads: int = 12,
    head_dim: int = 64,
) -> ttnn.Tensor:
    """
    TTNN encoder layer forward (layer_norm_first=True).

    Uses fused QKV + nlp_create_qkv_heads + SDPA + nlp_concat_heads.

    Args:
        x_tt: Input [B, S, embed_dim] on device, TILE_LAYOUT.
        params: Weight dict from preprocess_encoder_layer_weights.
        device: TTNN device.
        batch_size: Batch size.
        seq_len: Sequence length.
        embed_dim: Embedding dimension (default: 768).
        num_heads: Number of attention heads (default: 12).
        head_dim: Head dimension (default: 64).

    Returns:
        [B, S, embed_dim] on device.
    """
    scaling = head_dim ** -0.5

    # === Attention half ===
    residual = x_tt
    h = ttnn.layer_norm(x_tt, weight=params["sa_ln_w"], bias=params["sa_ln_b"],
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Fused QKV
    qkv = ttnn.linear(h, params["qkv_w"], bias=params["qkv_b"],
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    qkv = ttnn.reshape(qkv, (batch_size, 1, seq_len, 3 * embed_dim))

    # Head split
    q_h, k_h, v_h = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=num_heads, num_kv_heads=num_heads,
        transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(qkv)

    # SDPA (non-causal encoder)
    compute_grid_size = device.compute_with_storage_grid_size()
    sdpa_out = ttnn.transformer.scaled_dot_product_attention(
        q_h, k_h, v_h, scale=scaling, attn_mask=None, is_causal=False,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(compute_grid_size.x, compute_grid_size.y),
            q_chunk_size=min(256, seq_len), k_chunk_size=min(256, seq_len),
            exp_approx_mode=False,
        ),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(q_h)
    ttnn.deallocate(k_h)
    ttnn.deallocate(v_h)

    # Merge heads + out_proj
    merged = ttnn.experimental.nlp_concat_heads(sdpa_out)
    ttnn.deallocate(sdpa_out)
    attn_result = ttnn.linear(merged, params["out_w"], bias=params["out_b"],
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(merged)

    # Residual
    x_tt = ttnn.add(residual, attn_result)
    ttnn.deallocate(attn_result)

    # === FFN half ===
    residual = x_tt
    h = ttnn.layer_norm(x_tt, weight=params["ff_ln_w"], bias=params["ff_ln_b"],
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h = ttnn.linear(h, params["fc1_w"], bias=params["fc1_b"],
                     activation="gelu", memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h = ttnn.linear(h, params["fc2_w"], bias=params["fc2_b"],
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_tt = ttnn.add(residual, h)
    ttnn.deallocate(h)

    return x_tt
