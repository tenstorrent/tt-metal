# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Transformer encoder on ttnn device for SAM3.

Runs all 6 encoder layers entirely on device (LayerNorm, self-attention with
SDPA, cross-attention via matmul, FFN) with a single CPU↔device transfer at
the boundaries.

Encoder layer (forward_pre):
  norm1 → self_attn(q=k=tgt2+pos, v=tgt2) → residual
  norm2 → cross_attn(q=tgt2, k=v=memory) → residual
  norm3 → FFN(linear1→relu→linear2) → residual
"""

import math

import torch

import ttnn


def _make_compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def preprocess_encoder_weights(encoder):
    """Extract and format weights from encoder layers for device execution."""
    layers = []
    for layer in encoder.layers:
        p = {}
        sa = layer.self_attn
        d = sa.embed_dim

        p["sa_q_w"] = sa.in_proj_weight[:d, :].T.contiguous()
        p["sa_q_b"] = sa.in_proj_bias[:d].unsqueeze(0).unsqueeze(0)
        p["sa_k_w"] = sa.in_proj_weight[d : 2 * d, :].T.contiguous()
        p["sa_k_b"] = sa.in_proj_bias[d : 2 * d].unsqueeze(0).unsqueeze(0)
        p["sa_v_w"] = sa.in_proj_weight[2 * d :, :].T.contiguous()
        p["sa_v_b"] = sa.in_proj_bias[2 * d :].unsqueeze(0).unsqueeze(0)
        p["sa_o_w"] = sa.out_proj.weight.T.contiguous()
        p["sa_o_b"] = sa.out_proj.bias.unsqueeze(0).unsqueeze(0)

        ca = layer.cross_attn_image
        p["ca_q_w"] = ca.in_proj_weight[:d, :].T.contiguous()
        p["ca_q_b"] = ca.in_proj_bias[:d].unsqueeze(0).unsqueeze(0)
        p["ca_k_w"] = ca.in_proj_weight[d : 2 * d, :].T.contiguous()
        p["ca_k_b"] = ca.in_proj_bias[d : 2 * d].unsqueeze(0).unsqueeze(0)
        p["ca_v_w"] = ca.in_proj_weight[2 * d :, :].T.contiguous()
        p["ca_v_b"] = ca.in_proj_bias[2 * d :].unsqueeze(0).unsqueeze(0)
        p["ca_o_w"] = ca.out_proj.weight.T.contiguous()
        p["ca_o_b"] = ca.out_proj.bias.unsqueeze(0).unsqueeze(0)

        p["ff1_w"] = layer.linear1.weight.T.contiguous()
        p["ff1_b"] = layer.linear1.bias.unsqueeze(0).unsqueeze(0)
        p["ff2_w"] = layer.linear2.weight.T.contiguous()
        p["ff2_b"] = layer.linear2.bias.unsqueeze(0).unsqueeze(0)

        for i in range(1, 4):
            norm = getattr(layer, f"norm{i}")
            p[f"n{i}_w"] = norm.weight.unsqueeze(0).unsqueeze(0)
            p[f"n{i}_b"] = norm.bias.unsqueeze(0).unsqueeze(0)

        layers.append(p)
    return layers


def move_encoder_params_to_device(layers, device):
    """Transfer preprocessed encoder weights to device."""
    dev_layers = []
    for p in layers:
        dp = {}
        for key, tensor in p.items():
            if "_w" in key and "n1" not in key and "n2" not in key and "n3" not in key:
                dp[key] = ttnn.from_torch(tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
            else:
                dp[key] = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        dev_layers.append(dp)
    return dev_layers


def _pad_to_tile(tensor, dim, tile_size=32):
    """Pad tensor along dim to next multiple of tile_size."""
    size = tensor.shape[dim]
    padded = math.ceil(size / tile_size) * tile_size
    if padded == size:
        return tensor, size
    pad_widths = [0] * (2 * tensor.ndim)
    pad_idx = 2 * (tensor.ndim - 1 - dim)
    pad_widths[pad_idx + 1] = padded - size
    return torch.nn.functional.pad(tensor, pad_widths), size


def tt_encoder_forward(
    tgt_cpu, memory_cpu, query_pos_cpu, memory_mask_cpu, encoder_params, device, compute_config, sync=True
):
    """Run all encoder layers on device.

    Args:
        tgt_cpu: (1, S, d_model) image features, batch-first
        memory_cpu: (1, M, d_model) text features, batch-first
        query_pos_cpu: (1, S, d_model) positional embeddings
        memory_mask_cpu: (1, M) bool mask, True = ignore
        encoder_params: list of per-layer device weight dicts
        device: ttnn device
        compute_config: compute kernel config

    Returns:
        (1, S, d_model) tensor on CPU
    """
    S = tgt_cpu.shape[1]
    d_model = tgt_cpu.shape[2]
    n_heads = 8
    head_dim = d_model // n_heads
    M_orig = memory_cpu.shape[1]

    memory_padded, _ = _pad_to_tile(memory_cpu, dim=1)
    M_pad = memory_padded.shape[1]

    if memory_mask_cpu is not None:
        mask_padded = torch.ones(1, M_pad, dtype=torch.bool)
        mask_padded[:, :M_orig] = memory_mask_cpu
        attn_bias = torch.zeros(1, 1, 1, M_pad)
        attn_bias[:, :, :, mask_padded[0]] = -1e9
    else:
        attn_bias = torch.zeros(1, 1, 1, M_pad)

    tgt = ttnn.from_torch(tgt_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mem = ttnn.from_torch(memory_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pos = ttnn.from_torch(query_pos_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ca_bias = ttnn.from_torch(attn_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scale = head_dim**-0.5

    sdpa_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=True,
        q_chunk_size=64,
        k_chunk_size=64,
    )

    for p in encoder_params:
        tgt2 = ttnn.layer_norm(tgt, weight=p["n1_w"], bias=p["n1_b"])
        tgt2_pos = ttnn.add(tgt2, pos)

        q = ttnn.linear(tgt2_pos, p["sa_q_w"], bias=p["sa_q_b"], compute_kernel_config=compute_config)
        k = ttnn.linear(tgt2_pos, p["sa_k_w"], bias=p["sa_k_b"], compute_kernel_config=compute_config)
        v = ttnn.linear(tgt2, p["sa_v_w"], bias=p["sa_v_b"], compute_kernel_config=compute_config)

        q = ttnn.reshape(q, [1, S, n_heads, head_dim])
        q = ttnn.permute(q, [0, 2, 1, 3])
        k = ttnn.reshape(k, [1, S, n_heads, head_dim])
        k = ttnn.permute(k, [0, 2, 1, 3])
        v = ttnn.reshape(v, [1, S, n_heads, head_dim])
        v = ttnn.permute(v, [0, 2, 1, 3])

        sa_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, compute_kernel_config=compute_config, program_config=sdpa_config
        )
        sa_out = ttnn.transformer.concatenate_heads(sa_out)
        sa_out = ttnn.linear(sa_out, p["sa_o_w"], bias=p["sa_o_b"], compute_kernel_config=compute_config)
        tgt = ttnn.add(tgt, sa_out)

        tgt2 = ttnn.layer_norm(tgt, weight=p["n2_w"], bias=p["n2_b"])

        ca_q = ttnn.linear(tgt2, p["ca_q_w"], bias=p["ca_q_b"], compute_kernel_config=compute_config)
        ca_k = ttnn.linear(mem, p["ca_k_w"], bias=p["ca_k_b"], compute_kernel_config=compute_config)
        ca_v = ttnn.linear(mem, p["ca_v_w"], bias=p["ca_v_b"], compute_kernel_config=compute_config)

        ca_q = ttnn.reshape(ca_q, [1, S, n_heads, head_dim])
        ca_q = ttnn.permute(ca_q, [0, 2, 1, 3])
        ca_k = ttnn.reshape(ca_k, [1, M_pad, n_heads, head_dim])
        ca_k = ttnn.permute(ca_k, [0, 2, 1, 3])
        ca_v = ttnn.reshape(ca_v, [1, M_pad, n_heads, head_dim])
        ca_v = ttnn.permute(ca_v, [0, 2, 1, 3])

        ca_k_t = ttnn.permute(ca_k, [0, 1, 3, 2])
        scores = ttnn.matmul(ca_q, ca_k_t)
        scores = ttnn.multiply(scores, scale)
        scores = ttnn.add(scores, ca_bias)
        scores = ttnn.softmax(scores, dim=-1)
        ca_out = ttnn.matmul(scores, ca_v)

        ca_out = ttnn.permute(ca_out, [0, 2, 1, 3])
        ca_out = ttnn.reshape(ca_out, [1, S, d_model])
        ca_out = ttnn.linear(ca_out, p["ca_o_w"], bias=p["ca_o_b"], compute_kernel_config=compute_config)
        tgt = ttnn.add(tgt, ca_out)

        tgt2 = ttnn.layer_norm(tgt, weight=p["n3_w"], bias=p["n3_b"])
        ff = ttnn.linear(tgt2, p["ff1_w"], bias=p["ff1_b"], compute_kernel_config=compute_config)
        ff = ttnn.relu(ff)
        ff = ttnn.linear(ff, p["ff2_w"], bias=p["ff2_b"], compute_kernel_config=compute_config)
        tgt = ttnn.add(tgt, ff)

    if not sync:
        return tgt
    return ttnn.to_torch(tgt).float()
