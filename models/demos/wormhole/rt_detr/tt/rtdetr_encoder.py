# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

# Force maximum math fidelity and full FP32 accumulation to eliminate rounding drift
_precision_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, 
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True
)

def _layer_norm(x, norm_params, eps=1e-5):
    return ttnn.layer_norm(
        x, epsilon=eps, weight=norm_params.weight, bias=norm_params.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

def encoder_layer(x, p, device, num_heads=8, pos_embed=None):
    residual = x
    
    shape = x.shape
    b = shape[0] if len(shape) == 4 else shape[0]
    seq_len = shape[2] if len(shape) == 4 else shape[1]
    hidden = shape[-1]

    x_with_pos = x if pos_embed is None else ttnn.add(x, pos_embed, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Apply precision config to projections
    q = ttnn.linear(x_with_pos, p.self_attn.q.weight, bias=p.self_attn.q.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    k = ttnn.linear(x_with_pos, p.self_attn.k.weight, bias=p.self_attn.k.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    
    if pos_embed is not None:
        ttnn.deallocate(x_with_pos)

    v = ttnn.linear(x, p.self_attn.v.weight, bias=p.self_attn.v.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)

    head_dim = hidden // num_heads

    q = ttnn.transpose(ttnn.reshape(q, (b, seq_len, num_heads, head_dim)), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, (b, seq_len, num_heads, head_dim)), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, (b, seq_len, num_heads, head_dim)), 1, 2)

    attn = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    attn = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
    attn = ttnn.reshape(attn, (b, 1, seq_len, hidden))
    
    attn_out = ttnn.linear(
        attn, p.self_attn.out_proj.weight, bias=p.self_attn.out_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config
    )
    ttnn.deallocate(attn)

    x = ttnn.add(residual, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(attn_out)
    
    x = _layer_norm(x, p.norm1)

    # FFN
    ffn_residual = x
    ffn1 = ttnn.linear(x, p.linear1.weight, bias=p.linear1.bias, activation="gelu", memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    ffn2 = ttnn.linear(ffn1, p.linear2.weight, bias=p.linear2.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    ttnn.deallocate(ffn1)

    x = ttnn.add(ffn_residual, ffn2, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(ffn2)
    
    x = _layer_norm(x, p.norm2)

    return x

def run_aifi(x, layer_params, device, pos_embed=None):
    for p in layer_params:
        x = encoder_layer(x, p, device, pos_embed=pos_embed)
    return x