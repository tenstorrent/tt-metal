# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN functional implementation of common modules shared between TADA encoder and decoder.

Includes:
- Snake1d activation: x + sin(alpha * x)^2 / alpha^2
- RoPE (Rotary Position Embeddings) for LocalSelfAttention
- LocalSelfAttention with RoPE and segment masks
- LocalAttentionEncoderLayer (attention + FFN)
- LocalAttentionEncoder (stack of layers)
"""


import torch

import ttnn

TADA_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG

# High-fidelity compute config for decoder attention matmuls.
# fp32_dest_acc_en=True prevents bfloat16 precision loss that accumulates
# through 6 transformer layers (PCC drops 0.99→0.89 without this).
TADA_HIFI4_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def snake1d(x, alpha):
    """
    Snake activation: x + sin(alpha * x)^2 / alpha^2

    This is a non-standard activation from the DAC codec library.
    Decomposed into TTNN elementwise ops.

    Args:
        x: input tensor (B, C, T) or (B, 1, C*T) on device
        alpha: (C,) learned frequency parameter on device
    Returns:
        activated tensor, same shape as x
    """
    # sin(alpha * x)
    ax = ttnn.mul(x, alpha, memory_config=TADA_MEMORY_CONFIG)
    sin_ax = ttnn.sin(ax, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(ax)
    # sin(alpha * x)^2
    sin_sq = ttnn.mul(sin_ax, sin_ax, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(sin_ax)
    # alpha^2
    alpha_sq = ttnn.mul(alpha, alpha, memory_config=TADA_MEMORY_CONFIG)
    # sin^2 / alpha^2
    result = ttnn.div(sin_sq, alpha_sq, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(sin_sq)
    ttnn.deallocate(alpha_sq)
    # x + sin^2/alpha^2
    output = ttnn.add(x, result, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(result)
    return output


def compute_rope_freqs(head_dim, max_seq_len):
    """
    Precompute RoPE rotation frequencies on CPU.

    Returns:
        (max_seq_len, head_dim//2, 2) tensor with [cos, sin] pairs
    """
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)


def apply_rope_cpu(q_or_k_cpu, rope_cos_cpu, rope_sin_cpu):
    """
    Apply interleaved RoPE on CPU, matching the reference TADA implementation.

    The reference rotates interleaved pairs: (dim0,dim1), (dim2,dim3), ...
    This requires reshaping to (..., head_dim//2, 2), which is not feasible
    in TTNN's tile layout. So we do RoPE on CPU and transfer back.

    Args:
        q_or_k_cpu: (B, num_heads, seq_len, head_dim) float tensor on CPU
        rope_cos_cpu: (seq_len, head_dim//2) on CPU
        rope_sin_cpu: (seq_len, head_dim//2) on CPU
    Returns:
        (B, num_heads, seq_len, head_dim) rotated tensor on CPU
    """
    B, H, S, D = q_or_k_cpu.shape
    half_D = D // 2

    # Reshape to interleaved pairs: (B, H, S, D//2, 2)
    x = q_or_k_cpu.reshape(B, H, S, half_D, 2)
    x0 = x[..., 0]  # even indices: (B, H, S, D//2)
    x1 = x[..., 1]  # odd indices:  (B, H, S, D//2)

    # Broadcast cos/sin: (S, D//2) -> (1, 1, S, D//2)
    cos = rope_cos_cpu.unsqueeze(0).unsqueeze(0)
    sin = rope_sin_cpu.unsqueeze(0).unsqueeze(0)

    # Rotate: [x0*cos - x1*sin, x0*sin + x1*cos]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos

    # Stack back to interleaved order and reshape
    return torch.stack([r0, r1], dim=-1).reshape(B, H, S, D)


def local_self_attention(
    x,
    seq_len,
    attention_mask,
    rope_cos_cpu,
    rope_sin_cpu,
    *,
    parameters,
    device,
    input_mesh_mapper,
    output_mesh_composer,
):
    """
    Local self-attention with interleaved RoPE.

    Args:
        x: (B, 1, S, d_model) hidden states on device
        seq_len: actual sequence length (before padding)
        attention_mask: (B, num_heads, S, S) attention mask on device, or None
        rope_cos_cpu: (S, head_dim//2) precomputed cos on CPU
        rope_sin_cpu: (S, head_dim//2) precomputed sin on CPU
        parameters: contains qkv, out_proj, layer_norm weights
        device: TT device
        input_mesh_mapper: mesh mapper for transferring tensors to device
        output_mesh_composer: mesh composer for transferring tensors from device
    Returns:
        (B, 1, S, d_model) output
    """
    d_model = x.shape[-1]

    # Fused QKV projection
    compute_grid_size = x.device().compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x)
    qkv = ttnn.linear(
        x,
        parameters.qkv.weight,
        bias=parameters.qkv.bias,
        core_grid=core_grid,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
    )

    # Split into Q, K, V using nlp_create_qkv_heads
    num_heads = d_model // (d_model // 8)  # = 8 for TADA encoder/decoder
    query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
        qkv,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        transpose_k_heads=False,
        memory_config=TADA_MEMORY_CONFIG,
    )
    ttnn.deallocate(qkv)

    # Apply interleaved RoPE on CPU (TTNN tile layout doesn't support the 5D reshape needed)
    # Transfer Q, K to CPU, apply RoPE, transfer back. Data is small (~1MB each).
    q_cpu = ttnn.to_torch(query_states, mesh_composer=output_mesh_composer).float()
    k_cpu = ttnn.to_torch(key_states, mesh_composer=output_mesh_composer).float()
    ttnn.deallocate(query_states)
    ttnn.deallocate(key_states)

    q_rotated = apply_rope_cpu(q_cpu, rope_cos_cpu, rope_sin_cpu)
    k_rotated = apply_rope_cpu(k_cpu, rope_cos_cpu, rope_sin_cpu)

    query_states = ttnn.from_torch(
        q_rotated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )
    key_states = ttnn.from_torch(
        k_rotated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=input_mesh_mapper,
    )

    # Scaled dot-product attention
    head_dim = d_model // num_heads
    scaling = head_dim**-0.5

    # Transpose K for attention: (B, H, S, d) -> (B, H, d, S)
    key_states_t = ttnn.permute(key_states, [0, 1, 3, 2])
    ttnn.deallocate(key_states)

    # Q @ K^T
    attn_scores = ttnn.matmul(
        query_states,
        key_states_t,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
    )
    ttnn.deallocate(query_states)
    ttnn.deallocate(key_states_t)

    # Scale
    attn_scores = ttnn.mul(attn_scores, scaling, memory_config=TADA_MEMORY_CONFIG)

    # Apply attention mask
    if attention_mask is not None:
        attn_scores = ttnn.add(attn_scores, attention_mask, memory_config=TADA_MEMORY_CONFIG)

    # Softmax
    attn_weights = ttnn.softmax(
        attn_scores,
        dim=-1,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
    )
    ttnn.deallocate(attn_scores)

    # Attention output
    attn_output = ttnn.matmul(
        attn_weights,
        value_states,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
    )
    ttnn.deallocate(attn_weights)
    ttnn.deallocate(value_states)

    # Concat heads
    attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=TADA_MEMORY_CONFIG)

    # Output projection
    output = ttnn.linear(
        attn_output,
        parameters.out_proj.weight,
        bias=parameters.out_proj.bias,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
    )
    ttnn.deallocate(attn_output)

    # Residual + LayerNorm
    residual_sum = ttnn.add(x, output, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(output)
    output = ttnn.layer_norm(
        residual_sum,
        weight=parameters.layer_norm.weight,
        bias=parameters.layer_norm.bias,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    ttnn.deallocate(residual_sum)

    return output


def local_attention_encoder_layer(
    x,
    seq_len,
    attention_mask,
    rope_cos_cpu,
    rope_sin_cpu,
    *,
    parameters,
    device,
    input_mesh_mapper,
    output_mesh_composer,
):
    """
    Single encoder layer: LocalSelfAttention + FFN.

    Args:
        x: (B, 1, S, d_model) on device
        seq_len: actual sequence length
        attention_mask: (B, num_heads, S, S) on device or None
        rope_cos_cpu, rope_sin_cpu: precomputed RoPE frequencies on CPU
        parameters: layer parameters
        device: TT device
        input_mesh_mapper: mesh mapper for transferring tensors to device
        output_mesh_composer: mesh composer for transferring tensors from device
    Returns:
        (B, 1, S, d_model) output
    """
    # Self-attention (includes residual + layernorm)
    x = local_self_attention(
        x,
        seq_len,
        attention_mask,
        rope_cos_cpu,
        rope_sin_cpu,
        parameters=parameters.self_attn,
        device=device,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    # FFN: Linear -> GELU -> Linear
    ffn_out = ttnn.linear(
        x,
        parameters.ffn[0].weight,
        bias=parameters.ffn[0].bias,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
    )
    ffn_out = ttnn.gelu(ffn_out, memory_config=TADA_MEMORY_CONFIG)
    ffn_out = ttnn.linear(
        ffn_out,
        parameters.ffn[3].weight,
        bias=parameters.ffn[3].bias,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
    )

    # Residual + LayerNorm
    residual_sum = ttnn.add(x, ffn_out, memory_config=TADA_MEMORY_CONFIG)
    ttnn.deallocate(ffn_out)
    output = ttnn.layer_norm(
        residual_sum,
        weight=parameters.norm.weight,
        bias=parameters.norm.bias,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    ttnn.deallocate(residual_sum)

    return output


def local_attention_encoder(
    x, seq_len, attention_mask_torch, *, parameters, device, input_mesh_mapper, output_mesh_composer
):
    """
    Full LocalAttentionEncoder: input_proj + N layers + final_norm.

    Args:
        x: (B, 1, S, d_model) on device
        seq_len: actual sequence length
        attention_mask_torch: (B, S, S) boolean mask on CPU (True=masked) or None
        parameters: encoder parameters
        device: TT device
        input_mesh_mapper: mesh mapper for input tensors
        output_mesh_composer: mesh composer for reading tensors from device
    Returns:
        (B, 1, S, d_model) output
    """
    d_model = x.shape[-1]
    num_heads = 8  # TADA encoder/decoder uses 8 heads
    head_dim = d_model // num_heads

    # Precompute RoPE on CPU (kept as CPU tensors for interleaved rotation)
    rope_freqs = compute_rope_freqs(head_dim, seq_len)
    rope_cos_cpu = rope_freqs[:, :, 0]  # (S, D/2)
    rope_sin_cpu = rope_freqs[:, :, 1]  # (S, D/2)

    # Convert attention mask to float mask on device
    if attention_mask_torch is not None:
        # Convert boolean mask (True=masked) to additive float mask (-inf for masked)
        float_mask = torch.zeros_like(attention_mask_torch, dtype=torch.bfloat16)
        float_mask.masked_fill_(attention_mask_torch, float("-inf"))
        # Expand for heads: (B, S, S) -> (B, num_heads, S, S)
        float_mask = float_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attention_mask = ttnn.from_torch(
            float_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=input_mesh_mapper
        )
    else:
        attention_mask = None

    # Input projection (Identity for d_input == d_model)
    try:
        input_proj_weight = parameters.input_proj.weight
        input_proj_bias = parameters.input_proj.get("bias", None) if hasattr(parameters.input_proj, "get") else None
        x = ttnn.linear(
            x,
            input_proj_weight,
            bias=input_proj_bias,
            memory_config=TADA_MEMORY_CONFIG,
            compute_kernel_config=TADA_HIFI4_COMPUTE_CONFIG,
        )
    except (KeyError, AttributeError):
        pass

    # Process through layers
    for layer_params in parameters.layers:
        x = local_attention_encoder_layer(
            x,
            seq_len,
            attention_mask,
            rope_cos_cpu,
            rope_sin_cpu,
            parameters=layer_params,
            device=device,
            input_mesh_mapper=input_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )

    # Final LayerNorm
    x = ttnn.layer_norm(
        x,
        weight=parameters.final_norm.weight,
        bias=parameters.final_norm.bias,
        memory_config=TADA_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return x


def convert_to_ttnn(model, name):
    """Convert all modules except Conv1d layers."""
    if "conv" in name.lower():
        return False
    return True


def create_custom_mesh_preprocessor(weights_mesh_mapper):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, weights_mesh_mapper)

    return custom_mesh_preprocessor


def custom_preprocessor(torch_model, name, weights_mesh_mapper):
    """Custom preprocessor for common TADA modules."""
    parameters = {}

    # Handle Snake1d - extract alpha parameter
    try:
        from dac.nn.layers import Snake1d as DACSnake1d
    except ImportError:
        DACSnake1d = None

    if DACSnake1d is not None and isinstance(torch_model, DACSnake1d):
        if hasattr(torch_model, "alpha"):
            parameters["alpha"] = ttnn.from_torch(
                torch_model.alpha.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
            )

    return parameters
