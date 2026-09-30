# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Two-card head/channel tensor parallelism for Qwen-Image 2.1 DiT blocks.

Hidden states and modulation are replicated. QKV and gated MLP projections
are column sharded; their output projections are row sharded. Each residual
branch combines partial outputs with a device sum collective.
"""

import torch
import ttnn

from .tt_attention import AttentionWeights
from .tt_block import BlockWeights
from .tt_dit_components import rotary_split_half, split_half_indices, to_device


def prepare_weights(state: dict[str, torch.Tensor], device) -> BlockWeights:
    if device.get_num_devices() != 2:
        raise ValueError("tensor parallel blocks require exactly two devices")
    indices = split_half_indices(128)

    def upload(weight, dim):
        return ttnn.from_torch(
            weight.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim),
        )

    def qkv(name, permute):
        weight = state[f"attn.{name}.weight"]
        if permute:
            weight = weight.reshape(32, 128, 4096)[:, indices].reshape(4096, 4096)
        return upload(weight.T, 1)

    attention = AttentionWeights(
        q=qkv("to_q", True),
        k=qkv("to_k", True),
        v=qkv("to_v", False),
        out=upload(state["attn.to_out.0.weight"].T, 0),
        q_norm=to_device(state["attn.norm_q.weight"][indices].reshape(1, 1, 1, 128), device),
        k_norm=to_device(state["attn.norm_k.weight"][indices].reshape(1, 1, 1, 128), device),
    )
    return BlockWeights(
        attention=attention,
        mlp_gate=upload(state["img_mlp.gate_layer.weight"].T, 1),
        mlp_proj=upload(state["img_mlp.proj.weight"].T, 1),
        mlp_out=upload(state["img_mlp.out.weight"].T, 0),
    )


def combine(value):
    return ttnn.all_reduce(
        value,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def attention(hidden, weights, cos, sin, mask, sequence, compute):
    padded = (sequence + 31) // 32 * 32

    def project(weight, norm=None):
        value = ttnn.matmul(hidden, weight, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.reshape(value, (1, sequence, 16, 128))
        if norm is not None:
            value = ttnn.rms_norm(value, weight=norm, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.permute(value, (0, 2, 1, 3))
        if padded != sequence:
            value = ttnn.pad(value, ((0, 0), (0, 0), (0, padded - sequence), (0, 0)), 0.0)
        return rotary_split_half(value, cos, sin) if norm is not None else value

    q, k, v = project(weights.q, weights.q_norm), project(weights.k, weights.k_norm), project(weights.v)
    value = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=False,
        scale=128**-0.5,
        compute_kernel_config=compute,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    value = ttnn.reshape(ttnn.permute(value, (0, 2, 1, 3)), (1, padded, 2048))
    value = ttnn.slice(value, (0, 0, 0), (1, sequence, 2048))
    partial = ttnn.matmul(value, weights.out, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return combine(partial)


def prefill_block(hidden, weights, modulation, cos, sin, attention_mask, sequence, compute_kernel_config=None):
    scale1, gate1, scale2, gate2 = modulation
    norm = ttnn.layer_norm(hidden, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    branch = ttnn.multiply(norm, ttnn.add(scale1, 1.0))
    branch = attention(branch, weights.attention, cos, sin, attention_mask, sequence, compute_kernel_config)
    hidden = ttnn.add(hidden, ttnn.multiply(ttnn.tanh(gate1), branch))
    norm = ttnn.layer_norm(hidden, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    branch = ttnn.multiply(norm, ttnn.add(scale2, 1.0))
    gate = ttnn.matmul(
        branch, weights.mlp_gate, compute_kernel_config=compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    proj = ttnn.matmul(
        branch, weights.mlp_proj, compute_kernel_config=compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    partial = ttnn.matmul(
        ttnn.multiply(ttnn.silu(gate), proj),
        weights.mlp_out,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.add(hidden, ttnn.multiply(ttnn.tanh(gate2), combine(partial)))
