# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for the Llama 3.1 8B optimized decoder."""

from __future__ import annotations

import os
import sys

import torch
from transformers import AutoConfig

import ttnn

REPO_ROOT = os.environ.get("TT_METAL_REPO_ROOT", "/localdev/mvasiljevic/tt-metal")
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (  # noqa: E402
    EMITTED_BATCH_SIZE,
    EMITTED_CACHE_LEN,
    MODEL_ID,
    build_decode_attention_mask,
    build_decode_rope,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (  # noqa: E402
    OptimizedDecoder,
    OptimizedDecoderPolicy,
)

LAYER_IDX = int(os.environ.get("SHARD_ADVISE_LAYER", "0"))

_DECODER = None
_KWARGS = None


def _synthetic_state_dict(hf_config, layer_idx: int):
    torch.manual_seed(20260714 + layer_idx)
    h = hf_config.hidden_size
    kv = hf_config.num_key_value_heads * hf_config.head_dim
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        f"{prefix}.self_attn.q_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.self_attn.k_proj.weight": torch.randn(kv, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.self_attn.v_proj.weight": torch.randn(kv, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.self_attn.o_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.mlp.gate_proj.weight": torch.randn(hf_config.intermediate_size, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.mlp.up_proj.weight": torch.randn(hf_config.intermediate_size, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.mlp.down_proj.weight": torch.randn(h, hf_config.intermediate_size, dtype=torch.bfloat16) * 0.01,
    }


def _tt(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build(mesh_device):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_state_dict(hf_config, LAYER_IDX)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        cache_len=EMITTED_CACHE_LEN,
        policy=OptimizedDecoderPolicy(),
    )
    cache_position = EMITTED_CACHE_LEN - 1
    hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    key_cache = torch.zeros(
        EMITTED_BATCH_SIZE,
        hf_config.num_key_value_heads,
        EMITTED_CACHE_LEN,
        hf_config.head_dim,
        dtype=torch.bfloat16,
    )
    value_cache = torch.zeros_like(key_cache)
    tt_hidden = _tt(hidden.reshape(1, 1, EMITTED_BATCH_SIZE, hf_config.hidden_size), mesh_device)
    kwargs = {
        "key_cache": _tt(key_cache, mesh_device),
        "value_cache": _tt(value_cache, mesh_device),
        "cache_position": _tt(
            torch.tensor([cache_position], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "cos": build_decode_rope(hf_config, cache_position, mesh_device)[0],
        "sin": build_decode_rope(hf_config, cache_position, mesh_device)[1],
        "attention_mask": build_decode_attention_mask(cache_position, EMITTED_CACHE_LEN, mesh_device),
    }
    return decoder, kwargs, tt_hidden


def decode(hidden):
    decoder = _DECODER
    normed = ttnn.rms_norm(
        hidden,
        epsilon=decoder.cfg.rms_norm_eps,
        weight=decoder.input_layernorm_weight,
        memory_config=decoder.qvk_input_memcfg,
        compute_kernel_config=decoder.norm_compute_kernel_config,
    )
    qvk = decoder._matmul(
        normed,
        decoder.qvk_proj_weight,
        input_memory_config=decoder.qvk_input_memcfg,
        output_memory_config=decoder.qvk_output_memcfg,
        program_config=decoder.qvk_program_config,
        compute_kernel_config=decoder.attention_compute_kernel_config,
    )
    attn_out = ttnn.slice(
        qvk,
        [0, 0, 0, 0],
        [1, 1, EMITTED_BATCH_SIZE, decoder.cfg.hidden_size],
        [1, 1, 1, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_out = decoder._matmul(
        attn_out,
        decoder.o_proj_weight,
        input_memory_config=decoder.o_input_memcfg,
        output_memory_config=decoder.o_output_memcfg,
        program_config=decoder.o_program_config,
        compute_kernel_config=decoder.attention_compute_kernel_config,
    )
    hidden_states = ttnn.add(
        attn_out, hidden, dtype=decoder.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mlp_in = ttnn.rms_norm(
        hidden_states,
        epsilon=decoder.cfg.rms_norm_eps,
        weight=decoder.post_attention_layernorm_weight,
        memory_config=decoder.mlp_gate_input_memcfg,
        compute_kernel_config=decoder.norm_compute_kernel_config,
    )
    if decoder.policy.use_packed_gate_up_projection:
        gate_up = decoder._matmul(
            mlp_in,
            decoder.gate_up_proj_weight,
            input_memory_config=decoder.mlp_gate_input_memcfg,
            output_memory_config=decoder.mlp_gate_up_packed_output_memcfg,
            program_config=decoder.gate_up_packed_program_config,
            compute_kernel_config=decoder.mlp_compute_kernel_config,
        )
        gate, up = ttnn.split(
            gate_up,
            decoder.cfg.intermediate_size,
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        gate = decoder._matmul(
            mlp_in,
            decoder.gate_proj_weight,
            input_memory_config=decoder.mlp_gate_input_memcfg,
            output_memory_config=decoder.mlp_gate_output_memcfg,
            program_config=decoder.gate_program_config,
            compute_kernel_config=decoder.mlp_compute_kernel_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        up = decoder._matmul(
            mlp_in,
            decoder.up_proj_weight,
            input_memory_config=decoder.mlp_up_input_memcfg,
            output_memory_config=decoder.mlp_up_output_memcfg,
            program_config=decoder.up_program_config,
            compute_kernel_config=decoder.mlp_compute_kernel_config,
        )
    mlp = ttnn.multiply(gate, up, dtype=decoder.policy.activation_dtype, memory_config=decoder.mlp_down_input_memcfg)
    mlp = decoder._matmul(
        mlp,
        decoder.down_proj_weight,
        input_memory_config=decoder.mlp_down_input_memcfg,
        output_memory_config=decoder.mlp_down_output_memcfg,
        program_config=decoder.down_program_config,
        compute_kernel_config=decoder.mlp_compute_kernel_config,
    )
    return ttnn.add(mlp, hidden_states, dtype=decoder.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
