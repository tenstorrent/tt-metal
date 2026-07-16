# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture target for the rewritten Falcon3-10B dense decoder block."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("FALCON3_TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
HF_MODEL = "tiiuae/Falcon3-10B-Base"
LAYER_IDX = 20
BATCH = 32
MAX_CACHE_LEN = 128
POSITION = 17


def _synthetic_state_dict(config, layer_idx: int):
    generator = torch.Generator().manual_seed(20260716)

    def weight(shape):
        tensor = torch.empty(shape, dtype=torch.bfloat16)
        return tensor.normal_(mean=0.0, std=0.02, generator=generator)

    prefix = f"model.layers.{layer_idx}."
    hidden = config.hidden_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    intermediate = config.intermediate_size
    return {
        prefix + "self_attn.q_proj.weight": weight((q_width, hidden)),
        prefix + "self_attn.k_proj.weight": weight((kv_width, hidden)),
        prefix + "self_attn.v_proj.weight": weight((kv_width, hidden)),
        prefix + "self_attn.o_proj.weight": weight((hidden, hidden)),
        prefix + "mlp.gate_proj.weight": weight((intermediate, hidden)),
        prefix + "mlp.up_proj.weight": weight((intermediate, hidden)),
        prefix + "mlp.down_proj.weight": weight((hidden, intermediate)),
        prefix + "input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }


def _build(device):
    if TT_METAL_ROOT not in sys.path:
        sys.path.append(TT_METAL_ROOT)

    from models.autoports.tiiuae_falcon3_10b_base.tt.optimized_decoder import OptimizedDecoder

    class AdvisorDecoder(OptimizedDecoder):
        """Capture-only form with explicit moves instead of layout introspection.

        The interception tracer intentionally treats memory_config as unknown
        until the greedy optimizer assigns it, so production's no-op move
        guards cannot be evaluated while building TTIR.
        """

        def _decode_norm(self, residual, weight):
            norm_input = ttnn.to_memory_config(residual, self.norm_memory_config)
            output = ttnn.rms_norm(
                norm_input,
                epsilon=self.rms_norm_eps,
                weight=weight,
                program_config=self.norm_program_config,
                compute_kernel_config=self.norm_compute_config,
                memory_config=self.norm_memory_config,
            )
            ttnn.deallocate(norm_input, True)
            return output

        def _move_owned(self, tensor, memory_config):
            moved = ttnn.to_memory_config(tensor, memory_config)
            ttnn.deallocate(tensor, True)
            return moved

    # Keep the capture target independent of the experiment's transformers
    # installation.  These are the exact checked Falcon3-10B config fields
    # consumed by OptimizedDecoder.from_state_dict.
    config = SimpleNamespace(
        hidden_size=3072,
        num_hidden_layers=40,
        num_attention_heads=12,
        num_key_value_heads=4,
        head_dim=256,
        intermediate_size=23040,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=32768,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=1000042.0,
        rope_parameters={"rope_type": "default"},
    )
    state_dict = _synthetic_state_dict(config, LAYER_IDX)
    decoder = AdvisorDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        decode_matmul_mode="shard_advisor",
        max_cache_len=MAX_CACHE_LEN,
        precision_policy="bfp8_hifi2",
        use_packed_mlp=False,
        use_explicit_decode_mask=False,
    )
    key_cache, value_cache = decoder.allocate_kv_cache()
    hidden = torch.randn(BATCH, 1, config.hidden_size, dtype=torch.bfloat16)
    tt_hidden = ttnn.from_torch(
        hidden.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cache_position = ttnn.from_torch(
        torch.full((BATCH,), POSITION, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return decoder, key_cache, value_cache, cache_position, tt_hidden


_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None
_CACHE_POSITION = None


def decode(hidden):
    return _DECODER.decode_forward(
        hidden,
        key_cache=_KEY_CACHE,
        value_cache=_VALUE_CACHE,
        cache_position=_CACHE_POSITION,
        position_index=POSITION,
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE, _CACHE_POSITION
    _DECODER, _KEY_CACHE, _VALUE_CACHE, _CACHE_POSITION, hidden = _build(device)
    return (hidden,)
