# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 shipped-policy capture target for the Gemma-4 advisor challenger."""

from __future__ import annotations

import os
import sys
import json
from types import SimpleNamespace

import torch
import ttnn


TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # Keep the advisor environment's ttnn ahead of the repository namespace.
    sys.path.append(TT_METAL_ROOT)
MAIN_SITE_PACKAGES = f"{TT_METAL_ROOT}/python_env/lib/python3.12/site-packages"
if MAIN_SITE_PACKAGES not in sys.path:
    # Reuse model-harness dependencies absent from the deliberately minimal
    # advisor venv. ttnn has already resolved from the advisor environment.
    sys.path.append(MAIN_SITE_PACKAGES)

LAYER_IDX = int(os.environ["CHALLENGER_LAYER_IDX"])
BATCH = 1
SEQ_LEN = 1024

_DECODER = None
_DECODE_ARGS = None


def _build(device):
    from safetensors.torch import load_file
    from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
        FULL_BLOCK_SIZE,
        FULL_HEAD_DIM,
        FULL_NUM_KV_HEADS,
        HIDDEN_SIZE,
        SLIDING_BLOCK_SIZE,
        SLIDING_HEAD_DIM,
        SLIDING_NUM_KV_HEADS,
    )
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    config_path = f"{TT_METAL_ROOT}/models/demos/gemma4/configs/gemma-4-26B-A4B-it/config.json"
    with open(config_path) as fh:
        cfg = SimpleNamespace(**json.load(fh)["text_config"])
    layer_type = cfg.layer_types[LAYER_IDX]
    state = load_file(f"/tmp/gemma4_real_layer_cache/layer_{LAYER_IDX}.safetensors")
    attention_dtype = ttnn.bfloat16 if layer_type == "sliding_attention" else ttnn.bfloat8_b
    dram_roles = ("packed_mlp_gate_up", "mlp_down")
    if layer_type == "full_attention":
        dram_roles = ("o_proj", *dram_roles)

    # These are the effective shipped settings recorded in incumbent.json.
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        weight_dtype=ttnn.bfloat16,
        attention_weight_dtype=attention_dtype,
        mlp_weight_dtype=ttnn.bfloat8_b,
        mlp_down_weight_dtype=ttnn.bfloat8_b,
        prefill_expert_weight_dtype=ttnn.bfloat8_b,
        expert_weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        attention_math_fidelity=ttnn.MathFidelity.HiFi4,
        full_attention_math_fidelity=ttnn.MathFidelity.LoFi,
        mlp_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_gate_math_fidelity=ttnn.MathFidelity.LoFi,
        expert_math_fidelity=ttnn.MathFidelity.LoFi,
        packed_dense_gate_up=True,
        expert_decode_input_l1=True,
        expert_gate_in0_block_w=11,
        expert_down_in0_block_w=11,
        expert_gate_per_core_n=2,
        expert_down_per_core_n=2,
        dram_in0_block_w=11,
        dram_sharded_roles=dram_roles,
        residual_shard_cores=0,
    )

    torch.manual_seed(1000 + LAYER_IDX)
    decode_hidden = torch.randn(BATCH, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    if layer_type == "sliding_attention":
        num_kv_heads, block_size, head_dim = (
            SLIDING_NUM_KV_HEADS,
            SLIDING_BLOCK_SIZE,
            SLIDING_HEAD_DIM,
        )
    else:
        num_kv_heads, block_size, head_dim = FULL_NUM_KV_HEADS, FULL_BLOCK_SIZE, FULL_HEAD_DIM
    blocks_per_user = (SEQ_LEN + 1 + block_size - 1) // block_size

    def as_tt(tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.as_tensor(
            tensor,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    page_table = as_tt(
        torch.arange(blocks_per_user, dtype=torch.int32).view(BATCH, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (blocks_per_user, num_kv_heads, block_size, head_dim)
    kv_cache = (
        as_tt(torch.zeros(cache_shape, dtype=torch.bfloat16)),
        as_tt(torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    tt_decode_cos = torch.randn(1, 1, BATCH, head_dim, dtype=torch.bfloat16)
    tt_decode_sin = torch.randn(1, 1, BATCH, head_dim, dtype=torch.bfloat16)
    args = {
        "hidden_states": as_tt(decode_hidden.transpose(0, 1).unsqueeze(0)),
        "position_cos": as_tt(tt_decode_cos),
        "position_sin": as_tt(tt_decode_sin),
        "current_pos": as_tt(
            torch.full((BATCH,), SEQ_LEN, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    return decoder, args


def decode(hidden_states):
    # Mirror the shipped decode through the dense and router prefix. The
    # following routed expert op is ttnn.sparse_matmul, a documented terminal
    # in the advisor tracer, so it is deliberately recorded as uncapturable.
    residual = hidden_states
    attn_in = _DECODER._rms_norm(hidden_states, _DECODER.weights.input_ln)
    attn_out = _DECODER._attention_decode(attn_in, cache_position_modulo=None, **_DECODE_ARGS)
    attn_out = _DECODER._rms_norm(attn_out, _DECODER.weights.post_attn_ln)
    hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    residual = hidden_states
    mlp_in = _DECODER._rms_norm(hidden_states, _DECODER.weights.pre_ff_ln)
    mlp_out = _DECODER._dense_mlp(mlp_in)
    hidden_1 = _DECODER._rms_norm(mlp_out, _DECODER.weights.post_ff_ln_1)
    router_weights = _DECODER._router_weights(residual)
    moe_in = _DECODER._rms_norm(residual, _DECODER.weights.pre_ff_ln_2)
    ttnn.multiply(router_weights, router_weights)
    return ttnn.add(hidden_1, moe_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_inputs(device):
    global _DECODER, _DECODE_ARGS
    _DECODER, args = _build(device)
    hidden = args.pop("hidden_states")
    _DECODE_ARGS = args
    return (hidden,)
