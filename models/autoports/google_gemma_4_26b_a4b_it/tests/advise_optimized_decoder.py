# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shipped-precision advisor capture target for one Gemma 4 decode layer kind."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn


TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # Preserve the advisor environment's ttnn package precedence.
    sys.path.append(TT_METAL_ROOT)
_REPO_SITE_PACKAGES = str(Path(TT_METAL_ROOT) / "python_env/lib/python3.12/site-packages")
if _REPO_SITE_PACKAGES not in sys.path:
    # The advisor venv intentionally carries only compiler dependencies. Reuse
    # the repo environment for Transformers/test builders, after advisor ttnn.
    sys.path.append(_REPO_SITE_PACKAGES)

MODEL_ROOT = Path(TT_METAL_ROOT) / "models/autoports/google_gemma_4_26b_a4b_it"
INCUMBENT = json.loads((MODEL_ROOT / "doc/advisor_challenger/incumbent.json").read_text())
LAYER_KIND = os.environ.get("CHALLENGER_LAYER_KIND", "sliding_attention")
LAYER_IDX = {"sliding_attention": 0, "full_attention": 5}[LAYER_KIND]
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "1"))
CURRENT_POS = int(os.environ.get("CHALLENGER_CURRENT_POS", "1024"))

_DECODER = None
_POSITION_COS = None
_POSITION_SIN = None
_CURRENT_POS_TENSOR = None
_PAGE_TABLE = None
_KV_CACHE = None


def _build(device):
    from models.autoports.google_gemma_4_26b_a4b_it.tests.synthetic_weights import synthetic_layer_state_dict
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    hidden_size = 2816
    layer_types = ["sliding_attention"] * 30
    layer_types[5] = "full_attention"
    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=2112,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_hidden_layers=30,
        layer_types=layer_types,
        sliding_window=1024,
        rms_norm_eps=1e-6,
        num_experts=128,
        top_k_experts=8,
        moe_intermediate_size=704,
        enable_moe_block=True,
        hidden_size_per_layer_input=0,
        hidden_activation="gelu_pytorch_tanh",
        attention_k_eq_v=True,
    )
    assert cfg.layer_types[LAYER_IDX] == LAYER_KIND
    policy = INCUMBENT["shipped_policy"]
    dtype = {"BFLOAT16": ttnn.bfloat16, "BFLOAT8_B": ttnn.bfloat8_b}
    fidelity = {
        "HiFi4": ttnn.MathFidelity.HiFi4,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "LoFi": ttnn.MathFidelity.LoFi,
    }
    decoder = OptimizedDecoder.from_state_dict(
        synthetic_layer_state_dict(LAYER_IDX),
        hf_config=cfg,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        weight_dtype=dtype[policy["weight_dtype"]],
        activation_dtype=dtype[policy["activation_dtype"]],
        attention_weight_dtype=dtype[policy["attention_weight_dtype_by_layer_kind"][LAYER_KIND]],
        mlp_weight_dtype=dtype[policy["mlp_weight_dtype"]],
        mlp_down_weight_dtype=dtype[policy["mlp_down_weight_dtype"]],
        prefill_expert_weight_dtype=dtype[policy["prefill_expert_weight_dtype"]],
        expert_weight_dtype=dtype[policy["expert_weight_dtype"]],
        attention_math_fidelity=fidelity[policy["attention_math_fidelity"]],
        full_attention_math_fidelity=fidelity[policy["full_attention_math_fidelity"]],
        mlp_math_fidelity=fidelity[policy["mlp_math_fidelity"]],
        expert_gate_math_fidelity=fidelity[policy["expert_gate_math_fidelity"]],
        expert_math_fidelity=fidelity[policy["expert_math_fidelity"]],
        packed_dense_gate_up=policy["packed_dense_gate_up"],
        dram_in0_block_w=policy["dram_in0_block_w"],
        dram_sharded_roles=tuple(policy["dram_sharded_roles_by_layer_kind"][LAYER_KIND]),
        expert_decode_input_l1=policy["expert_decode_input_l1"],
        expert_gate_in0_block_w=policy["expert_gate_in0_block_w"],
        expert_down_in0_block_w=policy["expert_down_in0_block_w"],
        expert_gate_per_core_n=policy["expert_gate_per_core_n"],
        expert_down_per_core_n=policy["expert_down_per_core_n"],
    )

    generator = torch.Generator().manual_seed(20260730 + LAYER_IDX)
    decode_hidden = torch.randn(BATCH, 1, hidden_size, generator=generator, dtype=torch.bfloat16)
    rotary_width = 256 if LAYER_KIND == "sliding_attention" else 512
    decode_cos = torch.randn(BATCH, 1, rotary_width, generator=generator, dtype=torch.bfloat16)
    decode_sin = torch.randn(BATCH, 1, rotary_width, generator=generator, dtype=torch.bfloat16)
    if LAYER_KIND == "sliding_attention":
        tt_decode_cos = decode_cos.unsqueeze(0)
        tt_decode_sin = decode_sin.unsqueeze(0)
        shared_physical = True
    else:
        tt_decode_cos = decode_cos.transpose(0, 1).unsqueeze(0)
        tt_decode_sin = decode_sin.transpose(0, 1).unsqueeze(0)
        shared_physical = False

    token_capacity = CURRENT_POS + 1
    block_size, num_heads, head_dim = (
        (64, 8, 256) if shared_physical else (128, 2, 512)
    )
    one_user_shape = (
        (token_capacity + block_size - 1) // block_size,
        num_heads,
        block_size,
        head_dim,
    )
    blocks_per_user = one_user_shape[0]
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
        torch.arange(BATCH * blocks_per_user, dtype=torch.int32).view(BATCH, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (BATCH * one_user_shape[0], *one_user_shape[1:])
    kv_cache = (
        as_tt(torch.zeros(cache_shape, dtype=torch.bfloat16)),
        as_tt(torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    return (
        decoder,
        as_tt(decode_hidden.transpose(0, 1).unsqueeze(0)),
        as_tt(tt_decode_cos),
        as_tt(tt_decode_sin),
        as_tt(
            torch.full((BATCH,), CURRENT_POS, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        page_table,
        kv_cache,
    )


def decode(hidden):
    # Trace the shipped decode prefix through its dense MLP. Routed experts use
    # ttnn.sparse_matmul, which is terminal in the advisor tracer; the stage
    # records that uncapturable suffix and its 30/30-layer share separately.
    residual = hidden
    attn_in = _DECODER._rms_norm(hidden, _DECODER.weights.input_ln)
    attn_out = _DECODER._attention_decode(
        attn_in,
        position_cos=_POSITION_COS,
        position_sin=_POSITION_SIN,
        current_pos=_CURRENT_POS_TENSOR,
        page_table=_PAGE_TABLE,
        kv_cache=_KV_CACHE,
        cache_position_modulo=None,
    )
    attn_out = _DECODER._rms_norm(attn_out, _DECODER.weights.post_attn_ln)
    hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mlp_in = _DECODER._rms_norm(hidden, _DECODER.weights.pre_ff_ln)
    mlp_out = _DECODER._dense_mlp(mlp_in)
    return _DECODER._rms_norm(mlp_out, _DECODER.weights.post_ff_ln_1)


def make_inputs(device):
    global _DECODER, _POSITION_COS, _POSITION_SIN
    global _CURRENT_POS_TENSOR, _PAGE_TABLE, _KV_CACHE
    (
        _DECODER,
        hidden,
        _POSITION_COS,
        _POSITION_SIN,
        _CURRENT_POS_TENSOR,
        _PAGE_TABLE,
        _KV_CACHE,
    ) = _build(device)
    return (hidden,)
