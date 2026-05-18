# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC: TT ``TtDecoderLayer`` vs HF ``Ministral3DecoderLayer`` (prefill, random weights)."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3DecoderLayer,
    Ministral3RotaryEmbedding,
)

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.tt_ministral3_decoder_layer import TtDecoderLayer
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import TtRotaryEmbedding
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99


def _mesh_shape_from_env() -> tuple[int, int]:
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "P150x4": (1, 4),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE", "P150x4"), (1, 4))


def _extract_state_dict(ref: Ministral3DecoderLayer, layer_idx: int) -> dict:
    p = f"model.layers.{layer_idx}"
    a = ref.self_attn
    m = ref.mlp
    return {
        f"{p}.input_layernorm.weight": ref.input_layernorm.weight.data.clone(),
        f"{p}.post_attention_layernorm.weight": ref.post_attention_layernorm.weight.data.clone(),
        f"{p}.self_attn.q_proj.weight": a.q_proj.weight.data.clone(),
        f"{p}.self_attn.k_proj.weight": a.k_proj.weight.data.clone(),
        f"{p}.self_attn.v_proj.weight": a.v_proj.weight.data.clone(),
        f"{p}.self_attn.o_proj.weight": a.o_proj.weight.data.clone(),
        f"{p}.mlp.gate_proj.weight": m.gate_proj.weight.data.clone(),
        f"{p}.mlp.up_proj.weight": m.up_proj.weight.data.clone(),
        f"{p}.mlp.down_proj.weight": m.down_proj.weight.data.clone(),
    }


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_decoder_layer_prefill_pcc_random_weights(mesh_device, seq_len):
    hidden_size = 1024
    cfg = Ministral3Config(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,  # must divide TP (mesh dim); Quietbox=4.
        head_dim=hidden_size // 8,
        max_position_embeddings=4096,
    )

    torch.manual_seed(0)
    ref = Ministral3DecoderLayer(cfg, layer_idx=0).to(dtype=torch.bfloat16).eval()
    ref_rope = Ministral3RotaryEmbedding(cfg).eval()

    state_dict = _extract_state_dict(ref, layer_idx=0)
    args = Devstral2Args.from_hf_config(cfg, mesh_shape=tuple(mesh_device.shape), max_seq_len=max(512, seq_len))
    tt_ccl = TT_CCL(mesh_device)
    rope = TtRotaryEmbedding(args, mesh_device, max_position_embeddings=args.max_seq_len)
    tt_layer = TtDecoderLayer(args, mesh_device, state_dict, layer_idx=0, tt_ccl=tt_ccl, rotary_emb=rope)

    x = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(seq_len).unsqueeze(0)
    cos_ref, sin_ref = ref_rope(x.unsqueeze(1).float(), positions)
    # ``Ministral3DecoderLayer.forward`` forwards ``attention_mask`` straight to attention; the
    # parent ``Ministral3Model`` would normally build a causal mask. Build it explicitly so HF
    # matches the TT prefill path (``is_causal=True`` inside ``scaled_dot_product_attention``).
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16), diagonal=1).reshape(
        1, 1, seq_len, seq_len
    )
    ref_out = ref(
        x,
        attention_mask=causal_mask,
        position_ids=positions,
        position_embeddings=(cos_ref.to(torch.bfloat16), sin_ref.to(torch.bfloat16)),
    )

    tt_x = ttnn.from_torch(
        x.reshape(1, 1, seq_len, hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_layer(tt_x, mode="prefill", start_pos=0)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]
    tt_torch = tt_torch.reshape(1, seq_len, hidden_size)

    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"
