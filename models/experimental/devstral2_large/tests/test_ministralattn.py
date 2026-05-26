# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC: TT ``TtAttention`` vs HF ``Ministral3Attention`` on Devstral-2-123B layer-0 weights.

Covers prefill (full sequence, KV cache populated). Decode is exercised by the decoder-layer
and model tests.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3Attention,
    Ministral3RotaryEmbedding,
)

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tests._devstral_weights import (
    DEVSTRAL2_TEST_MAX_SEQ_LEN,
    require_attention_weights,
    require_text_config,
    replicated_tt_to_torch,
)
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import TtRotaryEmbedding
from models.experimental.devstral2_large.tt.tt_ministralattn import TtAttention
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99


def _mesh_shape_from_env() -> tuple[int, int]:
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "P150x4": (1, 4),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE", "T3K"), (1, 8))


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_attention_prefill_pcc_real_weights(mesh_device, seq_len):
    text_cfg = require_text_config()
    layer = 0
    state_dict = require_attention_weights(layer)

    ref = Ministral3Attention(text_cfg, layer_idx=layer).to(torch.bfloat16).eval()
    ref.q_proj.weight.data.copy_(state_dict[f"model.layers.{layer}.self_attn.q_proj.weight"])
    ref.k_proj.weight.data.copy_(state_dict[f"model.layers.{layer}.self_attn.k_proj.weight"])
    ref.v_proj.weight.data.copy_(state_dict[f"model.layers.{layer}.self_attn.v_proj.weight"])
    ref.o_proj.weight.data.copy_(state_dict[f"model.layers.{layer}.self_attn.o_proj.weight"])

    ref_rope = Ministral3RotaryEmbedding(text_cfg).eval()

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max(DEVSTRAL2_TEST_MAX_SEQ_LEN, seq_len),
    )
    tt_ccl = TT_CCL(mesh_device)
    rope = TtRotaryEmbedding(args, mesh_device, max_position_embeddings=args.max_seq_len)
    tt_attn = TtAttention(args, mesh_device, state_dict, layer_idx=layer, tt_ccl=tt_ccl, rotary_emb=rope)

    x = torch.randn(1, seq_len, text_cfg.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(seq_len).unsqueeze(0)
    cos_ref, sin_ref = ref_rope(x.unsqueeze(1).float(), positions)
    # ``Ministral3Attention.forward`` does not generate a causal mask internally — that's the
    # parent ``Ministral3Model``'s job. Build it explicitly so HF matches the TT prefill path
    # (``ttnn.transformer.scaled_dot_product_attention(..., is_causal=True)``).
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16), diagonal=1).reshape(
        1, 1, seq_len, seq_len
    )
    ref_out, _ = ref(
        hidden_states=x,
        position_embeddings=(cos_ref.to(torch.bfloat16), sin_ref.to(torch.bfloat16)),
        attention_mask=causal_mask,
        position_ids=positions,
    )

    tt_x = ttnn.from_torch(
        x.reshape(1, 1, seq_len, text_cfg.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_attn(tt_x, mode="prefill", start_pos=0, user_id=0)
    tt_torch = replicated_tt_to_torch(tt_out, reshape=(1, seq_len, text_cfg.hidden_size))

    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"
