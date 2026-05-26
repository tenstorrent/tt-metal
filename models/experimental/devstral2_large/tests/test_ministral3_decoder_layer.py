# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC: TT ``TtDecoderLayer`` vs HF ``Ministral3DecoderLayer`` on Devstral-2-123B layer-0 weights."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3DecoderLayer,
    Ministral3RotaryEmbedding,
)

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tests._devstral_weights import (
    DEVSTRAL2_TEST_MAX_SEQ_LEN,
    load_ministral3_decoder_layer_weights,
    require_layer_weights,
    require_text_config,
    replicated_tt_to_torch,
)
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
    }.get(os.environ.get("MESH_DEVICE", "T3K"), (1, 8))


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_decoder_layer_prefill_pcc_real_weights(mesh_device, seq_len):
    text_cfg = require_text_config()
    layer_idx = 0
    state_dict = require_layer_weights(layer_idx)

    ref = Ministral3DecoderLayer(text_cfg, layer_idx=layer_idx).to(torch.bfloat16).eval()
    load_ministral3_decoder_layer_weights(ref, state_dict, layer_idx)
    ref_rope = Ministral3RotaryEmbedding(text_cfg).eval()

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max(DEVSTRAL2_TEST_MAX_SEQ_LEN, seq_len),
    )
    tt_ccl = TT_CCL(mesh_device)
    rope = TtRotaryEmbedding(args, mesh_device, max_position_embeddings=args.max_seq_len)
    tt_layer = TtDecoderLayer(args, mesh_device, state_dict, layer_idx=layer_idx, tt_ccl=tt_ccl, rotary_emb=rope)

    x = torch.randn(1, seq_len, text_cfg.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(seq_len).unsqueeze(0)
    cos_ref, sin_ref = ref_rope(x.unsqueeze(1).float(), positions)
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
        x.reshape(1, 1, seq_len, text_cfg.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_layer(tt_x, mode="prefill", start_pos=0)
    tt_torch = replicated_tt_to_torch(tt_out, reshape=(1, seq_len, text_cfg.hidden_size))

    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"
