# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC: TT ``TtMinistral3Model`` vs HF ``Ministral3Model`` end-to-end (random weights, small config)."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99


def _mesh_shape_from_env() -> tuple[int, int]:
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "P150x4": (1, 4),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE", "P150x4"), (1, 4))


def _hf_state_dict(ref: Ministral3Model) -> dict:
    sd = ref.state_dict()
    # Re-prefix HF top-level (``embed_tokens``, ``norm``, ``layers``) under ``model.`` to match
    # the keys our TT modules expect.
    return {f"model.{k}": v.clone() for k, v in sd.items()}


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("seq_len", [32])
@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_full_model_prefill_pcc_random_weights(mesh_device, seq_len, num_layers):
    hidden_size = 512
    cfg = Ministral3Config(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=4,  # must divide TP (mesh dim); Quietbox=4.
        head_dim=hidden_size // 8,
        vocab_size=1024,
        max_position_embeddings=4096,
    )

    torch.manual_seed(0)
    ref = Ministral3Model(cfg).to(dtype=torch.bfloat16).eval()
    state_dict = _hf_state_dict(ref)

    args = Devstral2Args.from_hf_config(cfg, mesh_shape=tuple(mesh_device.shape), max_seq_len=max(512, seq_len))
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtMinistral3Model(args, mesh_device, state_dict, tt_ccl, num_layers=num_layers)

    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.long)
    ref_out = ref(input_ids=input_ids).last_hidden_state

    tt_out = tt_model(input_ids, mode="prefill", start_pos=0)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]
    tt_torch = tt_torch.reshape(1, seq_len, hidden_size)

    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"
