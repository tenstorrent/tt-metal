# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC test: TT ``TtRMSNorm`` vs HF ``Ministral3RMSNorm`` on Devstral-2-123B layer-0 weights.

Uses the real Hugging Face config and downloads only ``model.layers.0.input_layernorm.weight``
(one safetensor shard) via :func:`require_hf_weights` (pytest skip if download fails).
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.modeling_ministral3 import Ministral3RMSNorm

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tests._devstral_weights import (
    DEVSTRAL2_TEST_MAX_SEQ_LEN,
    require_hf_weights,
    require_text_config,
    replicated_tt_to_torch,
)
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.mem_config import get_decode_width_sharded_activation_mem_config
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import TtRMSNorm

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
    [{"l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_rmsnorm_pcc_real_weights(mesh_device, seq_len):
    text_cfg = require_text_config()
    weight_key = "model.layers.0.input_layernorm.weight"
    state_dict = require_hf_weights([weight_key])
    ref = Ministral3RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps).to(torch.bfloat16).eval()
    ref.weight.data.copy_(state_dict[weight_key])
    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max(DEVSTRAL2_TEST_MAX_SEQ_LEN, seq_len),
    )
    tt_norm = TtRMSNorm(args, mesh_device, state_dict, weight_key)

    x = torch.randn(1, 1, seq_len, text_cfg.hidden_size, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_norm(tt_x, memory_config=ttnn.DRAM_MEMORY_CONFIG, mode="prefill")
    tt_torch = replicated_tt_to_torch(tt_out)

    ref_out = ref(x)
    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_rmsnorm_pcc_prefill_width_sharded(mesh_device, seq_len):
    """Prefill-shaped activations (M=128) with width-sharded RMSNorm."""
    text_cfg = require_text_config()
    weight_key = "model.layers.0.input_layernorm.weight"
    state_dict = require_hf_weights([weight_key])
    ref = Ministral3RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps).to(torch.bfloat16).eval()
    ref.weight.data.copy_(state_dict[weight_key])
    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max(DEVSTRAL2_TEST_MAX_SEQ_LEN, seq_len),
    )
    tt_norm = TtRMSNorm(args, mesh_device, state_dict, weight_key)

    x = torch.randn(1, 1, seq_len, text_cfg.hidden_size, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_norm(tt_x, memory_config=ttnn.DRAM_MEMORY_CONFIG, mode="prefill")
    tt_torch = replicated_tt_to_torch(tt_out)

    ref_out = ref(x)
    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(f"prefill width-sharded PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_rmsnorm_pcc_decode_width_sharded(mesh_device):
    """Decode-shaped activations (M=32) with width-sharded RMSNorm."""
    text_cfg = require_text_config()
    weight_key = "model.layers.0.input_layernorm.weight"
    state_dict = require_hf_weights([weight_key])
    ref = Ministral3RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps).to(torch.bfloat16).eval()
    ref.weight.data.copy_(state_dict[weight_key])
    args = Devstral2Args.from_hf_config(text_cfg, mesh_shape=tuple(mesh_device.shape))
    tt_norm = TtRMSNorm(args, mesh_device, state_dict, weight_key)

    m = ttnn.TILE_SIZE
    x = torch.randn(1, 1, m, text_cfg.hidden_size, dtype=torch.bfloat16)
    sharded_mem = get_decode_width_sharded_activation_mem_config(args.hidden_size)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_x = ttnn.to_memory_config(tt_x, sharded_mem)
    tt_out = tt_norm(tt_x, mode="decode")
    tt_torch = replicated_tt_to_torch(tt_out)

    ref_out = ref(x)
    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(f"decode width-sharded PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"
