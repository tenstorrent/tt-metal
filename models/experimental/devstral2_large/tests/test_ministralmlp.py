# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC: TT ``TtMLP`` vs HF ``Ministral3MLP`` on Devstral-2-123B layer-0 weights."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.modeling_ministral3 import Ministral3MLP

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tests._devstral_weights import (
    load_hf_tensors_for_keys,
    load_text_config,
    to_bf16_host_if_fp8,
)
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.tt_ministralmlp import TtMLP
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99


def _mesh_shape_from_env() -> tuple[int, int]:
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "P150x4": (1, 4),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE", "P150x4"), (1, 4))


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_mlp_pcc_real_weights(mesh_device, seq_len):
    try:
        text_cfg = load_text_config()
    except Exception as exc:
        pytest.skip(f"Could not load Devstral-2-123B HF config: {exc}")

    layer = 0
    keys = [
        f"model.layers.{layer}.mlp.gate_proj.weight",
        f"model.layers.{layer}.mlp.up_proj.weight",
        f"model.layers.{layer}.mlp.down_proj.weight",
    ]
    try:
        raw = load_hf_tensors_for_keys(keys)
    except Exception as exc:
        pytest.skip(f"Could not download Devstral-2-123B layer-0 MLP weights: {exc}")
    state_dict = {k: to_bf16_host_if_fp8(v).to(torch.bfloat16) for k, v in raw.items()}

    ref = Ministral3MLP(text_cfg).to(torch.bfloat16).eval()
    ref.gate_proj.weight.data.copy_(state_dict[keys[0]])
    ref.up_proj.weight.data.copy_(state_dict[keys[1]])
    ref.down_proj.weight.data.copy_(state_dict[keys[2]])

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max(512, seq_len),
    )
    tt_ccl = TT_CCL(mesh_device)
    tt_mlp = TtMLP(args, mesh_device, state_dict, layer_idx=layer, tt_ccl=tt_ccl)

    x = torch.randn(1, 1, seq_len, text_cfg.hidden_size, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_mlp(tt_x)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    ref_out = ref(x)
    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"
