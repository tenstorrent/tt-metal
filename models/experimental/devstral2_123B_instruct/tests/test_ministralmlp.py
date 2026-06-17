# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TT ``TtMLP`` vs HF ``Ministral3MLP`` on Devstral-2-123B layer-0 weights.

Uses prefill width-sharded activations (same layout as post-attention RMSNorm at ``seq_len <= kv_block_size``).
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.modeling_ministral3 import Ministral3MLP

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    devstral2_test_model_args,
    devstral2_tt_weight_cache_dir,
    log_tt_weight_cache_status,
    require_mlp_weights,
    require_text_config,
    replicated_tt_to_torch,
)
from models.experimental.devstral2_123B_instruct.tt.mem_config import (
    get_prefill_width_sharded_activation_mem_config,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministralmlp import TtMLP
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
@pytest.mark.timeout(3600)
def test_mlp_pcc_real_weights(mesh_device, seq_len):
    text_cfg = require_text_config()
    layer = 0
    state_dict = require_mlp_weights(layer)

    ref = Ministral3MLP(text_cfg).to(torch.bfloat16).eval()
    ref.gate_proj.weight.data.copy_(state_dict[f"model.layers.{layer}.mlp.gate_proj.weight"])
    ref.up_proj.weight.data.copy_(state_dict[f"model.layers.{layer}.mlp.up_proj.weight"])
    ref.down_proj.weight.data.copy_(state_dict[f"model.layers.{layer}.mlp.down_proj.weight"])

    args = devstral2_test_model_args(text_cfg, mesh_device)
    weight_cache_path = devstral2_tt_weight_cache_dir(mesh_device, text_cfg)
    log_tt_weight_cache_status(weight_cache_path, int(text_cfg.num_hidden_layers))
    tt_ccl = TT_CCL(mesh_device)
    tt_mlp = TtMLP(args, mesh_device, state_dict, layer_idx=layer, tt_ccl=tt_ccl, weight_cache_path=weight_cache_path)

    x = torch.randn(1, 1, seq_len, text_cfg.hidden_size, dtype=torch.bfloat16)
    ws_mem = get_prefill_width_sharded_activation_mem_config(seq_len, text_cfg.hidden_size)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_x = ttnn.interleaved_to_sharded(tt_x, ws_mem)
    tt_out = tt_mlp(tt_x, mode="prefill")
    tt_torch = replicated_tt_to_torch(tt_out)

    ref_out = ref(x)
    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"
