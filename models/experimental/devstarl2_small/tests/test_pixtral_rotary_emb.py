# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF PixtralRotaryEmbedding vs TtPixtralRotaryEmbedding.

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_pixtral_rotary_emb import TtPixtralRotaryEmbedding

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _normalize_ref_shape(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 2 else x


def _tt_to_torch_3d(tt_tensor: ttnn.Tensor, mesh_device) -> torch.Tensor:
    out = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    if out.dim() >= 4:
        out = out[0]
    if out.dim() == 3 and out.shape[0] != 1:
        out = out[:1]
    while out.dim() > 3:
        out = out.squeeze(0)
    return out


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_pixtral_rotary_embedding_pcc(mesh_device, seq_len):
    hf_cfg = AutoConfig.from_pretrained(
        DEVSTRAL_REPO_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    vision_cfg = hf_cfg.vision_config
    hf_rope = PixtralRotaryEmbedding(vision_cfg).eval()

    head_dim = int(getattr(vision_cfg, "head_dim", vision_cfg.hidden_size // vision_cfg.num_attention_heads))
    x_ref = torch.zeros(1, seq_len, vision_cfg.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos_ref, sin_ref = hf_rope(x_ref, position_ids)
    cos_ref = _normalize_ref_shape(cos_ref).float()
    sin_ref = _normalize_ref_shape(sin_ref).float()

    tt_rope = TtPixtralRotaryEmbedding(mesh_device=mesh_device, config=vision_cfg, datatype=ttnn.bfloat16)
    x_tt = ttnn.from_torch(
        torch.zeros(1, seq_len, head_dim, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pos_tt = ttnn.from_torch(
        position_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    cos_tt, sin_tt = tt_rope(x_tt, pos_tt)
    cos_tt = _tt_to_torch_3d(cos_tt, mesh_device).float()
    sin_tt = _tt_to_torch_3d(sin_tt, mesh_device).float()

    cos_tt = cos_tt[:, : cos_ref.shape[1], : cos_ref.shape[2]]
    sin_tt = sin_tt[:, : sin_ref.shape[1], : sin_ref.shape[2]]

    pcc_required = 0.999
    cos_ok, cos_msg = comp_pcc(cos_ref, cos_tt, pcc_required)
    sin_ok, sin_msg = comp_pcc(sin_ref, sin_tt, pcc_required)
    logger.info(comp_allclose(cos_ref, cos_tt))
    logger.info(comp_allclose(sin_ref, sin_tt))
    logger.info(f"cos PCC: {cos_msg}")
    logger.info(f"sin PCC: {sin_msg}")
    assert cos_ok, f"cos PCC below {pcc_required}: {cos_msg}"
    assert sin_ok, f"sin PCC below {pcc_required}: {sin_msg}"
