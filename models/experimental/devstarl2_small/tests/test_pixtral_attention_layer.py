# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF PixtralAttentionLayer vs TtPixtralAttentionLayer.

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.pixtral.modeling_pixtral import PixtralAttentionLayer, PixtralRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_pixtral_attention_layer import TtPixtralAttentionLayer
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import (
    convert_vision_hf_to_meta,
    load_hf_state_dict_filtered,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
VISION_LAYER0_PREFIX = "vision_tower.transformer.layers.0."


def _load_layer0_vision_tensors(repo_id: str) -> dict:
    return load_hf_state_dict_filtered(repo_id, (VISION_LAYER0_PREFIX,), local_files_only=os.getenv("CI") == "true")


def _vision_layer0_hf_to_meta(raw: dict, text_head_dim: int) -> dict:
    return convert_vision_hf_to_meta(standardize_hf_keys_multimodal(raw), text_head_dim)


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
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_pixtral_attention_layer_pcc_devstral_weights(mesh_device, seq_len, batch_size, monkeypatch):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max(256, seq_len))

    hf_cfg = AutoConfig.from_pretrained(
        DEVSTRAL_REPO_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    text_head_dim = hf_cfg.text_config.hidden_size // hf_cfg.text_config.num_attention_heads

    hf_layer_sd = _load_layer0_vision_tensors(DEVSTRAL_REPO_ID)
    meta_state = _vision_layer0_hf_to_meta(hf_layer_sd, text_head_dim)

    hf_layer = PixtralAttentionLayer(hf_cfg.vision_config).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(
        {k[len(VISION_LAYER0_PREFIX) :]: v for k, v in hf_layer_sd.items() if k.startswith(VISION_LAYER0_PREFIX)},
        strict=True,
    )
    hf_rope = PixtralRotaryEmbedding(hf_cfg.vision_config).eval()

    tt_layer = TtPixtralAttentionLayer(
        mesh_device=mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=meta_state,
        state_dict_prefix=VISION_LAYER0_PREFIX,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
    )

    hidden_size = model_args.vision_dim
    pt_in = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos_ref, sin_ref = hf_rope(pt_in, position_ids)

    ref_out = hf_layer(pt_in, attention_mask=attn_mask, position_embeddings=(cos_ref, sin_ref))

    x_tt = ttnn.from_torch(
        pt_in.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cos_tt = ttnn.from_torch(
        cos_ref,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin_ref,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_layer(x_tt, attention_mask=None, position_embeddings=(cos_tt, sin_tt))
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_torch = tt_torch[:, :, :, :hidden_size].squeeze(0)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
