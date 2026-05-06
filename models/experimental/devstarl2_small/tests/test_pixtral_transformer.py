# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: HF ``PixtralTransformer`` (first N layers) vs ``TtPixtralTransformer``."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from torch import nn
from transformers import AutoConfig
from transformers.models.pixtral.modeling_pixtral import PixtralAttentionLayer, PixtralRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_pixtral_transformer import TtPixtralTransformer
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import (
    convert_vision_hf_to_meta,
    load_hf_state_dict_filtered,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
VISION_LAYERS_PREFIX = "vision_tower.transformer.layers."


def _load_vision_layer_prefixes(repo_id: str, n_layers: int) -> dict:
    prefixes = tuple(f"{VISION_LAYERS_PREFIX}{i}." for i in range(n_layers))
    return load_hf_state_dict_filtered(repo_id, prefixes, local_files_only=os.getenv("CI") == "true")


def _to_meta(raw: dict, text_head_dim: int) -> dict:
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
@pytest.mark.parametrize("n_layers", (2,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_pixtral_transformer_pcc_devstral_weights(mesh_device, seq_len, batch_size, n_layers, monkeypatch):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max(256, seq_len))

    hf_cfg = AutoConfig.from_pretrained(
        DEVSTRAL_REPO_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    vision_cfg = hf_cfg.vision_config
    text_head_dim = hf_cfg.text_config.hidden_size // hf_cfg.text_config.num_attention_heads

    hf_sd = _load_vision_layer_prefixes(DEVSTRAL_REPO_ID, n_layers)
    meta_state = _to_meta(hf_sd, text_head_dim)

    hf_blocks = nn.ModuleList([PixtralAttentionLayer(vision_cfg).to(torch.bfloat16).eval() for _ in range(n_layers)])
    for i in range(n_layers):
        pref = f"{VISION_LAYERS_PREFIX}{i}."
        hf_blocks[i].load_state_dict({k[len(pref) :]: v for k, v in hf_sd.items() if k.startswith(pref)}, strict=True)

    hf_rope = PixtralRotaryEmbedding(vision_cfg).eval()

    tt_tf = TtPixtralTransformer(
        mesh_device=mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=meta_state,
        configuration=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        n_layers=n_layers,
    )

    hidden_size = model_args.vision_dim
    pt_in = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos_ref, sin_ref = hf_rope(pt_in, position_ids)

    ref = pt_in
    for block in hf_blocks:
        ref = block(ref, attention_mask=attn_mask, position_embeddings=(cos_ref, sin_ref))

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

    tt_out = tt_tf(x_tt, attention_mask=None, position_embeddings=(cos_tt, sin_tt))
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_torch = tt_torch[:, :, :, :hidden_size].squeeze(0)

    pcc_required = 0.99
    passing, msg = comp_pcc(ref, tt_torch, pcc_required)
    logger.info(comp_allclose(ref, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {pcc_required}: {msg}"
