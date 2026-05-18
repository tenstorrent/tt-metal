# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: truncated HF ``PixtralVisionModel`` vs ``TtPixtralVisionModel`` (first ``n_layers`` blocks).

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.pixtral.modeling_pixtral import (
    PixtralVisionModel,
    generate_block_attention_mask,
    position_ids_in_meshgrid,
)
from transformers.utils.generic import is_flash_attention_requested

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_pixtral_vision_model import TtPixtralVisionModel
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import (
    convert_vision_hf_to_meta,
    load_hf_state_dict_filtered,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _prefixes(n_layers: int) -> tuple[str, ...]:
    return ("vision_tower.patch_conv.", "vision_tower.ln_pre.") + tuple(
        f"vision_tower.transformer.layers.{i}." for i in range(n_layers)
    )


def _to_meta(raw: dict, text_head_dim: int) -> dict:
    return convert_vision_hf_to_meta(standardize_hf_keys_multimodal(raw), text_head_dim)


def _max_patch_grid_side(cfg):
    iz = cfg.image_size
    if isinstance(iz, (tuple, list)):
        iz = iz[0]
    return iz // cfg.patch_size


def _hf_vision_forward_truncated(hf_vm: PixtralVisionModel, pixel_values, image_sizes, n_layers: int, **kwargs):
    cfg = hf_vm.config
    patch_sz = hf_vm.patch_size
    target_dtype = hf_vm.patch_conv.weight.dtype
    patch_embeds = hf_vm.patch_conv(pixel_values.to(dtype=target_dtype))
    patch_embeds_list = [
        embed[..., : (sz[0] // patch_sz), : (sz[1] // patch_sz)] for embed, sz in zip(patch_embeds, image_sizes)
    ]
    patch_embeds_cat = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)
    patch_embeds_cat = hf_vm.ln_pre(patch_embeds_cat)
    position_ids = position_ids_in_meshgrid(patch_embeds_list, max_width=_max_patch_grid_side(cfg))
    kwargs = {
        **kwargs,
        "position_ids": position_ids.unsqueeze(0).to(patch_embeds_cat.device, non_blocking=True),
    }
    position_embeddings = hf_vm.patch_positional_embedding(patch_embeds_cat, kwargs["position_ids"])
    if is_flash_attention_requested(cfg):
        attention_mask = None
    else:
        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds_cat
        )
    hidden = patch_embeds_cat
    for i in range(n_layers):
        hidden = hf_vm.transformer.layers[i](
            hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    return hidden


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
@pytest.mark.parametrize("n_layers", (2,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_pixtral_vision_model_pcc_devstral_weights(mesh_device, n_layers, monkeypatch):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)

    hf_cfg = AutoConfig.from_pretrained(
        DEVSTRAL_REPO_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    vision_cfg = hf_cfg.vision_config
    text_head_dim = hf_cfg.text_config.hidden_size // hf_cfg.text_config.num_attention_heads

    hf_sd = load_hf_state_dict_filtered(
        DEVSTRAL_REPO_ID, _prefixes(n_layers), local_files_only=os.getenv("CI") == "true"
    )
    meta_state = _to_meta(hf_sd, text_head_dim)

    hf_vm = PixtralVisionModel(vision_cfg).to(torch.bfloat16).eval()
    sd_vm = {k[len("vision_tower.") :]: v for k, v in hf_sd.items() if k.startswith("vision_tower.")}
    hf_vm.load_state_dict(sd_vm, strict=False)

    patch_sz = int(vision_cfg.patch_size)
    H = W = patch_sz * 32
    pixel_values = torch.randn(1, 3, H, W, dtype=torch.bfloat16)
    image_sizes = [(H, W)]

    target_dtype = hf_vm.patch_conv.weight.dtype
    pe_conv = hf_vm.patch_conv(pixel_values.to(dtype=target_dtype))
    plist = [e[..., : s[0] // patch_sz, : s[1] // patch_sz] for e, s in zip(pe_conv, image_sizes)]
    position_ids = position_ids_in_meshgrid(plist, max_width=_max_patch_grid_side(vision_cfg))
    pos_tt = ttnn.from_torch(
        position_ids.unsqueeze(0).to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    ref = _hf_vision_forward_truncated(hf_vm, pixel_values, image_sizes, n_layers)

    tt_vm = TtPixtralVisionModel(
        mesh_device=mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=meta_state,
        configuration=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        vision_config=vision_cfg,
        n_layers=n_layers,
    )
    tt_out = tt_vm(pixel_values, image_sizes, position_ids_tt=pos_tt)

    hidden = model_args.vision_dim
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_torch = tt_torch[:, :, :, :hidden].squeeze(0)

    pcc_required = 0.99
    passing, msg = comp_pcc(ref, tt_torch, pcc_required)
    logger.info(comp_allclose(ref, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {pcc_required}: {msg}"
