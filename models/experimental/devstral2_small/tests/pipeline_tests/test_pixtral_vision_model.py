# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF PixtralVisionModel + projector vs TT PixtralVisionModel + projector.

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
from transformers.models.pixtral.modeling_pixtral import (
    PixtralVisionModel,
    generate_block_attention_mask,
    position_ids_in_meshgrid,
)
from transformers.utils.generic import is_flash_attention_requested

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_small.tt.pipeline.tt_multimodal_projector import TTMistral3MultiModalProjector
from models.experimental.devstral2_small.tt.pipeline.tt_pixtral_vision_model import TtPixtralVisionModel
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import (
    convert_vision_hf_to_meta,
    load_hf_state_dict_filtered,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _prefixes(n_layers: int) -> tuple[str, ...]:
    return (
        "vision_tower.patch_conv.",
        "vision_tower.ln_pre.",
        "multi_modal_projector.",
        "model.multi_modal_projector.",
    ) + tuple(f"vision_tower.transformer.layers.{i}." for i in range(n_layers))


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
    [{"P150": (1, 1), "BH-QB": (1, 4)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("n_layers", (24,))
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
    standardized_sd = standardize_hf_keys_multimodal(hf_sd)
    meta_state = convert_vision_hf_to_meta(standardized_sd, text_head_dim)

    hf_vm = PixtralVisionModel(vision_cfg).to(torch.bfloat16).eval()
    sd_vm = {k[len("vision_tower.") :]: v for k, v in standardized_sd.items() if k.startswith("vision_tower.")}
    hf_vm.load_state_dict(sd_vm, strict=False)
    hf_projector = Mistral3MultiModalProjector(hf_cfg).to(torch.bfloat16).eval()
    sd_projector = {
        k[len("multi_modal_projector.") :]: v
        for k, v in standardized_sd.items()
        if k.startswith("multi_modal_projector.")
    }
    hf_projector.load_state_dict(sd_projector, strict=False)

    patch_sz = int(vision_cfg.patch_size)
    H = W = patch_sz * 32
    pixel_values = torch.randn(1, 3, H, W, dtype=torch.bfloat16)
    image_sizes = [(H, W)]

    target_dtype = hf_vm.patch_conv.weight.dtype
    pe_conv = hf_vm.patch_conv(pixel_values.to(dtype=target_dtype))
    plist = [e[..., : s[0] // patch_sz, : s[1] // patch_sz] for e, s in zip(pe_conv, image_sizes)]
    position_ids = position_ids_in_meshgrid(plist, max_width=_max_patch_grid_side(vision_cfg))

    ref_hidden = _hf_vision_forward_truncated(hf_vm, pixel_values, image_sizes, n_layers)
    ref = hf_projector(ref_hidden.squeeze(0), torch.tensor(image_sizes, dtype=torch.long))

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
    tt_out = tt_vm(pixel_values, image_sizes, position_ids=position_ids)
    seq_len, hidden = int(tt_out.shape[2]), int(tt_out.shape[3])
    tt_tokens = ttnn.reshape(tt_out, (seq_len, hidden))
    if tt_tokens.memory_config().buffer_type == ttnn.BufferType.L1:
        tt_tokens = ttnn.to_memory_config(tt_tokens, ttnn.DRAM_MEMORY_CONFIG)

    tt_projector = TTMistral3MultiModalProjector(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=meta_state,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        eps=float(hf_projector.norm.variance_epsilon),
    )
    tt_features = tt_projector(tt_tokens, image_sizes)
    tt_torch = ttnn.to_torch(tt_features, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
    while tt_torch.dim() > 2:
        tt_torch = tt_torch.squeeze(0)
    tt_torch = tt_torch[: ref.shape[0], : ref.shape[1]]

    pcc_required = 0.99
    passing, msg = comp_pcc(ref.float(), tt_torch, pcc_required)
    logger.info(comp_allclose(ref.float(), tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {pcc_required}: {msg}"
