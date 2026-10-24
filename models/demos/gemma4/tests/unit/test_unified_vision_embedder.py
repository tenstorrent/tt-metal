# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device Gemma-4 unified (encoder-free) vision embedder vs the HF reference."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.tt.vision.unified_embedder import PREFIX, PROJECTION_KEY, Gemma4UnifiedVisionEmbedder


def _hf_embedder():
    from transformers import AutoConfig

    try:
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedMultimodalEmbedder
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedVisionEmbedder as HFVisionEmbedder,
        )
    except ImportError:  # transformers without the unified Gemma 4 classes
        pytest.skip("transformers has no gemma4_unified module")

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-12B-it")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(getattr(cfg, "vision_config", None), "model_patch_size", None) is None:
        # Tower-style (SigLIP) config, e.g. the 31B or the checked-in config-only
        # dirs: the encoder-free embedder does not apply.
        pytest.skip(f"{model_path} is not a unified (encoder-free) Gemma 4 checkpoint")
    torch.manual_seed(0)
    vis = HFVisionEmbedder(cfg.vision_config, cfg.text_config)
    # Random init: the checkpoint's constructor leaves the position table at zero
    # and LayerNorms at identity, which would not exercise the lookup / affine paths.
    with torch.no_grad():
        for p in vis.parameters():
            p.normal_(0.0, 0.05)
    assert isinstance(vis.multimodal_embedder, Gemma4UnifiedMultimodalEmbedder)
    return vis, cfg


def _state_dict_from_hf(vis):
    sd = {}
    for k, v in vis.state_dict().items():
        if k.startswith("multimodal_embedder.embedding_projection."):
            sd[PROJECTION_KEY] = v.detach().clone()
        else:
            sd[PREFIX + k] = v.detach().clone()
    return sd


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_valid, num_patches", [(280, 280), (150, 288)], ids=["full", "padded"])
def test_unified_vision_embedder(mesh_device, num_valid, num_patches):
    vis, cfg = _hf_embedder()
    patch_dim = cfg.vision_config.model_patch_size**2 * 3
    torch.manual_seed(1)
    pixel_values = torch.randn(1, num_patches, patch_dim)
    side = int(cfg.vision_config.mm_posemb_size)
    pos = torch.stack(
        [torch.randint(0, side, (num_patches,)), torch.randint(0, side, (num_patches,))], dim=-1
    ).unsqueeze(0)
    pos[0, num_valid:] = -1  # padding patches

    with torch.no_grad():
        ref = vis(pixel_values.to(torch.bfloat16), pos).float()[0]  # [P, text_hidden]

    tt = Gemma4UnifiedVisionEmbedder(
        mesh_device,
        _state_dict_from_hf(vis),
        text_hidden_size=cfg.text_config.hidden_size,
    )
    out_tt, mask = tt.forward(pixel_values, pos)
    out = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if hasattr(mesh_device, "shape") else None
    )
    out = out[0, 0, :num_patches].float() if out.dim() == 4 else out[:num_patches].float()

    assert mask.shape == (1, out_tt.shape[2])
    assert int(mask.sum()) == num_valid
    passing, pcc = comp_pcc(ref[:num_valid], out[:num_valid], 0.99)
    logger.info(f"unified vision embedder PCC over {num_valid} valid patches: {pcc}")
    assert passing, f"PCC {pcc} < 0.99"
