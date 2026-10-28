# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device Gemma-4 unified (encoder-free) audio embedder vs the HF reference."""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.tt.vision.embedders import PROJECTION_KEYS, Gemma4UnifiedAudioEmbedder


def _hf_embedder():
    from transformers import AutoConfig

    try:
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedMultimodalEmbedder
    except ImportError:  # transformers without the unified Gemma 4 classes
        pytest.skip("transformers has no gemma4_unified module")
    model_path = os.environ.get("HF_MODEL", "google/gemma-4-12B-it")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(cfg, "audio_config", None) is None:
        pytest.skip(f"{model_path} has no audio_config (not a unified audio checkpoint)")
    torch.manual_seed(0)
    emb = Gemma4UnifiedMultimodalEmbedder(cfg.audio_config, cfg.text_config)
    with torch.no_grad():
        for p in emb.parameters():
            p.normal_(0.0, 0.05)
    return emb, cfg


def _state_dict_from_hf(emb):
    return {PROJECTION_KEYS[0]: emb.state_dict()["embedding_projection.weight"].detach().clone()}


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_valid, num_frames", [(750, 750), (400, 750), (37, 37)], ids=["full", "padded", "short"])
def test_unified_audio_embedder(mesh_device, num_valid, num_frames):
    emb, cfg = _hf_embedder()
    audio_dim = int(cfg.audio_config.output_proj_dims)
    torch.manual_seed(1)
    # Raw 16 kHz waveform frames, one soft token per 640 samples (40 ms); amplitude within [-1, 1].
    input_features = (torch.rand(1, num_frames, audio_dim) * 2 - 1) * 0.3
    mask = torch.zeros(1, num_frames, dtype=torch.bool)
    mask[0, :num_valid] = True
    with torch.no_grad():
        ref = emb(input_features.to(torch.bfloat16)).float()[0]  # [T, text_hidden]
    tt = Gemma4UnifiedAudioEmbedder(
        mesh_device,
        _state_dict_from_hf(emb),
        text_hidden_size=cfg.text_config.hidden_size,
        rms_eps=cfg.audio_config.rms_norm_eps,
    )
    out_tt, out_mask = tt.forward(input_features, mask)
    out = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if hasattr(mesh_device, "shape") else None
    )
    out = out[0, 0, :num_frames].float() if out.dim() == 4 else out[:num_frames].float()
    assert out_mask.shape == (1, out_tt.shape[2])
    assert int(out_mask.sum()) == num_valid
    passing, pcc = comp_pcc(ref[:num_valid], out[:num_valid], 0.99)
    logger.info(f"unified audio embedder PCC over {num_valid} valid frames: {pcc}")
    assert passing, f"PCC {pcc} < 0.99"
