# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-by-stage stats of the on-device image encoder on a REAL processor output.

The random-weight unit tests pass while the real image yields inf (31B) / zeros
(12B): print abs-mean / max / NaN counts after the encoder, after the projector
and after the valid-token gather, and compare the final tokens with HF.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.demo.vision_demo import _device_params
from models.demos.gemma4.tt.generator import Gemma4Generator, _vision_encoder_seq_len


def _host(t, mesh_device):
    return (
        ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        if hasattr(mesh_device, "shape")
        else ttnn.to_torch(t).float()
    )


def _stats(name, t):
    t = t.float()
    logger.info(
        f"{name}: shape {tuple(t.shape)} abs mean {t.abs().mean():.4f} max {t.abs().max():.4f} nan {int(torch.isnan(t).sum())} inf {int(torch.isinf(t).sum())}"
    )


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_vision_encoder_probe(mesh_device):
    ref = torch.load(os.environ["GEMMA4_VISION_REF"])
    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    pixel_values, pos = ref["pixel_values"], ref["image_position_ids"].to(torch.long)
    _stats("pixel_values (host)", pixel_values)
    logger.info(
        f"pixel dtype {pixel_values.dtype}; positions valid {(pos != -1).all(-1).sum().item()} / {pos.shape[1]}"
    )
    vdt = getattr(ttnn, os.environ.get("GEMMA4_PROBE_VISION_DTYPE", "bfloat8_b"))
    logger.info(f"vision dtype {vdt}")
    generator, _kv, _tok = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=4096,
        multimodal=True,
        vision_dtype=vdt,
    )
    tower = generator.vision_tower
    pv = pixel_values.to(torch.bfloat16)
    if getattr(tower, "projects_to_text", False):
        projected_tt, mask = tower(pv, pos)
        _stats("unified embedder out", _host(projected_tt, mesh_device))
    else:
        seq_len = _vision_encoder_seq_len(pos.shape[1])
        pooled_tt, mask = tower(pv, pos, seq_len)
        pooled = _host(pooled_tt, mesh_device)
        _stats("tower pooled out", pooled)
        projected_tt = generator.embed_vision(pooled_tt)
        _stats("projector out", _host(projected_tt, mesh_device))
    logger.info(f"mask valid {int(mask.sum())} / {mask.shape[-1]}")
    embeds_tt = generator.encode_vision(pv, pos)
    feats = _host(embeds_tt, mesh_device)
    _stats("after gather (encode_vision)", feats)
    hf = ref["image_features"].float()
    if feats.shape == hf.shape and torch.isfinite(feats).all():
        _, pcc = comp_pcc(hf, feats, 0.95)
        logger.info(f"final soft tokens PCC vs HF: {pcc}")
