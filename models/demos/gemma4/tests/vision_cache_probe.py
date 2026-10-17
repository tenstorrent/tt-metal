# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is the zero/inf encoder output a weight-cache or a host-cast problem?

Builds the 12B unified embedder from the REAL checkpoint tensors (bf16) with no
weight cache, with the generator's cache path, and from a float32 copy of the
weights, and compares each against the HF image features for the dog image.
"""
import glob
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.demo.vision_demo import _device_params
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.vision.unified_embedder import Gemma4UnifiedVisionEmbedder


def _host(t, mesh_device):
    return (
        ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        if hasattr(mesh_device, "shape")
        else ttnn.to_torch(t).float()
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
def test_vision_cache_probe(mesh_device):
    ref = torch.load(os.environ["GEMMA4_VISION_REF"])
    pv, pos = ref["pixel_values"].to(torch.bfloat16), ref["image_position_ids"].to(torch.long)
    hf = ref["image_features"].float()
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/*/"))[0]
    sd = Gemma4ModelArgs.load_state_dict(snap)
    w = sd["model.vision_embedder.patch_dense.weight"]
    logger.info(
        f"checkpoint patch_dense dtype {w.dtype} abs mean {w.float().abs().mean():.5f}; pos_embedding abs mean {sd['model.vision_embedder.pos_embedding'].float().abs().mean():.5f}"
    )
    cache_dir = os.environ.get("GEMMA4_CACHE_PROBE_DIR")

    def run(label, state_dict, cache):
        emb = Gemma4UnifiedVisionEmbedder(mesh_device, state_dict, text_hidden_size=3840, weight_cache_path=cache)
        w_dev = _host(emb.dense_w, mesh_device)
        out_tt, mask = emb.forward(pv, pos)
        out = _host(out_tt, mesh_device)[0, 0]
        valid = out[: int(mask.sum())]
        pcc = comp_pcc(hf, valid, 0.95)[1] if valid.shape == hf.shape and torch.isfinite(valid).all() else "n/a"
        logger.info(
            f"{label}: dense_w on device abs mean {w_dev.abs().mean():.5f}; out abs mean {valid.abs().mean():.4f} max {valid.abs().max():.4f} nan {int(torch.isnan(valid).sum())}; PCC vs HF {pcc}"
        )

    run("bf16 weights, no cache", sd, None)
    run("fp32 weights, no cache", {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}, None)
    if cache_dir:
        run("bf16 weights, generator cache dir", sd, cache_dir)
