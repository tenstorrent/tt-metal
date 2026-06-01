# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Bisect the connector path: isolate the video aggregate_embed linear projection.

Feeds the SAME random input to (a) the on-device connector.aggregate_embed and
(b) a faithful torch reference (W @ x + b using the raw checkpoint weights). A
clean PCC here clears the linear/weight-loading stage; a bad PCC localizes the
negative-PCC connector bug to aggregate_embed.
"""

import glob
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.tt_dit.encoders.gemma.feature_extractor import GemmaFeatureExtractor
from models.tt_dit.pipelines.ltx.pipeline_ltx_av import LTXAVPipeline

CONNECTOR_PREFIXES = (
    "text_embedding_projection.video_aggregate_embed.",
    "text_embedding_projection.audio_aggregate_embed.",
    "model.diffusion_model.video_embeddings_connector.",
    "model.diffusion_model.audio_embeddings_connector.",
)


def _gemma_path():
    c = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/")
    )
    return c[0].rstrip("/") if c else None


def _ltx_ckpt():
    c = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-dev.safetensors"
        )
    )
    return c[0] if c else None


def pcc(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    a_m, b_m = a_f - a_f.mean(), b_f - b_f.mean()
    d = (a_m.pow(2).sum() * b_m.pow(2).sum()).sqrt()
    return ((a_m * b_m).sum() / d).item() if d > 0 else 0.0


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_aggregate_embed_isolated(*, mesh_device):
    gemma, ckpt = _gemma_path(), _ltx_ckpt()
    if not gemma or not ckpt:
        pytest.skip("assets missing")

    pipe = LTXAVPipeline.create_pipeline(mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av")
    # Connectors need a gemma encoder loaded for sequence_length bookkeeping, but we
    # only exercise aggregate_embed here, so skip the (slow) encoder load.
    conn_state, raw = {}, {}
    with safe_open(ckpt, "pt") as f:
        for k in f.keys():
            if k.startswith(CONNECTOR_PREFIXES):
                conn_state[k] = f.get_tensor(k)
                if k.startswith("text_embedding_projection.video_aggregate_embed."):
                    raw[k] = f.get_tensor(k)
    pipe.load_embeddings_connectors(conn_state, audio_num_blocks=8)

    # The device aggregate_embed weight is permuted D-major→layer-major at load, so the
    # reference must use the same permuted weight to be an apples-to-apples linear check.
    Wv = GemmaFeatureExtractor._weight_to_layer_major(
        raw["text_embedding_projection.video_aggregate_embed.weight"].float(), 3840, 49
    )  # [4096, 188160], layer-major
    bv = raw["text_embedding_projection.video_aggregate_embed.bias"].float()  # [4096]
    in_dim = Wv.shape[1]

    torch.manual_seed(0)
    x = (torch.randn(1, 32, in_dim) * 0.1).bfloat16()  # small seq for speed

    # torch reference linear
    ref = (x.float() @ Wv.t()) + bv

    # device aggregate_embed (now owned by the feature extractor)
    tt_x = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_y = pipe.feature_extractor.video_aggregate_embed(tt_x)
    dev = ttnn.to_torch(ttnn.get_device_tensors(tt_y)[0]).float()

    logger.info(f"aggregate_embed: ref={tuple(ref.shape)} dev={tuple(dev.shape)}  PCC={pcc(dev, ref):.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
