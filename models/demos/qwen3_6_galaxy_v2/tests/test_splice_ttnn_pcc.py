# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for the on-device (TTNN) vision/text splice used by the qwen3.6 VLM
server path.

The host splice `splice_modalities_into_embeddings` in
`tt/qwen36_mm_pipeline.py` is the GOLDEN reference. The on-device equivalent
`splice_modalities_ttnn` must produce the same fused embeddings (PCC > 0.99)
while operating on the embedding/feature tensors entirely with TTNN ops on
device (host is only used for the position-index lookup from input_ids, which
is a small int INPUT tensor — trace-safe).

Three cases are covered: image-only, video-only, image+video.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_pipeline import (
    splice_modalities_into_embeddings,
    splice_modalities_ttnn,
)

H = 5120  # qwen3.6 text hidden size
IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057


def _build_inputs(case, *, seed=0):
    """Build synthetic [B,S,H] text embeddings, feature tensors, and input_ids.

    `case` is one of "image", "video", "both".
    Returns (text_embed, input_ids, image_features, video_features).
    """
    torch.manual_seed(seed)
    B = 1
    n_img = 6 if case in ("image", "both") else 0
    n_vid = 4 if case in ("video", "both") else 0

    # Layout: [txt, txt, <img run>, txt, <vid run>, txt]
    ids = [100, 101]
    if n_img:
        ids += [IMAGE_TOKEN_ID] * n_img
    ids += [200]
    if n_vid:
        ids += [VIDEO_TOKEN_ID] * n_vid
    ids += [201, 202]
    S = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long)

    text_embed = torch.randn(B, S, H, dtype=torch.float32)
    image_features = torch.randn(n_img, H, dtype=torch.float32) if n_img else None
    video_features = torch.randn(n_vid, H, dtype=torch.float32) if n_vid else None
    return text_embed, input_ids, image_features, video_features


@torch.no_grad()
@pytest.mark.hardware
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "BH_GLX": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("case", ["image", "video", "both"])
def test_splice_ttnn_pcc(mesh_device, case, reset_seeds, ensure_gc):
    text_embed, input_ids, image_features, video_features = _build_inputs(case)

    # ---- Host golden ----
    golden = splice_modalities_into_embeddings(
        text_embed,
        input_ids,
        image_features=image_features,
        video_features=video_features,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )

    # ---- On-device inputs (replicated across the mesh) ----
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    text_embed_tt = ttnn.from_torch(
        text_embed,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
    )
    image_features_tt = (
        ttnn.from_torch(
            image_features,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        if image_features is not None
        else None
    )
    video_features_tt = (
        ttnn.from_torch(
            video_features,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        if video_features is not None
        else None
    )

    fused_tt = splice_modalities_ttnn(
        text_embed_tt,
        input_ids,
        mesh_device=mesh_device,
        image_features=image_features_tt,
        video_features=video_features_tt,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )

    fused_host = ttnn.to_torch(fused_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # Replicated → take the first mesh shard.
    B, S, _ = text_embed.shape
    fused_host = fused_host[:B]

    passing, pcc_message = comp_pcc(golden, fused_host, 0.99)
    logger.info(f"[{case}] PCC: {pcc_message}")
    assert passing, f"splice_modalities_ttnn [{case}] PCC < 0.99: {pcc_message}"
