# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive prefill-pipeline parity for ALL qwen3.6 VL input use cases.

This test verifies that the multimodal prefill WIRING is correct for every
modality combination the VL server can receive, at a level that is INDEPENDENT
of the known decode-degeneracy ("blocker #1"). It does NOT run the text decoder
or the sampler; it checks the two host/device primitives that determine whether
vision features and rotary positions are placed correctly:

  (A) on-device splice `splice_modalities_ttnn` PCC > 0.99 vs the host golden
      `splice_modalities_into_embeddings` — proves vision features land at the
      right token positions for text-only / 1,2,3 images / video / image+video.

  (B) our `get_rope_index` 3D positions EXACTLY equal HF
      `Qwen3VLModel.get_rope_index` (torch.equal) for the same input_ids +
      grids — proves the 3D M-RoPE positions are correct for every case.

USE CASES (each parametrized below):
  1. text-only (no vision)        -> splice no-op, rope = 1D ramp
  2. 1 image
  3. 2 images (multi-image)
  4. 3 images (multi-image)
  5. 1 video (multi-frame)
  6. 1 image + 1 video (mixed)

Token-count rule (matches HF): an image/video grid [t, h, w] contributes
`t * (h // spatial_merge_size) * (w // spatial_merge_size)` placeholder tokens.
For video, HF repeat_interleaves the temporal dim and sets t=1 per frame, so a
video grid [T, h, w] contributes `T * (h//sms) * (w//sms)` placeholder tokens
(same total). We size each synthetic feature tensor to that exact count.
"""

import os
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_pipeline import (
    splice_modalities_into_embeddings,
    splice_modalities_ttnn,
)
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import get_rope_index

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    HF_AVAILABLE = True
except Exception:  # pragma: no cover
    HF_AVAILABLE = False

H = 5120  # qwen3.6 text hidden size
IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
VISION_START_TOKEN_ID = 248053
SPATIAL_MERGE_SIZE = 2

# Generic non-special filler text tokens.
TXT = [100, 101, 200, 201, 202, 203]


def _hf_holder():
    cfg = types.SimpleNamespace()
    cfg.vision_config = types.SimpleNamespace(spatial_merge_size=SPATIAL_MERGE_SIZE)
    cfg.image_token_id = IMAGE_TOKEN_ID
    cfg.video_token_id = VIDEO_TOKEN_ID
    cfg.vision_start_token_id = VISION_START_TOKEN_ID
    return types.SimpleNamespace(config=cfg)


def _ntok(grid):
    """Placeholder-token count for one grid [t, h, w]."""
    t, h, w = grid
    return t * (h // SPATIAL_MERGE_SIZE) * (w // SPATIAL_MERGE_SIZE)


# --- use-case definitions -------------------------------------------------
# Each: list of ("image"|"video", grid_thw). Order = visit order in the stream.
# Grids are realistic & small (divisible by spatial_merge_size=2).
CASES = {
    "text_only": [],
    "image_1": [("image", (1, 8, 8))],
    "image_2": [("image", (1, 8, 8)), ("image", (1, 4, 6))],
    "image_3": [("image", (1, 8, 8)), ("image", (1, 4, 6)), ("image", (1, 6, 4))],
    "video_1": [("video", (4, 16, 16))],
    "image_and_video": [("image", (1, 8, 8)), ("video", (4, 16, 16))],
}


def _build_case(case_name, *, seed=0):
    """Build (text_embed [B,S,H], input_ids [B,S], image_feats, video_feats,
    image_grid_thw, video_grid_thw) for a use case.

    The token stream interleaves <vision_start><pad-run> blocks so HF's
    get_rope_index sees the same structure a real processor would emit.
    """
    torch.manual_seed(seed)
    segs = CASES[case_name]

    ids = list(TXT[:2])  # leading text
    img_grids, vid_grids = [], []
    n_img_tok, n_vid_tok = 0, 0
    for kind, grid in segs:
        ids.append(VISION_START_TOKEN_ID)
        n = _ntok(grid)
        if kind == "image":
            ids += [IMAGE_TOKEN_ID] * n
            img_grids.append(grid)
            n_img_tok += n
        else:
            ids += [VIDEO_TOKEN_ID] * n
            vid_grids.append(grid)
            n_vid_tok += n
        ids += [TXT[2]]  # separating text after each vision block
    ids += list(TXT[3:5])  # trailing text
    S = len(ids)
    B = 1
    input_ids = torch.tensor([ids], dtype=torch.long)

    text_embed = torch.randn(B, S, H, dtype=torch.float32)
    image_features = torch.randn(n_img_tok, H, dtype=torch.float32) if n_img_tok else None
    video_features = torch.randn(n_vid_tok, H, dtype=torch.float32) if n_vid_tok else None
    image_grid_thw = torch.tensor(img_grids, dtype=torch.long) if img_grids else None
    video_grid_thw = torch.tensor(vid_grids, dtype=torch.long) if vid_grids else None
    return (
        text_embed,
        input_ids,
        image_features,
        video_features,
        image_grid_thw,
        video_grid_thw,
    )


# =========================================================================
# PART 1B: rope-index parity vs HF (CPU-only — no device needed).
# =========================================================================
@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers qwen3_vl not installed")
@pytest.mark.parametrize("case_name", list(CASES.keys()))
@torch.no_grad()
def test_get_rope_index_matches_hf(case_name):
    (_, input_ids, _, _, image_grid_thw, video_grid_thw) = _build_case(case_name)

    # HF mutates video_grid_thw in place; pass clones to each.
    hf_pos, hf_delta = Qwen3VLModel.get_rope_index(
        _hf_holder(),
        input_ids=input_ids,
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.clone(),
        video_grid_thw=None if video_grid_thw is None else video_grid_thw.clone(),
    )
    tt_pos, tt_delta = get_rope_index(
        input_ids,
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.clone(),
        video_grid_thw=None if video_grid_thw is None else video_grid_thw.clone(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    )
    assert torch.equal(
        tt_pos, hf_pos.to(tt_pos.dtype)
    ), f"[{case_name}] position_ids mismatch\nHF:\n{hf_pos}\nTT:\n{tt_pos}"
    assert torch.equal(
        tt_delta, hf_delta.to(tt_delta.dtype)
    ), f"[{case_name}] mrope_deltas mismatch HF={hf_delta.tolist()} TT={tt_delta.tolist()}"
    logger.info(f"[{case_name}] get_rope_index parity OK: pos {tuple(tt_pos.shape)} delta {tt_delta.tolist()}")


# =========================================================================
# PART 1A: on-device splice PCC vs host golden (device-bound).
# =========================================================================
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
@pytest.mark.parametrize("case_name", list(CASES.keys()))
def test_splice_ttnn_pcc_all_cases(mesh_device, case_name, reset_seeds, ensure_gc):
    (text_embed, input_ids, image_features, video_features, _, _) = _build_case(case_name)

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

    def _up(t):
        if t is None:
            return None
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )

    text_embed_tt = _up(text_embed)
    image_features_tt = _up(image_features)
    video_features_tt = _up(video_features)

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
    B = text_embed.shape[0]
    fused_host = fused_host[:B]  # replicated → first mesh shard

    passing, pcc_message = comp_pcc(golden, fused_host, 0.99)
    logger.info(f"[{case_name}] splice PCC: {pcc_message}")
    assert passing, f"splice_modalities_ttnn [{case_name}] PCC < 0.99: {pcc_message}"
