# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-MROPE video tests — CPU parity of our `get_rope_index` video branch against
HF `Qwen3VLModel.get_rope_index` for the same synthetic clip.

CPU-only: no Tenstorrent device is opened. We invoke HF's `get_rope_index`
as an unbound method on a minimal config holder (no weights loaded).
"""

import types

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import get_rope_index

SNAPSHOT = (
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057
VISION_START_TOKEN_ID = 248053
SPATIAL_MERGE_SIZE = 2


def _hf_holder():
    cfg = types.SimpleNamespace()
    cfg.vision_config = types.SimpleNamespace(spatial_merge_size=SPATIAL_MERGE_SIZE)
    cfg.image_token_id = IMAGE_TOKEN_ID
    cfg.video_token_id = VIDEO_TOKEN_ID
    cfg.vision_start_token_id = VISION_START_TOKEN_ID
    return types.SimpleNamespace(config=cfg)


def _make_video(n_frames: int = 4, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (n_frames, size, size, 3), dtype=np.uint8)


@torch.no_grad()
def test_video_rope_index_matches_hf():
    """Our get_rope_index video branch must byte-match HF for a synthetic clip."""
    proc = AutoProcessor.from_pretrained(SNAPSHOT, trust_remote_code=False)
    video = _make_video()
    out = proc(
        text="<|vision_start|><|video_pad|><|vision_end|>Describe this video",
        videos=[video],
        return_tensors="pt",
        padding=True,
    )
    input_ids = out["input_ids"]
    video_grid_thw = out["video_grid_thw"]
    attention_mask = out["attention_mask"]

    # HF mutates video_grid_thw in place (repeat_interleave + t<-1); pass clones.
    hf_pos, hf_delta = Qwen3VLModel.get_rope_index(
        _hf_holder(),
        input_ids=input_ids,
        video_grid_thw=video_grid_thw.clone(),
        attention_mask=attention_mask,
    )

    tt_pos, tt_delta = get_rope_index(
        input_ids,
        video_grid_thw=video_grid_thw.clone(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    )

    assert torch.equal(tt_pos, hf_pos.to(tt_pos.dtype)), f"position_ids mismatch\nHF:\n{hf_pos}\nTT:\n{tt_pos}"
    assert torch.equal(
        tt_delta, hf_delta.to(tt_delta.dtype)
    ), f"mrope_deltas mismatch HF={hf_delta.tolist()} TT={tt_delta.tolist()}"
    print(f"video get_rope_index parity OK: pos {tuple(tt_pos.shape)} delta {tt_delta.tolist()}")
