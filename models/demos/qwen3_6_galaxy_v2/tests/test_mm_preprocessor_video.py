# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-PREPROC video tests — CPU sanity that the multimodal preprocessor accepts a
video input and produces `pixel_values_videos` + `video_grid_thw` + a video
placeholder token in `input_ids`.

CPU-only: no Tenstorrent device is opened.
"""

import numpy as np
import torch

from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_preprocessor import Qwen36MMInputs, Qwen36MMPreprocessor

SNAPSHOT = (
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def _make_video(n_frames: int = 4, size: int = 224) -> np.ndarray:
    """Synthetic RGB clip: [n_frames, H, W, 3] uint8."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (n_frames, size, size, 3), dtype=np.uint8)


@torch.no_grad()
def test_preprocessor_with_video():
    """Prompt with one synthetic video — produces pixel_values_videos + video_grid_thw."""
    proc = Qwen36MMPreprocessor(SNAPSHOT)

    video = _make_video()
    out = proc("<|vision_start|><|video_pad|><|vision_end|>Describe this video", videos=[video])

    assert isinstance(out, Qwen36MMInputs)

    # Video features must be exposed and non-None.
    assert out.pixel_values_videos is not None, "expected pixel_values_videos for a video input"
    assert out.video_grid_thw is not None, "expected video_grid_thw for a video input"

    # patch_feat_dim = in_channels * temporal_patch_size * patch_size**2 = 3*2*16*16 = 1536
    assert out.pixel_values_videos.ndim == 2
    assert out.pixel_values_videos.shape[-1] == 1536

    # video_grid_thw is [num_videos, 3]
    assert out.video_grid_thw.ndim == 2
    assert out.video_grid_thw.shape[-1] == 3

    # The video placeholder token must appear in input_ids.
    video_token_id = proc.video_token_id
    assert (out.input_ids[0] == video_token_id).any(), "expected video_pad tokens in input_ids"

    # 3D positions are still produced with the right shape.
    assert out.position_ids_3d.shape == (3, *out.input_ids.shape)

    print(
        f"video-input OK: input_ids shape {tuple(out.input_ids.shape)}, "
        f"pixel_values_videos shape {tuple(out.pixel_values_videos.shape)}, "
        f"video_grid_thw {out.video_grid_thw.tolist()}"
    )
