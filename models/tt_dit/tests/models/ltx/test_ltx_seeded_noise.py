# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The seeded latent noise (explicit generator, optionally prefetched on a thread) must be bit-identical to
the global ``torch.manual_seed`` + bf16 ``torch.randn`` draws the reference and golden outputs were made with."""

import pytest
import torch

from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline


def _legacy_stage1(seed, video_shape, audio_shape):
    """The pre-generator draws: video in the noiser, then a reseed + video-sized skip for audio."""
    torch.manual_seed(seed)
    video = torch.randn(video_shape, dtype=torch.bfloat16)
    torch.manual_seed(seed)
    _ = torch.randn(video_shape, dtype=torch.bfloat16)
    audio = torch.randn(audio_shape, dtype=torch.bfloat16)
    return video, audio


# (1, 37, 5) is not a multiple of 16 elements, where torch's vectorized normal fill takes its tail path.
@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("video_shape,audio_shape", [((1, 9728, 128), (1, 151, 128)), ((1, 37, 5), (1, 3, 5))])
@pytest.mark.parametrize("prefetch", [False, True])
def test_seeded_noise_matches_global_seed_draws(seed, video_shape, audio_shape, prefetch):
    pipe = object.__new__(LTXDistilledPipeline)
    if prefetch:
        pipe._prefetch_noise(seed, [(video_shape, audio_shape)])
    video, audio = pipe._seeded_noise(seed, video_shape, audio_shape)
    ref_video, ref_audio = _legacy_stage1(seed, video_shape, audio_shape)
    assert torch.equal(video, ref_video) and torch.equal(audio, ref_audio)
