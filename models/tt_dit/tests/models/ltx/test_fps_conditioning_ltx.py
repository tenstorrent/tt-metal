# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free regression tests for LTX FPS conditioning.

FPS is not a container label: it sets the audio latent length
(``AudioLatentShape.from_video_pixel_shape`` divides frames by fps to get a duration) and
scales the A/V cross-PE temporal axis into seconds (``rope_ltx.prepare_video_rope`` /
``prepare_av_cross_pe``). Both are baked into captured traces, so the rate is fixed at
pipeline construction and a mismatched per-call value must be rejected rather than
substituted.

These run on CPU so the conditioning path is covered in CI without a Galaxy. The
end-to-end behaviour at the served shape (153 frames @ 25 fps) is exercised by
``test_pipeline_distilled``; this file guards the arithmetic and the guard, which is what
a future edit would silently break while every device test stayed green.
"""

import pytest

from ....models.transformers.ltx.rope_ltx import prepare_av_cross_pe, prepare_video_rope
from ....pipelines.ltx.pipeline_ltx import LTXPipeline
from ....utils.patchifiers import AudioLatentShape, VideoPixelShape


class _FpsOnly:
    """Minimal stand-in so ``_resolve_fps`` can be tested without a mesh device."""

    def __init__(self, fps: float):
        self.fps = fps


def _audio_frames(num_frames: int, fps: float) -> int:
    return AudioLatentShape.from_video_pixel_shape(
        VideoPixelShape(batch=1, frames=num_frames, height=1088, width=1920, fps=fps)
    ).frames


# (num_frames, fps, expected audio latent frames). The audio latent rate is
# 16000 / 160 / 4 = 25/s, so audio_frames = round(num_frames / fps * 25).
@pytest.mark.parametrize(
    "num_frames, fps, expected",
    [
        (145, 24, 151),  # today's legacy shape: lopsided, 145 video -> 151 audio
        (145, 25, 145),  # at 25 fps the two grids coincide exactly
        (153, 25, 153),  # the served Console shape
        (153, 24, 159),  # same frame count at 24 fps -> a DIFFERENT audio length
        (241, 25, 241),
    ],
)
def test_audio_latent_length_tracks_fps(num_frames, fps, expected):
    assert _audio_frames(num_frames, fps) == expected


def test_fps_changes_audio_length_for_a_fixed_frame_count():
    """The regression this guards: hardcoding 24 while serving 25.

    153 frames yields 159 audio latents at 24 fps and 153 at 25. If a future edit restores a
    literal 24 at the VideoPixelShape sites, the audio latent (and the A/V alignment built
    against it) is sized for the wrong timeline even though the container still says 25.
    """
    assert _audio_frames(153, 24) != _audio_frames(153, 25)


def test_audio_and_video_grids_coincide_at_25fps():
    """25 fps matches the audio latent rate, so the grids align 1:1 at any legal length."""
    for num_frames in (145, 153, 161, 241, 481):
        assert _audio_frames(num_frames, 25) == num_frames


@pytest.mark.parametrize("fps", [24, 25.0])
def test_resolve_fps_accepts_none_and_matching(fps):
    pipeline = _FpsOnly(float(fps))
    assert LTXPipeline._resolve_fps(pipeline, None) == float(fps)
    assert LTXPipeline._resolve_fps(pipeline, fps) == float(fps)


def test_resolve_fps_rejects_a_mismatch(expect_error):
    """Rejected, not substituted: generating at one rate while the caller and the MP4
    container believe another is exactly the desync the plumbing removes."""
    pipeline = _FpsOnly(25.0)
    with expect_error(ValueError, "does not match the pipeline's fps"):
        LTXPipeline._resolve_fps(pipeline, 24)


def test_served_shape_is_a_legal_frame_count():
    """(num_frames - 1) % 8 == 0 is required for the VAE to decode latent_frames exactly.

    6s x 25fps = 150 is illegal, which is why the served shape is 153 (6.12s) rather than
    150, and why 145f@25 (5.80s) would under-deliver a "6 second" product.
    """
    assert (153 - 1) % 8 == 0
    assert (150 - 1) % 8 != 0
    assert 153 / 25 == pytest.approx(6.12)


@pytest.mark.parametrize("fn", [prepare_video_rope, prepare_av_cross_pe])
def test_rope_builders_still_take_fps(fn):
    """Guard the cross-PE / rope rate plumbing against silent removal.

    ``prepare_av_cross_pe`` scales the video temporal axis into seconds by dividing by fps,
    which is what aligns audio against video. That parameter existed but was never passed
    for a long time, so the A/V alignment was built at 24 fps regardless of the requested
    rate -- the bug this suite exists to prevent regressing.

    Exercising the scaling itself needs a mesh device (both builders return ttnn tensors),
    so that is covered end-to-end by the 153f/25fps CI case. What this asserts is narrower
    but still useful on CPU: the keyword remains part of the contract, so a refactor that
    drops it fails here instead of silently reinstating a fixed 24 fps.
    """
    import inspect

    params = inspect.signature(fn).parameters
    assert "fps" in params, f"{fn.__name__} lost its fps parameter"
    assert params["fps"].kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{fn.__name__}'s fps must stay keyword-only so a positional call cannot silently "
        "bind the wrong argument to it"
    )
