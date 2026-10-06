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

from types import SimpleNamespace

import pytest
import torch

from ....models.transformers.ltx.rope_ltx import (
    prepare_av_cross_pe,
    prepare_video_rope,
    video_rope_freqs,
    video_rope_positions,
)
from ....pipelines.ltx import pipeline_ltx, pipeline_ltx_distilled
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


@pytest.mark.parametrize("fn", [prepare_video_rope, prepare_av_cross_pe])
def test_rope_builders_have_no_fps_default(fn):
    """A default fps is how video self-attention RoPE silently ran at 24 fps while serving 25."""
    import inspect

    assert inspect.signature(fn).parameters["fps"].default is inspect.Parameter.empty


@pytest.mark.parametrize("fps", [24.0, 25.0])
def test_video_rope_matches_diffusers(fps):
    """Production video self-attention RoPE equals the diffusers LTX-2 reference at the given fps.

    diffusers (like ltx-core) divides the temporal pixel coordinate by fps before the rotary
    frequencies. Small spatial grid; the served 20 latent frames so the temporal phase spans 6 s.
    """
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed

    F, H, W = 20, 4, 6
    ours_cos, ours_sin = video_rope_freqs(
        F, H, W, inner_dim=4096, num_attention_heads=32, theta=10000.0, max_pos=[20, 2048, 2048], fps=fps
    )
    # double_precision matches our builder (bit-exact); the fp32 diffusers path differs by ~5e-3.
    ref = LTX2AudioVideoRotaryPosEmbed(dim=4096, theta=10000.0, modality="video", rope_type="interleaved")
    ref_cos, ref_sin = ref(ref.prepare_video_coords(1, F, H, W, torch.device("cpu"), fps=fps))
    torch.testing.assert_close(ours_cos, ref_cos.float(), atol=1e-6, rtol=0)
    torch.testing.assert_close(ours_sin, ref_sin.float(), atol=1e-6, rtol=0)


def test_video_rope_fps_changes_temporal_phase():
    """25 vs 24 fps must move the temporal positions (the last latent frame sits at 6.04 s at 24)."""
    p24 = video_rope_positions(20, 2, 2, fps=24.0)
    p25 = video_rope_positions(20, 2, 2, fps=25.0)
    torch.testing.assert_close(p25[:, 0] * 25.0, p24[:, 0] * 24.0)
    assert not torch.allclose(p24[:, 0], p25[:, 0])
    torch.testing.assert_close(p24[:, 1:], p25[:, 1:])  # spatial axes are fps-independent


class _Captured(Exception):
    def __init__(self):
        super().__init__("prepare_video_rope reached")


def _capture_video_rope_fps(monkeypatch, module) -> dict:
    """Stub ``module.prepare_video_rope`` to record its kwargs and abort the caller there."""
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        raise _Captured()

    monkeypatch.setattr(module, "prepare_video_rope", fake)
    return seen


def _pipeline_stub(fps: float) -> SimpleNamespace:
    return SimpleNamespace(
        fps=fps,
        inner_dim=4096,
        num_attention_heads=32,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        mesh_device=None,
        parallel_config=None,
        in_channels=128,
        transformer=SimpleNamespace(image_conditioning=False),
        _sp_pad_len=lambda n: n,
    )


def test_distilled_pipeline_passes_fps_to_video_rope(monkeypatch, expect_error):
    seen = _capture_video_rope_fps(monkeypatch, pipeline_ltx_distilled)
    with expect_error(_Captured, "prepare_video_rope reached"):
        pipeline_ltx_distilled.LTXDistilledPipeline._prepare_stage_statics(
            _pipeline_stub(25.0),
            SimpleNamespace(tt_video_rope_cos=None),
            latent_frames=20,
            latent_h=17,
            latent_w=30,
            video_N=10240,
            video_N_real=10200,
            audio_N=256,
            audio_N_real=153,
            sp_axis=1,
        )
    assert seen.get("fps") == 25.0


def test_two_stage_pipeline_passes_fps_to_video_rope(monkeypatch, expect_error):
    seen = _capture_video_rope_fps(monkeypatch, pipeline_ltx)
    with expect_error(_Captured, "prepare_video_rope reached"):
        LTXPipeline.call_av(
            _pipeline_stub(25.0),
            video_prompt_embeds=None,
            audio_prompt_embeds=None,
            num_frames=153,
            height=1088,
            width=1920,
        )
    assert seen.get("fps") == 25.0
