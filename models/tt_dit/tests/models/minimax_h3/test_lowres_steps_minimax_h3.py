# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the low-res first steps (`MINIMAX_H3_LOWRES_STEPS`): the schedule split, the noise and
hand-off maths, and what each of the two denoise stages is given."""

from types import SimpleNamespace

import pytest
import torch

from ....models.transformers.minimax_h3 import convert_comfy_minimax_h3 as comfy_h3
from ....pipelines.minimax_h3.packing import MINIMAX_H3_AUDIO_CHANNELS, build_packed_sequence, patchify_video_latents
from ....pipelines.minimax_h3.pipeline_minimax_h3 import (
    LOWRES_STEPS_ENV,
    MiniMaxH3Pipeline,
    lowres_handoff_rows,
    lowres_noise_rows,
)
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

PATCH = (1, 2, 2)
CHANNELS, FRAMES, HEIGHT, WIDTH = 4, 3, 8, 12
SHAPE = (FRAMES, HEIGHT, WIDTH, CHANNELS, PATCH)
AUDIO_LATENTS = 5


def _scheduler(shift=12.0, grid_points=9):
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(grid_points)
    return scheduler


def _layout(height=HEIGHT, width=WIDTH, anchors=()):
    return build_packed_sequence(torch.ones(7, dtype=torch.long), FRAMES, height, width, AUDIO_LATENTS, PATCH, anchors)


@pytest.mark.parametrize("steps", [1, 2, 7])
def test_windows_split_the_schedule(steps):
    scheduler = _scheduler()
    total = scheduler.num_inference_steps
    first, rest = scheduler.window(0, steps), scheduler.window(steps, total)

    assert torch.equal(torch.cat([first.timesteps, rest.timesteps]), scheduler.timesteps)
    coefficients = [first.step_coefficient(i) for i in range(steps)]
    coefficients += [rest.step_coefficient(i) for i in range(total - steps)]
    assert coefficients == [scheduler.step_coefficient(i) for i in range(total)]
    assert (first.num_inference_steps, rest.num_inference_steps) == (steps, total - steps)


def test_window_to_clean_lands_on_the_clean_estimate():
    scheduler = _scheduler()
    window = scheduler.window(0, 2, to_clean=True)

    assert torch.equal(window.timesteps, scheduler.timesteps[:2])
    assert window.step_coefficient(0) == scheduler.step_coefficient(0)
    # x0 = x_t + sigma * v
    assert window.step_coefficient(1) == float(scheduler.sigmas[1])
    assert float(scheduler.sigmas[2]) > 0.0, "the parent schedule keeps its own sigmas"


def test_lowres_noise_is_unit_variance_at_half_size():
    video = torch.randn((1, CHANNELS, 16, 32, 48), generator=torch.Generator().manual_seed(0))
    rows = lowres_noise_rows(patchify_video_latents(video, PATCH), 16, 32, 48, CHANNELS, PATCH)

    assert rows.shape == (16 * 8 * 12, CHANNELS * 4)
    assert abs(float(rows.std()) - 1.0) < 0.02
    assert abs(float(rows.mean())) < 0.02


def test_handoff_is_the_upscale_renoised_to_sigma():
    lowres = torch.full((FRAMES * (HEIGHT // 4) * (WIDTH // 4), CHANNELS * 4), 3.0)
    noise = torch.randn(
        (FRAMES * (HEIGHT // 2) * (WIDTH // 2), CHANNELS * 4), generator=torch.Generator().manual_seed(1)
    )

    assert torch.equal(lowres_handoff_rows(lowres, noise, 0.0, *SHAPE), torch.full_like(noise, 3.0))
    assert torch.allclose(lowres_handoff_rows(lowres, noise, 0.25, *SHAPE), 0.75 * 3.0 + 0.25 * noise)


def test_handoff_upscales_in_space_only():
    # One value per low-res frame: every full-resolution pixel of that frame gets it, no frame bleeds into the next.
    frames = torch.arange(FRAMES, dtype=torch.float32).reshape(1, 1, FRAMES, 1, 1)
    lowres = patchify_video_latents(frames.expand(1, CHANNELS, FRAMES, HEIGHT // 2, WIDTH // 2), PATCH)
    expected = patchify_video_latents(frames.expand(1, CHANNELS, FRAMES, HEIGHT, WIDTH), PATCH)

    assert torch.equal(lowres_handoff_rows(lowres, torch.zeros_like(expected), 0.0, *SHAPE), expected)


def _stage_pipeline(calls, pdd=None):
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.transformer_config = {"patch_size": PATCH}
    pipeline.vae_config = SimpleNamespace(latent_channels=CHANNELS)
    pipeline._pdd_active = pdd

    def denoise(transformer, layout, prompt, video, audio, scheduler, audio_scheduler, **kwargs):
        calls.append((layout, prompt, video.clone(), scheduler, audio_scheduler, pipeline._pdd_active, kwargs))
        # The stand-in model: the low-res video comes out all 2.0, and each audio step adds one.
        video.fill_(2.0)
        audio.add_(float(audio_scheduler.num_inference_steps))
        return video, audio

    pipeline._denoise = denoise
    return pipeline


def _run_lowres(pipeline, scheduler, audio_scheduler, layout=None, width=WIDTH):
    layout = layout or _layout()
    noise = torch.randn(
        (FRAMES * (HEIGHT // 2) * (width // 2), CHANNELS * 4), generator=torch.Generator().manual_seed(2)
    )
    audio = torch.zeros((AUDIO_LATENTS * MINIMAX_H3_AUDIO_CHANNELS, 3))
    result = pipeline._denoise_lowres(
        "transformer",
        layout,
        "prompt",
        noise,
        audio,
        scheduler,
        audio_scheduler,
        FRAMES,
        HEIGHT,
        width,
        AUDIO_LATENTS,
        None,
    )
    return layout, noise, audio, result


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(LOWRES_STEPS_ENV, raising=False)
    calls = []
    # A bare pipeline: the default path reads nothing from it.
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline._denoise = lambda *args, **kwargs: calls.append(args)
    _layout_, noise, audio, result = _run_lowres(pipeline, "scheduler", "audio scheduler")

    assert result[0] is noise and result[1] is audio and result[2:] == ("scheduler", "audio scheduler")
    assert not calls


@pytest.mark.parametrize(
    "steps, kwargs",
    [
        ("8", {}),  # no full-resolution forward left
        ("2", {"layout": _layout(width=10), "width": 10}),  # 10 does not halve onto 2x2 patches
        ("2", {"layout": _layout(anchors=("first",))}),  # keyframe rows are full resolution
    ],
)
def test_requests_the_stage_does_not_apply_to_run_as_before(monkeypatch, steps, kwargs):
    monkeypatch.setenv(LOWRES_STEPS_ENV, steps)
    calls = []
    scheduler, audio_scheduler = _scheduler(12.0), _scheduler(3.0)
    _layout_, noise, audio, result = _run_lowres(_stage_pipeline(calls), scheduler, audio_scheduler, **kwargs)

    assert result[0] is noise and result[1] is audio and result[2:] == (scheduler, audio_scheduler)
    assert not calls


@pytest.mark.parametrize("steps", [1, 2])
def test_lowres_stage_hands_off_video_and_carries_audio(monkeypatch, steps):
    monkeypatch.setenv(LOWRES_STEPS_ENV, str(steps))
    calls = []
    scheduler, audio_scheduler = _scheduler(12.0), _scheduler(3.0)
    layout, noise, audio, result = _run_lowres(_stage_pipeline(calls), scheduler, audio_scheduler)
    video_rows, audio_rows, rest, audio_rest = result

    lowres_layout, prompt, lowres_video, first, audio_first, _pdd, kwargs = calls[0]
    assert len(calls) == 1 and kwargs["first_stage"] is True and prompt == "prompt"
    # Same text and audio rows, a quarter of the video rows.
    assert lowres_layout.sequence_length == layout.sequence_length - 3 * lowres_video.shape[0]
    assert torch.equal(lowres_layout.text_indices, layout.text_indices)
    assert torch.equal(lowres_video, lowres_noise_rows(noise, *SHAPE))

    # Video: its last low-res step lands on clean, then the upscale is re-noised to the next step's sigma.
    assert first.step_coefficient(steps - 1) == float(scheduler.sigmas[steps - 1])
    sigma = float(scheduler.sigmas[steps])
    assert torch.allclose(video_rows, (1.0 - sigma) * 2.0 + sigma * noise)
    assert torch.equal(rest.sigmas, scheduler.sigmas[steps:])

    # Audio: the same rows, stepped through the first `steps` of their own schedule and no further.
    assert audio_rows is audio and torch.equal(audio, torch.full_like(audio, float(steps)))
    assert torch.equal(torch.cat([audio_first.sigmas, audio_rest.sigmas[1:]]), audio_scheduler.sigmas)


def test_lowres_stage_pdd_heads(monkeypatch):
    monkeypatch.setenv(LOWRES_STEPS_ENV, "2")
    contract = {"pdd_steps": 32, "pdd_step_indices": list(range(0, 33, 4))}
    indices = comfy_h3.pdd_step_indices(contract, 8)
    weights = {"video": "video weights", "audio": "audio weights"}
    calls = []
    pipeline = _stage_pipeline(calls, pdd=({"video": indices, "audio": indices}, weights))
    _run_lowres(pipeline, _scheduler(12.0), _scheduler(3.0))

    # Low-res stage: the video's second step fuses every interval down to clean; the audio keeps its own two blocks.
    assert calls[0][5] == ({"video": [0, 4, 32], "audio": [0, 4, 8]}, weights)
    # What the full-resolution stage then reads: the remaining blocks of the trained partition, for both.
    assert pipeline._pdd_active == ({"video": indices[2:], "audio": indices[2:]}, weights)
