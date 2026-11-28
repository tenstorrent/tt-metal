# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the low-res first steps (`MINIMAX_H3_LOWRES_STEPS`): the schedule split, the noise and
hand-off maths, what each of the two denoise stages is given, and which requests run them by default."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ....models.transformers.minimax_h3 import convert_comfy_minimax_h3 as comfy_h3
from ....pipelines.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    build_packed_sequence,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from ....pipelines.minimax_h3.pipeline_minimax_h3 import (
    EROS_MAX_BETA7_FIX1,
    HYPERFLOW_V1,
    LOWRES_STEPS_ENV,
    MiniMaxH3Pipeline,
    lowres_handoff_rows,
    lowres_noise_rows,
)
from ....pipelines.minimax_h3.pipeline_minimax_h3_turbo import MiniMaxH3TurboPipeline
from ....pipelines.minimax_h3.policy import SolSchedule
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

PATCH = (1, 2, 2)
CHANNELS, FRAMES, HEIGHT, WIDTH = 4, 3, 8, 12
SHAPE = (FRAMES, HEIGHT, WIDTH, CHANNELS, PATCH, HEIGHT // 2, WIDTH // 2)
AUDIO_LATENTS = 5


def _scheduler(shift=12.0, grid_points=9):
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(grid_points)
    return scheduler


def _layout(height=HEIGHT, width=WIDTH, anchors=(), frames=FRAMES):
    return build_packed_sequence(torch.ones(7, dtype=torch.long), frames, height, width, AUDIO_LATENTS, PATCH, anchors)


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


@pytest.mark.parametrize("height, width", [(32, 48), (24, 36)])  # 16x24 is 1/2 of the first, 2/3 of the second
def test_lowres_noise_is_the_low_frequencies_of_the_noise(height, width):
    # Still unit variance, and below its own highest frequency it is the request's noise.
    video = torch.randn((1, CHANNELS, 16, height, width), generator=torch.Generator().manual_seed(0))
    rows = lowres_noise_rows(patchify_video_latents(video, PATCH), 16, height, width, CHANNELS, PATCH, 16, 24)
    lowres = unpatchify_video_tokens(rows, 16, 16, 24, CHANNELS, PATCH)

    assert abs(float(lowres.std()) - 1.0) < 0.02
    spectrum, lowres_spectrum = torch.fft.fft2(video, norm="ortho"), torch.fft.fft2(lowres, norm="ortho")
    assert torch.allclose(lowres_spectrum[..., :8, :12], spectrum[..., :8, :12], atol=1e-4)
    # The one frequency that is the highest on both axes has the power of every other.
    assert torch.allclose(lowres_spectrum[..., 8, 12].real, spectrum[..., -8, -12].real * 2**0.5, atol=1e-4)


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
    pipeline._log = lambda message: None

    def denoise(transformer, layout, prompt, video, audio, scheduler, audio_scheduler, **kwargs):
        calls.append((layout, prompt, video.clone(), scheduler, audio_scheduler, pipeline._pdd_active, kwargs))
        # The stand-in model: the low-res video comes out all 2.0, and each audio step adds one.
        video.fill_(2.0)
        audio.add_(float(audio_scheduler.num_inference_steps))
        return video, audio

    pipeline._denoise = denoise
    return pipeline


def _run_lowres(pipeline, scheduler, audio_scheduler, layout=None, width=WIDTH, frames=FRAMES, height=HEIGHT):
    layout = layout or _layout(height, width, frames=frames)
    noise = torch.randn(
        (frames * (height // 2) * (width // 2), CHANNELS * 4), generator=torch.Generator().manual_seed(2)
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
        frames,
        height,
        width,
        AUDIO_LATENTS,
        None,
    )
    return layout, noise, audio, result


@pytest.mark.parametrize(
    "steps, kwargs",
    [
        ("8", {}),  # no full-resolution forward left
        ("2", {"width": 10}),  # 10 does not halve onto 2x2 patches
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
    assert float(scheduler.sigmas[steps]) > 0.0, "the parent schedule keeps its own sigmas"
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


def test_lowres_stage_never_runs_sol():
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline._sol_schedule = SolSchedule()
    assert all(pipeline._is_sol_dense_step(step, first_stage=True, num_latent_frames=1) for step in range(4))


LIGHTX2V_V13 = "97b775707bb5732e01dba0cf3833d3fe2947c953859d4d05ea3bd74894c43d2b"


def _lowres_stages(weights, frames, forwards, height=48, width=84):
    """The low-res stages a request runs, each as (forwards, video rows); 48x84 is the latent of 1344x768."""
    calls = []
    pipeline = _stage_pipeline(calls)
    pipeline._lowres_weights = lambda: weights
    _run_lowres(pipeline, _scheduler(12.0, forwards + 1), _scheduler(3.0, forwards + 1), None, width, frames, height)
    return [(call[3].num_inference_steps, call[2].shape[0]) for call in calls]


@pytest.mark.parametrize(
    "weights, frames, forwards",
    [(EROS_MAX_BETA7_FIX1, 47, 4), (HYPERFLOW_V1, 47, 8), (HYPERFLOW_V1, 72, 8)],  # 47 latent frames: 6 s, 72: 10 s
)
def test_cells_that_passed_run_their_setting_by_default(monkeypatch, weights, frames, forwards):
    monkeypatch.delenv(LOWRES_STEPS_ENV, raising=False)
    # the first two forwards at 896x512: a 32x56 latent, 16x28 rows a frame
    assert _lowres_stages(weights, frames, forwards) == [(2, frames * 16 * 28)]

    # the switch wins: 0 turns the stage off, any other count runs as it did before there was a default (at 1/2)
    monkeypatch.setenv(LOWRES_STEPS_ENV, "0")
    assert _lowres_stages(weights, frames, forwards) == []
    monkeypatch.setenv(LOWRES_STEPS_ENV, "3")
    assert _lowres_stages(weights, frames, forwards) == [(3, frames * 12 * 21)]


@pytest.mark.parametrize(
    "weights, frames, forwards, size",
    [
        (None, 47, 4, (48, 84)),  # standard H3
        (LIGHTX2V_V13, 47, 4, (48, 84)),  # failed the look
        (EROS_MAX_BETA7_FIX1, 72, 8, (48, 84)),  # 10 s at 8 forwards failed the look
        (EROS_MAX_BETA7_FIX1, 72, 4, (48, 84)),  # 10 s at 4 forwards: not judged on enough seeds yet
        (EROS_MAX_BETA7_FIX1, 47, 8, (48, 84)),  # a forward count nobody judged
        (EROS_MAX_BETA7_FIX1, 47, 4, (36, 64)),  # a resolution nobody judged
        (HYPERFLOW_V1, 47, 4, (48, 84)),  # a forward count nobody judged
    ],
)
def test_other_requests_stay_at_full_resolution_by_default(monkeypatch, weights, frames, forwards, size):
    monkeypatch.delenv(LOWRES_STEPS_ENV, raising=False)
    assert _lowres_stages(weights, frames, forwards, *size) == []


def test_a_transformer_file_is_named_by_its_header_not_its_file_name(tmp_path):
    header = b'{"__metadata__":{"merge":"beta7 fix1"}}'
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    assert pipeline._lowres_weights() is None  # the plain weights

    pipeline.transformer_file = tmp_path / "10Eros_Max.safetensors"
    pipeline.transformer_file.write_bytes(len(header).to_bytes(8, "little") + header + bytes(64))
    assert pipeline._lowres_weights() == hashlib.sha256(header).hexdigest()
    pipeline.transformer_file = pipeline.transformer_file.rename(tmp_path / "q7a.safetensors")
    assert pipeline._lowres_weights() == hashlib.sha256(header).hexdigest()


@pytest.mark.parametrize(
    "strength, transformer_file, weights",
    [(1.0, None, "adapter"), (0.8, None, None), (1.0, Path("10Eros_Max.safetensors"), None)],
)
def test_an_adapter_names_the_weights_only_as_its_clips_were_judged(strength, transformer_file, weights):
    pipeline = MiniMaxH3TurboPipeline.__new__(MiniMaxH3TurboPipeline)
    pipeline.lora_strength, pipeline.transformer_file, pipeline._lora_digest = strength, transformer_file, "adapter"
    assert pipeline._lowres_weights() == weights
