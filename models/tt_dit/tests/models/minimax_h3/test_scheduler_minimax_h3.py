# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the MiniMax-H3 rectified-flow Euler scheduler: t = 1 - sigma (1 is clean),
x0 = x_t + sigma*v (a plus), and a sigma grid whose terminal 0 counts toward the step count.
Golden digests come from a run verified bit-exact against the minimax-h3 diffusers branch."""

import hashlib

import pytest
import torch

from ....pipelines.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

# shift -> (num sigmas, model evaluations, sha256[:16] of sigmas) at 50 grid points
GOLDEN = {
    12.0: (50, 49, "a7deafcaa754cc05"),
    3.0: (50, 49, "2e303820aa08a609"),
}

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def _digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("shift", [VIDEO_SHIFT, AUDIO_SHIFT])
def test_schedule_matches_golden(shift):
    num_sigmas, evaluations, sigma_sha = GOLDEN[shift]
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(50)

    assert scheduler.sigmas.numel() == num_sigmas
    assert _digest(scheduler.sigmas) == sigma_sha
    assert scheduler.num_inference_steps == evaluations
    assert scheduler.timesteps.numel() == evaluations


@pytest.mark.parametrize("shift", [VIDEO_SHIFT, AUDIO_SHIFT])
def test_sigma_grid_conventions(shift):
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(50)
    sigmas = scheduler.sigmas

    assert sigmas[0].item() == 1.0
    assert sigmas[-1].item() == 0.0
    assert bool((sigmas[1:] < sigmas[:-1]).all())
    assert torch.equal(scheduler.timesteps, 1.0 - sigmas[:-1])
    assert bool((scheduler.timesteps[1:] > scheduler.timesteps[:-1]).all())
    assert scheduler.timesteps[0].item() == 0.0


def test_video_and_audio_schedules_differ():
    video = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    audio = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
    video.set_timesteps(50)
    audio.set_timesteps(50)

    assert not torch.equal(video.sigmas, audio.sigmas)
    assert video.sigmas[25] > audio.sigmas[25]


def test_step_uses_data_ward_velocity():
    scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    scheduler.set_timesteps(50)
    sample = torch.zeros(1, 8)
    velocity = torch.ones(1, 8)

    stepped = scheduler.step(velocity, scheduler.timesteps[0], sample)
    assert bool((stepped > 0).all())


@pytest.mark.parametrize("timestep", [0.0, 0.5, MINIMAX_H3_KEYFRAME_NOISE_AUG, 1.0])
def test_scale_noise_is_the_forward_process(timestep):
    """noise_aug is a level, not a schedule entry -- never looked up in self.timesteps."""
    scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    sample = torch.randn(4, 96, generator=torch.Generator().manual_seed(7))
    noise = torch.randn(4, 96, generator=torch.Generator().manual_seed(8))

    out = scheduler.scale_noise(sample, timestep, noise)
    expected_t = torch.tensor(timestep, dtype=torch.float32)
    assert torch.equal(out, expected_t * sample + (1.0 - expected_t) * noise)


def test_scale_noise_needs_no_schedule():
    """It must work before set_timesteps, since conditioning is prepared first."""
    scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    sample = torch.ones(2, 8)
    out = scheduler.scale_noise(sample, 1.0, torch.zeros(2, 8))
    assert torch.equal(out, sample)


@pytest.mark.parametrize("shift", [VIDEO_SHIFT, AUDIO_SHIFT])
def test_explicit_sigmas_used_verbatim(shift):
    scheduler = MiniMaxH3Scheduler(shift=shift)
    sigmas = [1.0, 0.6, 0.3, 0.0]
    scheduler.set_timesteps(sigmas=sigmas)
    assert torch.equal(scheduler.sigmas, torch.tensor(sigmas, dtype=torch.float32))
    assert scheduler.num_inference_steps == 3


def test_matches_diffusers_reference():
    reference = pytest.importorskip(
        "diffusers.schedulers.scheduling_minimax_h3",
        reason="requires the minimax-h3 diffusers branch",
    )
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        ours = MiniMaxH3Scheduler(shift=shift)
        theirs = reference.MiniMaxH3Scheduler(shift=shift)
        ours.set_timesteps(50)
        theirs.set_timesteps(50)
        assert torch.equal(ours.sigmas, theirs.sigmas)
        assert torch.equal(ours.timesteps, theirs.timesteps)

        sample_ours = torch.randn(2, 128, generator=torch.Generator().manual_seed(11))
        sample_theirs = sample_ours.clone()
        generator = torch.Generator().manual_seed(12)
        for index, timestep in enumerate(ours.timesteps):
            velocity = torch.randn(2, 128, generator=generator)
            sample_ours = ours.step(velocity, timestep, sample_ours)
            sample_theirs = theirs.step(velocity, theirs.timesteps[index], sample_theirs, return_dict=False)[0]
        assert torch.equal(sample_ours, sample_theirs)
