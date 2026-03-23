# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from diffusers.pipelines.mochi.pipeline_mochi import linear_quadratic_schedule
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

import ttnn
from models.tt_dit.solvers import schedules
from models.tt_dit.solvers.euler import EulerSolver
from models.tt_dit.solvers.unipc import UniPCSolver, UniPCVariant
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

_NUM_STEPS = 17


# Schedule tests: linear


def test_linear() -> None:
    """linear() should produce evenly spaced sigmas."""
    expected = torch.arange(_NUM_STEPS, -1, -1) / _NUM_STEPS

    schedule = schedules.linear(_NUM_STEPS)

    assert torch.equal(expected, torch.tensor(schedule.sigmas))


def test_linear_matches_flow_match_euler() -> None:
    """linear(sigma_small=0.001) should match FlowMatchEulerDiscreteScheduler defaults."""
    reference = FlowMatchEulerDiscreteScheduler()
    reference.set_timesteps(_NUM_STEPS)

    schedule = schedules.linear(_NUM_STEPS, sigma_small=0.001)

    assert torch.equal(reference.sigmas, torch.tensor(schedule.sigmas))


def test_linear_alpha_sigma_sum_to_one() -> None:
    schedule = schedules.linear(_NUM_STEPS)
    for s, a in zip(schedule.sigmas, schedule.alphas, strict=True):
        assert s + a == 1


# Schedule tests: shifted_linear


@pytest.mark.parametrize("shift", [1.0, 3.0, 5.0, 12.0])
def test_shifted_linear_matches_flow_match_euler_static(shift: float) -> None:
    """shifted_linear() should match FlowMatchEulerDiscreteScheduler defaults.

    Diffusers double-applies the shift: once in __init__ (shifting sigma_small from 0.001
    to shift*0.001/(1+(shift-1)*0.001)), then again in set_timesteps. We match by using
    the pre-shifted sigma_small.
    """
    reference = FlowMatchEulerDiscreteScheduler(shift=shift)
    reference.set_timesteps(_NUM_STEPS)

    schedule = schedules.shifted_linear(_NUM_STEPS, shift=shift, sigma_small=reference.sigma_small)

    assert torch.equal(reference.sigmas, torch.tensor(schedule.sigmas))


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_shifted_linear_matches_flow_match_euler_dynamic(mu: float) -> None:
    """shifted_linear(shift=exp(mu)) should match dynamic shifting with exponential type."""
    reference = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, time_shift_type="exponential")
    reference.set_timesteps(_NUM_STEPS, mu=mu)

    sigma_small = np.float32(0.001).item()
    schedule = schedules.shifted_linear(_NUM_STEPS, shift=math.exp(mu), sigma_small=sigma_small)

    assert torch.equal(reference.sigmas, torch.tensor(schedule.sigmas))


@pytest.mark.parametrize("shift", [3.0, 12.0])
def test_shifted_linear_matches_unipc_flow_sigmas(shift: float) -> None:
    """shifted_linear() should match UniPCMultistepScheduler in flow-matching mode.

    UniPCMultistepScheduler uses linspace(1, 0.001, N+1)[:-1] as base, giving an
    effective sigma_small of 0.001 + 0.999/N, and subtracts 1e-6 from the first sigma
    to avoid log(1) = 0.
    """
    reference = UniPCMultistepScheduler(use_flow_sigmas=True, flow_shift=shift, prediction_type="flow_prediction")
    reference.set_timesteps(_NUM_STEPS)

    # Diffusers' effective sigma_small from its linspace(1, 0.001, N+1) discretization.
    unipc_sigma_small = 0.001 + 0.999 / _NUM_STEPS
    schedule = schedules.shifted_linear(_NUM_STEPS, shift=shift, sigma_small=unipc_sigma_small)

    schedule.sigmas[0] -= 1e-6

    assert torch.equal(reference.sigmas, torch.tensor(schedule.sigmas))


def test_shifted_linear_alpha_sigma_sum_to_one() -> None:
    schedule = schedules.shifted_linear(_NUM_STEPS, shift=5.0)
    for s, a in zip(schedule.sigmas, schedule.alphas, strict=True):
        assert s + a == 1


# Schedule tests: linear_quadratic


def test_linear_quadratic_matches_mochi_diffusers() -> None:
    """linear_quadratic() should match diffusers' linear_quadratic_schedule."""
    threshold_noise = 0.025

    reference = linear_quadratic_schedule(_NUM_STEPS, threshold_noise)

    schedule = schedules.linear_quadratic(_NUM_STEPS, threshold_noise=threshold_noise)

    assert schedule.sigmas == [*reference, 0.0]


# Euler solver tests


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_euler_matches_diffusers(mesh_device: ttnn.MeshDevice) -> None:
    """EulerSolver should match FlowMatchEulerDiscreteScheduler at every step."""
    torch.manual_seed(0)

    torch_latent = torch.randn(1, 1, 32, 32)

    reference = FlowMatchEulerDiscreteScheduler()
    reference.set_timesteps(_NUM_STEPS)

    schedule = schedules.linear(_NUM_STEPS, sigma_small=0.001)
    solver = EulerSolver()

    ref = torch_latent.clone()
    latent = tensor.from_torch(torch_latent, device=mesh_device, dtype=ttnn.float32)

    for step_idx in range(_NUM_STEPS):
        torch_velocity = torch.randn_like(torch_latent)

        # reference step
        ref = reference.step(torch_velocity, reference.timesteps[step_idx], ref, return_dict=False)[0]

        # our step
        velocity = tensor.from_torch(torch_velocity, device=mesh_device, dtype=ttnn.float32)
        latent = solver.step(
            step=step_idx,
            latent=latent,
            sigmas=schedule.sigmas,
            alphas=schedule.alphas,
            velocity_pred=velocity,
        )

        result_ours = ttnn.to_torch(latent)
        assert_quality(result_ours, ref, pcc=0.999_999, relative_rmse=1e-5)


# UniPC solver tests


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("variant", [UniPCVariant.B1, UniPCVariant.B2])
@pytest.mark.parametrize("shift", [5.0, 12.0])
def test_unipc_matches_diffusers(mesh_device: ttnn.MeshDevice, variant: UniPCVariant, shift: float) -> None:
    """UniPCSolver should match UniPCMultistepScheduler at every step."""
    torch.manual_seed(0)

    torch_latent = torch.randn(1, 1, 32, 32)

    scheduler = UniPCMultistepScheduler(
        use_flow_sigmas=True,
        flow_shift=shift,
        prediction_type="flow_prediction",
        solver_order=2,
        solver_type="bh1" if variant is UniPCVariant.B1 else "bh2",
    )
    scheduler.set_timesteps(_NUM_STEPS)

    (sigmas, alphas) = schedules.shifted_linear(_NUM_STEPS, shift=shift, sigma_small=0.001 + 0.999 / _NUM_STEPS)
    sigmas[0] -= 1e-6

    # diffusers uses float32 for the schedule
    sigmas = torch.tensor(sigmas, dtype=torch.float32).tolist()
    alphas = (1 - torch.tensor(sigmas, dtype=torch.float32)).tolist()

    solver = UniPCSolver(order=2, variant=variant)

    ref = torch_latent.clone()
    latent = tensor.from_torch(torch_latent, device=mesh_device, dtype=ttnn.float32)

    for step_idx in range(_NUM_STEPS):
        if step_idx == _NUM_STEPS - 1 and variant is UniPCVariant.B1:
            # Diffusers bh1 produces NaN on the final step; skip.
            break

        torch_velocity = torch.randn_like(torch_latent)

        # reference step
        ref = scheduler.step(torch_velocity, scheduler.timesteps[step_idx], ref, return_dict=False)[0]

        # our step
        velocity = tensor.from_torch(torch_velocity, device=mesh_device, dtype=ttnn.float32)
        latent = solver.step(
            step=step_idx,
            latent=latent,
            sigmas=sigmas,
            alphas=alphas,
            velocity_pred=velocity,
        )

        result_ours = ttnn.to_torch(latent)
        assert_quality(result_ours, ref, pcc=1 - 3e-14, relative_rmse=3e-7)
