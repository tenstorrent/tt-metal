# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from diffusers.pipelines.mochi.pipeline_mochi import linear_quadratic_schedule
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

import ttnn
from models.tt_dit.solvers.euler import EulerSolver
from models.tt_dit.solvers.unipc import UniPCSolver, UniPCVariant
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

_NUM_STEPS = 17


# Euler solver tests


def _motif_sigmas(*, step_count: int, linear_quadratic_emulating_steps: int) -> torch.Tensor:
    assert step_count % 2 == 0

    s = step_count
    n = linear_quadratic_emulating_steps
    a = s // 2 / n - 1

    sigmas1 = torch.linspace(1, 0, n + 1)[: s // 2]
    sigmas2 = torch.linspace(0, 1, s // 2 + 1).pow(2) * a - a

    return torch.concat([sigmas1, sigmas2])


def _assert_euler_matches_scheduler(
    mesh_device: ttnn.MeshDevice,
    *,
    schedule_kwargs: dict[str, object],
    expected_timesteps: torch.Tensor | None = None,
    expected_sigmas: torch.Tensor | None = None,
) -> None:
    solver = EulerSolver()
    ref_scheduler = FlowMatchEulerDiscreteScheduler()

    solver.set_schedule(**schedule_kwargs)
    ref_scheduler.set_timesteps(**schedule_kwargs)

    assert solver.timesteps is not None
    assert torch.allclose(solver.timesteps, ref_scheduler.timesteps)
    assert solver.sigmas == ref_scheduler.sigmas.tolist()
    assert solver.alphas == (1.0 - ref_scheduler.sigmas).tolist()

    if expected_timesteps is not None:
        assert torch.allclose(solver.timesteps, expected_timesteps)

    if expected_sigmas is not None:
        assert solver.sigmas == expected_sigmas.tolist()

    torch.manual_seed(0)
    torch_latent = torch.randn(1, 1, 32, 32)
    ref = torch_latent.clone()
    latent = tensor.from_torch(torch_latent, device=mesh_device, dtype=ttnn.float32)

    for step_idx in range(len(ref_scheduler.timesteps)):
        torch_velocity = torch.randn_like(torch_latent)
        ref = ref_scheduler.step(torch_velocity, ref_scheduler.timesteps[step_idx], ref, return_dict=False)[0]

        velocity = tensor.from_torch(torch_velocity, device=mesh_device, dtype=ttnn.float32)
        latent = solver.step(step=step_idx, latent=latent, velocity_pred=velocity)

        result_ours = ttnn.to_torch(latent)
        assert_quality(result_ours, ref, pcc=0.999_999, relative_rmse=1e-5)


def test_solver_set_schedule_caches_scheduler_outputs() -> None:
    """set_schedule should forward scheduler kwargs and cache schedule outputs."""
    scheduler = FlowMatchEulerDiscreteScheduler()
    solver = EulerSolver(scheduler=scheduler)

    sigmas = torch.linspace(1.0, 1 / _NUM_STEPS, _NUM_STEPS)
    mu = 0.75
    solver.set_schedule(sigmas=sigmas, mu=mu)

    assert solver.timesteps is scheduler.timesteps
    assert solver.sigmas == scheduler.sigmas.tolist()
    assert solver.alphas == (1.0 - scheduler.sigmas).tolist()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_euler_set_schedule_with_sigmas_and_mu_matches_scheduler(mesh_device: ttnn.MeshDevice) -> None:
    """EulerSolver should match scheduler outputs for the flux/qwen sigmas+mu path."""
    sigmas = torch.linspace(1.0, 1 / _NUM_STEPS, _NUM_STEPS).tolist()
    _assert_euler_matches_scheduler(mesh_device, schedule_kwargs={"sigmas": sigmas, "mu": 0.75})


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_euler_set_schedule_with_mochi_sigmas_matches_scheduler(mesh_device: ttnn.MeshDevice) -> None:
    """EulerSolver should match scheduler outputs for the mochi custom sigma path."""
    sigmas = linear_quadratic_schedule(_NUM_STEPS, 0.025)
    _assert_euler_matches_scheduler(mesh_device, schedule_kwargs={"sigmas": sigmas})


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_euler_set_schedule_with_motif_sigmas_matches_main(mesh_device: ttnn.MeshDevice) -> None:
    """EulerSolver should reproduce motif's main-branch timesteps and sigma schedule."""
    step_count = 18
    sigmas = _motif_sigmas(step_count=step_count, linear_quadratic_emulating_steps=1000)
    _assert_euler_matches_scheduler(
        mesh_device,
        schedule_kwargs={"sigmas": sigmas[:-1].tolist()},
        expected_timesteps=sigmas[:-1] * 1000,
        expected_sigmas=sigmas,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_euler_matches_diffusers(mesh_device: ttnn.MeshDevice) -> None:
    """EulerSolver should match FlowMatchEulerDiscreteScheduler at every step."""
    torch.manual_seed(0)

    torch_latent = torch.randn(1, 1, 32, 32)

    solver = EulerSolver()
    solver.set_schedule(_NUM_STEPS)
    scheduler = solver.scheduler

    ref = torch_latent.clone()
    latent = tensor.from_torch(torch_latent, device=mesh_device, dtype=ttnn.float32)

    for step_idx in range(_NUM_STEPS):
        torch_velocity = torch.randn_like(torch_latent)

        ref = scheduler.step(torch_velocity, scheduler.timesteps[step_idx], ref, return_dict=False)[0]

        velocity = tensor.from_torch(torch_velocity, device=mesh_device, dtype=ttnn.float32)
        latent = solver.step(step=step_idx, latent=latent, velocity_pred=velocity)

        result_ours = ttnn.to_torch(latent)
        assert_quality(result_ours, ref, pcc=0.999_999, relative_rmse=1e-5)


# UniPC solver tests


def test_unipc_constructor_validation() -> None:
    """UniPCSolver should reject incompatible scheduler configurations."""
    with pytest.raises(ValueError, match="UniPCMultistepScheduler"):
        UniPCSolver(scheduler=FlowMatchEulerDiscreteScheduler())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="use_flow_sigmas=True"):
        UniPCSolver(scheduler=UniPCMultistepScheduler())

    with pytest.raises(ValueError, match="only order 1 and 2 are supported"):
        UniPCSolver(
            scheduler=UniPCMultistepScheduler(
                use_flow_sigmas=True,
                prediction_type="flow_prediction",
                solver_order=3,
            )
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_unipc_set_schedule_resets_state(mesh_device: ttnn.MeshDevice) -> None:
    """set_schedule should reset logical history without reallocating state buffers."""
    solver = UniPCSolver(
        scheduler=UniPCMultistepScheduler(
            use_flow_sigmas=True,
            prediction_type="flow_prediction",
            solver_order=2,
            solver_type=UniPCVariant.BH2.value,
        )
    )
    solver.set_schedule(_NUM_STEPS)

    latent = tensor.from_torch(torch.randn(1, 1, 32, 32), device=mesh_device, dtype=ttnn.float32)
    velocity = tensor.from_torch(torch.randn(1, 1, 32, 32), device=mesh_device, dtype=ttnn.float32)
    solver.step(step=0, latent=latent, velocity_pred=velocity)

    assert solver._state is not None
    clean_preds = solver._state.clean_preds
    corrected = solver._state.corrected
    assert solver._state.oldest_idx == 1

    solver.set_schedule(_NUM_STEPS)

    assert solver._state is not None
    assert solver._state.clean_preds == clean_preds
    assert solver._state.corrected is corrected
    assert solver._state.oldest_idx == 0


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("variant", [UniPCVariant.BH1, UniPCVariant.BH2])
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
        solver_type=variant.value,
    )
    solver = UniPCSolver(scheduler=scheduler)
    solver.set_schedule(_NUM_STEPS)

    ref = torch_latent.clone()
    latent = tensor.from_torch(torch_latent, device=mesh_device, dtype=ttnn.float32)

    for step_idx in range(_NUM_STEPS):
        if step_idx == _NUM_STEPS - 1 and variant is UniPCVariant.BH1:
            # Diffusers bh1 produces NaN on the final step; skip.
            break

        torch_velocity = torch.randn_like(torch_latent)

        ref = scheduler.step(torch_velocity, scheduler.timesteps[step_idx], ref, return_dict=False)[0]

        velocity = tensor.from_torch(torch_velocity, device=mesh_device, dtype=ttnn.float32)
        latent = solver.step(step=step_idx, latent=latent, velocity_pred=velocity)

        result_ours = ttnn.to_torch(latent)
        assert_quality(result_ours, ref, pcc=1 - 3e-10, relative_rmse=3e-7)
