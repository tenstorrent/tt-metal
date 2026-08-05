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
from models.tt_dit.solvers.factory import CustomSigmaScheduler, solver_for_scheduler
from models.tt_dit.solvers.unipc import UniPCSolver, UniPCVariant
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

_NUM_STEPS = 17


def _unipc_scheduler(*, variant: UniPCVariant = UniPCVariant.BH2, flow_shift: float = 3.0) -> UniPCMultistepScheduler:
    return UniPCMultistepScheduler(
        use_flow_sigmas=True,
        flow_shift=flow_shift,
        prediction_type="flow_prediction",
        solver_order=2,
        solver_type=variant.value,
    )


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
    ref_scheduler = FlowMatchEulerDiscreteScheduler()

    solver = EulerSolver(scheduler=ref_scheduler)
    solver.set_schedule(**schedule_kwargs)

    if expected_timesteps is not None:
        assert torch.allclose(ref_scheduler.timesteps, expected_timesteps)

    if expected_sigmas is not None:
        assert ref_scheduler.sigmas.tolist() == expected_sigmas.tolist()

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


def test_euler_without_scheduler_accepts_only_sigmas(expect_error) -> None:
    """A scheduler-less EulerSolver takes sigmas verbatim and rejects scheduler arguments."""
    solver = EulerSolver()
    sigmas = [1.0, 0.5, 0.0]

    solver.set_schedule(sigmas=sigmas)
    assert solver.sigmas == (1.0, 0.5, 0.0)
    assert solver.alphas == (0.0, 0.5, 1.0)
    assert solver.timesteps == (1000.0, 500.0)

    with expect_error(ValueError, "accepts only `sigmas`"):
        solver.set_schedule(_NUM_STEPS)
    with expect_error(ValueError, "accepts only `sigmas`"):
        solver.set_schedule(sigmas=sigmas, shift=5.0)
    with expect_error(ValueError, "accepts only `sigmas`"):
        solver.set_schedule(sigmas=sigmas, mu=0.75)


def test_solver_without_schedule_rejects_access(expect_error) -> None:
    """Reading the schedule before setting one should fail loudly."""
    with expect_error(ValueError, "schedule must be set"):
        _ = EulerSolver().sigmas


@pytest.mark.parametrize("family", ["unipc", "euler"])
def test_omitted_shift_restores_construction_value(family: str) -> None:
    """Every solver spells the per-run shift `shift`, and it must not persist into the next run."""
    default_shift = 3.0
    if family == "unipc":
        solver = solver_for_scheduler(_unipc_scheduler(flow_shift=default_shift))
    else:
        solver = solver_for_scheduler(FlowMatchEulerDiscreteScheduler(shift=default_shift))

    solver.set_schedule(_NUM_STEPS)
    expected = solver.sigmas

    solver.set_schedule(_NUM_STEPS, shift=12.0)
    shifted = solver.sigmas
    assert shifted != expected

    solver.set_schedule(_NUM_STEPS)
    assert solver.sigmas == expected

    solver.set_schedule(_NUM_STEPS, shift=12.0)
    assert solver.sigmas == shifted


def test_solver_for_scheduler_dispatch(expect_error) -> None:
    """The scheduler class selects the solver family, and unsupported ones are rejected."""
    assert isinstance(solver_for_scheduler(_unipc_scheduler()), UniPCSolver)
    assert isinstance(solver_for_scheduler(FlowMatchEulerDiscreteScheduler()), EulerSolver)

    with expect_error(ValueError, "use_flow_sigmas=True"):
        solver_for_scheduler(UniPCMultistepScheduler())

    with expect_error(ValueError, "no solver available"):
        solver_for_scheduler(object())


def test_custom_sigma_scheduler_dispatches_to_a_scheduler_less_euler() -> None:
    """The marker asks for Euler over sigmas the caller supplies, taken as given."""
    solver = solver_for_scheduler(CustomSigmaScheduler())
    assert isinstance(solver, EulerSolver)
    assert solver.scheduler is None

    solver.set_schedule(sigmas=[1.0, 0.5, 0.0])
    assert solver.sigmas == (1.0, 0.5, 0.0)


def test_solver_for_scheduler_takes_solver_config_from_scheduler() -> None:
    """UniPC order and variant come from the scheduler, not from the caller."""
    solver = solver_for_scheduler(_unipc_scheduler(variant=UniPCVariant.BH1))
    assert solver.order == 2  # solver_order=2 in _unipc_scheduler
    assert solver.variant is UniPCVariant.BH1


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_euler_matches_diffusers(mesh_device: ttnn.MeshDevice) -> None:
    """EulerSolver should match FlowMatchEulerDiscreteScheduler at every step."""
    torch.manual_seed(0)

    torch_latent = torch.randn(1, 1, 32, 32)

    scheduler = FlowMatchEulerDiscreteScheduler()

    solver = EulerSolver(scheduler=scheduler)
    solver.set_schedule(_NUM_STEPS)

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


def test_unipc_constructor_validation(expect_error) -> None:
    """UniPCSolver should reject unsupported orders."""
    with expect_error(ValueError, "only order 1 and 2 are supported"):
        UniPCSolver(order=3, variant=UniPCVariant.BH2, scheduler=_unipc_scheduler())


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_unipc_set_schedule_resets_state(mesh_device: ttnn.MeshDevice) -> None:
    """set_schedule should reset logical history without reallocating state buffers."""
    solver = UniPCSolver(order=2, variant=UniPCVariant.BH2, scheduler=_unipc_scheduler())
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

    scheduler = _unipc_scheduler(variant=variant, flow_shift=shift)
    solver = UniPCSolver(order=2, variant=variant, scheduler=scheduler)
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
