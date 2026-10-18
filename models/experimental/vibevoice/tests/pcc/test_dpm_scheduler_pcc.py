# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1c — DPM Scheduler PCC test.

Tests scheduler math alone with synthetic eps tensors (same torch.manual_seed).
Latent PCC >= 0.99 after the configured number of steps.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tt.ttnn_dpm_scheduler import (
    TTDPMSolverMultistepScheduler,
)


def _build_reference_scheduler():
    from models.experimental.vibevoice.reference.schedule.dpm_solver import DPMSolverMultistepScheduler as RefScheduler

    return RefScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        solver_order=2,
        prediction_type="v_prediction",
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,
        timestep_spacing="linspace",
    )


LATENT_SIZE = 64


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("num_steps", [10, 15, 20])
def test_dpm_scheduler_pcc(mesh_device, num_steps):
    torch.manual_seed(42)

    # Reference scheduler
    ref_sched = _build_reference_scheduler()
    ref_sched.set_timesteps(num_steps)

    # TT scheduler (same configuration)
    tt_sched = TTDPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="cosine",
        solver_order=2,
        prediction_type="v_prediction",
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,
        timestep_spacing="linspace",
    )
    tt_sched.set_timesteps(num_steps)

    # Shared initial latent
    latent_torch = torch.randn(1, LATENT_SIZE, dtype=torch.float32)
    latent_ref = latent_torch.clone()
    latent_tt = ttnn.as_tensor(
        latent_torch.to(torch.bfloat16).view(1, 1, 1, LATENT_SIZE),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Pre-generate the same noise for each step
    all_eps = [torch.randn(1, LATENT_SIZE, dtype=torch.float32) for _ in range(num_steps)]

    # --- Reference loop ---
    for step_idx, t_val in enumerate(ref_sched.timesteps):
        eps = all_eps[step_idx]
        # Pass raw model output — step() calls convert_model_output internally
        result = ref_sched.step(eps, t_val, latent_ref)
        latent_ref = result.prev_sample

    # --- TT loop ---
    for step_idx in range(num_steps):
        eps = all_eps[step_idx]
        eps_tt = ttnn.as_tensor(
            eps.to(torch.bfloat16).view(1, 1, 1, LATENT_SIZE),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        latent_tt = tt_sched.step(eps_tt, latent_tt)

    # Compare
    latent_tt_torch = ttnn.to_torch(latent_tt).to(torch.float32).view(1, LATENT_SIZE)
    latent_ref_flat = latent_ref.view(1, LATENT_SIZE).to(torch.float32)

    passed, pcc_val = comp_pcc(latent_ref_flat, latent_tt_torch, pcc=0.99)
    assert passed, f"DPM scheduler PCC {pcc_val:.6f} < 0.99 (num_steps={num_steps})"
