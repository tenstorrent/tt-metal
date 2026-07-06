# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler

from models.tt_dit.tests.models.bria_fibo.test_vae import _fibo_local

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _calculate_shift(image_seq_len, scheduler):
    base = scheduler.config.get("base_image_seq_len", 256)
    mx = scheduler.config.get("max_image_seq_len", 4096)
    bs = scheduler.config.get("base_shift", 0.5)
    ms = scheduler.config.get("max_shift", 1.15)
    m = (ms - bs) / (mx - base)
    return image_seq_len * m + (bs - m * base)


def test_fibo_solver_matches_diffusers():
    from models.tt_dit.solvers.euler import EulerSolver

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(_fibo_local(), subfolder="scheduler")
    assert sched.config.use_dynamic_shifting and sched.config.base_shift == 0.5 and sched.config.max_shift == 1.15
    n, seq_len = 30, 4096
    mu = _calculate_shift(seq_len, sched)
    sched.set_timesteps(sigmas=np.linspace(1.0, 1 / n, n), mu=mu)
    sigmas = sched.sigmas.tolist()
    solver = EulerSolver()
    solver.set_schedule(sigmas)
    assert len(sigmas) == n + 1
    # Verify dynamic shifting: mu should equal max_shift for seq_len == max_image_seq_len
    assert abs(mu - sched.config.max_shift) < 1e-9
    # Verify solver schedule mirrors diffusers sigmas exactly
    for i, (s_solver, s_sched) in enumerate(zip(solver._sigmas, sched.sigmas.tolist())):
        assert abs(s_solver - s_sched) < 1e-6, f"sigma mismatch at index {i}: {s_solver} vs {s_sched}"
