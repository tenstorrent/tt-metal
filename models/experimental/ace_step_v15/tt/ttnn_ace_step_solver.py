# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 4a: the flow-matching denoising loop that drives the ACE-Step 1.5 DiT.

Block boundary (master doc §3.8):

    latents        [1, 1, T, 64]    x_1 ~ N(0, I) at t = 1
    context_latents[1, 1, T, 128]   cat([src_latents(64), chunk_masks(64)], -1)
    encoder_hidden_states [1, 1, enc_L, 2048]
      ->  final latents [1, 1, T, 64]  at t = 0, ready for the VAE decoder

Everything here was derived from ``golden/solver/s128`` and ``golden/pipeline/s128`` rather than
read off the reference source, then checked against it:

*   **The update is plain explicit Euler**, ``x_{i+1} = x_i + (t_{i+1} - t_i) * v_i``. Fitting
    ``dt`` per step by least squares against the goldens recovers exactly ``t_{i+1} - t_i`` with a
    residual of ~1e-7 (fp32 rounding), for all 8 steps. This is precisely
    ``tt_dit.solvers.euler.EulerSolver.step``, so that class is reused verbatim rather than
    reimplemented.
*   **The schedule is closed-form.** ``t_i = shift * s_i / (1 + (shift - 1) * s_i)`` with
    ``s_i = 1 - i / N``. At ``N=8, shift=3`` this reproduces the golden
    ``[1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]`` to fp32. The schedule handed to the
    solver appends the terminal ``0.0``, so it has ``N + 1`` entries as ``Solver.set_schedule``
    expects.
*   **``timestep_r`` is degenerate.** All 8 golden calls have ``timestep_r == timestep``; the DiT
    module therefore takes no ``timestep_r`` argument and folds its path into the weights. Same
    class of dead parameter as ``fix_nfe`` (see ACE_STEP_1_5_VARIANTS.md).
*   **No CFG.** Turbo runs at ``guidance_scale=1.0``, so there is one DiT call per step and no
    unconditional branch. A non-turbo variant would need two calls per step here.

Cross-attention K/V is step-invariant, so it is computed **once** before the loop
(``precompute_cross_kv``) rather than 8 times inside it -- that is 24 layers x 2 projections of
work saved per step.

TRAP-13 note: the DiT's SDPA runs with ``fp32_dest_acc_en=False`` (see ``sdpa_compute_config``).
Do not "optimise" that back on -- it deadlocks the windowed layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn
from models.tt_dit.solvers.euler import EulerSolver

from .ttnn_ace_step_common import capture_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .ttnn_ace_step_dit import AceStepTransformer1DModel

# Turbo defaults, from golden/solver/s<S>/meta.pt -> call_kwargs.
TURBO_NUM_STEPS = 8
TURBO_SHIFT = 3.0


def turbo_timesteps(num_steps: int = TURBO_NUM_STEPS, shift: float = TURBO_SHIFT) -> list[float]:
    """The ``num_steps`` timesteps at which the DiT is evaluated, descending from 1.0.

    ``t_i = shift * s_i / (1 + (shift - 1) * s_i)``, ``s_i = 1 - i / num_steps``. ``shift`` warps a
    uniform schedule toward t=1 (more steps spent on coarse structure); ``shift=1`` is uniform.
    """
    out = []
    for i in range(num_steps):
        s = 1.0 - i / num_steps
        out.append(shift * s / (1.0 + (shift - 1.0) * s))
    return out


def sigma_schedule(num_steps: int = TURBO_NUM_STEPS, shift: float = TURBO_SHIFT) -> list[float]:
    """:func:`turbo_timesteps` plus the terminal 0.0 -- ``num_steps + 1`` entries.

    ``Solver.set_schedule`` indexes ``[step]`` and ``[step + 1]``, so the last step needs an
    explicit endpoint. The goldens confirm the final step uses ``dt = 0.0 - 0.3 = -0.3``.
    """
    return [*turbo_timesteps(num_steps, shift), 0.0]


def make_solver(num_steps: int = TURBO_NUM_STEPS, shift: float = TURBO_SHIFT) -> EulerSolver:
    solver = EulerSolver()
    solver.set_schedule(sigma_schedule(num_steps, shift))
    return solver


def denoise(
    model: AceStepTransformer1DModel,
    latents_11TC: ttnn.Tensor,
    context_latents_11TC: ttnn.Tensor | None,
    *,
    encoder_hidden_states_11LC: ttnn.Tensor | None = None,
    cross_kv: list[tuple[ttnn.Tensor, ttnn.Tensor]] | None = None,
    num_steps: int = TURBO_NUM_STEPS,
    shift: float = TURBO_SHIFT,
    timesteps: Sequence[float] | None = None,
    capture: dict | None = None,
) -> ttnn.Tensor:
    """Run the full denoising loop and return the clean latents at t=0.

    Args:
        model: the DiT.
        latents_11TC: ``x_1``, ``[1, 1, T, 64]``, already on device.
        context_latents_11TC: ``[1, 1, T, 128]``; may be ``None`` for golden replay where
            ``latents_11TC`` is the pre-concatenated patchify input.
        encoder_hidden_states_11LC: condition-encoder output. Ignored when ``cross_kv`` is given.
        cross_kv: pre-computed cross-attention K/V. Computed here if omitted.
        num_steps, shift: schedule parameters; ignored when ``timesteps`` is given.
        timesteps: explicit schedule override. Must be **descending** and length ``num_steps``;
            the terminal 0.0 is appended automatically. Used by the PCC gate to replay the exact
            golden schedule instead of recomputing it.
        capture: records ``step_latents.call{i}`` (the latent *after* step ``i``) and
            ``velocity.call{i}``, matching the golden key names so the gate can compare per step
            and localise which step diverges.

    Returns:
        Final latents ``[1, 1, T, 64]``. Intermediates are deallocated as the loop proceeds, so
        only one latent and one velocity are live at a time.
    """
    if timesteps is not None:
        ts = list(timesteps)
        if len(ts) != num_steps:
            num_steps = len(ts)
        solver = EulerSolver()
        solver.set_schedule([*ts, 0.0])
    else:
        ts = turbo_timesteps(num_steps, shift)
        solver = make_solver(num_steps, shift)

    if cross_kv is None:
        if encoder_hidden_states_11LC is None:
            msg = "denoise() needs either cross_kv or encoder_hidden_states_11LC"
            raise ValueError(msg)
        cross_kv = model.precompute_cross_kv(encoder_hidden_states_11LC)

    x = latents_11TC
    for i, t in enumerate(ts):
        velocity = model(
            x,
            context_latents_11TC,
            t,
            cross_kv=cross_kv,
        )
        capture_tensor(capture, f"velocity.call{i}", velocity)

        x_next = solver.step(step=i, latent=x, velocity_pred=velocity)
        ttnn.deallocate(velocity)
        # `x` is the caller's tensor on the first iteration -- leave that one alone.
        if i > 0:
            ttnn.deallocate(x)
        x = x_next
        capture_tensor(capture, f"step_latents.call{i}", x)

    return x
