# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC gate for Block 4a: the denoising loop (``tt/ttnn_ace_step_solver.py``).

Block boundary (master doc §3.8):

    x_1 [1, T, 64] + context_latents [1, T, 128] + encoder_hidden_states [1, enc_L, 2048]
      ->  8 x (DiT velocity + Euler step)  ->  final latents [1, T, 64]

This is a **real-weight-only** gate: ``golden/solver/`` and ``golden/pipeline/`` were dumped from
the converted checkpoint, and there is no meaningful random-init equivalent of an 8-step
trajectory. It skips cleanly when the goldens or the checkpoint are absent.

What makes this gate different from ``test_dit_pcc``: error **accumulates** across steps. A single
DiT call at PCC 0.9996 is fine, but the same relative error compounding over 8 Euler steps is not
automatically fine, so the per-step table below is the point of the test -- it shows whether the
trajectory drifts or holds. Two independent checks per step:

* ``velocity.call{i}`` vs the golden DiT output -- is the *model* still right at step ``i``?
* ``step_latents.call{i}`` vs the golden latent -- is the *trajectory* still right after step ``i``?

The first isolates the DiT, the second the solver arithmetic on top of it. A clean velocity with a
drifting latent means the Euler step or the schedule is wrong; both drifting means the DiT is.

Two modes:

* **default (free-running)** -- feed only ``x_1``, let the loop produce every subsequent ``x_t``
  itself. This is the honest end-to-end number and what the pipeline will actually do.
* **``ACE_STEP_SOLVER_TEACHER=1``** -- reset ``x_t`` to the golden value before each step
  ("teacher forcing"), which decouples the steps so a single bad step cannot poison the rest.
  Use it to attribute a failure once the free-running mode is red.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.tests import block4_reference as B4
from models.experimental.ace_step_v15.tests import dit_reference as R
from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import (
    AceStepDiTConfig,
    Capture,
    to_device,
    to_host,
)
from models.experimental.ace_step_v15.tt.ttnn_ace_step_dit import AceStepTransformer1DModel
from models.experimental.ace_step_v15.tt.ttnn_ace_step_solver import denoise, turbo_timesteps

# The trajectory gate is the model-level one: 0.99 (master doc §5b).
TARGET_PCC = 0.99
SEQ_LENS = (R.SEQ_LEN_BLOCK,)  # S=128 / 10.24 s. S=768 exists but costs 8 full DiT passes.


def _teacher_forcing() -> bool:
    return os.environ.get("ACE_STEP_SOLVER_TEACHER", "0") == "1"


def _to_11sc(t: torch.Tensor, device) -> ttnn.Tensor:
    """``[1, X, C]`` -> device ``[1, 1, X, C]``."""
    return to_device(t.reshape(1, 1, *t.shape[-2:]), device)


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_solver_pcc(device, seq_len):
    try:
        sg = B4.SolverGoldens(seq_len)
        pg = B4.PipelineGoldens(seq_len)
        dit_goldens = R.DitGoldens(seq_len)
        state_dict = R.real_dit_state_dict()
    except (FileNotFoundError, KeyError) as exc:
        pytest.skip(f"Block 4 goldens or converted checkpoint unavailable: {exc}")

    config = AceStepDiTConfig.from_diffusers_config(sg.meta["transformer_config"])
    assert sg.meta["dit_tokens_S"] == seq_len
    num_steps = sg.num_steps
    teacher = _teacher_forcing()

    # The schedule is recomputed rather than read from meta, so a drifting closed form is caught
    # here instead of silently matching whatever was dumped.
    mine = turbo_timesteps(num_steps, float(sg.meta["call_kwargs"]["shift"]))
    sched_err = max(abs(a - b) for a, b in zip(sg.timesteps, mine))
    assert sched_err < 1e-5, f"recomputed schedule diverges from golden by {sched_err:.2e}"

    # ------------------------------------------------------------------------ TTNN model --
    tt_model = AceStepTransformer1DModel(config, mesh_device=device)
    tt_model.load_torch_state_dict(state_dict)
    tt_model.prepare_rope(seq_len)

    ctx_tt = _to_11sc(dit_goldens["kw_context_latents"], device)
    enc_tt = _to_11sc(dit_goldens["kw_encoder_hidden_states"], device)
    cross_kv = tt_model.precompute_cross_kv(enc_tt)

    keys = tuple(f"{p}.call{i}" for i in range(num_steps) for p in ("velocity", "step_latents"))

    # ------------------------------------------------------------------------ run the loop --
    if teacher:
        # One DiT call per step, each fed the *golden* x_t, so steps are fully independent.
        # The solver is deliberately not involved: with x_t supplied there is no trajectory to
        # advance, so only the velocity is meaningful here (hence the "-" latent column).
        capture = Capture(keys=keys)
        for i in range(num_steps):
            x_tt = _to_11sc(sg.x_at(i), device)
            velocity = tt_model(x_tt, ctx_tt, sg.timesteps[i], cross_kv=cross_kv)
            capture[f"velocity.call{i}"] = to_host(velocity)
            ttnn.deallocate(velocity)
            ttnn.deallocate(x_tt)
        final_host = None
    else:
        capture = Capture(keys=keys)
        x_tt = _to_11sc(sg.x_at(0), device)
        final = denoise(
            tt_model,
            x_tt,
            ctx_tt,
            cross_kv=cross_kv,
            timesteps=sg.timesteps,
            capture=capture,
        )
        final_host = to_host(final)

    # ------------------------------------------------------------------------ comparison --
    rows = []
    for i in range(num_steps):
        vk, sk = f"velocity.call{i}", f"step_latents.call{i}"
        v_pcc = l_pcc = None
        if vk in capture:
            _, p = comp_pcc(sg.velocity(i), capture[vk], pcc=0.0)
            v_pcc = float(p)
        if not teacher and sk in capture:
            _, p = comp_pcc(pg.step_latents(i), capture[sk], pcc=0.0)
            l_pcc = float(p)
        rows.append((i, sg.timesteps[i], v_pcc, l_pcc))

    print(
        f"\n=== solver PCC (S={seq_len}, T={seq_len * config.patch_size}, {num_steps} steps, "
        f"{'teacher-forced' if teacher else 'free-running'}, target {TARGET_PCC}) ==="
    )
    print(f"{'step':>4}  {'t':>8}  {'velocity':>10}  {'latent':>10}")
    for i, t, v, l in rows:
        vs = "-" if v is None else f"{v:.6f}"
        ls = "-" if l is None else f"{l:.6f}"
        print(f"{i:>4}  {t:>8.4f}  {vs:>10}  {ls:>10}")

    if final_host is not None:
        _, fp = comp_pcc(pg.final_latents, final_host, pcc=0.0)
        final_pcc = float(fp)
        print(f"{'FINAL latents':>26}  {final_pcc:.6f}")

    # Gate every step, so a mid-trajectory collapse cannot hide behind a good endpoint.
    worst = min((v for _, _, v, _ in rows if v is not None), default=1.0)
    assert worst >= TARGET_PCC, f"worst per-step velocity PCC {worst:.6f} < {TARGET_PCC}"
    if not teacher:
        worst_l = min((l for _, _, _, l in rows if l is not None), default=1.0)
        assert worst_l >= TARGET_PCC, f"worst per-step latent PCC {worst_l:.6f} < {TARGET_PCC}"
        assert final_pcc >= TARGET_PCC, f"final latents PCC {final_pcc:.6f} < {TARGET_PCC}"
