# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference DiT denoise loop for demo A/B vs TTNN."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from models.experimental.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import (
    dit_init_latents_host_f32,
    euler_subtract_v_dt_host,
)

_C_LAT = 64


def run_torch_denoise_loop(
    *,
    pipe,
    t_schedule: list[float],
    frames: int,
    enc_hs: torch.Tensor,
    ctx_lat: torch.Tensor,
    null_emb: torch.Tensor | None,
    do_cfg: bool,
    seed: int,
    cfg_fn: Optional[Callable[[int, float, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Euler denoise over :class:`AceStepV15TorchPipeline` (production host sampler semantics)."""
    num_steps = len(t_schedule)
    if num_steps < 1:
        raise ValueError("t_schedule must have at least one timestep")

    xt = dit_init_latents_host_f32(
        batch=1,
        frames=int(frames),
        channels=_C_LAT,
        seed=int(seed),
    )
    ctx_bf = ctx_lat.to(dtype=torch.bfloat16)
    enc_bf = enc_hs.to(dtype=torch.bfloat16)
    null_bf = null_emb.to(dtype=torch.bfloat16) if null_emb is not None else None

    with torch.inference_mode():
        for step_idx in range(num_steps - 1):
            t_curr = float(t_schedule[step_idx])
            t_next = float(t_schedule[step_idx + 1])
            dt = t_curr - t_next
            xt_bf = xt.to(dtype=torch.bfloat16)

            vt_cond = pipe.forward(
                xt_bt64=xt_bf,
                context_latents_bt128=ctx_bf,
                timestep_index=step_idx,
                encoder_hidden_states_btd=enc_bf,
            )
            if do_cfg and null_bf is not None:
                vt_uncond = pipe.forward(
                    xt_bt64=xt_bf,
                    context_latents_bt128=ctx_bf,
                    timestep_index=step_idx,
                    encoder_hidden_states_btd=null_bf,
                )
                vt = cfg_fn(step_idx, t_curr, xt_bf, vt_cond, vt_uncond) if cfg_fn is not None else vt_cond
            else:
                vt = vt_cond

            xt = euler_subtract_v_dt_host(xt=xt.float(), vt=vt.float(), dt=dt)

        step_idx = num_steps - 1
        t_curr = float(t_schedule[-1])
        xt_bf = xt.to(dtype=torch.bfloat16)
        vt_cond = pipe.forward(
            xt_bt64=xt_bf,
            context_latents_bt128=ctx_bf,
            timestep_index=step_idx,
            encoder_hidden_states_btd=enc_bf,
        )
        if do_cfg and null_bf is not None:
            vt_uncond = pipe.forward(
                xt_bt64=xt_bf,
                context_latents_bt128=ctx_bf,
                timestep_index=step_idx,
                encoder_hidden_states_btd=null_bf,
            )
            vt = cfg_fn(step_idx, t_curr, xt_bf, vt_cond, vt_uncond) if cfg_fn is not None else vt_cond
        else:
            vt = vt_cond

        xt = euler_subtract_v_dt_host(xt=xt.float(), vt=vt.float(), dt=t_curr)

    return xt.float()
