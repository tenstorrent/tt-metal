# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full Euler-step TTNN trace: pre-DiT cast, DiT body, CFG/APG/ADG, and Euler update.

Used by :class:`~models.experimental.ace_step_v1_5.ttnn_impl.e2e_model_tt._E2EDenoiseTrace`
when ``use_full_step=True`` (default with ``--use-trace``). Persistent buffers are
refreshed on CQ1 each replay; scalars ``dt`` and ``sigma`` (ADG) are copied as
single-element tiles so the captured graph stays shape-stable.

APG momentum is disabled inside the trace (``momentum_buffer=None``) so the graph
does not mutate host-visible state across replays.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

import ttnn

from .dit_sampling_ttnn import (
    adg_guidance_velocity_ttnn,
    apg_guidance_velocity_ttnn,
    concat_duplicate_batch,
    euler_subtract_v_dt,
    euler_subtract_v_dt_from_tile,
    fp32_tile_to_row_bf16,
    slice_batch_btc,
    typecast_bf16_any_to_fp32_tile,
)


def denoise_full_step_trace_graph(
    *,
    pipe: Any,
    xt_tile: ttnn.Tensor,
    temb: ttnn.Tensor,
    tp: ttnn.Tensor,
    enc_buf: ttnn.Tensor,
    ctx_buf: ttnn.Tensor,
    mask_buf: Optional[ttnn.Tensor],
    acoustic_out: ttnn.Tensor,
    dt_scalar_tile: ttnn.Tensor,
    mem: Any,
    frames_i: int,
    c_lat: int,
    do_cfg: bool,
    guidance_scale: float,
    apply_cfg: bool,
) -> ttnn.Tensor:
    """Trace-safe full step (no ADG): pre-DiT + body + APG/no-CFG + Euler with ``dt`` tensor."""
    xt_row = fp32_tile_to_row_bf16(xt_tile, dram=mem)
    if do_cfg:
        xt_pipe_in = concat_duplicate_batch(xt_row)
        try:
            ttnn.deallocate(xt_row)
        except Exception:
            pass
    else:
        xt_pipe_in = xt_row

    acoustic = pipe.forward_with_temb_tp(
        xt_bt64=xt_pipe_in,
        context_latents_bt128=ctx_buf,
        encoder_hidden_states_btd=enc_buf,
        temb_bd=temb,
        timestep_proj_b6d=tp,
        attention_mask_1d_bt=None,
        encoder_attention_mask_1d_bk=None,
        encoder_attention_mask_b1qk=mask_buf,
    )
    try:
        ttnn.deallocate(xt_pipe_in)
    except Exception:
        pass
    if acoustic is not acoustic_out:
        ttnn.copy(acoustic, acoustic_out)
        try:
            ttnn.deallocate(acoustic)
        except Exception:
            pass
        acoustic = acoustic_out

    if do_cfg and apply_cfg:
        vpc_rm = slice_batch_btc(acoustic, 0, 1, frames_i, c_lat)
        vpu_rm = slice_batch_btc(acoustic, 1, 2, frames_i, c_lat)
        vt_tt = apg_guidance_velocity_ttnn(
            vpc_rm,
            vpu_rm,
            float(guidance_scale),
            momentum_buffer=None,
            dims=[1],
            dram=mem,
        )
    elif do_cfg:
        vpc_rm = slice_batch_btc(acoustic, 0, 1, frames_i, c_lat)
        vt_tt = typecast_bf16_any_to_fp32_tile(vpc_rm, dram=mem)
    else:
        vt_tt = typecast_bf16_any_to_fp32_tile(acoustic, dram=mem)

    xt_new = euler_subtract_v_dt_from_tile(xt=xt_tile, vt=vt_tt, dt_tile=dt_scalar_tile, dram=mem)
    try:
        ttnn.deallocate(vt_tt)
    except Exception:
        pass
    if xt_new is not xt_tile:
        try:
            ttnn.deallocate(xt_tile)
        except Exception:
            pass
    return xt_new


def make_scalar_fp32_tile(device: Any, dram: Any, *, value: float = 0.0) -> ttnn.Tensor:
    """One-element TILE FP32 buffer for per-step scalar staging (``dt``, ``sigma``, etc.)."""
    if hasattr(ttnn, "full"):
        return ttnn.full(
            (1, 1, 1),
            float(value),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram,
        )
    import numpy as np

    rm = ttnn.as_tensor(
        np.asarray([[[float(value)]]], dtype=np.float32),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram,
    )
    return ttnn.to_layout(rm, layout=ttnn.TILE_LAYOUT)


def copy_scalar_into_tile(src_host_float: float, dst_tile: ttnn.Tensor, *, dram: Any) -> None:
    """Refresh a 1-element TILE tensor from a host float (CQ1-safe small upload)."""
    import numpy as np

    host_rm = ttnn.from_torch(
        np.asarray([[[float(src_host_float)]]], dtype=np.float32),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    if hasattr(ttnn, "copy_host_to_device_tensor"):
        ttnn.copy_host_to_device_tensor(host_rm, dst_tile, cq_id=1)
    else:
        ttnn.copy(host_rm, dst_tile)


def denoise_full_step_device(
    *,
    pipe: Any,
    xt_tile: ttnn.Tensor,
    temb: ttnn.Tensor,
    tp: ttnn.Tensor,
    enc_buf: ttnn.Tensor,
    ctx_buf: ttnn.Tensor,
    mask_buf: Optional[ttnn.Tensor],
    acoustic_out: ttnn.Tensor,
    dt_scalar_tile: ttnn.Tensor,
    sigma_scalar_tile: ttnn.Tensor,
    mem: Any,
    frames_i: int,
    c_lat: int,
    do_cfg: bool,
    use_adg: bool,
    guidance_scale: float,
    apply_cfg: bool,
    device: Any,
) -> ttnn.Tensor:
    """Run one denoise Euler step into persistent buffers (trace-safe, no Python scalars in graph).

    Reads/writes ``xt_tile`` in place for the Euler update. Writes DiT velocity output into
    ``acoustic_out`` (persistent). Caller must not deallocate ``acoustic_out`` between replays.
    """
    xt_row = fp32_tile_to_row_bf16(xt_tile, dram=mem)
    if do_cfg:
        xt_pipe_in = concat_duplicate_batch(xt_row)
        try:
            ttnn.deallocate(xt_row)
        except Exception:
            pass
    else:
        xt_pipe_in = xt_row

    acoustic = pipe.forward_with_temb_tp(
        xt_bt64=xt_pipe_in,
        context_latents_bt128=ctx_buf,
        encoder_hidden_states_btd=enc_buf,
        temb_bd=temb,
        timestep_proj_b6d=tp,
        attention_mask_1d_bt=None,
        encoder_attention_mask_1d_bk=None,
        encoder_attention_mask_b1qk=mask_buf,
    )
    try:
        ttnn.deallocate(xt_pipe_in)
    except Exception:
        pass

    # Overwrite persistent acoustic buffer via copy if capture bound a different tensor.
    if acoustic is not acoustic_out:
        ttnn.copy(acoustic, acoustic_out)
        try:
            ttnn.deallocate(acoustic)
        except Exception:
            pass
        acoustic = acoustic_out

    if do_cfg and apply_cfg:
        vpc_rm = slice_batch_btc(acoustic, 0, 1, frames_i, c_lat)
        vpu_rm = slice_batch_btc(acoustic, 1, 2, frames_i, c_lat)
        if use_adg:
            sigma_f = float(ttnn.to_torch(sigma_scalar_tile, dtype=torch.float32).reshape(-1)[0].item())
            vt_tt = adg_guidance_velocity_ttnn(
                xt_tile,
                vpc_rm,
                vpu_rm,
                sigma_f,
                float(guidance_scale),
                device=device,
                dram=mem,
            )
        else:
            vt_tt = apg_guidance_velocity_ttnn(
                vpc_rm,
                vpu_rm,
                float(guidance_scale),
                momentum_buffer=None,
                dims=[1],
                dram=mem,
            )
    elif do_cfg:
        vpc_rm = slice_batch_btc(acoustic, 0, 1, frames_i, c_lat)
        vt_tt = typecast_bf16_any_to_fp32_tile(vpc_rm, dram=mem)
    else:
        vt_tt = typecast_bf16_any_to_fp32_tile(acoustic, dram=mem)

    dt_f = float(ttnn.to_torch(dt_scalar_tile).reshape(-1)[0].item())
    xt_new = euler_subtract_v_dt(xt=xt_tile, vt=vt_tt, dt=dt_f, dram=mem)
    try:
        ttnn.deallocate(vt_tt)
    except Exception:
        pass
    if xt_new is not xt_tile:
        try:
            ttnn.deallocate(xt_tile)
        except Exception:
            pass
    return xt_new


__all__ = [
    "copy_scalar_into_tile",
    "denoise_full_step_device",
    "denoise_full_step_trace_graph",
    "make_scalar_fp32_tile",
]
