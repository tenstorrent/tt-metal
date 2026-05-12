# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN Euler, CFG tiling, APG / ADG guidance for ACE-Step flow sampling.

Latent ``x_t`` stays on device as FLOAT32 TILE; DiT consumes ROW_MAJOR BF16 ``x`` built per step.

ADG matches ``apply_norm=False`` (demo default); ``apply_norm=True`` raises.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import ttnn


class TtnnMomentumBufferApg:
    __slots__ = ("momentum", "running_tt")

    def __init__(self, *, momentum: float = -0.75) -> None:
        self.momentum = float(momentum)
        self.running_tt: ttnn.Tensor | None = None

    def reset(self) -> None:
        if self.running_tt is None:
            return
        try:
            ttnn.deallocate(self.running_tt)
        except Exception:
            pass
        self.running_tt = None


def concat_duplicate_batch(t: ttnn.Tensor) -> ttnn.Tensor:
    if hasattr(ttnn, "concat"):
        return ttnn.concat([t, t], dim=0)
    return ttnn.concatenate([t, t], dim=0)


def to_tile_fp32(t: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    tt = ttnn.to_layout(t, layout=ttnn.TILE_LAYOUT)
    return ttnn.typecast(tt, ttnn.float32, memory_config=dram)


def tile_fp32_from_numpy_bc(arr: np.ndarray, *, device: Any, dram: Any) -> ttnn.Tensor:
    tt = ttnn.as_tensor(
        np.asarray(arr, dtype=np.float32),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram,
    )
    return ttnn.to_layout(tt, layout=ttnn.TILE_LAYOUT)


def bf16_row_from_numpy_bc(arr_f32_np: np.ndarray, *, device: Any, dram: Any) -> ttnn.Tensor:
    return ttnn.as_tensor(
        np.asarray(arr_f32_np, dtype=np.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram,
    )


def typecast_bf16_any_to_fp32_tile(tt_bf16: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    tt = ttnn.to_layout(tt_bf16, layout=ttnn.TILE_LAYOUT)
    out = ttnn.typecast(tt, ttnn.float32, memory_config=dram)
    ttnn.deallocate(tt)
    return out


def fp32_tile_to_row_bf16(x_f32_tile: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    x_bf_tile = ttnn.typecast(x_f32_tile, ttnn.bfloat16, memory_config=dram)
    out = ttnn.to_layout(x_bf_tile, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(x_bf_tile)
    return out


def slice_batch_btc(vol: ttnn.Tensor, b0: int, b1_exc: int, t_: int, c: int) -> ttnn.Tensor:
    # ttnn.slice(input, slice_start, slice_end) — not starts=/ends= keyword args.
    return ttnn.slice(vol, (int(b0), 0, 0), (int(b1_exc), int(t_), int(c)))


def euler_subtract_v_dt(*, xt: ttnn.Tensor, vt: ttnn.Tensor, dt: float, dram: Any) -> ttnn.Tensor:
    step = ttnn.multiply(vt, float(dt), memory_config=dram)
    xt_new = ttnn.subtract(xt, step, memory_config=dram)
    ttnn.deallocate(step)
    return xt_new


def _sum_keepdim(x: ttnn.Tensor, dim: int, *, dram: Any) -> ttnn.Tensor:
    return ttnn.sum(x, dim=int(dim), keepdim=True)


def momentum_step_apg(buf: TtnnMomentumBufferApg, diff: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    if buf.running_tt is None:
        buf.running_tt = ttnn.clone(diff)
        ttnn.deallocate(diff)
        return buf.running_tt
    scaled = ttnn.multiply(buf.running_tt, float(buf.momentum), memory_config=dram)
    merged = ttnn.add(diff, scaled, memory_config=dram)
    ttnn.deallocate(scaled)
    ttnn.deallocate(buf.running_tt)
    buf.running_tt = merged
    return merged


def _normalize_l2_dim(x: ttnn.Tensor, dim: int, eps: float, *, dram: Any) -> ttnn.Tensor:
    sq = ttnn.multiply(x, x, memory_config=dram)
    rss = _sum_keepdim(sq, dim, dram=dram)
    ttnn.deallocate(sq)
    rss_eps = ttnn.add(rss, float(eps**2), memory_config=dram)
    ttnn.deallocate(rss)
    inv = ttnn.rsqrt(rss_eps, memory_config=dram)
    ttnn.deallocate(rss_eps)
    out = ttnn.multiply(x, inv, memory_config=dram)
    ttnn.deallocate(inv)
    return out


def _dot_keepdim_along_dim(x: ttnn.Tensor, y: ttnn.Tensor, dim: int, *, dram: Any) -> ttnn.Tensor:
    xy = ttnn.multiply(x, y, memory_config=dram)
    s = _sum_keepdim(xy, dim, dram=dram)
    ttnn.deallocate(xy)
    return s


def _project_parallel_orthogonal(
    v0: ttnn.Tensor,
    v1: ttnn.Tensor,
    dim: int,
    *,
    dram: Any,
    eps: float = 1e-12,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    v1n = _normalize_l2_dim(v1, dim, eps=eps, dram=dram)
    dotted = _dot_keepdim_along_dim(v0, v1n, dim, dram=dram)
    vpar = ttnn.multiply(dotted, v1n, memory_config=dram)
    ttnn.deallocate(v1n)
    ttnn.deallocate(dotted)
    vorth = ttnn.subtract(v0, vpar, memory_config=dram)
    return vpar, vorth


def apg_guidance_velocity_ttnn(
    pred_cond_bf16_rm: ttnn.Tensor,
    pred_uncond_bf16_rm: ttnn.Tensor,
    guidance_scale: float,
    *,
    momentum_buffer: TtnnMomentumBufferApg | None,
    dims: list[int],
    dram: Any,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
) -> ttnn.Tensor:
    dim = int(dims[0])
    p_c = to_tile_fp32(pred_cond_bf16_rm, dram=dram)
    p_u = to_tile_fp32(pred_uncond_bf16_rm, dram=dram)
    ttnn.deallocate(pred_cond_bf16_rm)
    ttnn.deallocate(pred_uncond_bf16_rm)

    diff = ttnn.subtract(p_c, p_u, memory_config=dram)
    ttnn.deallocate(p_u)
    if momentum_buffer is not None:
        diff = momentum_step_apg(momentum_buffer, diff, dram=dram)

    if norm_threshold > 0:
        rss = _sum_keepdim(ttnn.multiply(diff, diff, memory_config=dram), dim, dram=dram)
        diff_norm = ttnn.sqrt(ttnn.add(rss, 1e-20, memory_config=dram), memory_config=dram)
        ttnn.deallocate(rss)
        ratio = ttnn.div(float(norm_threshold), ttnn.add(diff_norm, 1e-20, memory_config=dram))
        factor = ttnn.minimum(ratio, ttnn.ones_like(diff))
        scaled = ttnn.multiply(diff, factor, memory_config=dram)
        ttnn.deallocate(diff)
        ttnn.deallocate(ratio)
        ttnn.deallocate(diff_norm)
        ttnn.deallocate(factor)
        diff = scaled

    par, orth = _project_parallel_orthogonal(diff, p_c, dim, dram=dram)
    if float(eta) != 0.0:
        eta_par = ttnn.multiply(par, float(eta), memory_config=dram)
        normed = ttnn.add(orth, eta_par, memory_config=dram)
        ttnn.deallocate(eta_par)
    else:
        normed = orth
    ttnn.deallocate(par)

    upd = ttnn.multiply(normed, float(guidance_scale - 1.0), memory_config=dram)
    guided = ttnn.add(p_c, upd, memory_config=dram)
    ttnn.deallocate(upd)
    ttnn.deallocate(normed)
    ttnn.deallocate(p_c)
    return guided


def _const_tile_bc(shape: tuple[int, int, int], value: float, *, device: Any, dram: Any) -> ttnn.Tensor:
    b_, t_, c_ = shape
    arr = np.full((b_, t_, c_), float(value), dtype=np.float32)
    rm = ttnn.as_tensor(
        arr,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram,
    )
    return ttnn.to_layout(rm, layout=ttnn.TILE_LAYOUT)


def adg_guidance_velocity_ttnn(
    xt_f32_tt: ttnn.Tensor,
    vpc_bf16_rm: ttnn.Tensor,
    vpu_bf16_rm: ttnn.Tensor,
    sigma_scalar: float,
    guidance_scale: float,
    *,
    device: Any,
    dram: Any,
    angle_clip: float = float(np.pi / 6.0),
    apply_clip: bool = True,
    apply_norm: bool = False,
) -> ttnn.Tensor:
    if apply_norm:
        raise NotImplementedError("TTNN ADG apply_norm=True is unsupported in ACE-Step demos.")
    vpc = typecast_bf16_any_to_fp32_tile(vpc_bf16_rm, dram=dram)
    vpu = typecast_bf16_any_to_fp32_tile(vpu_bf16_rm, dram=dram)
    ttnn.deallocate(vpc_bf16_rm)
    ttnn.deallocate(vpu_bf16_rm)

    b_, t__, c__ = int(xt_f32_tt.shape[0]), int(xt_f32_tt.shape[1]), int(xt_f32_tt.shape[2])
    sigma_tt = _const_tile_bc((b_, t__, c__), float(sigma_scalar), device=device, dram=dram)

    lhs_c = ttnn.multiply(vpc, sigma_tt, memory_config=dram)
    lhs_u = ttnn.multiply(vpu, sigma_tt, memory_config=dram)
    lh_t = ttnn.subtract(xt_f32_tt, lhs_c, memory_config=dram)
    lh_u = ttnn.subtract(xt_f32_tt, lhs_u, memory_config=dram)
    ttnn.deallocate(lhs_c)
    ttnn.deallocate(lhs_u)
    ttnn.deallocate(vpc)
    ttnn.deallocate(vpu)
    ttnn.deallocate(sigma_tt)

    lh_t_flat = ttnn.reshape(lh_t, (b_ * t__, c__))
    lh_u_flat = ttnn.reshape(lh_u, (b_ * t__, c__))
    nt = _normalize_l2_dim(lh_t_flat, dim=1, eps=1e-6, dram=dram)
    nu = _normalize_l2_dim(lh_u_flat, dim=1, eps=1e-6, dram=dram)
    dotted_flat = _dot_keepdim_along_dim(nt, nu, dim=1, dram=dram)
    ttnn.deallocate(nt)
    ttnn.deallocate(nu)
    cos_theta = ttnn.clip(dotted_flat, -1.0 + 1e-6, 1.0 - 1e-6)
    ttnn.deallocate(dotted_flat)
    theta_flat = ttnn.acos(cos_theta)
    ttnn.deallocate(cos_theta)
    latent_theta = ttnn.reshape(theta_flat, (b_, t__, 1))
    ttnn.deallocate(theta_flat)

    w = float(guidance_scale) - 1.0
    w_eff = float(max(w, 0.0)) + (1e-3 if w <= 0 else 0.0)
    latent_theta_scaled = (
        ttnn.clip(ttnn.multiply(latent_theta, float(w_eff), memory_config=dram), -float(angle_clip), float(angle_clip))
        if apply_clip
        else ttnn.multiply(latent_theta, float(w_eff), memory_config=dram)
    )

    ld = ttnn.subtract(lh_t, lh_u, memory_config=dram)
    latent_diff_flat = ttnn.reshape(ld, (b_ * t__, c__))
    lh_u_proj = lh_u_flat
    dotted_pv = _dot_keepdim_along_dim(latent_diff_flat, lh_u_proj, dim=1, dram=dram)
    sq_u = _sum_keepdim(ttnn.multiply(lh_u_proj, lh_u_proj, memory_config=dram), dim=1, dram=dram)
    denom = ttnn.add(sq_u, 1e-8, memory_config=dram)
    scale_pv = ttnn.div(dotted_pv, denom)
    ttnn.deallocate(denom)
    ttnn.deallocate(sq_u)
    proj_f = ttnn.multiply(scale_pv, lh_u_proj, memory_config=dram)
    ttnn.deallocate(scale_pv)
    ttnn.deallocate(dotted_pv)
    perp_flat = ttnn.subtract(latent_diff_flat, proj_f, memory_config=dram)
    ttnn.deallocate(proj_f)
    ttnn.deallocate(latent_diff_flat)
    ttnn.deallocate(ld)

    latent_v_new = ttnn.multiply(lh_t, ttnn.cos(latent_theta_scaled), memory_config=dram)

    sin_theta = ttnn.sin(latent_theta)
    sin_nn = ttnn.sin(latent_theta_scaled)

    denom_s = ttnn.add(sin_theta, 1e-20, memory_config=dram)
    ratio = ttnn.div(sin_nn, denom_s)

    perp_btc = ttnn.reshape(perp_flat, (b_, t__, c__))
    # ratio / sin_theta are [B, T, 1] (see apg_guidance.adg_forward); broadcast to C like PyTorch.
    ratio_btc = ttnn.repeat_interleave(ratio, int(c__), dim=2)
    prim = ttnn.multiply(perp_btc, ratio_btc, memory_config=dram)

    sin_btc = ttnn.repeat_interleave(sin_theta, int(c__), dim=2)
    ms = ttnn.le(sin_btc, float(1e-3))
    alt = ttnn.multiply(perp_btc, float(w_eff), memory_config=dram)
    latent_p = ttnn.where(ms, alt, prim, memory_config=dram)
    latent_new = ttnn.add(latent_v_new, latent_p, memory_config=dram)

    guided_delta = ttnn.subtract(xt_f32_tt, latent_new, memory_config=dram)
    guided_v = ttnn.multiply(guided_delta, float(1.0 / max(float(sigma_scalar), 1e-12)), memory_config=dram)
    ttnn.deallocate(guided_delta)
    return guided_v
