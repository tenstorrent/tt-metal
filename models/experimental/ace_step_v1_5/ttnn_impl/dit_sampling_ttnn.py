# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN Euler, CFG tiling, APG / ADG guidance for ACE-Step flow sampling.

Latent ``x_t`` stays on device as FLOAT32 TILE; DiT consumes TILE BF16 L1 ``[B,T,C]`` activations per step.

ADG matches ``apply_norm=False`` (demo default); ``apply_norm=True`` raises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import ttnn

from .math_perf_env import (
    ace_step_ensure_dit_activation,
    ace_step_from_torch_activation,
    ace_step_linear_l1_memory_config,
    ace_step_reshape_kwargs,
    ace_step_to_layout_kwargs,
)


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
    from models.experimental.ace_step_v1_5.tt_device import ace_step_device_num_chips, ace_step_synchronize_device

    tt = ttnn.as_tensor(
        np.asarray(arr, dtype=np.float32),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram,
    )
    tt = ttnn.to_layout(tt, layout=ttnn.TILE_LAYOUT)
    if ace_step_device_num_chips(device) > 1:
        ace_step_synchronize_device(ttnn, device)
    return tt


def _host_gaussian_latents_f32(shape: tuple[int, ...], *, seed: int) -> np.ndarray:
    """Standard-normal latents on CPU (parity with ``ttnn.randn`` / ``torch.randn``)."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    return rng.standard_normal(size=tuple(int(x) for x in shape)).astype(np.float32)


def dit_init_latents_host_f32(
    *,
    batch: int,
    frames: int,
    channels: int,
    seed: int,
) -> torch.Tensor:
    """Host FLOAT32 latents ``[B, T, C]`` for multi-device eager sampling."""
    shape = (int(batch), int(frames), int(channels))
    return torch.from_numpy(_host_gaussian_latents_f32(shape, seed=int(seed)))


def dit_init_latents_fp32_tile(
    *,
    batch: int,
    frames: int,
    channels: int,
    device: Any,
    dram: Any,
    seed: int,
) -> ttnn.Tensor:
    """Initialize DiT latent noise as FLOAT32 TILE on device.

    On multi-device meshes use :func:`dit_init_latents_host_f32` with the host latent sampler
    (``ace_step_mesh_use_host_latent_sampler``); device-side latent init is not supported there.
    """
    from models.experimental.ace_step_v1_5.tt_device import ace_step_device_num_chips

    shape = (int(batch), int(frames), int(channels))
    if ace_step_device_num_chips(device) > 1:
        host = _host_gaussian_latents_f32(shape, seed=int(seed))
        return tile_fp32_from_numpy_bc(host, device=device, dram=dram)
    if not hasattr(ttnn, "randn"):
        raise RuntimeError("This path needs ``ttnn.randn`` (Gaussian) for latent init; ``ttnn.rand`` is uniform-only.")
    return ttnn.randn(
        shape,
        device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
        seed=int(np.uint32(int(seed))),
    )


def _load_temb_weights_np_f32(
    *, checkpoint_safetensors_path: str, decoder_prefix: str = "decoder."
) -> dict[str, np.ndarray]:
    """Load only ``time_embed`` / ``time_embed_r`` MLP weights (not the full DiT checkpoint)."""
    import torch
    from safetensors import safe_open

    prefix = str(decoder_prefix)
    bases = ("time_embed.", "time_embed_r.")
    suffixes = (
        "linear_1.weight",
        "linear_1.bias",
        "linear_2.weight",
        "linear_2.bias",
        "time_proj.weight",
        "time_proj.bias",
    )
    out: dict[str, np.ndarray] = {}
    with safe_open(str(checkpoint_safetensors_path), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            if prefix and not str(key).startswith(prefix):
                continue
            rel = str(key)[len(prefix) :] if prefix else str(key)
            if not rel.startswith(bases):
                continue
            if not any(rel.endswith(s) for s in suffixes):
                continue
            out[rel] = sf.get_tensor(key).detach().to(torch.float32).cpu().numpy()
    if not out:
        raise KeyError(f"No time_embed / time_embed_r weights under prefix {prefix!r} in {checkpoint_safetensors_path}")
    return out


def _numpy_silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32, copy=False)


def _numpy_temb_mlp(
    *,
    sd: dict[str, np.ndarray],
    base: str,
    t_freq_4d: np.ndarray,
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the timestep MLP on CPU (matches ``TorchTimestepEmbeddingRef``)."""
    w1 = sd[f"{base}.linear_1.weight"]
    b1 = sd[f"{base}.linear_1.bias"].reshape(1, 1, 1, hidden_size)
    w2 = sd[f"{base}.linear_2.weight"]
    b2 = sd[f"{base}.linear_2.bias"].reshape(1, 1, 1, hidden_size)
    wt = sd[f"{base}.time_proj.weight"]
    bt = sd[f"{base}.time_proj.bias"].reshape(1, 1, 1, 6 * hidden_size)

    temb = np.matmul(t_freq_4d.astype(np.float32), w1.T.astype(np.float32)) + b1
    temb = _numpy_silu(temb)
    temb = np.matmul(temb, w2.T.astype(np.float32)) + b2
    h = _numpy_silu(temb)
    tp = np.matmul(h, wt.T.astype(np.float32)) + bt
    tp_out = tp.reshape(1, 6, hidden_size).astype(np.float32, copy=False)
    temb_out = temb.reshape(1, hidden_size).astype(np.float32, copy=False)
    return temb_out, tp_out


def _numpy_sinusoidal_t_freq(*, timestep_value: float, in_channels: int = 256, scale: float = 1000.0) -> np.ndarray:
    t = np.float32(float(timestep_value) * float(scale))
    half = int(in_channels) // 2
    freqs = np.exp((-np.log(10000.0)) * (np.arange(half, dtype=np.float32) / float(half)))
    args = t * freqs
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    return emb.reshape(1, 1, 1, int(in_channels)).astype(np.float32, copy=False)


def stage_host_temb_tp_row(
    temb_host: np.ndarray | torch.Tensor,
    tp_host: np.ndarray | torch.Tensor,
    *,
    device: Any,
    dram: Any,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Upload host ``(temb, tp)`` as TILE BF16 in L1 (matches ``compute_temb_tp`` / decoder AdaLN)."""

    def _to_f32_np(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=np.float32)
        return np.asarray(x.detach().to(dtype=torch.float32).cpu().numpy(), dtype=np.float32)

    l1_mc = ace_step_linear_l1_memory_config(ttnn) or dram
    temb_tt = ace_step_from_torch_activation(
        ttnn, _to_f32_np(temb_host), device=device, dtype=ttnn.bfloat16, l1_mc=l1_mc
    )
    tp_tt = ace_step_from_torch_activation(ttnn, _to_f32_np(tp_host), device=device, dtype=ttnn.bfloat16, l1_mc=l1_mc)
    return temb_tt, tp_tt


def stage_host_temb_steps_to_device(
    temb_host_steps: list[Any],
    tp_host_steps: list[Any],
    *,
    device: Any,
    dram: Any,
) -> tuple[list[ttnn.Tensor], list[ttnn.Tensor]]:
    """Upload host-precomputed ``(temb, tp)`` lists for trace / device-side denoise loops."""
    temb_dev: list[ttnn.Tensor] = []
    tp_dev: list[ttnn.Tensor] = []
    for temb_h, tp_h in zip(temb_host_steps, tp_host_steps):
        temb_tt, tp_tt = stage_host_temb_tp_row(temb_h, tp_h, device=device, dram=dram)
        temb_dev.append(temb_tt)
        tp_dev.append(tp_tt)
    return temb_dev, tp_dev


def precompute_dit_temb_steps_host(
    *,
    checkpoint_safetensors_path: str,
    timesteps_host: np.ndarray,
    num_steps: int,
    decoder_prefix: str = "decoder.",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pure NumPy temb precompute; only loads the 12 small time-embed weight tensors."""
    print("[ace_step_v1_5] loading time_embed weights from checkpoint …", flush=True)
    sd = _load_temb_weights_np_f32(
        checkpoint_safetensors_path=str(checkpoint_safetensors_path),
        decoder_prefix=str(decoder_prefix),
    )
    print(f"[ace_step_v1_5] time_embed weights ready ({len(sd)} tensors)", flush=True)
    hidden_size = int(sd["time_embed.linear_1.weight"].shape[0])
    ts_host = np.asarray(timesteps_host, dtype=np.float32)

    print("[ace_step_v1_5] computing time_embed_r (delta=0) on host …", flush=True)
    t_freq_r = _numpy_sinusoidal_t_freq(timestep_value=0.0)
    temb_r, tp_r = _numpy_temb_mlp(sd=sd, base="time_embed_r", t_freq_4d=t_freq_r, hidden_size=hidden_size)

    temb_per_step: list[np.ndarray] = []
    tp_per_step: list[np.ndarray] = []
    for idx in range(int(num_steps)):
        half = 128
        freqs = np.exp((-np.log(10000.0)) * (np.arange(half, dtype=np.float32) / float(half)))
        t_val = np.float32(float(ts_host[int(idx)]) * 1000.0)
        args = t_val * freqs
        emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        t_freq = emb.reshape(1, 1, 1, 256).astype(np.float32, copy=False)
        temb_t, tp_t = _numpy_temb_mlp(sd=sd, base="time_embed", t_freq_4d=t_freq, hidden_size=hidden_size)
        temb_per_step.append((temb_t + temb_r).astype(np.float32, copy=False))
        tp_per_step.append((tp_t + tp_r).astype(np.float32, copy=False))
        print(f"[ace_step_v1_5] host temb step {idx + 1}/{int(num_steps)} done", flush=True)
    return temb_per_step, tp_per_step


def precompute_dit_temb_steps(
    pipe: Any,
    *,
    num_steps: int,
    target_batch: int,
    device: Any,
    checkpoint_safetensors_path: str | None = None,
    timesteps_host: np.ndarray | None = None,
) -> tuple[list[Any], list[Any], bool]:
    """Precompute per-Euler-step ``(temb, timestep_proj)``; returns ``(temb, tp, on_host)``."""
    from models.experimental.ace_step_v1_5.tt_device import (
        ace_step_device_num_chips,
        ace_step_mesh_use_host_temb_precompute,
        ace_step_synchronize_device,
    )

    if ace_step_mesh_use_host_temb_precompute(device):
        if checkpoint_safetensors_path is None or timesteps_host is None:
            raise ValueError("Host temb precompute requires checkpoint_safetensors_path and timesteps_host.")
        print("[ace_step_v1_5] precomputing timestep embeddings on host CPU …", flush=True)
        temb_per_step, tp_per_step = precompute_dit_temb_steps_host(
            checkpoint_safetensors_path=str(checkpoint_safetensors_path),
            timesteps_host=np.asarray(timesteps_host, dtype=np.float32),
            num_steps=int(num_steps),
        )
        print("[ace_step_v1_5] host timestep embeddings ready (upload per denoise step)", flush=True)
        return temb_per_step, tp_per_step, True

    if ace_step_device_num_chips(device) > 1:
        ace_step_synchronize_device(ttnn, device)
    temb_per_step_dev: list[ttnn.Tensor] = []
    tp_per_step_dev: list[ttnn.Tensor] = []
    for idx in range(int(num_steps)):
        temb, tp = pipe.compute_temb_tp(int(idx), target_batch=int(target_batch))
        temb_per_step_dev.append(temb)
        tp_per_step_dev.append(tp)
        if ace_step_device_num_chips(device) > 1:
            ace_step_synchronize_device(ttnn, device)
    return temb_per_step_dev, tp_per_step_dev, False


def bf16_tile_l1_from_numpy_bc(arr_f32_np: np.ndarray, *, device: Any, dram: Any) -> ttnn.Tensor:
    """Upload ``[B,T,C]`` (or ``[B,S,D]``) host array as TILE BF16 in L1 for DiT activations."""
    from models.experimental.ace_step_v1_5.tt_device import ace_step_device_num_chips, ace_step_synchronize_device

    l1_mc = ace_step_linear_l1_memory_config(ttnn) or dram
    tt = ace_step_from_torch_activation(
        ttnn,
        np.asarray(arr_f32_np, dtype=np.float32),
        device=device,
        dtype=ttnn.bfloat16,
        l1_mc=l1_mc,
    )
    if ace_step_device_num_chips(device) > 1:
        ace_step_synchronize_device(ttnn, device)
    return tt


def bf16_row_from_numpy_bc(arr_f32_np: np.ndarray, *, device: Any, dram: Any) -> ttnn.Tensor:
    """Deprecated alias: DiT path uses :func:`bf16_tile_l1_from_numpy_bc` (TILE+L1, not ROW_MAJOR DRAM)."""
    return bf16_tile_l1_from_numpy_bc(arr_f32_np, device=device, dram=dram)


def typecast_bf16_any_to_fp32_tile(tt_bf16: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    # Place the intermediate tile in L1 so the TilizeWithValPadding kernel reads/writes
    # L1 rather than DRAM (eliminates TypecastDeviceOperation + TilizeWithValPadding
    # appearing as ``in0:dram_interleaved`` in Tracy perf reports).
    tt = ttnn.to_layout(tt_bf16, layout=ttnn.TILE_LAYOUT, **ace_step_to_layout_kwargs(ttnn))
    out = ttnn.typecast(tt, ttnn.float32, memory_config=dram)
    ttnn.deallocate(tt)
    return out


def fp32_tile_to_bf16_tile_l1(x_f32_tile: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    """Cast Euler latents FP32 TILE → BF16 TILE in L1 (no ROW_MAJOR / ``Tilize`` before DiT)."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn) or dram
    out = ttnn.typecast(x_f32_tile, ttnn.bfloat16, memory_config=l1_mc)
    return ace_step_ensure_dit_activation(ttnn, out, l1_mc)


def fp32_tile_to_row_bf16(x_f32_tile: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    """Legacy ROW_MAJOR path (VAE / host-only); DiT denoise uses :func:`fp32_tile_to_bf16_tile_l1`."""
    x_bf_tile = ttnn.typecast(x_f32_tile, ttnn.bfloat16, memory_config=dram)
    out = ttnn.to_layout(x_bf_tile, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(x_bf_tile)
    return out


def prepare_latents_for_ttnn_vae(latents_tt: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    """Convert DiT denoise output to ROW_MAJOR FP32 ``[B,T,C]`` for ``decode_tiled``."""
    from models.experimental.ace_step_v1_5.tt_device import ace_step_device_num_chips, ace_step_synchronize_device

    tt = latents_tt
    if tt.layout != ttnn.ROW_MAJOR_LAYOUT:
        tt = ttnn.to_layout(tt, layout=ttnn.ROW_MAJOR_LAYOUT)
    if tt.dtype != ttnn.float32:
        tt = ttnn.typecast(tt, ttnn.float32, memory_config=dram)
    try:
        dev = tt.device()
    except Exception:
        dev = None
    if dev is not None and ace_step_device_num_chips(dev) > 1:
        ace_step_synchronize_device(ttnn, dev)
    return tt


def refresh_fp32_tile_from_host(
    xt_host: "torch.Tensor",
    *,
    device: Any,
    dram: Any,
    buf: ttnn.Tensor | None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Upload host ``[B,T,C]`` latents to a persistent FP32 TILE buffer (stable mesh denoise)."""

    host_np = xt_host.detach().float().cpu().contiguous().numpy()
    fresh = tile_fp32_from_numpy_bc(host_np, device=device, dram=dram)
    if buf is None:
        return fresh, fresh
    try:
        ttnn.copy(fresh, buf)
        try:
            ttnn.deallocate(fresh)
        except Exception:
            pass
        return buf, buf
    except Exception:
        try:
            ttnn.deallocate(buf)
        except Exception:
            pass
        return fresh, fresh


def slice_batch_btc(vol: ttnn.Tensor, b0: int, b1_exc: int, t_: int, c: int) -> ttnn.Tensor:
    # ttnn.slice(input, slice_start, slice_end) — not starts=/ends= keyword args.
    return ttnn.slice(vol, (int(b0), 0, 0), (int(b1_exc), int(t_), int(c)))


def euler_subtract_v_dt(*, xt: ttnn.Tensor, vt: ttnn.Tensor, dt: float, dram: Any) -> ttnn.Tensor:
    step = ttnn.multiply(vt, float(dt), memory_config=dram)
    xt_new = ttnn.subtract(xt, step, memory_config=dram)
    ttnn.deallocate(step)
    return xt_new


def euler_subtract_v_dt_host(*, xt: torch.Tensor, vt: torch.Tensor, dt: float) -> torch.Tensor:
    return xt - vt.to(dtype=xt.dtype) * float(dt)


def apg_guidance_velocity_host(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    *,
    momentum_buffer: Any | None,
    dims: list[int] | None = None,
) -> torch.Tensor:
    """APG on host ``[B, T, C]`` tensors (permutes to ``[B, C, T]`` for the torch ref)."""
    from models.experimental.ace_step_v1_5.torch_ref._vendored_acestep.acestep.models.common.apg_guidance import (
        apg_forward,
    )

    dim = int((dims or [1])[0])
    if dim == 1:
        p_c = pred_cond.permute(0, 2, 1)
        p_u = pred_uncond.permute(0, 2, 1)
        guided = apg_forward(p_c, p_u, float(guidance_scale), momentum_buffer=momentum_buffer, dims=[-1])
        return guided.permute(0, 2, 1)
    guided = apg_forward(
        pred_cond, pred_uncond, float(guidance_scale), momentum_buffer=momentum_buffer, dims=dims or [-1]
    )
    return guided


def adg_guidance_velocity_host(
    xt: torch.Tensor,
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    sigma_scalar: float,
    guidance_scale: float,
) -> torch.Tensor:
    from models.experimental.ace_step_v1_5.torch_ref._vendored_acestep.acestep.models.common.apg_guidance import (
        adg_forward,
    )

    return adg_forward(
        xt,
        pred_cond,
        pred_uncond,
        float(sigma_scalar),
        float(guidance_scale),
        apply_norm=False,
    )


def euler_subtract_v_dt_from_tile(*, xt: ttnn.Tensor, vt: ttnn.Tensor, dt_tile: ttnn.Tensor, dram: Any) -> ttnn.Tensor:
    """Euler update with ``dt`` read from a persistent 1-element TILE tensor (trace replay safe)."""
    step = ttnn.multiply(vt, dt_tile, memory_config=dram)
    xt_new = ttnn.subtract(xt, step, memory_config=dram)
    ttnn.deallocate(step)
    return xt_new


def _sum_keepdim(x: ttnn.Tensor, dim: int, *, dram: Any) -> ttnn.Tensor:
    return ttnn.sum(x, dim=int(dim), keepdim=True)


def momentum_step_apg(buf: TtnnMomentumBufferApg, diff: ttnn.Tensor, *, dram: Any) -> ttnn.Tensor:
    """Update APG momentum state and return a **new** tensor for downstream ops.

    ``running_tt`` must outlive the returned ``diff`` (norm clamp / projection may deallocate it).
    """
    if buf.running_tt is None:
        buf.running_tt = ttnn.clone(diff)
        out = ttnn.clone(buf.running_tt)
        ttnn.deallocate(diff)
        return out
    scaled = ttnn.multiply(buf.running_tt, float(buf.momentum), memory_config=dram)
    merged = ttnn.add(diff, scaled, memory_config=dram)
    ttnn.deallocate(scaled)
    ttnn.deallocate(diff)
    ttnn.deallocate(buf.running_tt)
    buf.running_tt = ttnn.clone(merged)
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
        diff_denom = ttnn.add(diff_norm, 1e-20, memory_config=dram)
        ttnn.deallocate(diff_norm)
        inv_denom = ttnn.div(ttnn.ones_like(diff_denom), diff_denom, memory_config=dram)
        ratio = ttnn.multiply(inv_denom, float(norm_threshold), memory_config=dram)
        ttnn.deallocate(diff_denom)
        ttnn.deallocate(inv_denom)
        factor = ttnn.minimum(ratio, ttnn.ones_like(diff))
        scaled = ttnn.multiply(diff, factor, memory_config=dram)
        ttnn.deallocate(diff)
        ttnn.deallocate(ratio)
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
    """Fill a TILE FP32 tensor on device (no host ``numpy`` + H2D per ADG step)."""
    if hasattr(ttnn, "full"):
        return ttnn.full(
            shape,
            float(value),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram,
        )
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

    _sr = ace_step_reshape_kwargs(ttnn)

    b_, t__, c__ = int(xt_f32_tt.shape[0]), int(xt_f32_tt.shape[1]), int(xt_f32_tt.shape[2])

    # Scalar-broadcast multiply: ``ttnn.multiply(tensor, python_float, ...)`` is a pure device op
    # (same kernel as tensor*tensor multiply, just with the scalar baked into the kernel args).
    # Previously this path built a [B,T,C] NumPy tensor filled with ``sigma_scalar`` and uploaded
    # it via ``ttnn.as_tensor`` every Euler step — a host->device transfer in the hot loop that
    # the perf probe (``ACE_STEP_PROBE_FALLBACKS=1``) caught as 27 unexpected uploads / 3-generate
    # run. Switching to scalar form eliminates the per-step DMA and the host NumPy allocation, and
    # is also a precondition for wrapping the denoise loop in a TTNN trace.
    lhs_c = ttnn.multiply(vpc, float(sigma_scalar), memory_config=dram)
    lhs_u = ttnn.multiply(vpu, float(sigma_scalar), memory_config=dram)
    lh_t = ttnn.subtract(xt_f32_tt, lhs_c, memory_config=dram)
    lh_u = ttnn.subtract(xt_f32_tt, lhs_u, memory_config=dram)
    ttnn.deallocate(lhs_c)
    ttnn.deallocate(lhs_u)
    ttnn.deallocate(vpc)
    ttnn.deallocate(vpu)

    lh_t_flat = ttnn.reshape(lh_t, (b_ * t__, c__), **_sr)
    lh_u_flat = ttnn.reshape(lh_u, (b_ * t__, c__), **_sr)
    nt = _normalize_l2_dim(lh_t_flat, dim=1, eps=1e-6, dram=dram)
    nu = _normalize_l2_dim(lh_u_flat, dim=1, eps=1e-6, dram=dram)
    dotted_flat = _dot_keepdim_along_dim(nt, nu, dim=1, dram=dram)
    ttnn.deallocate(nt)
    ttnn.deallocate(nu)
    cos_theta = ttnn.clip(dotted_flat, -1.0 + 1e-6, 1.0 - 1e-6)
    ttnn.deallocate(dotted_flat)
    theta_flat = ttnn.acos(cos_theta)
    ttnn.deallocate(cos_theta)
    latent_theta = ttnn.reshape(theta_flat, (b_, t__, 1), **_sr)
    ttnn.deallocate(theta_flat)

    w = float(guidance_scale) - 1.0
    w_eff = float(max(w, 0.0)) + (1e-3 if w <= 0 else 0.0)
    latent_theta_scaled = (
        ttnn.clip(ttnn.multiply(latent_theta, float(w_eff), memory_config=dram), -float(angle_clip), float(angle_clip))
        if apply_clip
        else ttnn.multiply(latent_theta, float(w_eff), memory_config=dram)
    )

    ld = ttnn.subtract(lh_t, lh_u, memory_config=dram)
    latent_diff_flat = ttnn.reshape(ld, (b_ * t__, c__), **_sr)
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

    perp_btc = ttnn.reshape(perp_flat, (b_, t__, c__), **_sr)
    # ratio / sin_theta are [B, T, 1]; multiply/compare broadcast on dim 2 like PyTorch.
    prim = ttnn.multiply(perp_btc, ratio, memory_config=dram)

    ms = ttnn.le(sin_theta, float(1e-3))
    alt = ttnn.multiply(perp_btc, float(w_eff), memory_config=dram)
    latent_p = ttnn.where(ms, alt, prim, memory_config=dram)
    latent_new = ttnn.add(latent_v_new, latent_p, memory_config=dram)

    guided_delta = ttnn.subtract(xt_f32_tt, latent_new, memory_config=dram)
    guided_v = ttnn.multiply(guided_delta, float(1.0 / max(float(sigma_scalar), 1e-12)), memory_config=dram)
    ttnn.deallocate(guided_delta)
    return guided_v
