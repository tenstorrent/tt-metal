# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.istftnet.SineGen`
(``flag_for_pulse=False`` path only — the Kokoro istftnet default).

Reference math (for input ``f0 ∈ [B, T, 1]`` with ``dim = harmonic_num + 1``):

    fn  = f0 * [1, 2, ..., dim]                                  # [B, T, dim]
    rad = (fn / sampling_rate) % 1
    rad[:, 0, :] += rand_ini                                      # rand_ini[:, 0] forced to 0
    rad_down  = F.interpolate(rad.transpose(1, 2),
                              scale_factor=1/upsample_scale,
                              mode="linear").transpose(1, 2)      # [B, T_down, dim]
    phase     = cumsum(rad_down, dim=1) * 2π
    phase_up  = F.interpolate(phase.transpose(1, 2) * upsample_scale,
                              scale_factor=upsample_scale,
                              mode="linear").transpose(1, 2)      # [B, T, dim]
    sines     = sin(phase_up)
    sine_w    = sines * sine_amp
    uv        = (f0 > voiced_threshold).float()                   # [B, T, 1]
    noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    noise     = noise_amp * randn(B, T, dim)
    out       = sine_w * uv + noise

Downsample and cumsum are baked as fixed matmuls. The upsample step uses elementwise linear
interpolation instead of a matmul to avoid BF16 precision loss from the small K=T_down dimension.

Randomness
----------
The reference samples ``rand_ini ~ U[0,1)`` (with the fundamental forced to 0) and
``noise ~ N(0, 1)`` inside ``forward``. In this port, noise is generated once in ``__init__``
via ``torch.randn_like`` on a dummy tensor of the correct shape and uploaded to device; the
same tensor is reused every ``forward`` call (tiled along the batch dimension when ``B > 1``).
Callers may still override by passing ``noise_raw`` explicitly. Tests patch ``torch.rand`` and
``torch.randn_like`` during module construction to keep noise deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import ttnn

from .tt_trace_prep import traced_zeros as _traced_zeros

# Above this full-resolution ``T_har`` the monolithic phase chain (permute entire ``rad`` to
# ``[B, dim, T]``, strided downsample, cumsum, lerp concat) exceeds BH L1.  Process
# ``rad_down`` in ``_SINEGEN_TD_CHUNK``-sized blocks instead; upsample/sin per block and
# assemble on the height axis (dim=1 in ``[B, T, dim]``).
_SINEGEN_PHASE_MONOLITH_MAX_T = 65536
# Downsample-index chunk size: each block touches ~(chunk-1)*upsample_scale + 2 full-rate
# samples (~9k rows at scale=300) and emits chunk*upsample_scale upsampled phase rows.
_SINEGEN_TD_CHUNK = 32


@dataclass(frozen=True)
class TTSineGenParams:
    """Device-resident weights / scalars for :class:`TTSineGen`."""

    # Elementwise lerp weights for the upsample step (replaces the interp_up matmul).
    # ``lerp_alpha[k] = (k + 0.5) / upsample_scale`` for k=0..upsample_scale-1.
    # Shape: ``[1, upsample_scale, dim]`` — broadcasts over B.
    lerp_alpha: ttnn.Tensor
    lerp_one_minus_alpha: ttnn.Tensor
    # ``[1, clamp_len, dim]`` all-ones tensor for broadcasting the edge-clamped regions.
    lerp_clamp_ones: ttnn.Tensor
    clamp_len: int  # = upsample_scale // 2

    # Per-harmonic multiplier ``[1, 1, dim]`` (``[1, 2, ..., dim]``).
    harmonics: ttnn.Tensor

    # Broadcastable ``[1, 1, 1]`` scalar tensors.
    inv_sampling_rate: ttnn.Tensor
    one: ttnn.Tensor
    two_pi_times_scale: ttnn.Tensor  # ``2π * upsample_scale`` baked together.
    voiced_threshold: ttnn.Tensor
    sine_amp: ttnn.Tensor
    noise_std: ttnn.Tensor
    sine_amp_over_three: ttnn.Tensor

    # Mask ``[1, 1, dim]`` that zeroes the fundamental's column of ``rand_ini`` (matches the
    # reference's ``rand_ini[:, 0] = 0``).
    fundamental_zero_mask: ttnn.Tensor

    time_len: int
    time_len_down: int
    dim: int
    upsample_scale: int
    sampling_rate: float
    activation_dtype: ttnn.DataType = ttnn.bfloat16  # dtype for intermediate zeros / typecasts


def _upload_scalar(value: float, device, *, dtype: ttnn.DataType) -> ttnn.Tensor:
    """``[1, 1, 1]`` broadcastable scalar."""
    return ttnn.from_torch(
        torch.tensor([[[value]]], dtype=torch.float32),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_fp32_if_needed(x: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    if x.dtype == ttnn.float32:
        return x, False
    return ttnn.typecast(x, ttnn.float32, memory_config=memory_config), True


def _add_rand_ini_row0(
    rad: ttnn.Tensor,
    rand_ini: ttnn.Tensor,
    *,
    B: int,
    dim: int,
    time_len: int,
    fundamental_zero_mask: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """``rad[:, 0, :] += rand_ini`` without materializing a ``[B, T-1, dim]`` zero tail."""
    rand_masked = ttnn.multiply(rand_ini, fundamental_zero_mask, memory_config=memory_config)
    row0 = ttnn.slice(rad, [0, 0, 0], [B, 1, dim], [1, 1, 1], memory_config=memory_config)
    row0 = ttnn.add(row0, rand_masked, memory_config=memory_config)
    ttnn.deallocate(rand_masked)
    if time_len == 1:
        return row0
    tail = ttnn.slice(rad, [0, 1, 0], [B, time_len, dim], [1, 1, 1], memory_config=memory_config)
    out = ttnn.concat([row0, tail], dim=1, memory_config=memory_config)
    ttnn.deallocate(row0)
    ttnn.deallocate(tail)
    return out


def _downsample_rad_bdt_chunk(
    rad_bdt: ttnn.Tensor,
    *,
    td_count: int,
    upsample_scale: int,
    B: int,
    dim: int,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """Strided-slice downsample on a short ``[B, dim, T_chunk]`` window (see class docstring)."""
    up = upsample_scale
    local_hi = 0 if up % 2 else 1
    hi_end = local_hi + (td_count - 1) * up + 1
    hi = ttnn.slice(rad_bdt, [0, 0, local_hi], [B, dim, hi_end], [1, 1, up], memory_config=memory_config)
    if up % 2 == 0:
        lo_end = (td_count - 1) * up + 1
        lo = ttnn.slice(rad_bdt, [0, 0, 0], [B, dim, lo_end], [1, 1, up], memory_config=memory_config)
        summed = ttnn.add(lo, hi, memory_config=memory_config)
        ttnn.deallocate(lo)
        ttnn.deallocate(hi)
        return ttnn.multiply(summed, 0.5, memory_config=memory_config)
    return hi


def _lerp_upsample_phase_btd(
    phase_btd: ttnn.Tensor,
    *,
    td0: int,
    td_count: int,
    time_len_down: int,
    B: int,
    dim: int,
    lerp_alpha: ttnn.Tensor,
    lerp_one_minus: ttnn.Tensor,
    lerp_clamp: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig,
    phase_prev: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    """Elementwise lerp upsample for one ``rad_down`` chunk; concat segments on height (dim=1)."""
    lerp_segs: list[ttnn.Tensor] = []
    if phase_prev is not None:
        p_first = ttnn.slice(phase_btd, [0, 0, 0], [B, 1, dim], [1, 1, 1], memory_config=memory_config)
        seg = ttnn.add(
            ttnn.multiply(phase_prev, lerp_one_minus, memory_config=memory_config),
            ttnn.multiply(p_first, lerp_alpha, memory_config=memory_config),
            memory_config=memory_config,
        )
        ttnn.deallocate(p_first)
        lerp_segs.append(seg)

    if td0 == 0:
        p0 = ttnn.slice(phase_btd, [0, 0, 0], [B, 1, dim], [1, 1, 1], memory_config=memory_config)
        lerp_segs.append(ttnn.multiply(p0, lerp_clamp, memory_config=memory_config))
        ttnn.deallocate(p0)

    for s in range(td_count - 1):
        p_s = ttnn.slice(phase_btd, [0, s, 0], [B, s + 1, dim], [1, 1, 1], memory_config=memory_config)
        p_s1 = ttnn.slice(phase_btd, [0, s + 1, 0], [B, s + 2, dim], [1, 1, 1], memory_config=memory_config)
        seg = ttnn.add(
            ttnn.multiply(p_s, lerp_one_minus, memory_config=memory_config),
            ttnn.multiply(p_s1, lerp_alpha, memory_config=memory_config),
            memory_config=memory_config,
        )
        ttnn.deallocate(p_s)
        ttnn.deallocate(p_s1)
        lerp_segs.append(seg)

    if td0 + td_count == time_len_down:
        p_last = ttnn.slice(phase_btd, [0, td_count - 1, 0], [B, td_count, dim], [1, 1, 1], memory_config=memory_config)
        lerp_segs.append(ttnn.multiply(p_last, lerp_clamp, memory_config=memory_config))
        ttnn.deallocate(p_last)

    phase_up = ttnn.concat(lerp_segs, dim=1, memory_config=memory_config)
    for seg in lerp_segs:
        ttnn.deallocate(seg)
    return phase_up


def preprocess_tt_sinegen(
    *,
    device: ttnn.Device,
    sampling_rate: float,
    upsample_scale: int,
    harmonic_num: int,
    sine_amp: float,
    noise_std: float,
    voiced_threshold: float,
    time_len: int,
    weights_dtype=ttnn.bfloat16,
    phase_weights_dtype=ttnn.float32,
) -> TTSineGenParams:
    """Bake the three time-axis maps + broadcast scalars onto device.

    ``phase_weights_dtype`` (default fp32) is used for the precision-critical operands of the
    on-device phase chain — the lerp upsample weights and the ``2π·upsample_scale`` scalar. These
    are amplified by ``2π·upsample_scale (~1885)`` and the ``sin`` nonlinearity, so bf16 storage
    (e.g. bf16 rounds ``2π·300`` to ~1888, a ~0.2% error → ~1 rad at phase~600) collapses the
    non-fallback sine PCC. They are consumed against fp32 activations, so fp32 storage adds no
    mixed-dtype conversions. Bumping ``MathFidelity`` does *not* help — HiFi passes only recover
    mantissa bits that are stored. For full accuracy use ``use_torch_phase_fallback``.

    The downsample is done by exact strided slicing in ``forward`` (not a matmul), so no
    ``interp_down`` weight is uploaded: for integer ``upsample_scale`` the linear-interp downsample
    is a fixed 0.5/0.5 average of two mid-block samples (even scale) or the middle sample (odd),
    which stays in true fp32 — the old matmul truncated MAC multiplicands to bf16 (~2e-6 on
    rad_down), and cumsum + ×2π×scale amplified that into ~0.25 full-model waveform PCC loss.
    """
    if upsample_scale < 1:
        raise ValueError(f"upsample_scale must be >= 1 (got {upsample_scale})")
    if time_len % upsample_scale != 0:
        raise ValueError(
            f"time_len must be a multiple of upsample_scale (got time_len={time_len}, upsample_scale={upsample_scale})"
        )

    dim = int(harmonic_num) + 1
    time_len_down = time_len // upsample_scale

    # Downsample uses exact strided slicing in forward (see class docstring); no interp_down matmul.

    # Elementwise lerp weights for upsample — avoids the K=T_down matmul whose BF16 accumulation
    # introduces ~5e-4 fractional-unit error (×2π×300 ≈ 1 radian in phase_up, destroying sines).
    # alpha[k] = (k + 0.5) / upsample_scale  matches ``F.interpolate(mode='linear', align_corners=False)``.
    clamp_len = upsample_scale // 2
    alpha_1d = (np.arange(upsample_scale, dtype=np.float32) + 0.5) / float(upsample_scale)
    alpha_2d = np.tile(alpha_1d[:, None], (1, dim))  # [upsample_scale, dim]
    lerp_alpha_np = alpha_2d[None, :, :]  # [1, upsample_scale, dim]
    lerp_one_minus_alpha_np = 1.0 - lerp_alpha_np
    lerp_clamp_ones_np = np.ones((1, clamp_len, dim), dtype=np.float32)

    def _upload_3d(arr: np.ndarray, dtype=weights_dtype) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.from_numpy(arr.astype(np.float32)),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Phase-chain lerp weights kept at phase_weights_dtype (fp32): consumed against fp32 phase slices.
    lerp_alpha_tt = _upload_3d(lerp_alpha_np, dtype=phase_weights_dtype)
    lerp_one_minus_alpha_tt = _upload_3d(lerp_one_minus_alpha_np, dtype=phase_weights_dtype)
    lerp_clamp_ones_tt = _upload_3d(lerp_clamp_ones_np, dtype=phase_weights_dtype)

    harmonics = torch.arange(1, dim + 1, dtype=torch.float32).reshape(1, 1, dim)
    harmonics_tt = ttnn.from_torch(
        harmonics,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Zero the fundamental column of ``rand_ini`` (broadcast ``[1, 1, dim]``).
    fundamental_mask = torch.ones(1, 1, dim, dtype=torch.float32)
    fundamental_mask[..., 0] = 0.0
    fund_mask_tt = ttnn.from_torch(
        fundamental_mask,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return TTSineGenParams(
        lerp_alpha=lerp_alpha_tt,
        lerp_one_minus_alpha=lerp_one_minus_alpha_tt,
        lerp_clamp_ones=lerp_clamp_ones_tt,
        clamp_len=clamp_len,
        harmonics=harmonics_tt,
        inv_sampling_rate=_upload_scalar(1.0 / float(sampling_rate), device, dtype=weights_dtype),
        one=_upload_scalar(1.0, device, dtype=weights_dtype),
        # fp32: bf16 rounding of ``2π·scale`` (~1885) alone is ~0.2% → ~1 rad error at phase~600.
        two_pi_times_scale=_upload_scalar(2.0 * math.pi * float(upsample_scale), device, dtype=phase_weights_dtype),
        voiced_threshold=_upload_scalar(float(voiced_threshold), device, dtype=weights_dtype),
        sine_amp=_upload_scalar(float(sine_amp), device, dtype=weights_dtype),
        noise_std=_upload_scalar(float(noise_std), device, dtype=weights_dtype),
        sine_amp_over_three=_upload_scalar(float(sine_amp) / 3.0, device, dtype=weights_dtype),
        fundamental_zero_mask=fund_mask_tt,
        time_len=int(time_len),
        time_len_down=int(time_len_down),
        dim=int(dim),
        upsample_scale=int(upsample_scale),
        sampling_rate=float(sampling_rate),
        activation_dtype=weights_dtype,
    )


class TTSineGen:
    """Kokoro :class:`SineGen` on TT (``flag_for_pulse=False``).

    Set ``use_torch_phase_fallback=True`` to run the phase accumulation chain on CPU float32 when
    BF16 MAC rounding at large ``upsample_scale`` would corrupt ``phase_up``.
    """

    def __init__(
        self,
        device: ttnn.Device,
        params: TTSineGenParams,
        *,
        use_torch_phase_fallback: bool = False,
    ) -> None:
        self.device = device
        self.params = params
        self.use_torch_phase_fallback = use_torch_phase_fallback
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        # Pre-generate noise_raw [1, time_len, dim] once; reused every forward call.
        _noise_dummy = torch.zeros(1, params.time_len, params.dim, dtype=torch.float32)
        self._noise_raw = ttnn.from_torch(
            torch.randn_like(_noise_dummy),
            dtype=params.activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _torch_phase_fallback(
        self,
        f0_btd: ttnn.Tensor,
        rand_ini: "Optional[ttnn.Tensor]",
    ) -> ttnn.Tensor:
        """CPU float32 phase accumulation through lerp upsample; returns fp32 ``phase_up``."""
        import torch.nn.functional as F_torch

        p = self.params
        B = int(f0_btd.shape[0])
        f0_cpu = ttnn.to_torch(f0_btd).float().reshape(B, p.time_len, 1)
        harmonics_cpu = ttnn.to_torch(p.harmonics).float().reshape(p.dim)

        fn = f0_cpu * harmonics_cpu  # [B, T, dim]
        rad = (fn / p.sampling_rate) % 1.0  # [B, T, dim]

        if rand_ini is not None:
            rand_ini_cpu = ttnn.to_torch(rand_ini).float().reshape(B, 1, p.dim)
            rand_ini_cpu[..., 0] = 0.0  # zero fundamental
            rad[:, 0:1, :] = rad[:, 0:1, :] + rand_ini_cpu

        rad_t = rad.transpose(1, 2)  # [B, dim, T]
        rad_down_t = F_torch.interpolate(
            rad_t, scale_factor=1.0 / p.upsample_scale, mode="linear", align_corners=False
        )  # [B, dim, T_down]
        rad_down = rad_down_t.transpose(1, 2)  # [B, T_down, dim]

        phase = torch.cumsum(rad_down, dim=1) * (2.0 * math.pi)  # [B, T_down, dim]

        phase_up_t = F_torch.interpolate(
            phase.transpose(1, 2) * p.upsample_scale,
            scale_factor=float(p.upsample_scale),
            mode="linear",
            align_corners=False,
        )  # [B, dim, T]
        phase_up = phase_up_t.transpose(1, 2)  # [B, T, dim]

        return ttnn.from_torch(
            phase_up.contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _forward_phase_chain_monolith(
        self,
        rad_btd: ttnn.Tensor,
        *,
        B: int,
        memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        """Full-sequence on-device phase chain (fits L1 when ``time_len`` is modest)."""
        p = self.params
        rad_fp32, owns_rad = _to_fp32_if_needed(rad_btd, memory_config)
        if owns_rad:
            ttnn.deallocate(rad_btd)
        rad_bdt = ttnn.permute(rad_fp32, (0, 2, 1), memory_config=memory_config)
        ttnn.deallocate(rad_fp32)

        up = p.upsample_scale
        mid = up // 2
        Td = p.time_len_down
        hi = ttnn.slice(
            rad_bdt, [0, 0, mid], [B, p.dim, mid + (Td - 1) * up + 1], [1, 1, up], memory_config=memory_config
        )
        if up % 2 == 0:
            lo = ttnn.slice(
                rad_bdt,
                [0, 0, mid - 1],
                [B, p.dim, (mid - 1) + (Td - 1) * up + 1],
                [1, 1, up],
                memory_config=memory_config,
            )
            summed = ttnn.add(lo, hi, memory_config=memory_config)
            ttnn.deallocate(lo)
            ttnn.deallocate(hi)
            rad_down = ttnn.multiply(summed, 0.5, memory_config=memory_config)
            ttnn.deallocate(summed)
        else:
            rad_down = hi
        ttnn.deallocate(rad_bdt)

        lerp_alpha, owns_la = _to_fp32_if_needed(p.lerp_alpha, memory_config)
        lerp_one_minus, owns_lom = _to_fp32_if_needed(p.lerp_one_minus_alpha, memory_config)
        lerp_clamp, owns_lc = _to_fp32_if_needed(p.lerp_clamp_ones, memory_config)
        two_pi_scale, owns_tps = _to_fp32_if_needed(p.two_pi_times_scale, memory_config)

        rad_down_btd = ttnn.permute(rad_down, (0, 2, 1), memory_config=memory_config)
        ttnn.deallocate(rad_down)
        phase_btd = ttnn.cumsum(rad_down_btd, dim=1)
        ttnn.deallocate(rad_down_btd)

        phase_up = _lerp_upsample_phase_btd(
            phase_btd,
            td0=0,
            td_count=Td,
            time_len_down=Td,
            B=B,
            dim=p.dim,
            lerp_alpha=lerp_alpha,
            lerp_one_minus=lerp_one_minus,
            lerp_clamp=lerp_clamp,
            memory_config=memory_config,
        )
        ttnn.deallocate(phase_btd)
        phase_up = ttnn.multiply(phase_up, two_pi_scale, memory_config=memory_config)

        if owns_la:
            ttnn.deallocate(lerp_alpha)
        if owns_lom:
            ttnn.deallocate(lerp_one_minus)
        if owns_lc:
            ttnn.deallocate(lerp_clamp)
        if owns_tps:
            ttnn.deallocate(two_pi_scale)
        return phase_up

    def _forward_phase_chain_chunked(
        self,
        rad_btd: ttnn.Tensor,
        *,
        B: int,
        memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        """L1-fitting blocks over ``T_har``: downsample → cumsum → lerp/sin, concat on height."""
        p = self.params
        up = p.upsample_scale
        mid = up // 2
        Td = p.time_len_down
        T = p.time_len

        lerp_alpha, owns_la = _to_fp32_if_needed(p.lerp_alpha, memory_config)
        lerp_one_minus, owns_lom = _to_fp32_if_needed(p.lerp_one_minus_alpha, memory_config)
        lerp_clamp, owns_lc = _to_fp32_if_needed(p.lerp_clamp_ones, memory_config)
        two_pi_scale, owns_tps = _to_fp32_if_needed(p.two_pi_times_scale, memory_config)

        # Consumed zero carry state — via traced_zeros (clone of a cached template) so the initial
        # ``ttnn.zeros(device=...)`` host write happens once (under trace prep) not every forward.
        phase_carry = _traced_zeros(
            [B, 1, p.dim],
            dtype=ttnn.float32,
            device=self.device,
            memory_config=memory_config,
            key=(id(self), "sinegen_phase_carry", B, int(p.dim), str(memory_config)),
        )
        sine_chunks: list[ttnn.Tensor] = []

        for td0 in range(0, Td, _SINEGEN_TD_CHUNK):
            td_count = min(_SINEGEN_TD_CHUNK, Td - td0)
            t_rad_start = td0 * up + (mid - 1 if up % 2 == 0 else mid)
            t_rad_len = (td_count - 1) * up + (2 if up % 2 == 0 else 1)
            t_rad_end = min(T, t_rad_start + t_rad_len)

            rad_chunk = ttnn.slice(
                rad_btd, [0, t_rad_start, 0], [B, t_rad_end, p.dim], [1, 1, 1], memory_config=memory_config
            )
            rad_chunk_fp32, owns_chunk = _to_fp32_if_needed(rad_chunk, memory_config)
            if owns_chunk:
                ttnn.deallocate(rad_chunk)
            rad_bdt = ttnn.permute(rad_chunk_fp32, (0, 2, 1), memory_config=memory_config)
            ttnn.deallocate(rad_chunk_fp32)

            rad_down = _downsample_rad_bdt_chunk(
                rad_bdt,
                td_count=td_count,
                upsample_scale=up,
                B=B,
                dim=p.dim,
                memory_config=memory_config,
            )
            ttnn.deallocate(rad_bdt)

            rad_down_btd = ttnn.permute(rad_down, (0, 2, 1), memory_config=memory_config)
            ttnn.deallocate(rad_down)
            phase_chunk = ttnn.cumsum(rad_down_btd, dim=1)
            ttnn.deallocate(rad_down_btd)
            if td0 > 0:
                phase_chunk = ttnn.add(phase_chunk, phase_carry, memory_config=memory_config)

            phase_prev = phase_carry if td0 > 0 else None
            phase_up_chunk = _lerp_upsample_phase_btd(
                phase_chunk,
                td0=td0,
                td_count=td_count,
                time_len_down=Td,
                B=B,
                dim=p.dim,
                lerp_alpha=lerp_alpha,
                lerp_one_minus=lerp_one_minus,
                lerp_clamp=lerp_clamp,
                memory_config=memory_config,
                phase_prev=phase_prev,
            )
            new_carry = ttnn.slice(
                phase_chunk, [0, td_count - 1, 0], [B, td_count, p.dim], [1, 1, 1], memory_config=memory_config
            )
            ttnn.deallocate(phase_chunk)
            ttnn.deallocate(phase_carry)
            phase_carry = new_carry
            phase_up_chunk = ttnn.multiply(phase_up_chunk, two_pi_scale, memory_config=memory_config)
            sine_chunks.append(ttnn.sin(phase_up_chunk, memory_config=memory_config))
            ttnn.deallocate(phase_up_chunk)

        ttnn.deallocate(phase_carry)
        sines = ttnn.concat(sine_chunks, dim=1, memory_config=memory_config)
        for ch in sine_chunks:
            ttnn.deallocate(ch)

        if owns_la:
            ttnn.deallocate(lerp_alpha)
        if owns_lom:
            ttnn.deallocate(lerp_one_minus)
        if owns_lc:
            ttnn.deallocate(lerp_clamp)
        if owns_tps:
            ttnn.deallocate(two_pi_scale)
        return sines

    def forward(
        self,
        f0_btd: ttnn.Tensor,
        *,
        rand_ini: Optional[ttnn.Tensor] = None,
        noise_raw: Optional[ttnn.Tensor] = None,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            f0_btd: ``[B, T, 1]`` (``T == params.time_len``).
            rand_ini: optional ``[B, 1, dim]`` initial phase noise (zeros if ``None``); the
                fundamental's column is forced to 0 internally to match the reference.
            noise_raw: optional ``[B, T, dim]`` noise tensor before amplitude scaling
                (``randn_like(sine_waves)`` in the reference). Uses the pre-generated
                ``self._noise_raw`` (tiled to batch size) if ``None``.

        Returns:
            ``(sine_waves, uv, noise)`` matching the reference contract.
        """
        p = self.params
        B = int(f0_btd.shape[0])

        if int(f0_btd.shape[1]) != p.time_len:
            raise ValueError(f"f0_btd time length {int(f0_btd.shape[1])} != params.time_len {p.time_len}")

        # ``uv = (f0 > voiced_threshold).float()`` → [B, T, 1]
        uv_bool = ttnn.gt(f0_btd, p.voiced_threshold, memory_config=memory_config)
        uv = ttnn.typecast(uv_bool, p.activation_dtype, memory_config=memory_config)
        ttnn.deallocate(uv_bool)

        if self.use_torch_phase_fallback:
            phase_up = self._torch_phase_fallback(f0_btd, rand_ini)
            pre_sines = None
        else:
            # ``fn = f0 * harmonics`` → [B, T, dim]
            fn = ttnn.multiply(f0_btd, p.harmonics, memory_config=memory_config)

            # ``rad = (fn / sampling_rate) % 1`` → [B, T, dim]
            rad = ttnn.multiply(fn, p.inv_sampling_rate, memory_config=memory_config)
            ttnn.deallocate(fn)
            # ``remainder`` rejects mixed dtypes; match ``one`` to ``rad`` (fp32 when f0 is uploaded fp32).
            one = p.one if p.one.dtype == rad.dtype else ttnn.typecast(p.one, rad.dtype, memory_config=memory_config)
            rad = ttnn.remainder(rad, one, memory_config=memory_config)
            if one is not p.one:
                ttnn.deallocate(one)

            # ``rad[:, 0, :] += rand_ini`` with the fundamental's slot zeroed.
            if rand_ini is not None:
                rad = _add_rand_ini_row0(
                    rad,
                    rand_ini,
                    B=B,
                    dim=p.dim,
                    time_len=p.time_len,
                    fundamental_zero_mask=p.fundamental_zero_mask,
                    memory_config=memory_config,
                )

            if p.time_len > _SINEGEN_PHASE_MONOLITH_MAX_T:
                pre_sines = self._forward_phase_chain_chunked(rad, B=B, memory_config=memory_config)
                ttnn.deallocate(rad)
                phase_up = None
            else:
                phase_up = self._forward_phase_chain_monolith(rad, B=B, memory_config=memory_config)
                pre_sines = None

        sine_amp_fp32, owns_sa = _to_fp32_if_needed(p.sine_amp, memory_config)
        if pre_sines is None:
            sines = ttnn.sin(phase_up, memory_config=memory_config)
            ttnn.deallocate(phase_up)
        else:
            sines = pre_sines
        sine_waves_unmasked = ttnn.multiply(sines, sine_amp_fp32, memory_config=memory_config)
        ttnn.deallocate(sines)
        if owns_sa:
            ttnn.deallocate(sine_amp_fp32)
        if sine_waves_unmasked.dtype != p.activation_dtype:
            sine_bf16 = ttnn.typecast(sine_waves_unmasked, p.activation_dtype, memory_config=memory_config)
            ttnn.deallocate(sine_waves_unmasked)
            sine_waves_unmasked = sine_bf16

        # ``noise_amp = uv * noise_std + (1 - uv) * sine_amp/3`` → [B, T, 1]
        one_minus_uv = ttnn.subtract(p.one, uv, memory_config=memory_config)
        noise_amp = ttnn.add(
            ttnn.multiply(uv, p.noise_std, memory_config=memory_config),
            ttnn.multiply(one_minus_uv, p.sine_amp_over_three, memory_config=memory_config),
            memory_config=memory_config,
        )
        ttnn.deallocate(one_minus_uv)

        # ``noise = noise_amp * randn_like(sine_waves)`` → [B, T, dim] (broadcast on last dim)
        if noise_raw is None:
            if B == 1:
                noise_raw_local = self._noise_raw
                owns_noise = False
            else:
                noise_raw_local = ttnn.concat([self._noise_raw] * B, dim=0, memory_config=memory_config)
                owns_noise = True
        else:
            noise_raw_local = noise_raw
            owns_noise = False
        noise = ttnn.multiply(noise_amp, noise_raw_local, memory_config=memory_config)
        ttnn.deallocate(noise_amp)
        if owns_noise:
            ttnn.deallocate(noise_raw_local)

        # ``out = sine_waves * uv + noise`` (``uv`` is ``[B, T, 1]``; broadcasts over ``dim``).
        sine_waves_masked = ttnn.multiply(sine_waves_unmasked, uv, memory_config=memory_config)
        ttnn.deallocate(sine_waves_unmasked)
        out = ttnn.add(sine_waves_masked, noise, memory_config=memory_config)
        ttnn.deallocate(sine_waves_masked)

        return out, uv, noise

    __call__ = forward
