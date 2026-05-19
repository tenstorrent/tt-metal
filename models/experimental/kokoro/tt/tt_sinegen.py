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


@dataclass(frozen=True)
class TTSineGenParams:
    """Device-resident weights / scalars for :class:`TTSineGen`."""

    # Time-axis linear maps for downsample and cumsum (matmul along last axis of ``[B, dim, *]``).
    interp_down: ttnn.Tensor  # [T, T_down]

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


def _linear_interp_matrix(input_size: int, output_size: int) -> np.ndarray:
    """``[output_size, input_size]`` that matches ``F.interpolate(mode='linear', align_corners=False)``."""
    M = np.zeros((output_size, input_size), dtype=np.float64)
    scale = input_size / output_size
    for j in range(output_size):
        x = (j + 0.5) * scale - 0.5
        i0 = int(np.floor(x))
        w1 = x - i0
        w0 = 1.0 - w1
        i0_c = max(0, min(input_size - 1, i0))
        i1_c = max(0, min(input_size - 1, i0 + 1))
        M[j, i0_c] += w0
        M[j, i1_c] += w1
    return M


def _upload_matrix(arr: np.ndarray, device, *, dtype: ttnn.DataType, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.from_numpy(arr.astype(np.float32)),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _upload_scalar(value: float, device, *, dtype: ttnn.DataType) -> ttnn.Tensor:
    """``[1, 1, 1]`` broadcastable scalar."""
    return ttnn.from_torch(
        torch.tensor([[[value]]], dtype=torch.float32),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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
) -> TTSineGenParams:
    """Bake the three time-axis maps + broadcast scalars onto device."""
    if upsample_scale < 1:
        raise ValueError(f"upsample_scale must be >= 1 (got {upsample_scale})")
    if time_len % upsample_scale != 0:
        raise ValueError(
            f"time_len must be a multiple of upsample_scale (got time_len={time_len}, upsample_scale={upsample_scale})"
        )

    dim = int(harmonic_num) + 1
    time_len_down = time_len // upsample_scale

    M_down = _linear_interp_matrix(time_len, time_len_down)  # [T_down, T]

    # Store as the transpose so we can ``y @ M`` directly (matmul along last axis of activation).
    interp_down_tt = _upload_matrix(M_down.T, device, dtype=weights_dtype)  # [T, T_down]

    # Elementwise lerp weights for upsample — avoids the K=T_down matmul whose BF16 accumulation
    # introduces ~5e-4 fractional-unit error (×2π×300 ≈ 1 radian in phase_up, destroying sines).
    # alpha[k] = (k + 0.5) / upsample_scale  matches ``F.interpolate(mode='linear', align_corners=False)``.
    clamp_len = upsample_scale // 2
    alpha_1d = (np.arange(upsample_scale, dtype=np.float32) + 0.5) / float(upsample_scale)
    alpha_2d = np.tile(alpha_1d[:, None], (1, dim))  # [upsample_scale, dim]
    lerp_alpha_np = alpha_2d[None, :, :]  # [1, upsample_scale, dim]
    lerp_one_minus_alpha_np = 1.0 - lerp_alpha_np
    lerp_clamp_ones_np = np.ones((1, clamp_len, dim), dtype=np.float32)

    def _upload_3d(arr: np.ndarray) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.from_numpy(arr.astype(np.float32)),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    lerp_alpha_tt = _upload_3d(lerp_alpha_np)
    lerp_one_minus_alpha_tt = _upload_3d(lerp_one_minus_alpha_np)
    lerp_clamp_ones_tt = _upload_3d(lerp_clamp_ones_np)

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
        interp_down=interp_down_tt,
        lerp_alpha=lerp_alpha_tt,
        lerp_one_minus_alpha=lerp_one_minus_alpha_tt,
        lerp_clamp_ones=lerp_clamp_ones_tt,
        clamp_len=clamp_len,
        harmonics=harmonics_tt,
        inv_sampling_rate=_upload_scalar(1.0 / float(sampling_rate), device, dtype=weights_dtype),
        one=_upload_scalar(1.0, device, dtype=weights_dtype),
        two_pi_times_scale=_upload_scalar(2.0 * math.pi * float(upsample_scale), device, dtype=weights_dtype),
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

    **Precision note** — ``use_torch_phase_fallback``
    The lerp upsample step computes small cumsum values (< 0.05 cycles) and then multiplies by
    ``2π × upsample_scale``.  On BH hardware all MAC ops round float32 inputs to BF16 before the
    multiply unit; the absolute BF16 error on a 0.04-cycle value is ~3.3×10⁻⁵ cycles, which
    after ×(2π × 300) = ×1885 becomes ~0.06–0.25 radians — comparable to ``sine_amp = 0.1``.
    At Kokoro's scale (``upsample_scale=300``) this destroys phase PCC.

    When ``use_torch_phase_fallback=True`` the entire phase chain — downsample → cumsum → lerp
    upsample → sin — runs in CPU float32.  Only ``uv``, noise scaling, and the
    ``sine_waves * uv + noise`` mix stay on-device.  This is the minimal fallback needed to
    restore PCC > 0.99 at Kokoro scale.  At small upsample scales (≤ 10) the BF16 error is
    negligible and the fallback is not needed.
    """

    def __init__(self, device: ttnn.Device, params: TTSineGenParams, *, use_torch_phase_fallback: bool = False) -> None:
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
        """CPU float32 phase chain: rad → downsample → cumsum → lerp upsample → sin.

        Only the phase/sine computation runs on CPU.  ``uv``, noise, and the output mix
        remain in the caller's TTNN path.  Returns ``sine_waves`` as a device tensor.
        """
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

        sine_amp_float = float(ttnn.to_torch(p.sine_amp).flatten()[0].item())
        sines = torch.sin(phase_up) * sine_amp_float
        return ttnn.from_torch(
            sines.contiguous(),
            dtype=p.activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _zero_btd(self, B: int) -> ttnn.Tensor:
        return ttnn.zeros(
            [B, self.params.time_len, self.params.dim],
            dtype=self.params.activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _zero_b1d(self, B: int) -> ttnn.Tensor:
        return ttnn.zeros(
            [B, 1, self.params.dim],
            dtype=self.params.activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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

        # ``fn = f0 * harmonics`` → [B, T, dim]
        fn = ttnn.multiply(f0_btd, p.harmonics, memory_config=memory_config)

        # ``rad = (fn / sampling_rate) % 1`` → [B, T, dim]
        rad = ttnn.multiply(fn, p.inv_sampling_rate, memory_config=memory_config)
        ttnn.deallocate(fn)
        rad = ttnn.remainder(rad, p.one, memory_config=memory_config)

        # ``rad[:, 0, :] += rand_ini`` with the fundamental's slot zeroed.
        if rand_ini is not None:
            rand_masked = ttnn.multiply(rand_ini, p.fundamental_zero_mask, memory_config=memory_config)
            # Pad the ``[B, 1, dim]`` row to ``[B, T, dim]`` by concat with zeros.
            if p.time_len > 1:
                tail = ttnn.zeros(
                    [B, p.time_len - 1, p.dim],
                    dtype=rad.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=memory_config,
                )
                rand_pad = ttnn.concat([rand_masked, tail], dim=1, memory_config=memory_config)
                ttnn.deallocate(tail)
            else:
                rand_pad = rand_masked
            rad = ttnn.add(rad, rand_pad, memory_config=memory_config)
            ttnn.deallocate(rand_pad)
            if rand_pad is not rand_masked:
                ttnn.deallocate(rand_masked)

        if self.use_torch_phase_fallback:
            # CPU float32 phase chain: BF16 MAC error on small cumsum values (< 0.05 cycles)
            # is amplified by 2π × upsample_scale (= 1885 for Kokoro) → ~0.06–0.25 rad error
            # vs sine_amp=0.1, destroying PCC.  Only the phase chain falls back; uv/noise stay TT.
            ttnn.deallocate(rad)
            sine_waves_unmasked = self._torch_phase_fallback(f0_btd, rand_ini)
        else:
            # Permute to ``[B, dim, T]`` so the time-axis maps are last-axis matmuls.
            rad_bdt = ttnn.permute(rad, (0, 2, 1), memory_config=memory_config)
            ttnn.deallocate(rad)

            # Downsample along T: ``[B, dim, T] @ [T, T_down]`` → ``[B, dim, T_down]``
            rad_down = ttnn.matmul(
                rad_bdt,
                p.interp_down,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(rad_bdt)

            # Cumsum along T_down via sequential prefix adds — avoids K=T_down matmul whose BF16
            # per-tile multiply gives ~2e-3 relative error that amplifies to ~2 radians in phase_up.
            # Each r_t = rad_down[:, :, t:t+1] is [B, dim, 1]; prefix[t] = r_0 + ... + r_t.
            r_slices = [
                ttnn.slice(rad_down, [0, 0, t], [B, p.dim, t + 1], [1, 1, 1], memory_config=memory_config)
                for t in range(p.time_len_down)
            ]
            ttnn.deallocate(rad_down)
            prefix = [r_slices[0]]
            for t in range(1, p.time_len_down):
                prefix.append(ttnn.add(prefix[-1], r_slices[t], memory_config=memory_config))
            for t in range(1, p.time_len_down):
                ttnn.deallocate(r_slices[t])
            phase = ttnn.concat(prefix, dim=2, memory_config=memory_config)
            for t in range(p.time_len_down):
                ttnn.deallocate(prefix[t])

            # Upsample via elementwise linear interpolation (avoids K=T_down matmul whose BF16
            # accumulation causes ~1 radian phase_up error, destroying sines PCC).
            # Layout: permute phase [B, dim, T_down] → [B, T_down, dim] for dim=1 slicing.
            phase_btd = ttnn.permute(phase, (0, 2, 1), memory_config=memory_config)
            ttnn.deallocate(phase)

            # Clamped start: repeat phase[t=0] for clamp_len output steps.
            p0 = ttnn.slice(phase_btd, [0, 0, 0], [B, 1, p.dim], [1, 1, 1], memory_config=memory_config)
            start_seg = ttnn.multiply(p0, p.lerp_clamp_ones, memory_config=memory_config)
            ttnn.deallocate(p0)

            # Interior lerp segments: one upsample_scale-length segment per adjacent pair.
            lerp_segs = [start_seg]
            for s in range(p.time_len_down - 1):
                p_s = ttnn.slice(phase_btd, [0, s, 0], [B, s + 1, p.dim], [1, 1, 1], memory_config=memory_config)
                p_s1 = ttnn.slice(phase_btd, [0, s + 1, 0], [B, s + 2, p.dim], [1, 1, 1], memory_config=memory_config)
                seg = ttnn.add(
                    ttnn.multiply(p_s, p.lerp_one_minus_alpha, memory_config=memory_config),
                    ttnn.multiply(p_s1, p.lerp_alpha, memory_config=memory_config),
                    memory_config=memory_config,
                )
                ttnn.deallocate(p_s)
                ttnn.deallocate(p_s1)
                lerp_segs.append(seg)

            # Clamped end: repeat phase[t=T_down-1] for clamp_len output steps.
            p_last = ttnn.slice(
                phase_btd,
                [0, p.time_len_down - 1, 0],
                [B, p.time_len_down, p.dim],
                [1, 1, 1],
                memory_config=memory_config,
            )
            end_seg = ttnn.multiply(p_last, p.lerp_clamp_ones, memory_config=memory_config)
            ttnn.deallocate(p_last)
            ttnn.deallocate(phase_btd)
            lerp_segs.append(end_seg)

            phase_up = ttnn.concat(lerp_segs, dim=1, memory_config=memory_config)
            for seg in lerp_segs:
                ttnn.deallocate(seg)

            phase_up = ttnn.multiply(phase_up, p.two_pi_times_scale, memory_config=memory_config)
            sines = ttnn.sin(phase_up, memory_config=memory_config)
            ttnn.deallocate(phase_up)

            sine_waves_unmasked = ttnn.multiply(sines, p.sine_amp, memory_config=memory_config)
            ttnn.deallocate(sines)

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
