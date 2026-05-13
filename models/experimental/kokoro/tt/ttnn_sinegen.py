# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``SineGen`` (Kokoro ``flag_for_pulse=False`` path).

Both 1D linear resamples are 2-tap so we use ``ttnn.gather`` + two-point lerp (sparse weights from
``preprocess_source_module_hn_nsf_parameters``) for the downsample *and* upsample paths, avoiding wide
bf16-quantised matmuls that drift on long sequences / large phases. Only ``mat_cumsum`` (triu-of-ones)
remains as a fp32 matmul — its weights are exactly 0 or 1, so bf16 quantisation is lossless. Phase scaling
is two roundings (``× 2π`` then ``× upsample_scale``) to mirror PyTorch's evaluation order; fusing the two
into ``× (2π·upsample_scale)`` was empirically slightly worse on WH B0.
"""

from __future__ import annotations

from typing import Any, Tuple

import math
import torch
import ttnn

from models.experimental.kokoro.reference.kokoro_source_module_preprocess import _interp_up_lerp_from_g_up


def sinegen_fp32_matmul_cfg(device):
    """Interpolation / merge matmuls: HiFi3 + fp32 dest + L1 pack sync on Wormhole B0."""
    fidelity = ttnn.MathFidelity.HiFi3 if ttnn.device.is_wormhole_b0(device) else ttnn.MathFidelity.HiFi4
    if ttnn.device.is_wormhole_b0(device):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class KokoroTtnnSineGen:
    """
    Device ``SineGen`` matching ``kokoro_istftnet.SineGen`` when ``flag_for_pulse`` is false.

    Uses sparse ``interp_down_*`` and ``interp_up_*`` lerp tables plus ``mat_cumsum`` from
    ``preprocess_source_module_hn_nsf_parameters``. ``mat_down`` / ``mat_up`` are kept for legacy callers
    only and are used to lazily rebuild the sparse tables if they're missing.
    """

    def __init__(self, device, parameters: dict[str, Any]):
        self.device = device
        if bool(parameters.get("flag_for_pulse", False)):
            raise NotImplementedError("TTNN SineGen supports flag_for_pulse=False only (Kokoro default).")

        self.time_len = int(parameters["time_len"])
        self.dim = int(parameters["harmonic_num"]) + 1
        self.sine_amp = float(parameters["sine_amp"])
        self.noise_std = float(parameters["noise_std"])
        self.voiced_threshold = float(parameters["voiced_threshold"])
        self.upsample_scale = float(parameters["upsample_scale"])

        self._mat_down = parameters["mat_down"]
        self._mat_up = parameters["mat_up"]
        self._mat_cumsum = parameters.get("mat_cumsum")
        if self._mat_cumsum is None:
            td = int(
                parameters.get(
                    "t_down", int(round(float(parameters["time_len"]) / float(parameters["upsample_scale"])))
                )
            )
            if td <= 0:
                raise ValueError(
                    "Cannot infer t_down for mat_cumsum; preprocess_source_module_hn_nsf_parameters is stale"
                )
            cm = torch.triu(torch.ones(td, td, dtype=torch.float32))
            self._mat_cumsum = ttnn.from_torch(
                cm.contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self._harmonics = parameters["harmonics"]
        self._harmonic_rand_mask = parameters["harmonic_rand_mask"]
        dram = ttnn.DRAM_MEMORY_CONFIG
        self._two_pi = parameters.get("two_pi")
        if self._two_pi is None:
            self._two_pi = ttnn.from_torch(
                torch.tensor([[[2.0 * math.pi]]], dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=dram,
            )
        self._upsample_scale_tt = ttnn.from_torch(
            torch.tensor([[[float(parameters["upsample_scale"])]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )
        self._inv_sr = parameters["inv_sampling_rate"]
        self._one = parameters["one"]
        vt = float(parameters["voiced_threshold"])
        self._voiced_threshold = ttnn.from_torch(
            torch.tensor([[[vt]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._sine_amp_tt = ttnn.from_torch(
            torch.tensor([[[self.sine_amp]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._noise_std_tt = ttnn.from_torch(
            torch.tensor([[[self.noise_std]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._sine_amp_over_3 = ttnn.from_torch(
            torch.tensor([[[self.sine_amp / 3.0]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._compute_cfg = sinegen_fp32_matmul_cfg(device)

        self._interp_up_idx0, self._interp_up_idx1, self._interp_up_w0, self._interp_up_w1 = self._load_interp_tables(
            device, parameters, "interp_up_", self._mat_up, transpose_g=True
        )
        (
            self._interp_down_idx0,
            self._interp_down_idx1,
            self._interp_down_w0,
            self._interp_down_w1,
        ) = self._load_interp_tables(device, parameters, "interp_down_", self._mat_down, transpose_g=True)

    @staticmethod
    def _load_interp_tables(device, parameters, prefix: str, mat_fallback, *, transpose_g: bool):
        """Return ``(idx0, idx1, w0, w1)`` device tensors, building from ``mat_fallback`` if missing.

        Stored matrices in ``parameters`` are ``g.T`` (shape ``(t_in, t_out)``); ``_interp_up_lerp_from_g_up``
        expects ``(t_out, t_in)``, hence ``transpose_g=True`` for them.
        """
        i0 = parameters.get(f"{prefix}idx0")
        if i0 is not None:
            return (i0, parameters[f"{prefix}idx1"], parameters[f"{prefix}w0"], parameters[f"{prefix}w1"])
        g = ttnn.to_torch(mat_fallback).to(dtype=torch.float32)
        if transpose_g:
            g = g.T.contiguous()
        idx0, idx1, w0, w1 = _interp_up_lerp_from_g_up(g)
        dram = ttnn.DRAM_MEMORY_CONFIG
        return (
            ttnn.from_torch(
                idx0.view(1, 1, -1).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=dram,
            ),
            ttnn.from_torch(
                idx1.view(1, 1, -1).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=dram,
            ),
            ttnn.from_torch(
                w0.view(1, 1, -1).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=dram,
            ),
            ttnn.from_torch(
                w1.view(1, 1, -1).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=dram,
            ),
        )

    def _gather_lerp(
        self, src: ttnn.Tensor, idx0: ttnn.Tensor, idx1: ttnn.Tensor, w0: ttnn.Tensor, w1: ttnn.Tensor
    ) -> ttnn.Tensor:
        """``out[..., j] = w0[j]·src[..., idx0[j]] + w1[j]·src[..., idx1[j]]`` — exact 2-tap fp32 lerp.

        ``src`` is ``(B, dim, t_in)`` and idx/w are ``(1, 1, t_out)``; we expand to ``(B, dim, t_out)`` before
        gather so all harmonics share the table.
        """
        dram = ttnn.DRAM_MEMORY_CONFIG
        bsz, dim = int(src.shape[0]), int(src.shape[1])
        rep = [bsz, dim, 1]
        idx0_e = ttnn.repeat(idx0, rep, memory_config=dram)
        idx1_e = ttnn.repeat(idx1, rep, memory_config=dram)
        w0_e = ttnn.repeat(w0, rep, memory_config=dram)
        w1_e = ttnn.repeat(w1, rep, memory_config=dram)
        v0 = ttnn.gather(src, 2, index=idx0_e, memory_config=dram)
        v1 = ttnn.gather(src, 2, index=idx1_e, memory_config=dram)
        ttnn.deallocate(idx0_e)
        ttnn.deallocate(idx1_e)
        out = ttnn.add(
            ttnn.multiply(v0, w0_e, memory_config=dram),
            ttnn.multiply(v1, w1_e, memory_config=dram),
            memory_config=dram,
        )
        ttnn.deallocate(v0)
        ttnn.deallocate(v1)
        ttnn.deallocate(w0_e)
        ttnn.deallocate(w1_e)
        return out

    def __call__(
        self, f0_bt1: ttnn.Tensor, *, deterministic: bool = False
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            f0_bt1: ``(B, T, 1)`` float32 TILE on device (``T == time_len``).

        Returns:
            ``(sine_waves, uv, noise)`` — same contract as ``kokoro_istftnet.SineGen.forward``:
            ``sine_waves`` is amplitude-scaled sines mixed with ``noise``; ``uv`` is ``(B, T, 1)``.
        """
        l1 = ttnn.L1_MEMORY_CONFIG
        f0 = ttnn.to_memory_config(f0_bt1, l1)
        if f0.dtype != ttnn.float32:
            f0 = ttnn.typecast(f0, ttnn.float32, memory_config=l1)

        bsz = int(f0.shape[0])
        tlen = int(f0.shape[1])
        if tlen != self.time_len:
            raise ValueError(f"SineGen time {tlen} != time_len {self.time_len}")

        uv_gt = ttnn.gt(f0, self._voiced_threshold, memory_config=l1)
        uv = ttnn.typecast(uv_gt, ttnn.float32, memory_config=l1)
        ttnn.deallocate(uv_gt)

        fn = ttnn.multiply(f0, self._harmonics, memory_config=l1)
        rad = ttnn.multiply(fn, self._inv_sr, memory_config=l1)
        ttnn.deallocate(fn)
        rad = ttnn.remainder(rad, self._one, memory_config=l1)

        if deterministic:
            rand_row = ttnn.zeros([bsz, 1, self.dim], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        else:
            rand_row = ttnn.rand(
                [bsz, 1, self.dim],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=l1,
            )
        rand_row = ttnn.multiply(rand_row, self._harmonic_rand_mask, memory_config=l1)

        if tlen > 1:
            tail = ttnn.zeros(
                [bsz, tlen - 1, self.dim],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=l1,
            )
            rand_pad = ttnn.concat([rand_row, tail], dim=1, memory_config=l1)
            ttnn.deallocate(tail)
        else:
            rand_pad = rand_row
        rad = ttnn.add(rad, rand_pad, memory_config=l1)
        ttnn.deallocate(rand_pad)
        if tlen > 1:
            ttnn.deallocate(rand_row)

        rad_bdt = ttnn.permute(rad, [0, 2, 1], memory_config=l1)
        ttnn.deallocate(rad)
        dram = ttnn.DRAM_MEMORY_CONFIG
        # 2-tap sparse linear downsample (exact fp32) — replaces a wide bf16-quantised mat_down matmul.
        rad_down = self._gather_lerp(
            rad_bdt, self._interp_down_idx0, self._interp_down_idx1, self._interp_down_w0, self._interp_down_w1
        )
        ttnn.deallocate(rad_bdt)

        # Cumsum via ``x @ triu(1)``: weights are 0/1, so bf16 quantisation is lossless.
        phase_i = ttnn.linear(
            rad_down,
            self._mat_cumsum,
            bias=None,
            memory_config=dram,
            compute_kernel_config=self._compute_cfg,
        )
        ttnn.deallocate(rad_down)
        # Match PyTorch's ``cumsum * 2π * upsample_scale`` rounding order (two roundings, not one fused mul).
        phase_2pi = ttnn.multiply(phase_i, self._two_pi, memory_config=dram)
        ttnn.deallocate(phase_i)
        phase_scaled = ttnn.multiply(phase_2pi, self._upsample_scale_tt, memory_config=dram)
        ttnn.deallocate(phase_2pi)

        # 2-tap sparse linear upsample of the scaled phase.
        phase_up = self._gather_lerp(
            phase_scaled, self._interp_up_idx0, self._interp_up_idx1, self._interp_up_w0, self._interp_up_w1
        )
        ttnn.deallocate(phase_scaled)

        sines = ttnn.sin(phase_up, memory_config=dram)
        ttnn.deallocate(phase_up)

        sine_waves = ttnn.multiply(sines, self._sine_amp_tt, memory_config=dram)
        ttnn.deallocate(sines)

        one_m_uv = ttnn.subtract(self._one, uv, memory_config=l1)
        noise_amp = ttnn.add(
            ttnn.multiply(uv, self._noise_std_tt, memory_config=l1),
            ttnn.multiply(one_m_uv, self._sine_amp_over_3, memory_config=l1),
            memory_config=l1,
        )
        ttnn.deallocate(one_m_uv)

        if deterministic:
            noise_raw = ttnn.zeros(
                [bsz, tlen, self.dim], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        else:
            noise_raw = ttnn.rand(
                [bsz, tlen, self.dim],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=l1,
            )
        noise = ttnn.multiply(noise_amp, noise_raw, memory_config=l1)
        ttnn.deallocate(noise_amp)
        ttnn.deallocate(noise_raw)

        # ``sine_waves`` is ``(B, dim, T)`` (same layout as ``phase_up``); ``uv`` / ``noise`` are ``(B, T, dim)``.
        sine_btd = ttnn.permute(sine_waves, [0, 2, 1], memory_config=l1)
        ttnn.deallocate(sine_waves)

        uv_exp = ttnn.repeat(uv, [1, 1, self.dim], memory_config=l1)
        scaled = ttnn.multiply(sine_btd, uv_exp, memory_config=l1)
        ttnn.deallocate(uv_exp)
        ttnn.deallocate(sine_btd)
        out = ttnn.add(scaled, noise, memory_config=l1)
        ttnn.deallocate(scaled)

        return out, uv, noise
