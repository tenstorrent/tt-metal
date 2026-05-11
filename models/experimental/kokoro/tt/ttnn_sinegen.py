# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``SineGen`` (Kokoro ``flag_for_pulse=False`` path).

Numerics vs float32 PyTorch are dominated by TILE ``cumsum`` / ``sin`` and interpolation ``linear``.
Improvements applied here: host-fused ``(2π)·upsample_scale``, ``ttnn.remainder`` for ``% 1``,
``sinegen_fp32_matmul_cfg`` (HiFi3 + fp32 dest + packer L1 acc + dst sync on Wormhole B0),
and a DRAM-backed phase block (downsample → cumsum → scale → upsample → sin) to reduce L1 drift.
"""

from __future__ import annotations

from typing import Any, Tuple

import math
import torch
import ttnn


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

    Uses ``mat_down`` / ``mat_up`` from ``preprocess_source_module_hn_nsf_parameters``.
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
        self._harmonics = parameters["harmonics"]
        self._harmonic_rand_mask = parameters["harmonic_rand_mask"]
        self._two_pi_times_upsample = parameters.get("two_pi_times_upsample")
        if self._two_pi_times_upsample is None:
            v = 2.0 * math.pi * float(parameters["upsample_scale"])
            self._two_pi_times_upsample = ttnn.from_torch(
                torch.tensor([[[v]]], dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        rad_down = ttnn.linear(
            rad_bdt,
            self._mat_down,
            bias=None,
            memory_config=dram,
            compute_kernel_config=self._compute_cfg,
        )
        ttnn.deallocate(rad_bdt)

        phase_i = ttnn.cumsum(rad_down, dim=2, dtype=ttnn.float32)
        ttnn.deallocate(rad_down)
        phase_scaled = ttnn.multiply(phase_i, self._two_pi_times_upsample, memory_config=dram)
        ttnn.deallocate(phase_i)

        phase_up = ttnn.linear(
            phase_scaled,
            self._mat_up,
            bias=None,
            memory_config=dram,
            compute_kernel_config=self._compute_cfg,
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
