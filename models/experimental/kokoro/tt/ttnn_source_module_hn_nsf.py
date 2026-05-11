# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Kokoro ``SourceModuleHnNSF`` (harmonic-plus-noise source).

Reference: ``models.experimental.kokoro.reference.kokoro_istftnet.SourceModuleHnNSF``
and ``SineGen`` (hn-nsf, ``flag_for_pulse=False`` only).

Weights and resampling matrices are produced by
``models.experimental.kokoro.reference.kokoro_source_module_preprocess.preprocess_source_module_hn_nsf_parameters``
(PyTorch on host). This module imports only ``ttnn`` and the standard library.
"""

from __future__ import annotations

from typing import Tuple

import ttnn


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class SourceModuleHnNSF:
    """Kokoro harmonic-plus-noise source; forward is TTNN on device only."""

    def __init__(self, device, parameters: dict):
        self.device = device
        self.time_len = int(parameters["time_len"])
        self.t_down = int(parameters["t_down"])
        self.upsample_scale = float(parameters["upsample_scale"])
        self.dim = int(parameters["harmonic_num"]) + 1
        self.sine_amp = float(parameters["sine_amp"])
        self.noise_std = float(parameters["noise_std"])
        self.voiced_threshold = float(parameters["voiced_threshold"])
        self.compute_cfg = _compute_cfg(device)

        self.linear_w = parameters["linear_weight"]
        self.linear_b = parameters["linear_bias"]
        self.mat_down = parameters["mat_down"]
        self.mat_up = parameters["mat_up"]
        self.harmonics = parameters["harmonics"]
        self.harmonic_rand_mask = parameters["harmonic_rand_mask"]
        self.two_pi = parameters["two_pi"]
        self.inv_sr = parameters["inv_sampling_rate"]
        self.one = parameters["one"]
        self._rng = 0

    def _next_seed(self) -> int:
        self._rng += 1
        return self._rng

    def __call__(self, f0: ttnn.Tensor, *, deterministic: bool = False) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            f0: ``(batch, time_len, 1)`` float32 TILE, same ``time_len`` as preprocess.

        Returns:
            ``(sine_merge, noise_merge, uv)`` each ``(batch, time_len, 1)``.

        ``deterministic=True`` fixes random draws to zero (matches a seeded reference
        where ``torch.rand`` / ``torch.randn`` are zeroed); randomness still uses
        ``ttnn.rand`` / ``ttnn.randn`` on device.
        """
        bsz = int(f0.shape[0])
        tlen = int(f0.shape[1])
        if tlen != self.time_len:
            raise ValueError(f"f0 time dim {tlen} != preprocessed time_len {self.time_len}")
        if int(f0.shape[2]) != 1:
            raise ValueError("f0 must have shape (batch, time, 1)")

        x = ttnn.to_memory_config(f0, ttnn.L1_MEMORY_CONFIG)
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)

        harm = ttnn.to_memory_config(self.harmonics, ttnn.L1_MEMORY_CONFIG)
        fn = ttnn.multiply(x, harm, memory_config=ttnn.L1_MEMORY_CONFIG)
        inv = ttnn.to_memory_config(self.inv_sr, ttnn.L1_MEMORY_CONFIG)
        rad_acc = ttnn.multiply(fn, inv, memory_config=ttnn.L1_MEMORY_CONFIG)
        rad_acc = ttnn.remainder(rad_acc, self.one, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(fn)

        if deterministic:
            rand_ini = ttnn.zeros((bsz, self.dim), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        else:
            rand_ini = ttnn.rand(
                (bsz, self.dim),
                device=self.device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                low=0.0,
                high=1.0,
                seed=self._next_seed(),
            )
        hrm = ttnn.to_memory_config(self.harmonic_rand_mask, ttnn.L1_MEMORY_CONFIG)
        rand_ini = ttnn.multiply(rand_ini, hrm, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(hrm)

        ri = ttnn.reshape(rand_ini, [bsz, 1, self.dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(rand_ini)
        r0 = ttnn.slice(rad_acc, [0, 0, 0], [bsz, 1, self.dim])
        r0 = ttnn.add(r0, ri, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ri)
        if tlen > 1:
            rrest = ttnn.slice(rad_acc, [0, 1, 0], [bsz, tlen, self.dim])
            rad = ttnn.concat([r0, rrest], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(r0)
            ttnn.deallocate(rrest)
        else:
            rad = r0
        ttnn.deallocate(rad_acc)

        rad_bct = ttnn.permute(rad, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(rad)
        flat = bsz * self.dim
        rad_flat = ttnn.reshape(rad_bct, [flat, self.time_len], memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(rad_bct)
        md = ttnn.to_memory_config(self.mat_down, ttnn.L1_MEMORY_CONFIG)
        rad_low = ttnn.linear(
            rad_flat,
            md,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_cfg,
            dtype=ttnn.float32,
        )
        ttnn.deallocate(rad_flat)
        ttnn.deallocate(md)
        rad_low = ttnn.reshape(rad_low, [bsz, self.dim, self.t_down], memory_config=ttnn.L1_MEMORY_CONFIG)
        rad_bt_c = ttnn.permute(rad_low, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(rad_low)

        phase_low = ttnn.cumsum(rad_bt_c, dim=1, dtype=ttnn.float32)
        ttnn.deallocate(rad_bt_c)
        tp = ttnn.to_memory_config(self.two_pi, ttnn.L1_MEMORY_CONFIG)
        phase_low = ttnn.multiply(phase_low, tp, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tp)
        phase_scaled = ttnn.multiply(phase_low, self.upsample_scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(phase_low)

        ps_bct = ttnn.permute(phase_scaled, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(phase_scaled)
        ps_flat = ttnn.reshape(ps_bct, [flat, self.t_down], memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ps_bct)
        mu = ttnn.to_memory_config(self.mat_up, ttnn.L1_MEMORY_CONFIG)
        phase_full = ttnn.linear(
            ps_flat,
            mu,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_cfg,
            dtype=ttnn.float32,
        )
        ttnn.deallocate(ps_flat)
        ttnn.deallocate(mu)
        phase_full = ttnn.reshape(phase_full, [bsz, self.dim, self.time_len], memory_config=ttnn.L1_MEMORY_CONFIG)
        phase_btc = ttnn.permute(phase_full, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(phase_full)

        sines = ttnn.sin(phase_btc, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(phase_btc)
        sines = ttnn.multiply(sines, self.sine_amp, memory_config=ttnn.L1_MEMORY_CONFIG)

        uv_mask = ttnn.gt(x, self.voiced_threshold, memory_config=ttnn.L1_MEMORY_CONFIG)
        uv = ttnn.typecast(uv_mask, ttnn.float32)

        one_m_uv = ttnn.subtract(self.one, uv, memory_config=ttnn.L1_MEMORY_CONFIG)
        term_a = ttnn.multiply(uv, self.noise_std, memory_config=ttnn.L1_MEMORY_CONFIG)
        term_b = ttnn.multiply(one_m_uv, self.sine_amp / 3.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        noise_amp = ttnn.add(term_a, term_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(term_a)
        ttnn.deallocate(term_b)
        ttnn.deallocate(one_m_uv)

        if deterministic:
            noise_sine = ttnn.zeros(sines.shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        else:
            noise_sine = ttnn.randn(
                list(sines.shape),
                device=self.device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                seed=self._next_seed(),
            )
        noise_sine = ttnn.multiply(noise_amp, noise_sine, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(noise_amp)

        voiced_sines = ttnn.multiply(sines, uv, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(sines)
        sine_waves = ttnn.add(voiced_sines, noise_sine, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(voiced_sines)
        ttnn.deallocate(noise_sine)

        sine_merge = ttnn.linear(
            sine_waves,
            self.linear_w,
            bias=self.linear_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_cfg,
            dtype=ttnn.float32,
        )
        ttnn.deallocate(sine_waves)
        sine_merge = ttnn.tanh(sine_merge, memory_config=ttnn.L1_MEMORY_CONFIG)

        if deterministic:
            noise_merge = ttnn.zeros(uv.shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        else:
            noise_merge = ttnn.randn(
                list(uv.shape),
                device=self.device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                seed=self._next_seed(),
            )
        noise_merge = ttnn.multiply(noise_merge, self.sine_amp / 3.0, memory_config=ttnn.L1_MEMORY_CONFIG)

        return (
            ttnn.to_memory_config(sine_merge, ttnn.L1_MEMORY_CONFIG),
            ttnn.to_memory_config(noise_merge, ttnn.L1_MEMORY_CONFIG),
            ttnn.to_memory_config(uv, ttnn.L1_MEMORY_CONFIG),
        )
