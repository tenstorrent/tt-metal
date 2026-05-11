# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Kokoro ``SourceModuleHnNSF``: harmonic-plus-noise source for the generator.

``KokoroTtnnSineGen`` runs entirely on device; the final ``Linear`` + ``Tanh`` also run on TTNN.
Weights come from
``models.experimental.kokoro.reference.kokoro_source_module_preprocess.preprocess_source_module_hn_nsf_parameters``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import ttnn

from models.experimental.kokoro.tt.ttnn_sinegen import KokoroTtnnSineGen, sinegen_fp32_matmul_cfg


class SourceModuleHnNSF:
    """Kokoro harmonic-plus-noise source; all computation runs on TTNN device."""

    def __init__(self, device, parameters: dict):
        self.device = device
        self.time_len = int(parameters["time_len"])
        self.sine_amp = float(parameters["sine_amp"])
        self.dim = int(parameters["harmonic_num"]) + 1

        if bool(parameters.get("flag_for_pulse", False)):
            raise NotImplementedError("SourceModuleHnNSF: flag_for_pulse=True not supported on TTNN")

        self._ttnn_sg = KokoroTtnnSineGen(device, parameters)
        self._linear_weight = parameters["linear_weight"]
        self._linear_bias = parameters["linear_bias"]
        self._compute_cfg = sinegen_fp32_matmul_cfg(device)

        vt = float(parameters["voiced_threshold"])
        self._voiced_threshold = ttnn.from_torch(
            torch.tensor([[[vt]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sine_amp_over_3 = self.sine_amp / 3.0
        self._sine_amp_over_3 = ttnn.from_torch(
            torch.tensor([[[sine_amp_over_3]]], dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, f0: ttnn.Tensor, *, deterministic: bool = False) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            f0: ``(batch, time_len, 1)`` float32 TILE on device.

        Returns:
            ``(sine_merge, noise_merge, uv)`` each ``(batch, time_len, 1)`` float32 TILE on device.
        """
        bsz = int(f0.shape[0])
        tlen = int(f0.shape[1])
        if tlen != self.time_len:
            raise ValueError(f"f0 time dim {tlen} != preprocessed time_len {self.time_len}")
        if int(f0.shape[2]) != 1:
            raise ValueError("f0 must have shape (batch, time, 1)")

        l1 = ttnn.L1_MEMORY_CONFIG
        f0_l1 = ttnn.to_memory_config(f0, l1)
        if f0_l1.dtype != ttnn.float32:
            f0_l1 = ttnn.typecast(f0_l1, ttnn.float32, memory_config=l1)

        uv_gt = ttnn.gt(f0_l1, self._voiced_threshold, memory_config=l1)
        uv = ttnn.typecast(uv_gt, ttnn.float32, memory_config=l1)
        ttnn.deallocate(uv_gt)

        sine_wavs_tt, _, _ = self._ttnn_sg(f0_l1, deterministic=deterministic)

        pre = ttnn.linear(
            sine_wavs_tt,
            self._linear_weight,
            bias=self._linear_bias,
            memory_config=l1,
            compute_kernel_config=self._compute_cfg,
        )
        ttnn.deallocate(sine_wavs_tt)
        sine_merge = ttnn.tanh(pre, memory_config=l1)
        ttnn.deallocate(pre)

        if deterministic:
            noise_merge = ttnn.zeros(
                [bsz, tlen, 1],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=l1,
            )
        else:
            noise_raw = ttnn.rand(
                [bsz, tlen, 1],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=l1,
            )
            noise_merge = ttnn.multiply(noise_raw, self._sine_amp_over_3, memory_config=l1)
            ttnn.deallocate(noise_raw)

        return sine_merge, noise_merge, uv
