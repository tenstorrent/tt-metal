# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN-facing Kokoro ``SourceModuleHnNSF``: harmonic-plus-noise source for the generator.

The official path uses ``SineGen`` + ``Linear`` + ``Tanh`` in PyTorch with sensitive
``cumsum`` / ``sin`` numerics. On Wormhole, reproducing that on device was far from
reference PCC; this module therefore runs the **same** PyTorch ``SineGen`` on **CPU**,
then runs the final **Linear** + **Tanh** on **TTNN**, computes **uv** from ``f0`` on
device, and uploads only ``sine_wavs`` (and non-deterministic ``noise_merge`` when
needed) so the rest of ``KokoroGenerator`` stays on TTNN.

Weights come from
``models.experimental.kokoro.reference.kokoro_source_module_preprocess.preprocess_source_module_hn_nsf_parameters``.
"""

from __future__ import annotations

import unittest.mock
from typing import Tuple

import torch
import ttnn

from models.experimental.kokoro.reference.kokoro_istftnet import SineGen as TorchSineGen


def _zeros_rand(*args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
    return torch.zeros(*args, **kwargs)


def _zeros_randn(*args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
    return torch.zeros(*args, **kwargs)


def _zeros_randn_like(t, **kwargs):
    return torch.zeros_like(t)


def _source_linear_compute_cfg(device):
    """HiFi3 + fp32 acc on Wormhole B0 avoids known HiFi4 fp32 accuracy warning."""
    fidelity = ttnn.MathFidelity.HiFi3 if ttnn.device.is_wormhole_b0(device) else ttnn.MathFidelity.HiFi4
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class SourceModuleHnNSF:
    """Kokoro harmonic-plus-noise source; PyTorch ``SineGen`` on CPU, merge path on TTNN."""

    def __init__(self, device, parameters: dict):
        self.device = device
        self.time_len = int(parameters["time_len"])
        self.sine_amp = float(parameters["sine_amp"])
        self.dim = int(parameters["harmonic_num"]) + 1

        self._cpu_sg = TorchSineGen(
            int(parameters["sampling_rate"]),
            float(parameters["upsample_scale"]),
            int(parameters["harmonic_num"]),
            float(parameters["sine_amp"]),
            float(parameters["noise_std"]),
            float(parameters["voiced_threshold"]),
            bool(parameters["flag_for_pulse"]),
        )
        self._linear_weight = parameters["linear_weight"]
        self._linear_bias = parameters["linear_bias"]
        self._compute_cfg = _source_linear_compute_cfg(device)
        vt = float(parameters["voiced_threshold"])
        self._voiced_threshold = ttnn.from_torch(
            torch.tensor([[[vt]]], dtype=torch.float32),
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

        f0_cpu = ttnn.to_torch(f0_l1).to(torch.float32).contiguous()

        if deterministic:
            with (
                unittest.mock.patch("torch.rand", side_effect=_zeros_rand),
                unittest.mock.patch("torch.randn", side_effect=_zeros_randn),
                unittest.mock.patch("torch.randn_like", side_effect=_zeros_randn_like),
            ):
                with torch.inference_mode():
                    sine_wavs, _, _ = self._cpu_sg(f0_cpu)
        else:
            with torch.inference_mode():
                sine_wavs, _, _ = self._cpu_sg(f0_cpu)

        sine_wavs_tt = ttnn.from_torch(
            sine_wavs,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=l1,
        )
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
            noise_merge = ttnn.full(
                [bsz, tlen, 1],
                fill_value=0.0,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=l1,
            )
        else:
            noise_merge = ttnn.from_torch(
                torch.randn((bsz, tlen, 1), dtype=torch.float32, device="cpu") * (self.sine_amp / 3.0),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=l1,
            )

        return sine_merge, noise_merge, uv
