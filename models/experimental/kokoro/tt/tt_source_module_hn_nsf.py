# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.istftnet.SourceModuleHnNSF`.

Pipeline (with ``dim = harmonic_num + 1``):

    sine_wavs, uv, _ = l_sin_gen(f0)             # [B, T, dim], [B, T, 1]
    sine_merge       = tanh(l_linear(sine_wavs)) # [B, T, 1]
    noise            = randn_like(uv) * sine_amp / 3
    return sine_merge, noise, uv

Reuses :class:`~models.experimental.kokoro.tt.tt_sinegen.TTSineGen`. The output noise is a
*separate* tensor from the one mixed inside :class:`TTSineGen` (which is used internally for the
sine waveform); tests inject deterministic noise tensors when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

import ttnn

from .tt_sinegen import TTSineGen, TTSineGenParams, preprocess_tt_sinegen


@dataclass(frozen=True)
class TTSourceModuleHnNSFParams:
    """Device-resident weights for :class:`TTSourceModuleHnNSF`."""

    sinegen: TTSineGenParams
    linear_weight: ttnn.Tensor  # [1, dim] (``transpose_b=True``)
    linear_bias: ttnn.Tensor  # [1, 1, 1, 1]
    noise_scale: ttnn.Tensor  # [1, 1, 1] = sine_amp / 3
    dim: int
    sine_amp: float
    time_len: int


def preprocess_tt_source_module_hn_nsf(
    module: nn.Module,
    device: ttnn.Device,
    *,
    sampling_rate: float,
    upsample_scale: int,
    harmonic_num: int,
    voiced_threshold: float,
    time_len: int,
    weights_dtype=ttnn.bfloat16,
) -> TTSourceModuleHnNSFParams:
    """Upload reference ``SourceModuleHnNSF`` to device.

    ``sampling_rate`` / ``upsample_scale`` / ``harmonic_num`` / ``voiced_threshold`` / ``time_len``
    are passed explicitly because the reference stores them only inside ``l_sin_gen`` and the
    ``time_len`` axis must be known at construction time for :class:`TTSineGen`.
    """
    sine_amp = float(module.sine_amp)
    noise_std = float(module.noise_std)

    sinegen = preprocess_tt_sinegen(
        device=device,
        sampling_rate=sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=harmonic_num,
        sine_amp=sine_amp,
        noise_std=noise_std,
        voiced_threshold=voiced_threshold,
        time_len=time_len,
        weights_dtype=weights_dtype,
    )

    lin = module.l_linear
    w = ttnn.from_torch(
        lin.weight.detach().cpu(),  # [1, dim]
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        lin.bias.detach().cpu().reshape(1, 1, 1, -1),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    noise_scale = ttnn.from_torch(
        torch.tensor([[[sine_amp / 3.0]]], dtype=torch.float32),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return TTSourceModuleHnNSFParams(
        sinegen=sinegen,
        linear_weight=w,
        linear_bias=b,
        noise_scale=noise_scale,
        dim=int(harmonic_num) + 1,
        sine_amp=sine_amp,
        time_len=int(time_len),
    )


class TTSourceModuleHnNSF:
    """TTNN port of :class:`SourceModuleHnNSF`."""

    def __init__(
        self,
        device: ttnn.Device,
        params: TTSourceModuleHnNSFParams,
        *,
        use_torch_phase_fallback: bool = False,
        use_torch_sinegen_fallback: bool = False,
        use_torch_linear_fallback: bool = False,
        use_torch_tanh_fallback: bool = False,
    ) -> None:
        self.device = device
        self.params = params
        self._sinegen = TTSineGen(
            device,
            params.sinegen,
            use_torch_phase_fallback=use_torch_phase_fallback,
            use_torch_sinegen_fallback=use_torch_sinegen_fallback,
        )
        self._use_torch_linear_fallback = use_torch_linear_fallback
        self._use_torch_tanh_fallback = use_torch_tanh_fallback
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        # Pre-generate out_noise_raw [1, time_len, 1] once; reused every forward call.
        _out_noise_dummy = torch.zeros(1, params.time_len, 1, dtype=torch.float32)
        self._out_noise_raw = ttnn.from_torch(
            torch.randn_like(_out_noise_dummy),
            dtype=params.sinegen.activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(
        self,
        f0_bt1: ttnn.Tensor,
        *,
        sinegen_rand_ini: Optional[ttnn.Tensor] = None,
        sinegen_noise_raw: Optional[ttnn.Tensor] = None,
        out_noise_raw: Optional[ttnn.Tensor] = None,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            f0_bt1: ``[B, T, 1]`` fundamental frequency (Hz), ``T == params.time_len``.
            sinegen_rand_ini: optional ``[B, 1, dim]`` for SineGen's internal initial-phase noise.
            sinegen_noise_raw: optional ``[B, T, dim]`` raw noise mixed *inside* SineGen.
            out_noise_raw: optional ``[B, T, 1]`` raw Gaussian noise for the *output* noise branch
                (``randn_like(uv)`` in the reference). Uses the pre-generated
                ``self._out_noise_raw`` (tiled to batch size) if ``None``.

        Returns:
            ``(sine_merge, noise, uv)`` matching the reference contract — each ``[B, T, 1]``.
        """
        p = self.params
        B = int(f0_bt1.shape[0])

        sine_wavs, uv, _sine_noise = self._sinegen.forward(
            f0_bt1,
            rand_ini=sinegen_rand_ini,
            noise_raw=sinegen_noise_raw,
            memory_config=memory_config,
        )
        ttnn.deallocate(_sine_noise)

        # ``sine_merge = tanh(l_linear(sine_wavs))`` — Linear(dim, 1) over channel axis.
        if self._use_torch_linear_fallback:
            # CPU float32 linear: BH BF16 MACs on the dim=9 dot product introduce ~2%
            # relative error in sine_merge, which corrupts near-zero STFT bins downstream
            # even when torch.stft is used for the transform.
            dim = p.sinegen.dim
            x_cpu = ttnn.to_torch(sine_wavs).float().reshape(B * p.time_len, dim)
            w_cpu = ttnn.to_torch(p.linear_weight).float().reshape(1, dim)
            b_cpu = ttnn.to_torch(p.linear_bias).float().flatten()[:1]
            ttnn.deallocate(sine_wavs)
            merged_cpu = F_torch.linear(x_cpu, w_cpu, b_cpu).reshape(B, p.time_len, 1)
            if self._use_torch_tanh_fallback:
                sine_merge_cpu = torch.tanh(merged_cpu)
                sine_merge = ttnn.from_torch(
                    sine_merge_cpu.contiguous(),
                    dtype=p.sinegen.activation_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=memory_config,
                )
            else:
                merged = ttnn.from_torch(
                    merged_cpu.contiguous(),
                    dtype=p.sinegen.activation_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=memory_config,
                )
                sine_merge = ttnn.tanh(merged, memory_config=memory_config)
                ttnn.deallocate(merged)
        else:
            merged = ttnn.linear(
                sine_wavs,
                p.linear_weight,
                bias=p.linear_bias,
                transpose_b=True,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(sine_wavs)
            # ``ttnn.linear`` may pad to rank 4; squeeze leading singletons.
            while len(merged.shape) > 3:
                merged = ttnn.squeeze(merged, 0)
            if self._use_torch_tanh_fallback:
                merged_cpu = ttnn.to_torch(merged).float()
                ttnn.deallocate(merged)
                sine_merge_cpu = torch.tanh(merged_cpu)
                sine_merge = ttnn.from_torch(
                    sine_merge_cpu.contiguous(),
                    dtype=p.sinegen.activation_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=memory_config,
                )
            else:
                sine_merge = ttnn.tanh(merged, memory_config=memory_config)
                ttnn.deallocate(merged)

        # ``noise = randn_like(uv) * sine_amp / 3`` → [B, T, 1]
        if out_noise_raw is None:
            if B == 1:
                out_noise_raw_local = self._out_noise_raw
                owns_noise = False
            else:
                out_noise_raw_local = ttnn.concat([self._out_noise_raw] * B, dim=0, memory_config=memory_config)
                owns_noise = True
        else:
            out_noise_raw_local = out_noise_raw
            owns_noise = False
        noise = ttnn.multiply(out_noise_raw_local, p.noise_scale, memory_config=memory_config)
        if owns_noise:
            ttnn.deallocate(out_noise_raw_local)

        return sine_merge, noise, uv

    __call__ = forward
