# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN Kokoro ``Generator`` (ISTFTNet vocoder stack); forward uses only ``ttnn``."""

from __future__ import annotations

from typing import Any, Optional

import ttnn

from .ttnn_adain_resblk_encode import _TtConv1d
from .ttnn_adain_resblock1 import AdaINResBlock1
from .ttnn_kokoro_stft import KokoroConvStft
from .ttnn_source_module_hn_nsf import SourceModuleHnNSF


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class _StridedNoiseConv1d:
    """Strided Conv1d on the time axis (harmonic conditioning path)."""

    def __init__(self, device, spec: dict[str, Any]):
        self.device = device
        self.weight_rm = spec["weight"]
        self.bias_rm: Optional[ttnn.Tensor] = spec.get("bias")
        self.stride = int(spec["stride"])
        self.kernel_size = int(spec["kernel_size"])
        self.padding_h = int(spec["padding"])
        self.in_channels = int(spec["in_channels"])
        self.out_channels = int(spec["out_channels"])
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            config_tensors_in_dram=True,
            reshard_if_not_optimal=False,
            enable_kernel_stride_folding=False,
            force_split_reader=False,
            transpose_shards=False,
            enable_activation_reuse=False,
            full_inner_dim=False,
        )
        self.compute_cfg = _compute_cfg(device)
        self.dram_slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=8)
        self._prep_key: Optional[tuple[int, int]] = None
        self.weight_prepared = self.weight_rm
        self.bias_prepared: Optional[ttnn.Tensor] = self.bias_rm

    def __call__(self, x_bcl: ttnn.Tensor, batch_size: int, input_len: int) -> ttnn.Tensor:
        x_bcl = ttnn.to_memory_config(x_bcl, ttnn.L1_MEMORY_CONFIG)
        if x_bcl.dtype != ttnn.float32:
            x_bcl = ttnn.typecast(x_bcl, ttnn.float32)
        x_rm = ttnn.permute(x_bcl, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        x_rm = ttnn.reshape(x_rm, [batch_size, 1, input_len, self.in_channels], memory_config=ttnn.L1_MEMORY_CONFIG)
        x_rm = ttnn.to_layout(x_rm, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.to_memory_config(x_rm, ttnn.DRAM_MEMORY_CONFIG)
        key = (batch_size, input_len)
        has_bias = self.bias_rm is not None
        if self._prep_key != key:
            self.weight_prepared = ttnn.prepare_conv_weights(
                weight_tensor=self.weight_rm,
                input_memory_config=x_rm.memory_config(),
                input_layout=x_rm.layout,
                weights_format="OIHW",
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                input_height=input_len,
                input_width=1,
                kernel_size=(self.kernel_size, 1),
                stride=(self.stride, 1),
                padding=(self.padding_h, 0),
                dilation=(1, 1),
                has_bias=has_bias,
                groups=1,
                device=self.device,
                input_dtype=x_rm.dtype,
                conv_config=self.conv_config,
                compute_config=self.compute_cfg,
                slice_config=self.dram_slice_config,
            )
            if has_bias:
                self.bias_prepared = ttnn.prepare_conv_bias(
                    bias_tensor=self.bias_rm,
                    input_memory_config=x_rm.memory_config(),
                    input_layout=x_rm.layout,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    batch_size=batch_size,
                    input_height=input_len,
                    input_width=1,
                    kernel_size=(self.kernel_size, 1),
                    stride=(self.stride, 1),
                    padding=(self.padding_h, 0),
                    dilation=(1, 1),
                    groups=1,
                    device=self.device,
                    input_dtype=x_rm.dtype,
                    conv_config=self.conv_config,
                    compute_config=self.compute_cfg,
                )
            else:
                self.bias_prepared = None
            self._prep_key = key

        result, [oh, _ow], wpair = ttnn.conv2d(
            input_tensor=x_rm,
            weight_tensor=self.weight_prepared,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_len,
            input_width=1,
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
            padding=(self.padding_h, 0),
            dilation=(1, 1),
            bias_tensor=self.bias_prepared,
            conv_config=self.conv_config,
            compute_config=self.compute_cfg,
            slice_config=self.dram_slice_config,
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        self.weight_prepared = wpair[0]
        if has_bias and len(wpair) > 1:
            self.bias_prepared = wpair[1]
        oh_i = int(oh)
        result = ttnn.reshape(result, [batch_size, oh_i, self.out_channels], memory_config=ttnn.L1_MEMORY_CONFIG)
        result = ttnn.permute(result, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)


class _UpsConvTranspose1d:
    """ConvTranspose1d along time via ``conv_transpose2d``."""

    def __init__(self, device, spec: dict[str, Any]):
        self.device = device
        self.weight = spec["weight"]
        self.bias_rm: Optional[ttnn.Tensor] = spec.get("bias")
        self.in_channels = int(spec["in_channels"])
        self.out_channels = int(spec["out_channels"])
        self.kernel_size = int(spec["kernel_size"])
        self.stride = int(spec["stride"])
        self.padding_h = int(spec["padding"])
        self.output_padding = int(spec["output_padding"])
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            config_tensors_in_dram=True,
            reshard_if_not_optimal=False,
            enable_kernel_stride_folding=False,
            force_split_reader=False,
            transpose_shards=False,
            enable_activation_reuse=False,
            full_inner_dim=False,
        )
        self.compute_cfg = _compute_cfg(device)

    def __call__(self, x_bcl: ttnn.Tensor, batch_size: int, input_len: int) -> ttnn.Tensor:
        x_bcl = ttnn.to_memory_config(x_bcl, ttnn.L1_MEMORY_CONFIG)
        if x_bcl.dtype != ttnn.float32:
            x_bcl = ttnn.typecast(x_bcl, ttnn.float32)
        x = ttnn.permute(x_bcl, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, [batch_size, 1, input_len, self.in_channels], memory_config=ttnn.L1_MEMORY_CONFIG)
        result, [oh, ow], wpair = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias_rm,
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
            padding=(self.padding_h, 0),
            output_padding=(self.output_padding, 0),
            dilation=(1, 1),
            batch_size=batch_size,
            input_height=input_len,
            input_width=1,
            conv_config=self.conv_config,
            compute_config=self.compute_cfg,
            groups=1,
            mirror_kernel=False,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.float32,
        )
        self.weight = wpair[0]
        if self.bias_rm is not None and len(wpair) > 1:
            self.bias_rm = wpair[1]
        flat = int(oh) * int(ow)
        result = ttnn.reshape(result, [batch_size, flat, self.out_channels], memory_config=ttnn.L1_MEMORY_CONFIG)
        result = ttnn.permute(result, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)


def _reflect_pad1_left_bct(x: ttnn.Tensor) -> ttnn.Tensor:
    """``ReflectionPad1d((1,0))`` on the time axis for layout ``(B, C, T)``."""
    b = int(x.shape[0])
    c = int(x.shape[1])
    t = int(x.shape[2])
    if t < 2:
        return x
    left = ttnn.slice(x, [0, 0, 1], [b, c, 2])
    return ttnn.concat([left, x], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)


class KokoroGenerator:
    """Kokoro vocoder generator; ``__call__`` matches ``kokoro_istftnet.Generator.forward``."""

    def __init__(self, device, parameters: dict[str, Any]):
        self.device = device
        self.num_upsamples = int(parameters["num_upsamples"])
        self.num_kernels = int(parameters["num_kernels"])
        self.post_n_fft = int(parameters["post_n_fft"])
        self.half_bins = self.post_n_fft // 2 + 1
        sf = float(parameters["f0_up_scale"])
        self.f0_up_scale_int = int(round(sf))
        if abs(sf - float(self.f0_up_scale_int)) > 1e-6:
            raise ValueError(f"TTNN generator expects integer f0 upsample scale, got {sf!r}")
        self.inv_nk = ttnn.full(
            [1, 1, 1],
            fill_value=float(parameters["inv_num_kernels"]),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.m_source = SourceModuleHnNSF(device, parameters["m_source"])
        self.stft = KokoroConvStft(device, parameters["stft"])
        self.noise_convs = [_StridedNoiseConv1d(device, s) for s in parameters["noise_convs"]]
        self.noise_res = [AdaINResBlock1(device, p) for p in parameters["noise_res"]]
        self.ups = [_UpsConvTranspose1d(device, s) for s in parameters["ups"]]
        self.resblocks = [AdaINResBlock1(device, p) for p in parameters["resblocks"]]
        self.conv_post = _TtConv1d(device, parameters["conv_post"]["weight"], parameters["conv_post"]["bias"])

    def __call__(
        self,
        x: ttnn.Tensor,
        s: ttnn.Tensor,
        f0_coarse: ttnn.Tensor,
        *,
        deterministic: bool = False,
    ) -> ttnn.Tensor:
        """
        Args:
            x: ``(B, C, T)`` decoder output (last ``AdainResBlk1d``), float32 TILE.
            s: ``(B, style_dim)`` style, float32 TILE.
            f0_coarse: ``(B, T_coarse, 1)`` F0 curve before upsampling (nearest ``f0_up_scale``).
        """
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        s = ttnn.to_memory_config(s, ttnn.L1_MEMORY_CONFIG)
        if s.dtype != ttnn.float32:
            s = ttnn.typecast(s, ttnn.float32)
        f0_coarse = ttnn.to_memory_config(f0_coarse, ttnn.L1_MEMORY_CONFIG)
        if f0_coarse.dtype != ttnn.float32:
            f0_coarse = ttnn.typecast(f0_coarse, ttnn.float32)

        batch_size = int(x.shape[0])
        f0_up = ttnn.repeat_interleave(f0_coarse, self.f0_up_scale_int, dim=1)
        har_bt1, _noise, _uv = self.m_source(f0_up, deterministic=deterministic)
        har_b1t = ttnn.permute(har_bt1, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        t_audio = int(har_b1t.shape[2])
        mag, phase = self.stft.transform(har_b1t)
        har_cat = ttnn.concat([mag, phase], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(mag)
        ttnn.deallocate(phase)
        ttnn.deallocate(har_b1t)

        har_frames = int(har_cat.shape[2])

        for i in range(self.num_upsamples):
            x = ttnn.leaky_relu(x, negative_slope=0.1)
            x_src = self.noise_convs[i](har_cat, batch_size, har_frames)
            x_src = self.noise_res[i](x_src, s)
            seq = int(x.shape[2])
            x = self.ups[i](x, batch_size, seq)
            if i == self.num_upsamples - 1:
                x = _reflect_pad1_left_bct(x)
            x = ttnn.add(x, x_src, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(x_src)
            xs: Optional[ttnn.Tensor] = None
            base = i * self.num_kernels
            for j in range(self.num_kernels):
                y = self.resblocks[base + j](x, s)
                if xs is None:
                    xs = y
                else:
                    acc = ttnn.add(xs, y, memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(xs)
                    ttnn.deallocate(y)
                    xs = acc
            assert xs is not None
            x = ttnn.multiply(xs, self.inv_nk, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(xs)

        ttnn.deallocate(har_cat)

        x = ttnn.leaky_relu(x, negative_slope=0.01)
        seq2 = int(x.shape[2])
        x = self.conv_post(x, batch_size, seq2)
        spec_lin = ttnn.slice(x, [0, 0, 0], [batch_size, self.half_bins, seq2])
        phase_lin = ttnn.slice(x, [0, self.half_bins, 0], [batch_size, self.post_n_fft + 2, seq2])
        ttnn.deallocate(x)
        spec = ttnn.exp(spec_lin, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(spec_lin)
        phase = ttnn.sin(phase_lin, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(phase_lin)
        wave = self.stft.inverse(spec, phase, length=None)
        ttnn.deallocate(spec)
        ttnn.deallocate(phase)
        return ttnn.to_memory_config(wave, ttnn.L1_MEMORY_CONFIG)
