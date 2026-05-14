# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN ``AdaINResBlock1`` from Kokoro ``Generator`` / ``noise_res`` (not encoder ``AdainResBlk1d``)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import ttnn

from .ttnn_adain_resblk_encode import ADAIN_LONG_SEQ_DRAM_THRESHOLD, _adain_instance_norm


def _compute_cfg(device):
    # HiFi4 here matches PyTorch CPU more closely for these dilated conv1d stacks, despite the WH
    # HiFi4-fp32-accum HW bug warning — switching to HiFi3 regressed decoder e2e PCC.
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class _TtConv1dDilated:
    """Conv1d via conv2d; stride 1; dilation on the time axis.

    Long ``input_length`` (longer utterances) kept everything in L1 and collided with conv circular
    buffers. Match ``_StridedNoiseConv1d``: ROW_MAJOR activations in DRAM + ``prepare_conv_weights`` +
    height slice config so inference scales past short smoke phrases.
    """

    def __init__(
        self,
        device,
        weight_rm: ttnn.Tensor,
        bias_rm: Optional[ttnn.Tensor],
        *,
        kernel_size: int,
        padding_h: int,
        dilation_h: int,
    ):
        self.device = device
        self.weight_rm = weight_rm
        self.bias_rm = bias_rm
        self.out_channels = int(weight_rm.shape[0])
        self.in_channels = int(weight_rm.shape[1])
        self.kernel_size = int(kernel_size)
        self.padding_h = int(padding_h)
        self.dilation_h = int(dilation_h)
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
        self._prep_key: Optional[Tuple[int, int, int]] = None
        self.dram_slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=16)
        self.weight_prepared = weight_rm
        self.bias_prepared: Optional[ttnn.Tensor] = bias_rm

    def __call__(self, x: ttnn.Tensor, batch_size: int, input_length: int) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        l1 = ttnn.L1_MEMORY_CONFIG
        # Large (B,1,T,C) TILE in L1 collides with conv circular buffers on long harmonic frames.
        x = ttnn.to_memory_config(x, dram)
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32, memory_config=dram)
        x = ttnn.permute(x, [0, 2, 1], memory_config=dram)
        x = ttnn.reshape(x, [batch_size, 1, input_length, self.in_channels], memory_config=dram)
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.to_memory_config(x_rm, dram)

        key = (batch_size, input_length, self.dilation_h)
        has_bias = self.bias_rm is not None
        if self._prep_key != key:
            # ``num_slices`` must be ≤ the conv2d's output height (TT_FATAL in prepare_conv2d_weights).
            # Stride=1: ``out_h = input_length + 2*padding - dilation*(kernel-1)``.
            out_h = int(input_length) + 2 * int(self.padding_h) - int(self.dilation_h) * (int(self.kernel_size) - 1)
            out_h = max(1, out_h)
            target = max(16, (int(input_length) + 7) // 8)
            num_sl = max(2, min(min(512, target), out_h))
            self.dram_slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceHeight,
                num_slices=num_sl,
            )
            self.weight_prepared = ttnn.prepare_conv_weights(
                weight_tensor=self.weight_rm,
                input_memory_config=x_rm.memory_config(),
                input_layout=x_rm.layout,
                weights_format="OIHW",
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                input_height=input_length,
                input_width=1,
                kernel_size=(self.kernel_size, 1),
                stride=(1, 1),
                padding=(self.padding_h, 0),
                dilation=(self.dilation_h, 1),
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
                    input_height=input_length,
                    input_width=1,
                    kernel_size=(self.kernel_size, 1),
                    stride=(1, 1),
                    padding=(self.padding_h, 0),
                    dilation=(self.dilation_h, 1),
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
            input_height=input_length,
            input_width=1,
            kernel_size=(self.kernel_size, 1),
            stride=(1, 1),
            padding=(self.padding_h, 0),
            dilation=(self.dilation_h, 1),
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
        result = ttnn.reshape(result, [batch_size, oh_i, self.out_channels], memory_config=dram)
        result = ttnn.permute(result, [0, 2, 1], memory_config=dram)
        result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
        if input_length > ADAIN_LONG_SEQ_DRAM_THRESHOLD:
            return ttnn.to_memory_config(result, dram)
        return ttnn.to_memory_config(result, l1)


class AdaINResBlock1:
    """
    Matches ``AdaINResBlock1`` in ``kokoro_istftnet.py`` (three conv1 pairs + AdaIN + ``sin(alpha*x)**2``).
    """

    def __init__(self, device, parameters: dict[str, Any]):
        self.device = device
        self.channels = int(parameters["channels"])
        self.style_dim = int(parameters["style_dim"])
        self.eps = float(parameters.get("eps", 1e-5))
        self.compute_cfg = _compute_cfg(device)
        self.layers: List[dict[str, Any]] = parameters["layers"]
        self.convs1: List[_TtConv1dDilated] = []
        self.convs2: List[_TtConv1dDilated] = []
        for li in self.layers:
            self.convs1.append(
                _TtConv1dDilated(
                    device,
                    li["conv1"]["weight"],
                    li["conv1"]["bias"],
                    kernel_size=int(li["conv1"]["kernel_size"]),
                    padding_h=int(li["conv1"]["padding_h"]),
                    dilation_h=int(li["conv1"]["dilation_h"]),
                )
            )
            self.convs2.append(
                _TtConv1dDilated(
                    device,
                    li["conv2"]["weight"],
                    li["conv2"]["bias"],
                    kernel_size=int(li["conv2"]["kernel_size"]),
                    padding_h=int(li["conv2"]["padding_h"]),
                    dilation_h=int(li["conv2"]["dilation_h"]),
                )
            )

    def __call__(self, x: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
        seq = int(x.shape[2])
        mc = ttnn.DRAM_MEMORY_CONFIG if seq > ADAIN_LONG_SEQ_DRAM_THRESHOLD else ttnn.L1_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, mc)
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32, memory_config=mc)
        s = ttnn.to_memory_config(s, ttnn.L1_MEMORY_CONFIG)
        if s.dtype != ttnn.float32:
            s = ttnn.typecast(s, ttnn.float32)
        batch_size = int(x.shape[0])
        for i, li in enumerate(self.layers):
            h1 = ttnn.linear(
                s,
                li["norm1"]["fc_weight"],
                bias=li["norm1"]["fc_bias"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_cfg,
            )
            h2 = ttnn.linear(
                s,
                li["norm2"]["fc_weight"],
                bias=li["norm2"]["fc_bias"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_cfg,
            )
            xt = _adain_instance_norm(
                x,
                li["norm1"]["inst_weight"],
                li["norm1"]["inst_bias"],
                h1,
                self.channels,
                self.eps,
                memory_config=mc,
            )
            a1 = li["alpha1"]
            inv_a1 = li["inv_alpha1"]
            ax = ttnn.multiply(a1, xt, memory_config=mc)
            sn = ttnn.sin(ax, memory_config=mc)
            ttnn.deallocate(ax)
            sn2 = ttnn.multiply(sn, sn, memory_config=mc)
            ttnn.deallocate(sn)
            bump = ttnn.multiply(inv_a1, sn2, memory_config=mc)
            ttnn.deallocate(sn2)
            xt = ttnn.add(xt, bump, memory_config=mc)
            ttnn.deallocate(bump)
            xt = self.convs1[i](xt, batch_size, seq)
            xt = _adain_instance_norm(
                xt,
                li["norm2"]["inst_weight"],
                li["norm2"]["inst_bias"],
                h2,
                self.channels,
                self.eps,
                memory_config=mc,
            )
            a2 = li["alpha2"]
            inv_a2 = li["inv_alpha2"]
            ax2 = ttnn.multiply(a2, xt, memory_config=mc)
            snb = ttnn.sin(ax2, memory_config=mc)
            ttnn.deallocate(ax2)
            sn2b = ttnn.multiply(snb, snb, memory_config=mc)
            ttnn.deallocate(snb)
            bump2 = ttnn.multiply(inv_a2, sn2b, memory_config=mc)
            ttnn.deallocate(sn2b)
            xt = ttnn.add(xt, bump2, memory_config=mc)
            ttnn.deallocate(bump2)
            xt = self.convs2[i](xt, batch_size, seq)
            x = ttnn.add(x, xt, memory_config=mc)
            ttnn.deallocate(xt)
        return ttnn.to_memory_config(x, mc)
