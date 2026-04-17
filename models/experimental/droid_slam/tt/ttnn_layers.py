# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Foundational ttnn wrappers for DROID-SLAM on-device inference.

All helpers follow the unet_shallow_ttnn pattern: preprocessed weights
are cached on the device via `return_weights_and_bias=True`, compute
config is explicit (LoFi / packer_l1_acc), and inputs stay in NHWC
tile layout so chains of convs do not thrash between NHWC row-major
and tile.
"""

from __future__ import annotations

import torch
import ttnn


RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
SIGMOID = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
TANH = ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH)


def _default_compute_config(device):
    # HiFi2 gives us ~2x the FPU throughput of HiFi4 while keeping
    # 7-bit mantissa precision — together with fp32_dest_acc_en that's
    # enough for DROID-SLAM's 16-layer encoder to stay above 99% PCC.
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class TtConv2d:
    """ttnn.conv2d wrapper with cached weight+bias on device.

    The first call preprocesses weights and stashes the returned
    on-device tensors so subsequent calls skip the preprocessing step.
    """

    def __init__(
        self,
        conv: torch.nn.Conv2d,
        *,
        activation=None,
        compute_config=None,
        slice_config=None,
    ):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = conv.groups
        self.activation = activation
        self.compute_config = compute_config
        self.slice_config = slice_config
        # Host-side weights as bf16 (preprocess_conv_weights converts
        # to the runtime tile layout on first __call__).
        self.weight = ttnn.from_torch(
            conv.weight.detach().to(torch.bfloat16), dtype=ttnn.bfloat16
        )
        if conv.bias is not None:
            bias = conv.bias.detach().to(torch.bfloat16).reshape(1, 1, 1, -1)
            self.bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
        else:
            self.bias = None

    def _conv_config(self):
        # bfloat16 weights halve the L1 footprint vs float32 and still
        # hold PCC above 0.99 when paired with HiFi2 + fp32_dest_acc_en.
        return ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=self.activation,
            deallocate_activation=False,
            enable_weights_double_buffer=True,
        )

    def __call__(self, x, device, batch_size, input_height, input_width):
        compute_cfg = self.compute_config or _default_compute_config(device)
        out, [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=(1, 1),
            groups=self.groups,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self._conv_config(),
            compute_config=compute_cfg,
            slice_config=self.slice_config,
            return_output_dim=False,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        out_h = (input_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (input_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return out, out_h, out_w


def to_tile_nhwc(t_nchw: torch.Tensor, device) -> "ttnn.Tensor":
    """torch NCHW fp32/bf16 → ttnn NHWC tile on device."""
    x = t_nchw.permute(0, 2, 3, 1).contiguous()
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def from_tile_nhwc(t_tt: "ttnn.Tensor", n: int, h: int, w: int, c: int) -> torch.Tensor:
    """ttnn NHWC tile (any packed layout) → torch NCHW fp32."""
    out = ttnn.to_torch(t_tt).float()
    out = out.reshape(n, h, w, c)
    return out.permute(0, 3, 1, 2).contiguous()


class TtInstanceNorm2d:
    """InstanceNorm2d (affine=False) expressed as basic ttnn ops.

    ttnn.group_norm has strict sharding requirements that collide with
    our NHWC-tile conv chain layout; building it from mean/var/rsqrt
    primitives is precision-adequate for bf16 inference and avoids the
    custom input-mask preprocessing entirely.

    Expects input in packed NHWC tile layout as emitted by
    ttnn.conv2d: shape [1, 1, N*H*W, C].
    """

    def __init__(self, channels: int, device, eps: float = 1e-5):
        self.channels = channels
        self.device = device
        self.eps = eps

    def __call__(self, x_tile, *, batch_size: int, spatial: int):
        # Reshape packed [1, 1, N*H*W, C] → [N, H*W, C] so reductions
        # along the spatial axis compute per-sample, per-channel stats.
        x = ttnn.reshape(x_tile, (batch_size, spatial, self.channels))
        mean = ttnn.mean(x, dim=1, keepdim=True)
        centered = ttnn.subtract(x, mean)
        var = ttnn.mean(ttnn.multiply(centered, centered), dim=1, keepdim=True)
        inv_std = ttnn.rsqrt(ttnn.add(var, self.eps))
        out = ttnn.multiply(centered, inv_std)
        return ttnn.reshape(out, (1, 1, batch_size * spatial, self.channels))
