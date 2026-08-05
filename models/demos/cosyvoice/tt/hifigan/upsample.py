# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ConvTranspose1d for HiFT's upsampling stages.

TTNN has `conv1d`, `conv2d` and `conv_transpose2d` but **no native 1-D
transpose** -- an odd asymmetry, since the forward `conv1d` exists as a 1-D
specialisation and the precedent for the transposed twin is right there in the
tree. So the 1-D transpose is expressed as a 2-D one with `H = 1`, which is the
same trick istft.py uses for overlap-add and which was verified to work at
`H=1, in_ch=16, k=16, stride=4`.

HiFT upsamples twice, `upsample_rates [8, 8]` with `upsample_kernel_sizes
[16, 16]` and `padding = (k - u) // 2 = 4`, taking the mel frame rate up by 64.
The remaining factor of 4 comes from the iSTFT hop, for a total of 256 -- which
matches the mel `hop_size`.

Whether the `H=1` path carries large constant overhead is the measurement
`03_plan.md` P3 needs to justify proposing a native `ttnn.conv_transpose1d`.
"""
from __future__ import annotations

import torch

import ttnn

from .conv import accurate_compute_config, extract_conv_weights


class TtConvTranspose1d:
    """Transposed 1-D convolution via conv_transpose2d at H=1.

    Input and output are channels-last `[N, L, C]`, matching conv.py.
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        high_fidelity: bool = True,
    ):
        # torch ConvTranspose1d weight is [in_ch, out_ch/groups, k]; TTNN's
        # conv_transpose2d wants (C, O/G, K_H, K_W), so the extra H axis is 1.
        assert weight.dim() == 3, f"expected [in_ch, out_ch/groups, k], got {tuple(weight.shape)}"
        self.device = device
        self.in_channels, self.out_per_group, self.kernel_size = weight.shape
        self.out_channels = self.out_per_group * groups
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        self.dtype = dtype

        self.weight = ttnn.from_torch(weight.unsqueeze(2), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bias = None
        if bias is not None:
            self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.compute_config = accurate_compute_config(device) if high_fidelity else None

    @classmethod
    def from_module(cls, device, module: torch.nn.Module, **kw):
        w, b = extract_conv_weights(module)
        return cls(
            device,
            w,
            b,
            stride=int(module.stride[0]),
            padding=int(module.padding[0]),
            dilation=int(module.dilation[0]),
            groups=int(module.groups),
            **kw,
        )

    def out_length(self, length: int) -> int:
        return (length - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1

    def __call__(self, x, length: int, batch_size: int = 1):
        """x: ttnn [B, L, C_in] -> (ttnn [B, L_out, C_out], L_out)."""
        nhwc = ttnn.reshape(x, (batch_size, 1, length, self.in_channels))
        out = ttnn.conv_transpose2d(
            input_tensor=nhwc,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=length,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, self.dilation),
            groups=self.groups,
            compute_config=self.compute_config,
            dtype=self.dtype,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        lo = self.out_length(length)
        return ttnn.reshape(out, (batch_size, lo, self.out_channels)), lo

    @staticmethod
    def torch_reference(x: torch.Tensor, weight, bias, stride=1, padding=0) -> torch.Tensor:
        """[B, C, L] in and out, as the reference module works."""
        return torch.nn.functional.conv_transpose1d(x, weight, bias, stride=stride, padding=padding)
