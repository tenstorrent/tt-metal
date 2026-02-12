# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

try:
    import ttnn  # type: ignore
except Exception:  # pragma: no cover
    ttnn = None  # type: ignore


def _require_ttnn():
    if ttnn is None:
        raise RuntimeError("TTNN runtime is not available")


def _shape4(x) -> Tuple[int, int, int, int]:
    shape = tuple(int(v) for v in tuple(x.shape))
    if len(shape) != 4:
        raise ValueError(f"Expected rank-4 tensor, got shape={shape}")
    return shape  # type: ignore[return-value]


def _to_row_major(x):
    _require_ttnn()
    if hasattr(x, "layout") and x.layout == ttnn.TILE_LAYOUT:
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return x


def ensure_tt_device_tensor(x, tt_device):
    _require_ttnn()
    if tt_device is None:
        return x

    if isinstance(x, ttnn.Tensor):
        try:
            if x.storage_type() == ttnn.StorageType.DEVICE:
                return x
        except Exception:
            pass
        try:
            return x.to(tt_device)
        except Exception as exc:
            raise RuntimeError("Expected a TT tensor on device storage") from exc

    if torch.is_tensor(x):
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
        )

    raise TypeError(f"Unsupported tensor type for TT conversion: {type(x)!r}")


def tt_relu(x):
    _require_ttnn()
    try:
        return ttnn.relu(x, output_tensor=x)
    except TypeError:
        return ttnn.relu(x)


def _nchw_to_nhwc(x):
    _require_ttnn()
    x = _to_row_major(x)
    b, c, h, w = _shape4(x)
    x_nhwc = ttnn.permute(x, (0, 2, 3, 1))
    return x_nhwc, b, c, h, w


def _nhwc_to_nchw(x_nhwc, b: int, h: int, w: int, c: int):
    _require_ttnn()
    _shape4(x_nhwc)
    return ttnn.permute(x_nhwc, (0, 3, 1, 2))


def tt_upsample_nchw(
    x,
    *,
    scale_factor: int | Sequence[int] = 2,
    mode: str = "bilinear",
    memory_config=None,
):
    _require_ttnn()
    x = _to_row_major(x)
    b, c, h, w = _shape4(x)
    x_nhwc = ttnn.permute(x, (0, 2, 3, 1))

    sf = scale_factor
    if isinstance(sf, (tuple, list)):
        sf = [int(sf[0]), int(sf[1])]
    else:
        sf = int(sf)

    y_nhwc = ttnn.upsample(
        input_tensor=x_nhwc,
        scale_factor=sf,
        mode=mode,
        memory_config=memory_config,
    )

    out_h = h * (sf[0] if isinstance(sf, list) else sf)
    out_w = w * (sf[1] if isinstance(sf, list) else sf)
    y_nchw = ttnn.permute(y_nhwc, (0, 3, 1, 2))
    _shape4(y_nchw)
    if int(y_nchw.shape[-2]) != out_h or int(y_nchw.shape[-1]) != out_w:
        # Keep shape checks explicit for easier debug when runtime APIs change.
        raise RuntimeError(
            f"Unexpected TT upsample output shape: got {tuple(y_nchw.shape)} expected HxW={out_h}x{out_w}"
        )
    return y_nchw


def tt_resize_to_nchw(
    x,
    *,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
    memory_config=None,
):
    _require_ttnn()
    _, _, h, w = _shape4(x)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h == h and target_w == w:
        return x
    if target_h < h or target_w < w:
        raise RuntimeError(
            f"TT resize helper only supports upsample in fast path: src={h}x{w}, target={target_h}x{target_w}"
        )
    if target_h % h != 0 or target_w % w != 0:
        raise RuntimeError(
            f"Non-integer resize ratio is unsupported in TT fast path: src={h}x{w}, target={target_h}x{target_w}"
        )
    return tt_upsample_nchw(
        x,
        scale_factor=(target_h // h, target_w // w),
        mode=mode,
        memory_config=memory_config,
    )


@dataclass
class TTConv2dCached:
    weight_torch: torch.Tensor
    bias_torch: Optional[torch.Tensor]
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    mirror_kernel: bool = True

    _weight: object = None
    _bias: object = None

    @classmethod
    def from_conv(cls, conv: torch.nn.Conv2d) -> "TTConv2dCached":
        bias = conv.bias.detach() if conv.bias is not None else None
        return cls(
            weight_torch=conv.weight.detach(),
            bias_torch=bias,
            out_channels=int(conv.out_channels),
            kernel_size=(int(conv.kernel_size[0]), int(conv.kernel_size[1])),
            stride=(int(conv.stride[0]), int(conv.stride[1])),
            padding=(int(conv.padding[0]), int(conv.padding[1])),
            dilation=(int(conv.dilation[0]), int(conv.dilation[1])),
            groups=int(conv.groups),
        )

    @classmethod
    def from_tensors(
        cls,
        *,
        weight_torch: torch.Tensor,
        bias_torch: Optional[torch.Tensor],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> "TTConv2dCached":
        return cls(
            weight_torch=weight_torch.detach(),
            bias_torch=(bias_torch.detach() if bias_torch is not None else None),
            out_channels=int(weight_torch.shape[0]),
            kernel_size=(int(weight_torch.shape[2]), int(weight_torch.shape[3])),
            stride=(int(stride[0]), int(stride[1])),
            padding=(int(padding[0]), int(padding[1])),
            dilation=(int(dilation[0]), int(dilation[1])),
            groups=int(groups),
        )

    def __call__(self, x, *, device):
        _require_ttnn()
        x = ensure_tt_device_tensor(x, device)
        x_nhwc, batch_size, in_channels, input_h, input_w = _nchw_to_nhwc(x)

        if self._weight is None:
            self._weight = ttnn.from_torch(
                self.weight_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        if self.bias_torch is not None and self._bias is None:
            self._bias = ttnn.from_torch(
                self.bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

        out_nhwc, out_hw, weights_bias = ttnn.conv2d(
            input_tensor=x_nhwc,
            weight_tensor=self._weight,
            bias_tensor=self._bias,
            in_channels=in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_h,
            input_width=input_w,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            device=device,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )

        if isinstance(weights_bias, (tuple, list)) and len(weights_bias) == 2:
            self._weight, self._bias = weights_bias[0], weights_bias[1]

        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        return _nhwc_to_nchw(out_nhwc, batch_size, out_h, out_w, self.out_channels)


@dataclass
class TTConvTranspose2dCached:
    weight_torch: torch.Tensor
    bias_torch: Optional[torch.Tensor]
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    output_padding: Tuple[int, int]
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    mirror_kernel: bool = True

    _weight: object = None
    _bias: object = None

    @classmethod
    def from_conv_transpose(cls, conv_t: torch.nn.ConvTranspose2d) -> "TTConvTranspose2dCached":
        bias = conv_t.bias.detach() if conv_t.bias is not None else None
        return cls(
            weight_torch=conv_t.weight.detach(),
            bias_torch=bias,
            in_channels=int(conv_t.in_channels),
            out_channels=int(conv_t.out_channels),
            kernel_size=(int(conv_t.kernel_size[0]), int(conv_t.kernel_size[1])),
            stride=(int(conv_t.stride[0]), int(conv_t.stride[1])),
            padding=(int(conv_t.padding[0]), int(conv_t.padding[1])),
            output_padding=(int(conv_t.output_padding[0]), int(conv_t.output_padding[1])),
            dilation=(int(conv_t.dilation[0]), int(conv_t.dilation[1])),
            groups=int(conv_t.groups),
        )

    def __call__(self, x, *, device):
        _require_ttnn()
        x = ensure_tt_device_tensor(x, device)
        x_nhwc, batch_size, _in_channels, input_h, input_w = _nchw_to_nhwc(x)

        if self._weight is None:
            self._weight = ttnn.from_torch(
                self.weight_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        if self.bias_torch is not None and self._bias is None:
            self._bias = ttnn.from_torch(
                self.bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

        out_nhwc, out_hw, weights_bias = ttnn.conv_transpose2d(
            input_tensor=x_nhwc,
            weight_tensor=self._weight,
            bias_tensor=self._bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_h,
            input_width=input_w,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            device=device,
            mirror_kernel=self.mirror_kernel,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )

        if isinstance(weights_bias, (tuple, list)) and len(weights_bias) == 2:
            self._weight, self._bias = weights_bias[0], weights_bias[1]

        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        return _nhwc_to_nchw(out_nhwc, batch_size, out_h, out_w, self.out_channels)
