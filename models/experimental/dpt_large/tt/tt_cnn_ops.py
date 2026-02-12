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


def _ensure_interleaved(x, *, target_memory_config=None):
    _require_ttnn()
    if target_memory_config is None:
        target_memory_config = ttnn.DRAM_MEMORY_CONFIG
    try:
        mc = ttnn.get_memory_config(x)
    except Exception:
        mc = None
    if mc is not None:
        try:
            if mc.is_sharded():
                return ttnn.to_memory_config(x, target_memory_config)
        except Exception:
            pass
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
    # Keep ReLU out-of-place so pre-activation residual paths preserve the
    # original skip tensor semantics.
    return ttnn.relu(x)


def _nchw_to_nhwc(x):
    _require_ttnn()
    x = _to_row_major(x)
    b, c, h, w = _shape4(x)
    x_nhwc = ttnn.permute(x, (0, 2, 3, 1))
    return x_nhwc, b, c, h, w


def _nhwc_to_nchw(x_nhwc, b: int, h: int, w: int, c: int):
    _require_ttnn()
    x_nhwc = _to_row_major(x_nhwc)
    x_nhwc = _ensure_interleaved(x_nhwc)
    s_b, s1, s2, s3 = _shape4(x_nhwc)
    if s_b != int(b):
        raise RuntimeError(
            f"Unexpected TT NHWC batch dimension: got {tuple(x_nhwc.shape)} expected batch={int(b)}"
        )

    exp_h, exp_w, exp_c = int(h), int(w), int(c)
    if (s1, s2, s3) == (exp_h, exp_w, exp_c):
        pass
    elif (s1, s2, s3) == (1, exp_h * exp_w, exp_c):
        x_nhwc = ttnn.reshape(x_nhwc, (int(b), exp_h, exp_w, exp_c))
    elif (s1, s2, s3) == (exp_h * exp_w, 1, exp_c):
        x_nhwc = ttnn.reshape(x_nhwc, (int(b), exp_h, exp_w, exp_c))
    else:
        raise RuntimeError(
            f"Unexpected TT NHWC logical shape: got {tuple(x_nhwc.shape)} expected {(int(b), exp_h, exp_w, exp_c)}"
        )
    return ttnn.permute(x_nhwc, (0, 3, 1, 2))


def tt_canonicalize_nchw_spatial(
    x,
    *,
    expected_hw: Optional[Tuple[int, int]] = None,
    op_name: str = "tt_canonicalize_nchw_spatial",
):
    _require_ttnn()
    x = _to_row_major(x)
    x = _ensure_interleaved(x)
    b, c, h, w = _shape4(x)
    if expected_hw is None:
        return x

    exp_h, exp_w = int(expected_hw[0]), int(expected_hw[1])
    if exp_h <= 0 or exp_w <= 0:
        raise RuntimeError(f"{op_name}: invalid expected HxW={exp_h}x{exp_w}")
    if h == exp_h and w == exp_w:
        return x

    flat_hw = exp_h * exp_w
    if h == 1 and w == flat_hw:
        return ttnn.reshape(x, (b, c, exp_h, exp_w))
    if w == 1 and h == flat_hw:
        return ttnn.reshape(x, (b, c, exp_h, exp_w))

    raise RuntimeError(
        f"{op_name}: unexpected spatial shape={tuple(x.shape)} expected HxW={exp_h}x{exp_w}"
    )


def tt_upsample_nchw(
    x,
    *,
    scale_factor: int | Sequence[int] = 2,
    mode: str = "bilinear",
    align_corners: Optional[bool] = None,
    memory_config=None,
    expected_input_hw: Optional[Tuple[int, int]] = None,
    op_name: str = "tt_upsample_nchw",
):
    _require_ttnn()
    x = tt_canonicalize_nchw_spatial(
        x,
        expected_hw=expected_input_hw,
        op_name=f"{op_name}.input",
    )
    b, c, h, w = _shape4(x)
    x_nhwc = ttnn.permute(x, (0, 2, 3, 1))

    sf = scale_factor
    if isinstance(sf, (tuple, list)):
        sf = [int(sf[0]), int(sf[1])]
    else:
        sf = int(sf)
    sf_h = sf[0] if isinstance(sf, list) else sf
    sf_w = sf[1] if isinstance(sf, list) else sf
    out_h = h * sf_h
    out_w = w * sf_w

    # TTNN bilinear interpolation follows align_corners=False semantics. For
    # DPT paths that require align_corners=True, execute an exact host
    # interpolation and convert back to TT tensor to preserve numerical parity.
    if mode == "bilinear" and align_corners is True:
        x_torch = ttnn.to_torch(x).to(dtype=torch.float32)
        y_torch = torch.nn.functional.interpolate(
            x_torch,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=True,
        )
        tt_device = None
        try:
            tt_device = x.device()
        except Exception:
            tt_device = None
        return ttnn.from_torch(
            y_torch.to(dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
        )

    if memory_config is None:
        # Keep upsample outputs interleaved by default to avoid width-sharded
        # transpose constraints and large L1 halo allocations on N300.
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    try:
        y_nhwc = ttnn.upsample(
            input_tensor=x_nhwc,
            scale_factor=sf,
            mode=mode,
            memory_config=memory_config,
        )
    except RuntimeError as exc:
        # Bilinear upsample can require large L1 halo buffers on N300.
        # Fall back to nearest on TT to keep the practical hot path alive.
        if mode == "bilinear" and "Out of Memory" in str(exc):
            y_nhwc = ttnn.upsample(
                input_tensor=x_nhwc,
                scale_factor=sf,
                mode="nearest",
                memory_config=memory_config,
            )
        else:
            raise
    # Some TTNN upsample configurations return sharded outputs; permute/transpose
    # can be invalid for those shard specs. Ensure interleaved before NCHW permute.
    y_nhwc = _ensure_interleaved(y_nhwc)

    y_nchw = ttnn.permute(y_nhwc, (0, 3, 1, 2))
    y_nchw = tt_canonicalize_nchw_spatial(
        y_nchw,
        expected_hw=(out_h, out_w),
        op_name=f"{op_name}.output",
    )
    _shape4(y_nchw)
    if int(y_nchw.shape[-2]) != out_h or int(y_nchw.shape[-1]) != out_w:
        # Keep shape checks explicit for easier debug when runtime APIs change.
        raise RuntimeError(
            f"{op_name}: unexpected TT upsample output shape: got {tuple(y_nchw.shape)} expected HxW={out_h}x{out_w}"
        )
    return y_nchw


def tt_resize_to_nchw(
    x,
    *,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
    align_corners: Optional[bool] = None,
    memory_config=None,
    op_name: str = "tt_resize_to_nchw",
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
        align_corners=align_corners,
        memory_config=memory_config,
        expected_input_hw=(h, w),
        op_name=op_name,
    )


def tt_depth_to_space_nchw(
    x,
    block_size: int,
    *,
    expected_output_hw: Optional[Tuple[int, int]] = None,
    op_name: str = "tt_depth_to_space_nchw",
):
    _require_ttnn()
    scale = int(block_size)
    if expected_output_hw is not None:
        exp_h, exp_w = int(expected_output_hw[0]), int(expected_output_hw[1])
        if exp_h % scale != 0 or exp_w % scale != 0:
            raise RuntimeError(
                f"{op_name}: expected output HxW={exp_h}x{exp_w} not divisible by block_size={scale}"
            )
        x = tt_canonicalize_nchw_spatial(
            x,
            expected_hw=(exp_h // scale, exp_w // scale),
            op_name=f"{op_name}.input",
        )
    else:
        x = _to_row_major(x)
        x = _ensure_interleaved(x)

    b, c_mul, h, w = _shape4(x)
    scale_sq = scale * scale
    if c_mul % scale_sq != 0:
        raise RuntimeError(
            f"{op_name}: channel mismatch: channels={c_mul}, block_size={scale}, expected divisible by {scale_sq}"
        )

    c = c_mul // scale_sq
    x = ttnn.reshape(x, (b, c, scale, scale, h, w))
    x = ttnn.permute(x, (0, 1, 4, 2, 5, 3))
    y = ttnn.reshape(x, (b, c, h * scale, w * scale))
    if expected_output_hw is not None:
        y = tt_canonicalize_nchw_spatial(
            y,
            expected_hw=(int(expected_output_hw[0]), int(expected_output_hw[1])),
            op_name=f"{op_name}.output",
        )
    return y


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
            )
        if self.bias_torch is not None and self._bias is None:
            self._bias = ttnn.from_torch(
                self.bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
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
    _tt_pointwise_conv: "TTConv2dCached | None" = None

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

    def _can_use_pointwise_depth_to_space(self) -> bool:
        return (
            self.groups == 1
            and self.dilation == (1, 1)
            and self.padding == (0, 0)
            and self.output_padding == (0, 0)
            and self.kernel_size == self.stride
            and self.kernel_size[0] == self.kernel_size[1]
            and self.kernel_size[0] > 1
        )

    def _build_pointwise_equivalent(self):
        scale = int(self.kernel_size[0])
        out_channels = int(self.out_channels)
        in_channels = int(self.in_channels)

        # ConvTranspose2d (stride=kernel, no overlap) is equivalent to a 1x1
        # conv producing C_out * scale^2 channels followed by depth-to-space.
        # weight_torch layout: [C_in, C_out, K, K]
        repacked_weight = (
            self.weight_torch.permute(1, 2, 3, 0)
            .contiguous()
            .reshape(out_channels * scale * scale, in_channels, 1, 1)
            .contiguous()
        )
        repacked_bias = None
        if self.bias_torch is not None:
            repacked_bias = self.bias_torch.repeat_interleave(scale * scale).contiguous()

        self._tt_pointwise_conv = TTConv2dCached.from_tensors(
            weight_torch=repacked_weight,
            bias_torch=repacked_bias,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )
        return scale

    def __call__(
        self,
        x,
        *,
        device,
        expected_input_hw: Optional[Tuple[int, int]] = None,
        expected_output_hw: Optional[Tuple[int, int]] = None,
    ):
        _require_ttnn()
        x = ensure_tt_device_tensor(x, device)

        if self._can_use_pointwise_depth_to_space():
            scale = int(self.kernel_size[0])
            if self._tt_pointwise_conv is None:
                scale = self._build_pointwise_equivalent()
            x = tt_canonicalize_nchw_spatial(
                x,
                expected_hw=expected_input_hw,
                op_name="tt_conv_transpose2d_fast_path.input",
            )
            out = self._tt_pointwise_conv(x, device=device)
            return tt_depth_to_space_nchw(
                out,
                scale,
                expected_output_hw=expected_output_hw,
                op_name="tt_conv_transpose2d_fast_path.depth_to_space",
            )

        x = tt_canonicalize_nchw_spatial(
            x,
            expected_hw=expected_input_hw,
            op_name="tt_conv_transpose2d.input",
        )
        x_nhwc, batch_size, _in_channels, input_h, input_w = _nchw_to_nhwc(x)

        if self._weight is None:
            self._weight = ttnn.from_torch(
                self.weight_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        if self.bias_torch is not None and self._bias is None:
            self._bias = ttnn.from_torch(
                self.bias_torch.reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        out_nhwc, out_hw = ttnn.conv_transpose2d(
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
            return_weights_and_bias=False,
            dtype=ttnn.bfloat16,
        )

        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        y = _nhwc_to_nchw(out_nhwc, batch_size, out_h, out_w, self.out_channels)
        if expected_output_hw is not None:
            y = tt_canonicalize_nchw_spatial(
                y,
                expected_hw=expected_output_hw,
                op_name="tt_conv_transpose2d.output",
            )
        return y
