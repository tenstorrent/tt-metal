# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Self-contained convolution primitives for the XTTS-v2 HiFi-GAN decoder.

The GAN decoder is a deep chain of ``Conv1d`` + ``ConvTranspose1d`` layers (the
vocoder) plus a ``Conv2d`` SE-ResNet (the speaker encoder). ``ttnn`` has native
``conv1d``/``conv2d`` (with ``dilation``/``groups``) but no ``conv_transpose1d``,
so these primitives live here:

* :class:`TtConv1d`          — thin wrapper over ``ttnn.conv1d``.
* :class:`TtConvTranspose1d` — transpose conv expressed as a regular conv on the
  zero-stuffed input with a flipped, in/out-transposed kernel.
* :class:`TtConv2d`          — thin wrapper over ``ttnn.conv2d``.

Tensor convention: **channels-last** — ``[N, L, C]`` for 1D, ``[N, H, W, C]`` for
2D (ROW_MAJOR, on device), the layouts ttnn's convs consume — avoiding per-layer
transposes. Weights are PyTorch tensors (``Conv1d``: ``[out, in/groups, k]``;
``ConvTranspose1d``: ``[in, out, k]``; ``Conv2d``: ``[out, in/groups, kh, kw]``).

Defaults: **fp32 activations**, bf16 weights, HiFi4, ``fp32_dest_acc_en``. bf16
activations lose too much through this deep a chain (the ~36 residual adds + MRF
sums + tanh compound, and PCC drifts below 0.99 as the sequence lengthens); fp32
activations hold PCC ~0.999 at length. fp32 activations no longer OOM the wide
layers because conv1d auto-width-slices DRAM inputs — but they need a larger
``l1_small_size`` (32768) on the device. bf16 weights are kept: fp32 weights gave
no accuracy gain. Pass ``activations_dtype=ttnn.bfloat16`` for a faster, lower-
accuracy mode.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


def _interleaved_row_major(x: ttnn.Tensor, shape) -> ttnn.Tensor:
    """Bring a (possibly sharded/tiled) conv output back to interleaved ROW_MAJOR
    and reshape to ``shape`` so downstream element-wise ops and the next conv can
    consume it as ``[N, L, C]``."""
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(x, shape)


class TtConv1d(LightweightModule):
    """1D convolution over a channels-last ``[N, L, C]`` device tensor.

    ``padding`` follows PyTorch semantics (symmetric); for a "same"-length dilated
    conv pass ``padding = dilation * (kernel_size - 1) // 2``.
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [out_channels, in_channels // groups, kernel_size]
        bias: torch.Tensor | None = None,  # [out_channels]
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        activations_dtype: ttnn.DataType = ttnn.float32,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en: bool = True,
        packer_l1_acc: bool = True,
    ):
        super().__init__()
        assert weight.dim() == 3, f"expected Conv1d weight [out, in/groups, k], got {tuple(weight.shape)}"
        out_channels, in_per_group, kernel_size = weight.shape

        self.device = device
        self.in_channels = in_per_group * groups
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activations_dtype = activations_dtype

        # ttnn.conv1d takes the raw PyTorch weight layout and preprocesses it on
        # first call; we cache the preprocessed device weight for reuse.
        self.tt_weight = ttnn.from_torch(weight.float(), weights_dtype)
        self.tt_bias = None
        if bias is not None:
            self.tt_bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1).float(), weights_dtype)

        # No forced shard_layout: HEIGHT_SHARDED fails the DRAM slicer on the wide
        # (1024-channel) layers with short spatial extent. Auto-sharding picks a
        # valid layout per shape and gives PCC ~0.9999.
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            deallocate_activation=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, input_length, _ = x.shape
        out, out_length, [weight, bias] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.tt_weight,
            bias_tensor=self.tt_bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            batch_size=batch_size,
            input_length=input_length,
            dtype=self.activations_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.tt_weight = weight
        self.tt_bias = bias
        return _interleaved_row_major(out, [batch_size, out_length, self.out_channels])


class TtConvTranspose1d(LightweightModule):
    """``torch.nn.ConvTranspose1d`` with ``padding = (kernel_size - stride) // 2``
    (the HiFi-GAN upsampling convention, giving an exact ``stride``x upsample).

    Implemented as a plain :class:`TtConv1d` on the zero-stuffed input:
      * insert ``stride - 1`` zeros between input samples,
      * symmetric external zero-pad of ``k - 1 - (k - stride) // 2`` on each side,
      * convolve with the kernel flipped along ``k`` and transposed to ``[out, in, k]``.
    The flip is because ``Conv1d`` is cross-correlation while the zero-stuff
    equivalent of a transpose conv needs a flipped kernel. Requires ``k - stride``
    even (true for all XTTS upsample layers: k/stride = 16/8, 4/2).
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [in_channels, out_channels, kernel_size]
        bias: torch.Tensor | None = None,  # [out_channels]
        *,
        stride: int,
        **conv_kwargs,
    ):
        super().__init__()
        assert weight.dim() == 3, f"expected ConvTranspose1d weight [in, out, k], got {tuple(weight.shape)}"
        in_channels, out_channels, kernel_size = weight.shape
        assert (kernel_size - stride) % 2 == 0, f"need (k - stride) even, got k={kernel_size}, stride={stride}"

        self.stride = stride
        padding = (kernel_size - stride) // 2
        self.external_pad = kernel_size - 1 - padding

        # Conv1d cross-correlation on zero-stuffed input == ConvTranspose1d.
        weight_conv = torch.flip(weight, dims=[-1]).permute(1, 0, 2).contiguous()  # [out, in, k]
        self.conv = TtConv1d(device, weight_conv, bias, stride=1, padding=0, **conv_kwargs)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, input_length, channels = x.shape

        # Zero-stuff: [N, L, C] -> [N, L, 1, C] -> pad to [N, L, stride, C] ->
        # [N, L*stride, C], then drop the trailing (stride-1) zeros.
        x = ttnn.reshape(x, [batch_size, input_length, 1, channels])
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, self.stride - 1), (0, 0)], value=0.0)
        x = ttnn.reshape(x, [batch_size, input_length * self.stride, channels])
        stuffed_length = (input_length - 1) * self.stride + 1
        x = ttnn.slice(x, [0, 0, 0], [batch_size, stuffed_length, channels])

        if self.external_pad > 0:
            x = ttnn.pad(x, [(0, 0), (self.external_pad, self.external_pad), (0, 0)], value=0.0)

        return self.conv(x)


class TtConv2d(LightweightModule):
    """2D convolution over a channels-last ``[N, H, W, C]`` device tensor.

    ``stride``/``padding`` follow PyTorch semantics (symmetric). Used by the
    speaker-encoder SE-ResNet (all 3x3 / 1x1 convs).
    """

    def __init__(
        self,
        device,
        weight: torch.Tensor,  # [out_channels, in_channels // groups, kh, kw]
        bias: torch.Tensor | None = None,  # [out_channels]
        *,
        stride: int = 1,
        padding: int = 1,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        activations_dtype: ttnn.DataType = ttnn.float32,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en: bool = True,
        packer_l1_acc: bool = True,
    ):
        super().__init__()
        assert weight.dim() == 4, f"expected Conv2d weight [out, in, kh, kw], got {tuple(weight.shape)}"
        out_channels, in_channels, kh, kw = weight.shape

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.activations_dtype = activations_dtype

        self.tt_weight = ttnn.from_torch(weight.float(), weights_dtype)
        self.tt_bias = None
        if bias is not None:
            self.tt_bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1).float(), weights_dtype)

        self.conv_config = ttnn.Conv2dConfig(weights_dtype=weights_dtype, deallocate_activation=False)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, height, width, _ = x.shape
        out, (out_h, out_w), [weight, bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_weight,
            bias_tensor=self.tt_bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            dtype=self.activations_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.tt_weight = weight
        self.tt_bias = bias
        return _interleaved_row_major(out, [batch_size, out_h, out_w, self.out_channels])
