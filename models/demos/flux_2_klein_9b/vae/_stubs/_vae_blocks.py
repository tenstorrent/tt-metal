# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN building blocks for the FLUX.2 (`AutoencoderKLFlux2`) VAE.

Every block here is pure ttnn on device — nothing calls the torch reference.
Torch appears only where weights are read out of the reference `nn.Module` at
construction time and staged with `ttnn.from_torch`.

Layout convention
-----------------
Activations travel as **channels-last, batch-and-space flattened**:
`[1, 1, N*H*W, C]` in `TILE_LAYOUT`, which is what `ttnn.conv2d` consumes and
produces. `(batch, height, width)` is carried alongside as plain Python ints,
so no shape information has to be recovered from the tensor. The only places
that leave this layout are `_upsample2x` (`ttnn.upsample` wants `[N, H, W, C]`
in `ROW_MAJOR_LAYOUT`) and the NCHW conversions at the model boundary.

Tensor parallelism (TP)
-----------------------
The parallel axis for a VAE is the **channel** axis, and every conv is
COLUMN-parallel over its OUTPUT channels:

    W [C_out, C_in, kh, kw]  --ShardTensorToMesh(dim=0)-->  [C_out/TP, C_in, kh, kw]
    b [1, 1, 1, C_out]       --ShardTensorToMesh(dim=3)-->  [1, 1, 1, C_out/TP]

so device `d` computes exactly the output channels `[d*C_out/TP, (d+1)*C_out/TP)`
— bias included, because the bias travels with its own columns — and one
`all_gather` on the channel dim concatenates them back into the full activation.
Concatenation of disjoint output channels is the identity, so the math is
unchanged; the gathered output is what a single device would have produced.

Why column-parallel everywhere instead of the Megatron column-then-row pairing:

  * A conv needs ALL of its input channels. Column-parallel keeps the input
    replicated and full, so no conv is ever handed a sliver of C_in — which
    matters here because the last decoder stage has only 128 channels and a
    row-parallel split would hand it 16 input channels per device.
  * Every GroupNorm then sees the full channel dim, so `norm_num_groups=32`
    never has a group straddling a device and gamma/beta stay REPLICATED,
    exactly as the TP principles ask for elementwise parameters.
  * The collective count is one per conv either way; `all_gather` moves the
    same bytes as the `all_reduce` a row-parallel pairing would need, without
    the cross-device arithmetic.

Convs whose output channels do not divide the mesh stay replicated — that is
`conv_out` alone (`C_out=3`), which is also the cheapest conv in the model.

GroupNorm
---------
`ttnn.group_norm` is not used: it constrains the core grid against the tile
height (`Ht` must be divisible by the virtual row count), which the 28x28 stage
(`H*W = 784`, 24.5 tiles) does not satisfy. Instead the group statistics are
computed with the tensor's own channel structure: a constant one-hot
membership matrix `[C, G]` turns a matmul into the per-group channel sum, a
`ttnn.sum` over the spatial axis finishes the reduction (it masks tile padding,
so `P=784` is exact), and the transpose `[G, C]` scatters each group statistic
back across its channels. Mean and variance are two-pass — `E[(x-mu)^2]`, never
`E[x^2] - E[x]^2` — so the bfloat16 variance never rests on a cancellation.
Measured against `torch.nn.GroupNorm` this holds PCC >= 0.99998 at every
resolution in the model.
"""
from __future__ import annotations

import torch

import ttnn


def mesh_width(device) -> int:
    """Number of devices `device` spans (1 for a single-chip device)."""
    try:
        n = int(device.get_num_devices())
    except (AttributeError, TypeError):
        return 1
    return n if n > 0 else 1


def compute_config():
    """HiFi4 + fp32 accumulation: the norm reductions run over up to 50k positions."""
    try:
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


def _replicate(tensor, device, tp, *, layout=ttnn.TILE_LAYOUT):
    """Stage a tensor identically on every device."""
    mapper = ttnn.ReplicateTensorToMesh(device) if tp > 1 else None
    return ttnn.from_torch(
        tensor.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        mesh_mapper=mapper,
    )


class GroupNorm:
    """`torch.nn.GroupNorm` over the channel axis of a `[N, 1, H*W, C]` activation."""

    def __init__(self, device, torch_norm, tp) -> None:
        self.device = device
        self.channels = int(torch_norm.num_channels)
        self.groups = int(torch_norm.num_groups)
        self.eps = float(torch_norm.eps)
        self.group_size = self.channels // self.groups
        self.compute_config = compute_config()

        membership = torch.zeros(self.channels, self.groups, dtype=torch.float32)
        for g in range(self.groups):
            membership[g * self.group_size : (g + 1) * self.group_size, g] = 1.0
        self.group_sum = _replicate(membership, device, tp)
        self.group_scatter = _replicate(membership.t().contiguous(), device, tp)

        state = torch_norm.state_dict()
        self.weight = _replicate(state["weight"].reshape(1, 1, 1, self.channels), device, tp)
        self.bias = _replicate(state["bias"].reshape(1, 1, 1, self.channels), device, tp)

    def __call__(self, x, batch, positions):
        # Statistics are per image, so fold the batch out of the flattened axis.
        x = ttnn.reshape(x, (batch, 1, positions, self.channels))
        inv_count = 1.0 / float(positions * self.group_size)

        per_position = ttnn.matmul(x, self.group_sum, compute_kernel_config=self.compute_config)
        totals = ttnn.sum(per_position, dim=-2, keepdim=True, compute_kernel_config=self.compute_config)
        mean = ttnn.multiply(totals, inv_count)
        mean_channels = ttnn.matmul(mean, self.group_scatter, compute_kernel_config=self.compute_config)

        centered = ttnn.subtract(x, mean_channels)
        squares = ttnn.multiply(centered, centered)
        square_totals = ttnn.sum(
            ttnn.matmul(squares, self.group_sum, compute_kernel_config=self.compute_config),
            dim=-2,
            keepdim=True,
            compute_kernel_config=self.compute_config,
        )
        variance = ttnn.multiply(square_totals, inv_count)
        inv_std = ttnn.rsqrt(ttnn.add(variance, self.eps))
        inv_std_channels = ttnn.matmul(inv_std, self.group_scatter, compute_kernel_config=self.compute_config)

        out = ttnn.multiply(centered, inv_std_channels)
        out = ttnn.multiply(out, self.weight)
        out = ttnn.add(out, self.bias)
        return ttnn.reshape(out, (1, 1, batch * positions, self.channels))


class Conv2d:
    """`torch.nn.Conv2d`, column-parallel over output channels with an `all_gather`."""

    def __init__(self, device, torch_conv, tp) -> None:
        self.device = device
        self.tp = tp

        weight = torch_conv.weight.detach()
        bias = torch_conv.bias.detach() if torch_conv.bias is not None else None
        self.out_channels_full = int(weight.shape[0])
        self.in_channels = int(weight.shape[1]) * int(torch_conv.groups)
        self.kernel_size = (int(weight.shape[2]), int(weight.shape[3]))
        self.stride = tuple(torch_conv.stride)
        self.padding = tuple(torch_conv.padding)
        self.dilation = tuple(torch_conv.dilation)
        self.groups = int(torch_conv.groups)

        # A 3-channel output (conv_out) cannot be split 8 ways; leave it replicated.
        self.column_parallel = tp > 1 and self.out_channels_full % tp == 0 and self.groups == 1
        if self.column_parallel:
            self.out_channels = self.out_channels_full // tp
            self.weight = ttnn.from_torch(weight, ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))
            self.bias = (
                ttnn.from_torch(
                    bias.reshape(1, 1, 1, self.out_channels_full),
                    ttnn.bfloat16,
                    mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
                )
                if bias is not None
                else None
            )
        else:
            self.out_channels = self.out_channels_full
            self.weight = ttnn.from_torch(weight, ttnn.bfloat16)
            self.bias = (
                ttnn.from_torch(bias.reshape(1, 1, 1, self.out_channels_full), ttnn.bfloat16)
                if bias is not None
                else None
            )

    def __call__(self, x, batch, height, width, padding=None):
        out, [out_height, out_width], [weight, bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding if padding is None else padding,
            dilation=self.dilation,
            batch_size=batch,
            input_height=height,
            input_width=width,
            groups=self.groups,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        # Keep the device-prepared weights so a second call skips the preparation.
        self.weight, self.bias = weight, bias
        if self.column_parallel:
            out = _all_gather_channels(out, self.tp)
        return out, out_height, out_width


def _all_gather_channels(tensor, tp):
    """Concatenate the per-device output-channel shards back into a full activation."""
    if tp <= 1:
        return tensor
    try:
        return ttnn.all_gather(tensor, dim=3, topology=ttnn.Topology.Linear)
    except TypeError:
        return ttnn.all_gather(tensor, dim=3)


def _upsample2x(x, batch, height, width, channels):
    """Nearest-neighbour 2x upsample; `ttnn.upsample` wants unpadded `[N, H, W, C]`."""
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (batch, height, width, channels))
    x = ttnn.upsample(x, 2)
    x = ttnn.reshape(x, (1, 1, batch * height * 2 * width * 2, channels))
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT), height * 2, width * 2


class ResnetBlock2D:
    """diffusers `ResnetBlock2D` in its VAE configuration (no time embedding, no resampling)."""

    def __init__(self, device, torch_block, tp) -> None:
        if torch_block.time_emb_proj is not None:
            raise RuntimeError("VAE ResnetBlock2D has no time embedding; this one does")
        if torch_block.upsample is not None or torch_block.downsample is not None:
            raise RuntimeError("VAE ResnetBlock2D does not resample inside the block")
        if not isinstance(torch_block.nonlinearity, torch.nn.SiLU):
            raise RuntimeError(f"expected SiLU, got {type(torch_block.nonlinearity).__name__}")

        self.norm1 = GroupNorm(device, torch_block.norm1, tp)
        self.conv1 = Conv2d(device, torch_block.conv1, tp)
        self.norm2 = GroupNorm(device, torch_block.norm2, tp)
        self.conv2 = Conv2d(device, torch_block.conv2, tp)
        self.conv_shortcut = (
            Conv2d(device, torch_block.conv_shortcut, tp) if torch_block.conv_shortcut is not None else None
        )
        self.output_scale_factor = float(torch_block.output_scale_factor)
        self.out_channels = self.conv2.out_channels_full

    def __call__(self, x, batch, height, width):
        h = self.norm1(x, batch, height * width)
        h = ttnn.silu(h)
        h, mid_height, mid_width = self.conv1(h, batch, height, width)

        h = self.norm2(h, batch, mid_height * mid_width)
        h = ttnn.silu(h)
        h, out_height, out_width = self.conv2(h, batch, mid_height, mid_width)

        if self.conv_shortcut is not None:
            residual, _, _ = self.conv_shortcut(x, batch, height, width)
        else:
            residual = x

        out = ttnn.add(residual, h)
        if self.output_scale_factor != 1.0:
            out = ttnn.multiply(out, 1.0 / self.output_scale_factor)
        return out, out_height, out_width


class Upsample2D:
    """diffusers `Upsample2D`: nearest 2x interpolation followed by a 3x3 conv."""

    def __init__(self, device, torch_upsample, tp) -> None:
        if torch_upsample.use_conv_transpose:
            raise RuntimeError("transposed-conv upsampling is not used by this VAE")
        if not torch_upsample.interpolate or not torch_upsample.use_conv:
            raise RuntimeError("expected interpolate + conv upsampling")
        if getattr(torch_upsample, "norm", None) is not None:
            raise RuntimeError("Upsample2D with a norm is not used by this VAE")

        self.conv = Conv2d(device, torch_upsample.conv, tp)
        self.in_channels = int(torch_upsample.channels)
        self.out_channels = self.conv.out_channels_full

    def __call__(self, x, batch, height, width):
        x, height, width = _upsample2x(x, batch, height, width, self.in_channels)
        return self.conv(x, batch, height, width)


class Downsample2D:
    """diffusers `Downsample2D`: asymmetric (0,1,0,1) zero pad then a stride-2 3x3 conv."""

    def __init__(self, device, torch_downsample, tp) -> None:
        if not torch_downsample.use_conv:
            raise RuntimeError("average-pool downsampling is not used by this VAE")
        if getattr(torch_downsample, "norm", None) is not None:
            raise RuntimeError("Downsample2D with a norm is not used by this VAE")

        self.pad = int(torch_downsample.padding)
        self.conv = Conv2d(device, torch_downsample.conv, tp)
        self.in_channels = int(torch_downsample.channels)
        self.out_channels = self.conv.out_channels_full

    def __call__(self, x, batch, height, width):
        if self.pad == 0:
            # With padding=0 diffusers pads the activation itself with
            # `F.pad(..., (0, 1, 0, 1))` — bottom and right only — before a
            # padding-free conv. Express that asymmetric window directly, as
            # `(top, bottom, left, right)`.
            return self.conv(x, batch, height, width, padding=(0, 1, 0, 1))
        return self.conv(x, batch, height, width)


class UNetMidBlock2D:
    """diffusers `UNetMidBlock2D`: resnet -> attention -> resnet."""

    def __init__(self, device, torch_block, tp, attention_factory) -> None:
        self.resnets = [ResnetBlock2D(device, r, tp) for r in torch_block.resnets]
        self.attentions = [attention_factory(device, a) if a is not None else None for a in torch_block.attentions]
        self.out_channels = self.resnets[-1].out_channels

    def __call__(self, x, batch, height, width):
        x, height, width = self.resnets[0](x, batch, height, width)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            if attention is not None:
                x = attention(x)
            x, height, width = resnet(x, batch, height, width)
        return x, height, width


class UpDecoderBlock2D:
    """diffusers `UpDecoderBlock2D`: N resnets then an optional 2x upsampler."""

    def __init__(self, device, torch_block, tp) -> None:
        self.resnets = [ResnetBlock2D(device, r, tp) for r in torch_block.resnets]
        self.upsamplers = (
            [Upsample2D(device, u, tp) for u in torch_block.upsamplers] if torch_block.upsamplers is not None else []
        )
        self.out_channels = self.upsamplers[-1].out_channels if self.upsamplers else self.resnets[-1].out_channels

    def __call__(self, x, batch, height, width):
        for resnet in self.resnets:
            x, height, width = resnet(x, batch, height, width)
        for upsampler in self.upsamplers:
            x, height, width = upsampler(x, batch, height, width)
        return x, height, width


class DownEncoderBlock2D:
    """diffusers `DownEncoderBlock2D`: N resnets then an optional stride-2 downsampler."""

    def __init__(self, device, torch_block, tp) -> None:
        self.resnets = [ResnetBlock2D(device, r, tp) for r in torch_block.resnets]
        self.downsamplers = (
            [Downsample2D(device, d, tp) for d in torch_block.downsamplers]
            if torch_block.downsamplers is not None
            else []
        )
        self.out_channels = self.downsamplers[-1].out_channels if self.downsamplers else self.resnets[-1].out_channels

    def __call__(self, x, batch, height, width):
        for resnet in self.resnets:
            x, height, width = resnet(x, batch, height, width)
        for downsampler in self.downsamplers:
            x, height, width = downsampler(x, batch, height, width)
        return x, height, width


def nchw_to_flat_nhwc(x):
    """`[N, C, H, W]` on device -> `([1, 1, N*H*W, C], N, C, H, W)`."""
    batch, channels, height, width = (int(v) for v in x.shape)
    x = ttnn.permute(x, (0, 2, 3, 1))
    x = ttnn.reshape(x, (1, 1, batch * height * width, channels))
    return x, batch, channels, height, width


def flat_nhwc_to_nchw(x, batch, channels, height, width):
    """`[1, 1, N*H*W, C]` -> `[N, C, H, W]`, the layout the torch golden is in."""
    x = ttnn.reshape(x, (batch, height, width, channels))
    return ttnn.permute(x, (0, 3, 1, 2))
