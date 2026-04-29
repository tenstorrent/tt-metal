# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the AudioX Oobleck VAE decoder.

Tensor convention here is NHWC with H=T, W=1 (``[B, T, 1, C]``), matching
``ttnn.conv1d``'s expected layout. SnakeBeta runs on TILE_LAYOUT (pointwise),
conv1d / conv_transpose1d run on ROW_MAJOR_LAYOUT; we convert at boundaries.

TTNN has no native ``conv_transpose1d``, so the decoder upsample is emulated
with ``conv_transpose2d`` over a degenerate width-1 dim — mathematically
identical to the 1D op, kernel ``(2*stride, 1)`` and stride ``(stride, 1)``."""

import torch
import ttnn

from models.experimental.audiox.tt.common import to_tt


def reconstruct_wn_weight(state_dict: dict, prefix: str) -> torch.Tensor:
    """Rebuild a weight_norm-parameterized Conv1d weight from g and v.

    weight_norm with default ``dim=0`` stores ``weight_g`` (shape
    ``[out, 1, 1]``) and ``weight_v`` (full weight shape) and computes
    ``g * v / ||v||`` where the norm reduces over all dims except 0.
    Inference doesn't need the parameterization, so we collapse it here."""
    g = state_dict[f"{prefix}weight_g"]
    v = state_dict[f"{prefix}weight_v"]
    norm = v.norm(dim=tuple(range(1, v.ndim)), keepdim=True)
    return g * v / norm


class TtSnakeBeta:
    """Per-channel ``x + (1/beta) * sin(alpha*x)^2``. Alpha and beta are
    stored in log-space upstream; we pre-exp them at init so forward is
    just sin -> mul -> mul -> add."""

    def __init__(self, mesh_device, state_dict: dict, prefix: str = ""):
        eps = 1e-9
        # Reshape per-channel params [C] -> [1, 1, 1, C] for broadcast over
        # [B, T, 1, C].
        alpha = torch.exp(state_dict[f"{prefix}alpha"]).reshape(1, 1, 1, -1)
        beta = torch.exp(state_dict[f"{prefix}beta"]).reshape(1, 1, 1, -1)
        inv_beta = 1.0 / (beta + eps)
        self.alpha = to_tt(alpha, mesh_device)
        self.inv_beta = to_tt(inv_beta, mesh_device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        s = ttnn.sin(ttnn.multiply(x, self.alpha))
        s = ttnn.multiply(s, s)
        return ttnn.add(x, ttnn.multiply(s, self.inv_beta))


def _conv1d(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    device,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int,
    padding: int,
    batch_size: int,
    input_length: int,
):
    """Wrapper around ``ttnn.conv1d`` that returns an interleaved TILE
    tensor of shape ``[B, L_out, 1, C_out]`` so the next pointwise op can
    consume it directly."""
    out, out_length = ttnn.conv1d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_length=input_length,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=dilation,
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=False,
    )
    out = ttnn.sharded_to_interleaved(out)
    out = ttnn.reshape(out, (batch_size, out_length, 1, out_channels))
    return ttnn.to_layout(out, ttnn.TILE_LAYOUT)


class TtResidualUnit:
    """Snake -> dilated 7-tap Conv1d -> Snake -> 1-tap Conv1d, plus residual.
    Length is preserved by the upstream padding choice ``dilation*3``."""

    def __init__(self, mesh_device, state_dict: dict, channels: int, dilation: int, prefix: str = ""):
        sd = state_dict
        self.mesh_device = mesh_device
        self.channels = channels
        self.dilation = dilation
        self.padding = dilation * 3  # = (dilation * (7 - 1)) // 2

        self.act1 = TtSnakeBeta(mesh_device, sd, prefix + "act1.")
        self.act2 = TtSnakeBeta(mesh_device, sd, prefix + "act2.")

        # Conv weights/biases live on host until conv1d ships them to device
        # on first call. ROW_MAJOR + float32 matches the common pattern in
        # tt-metal for conv1d ingest.
        w1 = reconstruct_wn_weight(sd, prefix + "conv1.")
        b1 = sd[prefix + "conv1.bias"]
        w2 = reconstruct_wn_weight(sd, prefix + "conv2.")
        b2 = sd[prefix + "conv2.bias"]

        self.w1 = ttnn.from_torch(w1, dtype=ttnn.float32)
        self.b1 = ttnn.from_torch(b1.reshape(1, 1, 1, -1), dtype=ttnn.float32)
        self.w2 = ttnn.from_torch(w2, dtype=ttnn.float32)
        self.b2 = ttnn.from_torch(b2.reshape(1, 1, 1, -1), dtype=ttnn.float32)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: [B, T, 1, C] TILE.
        batch_size, input_length = x.shape[0], x.shape[1]

        h = self.act1(x)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = _conv1d(
            h,
            self.w1,
            self.b1,
            self.mesh_device,
            self.channels,
            self.channels,
            kernel_size=7,
            dilation=self.dilation,
            padding=self.padding,
            batch_size=batch_size,
            input_length=input_length,
        )

        h = self.act2(h)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = _conv1d(
            h,
            self.w2,
            self.b2,
            self.mesh_device,
            self.channels,
            self.channels,
            kernel_size=1,
            dilation=1,
            padding=0,
            batch_size=batch_size,
            input_length=input_length,
        )

        return ttnn.add(x, h)


def _conv_transpose1d(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    device,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    batch_size: int,
    input_length: int,
):
    """Emulate ConvTranspose1d via ``ttnn.conv_transpose2d`` over ``[B, T, 1, C]``.

    Kernel/stride/padding act on the H (time) dim with W=1; the W dim is
    a no-op (``kW=1``, ``sW=1``, ``pW=0``). Input is RM, output is converted
    back to interleaved TILE for the next pointwise op."""
    out_length = (input_length - 1) * stride - 2 * padding + kernel_size
    # No return flags -> single-tensor return. We compute out_length ourselves
    # from the standard ConvTranspose1d formula.
    out = ttnn.conv_transpose2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
        padding=(padding, 0),
        batch_size=batch_size,
        input_height=input_length,
        input_width=1,
    )
    out = ttnn.sharded_to_interleaved(out)
    out = ttnn.reshape(out, (batch_size, out_length, 1, out_channels))
    return ttnn.to_layout(out, ttnn.TILE_LAYOUT), out_length


class TtDecoderBlock:
    """Snake -> ConvTranspose1d (upsample) -> 3 dilated ResidualUnits.

    Length goes from ``T`` to ``T * stride`` (the upstream config uses even
    strides, so the formula collapses to that)."""

    def __init__(
        self, mesh_device, state_dict: dict, in_channels: int, out_channels: int, stride: int, prefix: str = ""
    ):
        sd = state_dict
        self.mesh_device = mesh_device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = 2 * stride
        self.padding = (stride + 1) // 2  # = ceil(stride / 2)

        self.act = TtSnakeBeta(mesh_device, sd, prefix + "act.")

        # ConvTranspose1d weight is [in, out, k]; reshape to [in, out, k, 1]
        # for ttnn.conv_transpose2d (which expects 4D NHWC weights).
        up_w = reconstruct_wn_weight(sd, prefix + "upsample.").unsqueeze(-1)
        up_b = sd[prefix + "upsample.bias"]
        self.up_w = ttnn.from_torch(up_w, dtype=ttnn.float32)
        self.up_b = ttnn.from_torch(up_b.reshape(1, 1, 1, -1), dtype=ttnn.float32)

        self.res1 = TtResidualUnit(mesh_device, sd, out_channels, dilation=1, prefix=prefix + "res1.")
        self.res2 = TtResidualUnit(mesh_device, sd, out_channels, dilation=3, prefix=prefix + "res2.")
        self.res3 = TtResidualUnit(mesh_device, sd, out_channels, dilation=9, prefix=prefix + "res3.")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, input_length = x.shape[0], x.shape[1]

        h = self.act(x)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h, _ = _conv_transpose1d(
            h,
            self.up_w,
            self.up_b,
            self.mesh_device,
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch_size,
            input_length=input_length,
        )
        h = self.res1(h)
        h = self.res2(h)
        return self.res3(h)


class TtOobleckDecoder:
    """Full decoder: ``in_conv`` -> N DecoderBlocks -> Snake -> ``out_conv``.

    The default AudioX HF config has 5 blocks with strides (2, 4, 4, 8, 8),
    a total upsample factor of 2048 — i.e. 1 latent frame -> 2048 audio
    samples. The TT primitives keep ``[B, T, 1, C]`` throughout."""

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 64,
        c_mults=(1, 2, 4, 8, 16),
        strides=(2, 4, 4, 8, 8),
    ):
        sd = state_dict
        self.mesh_device = mesh_device
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        c_mults = (1,) + tuple(c_mults)
        depth = len(c_mults)

        in_w = reconstruct_wn_weight(sd, "in_conv.")
        in_b = sd["in_conv.bias"]
        self.in_conv_channels_in = latent_dim
        self.in_conv_channels_out = c_mults[-1] * channels
        self.in_w = ttnn.from_torch(in_w, dtype=ttnn.float32)
        self.in_b = ttnn.from_torch(in_b.reshape(1, 1, 1, -1), dtype=ttnn.float32)

        self.blocks = []
        for j, i in enumerate(range(depth - 1, 0, -1)):
            self.blocks.append(
                TtDecoderBlock(
                    mesh_device=mesh_device,
                    state_dict=sd,
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    prefix=f"blocks.{j}.",
                )
            )

        self.out_act = TtSnakeBeta(mesh_device, sd, "out_act.")

        # out_conv has bias=False upstream; pass None.
        out_w = reconstruct_wn_weight(sd, "out_conv.")
        self.out_conv_channels_in = c_mults[0] * channels
        self.out_w = ttnn.from_torch(out_w, dtype=ttnn.float32)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, input_length = x.shape[0], x.shape[1]

        h = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        h = _conv1d(
            h,
            self.in_w,
            self.in_b,
            self.mesh_device,
            self.in_conv_channels_in,
            self.in_conv_channels_out,
            kernel_size=7,
            dilation=1,
            padding=3,
            batch_size=batch_size,
            input_length=input_length,
        )

        for block in self.blocks:
            h = block(h)

        h = self.out_act(h)
        out_length = h.shape[1]
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        return _conv1d(
            h,
            self.out_w,
            None,
            self.mesh_device,
            self.out_conv_channels_in,
            self.out_channels,
            kernel_size=7,
            dilation=1,
            padding=3,
            batch_size=batch_size,
            input_length=out_length,
        )
