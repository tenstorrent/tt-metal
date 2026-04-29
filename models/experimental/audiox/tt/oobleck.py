# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the AudioX Oobleck VAE decoder primitives.

Provides ``TtSnakeBeta`` and ``TtResidualUnit``. The full ``DecoderBlock``
adds an upsample (ConvTranspose1d) which TTNN does not yet support natively
in 1D — that lands in a follow-up chunk.

Tensor convention here is NHWC with H=1 (``[B, T, 1, C]``), matching
``ttnn.conv1d``'s expected layout. SnakeBeta runs on TILE_LAYOUT (pointwise),
conv1d runs on ROW_MAJOR_LAYOUT; we convert at boundaries."""

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
    out, out_length, _ = ttnn.conv1d(
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
