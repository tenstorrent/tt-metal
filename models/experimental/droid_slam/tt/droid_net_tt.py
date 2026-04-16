# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tenstorrent tt-nn port of DROID-SLAM.

Porting is done progressively — each autoresearch iteration migrates
one component from torch-CPU to tt-nn. Pieces that are not yet ported
fall through to the torch reference implementation. This file owns
the hybrid forward path and the torch→ttnn weight staging.
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.droid_slam.reference.droid_net_ref import DroidNet as ReferenceDroidNet


def _torch_to_ttnn(t: torch.Tensor, device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, layout=layout, dtype=dtype, device=device)


def _nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def _nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


class TtCorrEncoder:
    """ttnn port of UpdateModule.corr_encoder.

    Two-conv stack:  (196 -> 128, 1x1)+ReLU  ->  (128 -> 128, 3x3)+ReLU.
    All activations stay NHWC so we avoid extra transposes between convs.
    """

    IN_CHANNELS = 4 * (2 * 3 + 1) ** 2  # 196
    HIDDEN = 128

    def __init__(self, device, src_module: torch.nn.Sequential):
        self.device = device
        # src_module = nn.Sequential(Conv2d, ReLU, Conv2d, ReLU)
        c1, _r1, c2, _r2 = src_module
        self.w1 = ttnn.from_torch(c1.weight.detach(), dtype=ttnn.float32)
        self.b1 = ttnn.from_torch(c1.bias.detach().reshape(1, 1, 1, -1), dtype=ttnn.float32)
        self.w2 = ttnn.from_torch(c2.weight.detach(), dtype=ttnn.float32)
        self.b2 = ttnn.from_torch(c2.bias.detach().reshape(1, 1, 1, -1), dtype=ttnn.float32)
        self._conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )

    def __call__(self, corr: torch.Tensor) -> torch.Tensor:
        n, c, h, w = corr.shape
        assert c == self.IN_CHANNELS
        x_nhwc = _nchw_to_nhwc(corr)
        x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.w1,
            bias_tensor=self.b1,
            in_channels=self.IN_CHANNELS,
            out_channels=self.HIDDEN,
            device=self.device,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=n,
            input_height=h,
            input_width=w,
            conv_config=self._conv_config,
        )
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.w2,
            bias_tensor=self.b2,
            in_channels=self.HIDDEN,
            out_channels=self.HIDDEN,
            device=self.device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=n,
            input_height=h,
            input_width=w,
            conv_config=self._conv_config,
        )
        out = ttnn.to_torch(x).float()
        # ttnn.conv2d packs output into [1, 1, N*H*W, C]. Reshape back
        # to NCHW to match the torch reference.
        out = out.reshape(n, h, w, self.HIDDEN)
        return _nhwc_to_nchw(out)


class TtDroidNet:
    """Hybrid DROID-SLAM forward path.

    Components ported so far:
      * UpdateModule.corr_encoder (ttnn)
    Everything else still delegates to the torch reference.
    """

    def __init__(self, device, reference: ReferenceDroidNet):
        self.device = device
        self.reference = reference
        self.reference.eval()
        self._corr_encoder = TtCorrEncoder(device, reference.update.corr_encoder)
        # Patch the reference UpdateModule so its forward uses our
        # ttnn corr_encoder without duplicating the surrounding logic.
        reference.update.corr_encoder = _CallableModule(self._corr_encoder)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor):
        return self.reference.extract_features(images)

    @torch.no_grad()
    def update(self, net, inp, corr, flow, ii):
        return self.reference.update(net, inp, corr, flow, ii)


class _CallableModule(torch.nn.Module):
    """Wrap a plain callable so it slots into the reference nn.Sequential."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, x):
        return self._fn(x)
