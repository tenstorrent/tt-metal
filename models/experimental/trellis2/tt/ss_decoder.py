# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 sparse-structure decoder (`ss_dec_conv3d_16l8`: dense 3D CNN, 16^3 x 8 -> 64^3 x 1 logits) on one device.

Mirrors `reference/ss_decoder_host.functional_forward` op for op (DESIGN.md D07: the structure stage stays on device up
to the occupancy mask). Activations are channels-last; `ttnn.experimental.conv3d` takes them row-major as
[1, D, H, W, C] (C padded to 32) and the norms, SiLU and residual adds run on the tile-major view [1, 1, D*H*W, C].
D is zero-padded with a concat before each 3x3x3 conv, so the conv itself pads only H and W, as every tt_dit conv3d
does. The upsample conv's output channels are pre-ordered (i, j, k, c) on the host, so the 3D pixel shuffle is two
rank-4 permutes.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.experimental.trellis2.reference import ss_decoder_host as host
from models.tt_dit.layers.conv3d import get_conv3d_config

ALIGN = 32
TT_DIT_BLOCKING = (128, 256, 512, 768)  # input channel counts with a blocking in tt_dit get_conv3d_config


def _fp32_conv() -> bool:
    """Activations, weights and convs in fp32 (conv3d's fp32-exact path): bf16 gave 3x the CPU bf16 floor in 64^3 flips,
    fp32 is below it at 1.84 s vs 0.15 s (S10 G4a). TRELLIS2_SSDEC_FP32_CONV=0 restores bf16."""
    return os.environ.get("TRELLIS2_SSDEC_FP32_CONV", "1") == "1"


def _hifi4(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )


def _pad_channels(t: torch.Tensor, dim: int) -> torch.Tensor:
    pad = -t.shape[dim] % ALIGN
    if not pad:
        return t
    shape = list(t.shape)
    shape[dim] = pad
    return torch.cat([t, t.new_zeros(shape)], dim=dim)


class _Conv3d:
    """k x k x k conv with symmetric zero padding on a row-major [1, D, H, W, C] activation."""

    def __init__(self, w: torch.Tensor, b: torch.Tensor, *, device) -> None:
        self.device = device
        dtype = ttnn.float32 if _fp32_conv() else ttnn.bfloat16
        self.k = w.shape[-1]
        # Pad C_out to a multiple of 32 with zero filters (tt_dit never runs conv3d below 32); callers slice.
        w, b = _pad_channels(w, 0), _pad_channels(b, 0)
        self.out_channels = w.shape[0]
        self.out_dtype = dtype
        # Default blocking keeps whole channel blocks in L1, which overflows at 512 -> 1024; tt_dit's Wan table fits.
        in_channels = w.shape[1] + (-w.shape[1] % ALIGN)
        if dtype == ttnn.float32:
            self.config = get_conv3d_config(in_channels, device.compute_with_storage_grid_size(), weights_dtype=dtype)
        elif in_channels in TT_DIT_BLOCKING:
            self.config = get_conv3d_config(in_channels, device.compute_with_storage_grid_size())
        else:
            self.config = ttnn.Conv3dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            )
        weight = ttnn.from_torch(_pad_channels(w.float(), 1), dtype=dtype, pad_value=0)
        self.weight = ttnn.experimental.prepare_conv3d_weights(
            weight_tensor=weight, C_in_block=self.config.C_in_block, alignment=ALIGN, device=device
        )
        self.bias = ttnn.from_torch(
            b.float().reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0
        )
        self.compute_kernel_config = _hifi4(device)
        self._zero_planes: dict[tuple, ttnn.Tensor] = {}

    def _pad_depth(self, x: ttnn.Tensor) -> ttnn.Tensor:
        p = self.k // 2
        if p == 0:
            return x
        _, _, h, w, c = x.shape
        key = (h, w, c)
        if key not in self._zero_planes:
            self._zero_planes[key] = ttnn.from_torch(
                torch.zeros(1, p, h, w, c), dtype=x.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        zeros = self._zero_planes[key]
        return ttnn.concat([zeros, x, zeros], dim=1)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        p = self.k // 2
        return ttnn.experimental.conv3d(
            input_tensor=self._pad_depth(x),
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            config=self.config,
            dtype=self.out_dtype,
            output_channels=self.out_channels,
            kernel_size=(self.k,) * 3,
            padding=(0, p, p),
            padding_mode="zeros",
            compute_kernel_config=self.compute_kernel_config,
        )


class _NormSiLU:
    """Channel LayerNorm (affine, eps 1e-5) + SiLU on the tile-major [1, 1, N, C] view.

    In fp32 with one bf16 cast at the end, as TRELLIS.2's LayerNorm32: the bf16 version shrank every norm output by
    0.1-0.2% (S10 probe), which flips near-zero logits. TRELLIS2_SSDEC_FP32_NORM=0 restores bf16 for A/B runs.
    """

    def __init__(self, w: torch.Tensor, b: torch.Tensor, *, device) -> None:
        self.fp32 = os.environ.get("TRELLIS2_SSDEC_FP32_NORM", "1") == "1"
        dtype = ttnn.float32 if self.fp32 else ttnn.bfloat16
        to_dev = lambda t: ttnn.from_torch(
            t.float().reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.weight, self.bias = to_dev(w), to_dev(b)
        self.compute_kernel_config = _hifi4(device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.fp32 and x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        y = ttnn.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            epsilon=host.NORM_EPS,
            compute_kernel_config=self.compute_kernel_config,
        )
        y = ttnn.silu(y)
        return ttnn.typecast(y, ttnn.bfloat16) if self.fp32 and not _fp32_conv() else y


def _tokens(x: ttnn.Tensor) -> ttnn.Tensor:
    """Row-major [1, D, H, W, C] -> tile [1, 1, D*H*W, C]."""
    _, d, h, w, c = x.shape
    return ttnn.to_layout(ttnn.reshape(x, (1, 1, d * h * w, c)), ttnn.TILE_LAYOUT)


def _grid(x: ttnn.Tensor, dhw: tuple[int, int, int]) -> ttnn.Tensor:
    """Tile [1, 1, D*H*W, C] -> row-major [1, D, H, W, C]."""
    return ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, *dhw, x.shape[-1]))


def _shuffle(x: ttnn.Tensor, s: int = host.SCALE) -> ttnn.Tensor:
    """`host.shuffle_ndhwc` on device: [1, D, H, W, s^3 C] (i, j, k, c order) -> [1, sD, sH, sW, C]."""
    _, d, h, w, cc = x.shape
    c = cc // s**3
    x = ttnn.permute(ttnn.reshape(x, (d * h, w, s * s, s * c)), (0, 2, 1, 3))
    x = ttnn.permute(ttnn.reshape(x, (d, h, s, s * w * s * c)), (0, 2, 1, 3))
    return ttnn.reshape(x, (1, d * s, h * s, w * s, c))


class _ResBlock:
    def __init__(self, s: dict[str, torch.Tensor], p: str, *, device) -> None:
        self.norm1 = _NormSiLU(s[p + "norm1.weight"], s[p + "norm1.bias"], device=device)
        self.conv1 = _Conv3d(s[p + "conv1.weight"], s[p + "conv1.bias"], device=device)
        self.norm2 = _NormSiLU(s[p + "norm2.weight"], s[p + "norm2.bias"], device=device)
        self.conv2 = _Conv3d(s[p + "conv2.weight"], s[p + "conv2.bias"], device=device)

    def __call__(self, x: ttnn.Tensor, dhw: tuple[int, int, int]) -> ttnn.Tensor:
        """x: tile [1, 1, N, C] (residual stream) -> same."""
        h = _tokens(self.conv1(_grid(self.norm1(x), dhw)))
        h = _tokens(self.conv2(_grid(self.norm2(h), dhw)))
        return ttnn.add(x, h)


class TtSSDecoder:
    """Structure decoder on one device: z [1, 8, 16, 16, 16] (torch, reference layout) -> logits."""

    def __init__(self, state: dict[str, torch.Tensor], *, mesh_device: ttnn.MeshDevice) -> None:
        self.device = mesh_device
        s = {k: v.float() for k, v in state.items()}
        self.input_layer = _Conv3d(s["input_layer.weight"], s["input_layer.bias"], device=mesh_device)
        self.middle = [
            _ResBlock(s, f"middle_block.{i}.", device=mesh_device) for i in range(host.NUM_RES_BLOCKS_MIDDLE)
        ]
        self.blocks = []
        for i, (kind, _, _) in enumerate(host.block_layout()):
            if kind == "res":
                self.blocks.append(("res", _ResBlock(s, f"blocks.{i}.", device=mesh_device)))
            else:
                w, b = host.shuffle_weight_order(s[f"blocks.{i}.conv.weight"], s[f"blocks.{i}.conv.bias"])
                self.blocks.append(("up", _Conv3d(w, b, device=mesh_device)))
        self.out_norm = _NormSiLU(s["out_layer.0.weight"], s["out_layer.0.bias"], device=mesh_device)
        # Logits come out in the conv dtype: a bf16-input conv3d with an fp32 output returns garbage (scale 0 vs the
        # mirror, S10 probe); bf16 rounding never changes a logit's sign, so occupancy (logits > 0) is unaffected.
        self.out_conv = _Conv3d(s["out_layer.2.weight"], s["out_layer.2.bias"], device=mesh_device)

    def forward_device(self, z: ttnn.Tensor) -> ttnn.Tensor:
        """Row-major [1, 16, 16, 16, 32] (latent channels zero-padded to 32) -> row-major [1, 64, 64, 64, 32] (channel 0 is the logit)."""
        dhw = tuple(z.shape)[1:4]
        h = _tokens(self.input_layer(z))
        for block in self.middle:
            h = block(h, dhw)
        for kind, block in self.blocks:
            if kind == "res":
                h = block(h, dhw)
            else:
                g = _shuffle(block(_grid(h, dhw)))
                dhw = tuple(g.shape)[1:4]
                h = _tokens(g)
        return self.out_conv(_grid(self.out_norm(h), dhw))

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        x = _pad_channels(z.float().permute(0, 2, 3, 4, 1), 4)
        dtype = ttnn.float32 if _fp32_conv() else ttnn.bfloat16
        x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        out = ttnn.to_torch(ttnn.get_device_tensors(self.forward_device(x))[0]).float()
        return out[..., : host.OUT_CHANNELS].permute(0, 4, 1, 2, 3).contiguous()
