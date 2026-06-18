# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared VAE Conv3d layer (weight prep uses tt_dit Module hook; kept separate from model logic)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import ttnn
from models.common.utility_functions import is_blackhole
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import LATENT_H, LATENT_T, LATENT_W
from models.tt_dit.layers.audio_ops import prepare_conv3d_weight_state
from models.tt_dit.layers.module import Module, Parameter
from models.tt_dit.utils.conv3d import (
    _BLOCKINGS,
    _DEFAULT_BLOCKINGS,
    _ntuple,
    aligned_channels,
    get_conv3d_config,
    register_conv3d_configs,
)

# Max im2col elements (in_ch*T*H*W*kernel_vol) before a conv3d chunks over H.
_CONV3D_CHUNK_ELEMS = 2 * 1024 * 1024 * 1024

register_conv3d_configs(
    {
        (1024, 1024, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 1024, (1, 1, 1)): (256, 32, 1, 1, 1),
        (1024, 8192, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 4096, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 2048, (3, 3, 3)): (64, 32, 1, 2, 2),
        (512, 1024, (3, 3, 3)): (64, 32, 1, 2, 2),
        (512, 512, (3, 3, 3)): (64, 32, 1, 2, 2),
        (256, 512, (3, 3, 3)): (64, 32, 1, 2, 2),
        (256, 256, (3, 3, 3)): (64, 32, 1, 2, 2),
        (128, 512, (3, 3, 3)): (64, 32, 1, 2, 2),
        (128, 128, (3, 3, 3)): (64, 32, 1, 2, 2),
        (128, 3, (3, 3, 3)): (32, 32, 1, 2, 2),
        (32, 1024, (3, 3, 3)): (32, 32, 1, 2, 2),
    }
)


def promote_conv3d_fallback_to_exact(
    *,
    h_factor: int,
    w_factor: int,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    t: int,
    h: int,
    w: int,
) -> None:
    """Copy a channel-keyed fallback blocking into the exact table for this shape."""
    blocking_key = (h_factor, w_factor, in_channels, out_channels, kernel_size, t, h, w)
    if blocking_key in _BLOCKINGS:
        return
    channel_key = (in_channels, out_channels, kernel_size)
    fallback = _DEFAULT_BLOCKINGS.get(channel_key)
    if fallback is not None:
        _BLOCKINGS[blocking_key] = fallback


class HunyuanSymmetricConv3d(Module):
    """Conv3d with symmetric padding on T, H, W. Input/output layout: BTHWC ROW_MAJOR."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Sequence[int] | int = 3,
        stride: Sequence[int] | int = 1,
        padding: Sequence[int] | int = 1,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        t: int = LATENT_T,
        h: int = LATENT_H,
        w: int = LATENT_W,
    ) -> None:
        super().__init__()

        self.unpadded_in_channels = in_channels
        self.in_channels = aligned_channels(in_channels)
        self.unpadded_out_channels = out_channels
        self.out_channels = out_channels

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.padding = _ntuple(padding, 3)
        self.mesh_device = mesh_device
        self.dtype = dtype

        promote_conv3d_fallback_to_exact(
            h_factor=1,
            w_factor=1,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            t=t,
            h=h,
            w=w,
        )
        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
            T=t,
            H=h,
            W=w,
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if (is_blackhole() and dtype == ttnn.float32)
            else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        weight_elems = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(
            total_shape=[weight_elems, self.out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )
        self.bias = Parameter(
            total_shape=[1, self.out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, Any]) -> None:
        if "weight" in state:
            prepare_conv3d_weight_state(
                state,
                state["weight"],
                conv_config=self.conv_config,
                mesh_device=self.mesh_device,
                dtype=self.dtype,
                unpadded_out=self.unpadded_out_channels,
                out_channels=self.out_channels,
                unpadded_in=self.unpadded_in_channels,
                in_channels=self.in_channels,
            )
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def _conv(self, x_bthwc, padding, config):
        return ttnn.experimental.conv3d(
            input_tensor=x_bthwc,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        assert (
            x_bthwc.layout == ttnn.ROW_MAJOR_LAYOUT
        ), f"HunyuanSymmetricConv3d expects ROW_MAJOR, got {x_bthwc.layout}"

        b, t, h, w, _ = x_bthwc.shape
        kT, kH, kW = self.kernel_size
        pT, pH, pW = self.padding
        im2col_elems = self.in_channels * t * h * w * kT * kH * kW
        if im2col_elems <= _CONV3D_CHUNK_ELEMS or h <= 1 or self.stride[1] != 1 or pH == 0:
            return self._conv(x_bthwc, self.padding, self.conv_config)

        # Chunk over H to bound the conv's ~im2col DRAM buffer. Zero-pad H by pH
        # (true-boundary padding), then conv overlapping strips with padding_h=0;
        # interior strips read real neighbor rows from the padded tensor (halo).
        n_chunks = (im2col_elems + _CONV3D_CHUNK_ELEMS - 1) // _CONV3D_CHUNK_ELEMS
        hc = (h + n_chunks - 1) // n_chunks
        last = x_bthwc.shape[-1]
        zpad = ttnn.zeros(
            [b, t, pH, w, last], dtype=x_bthwc.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
        )
        x_pad = ttnn.concat([zpad, x_bthwc, zpad], dim=2)
        ttnn.deallocate(zpad)
        h_pad = x_pad.shape[2]
        grid = self.mesh_device.compute_with_storage_grid_size()
        outs = []
        for o in range(0, h, hc):
            oe = min(h, o + hc)
            in_slice = ttnn.slice(x_pad, [0, 0, o, 0, 0], [b, t, min(oe + 2 * pH, h_pad), w, last])
            cfg = get_conv3d_config(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.dtype,
                grid_size=grid,
                h_factor=1,
                w_factor=1,
                T=t,
                H=in_slice.shape[2],
                W=w,
            )
            outs.append(self._conv(in_slice, (pT, 0, pW), cfg))
            ttnn.deallocate(in_slice)
        ttnn.deallocate(x_pad)
        out = ttnn.concat(outs, dim=2)
        for o_t in outs:
            ttnn.deallocate(o_t)
        return out
