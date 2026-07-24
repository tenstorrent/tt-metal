# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 VAE encoder — TTNN implementation (single file)."""

from __future__ import annotations

import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.encoder import (
    BLOCK_OUT_CHANNELS,
    DownLevelSpec,
    IN_CHANNELS,
    NUM_RES_BLOCKS,
    OUT_PARAM_CHANNELS,
    PIXEL_H,
    PIXEL_T,
    PIXEL_W,
    encoder_down_level_specs,
    encoder_head_shape,
)
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import HunyuanSymmetricConv3d, promote_conv3d_fallback_to_exact
from models.experimental.hunyuan_image_3_0.tt.vae.decoder import (
    AttnBlockTTNN,
    ResnetBlockTTNN,
    dcae_down_shortcut_bthwc,
    dcae_space_to_depth_bthwc,
    encoder_head_shortcut_bthwc,
)
from models.experimental.hunyuan_image_3_0.tt.vae.spatial import norm_sharded
from models.experimental.hunyuan_image_3_0.tt.vae.encoder_weights import (
    init_encoder_conv_in as init_encoder_conv_in_weights,
    init_encoder_down as init_encoder_down_weights,
    init_encoder_head as init_encoder_head_weights,
    init_encoder_mid as init_encoder_mid_weights,
)
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import GroupNorm3D
from models.tt_dit.utils.conv3d import aligned_channels


class EncoderConvInTTNN(Module):
    """encoder.conv_in on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        pixel_t: int = PIXEL_T,
        pixel_h: int = PIXEL_H,
        pixel_w: int = PIXEL_W,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        promote_conv3d_fallback_to_exact(
            h_factor=1,
            w_factor=1,
            in_channels=aligned_channels(IN_CHANNELS),
            out_channels=BLOCK_OUT_CHANNELS[0],
            kernel_size=(3, 3, 3),
            t=pixel_t,
            h=pixel_h,
            w=pixel_w,
        )
        self.conv = HunyuanSymmetricConv3d(
            IN_CHANNELS,
            BLOCK_OUT_CHANNELS[0],
            kernel_size=3,
            stride=1,
            padding=1,
            mesh_device=mesh_device,
            dtype=dtype,
            t=pixel_t,
            h=pixel_h,
            w=pixel_w,
        )
        init_encoder_conv_in_weights(self)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv(x_bthwc)


class DownsampleDCAETTNN(Module):
    """DownsampleDCAE: Conv3d + space-to-depth rearrange + group-mean shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        add_temporal_downsample: bool,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        t: int,
        h: int,
        w: int,
    ) -> None:
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_temporal_downsample = add_temporal_downsample
        self.r1 = 2 if add_temporal_downsample else 1
        self.group_size = factor * in_channels // out_channels
        self.mesh_device = mesh_device
        self.dtype = dtype

        out_conv_channels = out_channels // factor
        promote_conv3d_fallback_to_exact(
            h_factor=1,
            w_factor=1,
            in_channels=aligned_channels(in_channels),
            out_channels=out_conv_channels,
            kernel_size=(3, 3, 3),
            t=t,
            h=h,
            w=w,
        )
        self.conv = HunyuanSymmetricConv3d(
            in_channels,
            out_conv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            mesh_device=mesh_device,
            dtype=dtype,
            t=t,
            h=h,
            w=w,
        )

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        # Shortcut before conv lowers peak DRAM (conv activations + mean buffer).
        shortcut = dcae_down_shortcut_bthwc(x_bthwc, self.out_channels, self.group_size, self.r1)
        conv_out = self.conv(x_bthwc)
        h_down = dcae_space_to_depth_bthwc(conv_out, self.r1)
        ttnn.deallocate(conv_out, force=False)
        return ttnn.add(h_down, shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class DownBlockTTNN(Module):
    """One encoder down level on replicated mesh."""

    def __init__(
        self,
        spec: DownLevelSpec,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.mesh_device = mesh_device
        self.dtype = dtype

        blocks = []
        in_ch = spec.in_channels
        for _ in range(num_res_blocks):
            blocks.append(
                ResnetBlockTTNN(
                    in_ch,
                    spec.block_channels,
                    mesh_device,
                    dtype=dtype,
                    t=spec.t,
                    h=spec.h,
                    w=spec.w,
                )
            )
            in_ch = spec.block_channels
        self.blocks = ModuleList(blocks)

        self.downsample = None
        if spec.has_downsample:
            assert spec.downsample_out_channels is not None
            self.downsample = DownsampleDCAETTNN(
                in_ch,
                spec.downsample_out_channels,
                add_temporal_downsample=spec.add_temporal_downsample,
                mesh_device=mesh_device,
                dtype=dtype,
                t=spec.t,
                h=spec.h,
                w=spec.w,
            )

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = x_bthwc
        for block in self.blocks:
            h = block(h)
        if self.downsample is None:
            return h
        return self.downsample(h)


class EncoderDownTTNN(Module):
    """All encoder down levels (post-conv_in) on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        pixel_t: int = PIXEL_T,
        pixel_h: int = PIXEL_H,
        pixel_w: int = PIXEL_W,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.down_blocks = ModuleList(
            [
                DownBlockTTNN(spec, mesh_device, dtype=dtype)
                for spec in encoder_down_level_specs(pixel_t, pixel_h, pixel_w)
            ]
        )
        init_encoder_down_weights(self)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = x_bthwc
        for down_block in self.down_blocks:
            h = down_block(h)
        return h


class EncoderMidBlockTTNN(Module):
    """encoder.mid on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        pixel_t: int = PIXEL_T,
        pixel_h: int = PIXEL_H,
        pixel_w: int = PIXEL_W,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        head_t, head_h, head_w, head_c = encoder_head_shape(pixel_t, pixel_h, pixel_w)

        self.block_1 = ResnetBlockTTNN(head_c, head_c, mesh_device, dtype=dtype, t=head_t, h=head_h, w=head_w)
        self.attn_1 = AttnBlockTTNN(head_c, mesh_device, dtype=dtype, t=head_t, h=head_h, w=head_w)
        self.block_2 = ResnetBlockTTNN(head_c, head_c, mesh_device, dtype=dtype, t=head_t, h=head_h, w=head_w)
        init_encoder_mid_weights(self)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = self.block_1(x_bthwc)
        h = self.attn_1(h)
        return self.block_2(h)


class EncoderHeadTTNN(Module):
    """encoder norm_out + conv_out + group-mean shortcut on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        pixel_t: int = PIXEL_T,
        pixel_h: int = PIXEL_H,
        pixel_w: int = PIXEL_W,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        head_t, head_h, head_w, head_c = encoder_head_shape(pixel_t, pixel_h, pixel_w)
        self.out_channels = OUT_PARAM_CHANNELS
        self.group_size = head_c // OUT_PARAM_CHANNELS

        self.norm_out = GroupNorm3D(
            num_channels=head_c,
            num_groups=32,
            input_nhw=head_t * head_h * head_w,
            num_batches=1,
            eps=1e-6,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.conv_out = HunyuanSymmetricConv3d(
            head_c,
            OUT_PARAM_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
            mesh_device=mesh_device,
            dtype=dtype,
            t=head_t,
            h=head_h,
            w=head_w,
        )
        init_encoder_head_weights(self)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        shortcut = encoder_head_shortcut_bthwc(x_bthwc, self.out_channels, self.group_size)

        ccl = getattr(self, "_sp_ccl", None)
        if ccl is None:
            h = self.norm_out(x_bthwc)
        else:
            h = norm_sharded(self.norm_out, x_bthwc, ccl, h_mesh_axis=self._sp_h, w_mesh_axis=self._sp_w)
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_out = self.conv_out(h)
        ttnn.deallocate(h, force=False)

        out = ttnn.add(conv_out, shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(conv_out, force=False)
        return out


class VAEEncoderTTNN(Module):
    """Full VAE encoder: conv_in -> down -> mid -> head.

    Default: replicated 2×2 mesh, DRAM interleaved activations.
    Set ``HY_ENCODER_W_SPATIAL=1`` (or pass ``h_mesh_axis=0``, ``w_mesh_axis=1``) for
    H/W spatial on a 2×2 mesh — convs use neighbor-pad halos; norms use distributed
    group_norm. Input must be uploaded with ``upload_bcthw_spatial`` or
    ``mesh_mapper_hw_spatial(..., h_mesh_axis=0, w_mesh_axis=1)``.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        pixel_t: int = PIXEL_T,
        pixel_h: int = PIXEL_H,
        pixel_w: int = PIXEL_W,
        ccl_manager=None,
        h_mesh_axis: int | None = None,
        w_mesh_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.pixel_t = pixel_t
        self.pixel_h = pixel_h
        self.pixel_w = pixel_w

        from models.experimental.hunyuan_image_3_0.tt.vae.spatial import enable_vae_spatial, encoder_w_spatial_enabled

        if encoder_w_spatial_enabled():
            if h_mesh_axis is None:
                h_mesh_axis = 0
            if w_mesh_axis is None:
                w_mesh_axis = 1
        self.w_mesh_axis = w_mesh_axis
        self.h_mesh_axis = h_mesh_axis
        self.ccl = ccl_manager

        enc_kw = dict(pixel_t=pixel_t, pixel_h=pixel_h, pixel_w=pixel_w)
        self.conv_in = EncoderConvInTTNN(mesh_device, dtype=dtype, **enc_kw)
        self.down = EncoderDownTTNN(mesh_device, dtype=dtype, **enc_kw)
        self.mid = EncoderMidBlockTTNN(mesh_device, dtype=dtype, **enc_kw)
        self.head = EncoderHeadTTNN(mesh_device, dtype=dtype, **enc_kw)

        if self.ccl is not None and (self.h_mesh_axis is not None or self.w_mesh_axis is not None):
            enable_vae_spatial(self, self.ccl, h_mesh_axis=self.h_mesh_axis, w_mesh_axis=self.w_mesh_axis)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        """BTHWC device tensor in -> BTHWC device tensor out."""
        h = self.conv_in(x_bthwc)
        h = self.down(h)
        h = self.mid(h)
        return self.head(h)
