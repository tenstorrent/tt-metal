# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 VAE decoder — TTNN implementation (single file)."""

from __future__ import annotations

import math

import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import (
    BLOCK_IN_CHANNELS,
    GN_EPS,
    LATENT_H,
    LATENT_T,
    LATENT_W,
    NUM_GROUPS,
    NUM_RES_BLOCKS,
    OUT_CHANNELS,
    UpLevelSpec,
    Z_CHANNELS,
    decoder_tail_shape,
    decoder_up_level_specs,
)
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import HunyuanSymmetricConv3d
from models.experimental.hunyuan_image_3_0.tt.vae.spatial import gather_hw, partition_hw, norm_sharded
from models.experimental.hunyuan_image_3_0.tt.vae.decoder_weights import (
    init_conv_in as init_conv_in_weights,
    init_decoder_tail as init_decoder_tail_weights,
    init_decoder_up as init_decoder_up_weights,
    init_mid_block as init_mid_block_weights,
)
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import GroupNorm3D


def bcthw_to_bthwc(x_bcthw: ttnn.Tensor) -> ttnn.Tensor:
    """BCTHW -> BTHWC (channels last for conv3d)."""
    return ttnn.permute(x_bcthw, (0, 2, 3, 4, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def bthwc_to_bcthw(x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
    """BTHWC -> BCTHW."""
    return ttnn.permute(x_bthwc, (0, 4, 1, 2, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)


# Max flat elements before depth-to-space chunks over H (the 8-D reshape pads the
# size-2 dims, so keep the pre-pad flat count well under the device limit).
_D2S_CHUNK_ELEMS = 32 * 1024 * 1024


def dcae_depth_to_space_bthwc(
    x_bthwc: ttnn.Tensor,
    *,
    out_channels: int,
    r1: int,
    r2: int = 2,
    r3: int = 2,
) -> ttnn.Tensor:
    """Depth-to-space on BTHWC: (B,T,H,W,r1*r2*r3*C) -> (B,T*r1,H*r2,W*r3,C).

    At 1024² the 8-D reshape+permute tilizes the size-2 (r2,r3) dims toward 32 and
    allocates ~16GB. H rows are independent (each input H expands to r2 output
    rows), so we chunk over input H and concat — bounding the buffer, exactly
    (no precision loss).
    """
    b, t, h, w, _ = x_bthwc.shape

    def _d2s(x):
        # Reference channel order is (r1 r2 r3 c) with c fastest — decoder.py
        # rearrange "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)". Split the
        # channel dim accordingly (NOT (c r1 r2 r3)).
        xb = x.shape[2]
        x = ttnn.reshape(x, (b, t, xb, w, r1, r2, r3, out_channels))
        x = ttnn.permute(x, (0, 1, 4, 2, 5, 3, 6, 7))  # -> b, t, r1, xb, r2, w, r3, out
        return ttnn.reshape(x, (b, t * r1, xb * r2, w * r3, out_channels))

    flat = b * t * h * w * out_channels * r1 * r2 * r3
    if flat <= _D2S_CHUNK_ELEMS or h <= 1:
        return _d2s(x_bthwc)

    n_chunks = (flat + _D2S_CHUNK_ELEMS - 1) // _D2S_CHUNK_ELEMS
    hc = (h + n_chunks - 1) // n_chunks
    last = x_bthwc.shape[-1]
    outs = []
    for o in range(0, h, hc):
        oe = min(h, o + hc)
        xs = ttnn.slice(x_bthwc, [0, 0, o, 0, 0], [b, t, oe, w, last])
        outs.append(_d2s(xs))
        ttnn.deallocate(xs)
    out = ttnn.concat(outs, dim=2)  # along H*r2
    for o_t in outs:
        ttnn.deallocate(o_t)
    return out


def dcae_space_to_depth_bthwc(
    x_bthwc: ttnn.Tensor,
    r1: int,
    r2: int = 2,
    r3: int = 2,
) -> ttnn.Tensor:
    """Space-to-depth on BTHWC: (B,T,H,W,C) -> (B,T/r1,H/r2,W/r3,r1*r2*r3*C)."""
    b, t, h, w, c = x_bthwc.shape
    x = ttnn.reshape(x_bthwc, (b, t // r1, r1, h // r2, r2, w // r3, r3, c))
    x = ttnn.permute(x, (0, 1, 3, 5, 2, 4, 6, 7))
    return ttnn.reshape(x, (b, t // r1, h // r2, w // r3, r1 * r2 * r3 * c))


# Max elements in the flattened (N, group_size) view before spatial chunking.
# Full 1024² encoder shortcut is ~268M elements and triggers ~16GB tilize buffers.
_GROUP_MEAN_MAX_FLAT_ELEMENTS = 8 * 1024 * 1024


def _group_mean_flat_bthwc(
    x_bthwc: ttnn.Tensor,
    *,
    b: int,
    t: int,
    h: int,
    w: int,
    out_channels: int,
    group_size: int,
) -> ttnn.Tensor:
    """Group-mean on a single spatial tile; last dim = out_channels * group_size."""
    n = b * t * h * w * out_channels
    x = ttnn.reshape(x_bthwc, (n, group_size))

    acc = ttnn.slice(x, (0, 0), (n, 1))
    for g in range(1, group_size):
        part = ttnn.slice(x, (0, g), (n, g + 1))
        new_acc = ttnn.add(acc, part, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(acc, force=False)
        ttnn.deallocate(part, force=False)
        acc = new_acc
    ttnn.deallocate(x, force=False)

    acc = ttnn.multiply(acc, 1.0 / group_size, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(acc, (b, t, h, w, out_channels))


def group_mean_groups_bthwc(
    x_bthwc: ttnn.Tensor,
    *,
    out_channels: int,
    group_size: int,
) -> ttnn.Tensor:
    """Mean over channel groups on BTHWC (last dim = out_channels * group_size).

    Uses slice+add instead of ttnn.mean on a 6D tensor to avoid huge tilize allocations
    at full encoder resolution. Large spatial tensors are processed in H strips.
    """
    if group_size == 1:
        return x_bthwc

    b, t, h, w, _ = x_bthwc.shape
    flat_elements = b * t * h * w * out_channels
    if flat_elements <= _GROUP_MEAN_MAX_FLAT_ELEMENTS:
        return _group_mean_flat_bthwc(x_bthwc, b=b, t=t, h=h, w=w, out_channels=out_channels, group_size=group_size)

    h_chunk = max(1, _GROUP_MEAN_MAX_FLAT_ELEMENTS // (b * t * w * out_channels))
    outputs: list[ttnn.Tensor] = []
    for h0 in range(0, h, h_chunk):
        h1 = min(h0 + h_chunk, h)
        chunk = ttnn.slice(x_bthwc, (0, 0, h0, 0, 0), (b, t, h1, w, x_bthwc.shape[-1]))
        chunk_mean = _group_mean_flat_bthwc(
            chunk,
            b=b,
            t=t,
            h=h1 - h0,
            w=w,
            out_channels=out_channels,
            group_size=group_size,
        )
        ttnn.deallocate(chunk, force=False)
        outputs.append(chunk_mean)

    if len(outputs) == 1:
        return outputs[0]

    result = outputs[0]
    for nxt in outputs[1:]:
        merged = ttnn.concat([result, nxt], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(result, force=False)
        ttnn.deallocate(nxt, force=False)
        result = merged
    return result


def dcae_down_shortcut_bthwc(
    x_bthwc: ttnn.Tensor,
    out_channels: int,
    group_size: int,
    r1: int,
) -> ttnn.Tensor:
    """Group-mean shortcut for DownsampleDCAE on BTHWC."""
    grouped = dcae_space_to_depth_bthwc(x_bthwc, r1)
    shortcut = group_mean_groups_bthwc(grouped, out_channels=out_channels, group_size=group_size)
    ttnn.deallocate(grouped, force=False)
    return shortcut


def encoder_head_shortcut_bthwc(
    x_bthwc: ttnn.Tensor,
    out_channels: int,
    group_size: int,
) -> ttnn.Tensor:
    """Channel-group mean shortcut for encoder head on BTHWC."""
    return group_mean_groups_bthwc(x_bthwc, out_channels=out_channels, group_size=group_size)


class ConvInTTNN(Module):
    """conv_in on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.repeats = BLOCK_IN_CHANNELS // Z_CHANNELS

        self.conv = HunyuanSymmetricConv3d(
            Z_CHANNELS,
            BLOCK_IN_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
            mesh_device=mesh_device,
            dtype=dtype,
            t=LATENT_T,
            h=LATENT_H,
            w=LATENT_W,
        )
        init_conv_in_weights(self)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        shortcut = ttnn.repeat_interleave(x_bthwc, self.repeats, dim=4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_out = self.conv(x_bthwc)
        out_bthwc = ttnn.add(conv_out, shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(conv_out, force=False)
        ttnn.deallocate(shortcut, force=False)
        return out_bthwc


class ResnetBlockTTNN(Module):
    """GroupNorm3D -> SiLU -> Conv3d x2 + residual (BTHWC ROW_MAJOR)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        mesh_device: ttnn.MeshDevice | None = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        t: int = LATENT_T,
        h: int = LATENT_H,
        w: int = LATENT_W,
        prefix: str = "",
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.input_nhw = t * h * w

        gn_kwargs = dict(
            num_groups=NUM_GROUPS,
            input_nhw=self.input_nhw,
            num_batches=1,
            eps=GN_EPS,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.norm1 = GroupNorm3D(num_channels=in_channels, **gn_kwargs)
        self.norm2 = GroupNorm3D(num_channels=out_channels, **gn_kwargs)
        conv_kwargs = dict(mesh_device=mesh_device, dtype=dtype, t=t, h=h, w=w)
        self.conv1 = HunyuanSymmetricConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, **conv_kwargs
        )
        self.conv2 = HunyuanSymmetricConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, **conv_kwargs
        )
        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = HunyuanSymmetricConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, **conv_kwargs
            )

        if prefix:
            self._prefix = prefix

    def _gn(self, norm, x):
        # Spatial-parallel: gather H/W -> GroupNorm -> re-shard (per-group stats need
        # full spatial). Replicated path (no _sp_ccl) runs the norm directly.
        ccl = getattr(self, "_sp_ccl", None)
        if ccl is None:
            return norm(x)
        return norm_sharded(norm, x, ccl, h_mesh_axis=self._sp_h, w_mesh_axis=self._sp_w)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        residual = self.nin_shortcut(x_bthwc) if self.nin_shortcut is not None else x_bthwc

        h = self._gn(self.norm1, x_bthwc)
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self.conv1(h)

        h = self._gn(self.norm2, h)
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self.conv2(h)

        return ttnn.add(residual, h, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class AttnBlockTTNN(Module):
    """GroupNorm3D + Q/K/V 1x1 Conv3d + SDPA + proj_out + residual."""

    ATTN_SCALE = 1.0 / math.sqrt(1024)

    def __init__(
        self,
        channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.spatial = LATENT_H * LATENT_W

        self.norm = GroupNorm3D(
            num_channels=channels,
            num_groups=NUM_GROUPS,
            input_nhw=LATENT_T * LATENT_H * LATENT_W,
            num_batches=1,
            eps=GN_EPS,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        conv_kwargs = dict(
            mesh_device=mesh_device,
            dtype=dtype,
            t=LATENT_T,
            h=LATENT_H,
            w=LATENT_W,
        )
        self.q = HunyuanSymmetricConv3d(channels, channels, kernel_size=1, stride=1, padding=0, **conv_kwargs)
        self.k = HunyuanSymmetricConv3d(channels, channels, kernel_size=1, stride=1, padding=0, **conv_kwargs)
        self.v = HunyuanSymmetricConv3d(channels, channels, kernel_size=1, stride=1, padding=0, **conv_kwargs)
        self.proj_out = HunyuanSymmetricConv3d(channels, channels, kernel_size=1, stride=1, padding=0, **conv_kwargs)

    def _to_sdpa(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        """BTHWC ROW_MAJOR -> [B, 1, T*H*W, C] TILE (matches ref AttnBlock)."""
        b, t, h, w, c = x_bthwc.shape
        x_btsc = ttnn.reshape(x_bthwc, (b, t, h * w, c))
        x_b1sc = ttnn.reshape(x_btsc, (b, 1, t * h * w, c))
        ttnn.deallocate(x_btsc, force=False)
        return ttnn.to_layout(x_b1sc, ttnn.TILE_LAYOUT)

    def _from_sdpa(self, x_b1sc: ttnn.Tensor, b: int, t: int, h: int, w: int, c: int) -> ttnn.Tensor:
        """[B, 1, T*H*W, C] TILE -> BTHWC ROW_MAJOR for conv3d."""
        x = ttnn.to_layout(x_b1sc, ttnn.ROW_MAJOR_LAYOUT)
        x_btsc = ttnn.reshape(x, (b, t, h * w, c))
        ttnn.deallocate(x, force=False)
        return ttnn.reshape(x_btsc, (b, t, h, w, c))

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        # Attention is global over T*H*W, so under spatial sharding gather the whole
        # block to full spatial (cheap — attn only runs at 64x64), then re-shard. The
        # 1x1 convs inside run correctly on the full tensor (padding 0 -> no halo).
        ccl = getattr(self, "_sp_ccl", None)
        if ccl is not None:
            x_full = gather_hw(ccl, x_bthwc, h_mesh_axis=self._sp_h, w_mesh_axis=self._sp_w)
            out_full = self._forward_impl(x_full)
            ttnn.deallocate(x_full, force=False)
            return partition_hw(out_full, h_mesh_axis=self._sp_h, w_mesh_axis=self._sp_w)
        return self._forward_impl(x_bthwc)

    def _forward_impl(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        residual = x_bthwc
        b, t, h, w, c = x_bthwc.shape

        normed = self.norm(x_bthwc)
        q = self._to_sdpa(self.q(normed))
        k = self._to_sdpa(self.k(normed))
        v = self._to_sdpa(self.v(normed))
        ttnn.deallocate(normed, force=False)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=self.ATTN_SCALE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q, force=False)
        ttnn.deallocate(k, force=False)
        ttnn.deallocate(v, force=False)

        attn_bthwc = self._from_sdpa(attn, b, t, h, w, c)
        ttnn.deallocate(attn, force=False)

        out = self.proj_out(attn_bthwc)
        ttnn.deallocate(attn_bthwc, force=False)
        return ttnn.add(residual, out, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class MidBlockTTNN(Module):
    """mid.block_1 -> mid.attn_1 -> mid.block_2 on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.block_1 = ResnetBlockTTNN(1024, 1024, mesh_device, dtype=dtype)
        self.attn_1 = AttnBlockTTNN(1024, mesh_device, dtype=dtype)
        self.block_2 = ResnetBlockTTNN(1024, 1024, mesh_device, dtype=dtype)
        init_mid_block_weights(self)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        x = self.block_1(x_bthwc)
        x = self.attn_1(x)
        return self.block_2(x)


class UpsampleDCAETTNN(Module):
    """UpsampleDCAE: Conv3d + depth-to-space rearrange + channel-repeat shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        add_temporal_upsample: bool,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        t: int,
        h: int,
        w: int,
    ) -> None:
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_temporal_upsample = add_temporal_upsample
        self.r1 = 2 if add_temporal_upsample else 1
        self.repeats = factor * out_channels // in_channels
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.conv = HunyuanSymmetricConv3d(
            in_channels,
            out_channels * factor,
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
        conv_out = self.conv(x_bthwc)
        h_up = dcae_depth_to_space_bthwc(conv_out, out_channels=self.out_channels, r1=self.r1)
        ttnn.deallocate(conv_out, force=False)

        shortcut_in = ttnn.repeat_interleave(x_bthwc, self.repeats, dim=4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shortcut_up = dcae_depth_to_space_bthwc(shortcut_in, out_channels=self.out_channels, r1=self.r1)
        ttnn.deallocate(shortcut_in, force=False)

        return ttnn.add(h_up, shortcut_up, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class UpBlockTTNN(Module):
    """One decoder up level on replicated mesh."""

    def __init__(
        self,
        spec: UpLevelSpec,
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
        for _ in range(num_res_blocks + 1):
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

        self.upsample = None
        if spec.has_upsample:
            assert spec.upsample_out_channels is not None
            self.upsample = UpsampleDCAETTNN(
                in_ch,
                spec.upsample_out_channels,
                add_temporal_upsample=spec.add_temporal_upsample,
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
        if self.upsample is None:
            return h
        return self.upsample(h)


class NormOutTTNN(Module):
    """GroupNorm3D + swish (x * sigmoid(x)) on replicated mesh."""

    def __init__(
        self,
        channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        t: int,
        h: int,
        w: int,
        num_batches: int = 1,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.norm = GroupNorm3D(
            num_channels=channels,
            num_groups=NUM_GROUPS,
            input_nhw=t * h * w,
            num_batches=num_batches,
            eps=GN_EPS,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        ccl = getattr(self, "_sp_ccl", None)
        if ccl is None:
            h = self.norm(x_bthwc)
        else:
            h = norm_sharded(self.norm, x_bthwc, ccl, h_mesh_axis=self._sp_h, w_mesh_axis=self._sp_w)
        return ttnn.mul(
            h, ttnn.sigmoid(h, memory_config=ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )


class ConvOutTTNN(Module):
    """Final Conv3d to RGB on replicated mesh."""

    def __init__(
        self,
        in_channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        *,
        out_channels: int = OUT_CHANNELS,
        t: int,
        h: int,
        w: int,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.conv = HunyuanSymmetricConv3d(
            in_channels,
            out_channels,
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
        return self.conv(x_bthwc)


class DecoderTailTTNN(Module):
    """norm_out + conv_out on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        tail_t, tail_h, tail_w, tail_c = decoder_tail_shape()

        self.norm_out = NormOutTTNN(tail_c, mesh_device, dtype=dtype, t=tail_t, h=tail_h, w=tail_w)
        self.conv_out = ConvOutTTNN(
            tail_c, mesh_device, dtype=dtype, out_channels=OUT_CHANNELS, t=tail_t, h=tail_h, w=tail_w
        )
        init_decoder_tail_weights(self)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = self.norm_out(x_bthwc)
        return self.conv_out(h)


class DecoderUpTTNN(Module):
    """All decoder up levels (post-mid) on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.up_blocks = ModuleList([UpBlockTTNN(spec, mesh_device, dtype=dtype) for spec in decoder_up_level_specs()])
        init_decoder_up_weights(self)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = x_bthwc
        for up_block in self.up_blocks:
            h = up_block(h)
        return h


class VAEDecoderUpTailTTNN(Module):
    """Decoder up path + tail (post-mid) on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.decoder_up = DecoderUpTTNN(mesh_device, dtype=dtype)
        self.decoder_tail = DecoderTailTTNN(mesh_device, dtype=dtype)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = self.decoder_up(x_bthwc)
        return self.decoder_tail(h)


class VAEDecoderTTNN(Module):
    """Full VAE decoder: conv_in -> mid -> up -> tail on replicated mesh."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.conv_in = ConvInTTNN(mesh_device, dtype=dtype)
        self.mid = MidBlockTTNN(mesh_device, dtype=dtype)
        self.decoder_up = DecoderUpTTNN(mesh_device, dtype=dtype)
        self.decoder_tail = DecoderTailTTNN(mesh_device, dtype=dtype)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        """BTHWC device tensor in -> BTHWC device tensor out."""
        h = self.conv_in(x_bthwc)
        h = self.mid(h)
        h = self.decoder_up(h)
        return self.decoder_tail(h)
