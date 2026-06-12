# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 VAE decoder — TTNN implementation (single file)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import ttnn
from einops import rearrange
from models.common.utility_functions import is_blackhole
from models.experimental.hunyuan_image_3_0.ref.ref_vae_decoder import (
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
    load_conv_in,
    load_decoder_tail,
    load_decoder_up,
    load_mid,
)
from models.tt_dit.layers.audio_ops import prepare_conv3d_weight_state
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import GroupNorm3D
from models.tt_dit.utils.conv3d import (
    _BLOCKINGS,
    _DEFAULT_BLOCKINGS,
    _ntuple,
    aligned_channels,
    get_conv3d_config,
    register_conv3d_configs,
)

# Replicated 1x4 mesh — conservative fallbacks for Hunyuan VAE decoder shapes.
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


def _promote_conv3d_fallback_to_exact(
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

        _promote_conv3d_fallback_to_exact(
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

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
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

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        assert (
            x_bthwc.layout == ttnn.ROW_MAJOR_LAYOUT
        ), f"HunyuanSymmetricConv3d expects ROW_MAJOR, got {x_bthwc.layout}"
        return ttnn.experimental.conv3d(
            input_tensor=x_bthwc,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )


def bcthw_to_bthwc(z_bcthw: torch.Tensor) -> torch.Tensor:
    return z_bcthw.float().permute(0, 2, 3, 4, 1).contiguous()


def bthwc_to_bcthw(x_bthwc: torch.Tensor) -> torch.Tensor:
    return x_bthwc.permute(0, 4, 1, 2, 3).contiguous()


def upload_bthwc(
    mesh_device: ttnn.MeshDevice,
    tensor_bthwc: torch.Tensor,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    host = tensor_bthwc.bfloat16() if dtype == ttnn.bfloat16 else tensor_bthwc.float()
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def download_bthwc(mesh_device: ttnn.MeshDevice, tensor_bthwc: ttnn.Tensor) -> torch.Tensor:
    out_host = ttnn.to_torch(
        ttnn.from_device(tensor_bthwc),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    num_devices = mesh_device.get_num_devices()
    return out_host[: out_host.shape[0] // num_devices]


def upload_bcthw(
    mesh_device: ttnn.MeshDevice,
    z_bcthw: torch.Tensor,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    return upload_bthwc(mesh_device, bcthw_to_bthwc(z_bcthw), dtype=dtype)


def download_bcthw(mesh_device: ttnn.MeshDevice, tensor_bthwc: ttnn.Tensor) -> torch.Tensor:
    return bthwc_to_bcthw(download_bthwc(mesh_device, tensor_bthwc).float())


def dcae_rearrange_up_bcthw(x_bcthw: torch.Tensor, out_channels: int, r1: int) -> torch.Tensor:
    return rearrange(x_bcthw, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2, c=out_channels)


def dcae_shortcut_bcthw(x_bcthw: torch.Tensor, repeats: int, out_channels: int, r1: int) -> torch.Tensor:
    shortcut = x_bcthw.repeat_interleave(repeats, dim=1)
    return dcae_rearrange_up_bcthw(shortcut, out_channels, r1)


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

        pt_conv = load_conv_in(dtype=torch.float32)
        self.conv.load_torch_state_dict(pt_conv.conv.state_dict())
        del pt_conv

    def forward(self, z_bcthw: torch.Tensor) -> torch.Tensor:
        z_bthwc = bcthw_to_bthwc(z_bcthw)
        shortcut_bthwc = z_bthwc.repeat_interleave(self.repeats, dim=-1)

        x = upload_bthwc(self.mesh_device, z_bthwc, dtype=self.dtype)
        conv_out = self.conv(x)
        ttnn.deallocate(x, force=False)

        shortcut = upload_bthwc(self.mesh_device, shortcut_bthwc, dtype=self.dtype)
        out_bthwc = ttnn.add(conv_out, shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(conv_out, force=False)
        ttnn.deallocate(shortcut, force=False)

        result = download_bcthw(self.mesh_device, out_bthwc)
        ttnn.deallocate(out_bthwc, force=False)
        return result


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

    def load_from_torch(self, torch_block) -> None:
        self.norm1.load_torch_state_dict(torch_block.norm1.state_dict())
        self.norm2.load_torch_state_dict(torch_block.norm2.state_dict())
        self.conv1.load_torch_state_dict(torch_block.conv1.state_dict())
        self.conv2.load_torch_state_dict(torch_block.conv2.state_dict())
        if self.nin_shortcut is not None:
            self.nin_shortcut.load_torch_state_dict(torch_block.nin_shortcut.state_dict())

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        residual = self.nin_shortcut(x_bthwc) if self.nin_shortcut is not None else x_bthwc

        h = self.norm1(x_bthwc)
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self.conv1(h)

        h = self.norm2(h)
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

    def load_from_torch(self, torch_block) -> None:
        self.norm.load_torch_state_dict(torch_block.norm.state_dict())
        self.q.load_torch_state_dict(torch_block.q.state_dict())
        self.k.load_torch_state_dict(torch_block.k.state_dict())
        self.v.load_torch_state_dict(torch_block.v.state_dict())
        self.proj_out.load_torch_state_dict(torch_block.proj_out.state_dict())

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

        pt_mid = load_mid(dtype=torch.float32)
        self.block_1.load_from_torch(pt_mid.block_1)
        self.attn_1.load_from_torch(pt_mid.attn_1)
        self.block_2.load_from_torch(pt_mid.block_2)
        del pt_mid

    def forward_device(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        x = self.block_1(x_bthwc)
        x = self.attn_1(x)
        return self.block_2(x)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        x = upload_bcthw(self.mesh_device, x_bcthw, dtype=self.dtype)
        out = self.forward_device(x)
        ttnn.deallocate(x, force=False)
        result = download_bcthw(self.mesh_device, out)
        ttnn.deallocate(out, force=False)
        return result


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

    def load_from_torch(self, torch_upsample) -> None:
        self.conv.load_torch_state_dict(torch_upsample.conv.state_dict())

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        x = upload_bcthw(self.mesh_device, x_bcthw, dtype=self.dtype)
        conv_out = self.conv(x)
        ttnn.deallocate(x, force=False)

        conv_bcthw = download_bcthw(self.mesh_device, conv_out)
        ttnn.deallocate(conv_out, force=False)

        h_up = dcae_rearrange_up_bcthw(conv_bcthw, self.out_channels, self.r1)
        shortcut_up = dcae_shortcut_bcthw(x_bcthw, self.repeats, self.out_channels, self.r1)
        return h_up + shortcut_up


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

    def load_from_torch(self, torch_block) -> None:
        for tt_block, pt_block in zip(self.blocks, torch_block.block):
            tt_block.load_from_torch(pt_block)
        if self.upsample is not None:
            self.upsample.load_from_torch(torch_block.upsample)

    def forward_device(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = x_bthwc
        for block in self.blocks:
            h = block(h)
        if self.upsample is None:
            return h

        h_bcthw = download_bcthw(self.mesh_device, h)
        ttnn.deallocate(h, force=False)
        out_bcthw = self.upsample(h_bcthw)
        return upload_bcthw(self.mesh_device, out_bcthw, dtype=self.dtype)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        x = upload_bcthw(self.mesh_device, x_bcthw, dtype=self.dtype)
        out = self.forward_device(x)
        ttnn.deallocate(x, force=False)
        result = download_bcthw(self.mesh_device, out)
        ttnn.deallocate(out, force=False)
        return result


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

    def load_from_torch(self, torch_norm_out) -> None:
        self.norm.load_torch_state_dict(torch_norm_out.norm.state_dict())

    def forward_device(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = self.norm(x_bthwc)
        return ttnn.mul(
            h, ttnn.sigmoid(h, memory_config=ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        x = upload_bcthw(self.mesh_device, x_bcthw, dtype=self.dtype)
        out = self.forward_device(x)
        ttnn.deallocate(x, force=False)
        result = download_bcthw(self.mesh_device, out)
        ttnn.deallocate(out, force=False)
        return result


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

    def load_from_torch(self, torch_conv_out) -> None:
        self.conv.load_torch_state_dict(torch_conv_out.conv.state_dict())

    def forward_device(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv(x_bthwc)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        x = upload_bcthw(self.mesh_device, x_bcthw, dtype=self.dtype)
        out = self.forward_device(x)
        ttnn.deallocate(x, force=False)
        result = download_bcthw(self.mesh_device, out)
        ttnn.deallocate(out, force=False)
        return result


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

        pt_tail = load_decoder_tail(dtype=torch.float32)
        self.norm_out.load_from_torch(pt_tail.norm_out)
        self.conv_out.load_from_torch(pt_tail.conv_out)
        del pt_tail

    def forward_device(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        h = self.norm_out.forward_device(x_bthwc)
        return self.conv_out.forward_device(h)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        x = upload_bcthw(self.mesh_device, x_bcthw, dtype=self.dtype)
        out = self.forward_device(x)
        ttnn.deallocate(x, force=False)
        result = download_bcthw(self.mesh_device, out)
        ttnn.deallocate(out, force=False)
        return result


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

        pt_up = load_decoder_up(dtype=torch.float32)
        for tt_block, pt_block in zip(self.up_blocks, pt_up.up):
            tt_block.load_from_torch(pt_block)
        del pt_up

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        h = x_bcthw
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

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        h = self.decoder_up(x_bcthw)
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

    def forward(self, z_bcthw: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z_bcthw)
        h = self.mid(h)
        h = self.decoder_up(h)
        return self.decoder_tail(h)
