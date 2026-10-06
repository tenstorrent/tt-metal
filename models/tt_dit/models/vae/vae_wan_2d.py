# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKLWan

import ttnn
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.models.vae.vae import (
    VaeContext,
    VaeConv2d,
    VaeDownBlock,
    VaeMidBlock,
    VaeNormDesc,
    VaeNormDescRms,
    VaeRmsNorm,
    VaeUpBlock,
    fold_quant_conv_mean,
)
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache, tensor
from models.tt_dit.utils.substate import pop_substate, rename_substate
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence


class WanDupUp2D(Module):
    """``DupUp3D`` from diffusers, restricted to 2D (no temporal upsample)."""

    def __init__(self, *, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()

        repeats, remainder = divmod(out_channels * factor * factor, in_channels)
        if remainder != 0 or repeats not in (factor, factor * factor):
            msg = (
                f"unsupported channel ratio: in_channels {in_channels}, out_channels {out_channels}, "
                f"factor {factor}"
            )
            raise ValueError(msg)

        self._repeats = repeats
        self._factor = factor

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        f = self._factor

        if self._repeats == f * f:
            return tensor.upsample(x, scale_factor=f)

        bs, h, w, c = x.shape
        x = ttnn.transpose(x, -2, -1)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [bs, h, c // f, f, w])
        x = ttnn.permute(x, (0, 1, 3, 2, 4))
        x = ttnn.reshape(x, [bs, h * f, c // f, w])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.transpose(x, -2, -1)
        x = ttnn.upsample(x, scale_factor=[1, f])
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


class WanAvgDown2D(Module):
    """``AvgDown3D`` from Wan 2.2's VAE on a single frame, halving height and width."""

    def __init__(self, *, in_channels: int, out_channels: int, temporal_factor: int, ctx: VaeContext) -> None:
        super().__init__()

        if out_channels != in_channels * temporal_factor:
            msg = f"unsupported channel ratio: in_channels {in_channels}, out_channels {out_channels}"
            raise ValueError(msg)

        self._temporal_factor = temporal_factor
        self._mask = (
            tensor.from_torch(
                (torch.arange(out_channels) % temporal_factor == temporal_factor - 1).float(), device=ctx.device
            )
            if temporal_factor > 1
            else None
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        b, h, w, c = x.shape
        x = ttnn.avg_pool2d(
            x,
            batch_size=b,
            input_h=h,
            input_w=w,
            channels=c,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
        )
        x = ttnn.reshape(x, [b, h // 2, w // 2, c])

        if self._mask is not None:
            x = ttnn.repeat_interleave(x, self._temporal_factor, dim=3) * self._mask

        return x


class WanUpBlock2D(VaeUpBlock):
    """Wan 2.1's up block (``WanUpBlock``), 2D-only."""

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if self.upsampler is not None:
            rename_substate(state, "upsamplers.0.resample.1", "upsamplers.0.conv")
            pop_substate(state, "upsamplers.0.time_conv")

        super()._prepare_torch_state(state)


class WanResidualUpBlock2D(VaeUpBlock):
    """Wan 2.2's residual up block (``WanResidualUpBlock``), 2D-only."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        upsample: bool,
        norm: VaeNormDesc,
        ctx: VaeContext,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            upsample=upsample,
            norm=norm,
            ctx=ctx,
        )

        self.avg_shortcut = (
            WanDupUp2D(in_channels=in_channels, out_channels=out_channels, factor=2) if upsample else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if self.upsampler is not None:
            rename_substate(state, "upsampler.resample.1", "upsampler.conv")
            pop_substate(state, "upsampler.time_conv")

        super()._prepare_torch_state(state)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.avg_shortcut is None:
            return super().forward(x)
        return super().forward(x) + self.avg_shortcut.forward(x)


class WanResidualDownBlock2D(VaeDownBlock):
    """Wan 2.2's residual down block (``WanResidualDownBlock``), 2D-only."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        downsample: bool,
        temporal_downsample: bool,
        norm: VaeNormDesc,
        ctx: VaeContext,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            downsample=downsample,
            norm=norm,
            ctx=ctx,
        )

        if downsample:
            self.avg_shortcut = WanAvgDown2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temporal_factor=2 if temporal_downsample else 1,
                ctx=ctx,
            )
        else:
            if in_channels != out_channels:
                msg = "without downsampling, the shortcut is the identity and cannot change the channel count"
                raise ValueError(msg)
            self.avg_shortcut = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if self.downsampler is not None:
            rename_substate(state, "downsampler.resample.1", "downsampler.conv")
            pop_substate(state, "downsampler.time_conv")

        super()._prepare_torch_state(state)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        shortcut = self.avg_shortcut.forward(x) if self.avg_shortcut is not None else x
        return super().forward(x) + shortcut


class WanVaeDecoder2D(Module):
    """Wan VAE decoder on a single frame.

    With ``is_residual`` it is the Wan 2.2 decoder, whose up blocks add a shortcut; without it, the
    Wan 2.1 decoder, whose upsamplers halve the channel count.
    """

    def __init__(
        self,
        *,
        base_dim: int,
        decoder_base_dim: int | None,
        z_dim: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        out_channels: int,
        is_residual: bool,
        parallel_config: VaeHWParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        super().__init__()

        ctx = _vae_context(parallel_config, device=device, ccl_manager=ccl_manager)

        dim = decoder_base_dim if decoder_base_dim is not None else base_dim
        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]
        eps = 1e-12

        self.post_quant_conv = Linear(z_dim, z_dim, mesh_device=device)
        self.conv_in = VaeConv2d(z_dim, dims[0], kernel_size=3, padding=1, ctx=ctx)
        self.mid_block = VaeMidBlock(
            num_channels=dims[0],
            norm=VaeNormDescRms(eps=eps),
            ctx=ctx,
        )

        self.up_blocks = ModuleList([])
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            upsample = i != len(dim_mult) - 1
            if is_residual:
                up_block = WanResidualUpBlock2D(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    num_layers=num_res_blocks + 1,
                    upsample=upsample,
                    norm=VaeNormDescRms(eps=eps),
                    ctx=ctx,
                )
            else:
                up_block = WanUpBlock2D(
                    in_channels=in_dim if i == 0 else in_dim // 2,
                    out_channels=out_dim,
                    upsampler_out_channels=out_dim // 2,
                    num_layers=num_res_blocks + 1,
                    upsample=upsample,
                    norm=VaeNormDescRms(eps=eps),
                    ctx=ctx,
                )
            self.up_blocks.append(up_block)

        self.conv_norm_out = VaeRmsNorm(out_dim, eps=eps, ctx=ctx, activation_fn="silu")
        self.conv_out = VaeConv2d(out_dim, out_channels, kernel_size=3, padding=1, ctx=ctx)

        self._ctx = ctx

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")
        rename_substate(state, "decoder", "")

        state["conv_norm_out.gamma"] = state.pop("norm_out.gamma")

        _slice_causal_convs(state)
        _convert_mid_block_attention(state, prefix="mid_block.attentions.0.")

        # ``post_quant_conv`` is a 1x1 conv, applied as a ``Linear``.
        state["post_quant_conv.weight"] = state["post_quant_conv.weight"].squeeze(2, 3)

    def forward(self, z: ttnn.Tensor) -> ttnn.Tensor:
        """Decode a latent into an image.

        Args:
            z: Latent of shape (B, h, w, z_dim), spatially sharded.

        Returns:
            The patchified image of shape (B, H, W * C), replicated, in row-major layout. H and W
            are h and w times 8.
        """
        z = self.post_quant_conv.forward(z)
        z = self.conv_in.forward(z)

        z = self.mid_block.forward(z)

        for block in self.up_blocks:
            z = block.forward(z)

        z = self.conv_norm_out.forward(z)
        z = self.conv_out.forward(z)
        z = ttnn.clamp(z, min=-1.0, max=1.0)

        # The host needs row-major data, and untilizing on the device is cheaper than on the host.
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)

        # A row-major page holds one row of the last dimension, and pages of only C elements make
        # the gather and the read to the host many times slower.
        b, h, w, c = z.shape
        z = ttnn.reshape(z, [b, h, w * c])

        ctx = self._ctx
        if ctx.h_factor > 1:
            z = ctx.ccl_manager.all_gather(z, dim=1, mesh_axis=ctx.h_mesh_axis, use_hyperparams=True)
        if ctx.w_factor > 1:
            z = ctx.ccl_manager.all_gather(z, dim=2, mesh_axis=ctx.w_mesh_axis, use_hyperparams=True)

        return z


class WanVaeEncoder2D(Module):
    """Wan 2.2 (residual) VAE encoder on a single frame, producing the latent mean."""

    def __init__(
        self,
        *,
        base_dim: int,
        z_dim: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        temperal_downsample: Sequence[bool],
        in_channels: int,
        parallel_config: VaeHWParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        super().__init__()

        ctx = _vae_context(parallel_config, device=device, ccl_manager=ccl_manager)

        dims = [base_dim * u for u in [1, *dim_mult]]
        eps = 1e-12

        self.conv_in = VaeConv2d(in_channels, dims[0], kernel_size=3, padding=1, ctx=ctx)

        self.down_blocks = ModuleList([])
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            downsample = i != len(dim_mult) - 1
            self.down_blocks.append(
                WanResidualDownBlock2D(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    num_layers=num_res_blocks,
                    downsample=downsample,
                    temporal_downsample=downsample and temperal_downsample[i],
                    norm=VaeNormDescRms(eps=eps),
                    ctx=ctx,
                )
            )

        self.mid_block = VaeMidBlock(num_channels=dims[-1], norm=VaeNormDescRms(eps=eps), ctx=ctx)

        self.norm_out = VaeRmsNorm(dims[-1], eps=eps, ctx=ctx, activation_fn="silu")
        self.conv_out = VaeConv2d(dims[-1], z_dim, kernel_size=3, padding=1, ctx=ctx)

        self._z_dim = z_dim

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pop_substate(state, "decoder")
        pop_substate(state, "post_quant_conv")
        rename_substate(state, "encoder", "")

        _slice_causal_convs(state)
        _convert_mid_block_attention(state, prefix="mid_block.attentions.0.")

        fold_quant_conv_mean(
            state, conv_out_prefix="conv_out.", quant_conv_prefix="quant_conv.", latent_channels=self._z_dim
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Encode an image into the mean of its latent distribution.

        Args:
            x: Patchified image of shape (B, H, W, C), spatially sharded, in row-major layout.

        Returns:
            The latent mean of shape (B, H / 8, W / 8, z_dim), spatially sharded like the input.
        """
        x = self.conv_in.forward(x)

        for block in self.down_blocks:
            x = block.forward(x)

        x = self.mid_block.forward(x)

        x = self.norm_out.forward(x)
        return self.conv_out.forward(x)


class WanVaeDecoder2DAdapter:
    """Torch-in (BHWC), torch-out (BCHW) decoder for the Wan VAE of a checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_name: str,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        use_torch: bool,
        load_weights: bool = True,
    ) -> None:
        self._name = checkpoint_name
        self._parallel_config = parallel_config
        self._ccl_manager = ccl_manager
        self._device = ccl_manager.mesh_device

        # Eager config-only read; full torch weights are only loaded if needed (use_torch=True or
        # cache.load_model cache miss).
        hf_config = AutoencoderKLWan.load_config(checkpoint_name, subfolder="vae")
        self._hf_config = hf_config
        self._latents_scaling = 1.0 / torch.tensor(hf_config["latents_std"])
        self._latents_shift = torch.tensor(hf_config["latents_mean"])
        self.z_dim: int = hf_config["z_dim"]
        self.patch_size: int = hf_config.get("patch_size") or 1
        # Wan 2.1 style configs, such as Qwen-Image's, omit the keys that default to the Wan 2.1 VAE.
        self.spatial_compression_ratio: int = hf_config.get("scale_factor_spatial", 8)

        if use_torch:
            self._torch_vae = AutoencoderKLWan.from_pretrained(
                checkpoint_name, subfolder="vae", torch_dtype=torch.float32
            )
            self._decoder = None
            self._tracer = None
            self._tt_latents_std = None
            self._tt_latents_mean = None
        else:
            self._torch_vae = None
            self._decoder = self._build_decoder()
            self._tracer = Tracer(self._rescale_and_decode, device=self._device, clone_prep_inputs=False)
            self._tt_latents_std = tensor.from_torch(torch.tensor(hf_config["latents_std"]), device=self._device)
            self._tt_latents_mean = tensor.from_torch(torch.tensor(hf_config["latents_mean"]), device=self._device)

            if load_weights:
                self.reload_weights()

    def is_loaded(self) -> bool:
        return self._torch_vae is not None or (self._decoder is not None and self._decoder.is_loaded())

    def reload_weights(self) -> None:
        if self.is_loaded():
            return

        if self._decoder is None:
            self._decoder = self._build_decoder()
        cache.load_model(
            self._decoder,
            get_torch_state_dict=self._load_torch_state_dict,
            model_name=self._name.split("/")[-1],
            subfolder="vae",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            mesh_device=self._device,
        )
        ttnn.synchronize_device(self._device)

    def deallocate_weights(self) -> None:
        """Drops the decoder, which frees its weights. ``reload_weights`` builds a new one.

        Also releases the trace, which would read the weights from their old addresses after a reload.
        """
        if self._torch_vae is not None or self._decoder is None:
            return

        self._tracer.release_trace()
        self._decoder = None
        ttnn.synchronize_device(self._device)

    def _build_decoder(self) -> WanVaeDecoder2D:
        hf_config = self._hf_config
        return WanVaeDecoder2D(
            base_dim=hf_config["base_dim"],
            decoder_base_dim=hf_config.get("decoder_base_dim"),
            z_dim=hf_config["z_dim"],
            dim_mult=hf_config["dim_mult"],
            num_res_blocks=hf_config["num_res_blocks"],
            out_channels=hf_config.get("out_channels", 3),
            is_residual=hf_config.get("is_residual", False),
            device=self._device,
            parallel_config=self._parallel_config,
            ccl_manager=self._ccl_manager,
        )

    def _load_torch_state_dict(self) -> dict[str, torch.Tensor]:
        torch_vae = AutoencoderKLWan.from_pretrained(self._name, subfolder="vae")
        return torch_vae.state_dict()

    def _rescale_and_decode(self, z: ttnn.Tensor) -> ttnn.Tensor:
        z = z * self._tt_latents_std + self._tt_latents_mean
        return self._decoder.forward(z)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, traced: bool) -> torch.Tensor:
        _, h, w, _ = latents.shape

        if self._torch_vae is not None:
            latents = latents / self._latents_scaling + self._latents_shift
            return self._torch_vae.decode(latents.permute(0, 3, 1, 2).unsqueeze(2)).sample[:, :, 0]

        _check_divisible(self._parallel_config, latents_h=h, latents_w=w)

        tt_latents = tensor.from_torch(
            latents,
            device=self._device,
            layout=ttnn.TILE_LAYOUT,
            mesh_axes=_bhwc_mesh_axes(self._parallel_config),
        )
        tt_out = self._tracer(tt_latents, traced=traced)
        torch_out = tensor.to_torch(tt_out)
        b, out_h, _ = torch_out.shape
        torch_out = torch_out.reshape(b, out_h, w * self.spatial_compression_ratio // self.patch_size, -1)
        return _unpatchify(torch_out, patch_size=self.patch_size)


class WanVaeEncoder2DAdapter:
    """Torch-in (BCHW), torch-out (BHWC) encoder for the Wan 2.2 VAE of a checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_name: str,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        use_torch: bool,
    ) -> None:
        self._name = checkpoint_name
        self._parallel_config = parallel_config
        self._ccl_manager = ccl_manager
        self._device = ccl_manager.mesh_device

        hf_config = AutoencoderKLWan.load_config(checkpoint_name, subfolder="vae")
        self._latents_mean = torch.tensor(hf_config["latents_mean"])
        self._latents_std = torch.tensor(hf_config["latents_std"])
        self.z_dim: int = hf_config["z_dim"]
        self.patch_size: int = hf_config.get("patch_size") or 1
        self.spatial_compression_ratio: int = hf_config["scale_factor_spatial"]

        if use_torch:
            self._torch_vae = AutoencoderKLWan.from_pretrained(
                checkpoint_name, subfolder="vae", torch_dtype=torch.float32
            )
            self._encoder = None
            self._tracer = None
            self._tt_latents_mean = None
            self._tt_latents_scaling = None
        else:
            self._torch_vae = None
            self._encoder = WanVaeEncoder2D(
                base_dim=hf_config["base_dim"],
                z_dim=hf_config["z_dim"],
                dim_mult=hf_config["dim_mult"],
                num_res_blocks=hf_config["num_res_blocks"],
                temperal_downsample=hf_config["temperal_downsample"],
                in_channels=hf_config.get("in_channels", 3),
                device=self._device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            self._tracer = Tracer(self._encode_and_rescale, device=self._device, clone_prep_inputs=False)
            self._tt_latents_mean = tensor.from_torch(self._latents_mean, device=self._device)
            self._tt_latents_scaling = tensor.from_torch(1.0 / self._latents_std, device=self._device)

            cache.load_model(
                self._encoder,
                get_torch_state_dict=self._load_torch_state_dict,
                model_name=self._name.split("/")[-1],
                subfolder="vae_encoder",
                parallel_config=self._parallel_config,
                mesh_shape=tuple(self._device.shape),
                mesh_device=self._device,
            )

    def _load_torch_state_dict(self) -> dict[str, torch.Tensor]:
        torch_vae = AutoencoderKLWan.from_pretrained(self._name, subfolder="vae")
        return torch_vae.state_dict()

    def _encode_and_rescale(self, x: ttnn.Tensor) -> ttnn.Tensor:
        z = self._encoder.forward(x)
        return (z - self._tt_latents_mean) * self._tt_latents_scaling

    @torch.no_grad()
    def encode(self, images: torch.Tensor, *, traced: bool) -> torch.Tensor:
        """Encode images with values in [-1, 1] to normalized latents of shape (B, H, W, C)."""
        _, _, h, w = images.shape

        if h % self.spatial_compression_ratio != 0 or w % self.spatial_compression_ratio != 0:
            msg = f"image size {h}x{w} not divisible by {self.spatial_compression_ratio}"
            raise ValueError(msg)

        if self._torch_vae is not None:
            latents = self._torch_vae.encode(images.unsqueeze(2)).latent_dist.mean[:, :, 0]
            latents = latents.permute(0, 2, 3, 1)
            return (latents - self._latents_mean) / self._latents_std

        _check_divisible(
            self._parallel_config,
            latents_h=h // self.spatial_compression_ratio,
            latents_w=w // self.spatial_compression_ratio,
        )
        mesh_axes = _bhwc_mesh_axes(self._parallel_config)

        tt_images = tensor.from_torch(
            _patchify(images, patch_size=self.patch_size),
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=mesh_axes,
        )
        tt_out = self._tracer(tt_images, traced=traced)
        return tensor.to_torch(tt_out, mesh_axes=mesh_axes)


def _vae_context(
    parallel_config: VaeHWParallelConfig, *, device: ttnn.MeshDevice, ccl_manager: CCLManager | None
) -> VaeContext:
    hp = parallel_config.height_parallel
    wp = parallel_config.width_parallel
    return VaeContext(
        tp_axis=None,
        h_mesh_axis=hp.mesh_axis,
        h_factor=hp.factor,
        w_mesh_axis=wp.mesh_axis,
        w_factor=wp.factor,
        device=device,
        ccl_manager=ccl_manager,
    )


def _check_divisible(parallel_config: VaeHWParallelConfig, *, latents_h: int, latents_w: int) -> None:
    hp = parallel_config.height_parallel
    wp = parallel_config.width_parallel
    if latents_h % hp.factor != 0:
        msg = f"latent height {latents_h} not divisible by {hp.factor}"
        raise ValueError(msg)
    if latents_w % wp.factor != 0:
        msg = f"latent width {latents_w} not divisible by {wp.factor}"
        raise ValueError(msg)


def _bhwc_mesh_axes(parallel_config: VaeHWParallelConfig) -> list[int | None]:
    return [None, parallel_config.height_parallel.mesh_axis, parallel_config.width_parallel.mesh_axis, None]


def _patchify(x: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    """``(B, C, H, W) -> (B, H/P, W/P, C * P * P)``."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 5, 3)
    return x.reshape(b, h // patch_size, w // patch_size, c * patch_size * patch_size)


def _unpatchify(x: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    """``(B, H/P, W/P, C * P * P) -> (B, C, H, W)``."""
    b, h, w, cpp = x.shape
    c = cpp // (patch_size * patch_size)
    x = x.reshape(b, h, w, c, patch_size, patch_size)
    x = x.permute(0, 3, 1, 5, 2, 4)
    return x.reshape(b, c, h * patch_size, w * patch_size)


def _slice_causal_convs(state: dict[str, torch.Tensor]) -> None:
    """Reduce the 3D causal convs to 2D."""
    for key, value in state.items():
        if value.ndim == 5:
            state[key] = value[:, :, -1, :, :]


def _convert_mid_block_attention(state: dict[str, torch.Tensor], *, prefix: str) -> None:
    """Convert the mid-block attention under ``prefix`` to the layout ``VaeAttention`` expects."""
    for suffix in ("weight", "bias"):
        key = f"{prefix}proj.{suffix}"
        if key in state:
            state[f"{prefix}to_out.0.{suffix}"] = state.pop(key)

    if f"{prefix}to_qkv.weight" in state:
        (
            state[f"{prefix}to_q.weight"],
            state[f"{prefix}to_k.weight"],
            state[f"{prefix}to_v.weight"],
        ) = (
            state.pop(f"{prefix}to_qkv.weight").squeeze(2, 3).chunk(3)
        )
        (
            state[f"{prefix}to_q.bias"],
            state[f"{prefix}to_k.bias"],
            state[f"{prefix}to_v.bias"],
        ) = state.pop(
            f"{prefix}to_qkv.bias"
        ).chunk(3)
    if f"{prefix}to_out.0.weight" in state:
        w = state[f"{prefix}to_out.0.weight"]
        if w.ndim == 4:
            state[f"{prefix}to_out.0.weight"] = w.squeeze(2, 3)
