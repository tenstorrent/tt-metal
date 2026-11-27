# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch

import ttnn

from ...layers.conv2d import configure_conv2d, warm_dram_slice_flatten, warm_fixed_slice_flatten
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.substate import pop_substate, rename_substate
from ...utils.tracing import traced_function
from .vae import (
    VaeContext,
    VaeConv2d,
    VaeDownBlock,
    VaeMidBlock,
    VaeNormDescGroup,
    VaeUpBlock,
    _norm,
    fold_quant_conv_mean,
    warm_token_gather,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L271
class Flux2VaeDecoder(Module):
    _PATCH_SIZE = 2

    # The output size per chip (height, width) of the fixed 2D slices of the convs at each resolution, from the
    # latent's (1/8 of the image) to the image's, or None for conv2d's own slicing, which adds program-cache entries
    # at every new image size. See Conv2d.fixed_slice_hw.
    fixed_slice_hw_per_level: tuple[tuple[int, int], ...] | None = ((32, 32), (32, 64), (64, 64), (128, 128))

    def __init__(
        self,
        *,
        out_channels: int = 3,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        z_channels: int = 32,
        parallel_config: Flux2VaeParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        use_conv3d: bool = False,
    ) -> None:
        super().__init__()

        ctx = VaeContext(
            tp_axis=parallel_config.tp_parallel.mesh_axis if parallel_config.tp_parallel is not None else None,
            device=device,
            ccl_manager=ccl_manager,
            h_mesh_axis=parallel_config.h_parallel.mesh_axis if parallel_config.h_parallel is not None else None,
            h_factor=parallel_config.h_parallel.factor if parallel_config.h_parallel is not None else 1,
            w_mesh_axis=parallel_config.w_parallel.mesh_axis if parallel_config.w_parallel is not None else None,
            w_factor=parallel_config.w_parallel.factor if parallel_config.w_parallel is not None else 1,
            use_conv3d=use_conv3d,
            # Layout conversions, the unpatchify permute and row-major upsamples serve every image size after
            # warming one.
            runtime_shapes=True,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        channel_counts = [block_out_channels[-1], *block_out_channels[::-1]]
        eps = 1e-6

        self.post_quant_conv = Linear(z_channels, z_channels, mesh_device=ctx.device)
        self.conv_in = VaeConv2d(z_channels, channel_counts[0], kernel_size=3, padding=1, ctx=ctx)

        self.mid_block = VaeMidBlock(
            num_channels=channel_counts[0],
            norm=VaeNormDescGroup(num_groups=32, eps=eps),
            ctx=ctx,
        )

        self.up_blocks = ModuleList(
            VaeUpBlock(
                in_channels=ch_in,
                out_channels=ch_out,
                upsample=i != len(channel_counts) - 2,
                num_layers=layers_per_block + 1,
                norm=VaeNormDescGroup(num_groups=32, eps=eps),
                ctx=ctx,
            )
            for i, (ch_in, ch_out) in enumerate(itertools.pairwise(channel_counts))
        )

        self.conv_norm_out = _norm(
            VaeNormDescGroup(num_groups=32, eps=1e-6),
            num_channels=channel_counts[-1],
            ctx=ctx,
            activation_fn="silu",
        )
        self.conv_out = VaeConv2d(
            channel_counts[-1], out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx
        )

        bn_size = self._PATCH_SIZE**2 * z_channels
        self.bn_running_mean = Parameter(total_shape=[bn_size], device=ctx.device)
        self.bn_running_var = Parameter(total_shape=[bn_size], device=ctx.device)
        self.bn_eps = 1e-4
        # The latent normalization's mean and standard deviation on the host, read once.
        self._bn_host: tuple[torch.Tensor, torch.Tensor] | None = None

        self._ctx = ctx

        # The pipeline traces the decoder at every image size, and each keeps its own conv config
        # tensors, which would not fit in L1_SMALL.
        configure_conv2d(self, config_tensors_in_dram=True)
        # Fixed shard schemes, so each conv prepares its weights once instead of per image size. On a
        # T3K (2x4) from 256x256 to 2048x1024 these are the schemes conv2d picks on its own, except
        # conv_out at 256x256, which it block-shards (21 ms faster there; block sharding costs 5-6% of
        # the decode from 512x512 up).
        configure_conv2d(self, shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED)
        configure_conv2d(self.conv_out, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
        # Always DRAM-slice the convs that conv2d leaves unsliced at 1024x1024 but slices at larger
        # sizes: the two with 512 input channels at the second-largest level, and conv_out. Their
        # slicing kernels then compile at the first size instead of at the first larger one. The
        # other convs that get sliced share their slicing kernels with convs already sliced at
        # 1024x1024. The areas are from a T3K (2x4) from 1024x1024 to 2048x2048. The largest slice
        # conv2d picked for conv_out was 512 rows by 256 columns, at 90% of the free L1; with the halo
        # counted on 64 cores (see Conv2d._width_slices) that is (512 + 64) * 256. On one chip, of
        # conv_out slices (height x width) 512x256, 128x704, 176x576 and 160x640 fit, and 128x1024
        # (the same 131,072 pixels as 512x256) and 176x704 did not, which this cap reproduces. It is
        # not scaled by the core count, so on Blackhole's 110 cores it slices more than it needs. For
        # the other two it covered 192x192 once, but 128x296 did not fit in one call, so their limit
        # is the next largest, 128x256, at 87%.
        configure_conv2d(self.up_blocks[1].upsampler, dram_slice_area=128 * 256)
        configure_conv2d(self.up_blocks[2].resnets[0].conv1, dram_slice_area=128 * 256)
        configure_conv2d(self.conv_out, dram_slice_area=(512 + 64) * 256)
        # The sliced convs end with a tiled reshape at every width per chip that is not whole tiles, which the
        # warm size (1024x1024) does not have at any level.
        warm_dram_slice_flatten(ctx.device)
        # Fixed 2D slices for every conv, which supersede the slicing above: each conv then runs as slices of one
        # output size per level at every image size, and adds no program-cache entries at a new size. Their
        # flatten's entries are built here, one per channel count, as the warm size never runs it.
        if self.fixed_slice_hw_per_level is not None:
            for level, modules in enumerate(self._conv_levels()):
                for module in modules:
                    configure_conv2d(module, fixed_slice_hw=self.fixed_slice_hw_per_level[level])
            warm_fixed_slice_flatten(ctx.device, [*channel_counts, out_channels])
        # So does the mid block's token gather where the token rows per chip are not whole tiles.
        warm_token_gather(ctx, channel_counts[0])

    def _conv_levels(self) -> list[list[Module]]:
        """The modules holding the convs at each resolution, from the latent's to the image's: each up block's
        resnets run at the resolution its upsampler leaves, which the next block's resnets share."""
        levels: list[list[Module]] = [[self.conv_in, self.mid_block]]
        for block in self.up_blocks:
            levels[-1].append(block.resnets)
            if block.upsampler is not None:
                levels.append([block.upsampler])
        levels[-1].append(self.conv_out)
        return levels

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # remove encoder state
        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")

        if "post_quant_conv.weight" in state:
            state["post_quant_conv.weight"] = state["post_quant_conv.weight"].squeeze(2, 3)

        if "bn.running_mean" in state:
            state["bn_running_mean"] = state.pop("bn.running_mean")
        if "bn.running_var" in state:
            state["bn_running_var"] = state.pop("bn.running_var")
        state.pop("bn.num_batches_tracked", None)
        # New statistics replace the ones read to the host.
        self._bn_host = None

        rename_substate(state, "decoder", "")

    def preprocess_and_unpatchify(self, z: torch.Tensor, *, height: int, width: int) -> ttnn.Tensor:
        """Undoes the latent normalization and the patchify of all tokens, [N, (H / P) * (W / P), C * P * P], on
        the host, and uploads the result split over the H and W mesh axes, [N, H / h_factor, W / w_factor, C] per
        device, row major. On the device this was a layout conversion, a permute and a mesh partition per axis, a
        program per image size each, for about 1 MB at 1024x1024; the transfers cost a few ms instead."""
        p = self._PATCH_SIZE
        assert height % p == 0 and width % p == 0, f"{height}x{width} is not whole patches"
        assert height % self._ctx.h_factor == 0, f"height {height} does not split over {self._ctx.h_factor} devices"
        assert width % self._ctx.w_factor == 0, f"width {width} does not split over {self._ctx.w_factor} devices"
        if self._bn_host is None:
            replicated = (None,)
            mean = tensor.to_torch(self.bn_running_mean.data, mesh_axes=replicated).float()
            var = tensor.to_torch(self.bn_running_var.data, mesh_axes=replicated).float()
            self._bn_host = (mean, torch.sqrt(var + self.bn_eps))
        mean, std = self._bn_host
        z = z.float() * std + mean
        z = z.reshape(z.shape[0], height // p, width // p, -1, p, p).permute(0, 1, 4, 2, 5, 3)
        # In bfloat16 already, since from_torch converts other dtypes on the device, at a program per shape.
        z = z.reshape(z.shape[0], height, width, -1).to(torch.bfloat16)
        # Row major: the decoder flattens it before tilizing (see forward).
        return tensor.from_torch(
            z,
            device=self._ctx.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=(None, self._ctx.h_mesh_axis, self._ctx.w_mesh_axis, None),
        )

    @traced_function(device=lambda self: self._ctx.device, clone_prep_inputs=False)
    def forward(self, z: ttnn.Tensor, /) -> ttnn.Tensor:
        # Input arrives from preprocess_and_unpatchify in row major, [N, H, W, C] split over the H and
        # W axes where the VAE has them. The tiled activations stay flat, [N, 1, H * W, C], and are
        # made 4D only in row major (see VaeConv2d.forward): a tiled reshape between the two moves data
        # whenever the local W is not whole tiles.
        n, h, w, c = z.shape
        hw = (h, w)
        # Tilizing the flat tensor pads only at its end, not per row of latents.
        z = ttnn.to_layout(ttnn.reshape(z, (n, 1, h * w, c)), ttnn.TILE_LAYOUT, runtime_shapes=self._ctx.runtime_shapes)
        # post_quant_conv is a per-position Linear, and the conv pyramid below exchanges halos.
        z = self.post_quant_conv.forward(z)

        z = self.conv_in.forward(z, hw)
        z = self.mid_block.forward(z, hw)

        for block in self.up_blocks:
            z = block.forward(z, hw)
            hw = block.output_hw(hw)

        z = self.conv_norm_out.forward(z)

        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            z = self._ctx.ccl_manager.all_gather(z, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)

        # The image stays split over the H and W axes: the host assembles it (see output_concat_dims),
        # which saves gathering it on the device, at a program per image size. It stays flat too, as
        # [N, 1, H * W, C] tiled; flat_to_row_major with output_hw makes it [N, H, W, C].
        return self.conv_out.forward(z, hw)

    @property
    def runtime_shapes(self) -> bool:
        """Whether the decoder's layout conversions, unpatchify permute and row-major upsamples use one program for
        every image size; the pipeline's float_to_uint8 follows it."""
        return self._ctx.runtime_shapes

    def output_hw(self, hw: tuple[int, int]) -> tuple[int, int]:
        """The local (H, W) of the image decoded from latents of local size `hw`."""
        for block in self.up_blocks:
            hw = block.output_hw(hw)
        return hw

    def output_concat_dims(self) -> list[int | None]:
        """Per mesh axis, the dimension of the [N, H, W, C] output its shards are split along, or None
        where the output is not split; the concat_dims of fast_device_to_host."""
        dims: list[int | None] = [None] * len(tuple(self._ctx.device.shape))
        if self._ctx.h_factor > 1 and self._ctx.h_mesh_axis is not None:
            dims[self._ctx.h_mesh_axis] = 1
        if self._ctx.w_factor > 1 and self._ctx.w_mesh_axis is not None:
            dims[self._ctx.w_mesh_axis] = 2
        return dims


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L189
class Flux2VaeEncoder(Module):
    """The FLUX.2 VAE encoder, producing the mean of the latent distribution."""

    # The output size per chip (height, width) of the fixed 2D slices of the convs at each resolution, from the
    # image's to the latent's (1/8 of the image), or None for conv2d's own slicing; the decoder's, in the order of the
    # resolutions. See Conv2d.fixed_slice_hw.
    fixed_slice_hw_per_level: tuple[tuple[int, int], ...] | None = ((128, 128), (64, 64), (32, 64), (32, 32))

    def __init__(
        self,
        *,
        in_channels: int = 3,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        z_channels: int = 32,
        parallel_config: Flux2VaeParallelConfig,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        super().__init__()

        ctx = VaeContext(
            tp_axis=parallel_config.tp_parallel.mesh_axis if parallel_config.tp_parallel is not None else None,
            device=device,
            ccl_manager=ccl_manager,
            h_mesh_axis=parallel_config.h_parallel.mesh_axis if parallel_config.h_parallel is not None else None,
            h_factor=parallel_config.h_parallel.factor if parallel_config.h_parallel is not None else 1,
            w_mesh_axis=parallel_config.w_parallel.mesh_axis if parallel_config.w_parallel is not None else None,
            w_factor=parallel_config.w_parallel.factor if parallel_config.w_parallel is not None else 1,
            # As in the decoder: the layout conversions and halo exchanges serve every image size after warming one.
            runtime_shapes=True,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        channel_counts = [block_out_channels[0], *block_out_channels]
        norm = VaeNormDescGroup(num_groups=32, eps=1e-6)

        self.conv_in = VaeConv2d(in_channels, channel_counts[0], kernel_size=3, padding=1, ctx=ctx)

        self.down_blocks = ModuleList(
            VaeDownBlock(
                in_channels=ch_in,
                out_channels=ch_out,
                num_layers=layers_per_block,
                downsample=i != len(channel_counts) - 2,
                norm=norm,
                ctx=ctx,
            )
            for i, (ch_in, ch_out) in enumerate(itertools.pairwise(channel_counts))
        )

        self.mid_block = VaeMidBlock(num_channels=channel_counts[-1], norm=norm, ctx=ctx)

        self.conv_norm_out = _norm(norm, num_channels=channel_counts[-1], ctx=ctx, activation_fn="silu")
        # The quant_conv mean half is folded into conv_out, which then has z_channels outputs, too few
        # to split across devices.
        self.conv_out = VaeConv2d(
            channel_counts[-1], z_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx
        )

        self._z_channels = z_channels
        self._ctx = ctx

        configure_conv2d(self, config_tensors_in_dram=True)
        # A fixed shard scheme, so each conv prepares its weights once instead of per image size. On a
        # T3K (2x4) for reference sizes from 256x256 to 1024x1024, square and not, these convs are
        # block-sharded by conv2d on its own. conv_in is left to conv2d: its 3 input channels are not
        # a whole tile, so without fixed slices it prepares its weights per size anyway.
        for module in (self.down_blocks, self.mid_block, self.conv_out):
            configure_conv2d(module, shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED)
        # Fixed 2D slices for every conv, as in the decoder, so the convs add no program-cache entries at a new image
        # size, and the warm-ups of what the warm size does not run: the slices' flatten, and the mid block's token
        # gather where the token rows per chip are not whole tiles.
        if self.fixed_slice_hw_per_level is not None:
            for level, modules in enumerate(self._conv_levels()):
                for module in modules:
                    configure_conv2d(module, fixed_slice_hw=self.fixed_slice_hw_per_level[level])
            warm_fixed_slice_flatten(ctx.device, [*channel_counts, z_channels])
        warm_token_gather(ctx, channel_counts[-1])

    def _conv_levels(self) -> list[list[Module]]:
        """The modules holding the convs at each resolution, from the image's to the latent's: each down block's
        resnets run at the resolution the previous block's downsampler leaves, and a downsampler's conv at the one it
        makes."""
        levels: list[list[Module]] = [[self.conv_in]]
        for block in self.down_blocks:
            levels[-1].append(block.resnets)
            if block.downsampler is not None:
                levels.append([block.downsampler])
        levels[-1].extend([self.mid_block, self.conv_out])
        return levels

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pop_substate(state, "decoder")
        pop_substate(state, "post_quant_conv")
        pop_substate(state, "bn")
        rename_substate(state, "encoder", "")

        for block in range(len(self.down_blocks)):
            rename_substate(state, f"down_blocks.{block}.downsamplers.0", f"down_blocks.{block}.downsampler")

        fold_quant_conv_mean(
            state, conv_out_prefix="conv_out.", quant_conv_prefix="quant_conv.", latent_channels=self._z_channels
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Encode an image into the mean of its latent distribution.

        Args:
            x: Image of shape (B, H, W, C) with values in [-1, 1], spatially sharded, in row-major
                layout.

        Returns:
            The latent mean, flat and tiled as (B, 1, H / 8 * W / 8, z_channels), split over the H and
            W axes like the input; flat_to_row_major with output_hw makes it (B, H / 8, W / 8,
            z_channels).
        """
        # The tiled activations stay flat, as in Flux2VaeDecoder.forward.
        _, h, w, _ = x.shape
        hw = (h, w)
        x = self.conv_in.forward(x, hw)

        for block in self.down_blocks:
            x = block.forward(x, hw)
            hw = block.output_hw(hw)

        x = self.mid_block.forward(x, hw)

        x = self.conv_norm_out.forward(x)

        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            x = self._ctx.ccl_manager.all_gather(x, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)

        # The latents stay split over the H and W axes for the host to assemble (see output_concat_dims),
        # which on a single host saves gathering them on the device, at a program per image size.
        return self.conv_out.forward(x, hw)

    @property
    def runtime_shapes(self) -> bool:
        """Whether the encoder's layout conversions and halo exchanges use one program for every image size; the
        pipeline's flat_to_row_major of the latents follows it."""
        return self._ctx.runtime_shapes

    def output_hw(self, hw: tuple[int, int]) -> tuple[int, int]:
        """The local (H, W) of the latents of an image of local size `hw`."""
        for block in self.down_blocks:
            hw = block.output_hw(hw)
        return hw

    def output_concat_dims(self) -> list[int | None]:
        """Per mesh axis, the dimension of the [N, H, W, C] output its shards are split along, or None
        where the output is not split; the concat_dims of fast_device_to_host."""
        dims: list[int | None] = [None] * len(tuple(self._ctx.device.shape))
        if self._ctx.h_factor > 1 and self._ctx.h_mesh_axis is not None:
            dims[self._ctx.h_mesh_axis] = 1
        if self._ctx.w_factor > 1 and self._ctx.w_mesh_axis is not None:
            dims[self._ctx.w_mesh_axis] = 2
        return dims
