# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import ttnn

from ...layers.conv2d import configure_conv2d
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
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
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


# https://github.com/black-forest-labs/flux2/blob/6bb103559da75b67d75bf77cebba0027ba412ebc/src/flux2/autoencoder.py#L271
class Flux2VaeDecoder(Module):
    _PATCH_SIZE = 2

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

        rename_substate(state, "decoder", "")

    def _inv_normalize(self, z: ttnn.Tensor) -> ttnn.Tensor:
        s = ttnn.sqrt(self.bn_running_var.data + self.bn_eps)
        m = self.bn_running_mean.data
        return z * s + m

    def preprocess_and_unpatchify(
        self, z: ttnn.Tensor, *, height: int, width: int, split_height: bool = False
    ) -> ttnn.Tensor:
        # N, (H / P) * (W / P), C * P * P -> N, H/h_factor, W/w_factor, C
        # The input is H-sharded in bands of whole token rows (H/h_factor latent rows per device), or
        # replicated with split_height set, in which case H is split here. Splitting after
        # unpatchifying needs only whole latent rows per device, not whole token rows. W is always
        # split here, since H and W are interleaved in the patchified tokens, and before tilizing:
        # W is a tiled dim, and a tiled split needs each device's share to be whole tiles.
        p = self._PATCH_SIZE
        h_factor = self._ctx.h_factor
        assert height % p == 0, f"height {height} must be divisible by patch {p}"
        if split_height:
            assert height % h_factor == 0, f"height {height} must be divisible by h_factor {h_factor}"
            height_in = height
        else:
            assert height % (p * h_factor) == 0, f"height {height} must be divisible by patch {p} * h_factor {h_factor}"
            height_in = height // h_factor
        assert width % p == 0, f"width {width} must be divisible by patch {p}"
        assert width % self._ctx.w_factor == 0, f"width {width} must be divisible by w_factor {self._ctx.w_factor}"
        z = self._inv_normalize(z)
        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = z.reshape([z.shape[0], height_in // p, width // p, -1, p, p])
        z = ttnn.permute(z, [0, 1, 4, 2, 5, 3])
        z = z.reshape([z.shape[0], height_in, width, -1])
        if split_height and h_factor > 1:
            z = ttnn.mesh_partition(z, dim=1, cluster_axis=self._ctx.h_mesh_axis)
        if self._ctx.w_factor > 1:
            z = ttnn.mesh_partition(z, dim=2, cluster_axis=self._ctx.w_mesh_axis)
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        return z

    @traced_function(device=lambda self: self._ctx.device, clone_prep_inputs=False)
    def forward(self, z: ttnn.Tensor, /) -> ttnn.Tensor:
        # Input arrives from preprocess_and_unpatchify, split over the H and W axes where the VAE has
        # them. post_quant_conv is a per-position Linear, and the conv pyramid below exchanges halos.
        z = self.post_quant_conv.forward(z)

        z = self.conv_in.forward(z)
        z = self.mid_block.forward(z)

        for block in self.up_blocks:
            z = block.forward(z)

        z = self.conv_norm_out.forward(z)

        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            z = self._ctx.ccl_manager.all_gather(z, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)

        # The image stays split over the H and W axes: the host assembles it (see output_concat_dims),
        # which saves gathering it on the device, at a program per image size.
        return self.conv_out.forward(z)

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
        # a whole tile, so it prepares its weights per size anyway.
        for module in (self.down_blocks, self.mid_block, self.conv_out):
            configure_conv2d(module, shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED)

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
            The latent mean of shape (B, H / 8, W / 8, z_channels), split over the H and W axes like
            the input.
        """
        x = self.conv_in.forward(x)

        for block in self.down_blocks:
            x = block.forward(x)

        x = self.mid_block.forward(x)

        x = self.conv_norm_out.forward(x)

        if self._ctx.ccl_manager is not None and self._ctx.tp_axis is not None:
            x = self._ctx.ccl_manager.all_gather(x, dim=-1, mesh_axis=self._ctx.tp_axis, use_hyperparams=True)

        # The latents stay split over the H and W axes for the host to assemble (see output_concat_dims),
        # which on a single host saves gathering them on the device, at a program per image size.
        return self.conv_out.forward(x)

    def output_concat_dims(self) -> list[int | None]:
        """Per mesh axis, the dimension of the [N, H, W, C] output its shards are split along, or None
        where the output is not split; the concat_dims of fast_device_to_host."""
        dims: list[int | None] = [None] * len(tuple(self._ctx.device.shape))
        if self._ctx.h_factor > 1 and self._ctx.h_mesh_axis is not None:
            dims[self._ctx.h_mesh_axis] = 1
        if self._ctx.w_factor > 1 and self._ctx.w_mesh_axis is not None:
            dims[self._ctx.w_mesh_axis] = 2
        return dims
