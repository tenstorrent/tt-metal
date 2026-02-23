# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import ttnn

from ...blocks.vae import VaeContext, VaeConv2d, VaeMidBlock, VaeNormDescGroup, VaeUpBlock
from ...layers.module import Module, ModuleList
from ...layers.normalization import GroupNorm
from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from diffusers.models.autoencoders import vae as diffusers_vae


class VAEDecoder(Module):
    def __init__(
        self,
        *,
        block_out_channels: Sequence[int] = (128, 256, 512, 512),
        in_channels: int = 16,
        out_channels: int = 3,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VAEParallelConfig | None,
        ccl_manager: CCLManager | None,
    ) -> None:
        """
        Initialize the VAEDecoder.

        Args:
            block_out_channels: The number of channels for the updecoder blocks. They are also used to support other layers and blocks
            in_channels: The number of channels in the input image.
            out_channels: The number of channels in the output image.
            layers_per_block: The number of Resnet layers (blocks) in each updecoder.
            norm_num_groups: The number of groups in the normalization layer.
            mesh_device: The device to use for the model.
            parallel_config: The parallel config to use for the model.
            ccl_manager: The ccl manager to use for the model.
        """
        super().__init__()

        ctx = VaeContext(
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            device=mesh_device,
            ccl_manager=ccl_manager,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        channel_counts = [block_out_channels[-1], *block_out_channels[::-1]]
        eps = 1e-6

        self.conv_in = VaeConv2d(in_channels, channel_counts[0], kernel_size=3, padding=1, ctx=ctx)

        self.mid_block = VaeMidBlock(
            num_channels=channel_counts[0],
            norm=VaeNormDescGroup(num_groups=norm_num_groups, eps=eps),
            ctx=ctx,
        )

        self.up_blocks = ModuleList(
            VaeUpBlock(
                in_channels=ch_in,
                out_channels=ch_out,
                upsample=i != len(channel_counts) - 2,
                num_layers=layers_per_block + 1,
                norm=VaeNormDescGroup(num_groups=norm_num_groups, eps=eps),
                ctx=ctx,
            )
            for i, (ch_in, ch_out) in enumerate(itertools.pairwise(channel_counts))
        )

        self.conv_norm_out = GroupNorm(
            num_groups=norm_num_groups,
            num_channels=channel_counts[-1],
            eps=1e-6,
            mesh_axis=ctx.tp_axis,
            mesh_device=ctx.device,
        )
        self.conv_out = VaeConv2d(
            channel_counts[-1], out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    @classmethod
    def from_torch(
        cls,
        torch_ref: diffusers_vae.Decoder,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> VAEDecoder:
        model = cls(
            block_out_channels=[block.resnets[0].conv2.out_channels for block in torch_ref.up_blocks][::-1],
            in_channels=torch_ref.conv_in.in_channels,
            out_channels=torch_ref.conv_out.out_channels,
            layers_per_block=torch_ref.layers_per_block,
            norm_num_groups=torch_ref.mid_block.resnets[0].norm1.num_groups,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        model.load_torch_state_dict(torch_ref.state_dict())
        return model

    def forward(self, z: ttnn.Tensor, /) -> ttnn.Tensor:
        z = self.conv_in.forward(z)

        z = self.mid_block.forward(z)

        for block in self.up_blocks:
            z = block.forward(z)

        z = self.conv_norm_out.forward(z)
        z = ttnn.silu(z)

        if self._ccl_manager is not None:
            z = self._ccl_manager.all_gather(z, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return self.conv_out.forward(z)
