# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import ttnn

from ...blocks.vae import VaeContext, VaeConv2d, VaeMidBlock, VaeNormDescGroup, VaeUpBlock
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import GroupNorm
from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate

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
        parallel_config: VAEParallelConfig | None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        super().__init__()

        ctx = VaeContext(
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            device=device,
            ccl_manager=ccl_manager,
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

        self.conv_norm_out = GroupNorm(
            num_groups=32, num_channels=channel_counts[-1], eps=1e-6, mesh_axis=ctx.tp_axis, mesh_device=ctx.device
        )
        self.conv_out = VaeConv2d(
            channel_counts[-1], out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx
        )

        bn_size = self._PATCH_SIZE**2 * z_channels
        self.bn_running_mean = Parameter(total_shape=[bn_size], device=ctx.device)
        self.bn_running_var = Parameter(total_shape=[bn_size], device=ctx.device)
        self.bn_eps = 1e-4

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

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

    def preprocess_and_unpatchify(self, z: ttnn.Tensor, *, height: int, width: int) -> ttnn.Tensor:
        # N, (H / P) * (W / P), C * P * P -> N, H, W, C

        n, _, _ = z.shape
        p = self._PATCH_SIZE

        z = self._inv_normalize(z)

        z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        z = z.reshape([n, height // p, width // p, -1, p, p])
        z = ttnn.permute(z, [0, 1, 4, 2, 5, 3])
        z = z.reshape([n, height, width, -1])
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)

        return z

    def forward(self, z: ttnn.Tensor, /) -> ttnn.Tensor:
        z = self.post_quant_conv.forward(z)
        z = self.conv_in.forward(z)

        z = self.mid_block.forward(z)

        for block in self.up_blocks:
            z = block.forward(z)

        z = self.conv_norm_out.forward(z)
        z = ttnn.silu(z)

        if self._ccl_manager is not None:
            z = self._ccl_manager.all_gather(z, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return self.conv_out.forward(z)
