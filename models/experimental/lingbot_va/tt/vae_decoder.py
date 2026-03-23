# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN VAE decoder for Lingbot-VA (WanVAE 2.2 with is_residual=True).

Based on tt_dit/models/vae/vae_wan2_1.py (WanDecoder3d + WanDecoder)
but uses WanResidualUpBlock instead of WanUpBlock.
"""

from __future__ import annotations

from typing import Sequence

import torch

import ttnn

from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import RMSNorm
from models.tt_dit.models.vae.vae_wan2_1 import (
    CACHE_T,
    WanCausalConv3d,
    WanMidBlock,
    conv3d_to_linear_weight,
)
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import aligned_channels, count_convs
from models.tt_dit.utils.substate import pop_substate

from .conv3d_configs import override_conv3d_configs
from .residual_up_block import WanResidualUpBlock


class WanResidualDecoder3d(Module):
    """WanDecoder3d with is_residual=True: uses WanResidualUpBlock."""

    def __init__(
        self,
        *,
        dim: int = 256,
        z_dim: int = 48,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence = (),
        temperal_upsample: Sequence[bool] = (True, True, False),
        out_channels: int = 12,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # Dimensions: [dim*mult[-1], dim*mult[-1], ..., dim*mult[0]]
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult[::-1])]

        # conv_in
        self.conv_in = WanCausalConv3d(
            z_dim,
            dims[0],
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        # mid_block
        self.mid_block = WanMidBlock(
            dim=dims[0],
            num_layers=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        # Override SDPA config for large head_dim (1024) to avoid L1 overflow.
        # Default k_chunk_size=256 with head_dim=1024 requires ~2.5MB per core,
        # exceeding the 1.5MB L1 limit. Reducing to 32 brings it to ~260KB.
        if dims[0] > 512:
            for attn in self.mid_block.attentions:
                attn.sdpa_program_config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32,
                    k_chunk_size=32,
                    exp_approx_mode=False,
                )

        # up_blocks (residual)
        self.up_blocks = ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(dim_mult) - 1
            up_block = WanResidualUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                temperal_upsample=temperal_upsample[i] if up_flag else False,
                up_flag=up_flag,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            self.up_blocks.append(up_block)

        # Output projection
        final_dim = dims[-1]
        self.norm_out = RMSNorm(
            embedding_dim=final_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.conv_out = WanCausalConv3d(
            final_dim,
            out_channels,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        override_conv3d_configs(self)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm_out.gamma" in state:
            state["norm_out.weight"] = state.pop("norm_out.gamma").squeeze()

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> tuple[ttnn.Tensor, int]:
        if feat_idx is None:
            feat_idx = [0]

        # conv_in
        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x = x_BTHWC[:, t_start:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x], dim=1)
            x_BTHWC = self.conv_in(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_in(x_BTHWC, logical_h)

        # WanCausalConv3d is ROW_MAJOR; WanMidBlock / residual path expect TILE (matches WanDecoder3d).
        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)

        # mid_block
        x_BTHWC = self.mid_block(x_BTHWC, logical_h, feat_cache, feat_idx)

        # up_blocks
        for up_block in self.up_blocks:
            x_BTHWC, logical_h = up_block(x_BTHWC, logical_h, feat_cache, feat_idx, first_chunk=first_chunk)

        # norm + silu + conv_out
        x_tile = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
        x_norm = self.norm_out(x_tile)
        x_silu = ttnn.silu(x_norm)
        x_BTHWC = ttnn.to_layout(x_silu, ttnn.ROW_MAJOR_LAYOUT)

        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x = x_BTHWC[:, t_start:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x], dim=1)
            x_BTHWC = self.conv_out(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_out(x_BTHWC, logical_h)

        return x_BTHWC, logical_h


class WanVAEDecoder(Module):
    """Top-level VAE decoder for Lingbot-VA.

    Wraps post_quant_conv + WanResidualDecoder3d + unpatchify + clamp.
    Follows the same pattern as tt_dit WanDecoder but for is_residual=True.
    """

    def __init__(
        self,
        *,
        base_dim: int = 160,
        decoder_base_dim: int | None = None,
        z_dim: int = 48,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence = (),
        temperal_downsample: Sequence[bool] = (False, True, True),
        out_channels: int = 12,
        patch_size: int = 2,
        latents_mean: Sequence[float] = (),
        latents_std: Sequence[float] = (),
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temperal_upsample = list(reversed(temperal_downsample))
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.latents_mean = list(latents_mean)
        self.latents_std = list(latents_std)
        decoder_base_dim = decoder_base_dim or base_dim

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.post_quant_conv = Linear(
            in_features=aligned_channels(z_dim),
            out_features=aligned_channels(z_dim),
            mesh_device=mesh_device,
        )

        self.decoder = WanResidualDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=self.temperal_upsample,
            out_channels=out_channels,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        self.cached_conv_count = count_convs(self.decoder)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "post_quant_conv.weight" in state and "post_quant_conv.bias" in state:
            state_sub = conv3d_to_linear_weight(pop_substate(state, "post_quant_conv"))
            state["post_quant_conv.weight"] = state_sub["weight"]
            state["post_quant_conv.bias"] = state_sub["bias"]

        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_cache = [None] * self.cached_conv_count

    def forward(self, z_BTHWC: ttnn.Tensor, logical_h: int) -> tuple[ttnn.Tensor, int]:
        B, T, H, W, C = z_BTHWC.shape

        self.clear_cache()
        z_tile = ttnn.to_layout(z_BTHWC, ttnn.TILE_LAYOUT)
        x_tile = self.post_quant_conv(z_tile)
        x_BTHWC = ttnn.to_layout(x_tile, ttnn.ROW_MAJOR_LAYOUT)

        output_BCTHW = None
        for i in range(T):
            self._conv_idx = [0]
            first_chunk = i == 0
            out_BTHWC, new_logical_h = self.decoder(
                x_BTHWC[:, i : i + 1, :, :, :],
                logical_h,
                feat_cache=self._feat_cache,
                feat_idx=self._conv_idx,
                first_chunk=first_chunk,
            )
            # Channels-first for concatenation
            out_BCTHW = ttnn.permute(out_BTHWC, (0, 4, 1, 2, 3))
            out_BCTHW = out_BCTHW[:, : self.out_channels, :, :, :]
            if output_BCTHW is None:
                output_BCTHW = out_BCTHW
            else:
                output_BCTHW = ttnn.concat([output_BCTHW, out_BCTHW], dim=2)

        # Clamp to [-1, 1]
        output_tile = ttnn.to_layout(output_BCTHW, ttnn.TILE_LAYOUT)
        output_BCTHW = ttnn.clamp(output_tile, min=-1.0, max=1.0)
        output_BCTHW = ttnn.to_layout(output_BCTHW, ttnn.ROW_MAJOR_LAYOUT)

        self.clear_cache()
        return output_BCTHW, new_logical_h
