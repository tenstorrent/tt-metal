# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
WanResidualUpBlock – TTNN decoder counterpart of WanResidualDownBlock.

Mirrors the host WanResidualUpBlock from vae_wan2_1_encoder_host.py.
Uses TtDupUp3D for the residual shortcut and WanResample for spatial/temporal upsampling.
"""

import ttnn

from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.models.vae.vae_wan2_1 import WanResample, WanResidualBlock
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.substate import rename_substate

from .dup_up_wan import TtDupUp3D


class WanResidualUpBlock(Module):
    """Residual up-sampling block for the WanVAE 2.2 decoder (is_residual=True).

    Structure mirrors WanResidualDownBlock but in reverse:
      resnets → upsampler → output + avg_shortcut(input_copy)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        mesh_device: ttnn.MeshDevice = None,
        parallel_config: VaeHWParallelConfig = None,
        ccl_manager: CCLManager = None,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.up_flag = up_flag

        if up_flag:
            self.avg_shortcut = TtDupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(
                    in_dim=current_dim,
                    out_dim=out_dim,
                    mesh_device=mesh_device,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
            )
            current_dim = out_dim
        self.resnets = ModuleList(resnets)

        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(
                dim=out_dim,
                mode=upsample_mode,
                resample_out_dim=out_dim,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
        else:
            self.upsampler = None

    def _prepare_torch_state(self, state: dict) -> None:
        # Host model stores upsampler inside a ModuleList-like structure;
        # our ttnn model stores it directly.
        rename_substate(state, "upsamplers.0", "upsampler")

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> tuple[ttnn.Tensor, int]:
        if feat_idx is None:
            feat_idx = [0]

        # Branch for avg_shortcut uses the block input (layout as received).
        x_copy = ttnn.clone(x_BTHWC)
        # Prior block may return ROW_MAJOR after shortcut add; resnets require TILE (WanUpBlock always feeds TILE).
        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)

        for resnet in self.resnets:
            x_BTHWC = resnet(x_BTHWC, logical_h, feat_cache, feat_idx)

        if self.upsampler is not None:
            # WanResidualBlock is TILE; WanResample/WanConv2d need ROW_MAJOR (same as WanUpBlock).
            x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
            x_upsampled_BTHWC, logical_h = self.upsampler(x_BTHWC, logical_h, feat_cache, feat_idx)
            x_BTHWC = ttnn.to_layout(x_upsampled_BTHWC, ttnn.TILE_LAYOUT)

        if self.avg_shortcut is not None:
            shortcut = self.avg_shortcut(x_copy, first_chunk=first_chunk)
            shortcut = ttnn.to_layout(shortcut, ttnn.TILE_LAYOUT)
            x_tile = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
            x_BTHWC = ttnn.add(x_tile, shortcut)
            x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        return x_BTHWC, logical_h
