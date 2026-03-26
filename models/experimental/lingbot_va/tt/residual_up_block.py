# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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

        self.avg_shortcut = (
            TtDupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
            if up_flag
            else None
        )

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

        upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
        self.upsampler = (
            WanResample(
                dim=out_dim,
                mode=upsample_mode,
                resample_out_dim=out_dim,
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            if up_flag
            else None
        )

    def _prepare_torch_state(self, state: dict) -> None:
        # Checkpoints store this as upsamplers.0; TT module keeps a direct attribute.
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

        x_copy = ttnn.clone(x_BTHWC)

        for resnet in self.resnets:
            x_BTHWC = resnet(x_BTHWC, logical_h, feat_cache, feat_idx)

        if self.upsampler is not None:
            # WanResample expects ROW_MAJOR; WanResidualBlock output is TILE (same as WanUpBlock).
            x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
            x_BTHWC, logical_h = self.upsampler(x_BTHWC, logical_h, feat_cache, feat_idx)
            x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)

        if self.avg_shortcut is not None:
            shortcut = self.avg_shortcut(x_copy, first_chunk=first_chunk)
            shortcut = ttnn.to_layout(shortcut, ttnn.TILE_LAYOUT)
            x_tile = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
            x_BTHWC = ttnn.add(x_tile, shortcut)

        return x_BTHWC, logical_h
