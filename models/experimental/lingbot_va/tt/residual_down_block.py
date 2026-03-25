import ttnn
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.models.vae.vae_wan2_1 import WanResample
from models.experimental.lingbot_va.tt.avg_down_wan import TtAvgDown3D
from models.experimental.lingbot_va.tt.residual_block import WanResidualBlock
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.parallel.config import VaeHWParallelConfig


class WanResidualDownBlock(Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_res_blocks,
        temperal_downsample: bool = False,
        down_flag: bool = False,
        mesh_device: ttnn.MeshDevice = None,
        parallel_config: VaeHWParallelConfig = None,
        ccl_manager: CCLManager = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.avg_shortcut = TtAvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(
                WanResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    mesh_device=mesh_device,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                    dtype=dtype,
                )
            )
            in_dim = out_dim
        self.resnets = ModuleList(resnets)
        if down_flag:
            self.downsampler = WanResample(
                dim=out_dim,
                mode="downsample3d" if temperal_downsample else "downsample2d",
                mesh_device=mesh_device,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
                dtype=dtype,
            )
        else:
            self.downsampler = None

    def forward(self, x, logical_h: int, feat_cache=None, feat_idx=None):
        # x is (B, T, H, W, C) TILE; logical_h is the valid spatial height for masking (must match WanEncoder3d / WanResample).
        if feat_idx is None:
            feat_idx = [0]
        x_copy = ttnn.clone(x)

        for resnet in self.resnets:
            x = resnet(x, logical_h=logical_h, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            # WanResample expects ROW_MAJOR; resnets output TILE. WanResample returns updated logical_h after H/2.
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x, logical_h = self.downsampler(x, logical_h, feat_cache=feat_cache, feat_idx=feat_idx)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x_copy = ttnn.to_layout(x_copy, ttnn.ROW_MAJOR_LAYOUT)
        avg_shortcut_out = self.avg_shortcut(x_copy)
        avg_shortcut_out = ttnn.to_layout(avg_shortcut_out, ttnn.TILE_LAYOUT)
        return x + avg_shortcut_out, logical_h
