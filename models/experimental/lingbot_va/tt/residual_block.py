# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_dit.layers.module import Module
from models.tt_dit.layers.normalization import RMSNorm
from models.tt_dit.models.vae.vae_wan2_1 import CACHE_T, WanCausalConv3d
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager


class WanResidualBlock(Module):
    """
    Lingbot-WAN residual block that keeps a causal 1x1 conv shortcut (reference parity).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mesh_device = mesh_device

        self.norm1 = RMSNorm(
            embedding_dim=in_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv1 = WanCausalConv3d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )
        self.norm2 = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv2 = WanCausalConv3d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        if in_dim != out_dim:
            self.conv_shortcut = WanCausalConv3d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=1,
                padding=0,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
            )
        else:
            self.conv_shortcut = None

        self.norm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, ttnn.Tensor]) -> None:
        if "norm1.gamma" in state:
            state["norm1.weight"] = state.pop("norm1.gamma").squeeze()

        if "norm2.gamma" in state:
            state["norm2.weight"] = state.pop("norm2.gamma").squeeze()

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] | None = None,
    ) -> ttnn.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        assert x_BTHWC.layout == ttnn.TILE_LAYOUT, f"WanResidualBlock expects TILE input, got {x_BTHWC.layout}"
        if self.conv_shortcut is not None:
            shortcut_rm_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
            h_rm_BTHWC = self.conv_shortcut(shortcut_rm_BTHWC, logical_h)
            h_tile_BTHWC = ttnn.to_layout(h_rm_BTHWC, ttnn.TILE_LAYOUT)
        else:
            h_tile_BTHWC = x_BTHWC

        x_norm_silu_tile_BTHWC = self.norm1(x_BTHWC, compute_kernel_config=self.norm_compute_kernel_config)
        x_BTHWC = ttnn.to_layout(x_norm_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

            x_conv_BTHWC = self.conv1(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_conv_BTHWC = self.conv1(x_BTHWC, logical_h)

        x_BTHWC = self.norm2(x_conv_BTHWC, compute_kernel_config=self.norm_compute_kernel_config)

        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

            x_conv_BTHWC = self.conv2(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_conv_BTHWC = self.conv2(x_BTHWC, logical_h)

        x_tile_BTHWC = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
        return ttnn.add(h_tile_BTHWC, x_tile_BTHWC)
