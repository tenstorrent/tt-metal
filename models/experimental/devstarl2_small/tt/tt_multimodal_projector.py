# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.tt.tt_patchmerger import TTMistral3PatchMerger
from models.experimental.devstarl2_small.tt.tt_rmsnorm import RMSNorm


class TTMistral3MultiModalProjector(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix="multi_modal_projector.",
        weight_cache_path=None,
        dtype=ttnn.bfloat16,
        eps=1e-5,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.state_dict_prefix = state_dict_prefix
        self.dtype = dtype

        norm_dim = state_dict[f"{state_dict_prefix}norm.weight"].numel()
        self.norm = RMSNorm(
            device=mesh_device,
            dim=norm_dim,
            state_dict=state_dict,
            weight_key=f"{state_dict_prefix}norm",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=eps,
        )
        self.patch_merger = TTMistral3PatchMerger(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}patch_merger.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        def as_linear_weight(name, mesh_mapper, cache_suffix=""):
            return ttnn.as_tensor(
                torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1),
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=None
                if weight_cache_path is None
                else weight_cache_path / f"{state_dict_prefix}{name}.weight{cache_suffix}",
            )

        self.linear_1_weight = as_linear_weight("linear_1", ttnn.ReplicateTensorToMesh(mesh_device))
        self.linear_2_weight = as_linear_weight(
            "linear_2", ttnn.ShardTensorToMesh(mesh_device, dim=-1), cache_suffix=".sharded_dim_-1"
        )
        self.linear_2_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=32,
            out_subblock_h=8,
            out_subblock_w=1,
            per_core_M=8,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        self.linear_2_large_m_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=16,
            per_core_N=10,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def forward(self, image_features: ttnn.Tensor, image_sizes) -> ttnn.Tensor:
        x = ttnn.reshape(image_features, (1, 1, image_features.shape[0], image_features.shape[1]))
        x = self.norm(x, mode="prefill")
        x = ttnn.reshape(x, image_features.shape)
        x = self.patch_merger(x, image_sizes)
        x = ttnn.linear(x, self.linear_1_weight, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.gelu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.typecast(x, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        m_tiles = (x.shape[0] + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        n_tiles = (self.linear_2_weight.shape[-1] + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        linear_2_program_config = (
            self.linear_2_large_m_program_config
            if ((m_tiles + 7) // 8) * n_tiles > 110
            else self.linear_2_program_config
        )
        x = ttnn.linear(
            x,
            self.linear_2_weight,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=linear_2_program_config,
        )
        return x


__all__ = ["TTMistral3MultiModalProjector"]
