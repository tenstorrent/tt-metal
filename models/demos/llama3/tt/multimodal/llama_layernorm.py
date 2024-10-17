# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule

import torch
import os

TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


class TtLayerNorm(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        state_dict_prefix,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat8_b,
        model_config=None,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.device = device
        self.eps = eps

        torch_weight = (
            state_dict[f"{state_dict_prefix}weight"].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        )
        torch_bias = state_dict[f"{state_dict_prefix}bias"].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        cache_name = None if weight_cache_path is None else weight_cache_path / state_dict_prefix

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name / "weight",
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.bias = ttnn.as_tensor(
            torch_bias,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name / "bias",
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        if model_config:
            self.sharded_input_config = model_config["SHARDED_NORM_INPUT_MEMCFG"]
            self.sharded_program_config = model_config["SHARDED_NORM_PRGM_CFG"]
            self.sharded_output_config = model_config["SHARDED_NORM_OUTPUT_MEMCFG"]
        else:
            assert (
                dim % SHARD_HEIGHT == 0
            ), f"Input dimension dim ({dim}) must be a multiple of SHARD_HEIGHT ({SHARD_HEIGHT})"
            shard_width_hidden_dim_across_32_cores = dim // SHARD_HEIGHT
            core_grid = ttnn.CoreGrid(x=8, y=SHARD_HEIGHT // 8)
            # core_grid = ttnn.CoreGrid(x=8, y=8)
            self.sharded_input_config = ttnn.create_sharded_memory_config(
                shape=(SHARD_HEIGHT, shard_width_hidden_dim_across_32_cores),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.sharded_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[core_grid.x, core_grid.y],
                subblock_w=shard_width_hidden_dim_across_32_cores // TILE,
                block_h=SHARD_HEIGHT // TILE,
                block_w=shard_width_hidden_dim_across_32_cores // TILE,
                inplace=False,
            )
            self.sharded_output_config = self.sharded_input_config

    def forward(self, x):
        return self.forward_tt(x)
        if os.environ.get("LN") == "tt":
            return self.forward_tt(x)
        else:
            return self.forward_pt(x)

    def forward_pt(self, x: ttnn.Tensor, in_sharded=False, out_sharded=False) -> ttnn.Tensor:
        # If input is sharded do sharded RMSNorm and optionally return sharded output

        x = ttnn.to_torch(
            x,
            device=self.device,
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )[0].float()
        weight = ttnn.to_torch(
            self.weight,
            device=self.device,
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )[0, 0].float()
        bias = ttnn.to_torch(
            self.bias,
            device=self.device,
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )[0, 0].float()

        out = torch.nn.functional.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=self.eps)
        out = out

        out = ttnn.from_torch(
            out,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        return out

    def forward_tt(self, x: ttnn.Tensor, in_sharded=False, out_sharded=False) -> ttnn.Tensor:
        if in_sharded:
            x = ttnn.layer_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                bias=self.bias,
                program_config=self.sharded_program_config,
                memory_config=self.sharded_output_config,
            )
            if out_sharded:
                return x
            x_interleaved = ttnn.sharded_to_interleaved(x)
            x.deallocate(True)
            return x_interleaved
        else:  # Interleaved rmsnorm does not need program or memory configs
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"
            x = ttnn.layer_norm(
                x,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                ),
            )
            return x
