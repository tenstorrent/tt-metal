# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
import torch
import ttnn
from ttnn import ReplicateTensorToMesh
from models.demos.t3000.llama2_70b.tt.llama_common import ShardTensor2dMesh, ConcatMesh2DToTensor
from models.utility_functions import nearest_32


class TtLlamaMLP_galaxy:
    def __init__(
        self,
        device_mesh,
        cluster_shape,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        cache_path=None,
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        assert self.num_devices == 32, "Only 32 devices supported for TG"
        self.model_config = model_config
        self.read_cache = read_cache
        self.cluster_shape = cluster_shape

        self.hidden_size = hidden_size

        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path

        self.get_mlp_model_config()
        self.load_weights()

    def set_model_config(self, model_config):
        self.model_config = model_config

    def get_mlp_model_config(self):
        if self.model_config["LLM_MODE"] == "decode":
            # Weight Sharding
            weight_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(
                            self.device_mesh.get_device(0).dram_grid_size().x - 1,
                            self.device_mesh.get_device(0).dram_grid_size().y - 1,
                        ),
                    )
                }
            )
            M, K, N = 32, 8192, 32768

            K = K // self.cluster_shape[0]
            N = N // self.cluster_shape[1]
            shard_shape = (K, nearest_32(N // 12))  # padded cols to divide by 12
            shard_spec = ttnn.ShardSpec(weight_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
            self.w1_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec
            )

            w2_K, w2_N = N, K
            shard_shape = (w2_K, nearest_32(w2_N // 12))  # padded cols to divide by 12
            shard_spec = ttnn.ShardSpec(weight_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
            self.w2_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec
            )

            self.DRAM_SHARDED_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=K
                // 8
                // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
                per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
                per_core_N=N // 8 // 32,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
                fused_activation=None,
            )

            self.COMPUTE_KERNEL_LOFI = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def load_weights(self):
        assert not hasattr(self, "w1_list"), "w1_list is already an attribute of this object"
        assert not hasattr(self, "w3_list"), "w3_list is already an attribute of this object"
        assert not hasattr(self, "w2_list"), "w2_list is already an attribute of this object"

        w1_str = f"{self.layer_name}.feed_forward.w1.weight"
        w2_str = f"{self.layer_name}.feed_forward.w2.weight"
        w3_str = f"{self.layer_name}.feed_forward.w3.weight"

        # TODO: Reenable when DRAM-SHARDED PCC issues resolves
        # w1_cache_str = f"{self.layer_name}.feed_forward.w1_galaxy_dram_shard.weight"
        # w2_cache_str = f"{self.layer_name}.feed_forward.w2_galaxy_dram_shard.weight"
        # w3_cache_str = f"{self.layer_name}.feed_forward.w3_galaxy_dram_shard.weight"
        w1_cache_str = f"{self.layer_name}.feed_forward.w1_galaxy.weight"
        w2_cache_str = f"{self.layer_name}.feed_forward.w2_galaxy.weight"
        w3_cache_str = f"{self.layer_name}.feed_forward.w3_galaxy.weight"

        w1_dtype = ttnn.bfloat4_b
        w2_dtype = ttnn.bfloat8_b
        w3_dtype = ttnn.bfloat4_b

        padded_w1 = None
        padded_w2 = None
        padded_w3 = None
        if not self.read_cache:
            # Do padding
            H = 8 * 1024
            PADDED_H4 = 32 * 1024
            H4 = 28 * 1024
            padded_w1 = torch.zeros(1, 1, H, PADDED_H4)
            padded_w2 = torch.zeros(1, 1, PADDED_H4, H)
            padded_w3 = torch.zeros(1, 1, H, PADDED_H4)
            padded_w1[:, :, :, :H4] = self.state_dict[w1_str].transpose(-2, -1)
            padded_w2[:, :, :H4, :] = self.state_dict[w2_str].transpose(-2, -1)
            padded_w3[:, :, :, :H4] = self.state_dict[w3_str].transpose(-2, -1)

        self.w1 = ttnn.as_tensor(
            padded_w1,
            dtype=w1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.w1_mem_config, # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / w1_cache_str,
        )

        self.w3 = ttnn.as_tensor(
            padded_w3,
            dtype=w3_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.w1_mem_config, # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / w3_cache_str,
        )

        self.w2 = ttnn.as_tensor(
            padded_w2,
            dtype=w2_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.w2_mem_config, # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=(3, 2), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / w2_cache_str,
        )

    def __call__(self, x: List[ttnn.Tensor]) -> List[ttnn.Tensor]:
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(x)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def tt_all_reduce(self, tensors, cluster_axis, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG):
        """
        reduction of a multi-device tensor
        """
        concat_dim = (1, 3) if cluster_axis == 0 else (3, 1)

        out = ttnn.to_torch(
            tensors,
            mesh_composer=ConcatMesh2DToTensor(self.device_mesh, dims=concat_dim, cluster_shape=self.cluster_shape),
        )
        out = torch.sum(out, dim=1, keepdim=True)

        shape = (
            out.shape[2],
            out.shape[3] // 8 // (self.cluster_shape[1] if cluster_axis == 0 else self.cluster_shape[0]),
        )

        if memory_config == ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG:
            act_mem_config = ttnn.create_sharded_memory_config(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            act_mem_config = None

        act_shard_dim = (None, 3) if cluster_axis == 0 else (3, None)
        out_tt = ttnn.from_torch(
            out,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=act_mem_config,
            device=self.device_mesh,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=act_shard_dim, cluster_shape=self.cluster_shape),
        )

        return out_tt

    def tt_all_gather(self, tensors, dim, cluster_axis, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG):
        """
        gather of a multi-device tensor
        """
        concat_dim = (dim, 1) if cluster_axis == 0 else (1, dim)
        shard_dim = (None, 1) if cluster_axis == 0 else (1, None)

        out = ttnn.to_torch(
            tensors,
            mesh_composer=ConcatMesh2DToTensor(self.device_mesh, dims=concat_dim, cluster_shape=self.cluster_shape),
        )
        shape = (out.shape[2], out.shape[3] // 32)

        if memory_config == ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG:
            act_mem_config = ttnn.create_sharded_memory_config(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            act_mem_config = None

        out_tt = ttnn.from_torch(
            out,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=act_mem_config,
            device=self.device_mesh,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=shard_dim, cluster_shape=self.cluster_shape),
        )

        return out_tt

    def decode_forward(self, x: List[ttnn.Tensor]) -> List[ttnn.Tensor]:
        w1_out = ttnn.matmul(
            x,
            self.w1,
            # program_config=self.DRAM_SHARDED_PROGCFG,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.COMPUTE_KERNEL_LOFI,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            # program_config=self.DRAM_SHARDED_PROGCFG, # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.COMPUTE_KERNEL_LOFI,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        x.deallocate(True)

        w1_out = self.tt_all_reduce(w1_out, cluster_axis=0, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        w3_out = self.tt_all_reduce(w3_out, cluster_axis=0, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

        # w1_out = ttnn.all_reduce(w1_out, cluster_axis=0, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        # w3_out = ttnn.all_reduce(w3_out, cluster_axis=0, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

        w1_out = ttnn.silu(w1_out)

        hidden_states = ttnn.mul(
            w1_out,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states = ttnn.matmul(
            hidden_states,
            self.w2,
            # program_config=self.DRAM_SHARDED_PROGCFG, # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.COMPUTE_KERNEL_LOFI,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        hidden_states = self.tt_all_reduce(
            hidden_states, cluster_axis=1, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )
        # hidden_states = ttnn.all_reduce(hidden_states, cluster_axis=1, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

        return hidden_states
