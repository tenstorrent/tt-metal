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
            M, K, N = 32, self.model_config["HIDDEN_SIZE"], self.model_config["FFN_EXPANDED_HIDDEN_SIZE"]

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

            self.FF1_DRAM_SHARDED_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=K
                // 8
                // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
                per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
                per_core_N=N // 8 // 32,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
                fused_activation=None,
            )

            self.FF2_DRAM_SHARDED_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=w2_K
                // 8
                // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
                per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
                per_core_N=w2_N // 8 // 32,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
                fused_activation=None,
            )

            self.COMPUTE_KERNEL_LOFI = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

            full_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    )
                }
            )
            self.FULL_GRID_MEMCFG = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    full_grid,
                    [
                        32,
                        nearest_32(56),
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )

            self.FF2_ACT_MEMCFG = ttnn.create_sharded_memory_config(
                shape=(M, N // 8),
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.FF1_ACT_MEMCFG = ttnn.create_sharded_memory_config(
                shape=(32, 2048 // 8),
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

    def load_weights(self):
        assert not hasattr(self, "w1_list"), "w1_list is already an attribute of this object"
        assert not hasattr(self, "w3_list"), "w3_list is already an attribute of this object"
        assert not hasattr(self, "w2_list"), "w2_list is already an attribute of this object"

        w1_str = f"{self.layer_name}.feed_forward.w1.weight"
        w2_str = f"{self.layer_name}.feed_forward.w2.weight"
        w3_str = f"{self.layer_name}.feed_forward.w3.weight"

        # TODO: Reenable when DRAM-SHARDED PCC issues resolves
        # w1_cache_str = f"{self.layer_name}.feed_forward.w1_galaxy_dram_shard_unpadded.weight"
        # w2_cache_str = f"{self.layer_name}.feed_forward.w2_galaxy_dram_shard_unpadded.weight"
        # w3_cache_str = f"{self.layer_name}.feed_forward.w3_galaxy_dram_shard_unpadded.weight"
        w1_cache_str = f"{self.layer_name}.feed_forward.w1_galaxy_unpadded.weight"
        w2_cache_str = f"{self.layer_name}.feed_forward.w2_galaxy_unpadded.weight"
        w3_cache_str = f"{self.layer_name}.feed_forward.w3_galaxy_unpadded.weight"

        w1_dtype = ttnn.bfloat4_b
        w2_dtype = ttnn.bfloat8_b
        w3_dtype = ttnn.bfloat4_b

        w1 = None
        w2 = None
        w3 = None
        if not self.read_cache:
            w1 = self.state_dict[w1_str].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            w2 = self.state_dict[w2_str].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            w3 = self.state_dict[w3_str].transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        self.w1 = ttnn.as_tensor(
            w1,
            dtype=w1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.w1_mem_config,  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / w1_cache_str,
        )

        self.w3 = ttnn.as_tensor(
            w3,
            dtype=w3_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.w1_mem_config,  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / w3_cache_str,
        )

        self.w2 = ttnn.as_tensor(
            w2,
            dtype=w2_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.w2_mem_config,  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
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

    def tt_all_reduce(self, input_tensor, cluster_axis, dim=0, memory_config=None):
        # Ensure the input tensor is in the correct memory configuration
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

        # Get the full device tensors list from the input tensor
        device_tensors = ttnn.get_device_tensors(input_tensor)

        num_rows, num_cols = self.cluster_shape[1], self.cluster_shape[0]

        def reduce_tensors(indices):
            tensors = ttnn.aggregate_as_tensor([device_tensors[i] for i in indices])
            gathered_tensor = ttnn.line_all_gather(
                tensors,
                dim,
                num_links=2,
                memory_config=ttnn.MemoryConfig(buffer_type=ttnn.experimental.tensor.BufferType.DRAM),
            )
            reduced_tensors = ttnn.experimental.tensor.fast_reduce_nc(
                gathered_tensor, dims=[dim], output=None, compute_kernel_config=None
            )

            return reduced_tensors

        aggregated_outputs = []

        if cluster_axis == 0:
            # Process row-wise when cluster_axis is 0
            for row in range(num_rows):
                start_idx = row * num_cols
                end_idx = start_idx + num_cols
                indices = range(start_idx, end_idx)
                reduced_tensors = reduce_tensors(indices)
                aggregated_outputs.append(reduced_tensors)

        elif cluster_axis == 1:
            # Process column-wise when cluster_axis is 1
            for col in range(num_cols):
                indices = range(col, len(device_tensors), num_cols)
                reduced_tensors = reduce_tensors(indices)
                aggregated_outputs.append(reduced_tensors)

        # Flatten device tensors
        if cluster_axis == 0:
            flattened_tensors = [tensor for output in aggregated_outputs for tensor in ttnn.get_device_tensors(output)]
        elif cluster_axis == 1:

            def flatten_column_major(array):
                flattened_list = []

                for col in range(len(array[0])):
                    for row in range(len(array)):
                        flattened_list.append(array[row][col])

                return flattened_list

            flattened_tensors = flatten_column_major([ttnn.get_device_tensors(tensor) for tensor in aggregated_outputs])

        final_output_tensor = ttnn.aggregate_as_tensor(flattened_tensors)

        return final_output_tensor

    def decode_forward(self, x: List[ttnn.Tensor]) -> List[ttnn.Tensor]:
        w1_out = ttnn.matmul(
            x,
            self.w1,
            # program_config=self.FF1_DRAM_SHARDED_PROGCFG,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.COMPUTE_KERNEL_LOFI,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            # program_config=self.FF1_DRAM_SHARDED_PROGCFG,  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.COMPUTE_KERNEL_LOFI,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        x.deallocate(True)

        w1_out = self.tt_all_reduce(w1_out, cluster_axis=0, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        w3_out = self.tt_all_reduce(w3_out, cluster_axis=0, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

        w1_out = ttnn.to_memory_config(w1_out, self.FULL_GRID_MEMCFG)
        w1_out = ttnn.silu(w1_out)

        w1_out = ttnn.to_memory_config(w1_out, self.FF2_ACT_MEMCFG)
        w3_out = ttnn.to_memory_config(w3_out, self.FF2_ACT_MEMCFG)

        hidden_states = ttnn.mul(
            w1_out,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states = ttnn.matmul(
            hidden_states,
            self.w2,
            # program_config=self.FF2_DRAM_SHARDED_PROGCFG,  # TODO: Reenable when DRAM-SHARDED PCC issues resolves
            core_grid=ttnn.CoreGrid(y=1, x=8),
            compute_kernel_config=self.COMPUTE_KERNEL_LOFI,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        hidden_states = self.tt_all_reduce(
            hidden_states, cluster_axis=1, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )

        hidden_states = ttnn.to_memory_config(hidden_states, self.FF1_ACT_MEMCFG)

        return hidden_states
