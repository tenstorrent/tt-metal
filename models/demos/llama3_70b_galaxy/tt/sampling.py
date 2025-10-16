# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TTSampling(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        k=torch.ones(32),
        p=torch.zeros(32),
        temp=torch.ones(32),
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices
        self.max_batch_size = args.max_batch_size
        self.max_top_k = args.max_top_k

        max_num_gather_links = args.model_config["GALAXY_NUM_LINKS"]
        self.num_gather_links = (
            self.max_top_k // 32 if self.max_top_k // 32 <= max_num_gather_links else max_num_gather_links
        )

        self.k_tensor = ttnn.from_torch(
            k,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
        )
        self.p_tensor = ttnn.from_torch(
            p,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
        )

        self.temp_tensor = ttnn.from_torch(
            temp,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
        )

        # Create indices tensor
        indices_device_offsets = torch.ones(
            1, 1, self.max_batch_size, self.max_top_k * self.args.cluster_shape[0], dtype=torch.int64
        )
        per_device_vocab_size = self.args.padded_vocab_size // self.args.cluster_shape[0]
        for device_id in range(self.args.cluster_shape[0]):
            indices_device_offsets[:, :, :, device_id * self.max_top_k : (device_id + 1) * self.max_top_k] = (
                device_id * per_device_vocab_size
            )
        self.tt_indices_device_offsets = ttnn.from_torch(
            indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        indices_tensor_torch = torch.zeros(1, 1, self.max_batch_size, per_device_vocab_size, dtype=torch.int32)
        for i in range(per_device_vocab_size):
            indices_tensor_torch[:, :, :, i] = i
        self.tt_indices_tensor = ttnn.from_torch(
            indices_tensor_torch,
            dtype=ttnn.uint16,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.warmup_done = False

    def reset_params(self, k, p, temp):
        self.k_tensor_new = ttnn.from_torch(
            torch.tensor(k),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.p_tensor_new = ttnn.from_torch(
            torch.tensor(p),
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.temp_tensor_new = ttnn.from_torch(
            torch.tensor(temp),
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        ttnn.copy_host_to_device_tensor(self.k_tensor_new, self.k_tensor)
        ttnn.copy_host_to_device_tensor(self.p_tensor_new, self.p_tensor)
        ttnn.copy_host_to_device_tensor(self.temp_tensor_new, self.temp_tensor)

    def forward(
        self,
        x: ttnn.Tensor,
        seed: int = 0,
        tt_out_tok: ttnn.Tensor = None,
    ):
        if self.warmup_done is False:
            self.warmup_done = True
            self.forward(x, seed=42, tt_out_tok=tt_out_tok)

        x_bf16 = ttnn.typecast(x, dtype=ttnn.bfloat16, sub_core_grids=self.args.sub_core_grids)

        # Local top k
        topk_values, topk_indices = ttnn.topk(
            x_bf16,
            k=self.max_top_k,
            dim=-1,
            sub_core_grids=self.args.sub_core_grid_topk,
            indices_tensor=self.tt_indices_tensor,
        )
        # Gather values
        # Note: Persistent output buffer used, do not deallocate output!
        topk_values_gathered = self.tt_ccl.line_all_gather(
            topk_values,
            dim=3,
            num_links=self.num_gather_links,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="SAMPLING_VALUES",
        )
        ttnn.deallocate(topk_values)

        # Convert values to bfloat16
        topk_values_gathered_bf16 = ttnn.to_memory_config(
            topk_values_gathered,
            memory_config=self.args.model_config["DECODE_SAMPLING_INPUT_MEMCFG"],
            dtype=ttnn.bfloat16,
        )
        topk_values_gathered_bf16_interleaved = ttnn.to_memory_config(
            topk_values_gathered_bf16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(topk_values_gathered_bf16)

        # Gather indices
        # Note: Persistent output buffer used, do not deallocate output!
        topk_indices_gathered = self.tt_ccl.line_all_gather(
            topk_indices,
            dim=3,
            num_links=self.num_gather_links,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="SAMPLING_INDICES",
        )
        ttnn.deallocate(topk_indices)

        topk_indices_gathered_uint32 = ttnn.typecast(
            topk_indices_gathered, dtype=ttnn.uint32, sub_core_grids=self.args.sub_core_grids
        )
        topk_indices_gathered_int32 = ttnn.typecast(
            topk_indices_gathered_uint32, dtype=ttnn.int32, sub_core_grids=self.args.sub_core_grids
        )
        ttnn.deallocate(topk_indices_gathered_uint32)

        topk_indices_gathered_int32_sharded = ttnn.to_memory_config(
            topk_indices_gathered_int32, self.args.model_config["DECODE_SAMPLING_INPUT_MEMCFG"]
        )
        ttnn.deallocate(topk_indices_gathered_int32)

        # Add device offsets for global indices
        topk_global_indices = ttnn.add(
            self.tt_indices_device_offsets, topk_indices_gathered_int32_sharded, dtype=ttnn.int32
        )
        ttnn.deallocate(topk_indices_gathered_int32_sharded)

        topk_global_indices_interleaved = ttnn.to_memory_config(topk_global_indices, ttnn.DRAM_MEMORY_CONFIG)
        # do not deallocate topk_global_indices

        # Untilize
        topk_global_indices_interleaved_untilised = ttnn.untilize(
            topk_global_indices_interleaved, use_multicore=True, sub_core_grids=self.args.sub_core_grids
        )
        ttnn.deallocate(topk_global_indices_interleaved)
        # Sampling
        tt_out_tok = ttnn.sampling(
            topk_values_gathered_bf16_interleaved,
            topk_global_indices_interleaved_untilised,
            k=self.k_tensor,
            p=self.p_tensor,
            temp=self.temp_tensor,
            seed=seed,
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.args.start_core, self.max_batch_size, self.args.sub_core_grids, row_wise=True
            ),
            output_tensor=tt_out_tok,
        )
        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)

        return tt_out_tok
