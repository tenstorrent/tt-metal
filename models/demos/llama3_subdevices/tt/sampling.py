# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule


class TTSampling(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        sampling_params,
        tt_ccl,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices
        self.max_batch_size = args.max_batch_size
        self.k = [sampling_params["top_k"]] * self.max_batch_size
        self.p = [sampling_params["top_p"]] * self.max_batch_size
        self.seed = sampling_params["seed"]

        # Create indices tensor
        num_local_top_k = 32
        indices_device_offsets = torch.ones(
            1, 1, self.max_batch_size, num_local_top_k * self.args.cluster_shape[0], dtype=torch.int64
        )
        per_device_vocab_size = self.args.padded_vocab_size // self.args.cluster_shape[0]
        for device_id in range(self.args.cluster_shape[0]):
            indices_device_offsets[:, :, :, device_id * num_local_top_k : (device_id + 1) * num_local_top_k] = (
                device_id * per_device_vocab_size
            )
        self.tt_indices_device_offsets = ttnn.from_torch(
            indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor):
        # # Fall back to host for now
        # out_torchs_list = ttnn.to_torch(x, mesh_composer=ttnn.ListMeshToTensor(self.mesh_device))

        # topk_values_list = []
        # topk_indices_list = []
        # for i in range(len(out_torchs_list)):
        #     if i % 4 == 0:
        #         topk_values_device, topk_indices_device = out_torchs_list[i].topk(k=32, dim=-1)
        #         topk_values_list.append(topk_values_device)
        #         topk_indices_list.append(topk_indices_device)

        # topk_values_tensor = torch.cat(topk_values_list, dim=3)
        # topk_indices_tensor = torch.cat(topk_indices_list, dim=3)

        # topk_values = ttnn.from_torch(
        #     topk_values_tensor,
        #     device=self.mesh_device,
        #     dtype=ttnn.bfloat16,
        #     layout=ttnn.TILE_LAYOUT,
        #     mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-1, None), mesh_shape=self.args.cluster_shape),
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )
        # topk_indices = ttnn.from_torch(
        #     topk_indices_tensor,
        #     device=self.mesh_device,
        #     dtype=ttnn.uint16,
        #     layout=ttnn.TILE_LAYOUT,
        #     mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-1, None), mesh_shape=self.args.cluster_shape),
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )

        # Local top k
        topk_values, topk_indices = ttnn.topk(x, k=32, dim=-1, sub_core_grids=self.args.sub_core_grid_topk)
        ttnn.deallocate(x)

        topk_values_gathered = self.tt_ccl.line_all_gather(
            topk_values, dim=3, num_links=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(topk_values)

        topk_values_gathered_bf16 = ttnn.to_memory_config(
            topk_values_gathered,
            memory_config=self.args.model_config["DECODE_SAMPLING_INPUT_MEMCFG"],
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(topk_values_gathered)
        topk_values_gathered_bf16_interleaved = ttnn.to_memory_config(
            topk_values_gathered_bf16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(topk_values_gathered_bf16)

        topk_indices_gathered = self.tt_ccl.line_all_gather(
            topk_indices, dim=3, num_links=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(topk_indices)

        topk_indices_gathered_sharded = ttnn.to_memory_config(
            topk_indices_gathered, self.args.model_config["DECODE_SAMPLING_INPUT_MEMCFG"], dtype=ttnn.bfloat16
        )
        ttnn.deallocate(topk_indices_gathered)

        topk_global_indices = ttnn.add(topk_indices_gathered_sharded, self.tt_indices_device_offsets, dtype=ttnn.uint32)
        ttnn.deallocate(topk_indices_gathered_sharded)

        topk_global_indices_interleaved = ttnn.to_memory_config(topk_global_indices, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(topk_global_indices)

        topk_global_indices_interleaved_untilised = ttnn.untilize(topk_global_indices_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved)

        tt_out_tok = ttnn.sampling(
            topk_values_gathered_bf16_interleaved,
            topk_global_indices_interleaved_untilised,
            k=self.k,
            p=self.p,
            seed=self.seed,
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.args.start_core, self.max_batch_size, self.args.sub_core_grids, row_wise=True
            ),
        )
        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)

        return tt_out_tok
