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
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.args.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor, tt_out_tok: ttnn.Tensor = None):
        # Local top k
        topk_values, topk_indices = ttnn.topk(x, k=32, dim=-1, sub_core_grids=self.args.sub_core_grid_topk)

        # Gather values
        # Note: Persistent output buffer used, do not deallocate output!
        topk_values_gathered = self.tt_ccl.line_all_gather(
            topk_values,
            dim=3,
            num_links=2,
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
            num_links=2,
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
            k=self.k,
            p=self.p,
            seed=self.seed,
            # seed=np.random.randint(0, 2**32 - 1), # TODO: find solution for constant outputs for constant seed
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.args.start_core, self.max_batch_size, self.args.sub_core_grids, row_wise=True
            ),
            output_tensor=tt_out_tok,
        )
        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)

        return tt_out_tok
