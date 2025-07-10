# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


class TT_CCL:
    def __init__(
        self,
        mesh_device,
        model_config,
        seq_len,  # if seq_len is not None, it will be used to set the size of the prefill persistent buffers
    ):
        self.mesh_device = mesh_device
        self.worker_sub_device_id = ttnn.SubDeviceId(0)
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        self.mesh_device.compute_with_storage_grid_size().x - 1,
                        self.mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )

        self.ag_semaphores_idx = 0
        self.ag_semaphore_handles = [[], []]

        self.rs_semaphores_idx = 0
        self.rs_semaphore_handles = [[], []]

        self.model_config = model_config

        self.ag_output_pbs = {}
        self.rs_output_pbs = {}
        self.rs_intermediate_pbs = {}

        self.seq_len = 32  # TODO: Why is the falcon demo dim 2 seq_len hardcoded to 32?

        self.create_persistent_buffers()

        for i in range(2):
            for _ in range(2):
                self.ag_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
            for _ in range(3):
                self.rs_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

        worker_sub_device = ttnn.SubDevice([self.sub_device_crs])
        sub_device_manager = self.mesh_device.create_sub_device_manager([worker_sub_device], 0)
        self.mesh_device.load_sub_device_manager(sub_device_manager)
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    def get_and_cycle_ag_semaphore_handles(self):
        current_idx = self.ag_semaphores_idx
        self.ag_semaphores_idx = (self.ag_semaphores_idx + 1) % 2
        return self.ag_semaphore_handles[current_idx]

    def get_and_cycle_rs_semaphore_handles(self):
        current_idx = self.rs_semaphores_idx
        self.rs_semaphores_idx = (self.rs_semaphores_idx + 1) % 2
        return self.rs_semaphore_handles[current_idx]

    def create_persistent_buffer(self, shape, mem_config, dtype, distributed=False):
        if distributed:
            shape[3] *= self.mesh_device.get_num_devices()
            cluster_shape = list(self.mesh_device.shape)
            mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 3), mesh_shape=cluster_shape)
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return ttnn.from_torch(
            torch.zeros(shape),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )

    def create_persistent_buffers(self):
        shard_spec_32_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 3),
                ),
            }
        )

        # prefill
        self.ag_output_pbs["DECODER_FWD_PREFILL_AG"] = self.create_persistent_buffer(
            shape=[1, 1, self.seq_len, 8192],
            mem_config=self.model_config["DEFAULT_MEMCFG"],
            dtype=ttnn.DataType.BFLOAT8_B,
        )
        self.ag_output_pbs["ATTN_FWD_PREFILL_AG"] = self.create_persistent_buffer(
            shape=[1, 1, self.seq_len, 8192],
            mem_config=self.model_config["DEFAULT_MEMCFG"],
            dtype=ttnn.DataType.BFLOAT8_B,
        )

        self.rs_intermediate_pbs["MLP_FWD_PREFILL_RS"] = self.create_persistent_buffer(
            shape=[1, 1, self.seq_len, 8192],
            mem_config=self.model_config["DEFAULT_MEMCFG"],
            dtype=ttnn.DataType.BFLOAT8_B,
        )
        self.rs_output_pbs["MLP_FWD_PREFILL_RS"] = self.create_persistent_buffer(
            shape=[1, 1, self.seq_len, 1024],
            mem_config=self.model_config["DEFAULT_MEMCFG"],
            dtype=ttnn.DataType.BFLOAT8_B,
        )

        self.ag_output_pbs["MODEL_FWD_PREFILL_AG"] = self.create_persistent_buffer(
            shape=[1, 1, self.seq_len, 8192],
            mem_config=self.model_config["DEFAULT_MEMCFG"],
            dtype=ttnn.DataType.BFLOAT8_B,
        )

        # decode
        shard_shape = [32, 256]
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.ag_output_pbs["DECODER_FWD_DECODE_AG"] = self.create_persistent_buffer(
            shape=[1, 1, 32, 8192],
            mem_config=mem_config,
            dtype=ttnn.DataType.BFLOAT8_B,
        )
        print("DECODER_FWD_DECODE_AG mem_cfg: ", self.ag_output_pbs["DECODER_FWD_DECODE_AG"].memory_config())

        self.ag_output_pbs["ATTN_FWD_DECODE_AG"] = self.create_persistent_buffer(
            shape=[1, 1, 32, 8192],
            mem_config=mem_config,
            dtype=ttnn.DataType.BFLOAT8_B,
        )

        self.ag_output_pbs["MODEL_FWD_DECODE_AG"] = self.create_persistent_buffer(
            shape=[1, 1, 32, 8192],
            mem_config=mem_config,
            dtype=ttnn.DataType.BFLOAT8_B,
        )

        self.rs_intermediate_pbs["MLP_FWD_DECODE_RS"] = self.create_persistent_buffer(
            shape=[1, 1, 32, 8192],
            mem_config=mem_config,
            dtype=ttnn.DataType.BFLOAT8_B,
            distributed=True,
        )
        shard_shape = [32, 32]
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_32_cores_grid,
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.rs_output_pbs["MLP_FWD_DECODE_RS"] = self.create_persistent_buffer(
            shape=[1, 1, 32, 1024],
            mem_config=mem_config,
            dtype=ttnn.DataType.BFLOAT8_B,
            distributed=True,
        )

    def close(self):
        self.mesh_device.reset_sub_device_stall_group()
