# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, tt_ccl, TG=False):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl

        if TG:
            core_grid_ln = (
                min(4, args.dim // 4 // 32 // 8),
                8,
            )  # dividing by 4 and 8 for num_cols and num_rows of mesh, and 32 for tile size
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            hidden_size_per_device_distributed_ln = args.dim // 4
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
                core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        self.TG = TG
        self.norm_gather_buffer = None

    def _ensure_norm_gather_buffer(self, shape, dtype, layout, memory_config):
        shape = tuple(shape)
        if self.norm_gather_buffer is None or tuple(self.norm_gather_buffer.shape) != shape:
            if self.norm_gather_buffer is not None:
                ttnn.deallocate(self.norm_gather_buffer)
            self.norm_gather_buffer = ttnn.empty(
                shape,
                dtype=dtype,
                layout=layout,
                device=self.args.mesh_device,
                memory_config=memory_config,
            )
        return self.norm_gather_buffer

    def forward(self, x, mode):
        """Apply a norm, possibly gathering inputs if required."""
        if self.TG:
            if mode == "decode":
                return tt_sharded_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                    ln_sharded_progcfg=self.ln_prg_cfg,
                    ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                )
            else:
                return tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    compute_kernel_config=self.ln_cfg,
                )

        input_mem_cfg = self.norm.sharded_output_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Ensure x is on device with correct memory config
        x = ttnn.to_device(x, self.args.mesh_device, memory_config=input_mem_cfg)

        # If norm is not distributed and we are multichip, gather input before norm
        cluster_axis = 0 if self.args.is_galaxy else None
        num_links = 2 if self.args.is_galaxy else 1
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            # Allocate reusable buffer for all_gather_async output
            output_shape = list(x.shape)
            output_shape[3] *= self.args.num_devices  # Gather along dim=3
            persistent_buffer = self._ensure_norm_gather_buffer(output_shape, x.dtype, x.layout, input_mem_cfg)
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=persistent_buffer,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=num_links,
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                cluster_axis=cluster_axis,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        x = self.norm(x, mode=mode, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))
        return x
