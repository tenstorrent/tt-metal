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
        self.use_fused_rms_norm = True

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
        # Stats buffer for fused_rms_minimal is shared via tt_ccl.stats_buffer

    def forward(self, x, mode):
        """Apply a norm, possibly gathering inputs if required."""

        if mode == "prefill" and self.TG:
            return tt_distributed_rmsnorm(
                x,
                epsilon=self.norm.eps,
                gamma=self.norm.weight_distributed,
                mesh_device=self.args.mesh_device,
                tt_ccl=self.tt_ccl,
                compute_kernel_config=self.ln_cfg,
            )

        input_mem_cfg = self.norm.sharded_output_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Decode mode with fused RMS norm
        if mode == "decode" and self.use_fused_rms_norm:
            # fused_rms_minimal does internal all-gather of STATS (for computing norm)
            # but the OUTPUT is still sharded (1024-wide per device)
            # We need to all_gather the output data to get full hidden dim (4096)
            x = tt_sharded_distributed_rmsnorm(
                x,
                epsilon=self.norm.eps,
                gamma=self.norm.weight_distributed,
                mesh_device=self.args.mesh_device,
                tt_ccl=self.tt_ccl,
                ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                ln_sharded_progcfg=self.ln_prg_cfg,
                ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                output_mem_config=self.gather_in_mem_cfg,
                use_fused_rms_norm=self.use_fused_rms_norm,
                stats_buffer=self.tt_ccl.stats_buffer,
            )
            # All-gather the normalized output data (fused op only gathers stats, not data)
            cluster_axis = 1
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=1,
                cluster_axis=cluster_axis,
                topology=self.args.ccl_topology(),
                memory_config=self.norm.sharded_output_config,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            return x

        # Prefill mode OR decode mode without fused norm: use standard path
        # First, all_gather if needed (multi-chip, non-distributed norm)
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        x = self.norm(x, mode=mode, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        return x
