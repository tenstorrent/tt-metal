# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm
from models.tt_transformers.tt.common import Mode


class DistributedNorm(LightweightModule):
    def __init__(
        self,
        norm,
        args,
        tt_ccl,
        prefetcher=None,
        TG=False,
        ag_config_key=None,
        enable_all_gather=True,
        use_fused_rms=False,
    ):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.ag_config_key = ag_config_key

        # Flag to control whether all_gather is performed after distributed norm (can be disabled when output should remain sharded)
        self.enable_all_gather = enable_all_gather

        # Fused RMS: combines norm + all-gather + residual add in one kernel
        self.use_fused_rms = use_fused_rms and not TG and args.is_multichip and prefetcher is None
        if self.use_fused_rms:
            cluster_shape = list(args.mesh_device.shape)
            self.fused_cluster_axis = 0 if cluster_shape[0] > 1 and cluster_shape[1] == 1 else 1
            dim_per_device = args.dim // cluster_shape[self.fused_cluster_axis]
            # Find core grid where dim_per_device / num_cores is tile-aligned and fits BH grid (12x10)
            max_x, max_y = 12, 10
            grid_x, grid_y, num_cores = 1, 1, 1
            for n in [40, 20, 10, 8, 5, 4, 2, 1]:
                if dim_per_device % n == 0 and (dim_per_device // n) % 32 == 0:
                    # Find a grid that fits in max_x × max_y
                    found = False
                    for gy in range(1, max_y + 1):
                        if n % gy == 0:
                            gx = n // gy
                            if gx <= max_x:
                                grid_x, grid_y, num_cores = gx, gy, n
                                found = True
                                break
                    if found:
                        break
            shard_width = dim_per_device // num_cores
            self.fused_ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(grid_x, grid_y),
                subblock_w=min(shard_width // 32, 4),
                block_h=1,
                block_w=shard_width // 32,
                inplace=False,
            )
            self.fused_input_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, dim_per_device),
                core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            # Create dedicated semaphore for fused RMS (single, not a list)
            sub_device_crs = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
            )
            self.fused_semaphore = ttnn.create_global_semaphore(args.mesh_device, sub_device_crs, 0)

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

    def forward(self, x, mode: Mode, norm_config=None, residual=None):
        """Apply a norm, possibly gathering inputs if required.
        If residual is provided and use_fused_rms is enabled, fuses norm + all-gather + residual add.
        """

        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None

        # Fused RMS path: norm + all-gather + residual add in one kernel
        if self.use_fused_rms and mode == Mode.DECODE and residual is not None:
            x = ttnn.to_memory_config(x, self.fused_input_mem_cfg)
            tt_out = ttnn.fused_rms_minimal(
                x,
                self.fused_ln_prg_cfg,
                self.fused_cluster_axis,
                self.args.mesh_device,
                self.fused_semaphore,
                topology=self.args.ccl_topology(),
                residual_input_tensor=residual,
                num_links=self.tt_ccl.get_num_links(self.fused_cluster_axis),
                epsilon=self.norm.eps,
                weight=self.norm.weight_distributed,
                memory_config=sharded_output_config,
            )
            return tt_out

        if self.TG:
            if mode == Mode.DECODE:
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

        input_mem_cfg = sharded_output_config if mode == Mode.DECODE else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.args.model_config[self.ag_config_key]["num_links"]
                if self.ag_config_key and mode == "decode"
                else self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=self.args.model_config[self.ag_config_key]["chunks_per_sync"]
                if self.ag_config_key and mode == "decode"
                else 10,
                num_workers_per_link=self.args.model_config[self.ag_config_key]["num_workers_per_link"]
                if self.ag_config_key and mode == "decode"
                else 2,
                num_buffers_per_channel=2,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        x = self.norm(
            x, mode=mode, in_sharded=(mode == Mode.DECODE), out_sharded=(mode == Mode.DECODE), norm_config=norm_config
        )

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode) and self.enable_all_gather:
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
