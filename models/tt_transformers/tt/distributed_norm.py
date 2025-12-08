# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, tt_ccl, TG=False, all_gather_config_key=None):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.all_gather_config_key = all_gather_config_key

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

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            # Get num_workers_per_link from config if available, otherwise use default
            num_workers = 1  # Default
            if self.all_gather_config_key and self.all_gather_config_key in self.args.model_config:
                num_workers = self.args.model_config[self.all_gather_config_key].get("num_workers_per_link", 1)

            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=4,
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=self.args.model_config[self.all_gather_config_key]["chunks_per_sync"]
                if self.all_gather_config_key
                else 10,
                num_workers_per_link=self.args.model_config[self.all_gather_config_key]["num_workers_per_link"]
                if self.all_gather_config_key
                else 1,
                num_buffers_per_channel=self.args.model_config[self.all_gather_config_key]["num_buffers_per_channel"]
                if self.all_gather_config_key
                else 2,
            )
            # 2 faktora
            # 4096 optimalna velicina paketa, tad je maks troughput (a. i  razlog), inace mozda/uvek opadne
            # 16 tiles po 2kb = 32kb. workload se deli ravnomerno medju workerima. Jednacina: nr_of_tiles(input per device)*size_of_tile / nr_workers == 4096
            # vise workers je losije zbog razloga gore i zbog multiplexing overheaad
            # a. slika ona druga i sta je tu round robin. nr_workers=1 je special case jer iz dram ide a ne cache, a inace cita iz cache. Kod nr_workers_per_link=1 imas prednost da nemas LATENCY ovo da se prebacuje
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
                num_links=1,
                topology=self.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=self.args.model_config[self.all_gather_config_key]["chunks_per_sync"],
                num_workers_per_link=self.args.model_config[self.all_gather_config_key]["num_workers_per_link"],
                num_buffers_per_channel=self.args.model_config[self.all_gather_config_key]["num_buffers_per_channel"],
            )

        return x
