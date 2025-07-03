# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_subdevices.tt.llama_ccl import tt_sharded_distributed_rmsnorm, tt_distributed_rmsnorm


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, TG=False, tt_ccl=None, ccl_topology=None):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.ccl_topology = ccl_topology

        if TG:
            core_grid_ln, grid_offset = (8, 2), ttnn.CoreCoord(1, 0)
            core_range = ttnn.CoreRange(
                grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
            )
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            hidden_size_per_device_distributed_ln = args.dim // 4
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device_distributed_ln // num_cores_ln),
                core_grid=ttnn.CoreRangeSet(
                    {
                        core_range,
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            self.ln_sharded_stats_memcfg = None
            # ttnn.create_sharded_memory_config(
            #     shape=[1, 1, 32, 32 * 4],
            #     core_grid=ttnn.CoreGrid(y=1, x=1),
            #     strategy=ttnn.ShardStrategy.WIDTH,
            # )
            self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        self.TG = TG

    def forward(self, x, res, mode):
        """Apply a norm, possibly gathering inputs if required."""
        if self.TG:
            if mode == "decode":
                return tt_sharded_distributed_rmsnorm(
                    x,
                    res,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                    ln_sharded_progcfg=self.ln_prg_cfg,
                    ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                    tt_ccl=self.tt_ccl,
                    output_mem_config=self.norm.output_mem_config,
                    ccl_topology=self.ccl_topology,
                )
            else:
                return tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    compute_kernel_config=self.ln_cfg,
                    tt_ccl=self.tt_ccl,
                )

        input_mem_cfg = self.norm.sharded_output_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.all_gather(x, dim=3, num_links=1, topology=self.args.ccl_topology(), memory_config=input_mem_cfg)
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        x = self.norm(x, mode=mode, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode):
            x = ttnn.all_gather(x, dim=3, num_links=1, topology=self.args.ccl_topology())

        return x
