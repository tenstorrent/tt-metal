# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.grok.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, tt_ccl):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl

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

    def forward(self, x, mode):
        """Apply a norm, possibly gathering inputs if required."""
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
