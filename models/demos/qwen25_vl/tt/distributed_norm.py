# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.qwen25_vl.tt.model_config import find_qwen_vl_width_grid
from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm


class QwenVLDistributedNorm(DistributedNorm):
    def __init__(self, norm, args, tt_ccl, prefetcher=None, TG=False, ag_config_key=None, enable_all_gather=True):
        if not TG:
            super().__init__(norm, args, tt_ccl, prefetcher, TG, ag_config_key, enable_all_gather)
            return

        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.ag_config_key = ag_config_key
        self.enable_all_gather = enable_all_gather

        hidden_size_per_device_distributed_ln = args.dim // args.cluster_shape[1]
        core_grid_ln = find_qwen_vl_width_grid(
            hidden_size_per_device_distributed_ln,
            ttnn.TILE_SIZE,
            max_rows=4,
            max_cols=8,
        )
        num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
        self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
            core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
        )
        self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
            subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // ttnn.TILE_SIZE,
            block_h=1,
            block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // ttnn.TILE_SIZE,
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

    def forward(self, x, mode: Mode, norm_config=None):
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
            return tt_distributed_rmsnorm(
                x,
                epsilon=self.norm.eps,
                gamma=self.norm.weight_distributed,
                mesh_device=self.args.mesh_device,
                tt_ccl=self.tt_ccl,
                compute_kernel_config=self.ln_cfg,
            )

        return super().forward(x, mode, norm_config)
