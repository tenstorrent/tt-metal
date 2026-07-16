# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_70b_galaxy.tt.llama_ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, tt_ccl=None, ccl_topology=None, use_sharded_decode=True):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.ccl_topology = ccl_topology
        self.use_sharded_decode = use_sharded_decode
        if args.qk_norm:
            core_grid_ln, grid_offset = (5, 2), ttnn.CoreCoord(1, 0)
        else:
            core_grid_ln, grid_offset = (8, 2), ttnn.CoreCoord(2, 0)
        core_range = ttnn.CoreRange(
            grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
        )
        num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
        hidden_size_per_device_distributed_ln = args.dim // 4
        # decode_shard_height (and thus gather_in_mem_cfg / ln_prg_cfg.block_h below) is only consumed
        # by the sharded-decode path (tt_sharded_distributed_rmsnorm), which runs when
        # use_sharded_decode=True. On that path decode_shard_height is always 32; the non-32 values
        # derived below are inert (the no-prefetcher decode uses the distributed, non-sharded norm).
        if not args.blackhole_no_prefetcher:
            # Wormhole / prefetcher path keeps main's fixed decode shard height (32 -> block_h 1).
            decode_shard_height = 32
        elif norm.output_mem_config is not None and norm.output_mem_config.shard_spec is not None:
            decode_shard_height = norm.output_mem_config.shard_spec.shape[0]
        else:
            decode_residual_memcfg = args.get_model_config().get("DECODE_RESIDUAL_MEMCFG", None)
            if decode_residual_memcfg is not None and decode_residual_memcfg.shard_spec is not None:
                decode_shard_height = decode_residual_memcfg.shard_spec.shape[0]
            else:
                decode_shard_height = 128 if args.is_blackhole else 32
        # block_h = decode_shard_height // 32 truncates silently otherwise.
        assert decode_shard_height % 32 == 0, f"decode_shard_height must be a multiple of 32, got {decode_shard_height}"
        self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(1, 1, decode_shard_height, hidden_size_per_device_distributed_ln // num_cores_ln),
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
            block_h=decode_shard_height // 32,
            block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
            inplace=False,
        )
        self.ln_sharded_stats_memcfg = None
        # self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
        #     shape=[1, 1, 32, 32 * 4],
        #     core_grid=ttnn.CoreGrid(y=1, x=1),
        #     strategy=ttnn.ShardStrategy.WIDTH,
        # )
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

    def forward(self, x, res, mode):
        """Apply a norm, possibly gathering inputs if required."""
        # On the BH no-prefetch path the residual stream is bf16 (kept high-precision to avoid bf8
        # accumulation error over 64 layers). The norm output only feeds matmuls and is not part of
        # the residual, so force it to bf8 to keep activation/CB footprint small (avoids L1 clashes
        # at long prefill sequence lengths). Other paths keep their input-derived dtype (None).
        norm_output_dtype = ttnn.bfloat8_b if self.args.blackhole_no_prefetcher else None
        if mode == "decode":
            if not self.use_sharded_decode:
                # BH no-prefetch decode. The residual stream is column-fractured (dim/4 per
                # device), so a plain local rms_norm would (incorrectly) normalize over only
                # dim/4. Add the residual, then run the distributed RMS norm: per-device partial
                # stats are gathered across columns and combined so the normalization is over the
                # full hidden dim. That column-axis stats all_gather runs on device over the 2D-torus
                # fabric (see TT_CCL.line_all_gather -> ttnn.all_gather on the no-prefetch branch).
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                if res is not None:
                    res = ttnn.to_memory_config(res, ttnn.DRAM_MEMORY_CONFIG)
                    x = ttnn.add(x, res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                x, _ = tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    compute_kernel_config=self.ln_cfg,
                    tt_ccl=self.tt_ccl,
                    output_dtype=norm_output_dtype,
                )
                if self.norm.output_mem_config is not None:
                    x = ttnn.to_memory_config(x, self.norm.output_mem_config)
                return x, None
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
                output_dtype=norm_output_dtype,
            )
