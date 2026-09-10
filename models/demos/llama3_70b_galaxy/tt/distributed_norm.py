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
        # Blackhole prefetcher path: columns 1-3 are the dram_prefetcher's global-CB receiver cores
        # and columns 5-6 hold the persistent CCL buffers (sub_core_grids). The default LN grid on
        # columns 1-2 puts the LayerNorm static CB on top of the resident global CB, which only shows
        # up during trace capture (all programs resident at once) as a small CB/L1 clash. Move the norm
        # grid onto free worker columns (7+) that carry no resident prefetcher/CCL L1. Same core count
        # and shard math, just a different origin, so gather_in_mem_cfg / ln_prg_cfg / stats configs
        # below all follow automatically.
        if getattr(args, "is_blackhole", False) and getattr(args, "use_prefetcher", False):
            grid_offset = ttnn.CoreCoord(7, 0)
        core_range = ttnn.CoreRange(
            grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
        )
        num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
        hidden_size_per_device_distributed_ln = args.dim // 4
        # gather_in_mem_cfg / ln_prg_cfg are consumed only by the sharded-decode path
        # (tt_sharded_distributed_rmsnorm in forward), which runs iff use_sharded_decode=True. On that
        # path the decode shard height is always 32 (block_h=1), exactly as on main. The no-prefetcher
        # decode uses the non-sharded distributed norm and never reads these, so only build them when
        # they can actually be used (avoids computing an unused, misleading shard height).
        if self.use_sharded_decode:
            decode_shard_height = 32  # fixed on the sharded path; block_h = 1
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
        else:
            # No-prefetcher decode uses tt_distributed_rmsnorm; these are never consumed.
            self.gather_in_mem_cfg = None
            self.ln_prg_cfg = None
        # Sharded stats config for the unfused BH-prefetcher distributed norm. rms_norm_post_all_gather
        # with a sharded program config requires the gathered stats to be sharded too. Place the single
        # 32x(32*4) stats shard on the first worker core of the LN grid (grid_offset, e.g. (1,0)) so it
        # never lands on the prefetcher sender column. Only needed/consumed on the BH unfused path.
        if self.use_sharded_decode:
            self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, 32 * 4),
                core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(grid_offset, grid_offset)}),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            self.ln_sharded_stats_memcfg = None
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
        # BH prefetcher path: use_sharded_decode is True but the fused RMSAllGather hangs on the 2D-torus
        # fabric, so fall through to the unfused distributed rmsnorm (stable all_gather) and reshard to
        # the sharded output the downstream matmul expects. Only triggers on BH + sharded-decode.
        bh_unfused_norm = (
            mode == "decode"
            and self.use_sharded_decode
            and getattr(self.args, "is_blackhole", False)
            and getattr(self.args, "use_unfused_ccl", False)
        )
        if mode == "decode":
            if (not self.use_sharded_decode) or bh_unfused_norm:
                # BH no-prefetch decode. The residual stream is column-fractured (dim/4 per
                # device), so a plain local rms_norm would (incorrectly) normalize over only
                # dim/4. Add the residual, then run the distributed RMS norm: per-device partial
                # stats are gathered across columns and combined so the normalization is over the
                # full hidden dim. That column-axis stats all_gather runs on device over the 2D-torus
                # fabric (see TT_CCL.line_all_gather -> ttnn.all_gather on the no-prefetch branch).
                #
                # On the BH prefetcher path (bh_unfused_norm) the resident dram_prefetcher owns the
                # sender column, so the norm compute must be confined to the worker sub-device grid.
                # Reshard the input onto gather_in_mem_cfg (LN grid origin, cols 7+ on BH prefetcher)
                # and hand the sharded LayerNorm
                # program config to the pre/post ops so their compute never lands on a sender core.
                if bh_unfused_norm:
                    norm_input_memcfg = self.gather_in_mem_cfg
                    norm_program_config = self.ln_prg_cfg
                    norm_stats_memcfg = self.ln_sharded_stats_memcfg
                else:
                    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                    if res is not None:
                        res = ttnn.to_memory_config(res, ttnn.DRAM_MEMORY_CONFIG)
                        x = ttnn.add(x, res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    norm_input_memcfg = None
                    norm_program_config = None
                    norm_stats_memcfg = None
                x, _ = tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    compute_kernel_config=self.ln_cfg,
                    tt_ccl=self.tt_ccl,
                    output_dtype=norm_output_dtype,
                    force_stable_ag=bh_unfused_norm,
                    input_memcfg=norm_input_memcfg,
                    program_config=norm_program_config,
                    stats_memcfg=norm_stats_memcfg,
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
