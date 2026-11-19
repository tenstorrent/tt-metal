# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Coherent rectangular residual/MLP shards without intermediate DRAM restores."""
import math

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_gdn import GdnCandidate


class MovementCandidate(GdnCandidate):
    carry_mlp = False

    @classmethod
    def from_state_dict(cls, sd, **kw):
        self = super().from_state_dict(sd, **kw)
        if self.carry_mlp:
            cg = ttnn.CoreGrid(x=8, y=self.residual_cores // 8)

            def layout(width):
                sw = math.ceil(width / (32 * self.residual_cores))
                return (
                    ttnn.create_sharded_memory_config(
                        (32, sw * 32), cg, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
                    ),
                    sw,
                )

            for name, w in self.dram_weights.items():
                im, iw = layout(w.shape[-2])
                om, ow = layout(w.shape[-1])
                role = self.role_configs.get(name, {})
                readers = role.get("readers", self.decode_readers)
                self.dram_layouts[name] = (
                    im,
                    om,
                    ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=role.get("block", self.decode_block),
                        per_core_M=1,
                        per_core_N=ow,
                        num_workers_per_dram_bank=readers,
                        use_multicore_pipeline=readers > 1,
                    ),
                )
        return self

    def linear(self, x, name):
        if not self.carry_mlp or x.shape[2] != 1 or name not in self.dram_weights:
            return super().linear(x, name)
        im, om, cfg = self.dram_layouts[name]
        out = ttnn.linear(
            ttnn.to_memory_config(x, im),
            self.dram_weights[name],
            program_config=cfg,
            compute_kernel_config=self.matmul_compute(name),
            memory_config=om,
        )
        if name in ("packed_input", "gate_up"):
            return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        return out
