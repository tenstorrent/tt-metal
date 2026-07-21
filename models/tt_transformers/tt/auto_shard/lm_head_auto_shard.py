# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Auto-shard LM head: the stock LMHead with its tail all-reduce removed.

The weight is vocab-sharded (ShardTensorToMesh(dim=-1), k = full hidden), and the auto-shard stack
hands it a *replicated* full-hidden input, so each device computes the correct, complete logits for
its own vocab slice -- there are no partial sums to combine. The stock LMHead ends with a
`tt_all_reduce(cluster_axis=1, dim=3 if is_galaxy else 0)`; on a flat (non-galaxy) mesh that reduce is
either a degenerate no-op (1xN line: scatter a size-1 batch dim) or outright wrong (2x2: it sums
distinct vocab slices across axis 1). It only exists for the galaxy layout, where axis 1 is a
replication axis orthogonal to the vocab shard.

Assembling the full logits is done downstream exactly as for the stock model, on any mesh:
  * decode host sampling: model.ttnn_decode_forward all_gathers dim=3 over the whole mesh
    (cluster_axis=None) so every device holds full logits;
  * prefill: concat_host_output stitches the per-device vocab shards on the host;
  * on-device sampling: consumes the vocab-sharded logits directly.

So this head just drops the tail reduce and returns the vocab-sharded logits. Rebinding model.LMHead
(see model_auto_shard) makes Transformer.__init__ build this instead of the stock one.
"""

import ttnn
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead


class LMHeadAutoShard(LMHead):
    def forward(self, x: ttnn.Tensor, debug_input_torch=None, debug_weight_torch=None):
        use_prefetcher = self.prefetcher is not None and self.prefetcher.mode == Mode.DECODE
        split_sizes = self.split_sizes_ring_mm if use_prefetcher else self.split_sizes_dram_sharded
        program_configs = [
            self.args.get_lm_head_program_config(split_size, self.prefetcher if use_prefetcher else None)
            for split_size in split_sizes
        ]
        output_weights = self.output_weights_ring_mm if use_prefetcher else self.output_weights_dram_sharded

        self.lm_head_output_memory_config = self.args.get_lm_head_output_mem_config(
            Mode.DECODE if use_prefetcher else Mode.PREFILL, self.prefetcher if use_prefetcher else None
        )

        outputs = []
        for i, (weight, pc) in enumerate(zip(output_weights, program_configs)):
            output = ttnn.linear(
                x,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc,
                memory_config=self.lm_head_output_memory_config,
                dtype=self.args.lm_head_dtype if hasattr(self.args, "lm_head_dtype") else ttnn.bfloat8_b,
                sub_device_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
            )
            output = ttnn.to_memory_config(
                output,
                memory_config=self.args.get_lm_head_sharded_output_mem_config(
                    self.prefetcher if use_prefetcher else None
                ),
            )
            outputs.append(output)

        ttnn.deallocate(x)

        # Concatenate this device's weight-splits into its full vocab slice.
        output = ttnn.concat(
            outputs,
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG if not use_prefetcher else ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=self.prefetcher.all_worker_cores_range_set if use_prefetcher else None,
        )
        if use_prefetcher:
            output = ttnn.to_memory_config(
                output,
                memory_config=self.args.get_lm_head_reshard_mem_config(self.prefetcher),
            )

        # No tail all-reduce: the vocab shards are assembled downstream (see module docstring).
        return output
