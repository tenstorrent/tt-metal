# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The chunk outputs and the rank's final state from the summary's emitted terms, once the SP prefix is known
(ttnn.experimental.kda.entry_correction): every chunk's output from its entry state, all chunks in parallel.

Each head's chunks are spread over `cores_per_head` cores; the summary writes its emitted terms sharded so each core's
chunk rows are in its own L1 (terms_memory_config).
"""

from __future__ import annotations

import ttnn


class EntryCorrection:
    """The correction's cores and memory layout for one mesh and geometry (one group per head)."""

    def __init__(
        self,
        mesh_device,
        *,
        sequence_parallel_axis: int,
        batch_heads: int,
        num_chunks: int,
        key_dim: int,
        local_rows: int,
        cores: list,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    ):
        self.axis = sequence_parallel_axis
        self.local_rows = local_rows
        if len(cores) < batch_heads:
            raise ValueError(f"entry correction needs a core per head: {batch_heads} heads, {len(cores)} cores")
        # Cores per head: the most that divide the head's chunks evenly (so each core's chunk rows are one shard).
        self.cores_per_head = max(
            d for d in range(1, max(1, min(num_chunks, len(cores) // batch_heads)) + 1) if num_chunks % d == 0
        )
        self.cores = cores[: batch_heads * self.cores_per_head]
        core_set = ttnn.CoreRangeSet([]).merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores]))
        per_core = num_chunks // self.cores_per_head
        # The emitted terms land in the L1 of the core that consumes them: each core takes whole chunks (cores_per_head
        # divides them), so a core's shard is its chunk rows and the correction reads nothing over the NoC for them.
        self.terms_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_set, (per_core * ttnn.TILE_SIZE, key_dim), ttnn.ShardOrientation.ROW_MAJOR),
        )
        # The products accumulate in an FP32 DST at the scan's fidelity.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True
        )
        # Where the summary writes the emitted terms, and where the chunk outputs go.
        self.memory_config = ttnn.L1_MEMORY_CONFIG
        self.output_memory_config = ttnn.L1_MEMORY_CONFIG

    def __call__(self, zero_output, identity_output, final_a, final_b, entry, tail_entry, actual_start, actual_end):
        """Return (output [BH, NC, 32, V] BF16, final state [BH, K, V] FP32)."""
        return ttnn.experimental.kda.entry_correction(
            zero_output,
            identity_output,
            final_a,
            final_b,
            entry,
            tail_entry,
            actual_start=actual_start,
            local_rows=self.local_rows,
            cores=self.cores,
            actual_end=actual_end,
            sequence_parallel_axis=self.axis,
            memory_config=self.output_memory_config,
            compute_kernel_config=self.compute_config,
        )
