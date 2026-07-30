# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS expert program configurations."""

from dataclasses import dataclass

from models.demos.gpt_oss.tt.experts.config import ProgramConfig


@dataclass
class GPTOSSProgramConfig(ProgramConfig):
    """
    GPT-OSS expert configuration.

    Optimized for: hidden=2088, intermediate=360
    """

    # Decode
    # Grids tuned via models/demos/gpt_oss/tests/sweeps/moe_sparse_matmul_sweep.py on
    # a single p150 (TP=1).
    # Re-swept after fixing out_block_w (was hardcoded 1, which made
    # in1_num_subblocks = out_block_w/out_subblock_w == 0, so out_subblock_w was a
    # no-op). With the axis live, 60-config sweeps per projection found:
    #   gate_up: 30 cores (10x3), ib=15, osw=3 -> 0.1709ms (was 0.1935 @ ib45/osw1)
    #   down:    45 cores (9x5),  ib=30, osw=2 -> 0.1764ms (was 0.1795)
    # In-model: SparseMatmul 6.458 -> 6.147 ms/tok, decode 16.178 -> 15.846 ms/tok,
    # 57.95-58.11 -> 58.93-59.16 tok/s/u (n=4, no overlap). Top-1 0.9333 -> 0.9667,
    # Top-5 1.0000 unchanged.
    # SAFETY: both keep num_blocks_w_dim = per_core_N/out_block_w = 1, which avoids
    # the PR #51514 wide-subblock corruption (needs osw>1 AND nbwd>1 AND nbid>1).
    decode_gate_up_cores: tuple[int, int] = (10, 3)
    decode_gate_up_in0_block_w: int = 15
    decode_gate_up_subblock_w: int = 3
    decode_down_cores: tuple[int, int] = (9, 5)
    decode_down_in0_block_w: int = 30
    decode_down_subblock_w: int = 2

    # Prefill
    # Same 2880x2880 expert GEMM as decode (Nt=90); on single p150 (TP=1) the old
    # galaxy grids (12/30 cores) left prefill sparse_matmul at ~7.3ms/3.9ms per op
    # (tracy: 888ms = ~80% of device time). Match decode's 90-core (10x9) grid.
    prefill_gate_up_cores: tuple[int, int] = (10, 9)
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_cores: tuple[int, int] = (10, 9)
    prefill_down_in0_block_w: int = 12

    # Memory
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024
