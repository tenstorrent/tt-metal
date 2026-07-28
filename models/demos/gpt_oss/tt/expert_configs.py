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
    # a single p150 (TP=1). A WIDER re-sweep (adding the high in0_block_w axis the
    # original narrow {1,2,3,5} sweep missed) found in0_block_w=45 is the winning knob:
    #   gate_up: 36 cores (12x3), ib=45 -> 0.257ms  (was 0.38ms @ 90c/ib30, ~32% faster)
    #   down:    30 cores (10x3), ib=45 -> 0.176ms  (was 0.38ms @ 90c/ib12, ~54% faster)
    # Both PCC ~0.9935. Standalone medians; verify in-model via signpost tracy + checks.sh.
    decode_gate_up_cores: tuple[int, int] = (12, 3)
    decode_gate_up_in0_block_w: int = 45
    decode_down_cores: tuple[int, int] = (10, 3)
    decode_down_in0_block_w: int = 45

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
