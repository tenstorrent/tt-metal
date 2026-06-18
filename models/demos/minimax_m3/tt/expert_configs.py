# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M2 expert program configurations."""

from dataclasses import dataclass

from models.demos.minimax_m3.tt.experts.config import ProgramConfig


@dataclass
class MiniMaxM3ExpertProgramConfig(ProgramConfig):
    """MiniMax-M2 prefill expert configuration."""

    # Prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_cores: tuple[int, int] = (5, 6)
    prefill_down_in0_block_w: int = 12

    # Memory
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024
