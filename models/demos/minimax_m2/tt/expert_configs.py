# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M2 expert program configurations."""

from dataclasses import dataclass

from models.demos.minimax_m2.tt.experts.config import ProgramConfig


@dataclass
class MiniMaxM2ExpertProgramConfig(ProgramConfig):
    """
    MiniMax-M2 expert configuration.

    Optimized for: hidden=2088, intermediate=360
    """

    # Decode
    decode_gate_up_cores: tuple[int, int] = (3, 4)
    decode_gate_up_in0_block_w: int = 30
    decode_down_cores: tuple[int, int] = (5, 6)
    decode_down_in0_block_w: int = 12

    # Prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_cores: tuple[int, int] = (5, 6)
    prefill_down_in0_block_w: int = 12

    # Memory
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024
