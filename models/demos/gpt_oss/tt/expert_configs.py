# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
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
    decode_gate_up_cores: tuple[int, int] = (3, 5)
    decode_gate_up_in0_block_w: int = 30
    decode_down_cores: tuple[int, int] = (5, 6)
    decode_down_in0_block_w: int = 9

    # Prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 5)
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_cores: tuple[int, int] = (5, 6)
    prefill_down_in0_block_w: int = 9

    # Memory
    sequence_chunk_size: int = 4 * 1024
    down_split_size: int = 1024
