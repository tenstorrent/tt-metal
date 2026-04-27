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
    decode_gate_up_cores: tuple[int, int] = (3, 4)
    decode_gate_up_in0_block_w: int = 30
    decode_down_cores: tuple[int, int] = (5, 6)
    # The down-projection K dim is intermediate_size_per_device. With TP=4 (P300x2
    # mesh (1,4)) that is 720 -> Kt=23 (prime); the sparse_matmul kernel asserts
    # Kt % in0_block_w == 0, so 12 (the T3K-tuned value where Kt=12) fails. 1 is
    # the only divisor of 23 <= 12, slightly slower on T3K but correct on both.
    decode_down_in0_block_w: int = 23

    # Prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_cores: tuple[int, int] = (5, 6)
    prefill_down_in0_block_w: int = 23

    # Memory
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024
