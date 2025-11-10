# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific expert program configurations.

This file defines the program configurations optimized for GPT-OSS architecture.
Other models can create similar files with their own configurations.
"""

from dataclasses import dataclass

from models.demos.gpt_oss.tt.experts.config import ProgramConfig


@dataclass
class GPTOSSProgramConfig(ProgramConfig):
    """GPT-OSS specific program configuration with optimized in0_block_w values."""

    # GPT-OSS specific input block widths (optimized for performance)
    decode_gate_up_in0_block_w: int = 30
    decode_down_in0_block_w: int = 12
    prefill_gate_up_in0_block_w: int = 30
    prefill_down_in0_block_w: int = 12


# Example: If you need custom settings for a specific variant
def get_gptoss_large_config():
    """
    Example: GPT-OSS with larger grids for bigger models.

    Usage:
        program_config = get_gptoss_large_config()
        experts = Experts(..., program_config=program_config)
    """
    return GPTOSSProgramConfig(
        decode_gate_up_cores=(4, 6),
        decode_down_cores=(6, 8),
        prefill_gate_up_cores=(4, 6),
        prefill_down_cores=(6, 8),
    )


def get_gptoss_memory_optimized_config():
    """
    Example: GPT-OSS with smaller chunks for memory-constrained environments.

    Usage:
        program_config = get_gptoss_memory_optimized_config()
        experts = Experts(..., program_config=program_config)
    """
    return GPTOSSProgramConfig(
        decode_gate_up_cores=(3, 4),
        decode_down_cores=(5, 6),
        sequence_chunk_size=2 * 1024,  # Smaller chunks
        down_split_size=512,  # More splits
    )
