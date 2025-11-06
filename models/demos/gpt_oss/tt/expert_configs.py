# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific expert program configurations.

This file defines the program configurations optimized for GPT-OSS architecture.
Other models can create similar files with their own configurations.
"""

from models.demos.gpt_oss.tt.experts.config import ProgramConfig

# GPT-OSS uses the default ProgramConfig values, so we can just use it directly
# or create an alias for clarity
GPTOSSProgramConfig = ProgramConfig


# Example: If you need custom settings for a specific variant
def get_gptoss_large_config():
    """
    Example: GPT-OSS with larger grids for bigger models.

    Usage:
        program_config = get_gptoss_large_config()
        experts = Experts(..., program_config=program_config)
    """
    return ProgramConfig(
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
    return ProgramConfig(
        decode_gate_up_cores=(3, 4),
        decode_down_cores=(5, 6),
        sequence_chunk_size=2 * 1024,  # Smaller chunks
        down_split_size=512,  # More splits
    )
