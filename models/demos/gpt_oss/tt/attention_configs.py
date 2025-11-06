# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific attention configurations.

This file contains model-specific SDPA program configurations that can be
customized without modifying the generic attention module.
"""

from models.demos.gpt_oss.tt.attention.config import ProgramConfig

# GPT-OSS uses the default ProgramConfig values
GPTOSSAttentionProgramConfig = ProgramConfig


# Example: Variant with different chunk sizes for larger models
def get_gptoss_large_attention_config():
    """
    GPT-OSS configuration for larger models.

    Uses larger chunk sizes for better performance on bigger models.
    """
    return ProgramConfig(
        decode_k_chunk_size=256,
        prefill_q_chunk_size_large=512,
        prefill_k_chunk_size_large=512,
    )


# Example: Memory-optimized configuration
def get_gptoss_memory_optimized_config():
    """
    GPT-OSS configuration optimized for memory-constrained environments.

    Uses smaller chunk sizes to reduce memory usage.
    """
    return ProgramConfig(
        decode_k_chunk_size=64,
        prefill_q_chunk_size_small=16,
        prefill_k_chunk_size_small=16,
        prefill_q_chunk_size_large=128,
        prefill_k_chunk_size_large=128,
    )


# Example: Performance-optimized configuration
def get_gptoss_performance_config():
    """
    GPT-OSS configuration optimized for maximum performance.

    Uses larger chunk sizes and higher precision for best quality.
    """
    return ProgramConfig(
        decode_k_chunk_size=256,
        prefill_q_chunk_size_small=64,
        prefill_k_chunk_size_small=64,
        prefill_q_chunk_size_large=512,
        prefill_k_chunk_size_large=512,
        prefill_threshold=1024,  # Lower threshold for large chunks
    )
