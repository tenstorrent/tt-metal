# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific attention program configurations.

This file contains model-specific SDPA and matmul program configurations
optimized for GPT-OSS architecture.
"""

from dataclasses import dataclass

from models.demos.gpt_oss.tt.attention.config import ProgramConfig


@dataclass
class GPTOSSAttentionProgramConfig(ProgramConfig):
    """
    GPT-OSS optimized attention configuration.

    Optimized for:
    - Wormhole architecture
    - 4x8 or 8x8 mesh configurations
    - GPT-OSS model dimensions (hidden_size=4096, num_heads=32, head_dim=128)
    """

    # SDPA configurations (keep defaults, already optimized)
    decode_q_chunk_size: int = 0
    decode_k_chunk_size: int = 128

    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Matmul program configs for QKV and output projections
    # These are optional - set to None to use TTNN defaults
    # For GPT-OSS, we leave as None to let TTNN optimize automatically
    decode_qkv_cores: tuple[int, int] | None = None
    decode_out_cores: tuple[int, int] | None = None
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None


# For models that want explicit matmul control
@dataclass
class GPTOSSAttentionWithExplicitMatmuls(ProgramConfig):
    """
    GPT-OSS attention with explicit matmul program configs.

    Use this if you want fine-grained control over QKV and output projection matmuls.
    """

    # SDPA configs
    decode_k_chunk_size: int = 128
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256

    # Decode QKV projection (1x32x4096 @ 4096x12288)
    decode_qkv_cores: tuple[int, int] = (8, 8)
    decode_qkv_in0_block_w: int = 4  # 4096 / 32 / 32 = 4
    decode_qkv_out_subblock_w: int = 2

    # Decode output projection (1x32x4096 @ 4096x4096)
    decode_out_cores: tuple[int, int] = (8, 8)
    decode_out_in0_block_w: int = 4
    decode_out_out_subblock_w: int = 2

    # Prefill QKV projection
    prefill_qkv_cores: tuple[int, int] = (8, 8)
    prefill_qkv_in0_block_w: int = 4
    prefill_qkv_out_subblock_w: int = 2

    # Prefill output projection
    prefill_out_cores: tuple[int, int] = (8, 8)
    prefill_out_in0_block_w: int = 4
    prefill_out_out_subblock_w: int = 2


def get_gptoss_large_attention_config():
    """
    Configuration for larger GPT-OSS models or longer sequences.

    Usage:
        program_config = get_gptoss_large_attention_config()
        attention = Attention(..., program_config=program_config)
    """
    return GPTOSSAttentionProgramConfig(
        decode_k_chunk_size=256,
        prefill_q_chunk_size_large=512,
        prefill_k_chunk_size_large=512,
    )


def get_gptoss_memory_optimized_config():
    """
    Memory-optimized configuration for constrained environments.

    Usage:
        program_config = get_gptoss_memory_optimized_config()
        attention = Attention(..., program_config=program_config)
    """
    return GPTOSSAttentionProgramConfig(
        decode_k_chunk_size=64,
        prefill_q_chunk_size_small=16,
        prefill_k_chunk_size_small=16,
        prefill_q_chunk_size_large=128,
        prefill_k_chunk_size_large=128,
    )
