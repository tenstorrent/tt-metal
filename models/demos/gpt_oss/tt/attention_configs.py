# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS attention program configurations."""

from dataclasses import dataclass

from models.demos.gpt_oss.tt.attention.config import ProgramConfig


@dataclass
class GPTOSSAttentionProgramConfig(ProgramConfig):
    """
    GPT-OSS attention configuration.

    Optimized for: hidden=2088, heads=84, head_dim=64
    Uses TTNN auto-tuning for matmuls (cores=None).
    """

    # SDPA chunk sizes
    decode_k_chunk_size: int = 128
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Matmul configs - None = auto-optimize (recommended)
    decode_qkv_cores: tuple[int, int] | None = None
    decode_out_cores: tuple[int, int] | None = None
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None
