# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 attention program configurations."""

from dataclasses import dataclass

from models.demos.minimax_m3.tt.attention.config import ProgramConfig


@dataclass
class MiniMaxM3AttentionProgramConfig(ProgramConfig):
    """
    MiniMax-M3 prefill attention configuration.

    Uses TTNN auto-tuning for matmuls (cores=None).
    """

    # Prefill SDPA chunk sizes
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Matmul configs - None = auto-optimize (recommended)
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None
