# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    # Prefill SDPA chunks (min_seq_len, q_chunk, k_chunk) for the full-attention and the sliding-window (128) layers.
    # From 4K up: the fastest config of a standalone sweep of the op at the 1x4 per-device shapes (16 Q / 2 KV heads,
    # head_dim 64, sink, bf16 Q, bf8 K/V, HiFi4) on the full Blackhole grid (11x10), 4K-64K. Below 4K the original
    # 32/32 and 256/256 stay: the run is host-bound there and smaller chunks gave no measurable TTFT gain.
    # Blackhole only: on other architectures get_prefill_sdpa_config ignores these tables and uses the base class
    # small/large rule (32/32 below 2048, 256/256 from 2048) on an 8x8 grid.
    prefill_chunk_regimes: tuple[tuple[int, int, int], ...] | None = (
        (0, 32, 32),
        (2048, 256, 256),
        (4096, 128, 512),
        (16384, 256, 1024),
    )
    prefill_sliding_chunk_regimes: tuple[tuple[int, int, int], ...] | None = (
        (0, 32, 32),
        (2048, 256, 256),
        (4096, 128, 128),
    )

    # Prefill SDPA (QK^T and PV matmuls) at HiFi2 on Blackhole; decode SDPA stays at math_fidelity (HiFi4).
    prefill_sdpa_math_fidelity: str | None = "HiFi2"

    # Matmul configs - None = auto-optimize (recommended)
    decode_qkv_cores: tuple[int, int] | None = None
    decode_out_cores: tuple[int, int] | None = None
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None
    # Prefill QKV / o_proj 2D multicast configs on Blackhole (attention/config.py prefill_proj_program_config).
    prefill_proj_2d: bool = True
