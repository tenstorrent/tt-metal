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
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Matmul configs - None = auto-optimize (recommended)
    # decode qkv [32,2880]x[2880,5120]. IN0 is L1_INTERLEAVED in-model, which is the
    # layout the offline sweep used, so the sweep result applies (unlike o_proj,
    # whose IN0 is L1_WIDTH_SHARDED and takes a different program factory).
    # .auto/qkv_sweep.py: 8x5 cores, in0_block_w=2, out_subblock_w=4 -> 47.79 us/op
    # vs 62.88 default at the model's exact shape, pcc 0.9998 vs the device output.
    # per_core_N = Nt/cores = 160/40 = 4 exactly. out_block_w = osw = 4 gives
    # num_blocks_w_dim = 1, so the PR #51514 corruption cannot trigger.
    decode_qkv_cores: tuple[int, int] | None = (8, 5)
    decode_qkv_in0_block_w: int = 2
    decode_qkv_out_subblock_w: int = 4
    decode_out_cores: tuple[int, int] | None = None
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None
