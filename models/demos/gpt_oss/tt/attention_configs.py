# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS attention program configurations."""

from dataclasses import dataclass

import ttnn
from models.demos.gpt_oss.tt.attention.config import ProgramConfig


@dataclass
class GPTOSSAttentionProgramConfig(ProgramConfig):
    """
    GPT-OSS attention configuration.

    Optimized for: hidden=2088, heads=84, head_dim=64
    Uses minimal_matmul with per-seq_len swept configs for both QKV and output projections.
    Swept on 4x8 galaxy mesh (32 Wormhole B0 devices).
    """

    # SDPA chunk sizes
    decode_k_chunk_size: int = 128
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Matmul configs - None = auto-optimize
    decode_qkv_cores: tuple[int, int] | None = None
    decode_out_cores: tuple[int, int] | None = None
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = (8, 8)
    prefill_out_in0_block_w: int = 8
    prefill_out_out_subblock_h: int = 1
    prefill_out_out_subblock_w: int = 4

    # Use minimal_matmul for all prefill seq_lens (39-78% faster than standard matmul)
    minimal_matmul_threshold: int = 128  # Effectively always use minimal_matmul in prefill

    def get_prefill_sdpa_chunks(self, seq_len: int) -> tuple[int, int]:
        """Per-seq_len SDPA chunk sizes. Swept on 4x8 galaxy mesh."""
        if seq_len <= 128:
            return 32, 32
        elif seq_len <= 256:
            return 128, 64
        elif seq_len <= 512:
            return 256, 256
        elif seq_len <= 1024:
            return 64, 64
        elif seq_len <= 2048:
            return 128, 128
        else:
            return 256, 512

    def get_prefill_out_minimal_config(self, seq_len: int):
        """Per-seq_len MinimalMatmulConfig for output projection [seq_len, 512] x [512, 3072].
        Swept on 4x8 galaxy mesh."""
        if seq_len <= 128:
            return ttnn.MinimalMatmulConfig(
                M_block_size=2,
                K_block_size=4,
                N_block_size=4,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
            )
        elif seq_len <= 256:
            return ttnn.MinimalMatmulConfig(
                M_block_size=2,
                K_block_size=2,
                N_block_size=8,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
            )
        elif seq_len <= 512:
            return ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=4,
                N_block_size=8,
                subblock_h=4,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
            )
        elif seq_len <= 1024:
            return ttnn.MinimalMatmulConfig(
                M_block_size=2,
                K_block_size=4,
                N_block_size=8,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
            )
        elif seq_len <= 2048:
            return ttnn.MinimalMatmulConfig(
                M_block_size=2,
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
            )
        else:
            return ttnn.MinimalMatmulConfig(
                M_block_size=16,
                K_block_size=4,
                N_block_size=8,
                subblock_h=4,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )

    def get_prefill_qkv_minimal_config(self, seq_len: int):
        """Per-seq_len MinimalMatmulConfig for QKV projection [seq_len, 2880] x [2880, 640].
        Swept on 4x8 galaxy mesh."""
        if seq_len <= 128:
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=2,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
        elif seq_len <= 256:
            return ttnn.MinimalMatmulConfig(
                M_block_size=2,
                K_block_size=2,
                N_block_size=8,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
            )
        elif seq_len <= 512:
            return ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=2,
                N_block_size=4,
                subblock_h=4,
                subblock_w=1,
                compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
            )
        elif seq_len <= 1024:
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=2,
                N_block_size=8,
                subblock_h=2,
                subblock_w=1,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 7),
            )
        elif seq_len <= 2048:
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=2,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
        else:
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=2,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            )
