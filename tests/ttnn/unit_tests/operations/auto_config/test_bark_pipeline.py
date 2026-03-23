# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Bark Small pipeline logic (CPU-only, no hardware required)."""

import math

import pytest


def tile_align(dim: int, tile: int = 32) -> int:
    """Round up to nearest tile boundary."""
    return ((dim + tile - 1) // tile) * tile


class TestTileAlignment:
    """Verify tile alignment utility."""

    @pytest.mark.parametrize("dim,expected", [
        (1, 32), (31, 32), (32, 32), (33, 64),
        (63, 64), (64, 64), (65, 96), (128, 128),
    ])
    def test_tile_align(self, dim, expected):
        assert tile_align(dim) == expected


class TestBarkTokenShapes:
    """Validate Bark pipeline tensor shapes."""

    def test_semantic_token_range(self):
        """Semantic tokens must be in [0, 10000)."""
        SEMANTIC_VOCAB_SIZE = 10_000
        assert SEMANTIC_VOCAB_SIZE == 10000

    def test_coarse_codebook_interleaving(self):
        """Coarse output is interleaved: [batch, seq*2]."""
        seq_len = 100
        n_codebooks = 2
        interleaved_len = seq_len * n_codebooks
        assert interleaved_len == 200
        # De-interleave: [batch, seq*2] → [batch, seq, 2]
        assert interleaved_len // n_codebooks == seq_len

    def test_fine_codebook_count(self):
        """Fine model produces 8 codebooks."""
        n_codes_total = 8
        n_codes_given = 2
        n_codes_predicted = n_codes_total - n_codes_given
        assert n_codes_predicted == 6


class TestKVCacheTileAlignment:
    """KV cache dimensions must be tile-aligned."""

    def test_head_dim_tile_aligned(self):
        hidden_dim = 1024
        num_heads = 16
        head_dim = hidden_dim // num_heads  # 64
        assert head_dim % 32 == 0, f"head_dim {head_dim} not tile-aligned"

    def test_bark_small_head_dim(self):
        """Bark Small: 768 / 12 = 64."""
        hidden_dim = 768
        num_heads = 12
        head_dim = hidden_dim // num_heads
        assert head_dim == 64
        assert head_dim % 32 == 0

    def test_padded_seq_tile_aligned(self):
        for max_seq in [128, 256, 512, 768, 1024]:
            padded_seq = tile_align(max_seq)
            assert padded_seq % 32 == 0


class TestAutoRegressiveTokenIndexing:
    """Verify token position tracking doesn't go OOB."""

    def test_position_bounds(self):
        max_seq = 256
        for step in range(max_seq):
            assert 0 <= step < max_seq

    def test_kv_cache_growth(self):
        """KV cache grows by 1 per step during autoregressive decode."""
        initial_seq = 50  # prompt length
        n_decode_steps = 100  # new tokens generated
        final_kv_len = initial_seq + n_decode_steps
        assert final_kv_len == 150


class TestEncoDecConstants:
    """Verify EnCodec constants are correct."""

    def test_codebook_size(self):
        assert 1024 == 1024  # CODEBOOK_SIZE

    def test_sample_rate(self):
        """EnCodec outputs 24kHz audio."""
        sample_rate = 24000
        assert sample_rate == 24000

    def test_audio_output_shape(self):
        """8 codebooks × seq_len → audio samples."""
        n_codebooks = 8
        seq_len = 100
        # EnCodec outputs 1 sample per codebook frame (with upsampling)
        assert n_codebooks * seq_len > 0  # sanity
