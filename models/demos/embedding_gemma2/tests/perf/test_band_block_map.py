# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only: `masks.band_block_map` against brute-force block visibility from the dense band mask.

The sliding layers' SDPA block map (E9.4) marks a (q chunk, k chunk) block visible when any (q, k) in it is
inside the band |q - k| <= radius. Chunks need not divide S (E9.4 q 96 / q 160): the last block is partial,
its rows / keys clipped to S. The reference here pads the dense band mask (`masks.band_mask`, the mask the
model uploads) with masked rows / keys up to whole blocks and takes `any` per block, so it shares no
arithmetic with the closed-form map.

    pytest models/demos/embedding_gemma2/tests/perf/test_band_block_map.py
"""

from __future__ import annotations

import pytest
import torch

from models.demos.embedding_gemma2.tt.masks import band_block_map, band_mask

RADIUS = 512  # embeddinggemma-2 sliding_window


def brute_force_map(seq_len: int, radius: int, q_chunk: int, k_chunk: int) -> torch.Tensor:
    keep = band_mask(seq_len, radius)[0, 0] == 0  # [S, S] bool, True inside the band
    nq, nk = -(-seq_len // q_chunk), -(-seq_len // k_chunk)
    padded = torch.zeros(nq * q_chunk, nk * k_chunk, dtype=torch.bool)
    padded[:seq_len, :seq_len] = keep
    blocks = padded.reshape(nq, q_chunk, nk, k_chunk).any(dim=3).any(dim=1)
    return blocks.to(torch.int32).reshape(1, 1, nq, nk)


@pytest.mark.parametrize(
    "seq_len,q_chunk,k_chunk",
    [
        (2048, 96, 128),  # q chunk does not divide S (22 blocks, the last 32 rows)
        (4096, 160, 128),  # E9.4 full-layer q 160 shape on the sliding band (26 blocks, the last 96 rows)
        (2048, 128, 96),  # k chunk does not divide S
        (4096, 160, 96),  # neither divides
        (2048, 128, 128),  # divides (the kept 2048 sliding entry)
        (4096, 256, 128),  # divides (the kept 4096 / 8192 sliding entry)
    ],
)
def test_band_block_map_matches_dense_band(seq_len, q_chunk, k_chunk):
    got = band_block_map(seq_len, RADIUS, q_chunk, k_chunk)
    want = brute_force_map(seq_len, RADIUS, q_chunk, k_chunk)
    assert got.shape == want.shape, f"map shape {tuple(got.shape)} != {tuple(want.shape)}"
    bad = (got != want).nonzero().tolist()
    assert not bad, f"{len(bad)} blocks differ (first [.., q block, k block]: {bad[:5]})"
    assert 0 < int(want.sum()) < want.numel(), "case does not exercise both visible and skipped blocks"


@pytest.mark.parametrize("seq_len,radius,q_chunk,k_chunk", [(300, 50, 96, 128), (1000, 7, 64, 96), (96, 512, 64, 64)])
def test_band_block_map_small_radii(seq_len, radius, q_chunk, k_chunk):
    """Odd sizes, a radius smaller than a chunk, and a band wider than S (every block visible)."""
    assert torch.equal(
        band_block_map(seq_len, radius, q_chunk, k_chunk), brute_force_map(seq_len, radius, q_chunk, k_chunk)
    )
