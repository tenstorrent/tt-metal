# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free unit tests for compact_io and shuffle_dram_assignment.

No TT device is needed — all tests run on CPU with numpy/torch only.
"""

from __future__ import annotations

import numpy as np
import torch

from models.demos.deepseek_v3_b1.compressed_tensor.compact_io import (
    bfp4_tile_byte_count,
    compact_tile_byte_count,
    pack_compact_tiles,
    unpack_compact_tiles,
)
from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import (
    BFP_MANT_BITS,
    DEFAULT_TILE_HW,
    bfp_tile_packed_size,
    unpack_bfp_tile,
)
from models.demos.deepseek_v3_b1.weights.transforms.moe import shuffle_dram_assignment, shuffle_dram_tiles

TILE_HW = DEFAULT_TILE_HW  # 32


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _random_weight(tiles_h: int, tiles_w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((tiles_h * TILE_HW, tiles_w * TILE_HW)).astype(np.float32)


_CODE_TO_FMT = {1: "bfp4", 2: "bfp2", 3: "bfp0"}


def _unpack_tile_direct(packed_bytes: bytes, code: int) -> np.ndarray:
    """Unpack a single tile using the tile-level primitive."""
    mant_bits = BFP_MANT_BITS[_CODE_TO_FMT[code]]
    arr = np.frombuffer(packed_bytes, dtype=np.uint8).copy()
    return unpack_bfp_tile(arr, mant_bits)


# ─── compact_io roundtrip tests ───────────────────────────────────────────────


def test_pack_unpack_all_bfp4():
    """Uniform BFP4 assignment: pack→unpack matches per-tile pack/unpack."""
    tiles_h, tiles_w = 4, 4
    w = _random_weight(tiles_h, tiles_w, seed=1)
    assignment = np.ones((tiles_h, tiles_w), dtype=np.int8)  # all BFP4

    packed = pack_compact_tiles(w, assignment)
    reconstructed = unpack_compact_tiles(packed, assignment)

    # Verify tile by tile against the primitive
    tile_size = bfp_tile_packed_size(BFP_MANT_BITS["bfp4"], TILE_HW)
    for r in range(tiles_h):
        for c in range(tiles_w):
            tile_bytes = packed[r * tiles_w * tile_size + c * tile_size : r * tiles_w * tile_size + (c + 1) * tile_size]
            expected = _unpack_tile_direct(tile_bytes, code=1)
            actual = reconstructed[r * TILE_HW : (r + 1) * TILE_HW, c * TILE_HW : (c + 1) * TILE_HW]
            np.testing.assert_array_equal(actual, expected, err_msg=f"tile ({r},{c}) mismatch")


def test_pack_unpack_mixed_formats():
    """Mixed BFP4/BFP2/zero: pack→unpack exactly matches per-tile primitives."""
    tiles_h, tiles_w = 3, 4
    rng = np.random.default_rng(42)
    w = _random_weight(tiles_h, tiles_w, seed=7)
    assignment = rng.choice([1, 2, 3], size=(tiles_h, tiles_w)).astype(np.int8)

    packed = pack_compact_tiles(w, assignment)
    reconstructed = unpack_compact_tiles(packed, assignment)

    # Re-derive expected output tile by tile using the primitives directly
    offset = 0
    fmt_map = {1: "bfp4", 2: "bfp2", 3: "bfp0"}
    for r in range(tiles_h):
        for c in range(tiles_w):
            code = int(assignment[r, c])
            tile_orig = w[r * TILE_HW : (r + 1) * TILE_HW, c * TILE_HW : (c + 1) * TILE_HW]
            tile_out = reconstructed[r * TILE_HW : (r + 1) * TILE_HW, c * TILE_HW : (c + 1) * TILE_HW]

            if code == 3:  # zero tile — should be all zeros
                np.testing.assert_array_equal(tile_out, np.zeros_like(tile_out), err_msg=f"tile ({r},{c}) zero fill")
            else:
                mant_bits = BFP_MANT_BITS[fmt_map[code]]
                nbytes = bfp_tile_packed_size(mant_bits, TILE_HW)
                tile_bytes = np.frombuffer(packed[offset : offset + nbytes], dtype=np.uint8).copy()
                expected = unpack_bfp_tile(tile_bytes, mant_bits)
                np.testing.assert_array_equal(tile_out, expected, err_msg=f"tile ({r},{c}) code={code} mismatch")
                offset += nbytes


def test_zero_tiles_produce_no_bytes_and_zero_fill():
    """All-zero assignment: packed output is empty; unpack fills with zeros."""
    tiles_h, tiles_w = 2, 3
    w = _random_weight(tiles_h, tiles_w, seed=5)
    assignment = np.full((tiles_h, tiles_w), fill_value=3, dtype=np.int8)  # all zero tiles

    packed = pack_compact_tiles(w, assignment)
    assert len(packed) == 0, f"Expected 0 bytes for all-zero assignment, got {len(packed)}"

    reconstructed = unpack_compact_tiles(packed, assignment)
    np.testing.assert_array_equal(
        reconstructed, np.zeros_like(reconstructed), err_msg="All-zero assignment must unpack to zeros"
    )


def test_compact_tile_byte_count_matches_packed_length():
    """compact_tile_byte_count must equal len(pack_compact_tiles(...))."""
    tiles_h, tiles_w = 4, 6
    rng = np.random.default_rng(13)
    w = _random_weight(tiles_h, tiles_w, seed=13)
    assignment = rng.choice([1, 2, 3], size=(tiles_h, tiles_w), p=[0.5, 0.3, 0.2]).astype(np.int8)

    packed = pack_compact_tiles(w, assignment)
    predicted = compact_tile_byte_count(assignment)
    assert predicted == len(packed), f"compact_tile_byte_count={predicted} != actual packed length={len(packed)}"


def test_bfp4_tile_byte_count_formula():
    """bfp4_tile_byte_count(h, w) == h * w * bfp_tile_packed_size(BFP4_MANT_BITS, TILE_HW)."""
    tile_bytes = bfp_tile_packed_size(BFP_MANT_BITS["bfp4"], TILE_HW)
    for tiles_h, tiles_w in [(2, 4), (8, 8), (1, 16)]:
        expected = tiles_h * tiles_w * tile_bytes
        actual = bfp4_tile_byte_count(tiles_h, tiles_w)
        assert actual == expected, f"({tiles_h},{tiles_w}): expected {expected}, got {actual}"


def test_compact_smaller_than_bfp4_for_mixed_assignment():
    """Compact format is strictly smaller than uniform BFP4 when assignment contains BFP2/zero tiles."""
    tiles_h, tiles_w = 8, 8
    rng = np.random.default_rng(99)
    w = _random_weight(tiles_h, tiles_w, seed=99)
    # 50% BFP4, 30% BFP2, 20% zero
    assignment = rng.choice([1, 2, 3], size=(tiles_h, tiles_w), p=[0.50, 0.30, 0.20]).astype(np.int8)

    compact_bytes = compact_tile_byte_count(assignment)
    bfp4_bytes = bfp4_tile_byte_count(tiles_h, tiles_w)
    assert compact_bytes < bfp4_bytes, f"Compact ({compact_bytes} B) should be < BFP4 baseline ({bfp4_bytes} B)"


# ─── shuffle_dram_assignment correctness test ─────────────────────────────────


def test_shuffle_dram_assignment_matches_shuffle_dram_tiles():
    """shuffle_dram_assignment applies the same permutation as shuffle_dram_tiles.

    Strategy: fill each 32×32 tile block with a unique constant (its flat tile index).
    After shuffling the weight tensor, read which original tile ended up at each output
    position, then check that shuffle_dram_assignment maps the same code to that position.
    """
    # num_banks=4, per_N_tiles=2 → non-trivial permutation (per_N_tiles=1 is identity)
    num_banks = 4
    tiles_h = 3
    per_N_tiles = 2
    tiles_w = num_banks * per_N_tiles  # 8
    K = tiles_h * TILE_HW  # 96
    N = tiles_w * TILE_HW  # 256

    # Weight where tile (r, c) is filled with unique constant = r * tiles_w + c
    w_torch = torch.zeros(K, N, dtype=torch.float32)
    for r in range(tiles_h):
        for c in range(tiles_w):
            w_torch[r * TILE_HW : (r + 1) * TILE_HW, c * TILE_HW : (c + 1) * TILE_HW] = float(r * tiles_w + c)

    # Assignment: codes cycling through 1/2/3 per tile
    assignment = np.array([[(r * tiles_w + c) % 3 + 1 for c in range(tiles_w)] for r in range(tiles_h)], dtype=np.int8)

    # Shuffle both
    shuffled_w = shuffle_dram_tiles(w_torch, TILE_HW, num_banks)  # (K, N)
    shuffled_asgn = shuffle_dram_assignment(assignment, num_banks)

    # Verify: at each output tile position, the code matches the original tile's code
    for r in range(tiles_h):
        for c in range(tiles_w):
            tile_block = shuffled_w[r * TILE_HW : (r + 1) * TILE_HW, c * TILE_HW : (c + 1) * TILE_HW]
            orig_tile_idx = int(tile_block[0, 0].item())  # all elements equal unique constant
            orig_r = orig_tile_idx // tiles_w
            orig_c = orig_tile_idx % tiles_w
            expected_code = int(assignment[orig_r, orig_c])
            actual_code = int(shuffled_asgn[r, c])
            assert actual_code == expected_code, (
                f"Tile ({r},{c}): weight came from original ({orig_r},{orig_c}), "
                f"expected code {expected_code}, got {actual_code}"
            )


def test_shuffle_dram_assignment_is_permutation():
    """shuffle_dram_assignment is a permutation: it preserves all tile counts."""
    rng = np.random.default_rng(77)
    tiles_h, tiles_w, num_banks = 4, 8, 4
    assignment = rng.choice([1, 2, 3], size=(tiles_h, tiles_w), p=[0.6, 0.25, 0.15]).astype(np.int8)

    shuffled = shuffle_dram_assignment(assignment, num_banks)

    assert shuffled.shape == assignment.shape
    for code in [1, 2, 3]:
        orig_count = int(np.sum(assignment == code))
        shuf_count = int(np.sum(shuffled == code))
        assert orig_count == shuf_count, f"Code {code}: count changed from {orig_count} to {shuf_count}"


def test_shuffle_dram_assignment_one_tile_per_bank_is_identity():
    """When tiles_w == num_banks (per_N_tiles=1), shuffle_dram_assignment is the identity.

    The permutation is source_idx[i] = (i % K_tiles) * per_N_tiles + (i // K_tiles).
    With per_N_tiles=1 and i < K_tiles this reduces to source_idx[i] = i.
    """
    rng = np.random.default_rng(3)
    tiles_h = 4
    num_banks = 4
    tiles_w = num_banks  # per_N_tiles = 1 → identity permutation
    assignment = rng.choice([1, 2, 3], size=(tiles_h, tiles_w)).astype(np.int8)

    shuffled = shuffle_dram_assignment(assignment, num_banks=num_banks)
    np.testing.assert_array_equal(
        shuffled, assignment, err_msg="tiles_w==num_banks (per_N_tiles=1) must yield identity shuffle"
    )
