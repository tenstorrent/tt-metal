# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free unit tests for compact_io, shuffle_dram_assignment, and bspm_loader.

No TT device is needed — all tests run on CPU with numpy/torch only.
"""

from __future__ import annotations

import struct
from pathlib import Path

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


# ─── bspm_loader tests ────────────────────────────────────────────────────────

from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import (  # noqa: E402
    load_bspm_for_layer,
    remap_bspm_to_ttnn,
)

_BSPM_STRUCT = struct.Struct("<4sIIIIIIIIB3xII")
_BSPM_HEADER_SIZE = 64  # 48 bytes fields + 16 bytes reserved


def _write_synthetic_bspm(
    path: Path,
    layer_idx: int,
    n_experts: int,
    tile_rows: int,
    tile_cols: int,
    codes_bitsculpt: np.ndarray,
    variant_code: int = 1,
    budget_millibits: int = 3500,
) -> None:
    """Write a minimal valid .bspm binary file for testing."""
    tiles_per_proj = tile_rows * tile_cols
    header_fields = _BSPM_STRUCT.pack(
        b"BSPM",
        1,  # version
        layer_idx,
        n_experts,
        3,  # n_projections
        tiles_per_proj,
        tile_rows,
        tile_cols,
        32,  # tile_size
        variant_code,
        budget_millibits,
        budget_millibits,  # actual_millibits
    )
    header = header_fields + b"\x00" * (_BSPM_HEADER_SIZE - len(header_fields))
    path.write_bytes(header + codes_bitsculpt.astype(np.uint8).tobytes())


def test_load_bspm_for_layer_parses_synthetic_file(tmp_path):
    """load_bspm_for_layer correctly parses all header fields and body from a synthetic file."""
    n_experts, tile_rows, tile_cols = 4, 8, 8
    tiles_per_proj = tile_rows * tile_cols
    rng = np.random.default_rng(7)
    codes_bs = rng.integers(0, 3, size=(n_experts, 3, tiles_per_proj), dtype=np.uint8)

    bspm_path = tmp_path / "precision_map_B_3.5.bspm"
    _write_synthetic_bspm(
        bspm_path, layer_idx=3, n_experts=n_experts, tile_rows=tile_rows, tile_cols=tile_cols, codes_bitsculpt=codes_bs
    )

    data = load_bspm_for_layer(str(bspm_path))

    assert data["magic"] == "BSPM"
    assert data["version"] == 1
    assert data["layer_idx"] == 3
    assert data["n_experts"] == n_experts
    assert data["n_projections"] == 3
    assert data["tiles_per_proj"] == tiles_per_proj
    assert data["tile_rows"] == tile_rows
    assert data["tile_cols"] == tile_cols
    assert data["variant"] == "B"
    assert abs(data["budget"] - 3.5) < 1e-3
    assert data["codes"].shape == (n_experts, 3, tiles_per_proj)
    assert data["codes_bitsculpt"].shape == (n_experts, 3, tiles_per_proj)
    np.testing.assert_array_equal(data["codes_bitsculpt"], codes_bs)


def test_load_bspm_for_layer_code_remapping(tmp_path):
    """BitSculpt→tt-metal code remapping is correct: BS[0,1,2,3] → ttnn[3,2,1,0]."""
    n_experts, tile_rows, tile_cols = 1, 4, 4
    tiles_per_proj = tile_rows * tile_cols

    # One tile per BitSculpt code, repeated to fill tiles_per_proj
    bs_cycle = np.array([0, 1, 2, 3] * (tiles_per_proj // 4), dtype=np.uint8)
    codes_bs = bs_cycle[np.newaxis, np.newaxis, :].repeat(n_experts, 0).repeat(3, 1)

    bspm_path = tmp_path / "precision_map_B_3.5.bspm"
    _write_synthetic_bspm(
        bspm_path, layer_idx=5, n_experts=n_experts, tile_rows=tile_rows, tile_cols=tile_cols, codes_bitsculpt=codes_bs
    )

    data = load_bspm_for_layer(str(bspm_path))
    ttnn_codes = data["codes"][0, 0]  # (tiles_per_proj,)
    bs_codes = data["codes_bitsculpt"][0, 0]

    # Verify mapping: ttnn = 3 - bs (they are mirrors of each other)
    expected_ttnn = (3 - bs_codes.astype(np.int8)).astype(np.int8)
    np.testing.assert_array_equal(ttnn_codes, expected_ttnn, err_msg="BitSculpt→tt-metal remapping incorrect")

    # Also verify remap_bspm_to_ttnn is its own inverse (it is: [3,2,1,0] applied twice = identity)
    double_mapped = remap_bspm_to_ttnn(ttnn_codes.astype(np.uint8))
    np.testing.assert_array_equal(double_mapped, bs_codes, err_msg="remap_bspm_to_ttnn should be its own inverse")


def test_load_bspm_for_layer_invalid_magic(tmp_path):
    """load_bspm_for_layer raises ValueError on wrong magic bytes."""
    bspm_path = tmp_path / "bad.bspm"
    bad_header = _BSPM_STRUCT.pack(b"NOPE", 1, 0, 2, 3, 64, 8, 8, 32, 1, 3500, 3500)
    bspm_path.write_bytes(bad_header + b"\x00" * 16)

    try:
        load_bspm_for_layer(str(bspm_path))
        assert False, "Expected ValueError for bad magic"
    except ValueError as e:
        assert "magic" in str(e).lower()


def test_load_bspm_for_layer_lfs_pointer_detected(tmp_path):
    """load_bspm_for_layer raises RuntimeError on Git LFS pointer files."""
    lfs_path = tmp_path / "lfs.bspm"
    lfs_path.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")

    try:
        load_bspm_for_layer(str(lfs_path))
        assert False, "Expected RuntimeError for LFS pointer"
    except RuntimeError as e:
        assert "LFS" in str(e) or "lfs" in str(e).lower()
