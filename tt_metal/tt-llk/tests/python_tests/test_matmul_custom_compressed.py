# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.compressed_utils import (
    DEEPSEEK_T420,
    FMT_CODE,
    assign_clustered,
    assign_interleaved,
    assign_random,
    encode_tile_meta,
    generate_exact_assignment,
    run_compressed,
)
from helpers.param_config import parametrize
from helpers.test_variant_parameters import CUSTOM_MM_CALLS
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM


def promote_assignment(assignment, ct):
    # Promote one tile to bfp2 if all tiles are bfp0
    if sum(assignment) == 0:
        assignment[0] = FMT_CODE["bfp2"]
    return assignment


def pack_b(tiles):
    # Just pack contiguously independent of the format, no aux data
    return b"".join(full for _, full in tiles), None


def encode_meta(assignment, ct, kt, aux):
    # run_compressed's make_meta hook; the packing itself is shared with the compressed_custom_mm
    # advance test. kt and aux are unused here -- the tile kernel's meta is a flat per-tile stream.
    return encode_tile_meta(assignment, ct)


COMPRESSION_GRANULARITY = DEFAULT_TILE_C_DIM
SUPPORTED_M = {1, 2, 4, 8}
# The tile kernel handles all four BFP precisions.
SUPPORTED_FORMATS = {FMT_CODE[f] for f in ("bfp0", "bfp2", "bfp4", "bfp8")}


def run_tile_compressed(M, K, N, assignment, pcc_threshold=None):
    run_compressed(
        M,
        K,
        N,
        assignment,
        "sources/matmul_custom_compressed_test.cpp",
        COMPRESSION_GRANULARITY,
        SUPPORTED_M,
        SUPPORTED_FORMATS,
        promote_assignment,
        pack_b,
        encode_meta,
        pcc_threshold=pcc_threshold,
    )


BASE_SHAPES = [
    (1, 64, 32),  #   2x1
    (1, 64, 64),  #   2x2
    (1, 256, 32),  #   8x1
    (1, 256, 128),  #   8x4
    (1, 512, 256),  #  16x8
    (1, 7168, 32),  # 224x1
    (1, 7168, 64),  # 224x2
    # (1, 7168, 256), # 224x8 OOM
]

DEEPSEEK_SHAPES = [
    (1, 256, 64),  #  8x2
    (1, 896, 32),  # 28x1
    (1, 256, 224),  #  8x7
    (1, 1792, 32),  # 56x1
]

EXT_SHAPES = [
    # (1,  128, 512), #   4x16
    (1, 512, 128),  #  16x 4
    (1, 1536, 128),  #  48x 4
    (1, 2048, 32),  #  64x 1
    (1, 3584, 32),  # 112x 1
    # (1, 7168, 160), # 224x 5
    (1, 8192, 64),  # 256x 2
    # (8,  256, 512), #   8x16
    # (8,  512, 512), #  16x16
    (8, 576, 256),  #  18x 8
    # (8,  576, 512), #  18x16
]

# The format lists below are kept identical to test_matmul_face_compressed's, so the two kernels can be
# compared on the same compression. That makes them the intersection of what both support, which excludes
# bfp8 even though this kernel handles it (SUPPORTED_FORMATS above) -- so bfp8 goes unexercised by this
# suite. Adding tile-only bfp8 coverage is left out of this change rather than dropped.
SINGLE_FORMATS = [
    # ("bfp8",), excluded for the parity reason above, not because this kernel lacks it
    ("bfp4",),
    ("bfp2",),
]

BASE_MULTI_FORMATS = [
    ("bfp4", "bfp2"),
    ("bfp4", "bfp0"),
    ("bfp2", "bfp0"),
    ("bfp4", "bfp2", "bfp0"),
]

EXT_MULTI_FORMATS = [
    ("bfp8", "bfp4"),
    ("bfp8", "bfp2"),
    ("bfp8", "bfp0"),
    ("bfp8", "bfp4", "bfp2"),
    ("bfp8", "bfp4", "bfp0"),
    ("bfp8", "bfp2", "bfp0"),
    ("bfp8", "bfp4", "bfp2", "bfp0"),
]

SHAPES = BASE_SHAPES + DEEPSEEK_SHAPES + EXT_SHAPES
MULTI_FORMATS = BASE_MULTI_FORMATS  # EXT_MULTI_FORMATS is not supported in face version


@blackhole_only
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=SINGLE_FORMATS,
)
def test_matmul_custom_compressed_single(shape, formats):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(M, K, N, assignment)


@blackhole_only
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_custom_compressed_random(shape, formats):
    M, K, N = shape
    assignment = assign_random(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(M, K, N, assignment)


@blackhole_only
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_custom_compressed_clustered(shape, formats):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(M, K, N, assignment)


@blackhole_only
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
    interleave_n=[1, 2, 4, 8, 16, 32],
)
def test_matmul_custom_compressed_interleaved(shape, formats, interleave_n):
    M, K, N = shape
    assignment = assign_interleaved(
        K, N, formats, COMPRESSION_GRANULARITY, interleave_n
    )
    run_tile_compressed(M, K, N, assignment)


# ---------------------------------------------------------------------------
# Realistic DeepSeek-R1 tile assignment on the native 32x32-tile kernel. Same
# exact-count generator + seed as test_matmul_face_compressed's switch_mult=1.0 row,
# so tile-kernel vs face-kernel on identical compression is a controlled comparison.
# Exact-count (not sampled) so the small shapes below still hit the target shares.


@blackhole_only
@parametrize(
    shape=DEEPSEEK_SHAPES,
    seed=[0],
)
def test_matmul_custom_compressed_deepseek(shape, seed):
    M, K, N = shape
    assignment = generate_exact_assignment(K, N, DEEPSEEK_T420, seed=seed)
    run_tile_compressed(M, K, N, assignment)


# ---------------------------------------------------------------------------
# Metadata-word boundary.
#
# The unpacker walks the format metadata 10 tile indices per u32
# (llk_unpack_AB_compressed_custom_mm.h): full_iters = kt*ct / 10 whole words, then a
# remainder word for the leftover kt*ct % 10 tiles. encode_meta above sizes the buffer
# at ceil(kt*ct / 10) words to match, so when kt*ct is an exact multiple of 10 there is
# no remainder word at all -- the buffer ends after full_iters.
#
# That is the case the kernel used to read one word past (fixed on this branch: the
# remainder load is now guarded on rem_iters != 0). Nothing else reaches it: every shape
# in SHAPES has kt*ct in {2,4,8,16,28,32,56,64,112,128,192,224,448,512}, none divisible
# by 10. Hence this test.


METADATA_WORD_BOUNDARY_SHAPES = [
    (1, 320, 32),  # kt=10, ct=1 -> 10 tiles, exactly 1 metadata word
    (1, 64, 160),  # kt= 2, ct=5 -> 10 tiles, and ct>1 so the use_b bit cycles
    (1, 320, 64),  # kt=10, ct=2 -> 20 tiles, exactly 2 metadata words
]


@blackhole_only
@parametrize(
    shape=METADATA_WORD_BOUNDARY_SHAPES,
    formats=SINGLE_FORMATS,
)
def test_matmul_custom_compressed_metadata_word_boundary(shape, formats):
    """kt*ct an exact multiple of 10 -> no remainder metadata word.

    Be clear about what this does and does not catch. It does **not** detect the
    out-of-bounds read itself, and cannot: at rem_iters == 0 the remainder loop never
    runs, so the word loaded from past the buffer is never used, and the result is
    unaffected by construction. Verified rather than assumed -- these six variants pass
    against the unguarded kernel too. Catching that read needs a memory-safety check on
    L1, not a golden comparison.

    What it does buy is coverage of the exact-multiple shape class, which nothing else
    reached, so a future change to the metadata walk that mishandles it -- consuming a
    stale word, or dropping the final group of 10 tiles -- fails here on the golden.
    """
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(M, K, N, assignment)


# Multi-call: K splits over back-to-back calls that accumulate into one DEST, every call repeating one
# tile pattern with its metadata restarting on the previous-format sentinel, as the blaze host builds
# it. The kernel holds math back before each call, so the unpacker reaches the next call while math
# still holds the previous one's banks (see CUSTOM_MM_CALLS).
#   race      bfp8/bfp4 only, so nothing inside a call clears SrcA: a both-bank SrcB clear issued while
#             math still holds the banks drops the SrcA writes it overlaps
#   rearm     the same with a -inf SrcA clear after every call, which the end-of-call stall's clear to 0
#             would otherwise mask
#   boundary  each call ends on bfp2: only the end-of-call stall separates that unpack from the next
#             call's bfp4/bfp8 format change
MULTI_CALL_CASES = [
    # (id, M, K, N, per-call tiles (row-major kt_per_call x ct), rearm)
    ("race", 1, 1024, 32, ["bfp8", "bfp4"], False),
    ("race-m8", 8, 1024, 32, ["bfp8", "bfp4"], False),
    ("rearm", 1, 1024, 32, ["bfp8", "bfp4"], True),
    ("rearm-m8", 8, 1024, 32, ["bfp8", "bfp4"], True),
    ("boundary-bfp4", 1, 2048, 32, ["bfp4", "bfp4", "bfp4", "bfp2"], False),
    ("boundary-bfp8-m8", 8, 2048, 32, ["bfp8", "bfp8", "bfp8", "bfp2"], False),
    ("boundary-ct2", 1, 1024, 64, ["bfp4", "bfp8", "bfp4", "bfp2"], False),
]
# Per-call weight slots are sized for the largest compressed tile (bfp8, 32x32).
BFP8_TILE_BYTES = 1088
# A missing stall corrupts part of one tile per boundary, which the default tolerance misses on
# some shapes; on Blackhole these land at PCC >= 0.9998 with the stall and <= 0.998 without it.
# The race cases fall far below that.
MULTI_CALL_PCC = 0.9995


@blackhole_only
@pytest.mark.parametrize(
    "M,K,N,pattern,rearm",
    [pytest.param(*case[1:], id=case[0]) for case in MULTI_CALL_CASES],
)
def test_matmul_custom_compressed_multi_call(M, K, N, pattern, rearm):
    """Back-to-back calls into one DEST, checked against the full-K golden."""
    kt, ct = K // DEFAULT_TILE_R_DIM, N // DEFAULT_TILE_C_DIM
    tiles_per_call = len(pattern)
    num_calls = kt * ct // tiles_per_call

    def pack_b_per_call(tiles):
        # Contiguous within a call, one bfp8-sized slot per call.
        calls = [
            tiles[c * tiles_per_call : (c + 1) * tiles_per_call]
            for c in range(num_calls)
        ]
        return (
            b"".join(
                b"".join(full for _, full in call).ljust(
                    tiles_per_call * BFP8_TILE_BYTES, b"\0"
                )
                for call in calls
            ),
            None,
        )

    def encode_meta_per_call(assignment, ct, kt, aux):
        # One metadata stream per call, each restarting on the previous-format sentinel.
        return b"".join(
            encode_tile_meta(
                assignment[c * tiles_per_call : (c + 1) * tiles_per_call], ct
            )
            for c in range(num_calls)
        )

    run_compressed(
        M,
        K,
        N,
        [FMT_CODE[f] for f in pattern] * num_calls,
        "sources/matmul_custom_compressed_test.cpp",
        COMPRESSION_GRANULARITY,
        SUPPORTED_M,
        SUPPORTED_FORMATS,
        promote_assignment,
        pack_b_per_call,
        encode_meta_per_call,
        pcc_threshold=MULTI_CALL_PCC,
        calls=CUSTOM_MM_CALLS(num_calls=num_calls, rearm=rearm),
    )
