# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from helpers.compressed_utils import (  # noqa: F401 (autouse fixture — imported to activate in this module)
    DEEPSEEK_T420,
    FMT_CODE,
    assign_clustered,
    assign_interleaved,
    assign_random,
    compressed_mm_include_paths,
    generate_exact_assignment,
    run_compressed,
)
from helpers.param_config import parametrize
from helpers.tile_constants import DEFAULT_TILE_C_DIM


def promote_assignment(assignment, ct):
    # Promote one tile to bfp2 if all tiles are bfp0
    if sum(assignment) == 0:
        assignment[0] = FMT_CODE["bfp2"]
    return assignment


def pack_b(tiles):
    # Just pack contiguously independent of the format, no aux data
    return b"".join(full for _, full in tiles), None


def encode_meta(assignment, ct, kt, aux):
    total = len(assignment)
    num_u32 = (total + 9) // 10
    meta = [0] * num_u32
    prev_fmt = 0
    for i in range(total):
        u, j = divmod(i, 10)
        if j == 0:
            meta[u] |= prev_fmt & 0b11
        fmt = assignment[i] & 0b11
        use_b = 1 if (i % ct) == 0 else 0
        meta[u] |= use_b << (3 * j + 2)
        meta[u] |= fmt << (3 * j + 3)
        prev_fmt = fmt
    return np.array(meta, dtype=np.uint32).tobytes()


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

SINGLE_FORMATS = [
    # ("bfp8",), unsupported in face version
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


@parametrize(
    shape=SHAPES,
    formats=SINGLE_FORMATS,
)
def test_matmul_custom_compressed_single(shape, formats):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(M, K, N, assignment)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_custom_compressed_random(shape, formats):
    M, K, N = shape
    assignment = assign_random(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(M, K, N, assignment)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_custom_compressed_clustered(shape, formats):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(M, K, N, assignment)


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


@parametrize(
    shape=DEEPSEEK_SHAPES,
    seed=[0],
)
def test_matmul_custom_compressed_deepseek(shape, seed):
    M, K, N = shape
    assignment = generate_exact_assignment(K, N, DEEPSEEK_T420, seed=seed)
    run_tile_compressed(M, K, N, assignment)
