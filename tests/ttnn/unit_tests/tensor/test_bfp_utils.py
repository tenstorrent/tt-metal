# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ttnn

bfp_utils = ttnn._ttnn.bfp_utils


def tile_faces(matrix):
    """[32, W] row-major -> 32x32 tiles along W, each as four 16x16 faces."""
    return np.concatenate(
        [
            matrix[row : row + 16, col + face_col : col + face_col + 16].ravel()
            for col in range(0, matrix.shape[1], 32)
            for row, face_col in ((0, 0), (0, 16), (16, 0), (16, 16))
        ]
    )


def test_unpack_bfp8_with_explicit_alignment():
    tile = np.full(1088 // 4, 0x40404040, dtype=np.uint32)
    tile[: 64 // 4] = 0x7F7F7F7F
    np.testing.assert_array_equal(
        bfp_utils.unpack_bfp8(tile, row_major_output=True, l1_alignment=16), np.ones(1024, dtype=np.float32)
    )


def test_unpack_bfp8_explicit_alignment_matches_hal():
    reference = np.random.default_rng(0).standard_normal((32, 96)).astype(np.float32)
    words = bfp_utils.pack_bfp8(tile_faces(reference))
    explicit = bfp_utils.unpack_bfp8(words, l1_alignment=16)
    np.testing.assert_array_equal(explicit, bfp_utils.unpack_bfp8(words))
    np.testing.assert_allclose(bfp_utils.untilize(explicit, 32, 96), reference, rtol=0, atol=0.1)


def test_untilize_bf16_face_order():
    expected = np.arange(32 * 64, dtype=np.uint16).reshape(32, 64)
    np.testing.assert_array_equal(bfp_utils.untilize(tile_faces(expected), 32, 64), expected)
