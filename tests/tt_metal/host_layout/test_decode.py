# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from tt_host_layout import decode_chunk, row_bytes


def test_bfp8_shared_exponent_reconstructs_one():
    tile = bytes([127] * 64) + bytes([0x40] * 1024)
    decoded = decode_chunk(tile, dtype="bfp8", width=32, storage="tile")
    np.testing.assert_allclose(decoded, np.ones((32, 32), dtype=np.float32))


def test_bf16_tile_face_order():
    expected = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    expected = ((expected.view(np.uint32) >> 16) << 16).view(np.float32)
    encoded = b""
    for col in (0, 32):
        for row, face_col in ((0, 0), (0, 16), (16, 0), (16, 16)):
            face = expected[row : row + 16, col + face_col : col + face_col + 16].copy()
            encoded += (face.view(np.uint32) >> 16).astype("<u2").tobytes()
    np.testing.assert_equal(decode_chunk(encoded, dtype="bf16", width=64, storage="tile"), expected)


def test_row_major_bf16_and_partial_rows():
    expected = np.arange(32 * 576, dtype=np.float32).reshape(32, 576)
    encoded = (expected.view(np.uint32) >> 16).astype("<u2").tobytes()
    actual = decode_chunk(encoded, dtype="bf16", width=576, storage="row-major")
    np.testing.assert_equal(actual, ((expected.view(np.uint32) >> 16) << 16).view(np.float32))
    assert row_bytes(encoded, 2, 3, dtype="bf16", width=576, storage="row-major") == encoded[2304:3456]
