# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The per-chunk H2D metadata layout shared with the engine (host-only, no ttnn)."""

import struct

from models.demos.common.prefill.media import MediaItem
from models.demos.common.prefill.media_metadata import chunk_metadata_size, pack_chunk_metadata, unpack_media_items

KEY_A, KEY_B = bytes([0xA]) * 32, bytes([0xB]) * 32


def test_example_chunk_bytes():
    # Chunk [0, 5120) of slot 3 with two images; the second continues into the next chunk.
    items = [MediaItem(100, 576, KEY_A), MediaItem(4900, 576, KEY_B)]
    raw = pack_chunk_metadata(3, 0, 5120, items, k=10)

    assert len(raw) == chunk_metadata_size(10) == 416
    assert struct.unpack_from("<iiiI", raw, 0) == (3, 0, 5120, 2)
    assert struct.unpack_from("<II32s", raw, 16) == (100, 576, KEY_A)
    assert struct.unpack_from("<II32s", raw, 56) == (4900, 576, KEY_B)
    assert raw[96:] == bytes(416 - 96)  # unused item slots are zero
    assert unpack_media_items(raw, k=10) == items


def test_text_only_keeps_the_12_byte_header():
    assert chunk_metadata_size(0) == 12
    assert pack_chunk_metadata(3, 0, 5120, [], k=0) == struct.pack("<3I", 3, 0, 5120)
    assert unpack_media_items(pack_chunk_metadata(3, 0, 5120, [], k=10), k=10) == []


def test_sentinels_list_no_media():
    # The producer's shutdown sentinel is every word -1, media count included.
    sentinel = struct.pack(f"<{chunk_metadata_size(10) // 4}i", *([-1] * (chunk_metadata_size(10) // 4)))
    assert unpack_media_items(sentinel, k=10) == []


def test_more_images_than_k_is_rejected(expect_error):
    items = [MediaItem(i * 10, 5, KEY_A) for i in range(3)]
    with expect_error(AssertionError, "3 images in one chunk, the header holds 2"):
        pack_chunk_metadata(3, 0, 5120, items, k=2)
