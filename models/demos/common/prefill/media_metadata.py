# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chunk H2D metadata for image prefill: the chunk header plus the images the chunk overlaps.

The engine (C++) writes these bytes and the runner reads them, so the layout below is the contract
between the two. Little-endian, 16 + 40 * K bytes:

    offset  size  field
    0       4     slot_id          i32   negative values are the shutdown / warm-up sentinels
    4       4     actual_start     i32
    8       4     actual_end       i32
    12      4     media_count      u32   valid items below, 0..K
    16      40*K  items[K]:
                    +0   4   start_position  u32   first placeholder position, absolute in the prompt
                    +4   4   num_tokens      u32   the image's row count N
                    +8   32  media_key       32 B  the image's key in the embedding store

Each item describes the whole image, also when only part of it falls in this chunk; the runner works
out which rows the chunk needs. Unused item slots are zero. K = 0 keeps today's 12-byte text-only
header {slot_id, actual_start, actual_end}.
"""

import struct

from models.demos.common.prefill.media import MediaItem

MEDIA_KEY_BYTES = 32
TEXT_HEADER = struct.Struct("<iii")  # slot_id, actual_start, actual_end
MEDIA_HEADER = struct.Struct("<iiiI")  # the same three words, then media_count
MEDIA_ITEM = struct.Struct(f"<II{MEDIA_KEY_BYTES}s")  # start_position, num_tokens, media_key


def chunk_metadata_size(k: int) -> int:
    """Bytes per chunk for at most k images per chunk."""
    assert k >= 0, f"max media items per chunk is {k}"
    return TEXT_HEADER.size if k == 0 else MEDIA_HEADER.size + k * MEDIA_ITEM.size


def pack_chunk_metadata(slot_id: int, actual_start: int, actual_end: int, items: list[MediaItem], k: int) -> bytes:
    """The engine's side of the contract; the test producer uses it to send what the engine would."""
    if k == 0:
        assert not items, "images need PREFILL_MEDIA_K > 0"
        return TEXT_HEADER.pack(slot_id, actual_start, actual_end)
    assert len(items) <= k, f"{len(items)} images in one chunk, the header holds {k}"
    out = bytearray(chunk_metadata_size(k))
    MEDIA_HEADER.pack_into(out, 0, slot_id, actual_start, actual_end, len(items))
    for i, m in enumerate(items):
        assert len(m.media_id) == MEDIA_KEY_BYTES, f"media key is {len(m.media_id)} bytes, want {MEDIA_KEY_BYTES}"
        MEDIA_ITEM.pack_into(out, MEDIA_HEADER.size + i * MEDIA_ITEM.size, m.start_position, m.num_tokens, m.media_id)
    return bytes(out)


def unpack_media_items(raw: bytes, k: int) -> list[MediaItem]:
    """The images listed in one chunk's metadata. Empty for text-only chunks and for the sentinels."""
    if k == 0:
        return []
    slot_id, _, _, count = MEDIA_HEADER.unpack_from(raw, 0)
    if slot_id < 0:  # shutdown / warm-up sentinels fill every word with the same negative value
        return []
    assert count <= k, f"metadata lists {count} images, the header holds {k}"
    return [MediaItem(*MEDIA_ITEM.unpack_from(raw, MEDIA_HEADER.size + i * MEDIA_ITEM.size)) for i in range(count)]
