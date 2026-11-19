# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Media keys of the test producer's synthetic images: the key carries the image's token count, so a stub
embedding reader can rebuild the image's value from the key alone."""

import struct

from models.demos.common.prefill.prefill_chunk import MEDIA_KEY_BYTES

_STUB_KEY = struct.Struct("<4sII")
_STUB_MAGIC = b"STUB"


def stub_media_key(num_tokens: int, index: int) -> bytes:
    return _STUB_KEY.pack(_STUB_MAGIC, num_tokens, index).ljust(MEDIA_KEY_BYTES, b"\0")


def stub_key_tokens(media_key: bytes) -> int:
    """The token count a stub key carries; ValueError for any other key."""
    magic, num_tokens, _ = _STUB_KEY.unpack_from(media_key)
    if magic != _STUB_MAGIC:
        raise ValueError(f"not a stub media key: {media_key.hex()}")
    return num_tokens
