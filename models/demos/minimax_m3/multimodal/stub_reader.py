# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stands in for the media embedding store: builds a stub key's value in format.py's layout, with deterministic rows
seeded from the key. Implements models/common/multimedia_store/reader.py's EmbeddingReader."""

import hashlib
import struct
from collections import OrderedDict

import torch

from models.common.multimedia_store.reader import FetchRequest, Missing
from models.demos.common.prefill.media_stub import stub_key_tokens
from models.demos.minimax_m3.multimodal import format


class StubEmbeddingReader:
    """Values are kept, least recently used dropped first beyond max_bytes, so a repeated fetch costs a lookup
    rather than a fresh randn of the whole value — the store it stands in for returns a view of what it holds."""

    def __init__(self, hidden: int, max_bytes: int = 1 << 30):
        self.hidden = hidden
        self.max_bytes = max_bytes
        self._values: OrderedDict[bytes, memoryview] = OrderedDict()
        self._bytes = 0

    def fetch(self, media_key: bytes) -> memoryview:
        return self.fetch_batch([FetchRequest(media_key)])[0]

    def fetch_batch(self, requests) -> list[memoryview]:
        out = []
        for media_key, offset, length in (FetchRequest(*r) for r in requests):
            value = self._value(media_key)
            out.append(value[offset:] if length == -1 else value[offset : offset + length])
        return out

    def _value(self, media_key: bytes) -> memoryview:
        cached = self._values.get(media_key)
        if cached is not None:
            self._values.move_to_end(media_key)
            return cached
        try:
            n = stub_key_tokens(media_key)
        except (ValueError, struct.error) as e:
            raise Missing(media_key.hex()) from e
        buf = memoryview(bytearray(format.stored_nbytes(n, self.hidden)))
        format.Header("image").write(buf)
        gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(media_key).digest()[:7], "little"))
        rows = (torch.randn(n, self.hidden, generator=gen) * 0.02).to(torch.bfloat16)
        format.embeddings(buf, self.hidden)[:] = rows.view(torch.int16).numpy()
        self._values[media_key] = buf
        self._bytes += len(buf)
        while self._bytes > self.max_bytes and len(self._values) > 1:
            self._bytes -= len(self._values.popitem(last=False)[1])
        return buf
