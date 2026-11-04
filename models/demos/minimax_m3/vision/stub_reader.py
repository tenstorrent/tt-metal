# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stands in for the media embedding store: builds a stub key's value in format.py's layout, with deterministic rows
seeded from the key. Implements models/common/multimedia_store/reader.py's EmbeddingReader."""

import hashlib

import torch

from models.common.multimedia_store.reader import FetchRequest, Missing
from models.demos.common.prefill.media_stub import stub_key_tokens
from models.demos.minimax_m3.vision import format


class StubEmbeddingReader:
    def __init__(self, hidden: int):
        self.hidden = hidden

    def fetch(self, media_key: str) -> memoryview:
        return self.fetch_batch([FetchRequest(media_key)])[0]

    def fetch_batch(self, requests) -> list[memoryview]:
        out = []
        for media_key, offset, length in (FetchRequest(*r) for r in requests):
            value = self._value(media_key)
            out.append(value[offset:] if length == -1 else value[offset : offset + length])
        return out

    def _value(self, media_key: str) -> memoryview:
        try:
            key = bytes.fromhex(media_key)
            n = stub_key_tokens(key)
        except (ValueError, TypeError) as e:
            raise Missing(media_key) from e
        buf = memoryview(bytearray(format.value_size(n, self.hidden)))
        h = format.Header(1, n, n, self.hidden)
        format.write_header(buf, h)
        gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:7], "little"))
        rows = (torch.randn(n, self.hidden, generator=gen) * 0.02).to(torch.bfloat16)
        format.embeddings(buf, h)[:] = rows.view(torch.int16).numpy()
        return buf
