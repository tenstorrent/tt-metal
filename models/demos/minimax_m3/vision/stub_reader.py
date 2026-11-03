# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stands in for the embedding store until its reader exists: builds a stub key's value in format.py's layout,
with deterministic rows seeded from the key."""

import hashlib

import torch

from models.common.media_store.reader import Missing
from models.demos.common.prefill.media_stub import stub_key_tokens
from models.demos.minimax_m3.vision import format


class StubEmbeddingReader:
    def __init__(self, hidden: int):
        self.hidden = hidden

    def fetch(self, media_key: bytes) -> memoryview:
        try:
            n = stub_key_tokens(media_key)
        except ValueError as e:
            raise Missing(media_key.hex()) from e
        buf = memoryview(bytearray(format.value_size(n, self.hidden)))
        h = format.Header(1, n, n, self.hidden)
        format.write_header(buf, h)
        gen = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(media_key).digest()[:7], "little"))
        rows = (torch.randn(n, self.hidden, generator=gen) * 0.02).to(torch.bfloat16)
        format.embeddings(buf, h)[:] = rows.view(torch.int16).numpy()
        return buf
