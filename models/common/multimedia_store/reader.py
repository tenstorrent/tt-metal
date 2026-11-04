# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How prefill reads the media embeddings the encoder service stored for this host."""

from typing import NamedTuple, Protocol, Sequence


class Missing(KeyError):
    """No value is stored under the media key."""


class StoreUnavailable(RuntimeError):
    """The embedding store cannot be reached."""


class FetchRequest(NamedTuple):
    """Bytes [offset, offset + length) of the value under media_key; length -1 runs to the end of the value."""

    media_key: str
    """ASCII; the module that generates media keys guarantees it."""
    offset: int = 0
    length: int = -1


class EmbeddingReader(Protocol):
    def fetch(self, media_key: str) -> memoryview:
        """The stored value under media_key, in the model's format. Raises Missing or StoreUnavailable."""

    def fetch_batch(self, requests: Sequence[FetchRequest]) -> list[memoryview]:
        """The requested bytes for each request, in order, for a forward that holds several requests' media. Raises
        Missing or StoreUnavailable for the first request that fails, failing the whole batch."""
