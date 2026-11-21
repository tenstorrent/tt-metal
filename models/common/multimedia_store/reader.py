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

    media_key: bytes
    """Any bytes, e.g. the 32-byte SHA-256 the encoder service returns."""
    offset: int = 0
    length: int = -1


class EmbeddingReader(Protocol):
    """Reads media embeddings by media key.

    Fetches return borrowed, read-only views, not copies. A view may point into the store's own memory, which stays
    put only for a while after the fetch (with Mooncake, the lookup's lease, then the writer's soft pin), or into a
    buffer the reader reuses, which the next fetch overwrites. close() can invalidate either kind. So copy what you
    need (e.g. to device staging) right after a fetch and before the next one, and keep no view past that."""

    def fetch(self, media_key: bytes) -> memoryview:
        """The stored value under media_key, in the model's format, as a borrowed view (see the class docstring).
        Raises Missing or StoreUnavailable."""

    def fetch_batch(self, requests: Sequence[FetchRequest]) -> list[memoryview]:
        """The requested bytes for each request, in order, for a forward that holds several requests' media, as
        borrowed views (see the class docstring). Raises Missing or StoreUnavailable for the first request that
        fails, failing the whole batch."""

    def size(self, media_key: bytes) -> int:
        """Bytes stored under media_key, e.g. to size a staging buffer before fetching. Raises Missing or
        StoreUnavailable."""

    def close(self) -> None:
        """Leaves the store; later reads raise StoreUnavailable. Views a fetch returned may be invalid afterwards."""
