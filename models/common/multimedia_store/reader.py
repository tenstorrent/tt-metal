# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How prefill reads the media embeddings the encoder service stored for this host."""

from typing import NamedTuple, Protocol, Sequence, Union


class Missing(KeyError):
    """No value is stored under the media key."""


class StoreUnavailable(RuntimeError):
    """The embedding store cannot be reached."""


class FetchRequest(NamedTuple):
    """Bytes [offset, offset + length) of the value under media_key; length -1 runs to the end of the value."""

    media_key: bytes
    offset: int = 0
    length: int = -1


FetchResult = Union[memoryview, Missing, StoreUnavailable]


class EmbeddingReader(Protocol):
    def fetch(self, media_key: bytes) -> memoryview:
        """The stored value under media_key, in the model's format. Raises Missing or StoreUnavailable."""

    def fetch_batch(self, requests: Sequence[FetchRequest]) -> list[FetchResult]:
        """One result per request, in order: the requested bytes, or the Missing or StoreUnavailable error for that
        request alone, so the caller can fail just the requests it belongs to. For a forward that holds several
        requests' media."""
