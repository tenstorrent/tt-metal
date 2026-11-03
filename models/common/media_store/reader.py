# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How prefill reads the media embeddings the encoder service stored for this host."""

from typing import Protocol


class Missing(KeyError):
    """No value is stored under the media key."""


class StoreUnavailable(RuntimeError):
    """The embedding store cannot be reached."""


class EmbeddingReader(Protocol):
    def fetch(self, media_key: bytes) -> memoryview:
        """The stored value under media_key, in the model's format. Raises Missing or StoreUnavailable."""
