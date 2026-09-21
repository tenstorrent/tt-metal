# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chunk context handed to every layer.

One object rather than a growing kwarg list, so the two token mixers can share a single
``mix(x, ctx)`` signature even though they need different things out of it — the attention layers
read ``cos``/``sin`` and the KV cache, the Gated DeltaNet layers read their own recurrent state.
That is what lets ``DecoderLayer`` hold a mixer chosen once at construction instead of branching
on the layer type in ``forward``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ttnn

from .caches import PrefillCaches


@dataclass(frozen=True)
class ChunkContext:
    """``cached_len`` is the valid GLOBAL prefix length before this chunk — 0 for one-shot, and
    what selects the cache-read attention path and the seeded GDN state."""

    cos: Optional[ttnn.Tensor]
    sin: Optional[ttnn.Tensor]
    caches: Optional[PrefillCaches] = None
    user_id: int = 0
    cached_len: int = 0
