# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-forward prefill context threaded through the layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PrefillCtx:
    caches: Any  # Qwen38Caches
    user_id: int
    start: int  # global position of this forward's first token (actual_start); chunk-aligned
    valid_end: int  # global end of the valid tokens (actual_end); tokens past it are padding
    tokens: int  # tokens in this forward (sp * s_local)
    cos: Any = None
    sin: Any = None
    cache_mask: Any = None  # position mask for the composed cache read (chunks after the first)

    @property
    def valid_len(self) -> int:
        return self.valid_end - self.start
