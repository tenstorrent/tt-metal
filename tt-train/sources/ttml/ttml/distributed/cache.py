# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Simple LRU plan cache for sharding plans."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Tuple


class PlanCache:
    """Cache mapping (op_name, input_layouts, kwargs_hash) -> ShardingPlan."""

    def __init__(self, maxsize: int = 2048):
        self._maxsize = maxsize
        self._cache: OrderedDict[Tuple, Any] = OrderedDict()

    def get(self, key: Tuple) -> Optional[Any]:
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def put(self, key: Tuple, plan: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = plan

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
