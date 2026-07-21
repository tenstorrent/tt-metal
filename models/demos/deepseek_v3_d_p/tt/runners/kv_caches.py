# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import ttnn
from models.demos.common.prefill.adapter import KvCaches

if TYPE_CHECKING:
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCache


@dataclass
class MlaKvCaches(KvCaches):
    """DeepSeek-family prefill caches owned by the common prefill engine.

    ``kvpe`` is the primary MLA cache with explicit physical encoding. ``index`` is the optional DSA indexer cache used by GLM variants.
    """

    kvpe: MlaKvCache
    index: Optional[ttnn.Tensor] = None
