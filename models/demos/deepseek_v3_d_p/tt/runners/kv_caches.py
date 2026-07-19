# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import ttnn
from models.demos.common.prefill.adapter import KvCaches


@dataclass
class MlaKvCaches(KvCaches):
    """Concrete KvCaches for the MLA/DSA prefill runner: the primary MLA KVPE cache (``kvpe``) and an
    optional sparse-DSA indexer key cache (``index``, ``None`` for dense MLA). The adapters' allocate_kv_cache
    builds it and TtPrefillRuntime consumes it via ``.kvpe`` / ``.index``."""

    kvpe: ttnn.Tensor
    index: Optional[ttnn.Tensor] = None
