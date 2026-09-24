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

    ``kvpe`` is the primary MLA cache with explicit physical encoding. ``index`` is the optional DSA
    indexer cache used by GLM variants.
    """

    kvpe: MlaKvCache
    index: Optional[ttnn.Tensor] = None


@dataclass
class V4FlashKvCaches(KvCaches):
    """DeepSeek-V4-Flash prefill caches owned by the common prefill engine -- one tensor per KV group in the fixed
    order of ``tt.v4.kv_contract.KV_GROUPS`` (the prefill <-> decode contract). Each is
    ``[users * layers_in_group, 1, rows, width]``, REPLICATED over SP and addressed by unified row; ``geometry``
    maps a group's batch index back to the GLOBAL layer it holds (``geometry.layers(group)[batch % n_layers]``).

    ``swa_window``   layers 0,1: the 128-row window ring                      (_CACHE_DTYPE tile, 512 wide)
    ``hca_unified``  HCA layers: window ring + compressed entries (1 per 128)  (_CACHE_DTYPE tile, 512 wide)
    ``csa_unified``  CSA layers: window ring + compressed entries (1 per 4)    (bf16 ROW_MAJOR, 512 wide)
    ``csa_index_k``  CSA layers: lightning-indexer keys, one per entry         (bfp8 tile, 128 wide)
    ``csa_pending`` / ``hca_pending``: compressor partial-window state (bf16 ROW_MAJOR, 1024 wide) -- contract
    gap DS4F-0242, allocated so the prefill side can export it once decode says where it lands.
    A group whose kind has no layer on this rank is None.
    """

    geometry: object  # tt.v4.layer_kinds.V4FlashKvGeometry (kept untyped: this module must import without torch)
    swa_window: Optional[ttnn.Tensor] = None
    hca_unified: Optional[ttnn.Tensor] = None
    csa_unified: Optional[ttnn.Tensor] = None
    csa_index_k: Optional[ttnn.Tensor] = None
    csa_pending: Optional[ttnn.Tensor] = None
    hca_pending: Optional[ttnn.Tensor] = None

    def group_tensors(self, *, include_pending: bool = True) -> dict:
        """``{group_name: tensor}`` for the groups this rank holds, in contract order."""
        names = ("swa_window", "hca_unified", "csa_unified", "csa_index_k", "csa_pending", "hca_pending")
        out = {}
        for n in names:
            if not include_pending and n.endswith("_pending"):
                continue
            t = getattr(self, n)
            if t is not None:
                out[n] = t
        return out
