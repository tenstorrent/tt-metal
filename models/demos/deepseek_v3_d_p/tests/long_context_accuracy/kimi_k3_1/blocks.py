# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 device Blocks of the long-context cases (``common/prefill/long_context_accuracy/block.DeviceBlock``),
each fed host rows from the FP64 chain golden ``k31-l0-3-fp64-1m`` and returning host ``BlockOutputs`` in global
row order:

* ``KimiK31MoeBlock`` (m06n.11.4): ``TtMoEBlockFfn`` of layer 1 or 2 on one 4x4 stage; ``extras`` hold the routing
  (expert ids and weights); no carried state.
* ``KimiK31DsaBlock`` (m06n.11.5): ``ttMLA`` of layer 3 with the adapter's caches, sized from the context length;
  ``extras`` hold the indexer top-k; the carried state is the KV and index caches.

Each Block's code, including its imports, stays inside its own section below.
"""

from __future__ import annotations

# ==== MoE: KimiK31MoeBlock (m06n.11.4) ==============================================================================
# ==== end of MoE ====================================================================================================

# ==== DSA: KimiK31DsaBlock (m06n.11.5) ==============================================================================
# ==== end of DSA ====================================================================================================
