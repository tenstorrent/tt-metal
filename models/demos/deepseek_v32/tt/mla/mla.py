# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3.2 MLA: v3 MLA + DeepSeek Sparse Attention (DSA).

Subclasses the v3 ttMLA so projections, RoPE, KV cache, and the e2e plumbing
stay imports from deepseek_v3_d_p; only the V3.2 deltas live here. The class
keeps the name ``ttMLA`` so the copied composition files differ from v3 by a
single import line.

Currently a passthrough (numerically identical to v3) so the e2e path runs and
exposes integration gaps. V3.2 deltas to land here:

  - Indexer: wq_b/wk/k_norm/weights_proj weights -> top-k (k=2048) token index
    selection per query (see reference_tt_single_chip/indexer.py and spec.md).
  - Sparse SDPA: attend only over selected indices instead of dense/ring MLA
    (FlashMLA-style; indices replace masks — missing TT op as of now).
  - Weight cache: indexer weights in _convert_and_cache_weights/build_ttnn_cache.
"""

from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA as _ttMLAv3


class ttMLA(_ttMLAv3):
    """V3.2 MLA with DSA. Passthrough of v3 until the indexer is wired in."""
