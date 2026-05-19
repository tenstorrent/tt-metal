# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused multi-head concat (Track A) — batched-barrier kernels.

Sweep target: replace the device-time bucket of
``ttnn.experimental.nlp_concat_heads`` (~10 µs/iter × 24 layers ≈ 250 µs total
in the B1/S512 trace) by collapsing the stock per-tile NoC barriers in the
reader into a single barrier per block.

Public surface:

    bge_concat_heads_stock(context, *, out_memcfg) -> context_concat
        Baseline: calls ttnn.experimental.nlp_concat_heads.

    bge_concat_heads_tracka(context, *, out_memcfg) -> context_concat
        Custom batched-barrier reader+writer via ttnn.generic_op.
"""

from .op import bge_concat_heads_headsplit, bge_concat_heads_stock, bge_concat_heads_tracka

__all__ = [
    "bge_concat_heads_stock",
    "bge_concat_heads_tracka",
    "bge_concat_heads_headsplit",
]
