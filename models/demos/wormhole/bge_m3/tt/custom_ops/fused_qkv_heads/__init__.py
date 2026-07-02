# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 fused QKV-matmul → Q/K/V head-split (scatter-writer) op.

Sweep target: replace the device-time bucket of
``ttnn.experimental.nlp_create_qkv_heads`` (~29 µs/iter × 24 layers = ~702 µs
total in the B1/S512 trace) by routing matmul output tiles directly into
per-head Q/K/V buffers via a custom matmul writer kernel.

Public surface:

    bge_qkv_heads_stock(qkv_fused, *, num_heads=16) -> (q, k, v)
        Baseline: calls ttnn.experimental.nlp_create_qkv_heads. Used by
        sweeps to lock in the stock device-time.

    bge_qkv_heads_scatter(...) -> (q, k, v)
        (Stub) Will route QKV-matmul output tiles directly to Q/K/V via a
        custom writer .cpp. Not implemented in this scaffold pass; the
        sweep file's "scatter" variants currently raise NotImplementedError
        and are recorded as `status="skipped"` rows.
"""

from .op import (
    bge_qkv_heads_headsplit,
    bge_qkv_heads_scatter,
    bge_qkv_heads_stock,
    bge_qkv_heads_tracka,
)

__all__ = [
    "bge_qkv_heads_stock",
    "bge_qkv_heads_tracka",
    "bge_qkv_heads_headsplit",
    "bge_qkv_heads_scatter",
]
