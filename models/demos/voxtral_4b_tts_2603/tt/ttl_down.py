# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The tt-lang rung on the acoustic FFN down projection `gated @ W2` (96 x 9216 x 3072).

One ttl operation on an 8x6 grid: each core owns every output row and 2 of the 96 output tile
columns, streams the whole activation and its own weight columns in K blocks of 8, and
accumulates over K in a reserved block. ttl's `@` needs one element type on both operands and
the output, so this path takes a bf16 W2 (the stock path's bf4_b weight cannot be fed to it) and
returns bf16, typecast to the stock path's float32.

Off unless VOXTRAL_TTL_DOWN=1. Measured 2026-09-27 against the stock bf4_b 1D-mcast linear: device
283.45 -> 340.37 ms, trace acoustic 23.04 -> 36.85 ms, full pipeline 227.79 -> 243.36 ms. 4x the
weight bytes on 48 of 110 cores, with no multicast of the activation.
"""
from __future__ import annotations

import os

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import ttl_swiglu

try:
    import ttl

    _HAVE_TTL = ttl_swiglu._HAVE_TTL
except ImportError:  # pragma: no cover - depends on the environment
    _HAVE_TTL = False

TILE = 32
_GX, _GY = 8, 6
_MB, _NB, _KB = 3, 2, 8


def enabled() -> bool:
    return _HAVE_TTL and os.environ.get("VOXTRAL_TTL_DOWN") == "1"


if _HAVE_TTL:

    @ttl.operation(grid=(_GX, _GY))
    def down_matmul(a: ttnn.Tensor, w: ttnn.Tensor, y: ttnn.Tensor) -> None:
        m_blocks = a.shape[0] // TILE // _MB
        k_blocks = a.shape[1] // TILE // _KB
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(_MB, _KB), block_count=2)
        w_dfb = ttl.make_dataflow_buffer_like(w, shape=(_KB, _NB), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(_MB, _NB), block_count=2)

        @ttl.datamovement()
        def read():
            x, cy = ttl.node(dims=2)
            c = (cy * _GX + x) * _NB
            for mb in range(m_blocks):
                r = mb * _MB
                for kb in range(k_blocks):
                    k = kb * _KB
                    with a_dfb.reserve() as ab, w_dfb.reserve() as wb:
                        ta = ttl.copy(a[r : r + _MB, k : k + _KB], ab)
                        tw = ttl.copy(w[k : k + _KB, c : c + _NB], wb)
                        ta.wait()
                        tw.wait()

        @ttl.compute()
        def compute():
            for _ in range(m_blocks):
                with acc_dfb.reserve() as acc:
                    with a_dfb.wait() as ab, w_dfb.wait() as wb:
                        acc.store(ab @ wb)
                    for _ in range(k_blocks - 1):
                        with a_dfb.wait() as ab, w_dfb.wait() as wb:
                            acc += ab @ wb

        @ttl.datamovement()
        def write():
            x, cy = ttl.node(dims=2)
            c = (cy * _GX + x) * _NB
            for mb in range(m_blocks):
                r = mb * _MB
                with acc_dfb.wait() as yb:
                    ttl.copy(yb, y[r : r + _MB, c : c + _NB]).wait()


def weight(w_t, device, from_torch):
    """The bf16 `[K, N]` weight this path reads, or None when it is off."""
    return from_torch(w_t, device, dtype=ttnn.bfloat16) if enabled() else None


def supports(x, w) -> bool:
    if not enabled() or w is None:
        return False
    rows = 1
    for d in list(x.shape)[:-1]:
        rows *= int(d)
    return (
        x.dtype == ttnn.bfloat16
        and not x.memory_config().is_sharded()
        and rows % (TILE * _MB) == 0
        and int(w.shape[-1]) == TILE * _NB * _GX * _GY
        and int(x.shape[-1]) % (TILE * _KB) == 0
    )


def apply(x, w):
    """`x @ w` as float32, same shape contract as the stock linear."""
    dims = [int(d) for d in x.shape]
    rows = 1
    for d in dims[:-1]:
        rows *= d
    a = ttnn.to_memory_config(ttnn.reshape(x, (rows, dims[-1])), ttnn.DRAM_MEMORY_CONFIG)
    n = int(w.shape[-1])
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, a.device(), ttnn.DRAM_MEMORY_CONFIG
    )
    with ttl_swiglu.repaired_kernel_writes():
        down_matmul(a, w, y)
    return ttnn.reshape(ttnn.typecast(y, ttnn.float32), tuple(dims[:-1] + [n]))
