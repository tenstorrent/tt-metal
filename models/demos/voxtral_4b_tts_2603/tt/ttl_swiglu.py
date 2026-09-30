# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The tt-lang rung on the prefill fused SwiGLU `silu(h @ Wg) * (h @ Wu)`.

One ttl operation: each core of a 9x8 grid owns a 32x32-tile rectangle of the gated output and
streams its own activation rows and its own gate/up weight columns, accumulating both products
over K in temporaries and writing only `silu(g) * u`. ttl's `@` needs one element type on both
operands and the output, so this path takes bf16 gate/up weights (the stock path's bf4_b weight
cannot be fed to it).

Off unless VOXTRAL_TTL_SWIGLU=1. Measured 2026-09-25 against the stock bf4_b minimal_matmul: PCC
0.999707 (HEAD 0.999784), trace prefill 324.2 -> 2002.4 ms, device 477.0 -> 1078.4 ms. No multicast,
so each core re-reads its activation rows once per 2-tile N block and its weights once per 2-row M
block (~20 GB per layer against ~1 GB), on bf16 weights (4x the bytes) and 72 of 110 cores.

ttl 1.0.1 on device rejects `+=` on a temporary ("must be called on a block acquired from
reserve()") while ttl.sim rejects it on a reserved block: peel k=0 with store() and accumulate into
reserved blocks.
"""
from __future__ import annotations

import os
import re

import ttnn

try:
    import ttl
    import ttl.ttl_api as _ttl_api

    _HAVE_TTL = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAVE_TTL = False

TILE = 32
_GX, _GY = 9, 8
_MB, _NB, _KB = 2, 2, 8


def enabled() -> bool:
    return _HAVE_TTL and os.environ.get("VOXTRAL_TTL_SWIGLU") == "1"


def _repair(source: str) -> str:
    """ttl 1.0.1 emits the pre-rename matmul init API; rewrite it to this tt-metal's."""
    source = re.sub(r"\bmm_block_init_short\(", "matmul_block_init(", source)
    source = re.sub(r"\bmm_block_init\(([^,]+),([^,]+),[^,]+,", r"matmul_block_init(\1,\2,", source)
    if "matmul_block_init(" in source and "compute_kernel_hw_startup" not in source:
        m = re.search(r"matmul_block_init\(([^,]+),([^,]+),", source)
        body = re.search(r"void kernel_main\(\)\s*\{", source)
        if m and body:
            startup = f"\n  compute_kernel_hw_startup<SrcOrder::Reverse>({m.group(1)},{m.group(2)},{m.group(1)});"
            source = source[: body.end()] + startup + source[body.end() :]
    return source


if _HAVE_TTL:
    _orig_write = _ttl_api._write_kernel_to_tmp

    def _write_repaired(name, source):
        return _orig_write(name, _repair(source))

    _ttl_api._write_kernel_to_tmp = _write_repaired

    @ttl.operation(grid=(_GX, _GY))
    def swiglu_matmul(a: ttnn.Tensor, wg: ttnn.Tensor, wu: ttnn.Tensor, y: ttnn.Tensor) -> None:
        m_core = a.shape[0] // TILE // _GY
        n_core = wg.shape[1] // TILE // _GX
        k_tiles = a.shape[1] // TILE
        m_blocks = m_core // _MB
        n_blocks = n_core // _NB
        k_blocks = k_tiles // _KB
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(_MB, _KB), block_count=2)
        g_dfb = ttl.make_dataflow_buffer_like(wg, shape=(_KB, _NB), block_count=2)
        u_dfb = ttl.make_dataflow_buffer_like(wu, shape=(_KB, _NB), block_count=2)
        ga_dfb = ttl.make_dataflow_buffer_like(y, shape=(_MB, _NB), block_count=2)
        ua_dfb = ttl.make_dataflow_buffer_like(y, shape=(_MB, _NB), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(_MB, _NB), block_count=2)

        @ttl.datamovement()
        def read():
            x, cy = ttl.node(dims=2)
            for mb in range(m_blocks):
                r = cy * m_core + mb * _MB
                for nb in range(n_blocks):
                    c = x * n_core + nb * _NB
                    for kb in range(k_blocks):
                        k = kb * _KB
                        with a_dfb.reserve() as ab, g_dfb.reserve() as gb, u_dfb.reserve() as ub:
                            ta = ttl.copy(a[r : r + _MB, k : k + _KB], ab)
                            tg = ttl.copy(wg[k : k + _KB, c : c + _NB], gb)
                            tu = ttl.copy(wu[k : k + _KB, c : c + _NB], ub)
                            ta.wait()
                            tg.wait()
                            tu.wait()

        @ttl.compute()
        def compute():
            for _ in range(m_blocks):
                for _ in range(n_blocks):
                    with ga_dfb.reserve() as ga, ua_dfb.reserve() as ua:
                        with a_dfb.wait() as ab, g_dfb.wait() as gb, u_dfb.wait() as ub:
                            ga.store(ab @ gb)
                            ua.store(ab @ ub)
                        for _ in range(k_blocks - 1):
                            with a_dfb.wait() as ab, g_dfb.wait() as gb, u_dfb.wait() as ub:
                                ga += ab @ gb
                                ua += ab @ ub
                    with ga_dfb.wait() as gv, ua_dfb.wait() as uv, y_dfb.reserve() as yb:
                        yb.store(ttl.math.silu(gv) * uv)

        @ttl.datamovement()
        def write():
            x, cy = ttl.node(dims=2)
            for mb in range(m_blocks):
                r = cy * m_core + mb * _MB
                for nb in range(n_blocks):
                    c = x * n_core + nb * _NB
                    with y_dfb.wait() as yb:
                        ttl.copy(yb, y[r : r + _MB, c : c + _NB]).wait()


def supports(h, wg) -> bool:
    if not enabled() or wg is None:
        return False
    rows = 1
    for d in list(h.shape)[:-1]:
        rows *= int(d)
    k = int(h.shape[-1])
    n = int(wg.shape[-1])
    return (
        h.dtype == ttnn.bfloat16
        and wg.dtype == ttnn.bfloat16
        and not h.memory_config().is_sharded()
        and rows % (TILE * _GY * _MB) == 0
        and n % (TILE * _GX * _NB) == 0
        and k % (TILE * _KB) == 0
    )


def apply(h, wg, wu):
    """`silu(h @ wg) * (h @ wu)` in bf16, same shape contract as the stock fused op."""
    dims = [int(d) for d in h.shape]
    rows = 1
    for d in dims[:-1]:
        rows *= d
    a = ttnn.reshape(h, (rows, dims[-1])) if len(dims) != 2 else h
    n = int(wg.shape[-1])
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, a.device(), ttnn.DRAM_MEMORY_CONFIG
    )
    swiglu_matmul(a, wg, wu, y)
    return y if len(dims) == 2 else ttnn.reshape(y, tuple(dims[:-1] + [n]))
