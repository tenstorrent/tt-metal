# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Micro-benchmark for the AdaLN six-way split at ``transformer_wan.py:209``.

An op-level profile of the untraced denoise put ``ttnn.chunk`` at ~950 us per call on a
**4608-element** tensor (1, 1, 6, 768), 60 calls per step. A tensor that small cannot be
device-bound, so the cost is the composite op itself; in TILE layout dim 2 is a tile
dimension, so a 6-row tensor is padded to 32 and six size-1 slices are six kernel
launches over mostly padding.

Measures the current form against alternatives on the exact production shape, in
isolation (no queue contention from other ops), and checks every alternative returns
identical values. Reported two ways:

  enqueue  N calls with a single sync at the end -- amortised cost, what a stream of
           these actually costs when the host is not stalled.
  sync     one call with a sync after it -- serialised host + device.

    pytest models/tt_dit/tests/models/wan2_2/test_adaln_chunk_ti2v_5b.py -sv --timeout=0
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

# T2V: temb is (1, B, 6, D/tp) with D=3072, tp=4 -> table_width 768. The six outputs are
# shift/scale/gate for self-attn and for the FFN.
_B, _CHUNKS, _WIDTH = 1, 6, 768
_ITERS = 50


def _variants(x_tile, x_rm, x_flat):
    """name -> callable returning the six (1,1,1,768) pieces."""

    def current():
        return ttnn.chunk(x_tile, _CHUNKS, dim=2)

    def slices_dim2():
        return [ttnn.slice(x_tile, [0, 0, i, 0], [1, _B, i + 1, _WIDTH], [1, 1, 1, 1]) for i in range(_CHUNKS)]

    def chunk_flat_dim3():
        # The per-token branch at transformer_wan.py:206 already uses this layout: one
        # row of 6*768, split along the last dim, which is tile-aligned at 768 = 24 tiles.
        return ttnn.chunk(x_flat, _CHUNKS, dim=3)

    def slices_flat_dim3():
        return [
            ttnn.slice(x_flat, [0, 0, 0, i * _WIDTH], [1, _B, 1, (i + 1) * _WIDTH], [1, 1, 1, 1])
            for i in range(_CHUNKS)
        ]

    def chunk_row_major():
        return ttnn.chunk(x_rm, _CHUNKS, dim=2)

    return {
        "chunk dim2 TILE (current)": current,
        "6x slice dim2 TILE": slices_dim2,
        "chunk dim3 TILE flat": chunk_flat_dim3,
        "6x slice dim3 TILE flat": slices_flat_dim3,
        "chunk dim2 ROW_MAJOR": chunk_row_major,
    }


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["single"], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
def test_adaln_chunk_variants(mesh_device, dtype):
    torch.manual_seed(0)
    torch_x = torch.randn(1, _B, _CHUNKS, _WIDTH, dtype=torch.float32)

    x_tile = ttnn.from_torch(torch_x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    x_rm = ttnn.from_torch(torch_x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    x_flat = ttnn.from_torch(
        torch_x.reshape(1, _B, 1, _CHUNKS * _WIDTH), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )

    ref = None
    results = {}
    for name, fn in _variants(x_tile, x_rm, x_flat).items():
        try:
            out = fn()  # warm: compile programs for this shape
            ttnn.synchronize_device(mesh_device)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ADALN {name}: unsupported ({e!r})")
            continue

        # Values must match the current form exactly -- this is pure data movement.
        got = torch.stack([ttnn.to_torch(t).float().reshape(-1) for t in out])
        if ref is None:
            ref = got
            max_abs = 0.0
        else:
            max_abs = (got - ref).abs().max().item()

        t0 = time.perf_counter()
        for _ in range(_ITERS):
            fn()
        ttnn.synchronize_device(mesh_device)
        enqueue_us = 1e6 * (time.perf_counter() - t0) / _ITERS

        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh_device)
        sync_us = 1e6 * (time.perf_counter() - t0)

        results[name] = (enqueue_us, sync_us, max_abs)
        logger.info(f"ADALN {name}: enqueue={enqueue_us:.1f}us sync={sync_us:.1f}us max_abs_diff={max_abs:.3e}")

    assert results, "no variant ran"
    base = results.get("chunk dim2 TILE (current)")
    if base:
        for name, (enq, _, _) in results.items():
            if name != "chunk dim2 TILE (current)" and enq > 0:
                logger.info(f"ADALN_SPEEDUP {name}: {base[0] / enq:.2f}x vs current")
    for name, (_, _, max_abs) in results.items():
        assert max_abs == 0.0, f"{name} disagrees with the current form (max_abs_diff={max_abs:.3e})"
