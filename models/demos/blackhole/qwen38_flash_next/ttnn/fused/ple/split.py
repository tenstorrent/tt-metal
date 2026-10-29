# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``QWEN38_PLE_SPLIT=1`` (default off): the PLE gate (``ple.gate``, one core, 49 us) as two programs.  Program 1 is the
elementwise products ``fp32(key) * fp32(query)`` of the 80 column tiles, spread over ``PRODUCT_CORES`` cores (each tile the
same unpack, ``mul_binary_tile`` and fp32 pack as the one-core kernel, so the bytes are the same); program 2 is the one-core
gate kernel without that step: the accurate fp32 fold over the 80 product tiles in tile order, the scalar chain and the
gated block, unchanged.  Same inputs, same output layout as ``ple.gate``."""

from __future__ import annotations

import os

import ttnn

from .. import program as fp
from . import (
    BF16,
    BRANCHES,
    FP32,
    GATE_WRITER,
    GLOBAL_TILES,
    HIDDEN,
    LOCAL_HIDDEN,
    LOCAL_TILES,
    NAME,
    TILE_BF16,
    TILE_FP32,
    _expect,
    constants,
)

VARIABLE = "QWEN38_PLE_SPLIT"
PRODUCT_CORES = int(
    os.environ.get("QWEN38_PLE_SPLIT_CORES", "80")
)  # 80 tiles: one per core (x57: 41.5 us vs 44.1 at 20 cores, 51.2 one-core gate)
PROD_READER = fp.kernel_source(NAME, "gate_prod_reader.cpp")
PROD_COMPUTE = fp.kernel_source(NAME, "gate_prod_compute.cpp")
PROD_WRITER = fp.kernel_source(NAME, "gate_prod_writer.cpp")
SUM_READER = fp.kernel_source(NAME, "gate_sum_reader.cpp")
SUM_COMPUTE = fp.kernel_source(NAME, "gate_sum_compute.cpp")


def enabled() -> bool:
    return os.environ.get(VARIABLE) == "1"


def products(key_global, query_global):
    """The fp32 products ``[1,1,4,2560]`` TILE of the two global norms, spread over cores."""

    mesh = key_global.device()
    out = fp.allocate((1, 1, BRANCHES, HIDDEN), FP32, ttnn.TILE_LAYOUT, mesh)
    work = fp.split_work(GLOBAL_TILES, mesh, cores=PRODUCT_CORES)
    cores = fp.core_rectangle(work, mesh)
    n = work[0].count
    if any(w.count != n for w in work):
        raise ValueError("the products split needs equal tile blocks")
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, n, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, n, cores),
        fp.cb_descriptor(2, FP32, TILE_FP32, n, cores),
    ]
    reader = fp.reader_kernel(
        PROD_READER,
        cores,
        [n] + fp.accessor_args(key_global) + fp.accessor_args(query_global),
        [(w.core, [key_global.buffer_address(), query_global.buffer_address(), w.start]) for w in work],
    )
    compute = fp.compute_kernel(PROD_COMPUTE, cores, [n], fp32_dest=True)
    writer = fp.writer_kernel(
        PROD_WRITER,
        cores,
        [n] + fp.accessor_args(out),
        [(w.core, [out.buffer_address(), w.start]) for w in work],
    )
    meta = fp.program_meta(
        NAME,
        "gate_products",
        BRANCHES,
        reads=(key_global, query_global),
        writes=(out,),
        flops=BRANCHES * HIDDEN,
        cores=len(work),
    )
    return fp.run_program(
        [key_global, query_global, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta
    )


def gate(key_global, query_global, value, *, debug: bool = False):
    """``ple.gate`` as the products program plus the one-core fold/scalar/gated program."""

    _expect(key_global, (1, 1, BRANCHES, HIDDEN), BF16, "PLE key global norm")
    _expect(query_global, (1, 1, BRANCHES, HIDDEN), BF16, "PLE query global norm")
    _expect(value, (1, 1, 1, LOCAL_HIDDEN), BF16, "PLE value projection")
    mesh = key_global.device()
    scale, zero = constants(mesh)
    prods = products(key_global, query_global)
    out = fp.allocate((1, 1, BRANCHES, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    dbg = (
        [fp.allocate((1, 1, fp.TILE, fp.TILE), FP32, ttnn.TILE_LAYOUT, mesh) for _ in range(2)] if debug else [out, out]
    )
    core = ttnn.CoreCoord(0, 0)
    one = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    cbs = [
        fp.cb_descriptor(2, FP32, TILE_FP32, GLOBAL_TILES, one),
        fp.cb_descriptor(3, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(4, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(5, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(6, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(7, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(8, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(9, BF16, TILE_BF16, LOCAL_TILES, one),
        fp.cb_descriptor(16, BF16, TILE_BF16, LOCAL_TILES, one),
    ]
    if debug:
        cbs += [fp.cb_descriptor(17, FP32, TILE_FP32, 1, one), fp.cb_descriptor(18, FP32, TILE_FP32, 1, one)]
    tensors = (prods, value, scale, zero)
    reader = fp.reader_kernel(
        SUM_READER,
        one,
        [GLOBAL_TILES, LOCAL_TILES] + sum((fp.accessor_args(t) for t in tensors), []),
        [
            (
                core,
                [
                    prods.buffer_address(),
                    0,
                    value.buffer_address(),
                    scale.buffer_address(),
                    zero.buffer_address(),
                    0,
                    0,
                ],
            )
        ],
    )
    compute = fp.compute_kernel(
        SUM_COMPUTE,
        one,
        [GLOBAL_TILES, LOCAL_TILES, int(debug)],
        fp32_dest=True,
        unpack_to_dest_fp32=(2, 4, 5, 6, 7, 8),
    )
    writer = fp.writer_kernel(
        GATE_WRITER,
        one,
        [LOCAL_TILES, int(debug)] + sum((fp.accessor_args(t) for t in (out, dbg[0], dbg[1])), []),
        [(core, [out.buffer_address(), dbg[0].buffer_address(), dbg[1].buffer_address(), 0])],
    )
    io = [prods, value, scale, zero, out] + (dbg if debug else [])
    meta = fp.program_meta(
        NAME,
        "gate_fold",
        BRANCHES,
        reads=(prods, value, scale, zero),
        writes=(out, *(dbg if debug else [])),
        flops=BRANCHES * HIDDEN + 8 * BRANCHES + BRANCHES * LOCAL_HIDDEN,
        cores=1,
    )
    fp.run_program(io, fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)
    ttnn.deallocate(prods)
    return (out, dbg[0], dbg[1]) if debug else out
