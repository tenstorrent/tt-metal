# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``untilize_rows``: one row tile ``[1, 1, rows, W]`` TILE -> ``[1, 1, rows, W]`` ROW_MAJOR in one program.

The composed chain is ``ttnn.to_layout``.  Pure data movement (the stock interleaved reader feeds one tile per CB
page; the writer scatters the valid rows out of the tile's faces), so the output is bitwise by construction: the
foundation's worked example of the plumbing.  Elements of 2 or 4 bytes, widths of whole tiles.
"""

from __future__ import annotations

import torch

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, GateSpec, register

NAME = "untilize_rows"
READER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
WRITER = fp.kernel_source(NAME, "writer_rows.cpp")
ROUTER_WIDTH = 512


def untilize_rows(tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    rows = fp.rows_of(tensor)
    width = fp.tile_width_of(tensor)
    if tensor.layout != ttnn.TILE_LAYOUT or tensor.dtype not in fp.ELEMENT_BYTES:
        raise ValueError(
            f"untilize_rows takes a TILE tensor of 2- or 4-byte elements, got {tensor.layout} {tensor.dtype}"
        )
    mesh = tensor.device()
    out = fp.allocate((1, 1, rows, width), tensor.dtype, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    work = fp.split_work(width // fp.TILE, mesh)
    cores = fp.core_set(work)
    tile_cb = fp.cb_descriptor(0, tensor.dtype, fp.TILE_BYTES[tensor.dtype], 2, cores)
    reader = fp.reader_kernel(
        READER, cores, fp.accessor_args(tensor), [(w.core, [tensor.buffer_address(), w.count, w.start]) for w in work]
    )
    writer = fp.writer_kernel(
        WRITER,
        cores,
        [fp.ELEMENT_BYTES[tensor.dtype], rows, *fp.accessor_args(out)],
        [(w.core, [out.buffer_address(), w.count, w.start]) for w in work],
    )
    meta = fp.program_meta(NAME, "rows", rows, reads=(tensor,), writes=(out,), cores=len(work))  # data movement
    return fp.run_program([tensor, out], fp.program_descriptor([reader, writer], cbs=[tile_cb]), meta=meta)


def untilize_rows_composed(tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)


def _gate_inputs(mesh, capture, positions, layer):
    """The captured fp32 router logits of ``positions`` at ``layer`` as one row tile ``[1, 1, rows, 512]``."""

    index = [capture["positions"].index(p) for p in positions]
    logits = capture["router_logits"][index, layer].float().reshape(1, 1, len(positions), ROUTER_WIDTH)
    return {"tensor": ttnn.from_torch(logits, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)}


def _gate_reference(oracle, positions, layer):
    index = [oracle["positions"].index(p) for p in positions]
    return oracle["router_logits"][index, layer].float()


register(
    FusedKernel(
        name=NAME,
        replaces="ttnn.to_layout(TILE -> ROW_MAJOR) of one row tile [1, 1, rows, W]",
        tolerance=BITWISE,
        fused=untilize_rows,
        composed=untilize_rows_composed,
        gate=GateSpec(
            inputs=_gate_inputs,
            output=lambda out: ttnn.to_torch(out).reshape(-1, ROUTER_WIDTH),
            reference=_gate_reference,
            layers=tuple(range(48)),
        ),
    )
)
