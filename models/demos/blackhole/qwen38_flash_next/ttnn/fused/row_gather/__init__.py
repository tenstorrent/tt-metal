# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``row_gather``: the per-block hidden all-gather of a decode row as one program, moving only the valid rows.

Stands in for ``ttnn.all_gather(x, dim=3, cluster_axis=<line>, memory_config=<caller's>)`` of a local ``[1, 1, rows,
W]`` bf16 TILE tensor (``rows`` 1..32 valid rows of one padded tile row) -> ``[1, 1, rows, W * TP_SIZE]``, device d's
shard at columns ``[W * d, W * (d + 1))`` (page ``t + (W / 32) * d`` of the tile layout).  The program is
``gr_fold``'s line transport (``gr_fold/kernels/transport_phase.h``: the per-call barrier, the fused write + atomic
increment over the fabric, the data semaphore) with the packed row mode (ROW_MODE 3): every device writes its OWN
shard's tiles in full, and the valid rows of each remote shard cross the fabric as ONE packet per direction per link
into the peers' transport-core L1 and are unpacked into the gathered tensor's tile pages at the byte offsets a
full-tile write uses.  Rows >= ``rows`` of the remote shards are not written; every consumer of the gathered hidden is
row independent (the projections' matmuls take it as in0, the routed untilize and slice read rows < ``rows``).
Tolerance class BITWISE.  ``QWEN38_ROW_GATHER=0`` keeps ``ttnn.all_gather`` everywhere; a gather whose packed rows do
not fit one fabric packet payload (``ttnn.get_tt_fabric_max_payload_size_bytes``) or whose shapes are outside the
contract takes ``ttnn.all_gather`` too (``admits``).
"""

from __future__ import annotations

import os

import ttnn

from ...contracts import TP_SIZE
from .. import gr_fold
from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "row_gather"
ENV = "QWEN38_ROW_GATHER"
KERNEL = fp.kernel_source(NAME, "transport.cpp")
PHASE = "rowgather"
SCRATCH_CB, PACK_CB = 0, 1
LINKS = 2


def _mode(environ=None) -> str:
    """QWEN38_ROW_GATHER: unset / 0 = off (the stock all_gather), 1 = the barrier-free ``row_gather2`` program, v1 = this
    module's earlier barrier + unpack program (kept for the A/B)."""

    return (os.environ if environ is None else environ).get(ENV, "1")  # default row_gather2 (x38 A/B -0.35 ms)


def enabled_by_env(environ=None) -> bool:
    return _mode(environ) in ("1", "v1")


def prepare(mesh, tiles: int) -> None:
    """Create the barrier-free program's persistent landing buffer and semaphores (before any trace capture)."""

    if _mode() == "1":
        from .. import row_gather2

        row_gather2.prepare(mesh, tiles)


def _tiles_wide(x) -> int:
    return int(x.padded_shape[3]) // fp.TILE


def _spec(x, out, links: int):
    """The transport spec of one gather and its (mode, slot bytes)."""

    tiles = _tiles_wide(x)
    transports = gr_fold.TRANSPORT["partials"][:links]
    rows = int(x.shape[2])
    spec = gr_fold.Transport(
        out,
        x,
        SCRATCH_CB,
        tiles,
        gr_fold.SOURCE_TENSOR,
        PHASE,
        transports,
        page_strides=(1, tiles),
        rows=rows,
        pack_cb=PACK_CB,
        kernel=KERNEL,
    )
    mode, slot = gr_fold._row_mode(spec, links, 3) if rows < fp.TILE else (0, 0)
    return spec, transports, rows, mode, slot


def admits(hidden, *, cluster_axis=1, memory_config=None) -> bool:
    """The contract: ``[1, 1, rows, W]`` bf16 TILE on the TP line, rows in 1..16 whose packed rows fit one packet
    per link (a full tile row takes the all-gather: nothing to trim)."""

    if not enabled_by_env():
        return False
    if _mode() == "1":
        from .. import row_gather2

        return row_gather2.admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config)
    try:
        shape = tuple(int(v) for v in hidden.shape)
        padded = tuple(int(v) for v in hidden.padded_shape)
        if (
            cluster_axis != gr_fold.TP_AXIS
            or len(shape) != 4
            or shape[:2] != (1, 1)
            or not 1 <= shape[2] <= 16
            or padded[2] != fp.TILE
            or padded[3] % fp.TILE
            or hidden.dtype != ttnn.bfloat16
            or hidden.layout != ttnn.TILE_LAYOUT
            or tuple(int(v) for v in hidden.device().shape) != (1, TP_SIZE)
        ):
            return False
        links = min(LINKS, gr_fold.line(hidden.device()).links)
        tiles = _tiles_wide(hidden)
        if links > tiles:
            return False
        tile_bytes = fp.TILE_BYTES[hidden.dtype]
        face, row = tile_bytes // 4, tile_bytes // 64
        packed = 2 * shape[2] * row
        return -(-tiles // links) * packed <= int(ttnn.get_tt_fabric_max_payload_size_bytes())
    except Exception:
        return False


def composed(hidden, *, cluster_axis=1, memory_config=None):
    return ttnn.all_gather(hidden, dim=3, cluster_axis=cluster_axis, memory_config=memory_config)


def row_gather(hidden, *, cluster_axis=1, memory_config=None):
    shape = tuple(int(v) for v in hidden.shape)
    if not admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config):
        raise ValueError(f"row_gather does not admit {shape} {hidden.dtype} {hidden.layout}")
    if _mode() == "1":
        from .. import row_gather2

        return row_gather2.row_gather(hidden, cluster_axis=cluster_axis, memory_config=memory_config)
    mesh = hidden.device()
    links = min(LINKS, gr_fold.line(mesh).links)
    out = fp.allocate(
        (1, 1, shape[2], shape[3] * TP_SIZE),
        hidden.dtype,
        ttnn.TILE_LAYOUT,
        mesh,
        memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )
    spec, transports, rows, mode, slot = _spec(hidden, out, links)
    if mode != 3:
        raise ValueError(f"row_gather needs the packed mode for {shape}")
    tiles = _tiles_wide(hidden)
    tile_bytes = fp.TILE_BYTES[hidden.dtype]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports])
    spec = gr_fold.Transport(
        out,
        hidden,
        SCRATCH_CB,
        tiles,
        gr_fold.SOURCE_TENSOR,
        PHASE,
        transports,
        page_strides=(1, tiles),
        rows=rows,
        pack_cb=PACK_CB,
        kernel=KERNEL,
        extra_args=(rows, mode, PACK_CB, slot),
    )
    cbs = [
        fp.cb_descriptor(SCRATCH_CB, hidden.dtype, tile_bytes, tiles, cores),
        fp.cb_descriptor(PACK_CB, hidden.dtype, (TP_SIZE + 2) * slot, 1, cores),
    ]
    semaphores = [
        fp.semaphore_descriptor(sem, cores) for sem in (gr_fold.SEM_GO, gr_fold.SEM_SCRATCH, gr_fold.SEM_DONE)
    ]
    meta = fp.program_meta(
        NAME,
        "row_gather",
        rows,
        reads=(hidden,),
        dram_bytes=TP_SIZE * tiles * tile_bytes,
        l1_bytes=tiles * tile_bytes + (TP_SIZE + 2) * slot,
        cores=links,
        outputs=((out, None),),
    )
    fp.run_program(
        [hidden, out],
        gr_fold.transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, links=links),
        meta=meta,
    )
    return fp.stamp_topology(out, hidden)


def row_gather_or_all_gather(hidden, *, cluster_axis=1, memory_config=None):
    """The call sites' entry: the packed program within its contract, ``ttnn.all_gather`` otherwise."""

    if admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config):
        return row_gather(hidden, cluster_axis=cluster_axis, memory_config=memory_config)
    return composed(hidden, cluster_axis=cluster_axis, memory_config=memory_config)


register(
    FusedKernel(
        name=NAME,
        replaces="ttnn.all_gather(dim=3) of the per-block decode hidden (gdn / moe / qsa _all_gather_hidden), one "
        "program with only the valid rows crossing the fabric",
        tolerance=BITWISE,
        fused=row_gather,
        composed=composed,
        gate=None,
        admits=admits,
    )
)
