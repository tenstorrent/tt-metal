# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``qsa_block``: the QSA decode layer's glue around its kept kernels as a handful of ``generic_op`` programs.

The chain (``Qwen38TTNNQSA.forward_decode_generic``) is 78 programs per layer; the collectives, the six DRAM-sharded
linears, the indexer, the top-k and sparse_sdpa stay, and the norms, RoPE, one-hot cache/ring updates, layout moves and
integer selection glue become six programs (work/FUSION-WAVE2-SCOPING-20260914.md section 2).  Two chain kernels
decide whether those programs are bitwise: ``ttnn.rms_norm`` (layernorm.cpp's RMSNORM path in an fp32 dest) and
``rotary_embedding_hf`` (FPU multiplies and an add in the op's 16-bit dest).  ``rms_norm_rows`` and ``rope64`` below
are their call-for-call mirrors on the foundation's stream reader/writer; the dev tool fused_qsa_llk_pins runs them
against the ops on a chip.

The served decode path computes the five K = 2560 projections as one linear (ttnn/qsa.py ``_project_merged``): the
tail programs take the first tile of each projection's window in the tensor they are handed (``index_q_first``,
``qg_first``, ...), 0 for a separate projection shard, the window's tile for the merged shard passed in every slot.
"""

from __future__ import annotations

import struct

import ttnn

from .. import program as fp
from ..gr_read import CONST_COL_SCALAR, CONST_SCALER, NOC_PROBE, _reader, _stream, _writer
from ..registry import BITWISE, FusedKernel, register

NAME = "qsa_block"
RMS_NORM = fp.kernel_source(NAME, "rms_norm_compute.cpp")
ROPE = fp.kernel_source(NAME, "rope_compute.cpp")
INDEX_TAIL = {
    role: fp.kernel_source(NAME, f"index_tail_{role}.cpp")
    for role in ("reader_norm", "compute_norm", "writer_norm", "reader_rope", "compute_rope", "writer_rope")
}
POST_ATTENTION_READER = fp.kernel_source(NAME, "post_attention_reader.cpp")
POST_ATTENTION_COMPUTE = fp.kernel_source(NAME, "post_attention_compute.cpp")
WIDEN_COMPUTE = fp.kernel_source(NAME, "widen_compute.cpp")
SELECTION_ROW = fp.kernel_source(NAME, "selection_row.cpp")
LANE_SCORE_ROWS = fp.kernel_source(NAME, "lane_score_rows.cpp")
LANE_COLUMNS = 8  # lane u's core sits in column u % 8 of its core row: up to 32 lanes on 4 core rows
SCORE_MERGE = {role: fp.kernel_source(NAME, f"score_merge_{role}.cpp") for role in ("reader", "compute", "writer")}
MAIN_TAIL = {
    role: fp.kernel_source(NAME, f"main_tail_{role}.cpp")
    for role in (
        "reader_norm",
        "compute_norm",
        "writer_norm",
        "reader_rope",
        "compute_rope",
        "writer_rope",
        "reader_staging",
        "compute_staging",
        "writer_staging",
    )
}
BF16, FP32 = ttnn.bfloat16, ttnn.float32
TILE_BF16, TILE_FP32 = fp.TILE_BYTES[BF16], fp.TILE_BYTES[FP32]
ROPE_DIM = 64
ROPE_TILES = ROPE_DIM // fp.TILE
EPS = 1e-6
ROPE_DEST_MODES = ("dest16", "dest32_pack", "dest32_rne")
WRAP_MINUS_ONE = 0xFFFFFFFF  # a stream stride of -1 in the reader's uint32 page arithmetic


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _row_tiles(tensor, width: int, label: str) -> int:
    shape, padded = tuple(tensor.shape), tuple(tensor.padded_shape)
    if len(shape) != 4 or shape[0] != 1 or padded[2] != fp.TILE or shape[3] != width or padded[3] != width:
        raise ValueError(f"{label} must be [1, n, <=32, {width}] padded to one row tile, got {shape} padded {padded}")
    if tensor.layout != ttnn.TILE_LAYOUT or tensor.dtype != BF16:
        raise ValueError(f"{label} must be TILE bfloat16")
    return shape[1]


def rms_norm_rows(x, weight, *, eps: float = EPS):
    """``ttnn.rms_norm(x, epsilon=eps, weight=weight)`` on ``x`` [1, n, rows, W] bf16 TILE (every row of every tile
    normalized over W) with the bf16 weight row [1, 1, 1, W]; one core per tile row.  The mirror of the chain's
    fp32-dest LayerNorm program on the same CB formats."""

    width = x.shape[-1]
    n = _row_tiles(x, width, "rms_norm input")
    if tuple(weight.shape) != (1, 1, 1, width) or weight.dtype != BF16 or weight.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"rms_norm weight must be TILE bfloat16 [1, 1, 1, {width}], got {tuple(weight.shape)}")
    wt = width // fp.TILE
    mesh = x.device()
    out = fp.allocate(tuple(x.shape), BF16, ttnn.TILE_LAYOUT, mesh)
    work = fp.split_work(n, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, wt, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, 1, cores),
        fp.cb_descriptor(2, BF16, TILE_BF16, 1, cores),
        fp.cb_descriptor(3, BF16, TILE_BF16, wt, cores),
        fp.cb_descriptor(4, FP32, TILE_FP32, wt, cores),
        fp.cb_descriptor(5, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(6, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(7, FP32, TILE_FP32, 8, cores),
        fp.cb_descriptor(16, BF16, TILE_BF16, wt, cores),
    ]
    reader = _reader(
        cores,
        [(x, 0), (weight, 3)],
        [(CONST_SCALER, 1), (CONST_COL_SCALAR, 2)],
        [
            (
                w.core,
                (
                    [_stream(x, w.count, wt, w.start * wt, 1, wt, wt), _stream(weight, 1, wt, 0, 1, 0, wt)],
                    [_bits(1.0), _bits(eps)],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(RMS_NORM, cores, [wt, width], [(w.core, [w.count]) for w in work], fp32_dest=True)
    writer = _writer(cores, [(out, 16)], [(w.core, [(w.count * wt, w.start * wt, 1, wt)]) for w in work])
    meta = fp.program_meta(  # the tiles in, the weight once per core, the tiles out; four passes per element
        NAME,
        "rms_norm_rows",
        1,
        reads=(x,),
        writes=(out,),
        dram_bytes=len(work) * fp.tensor_bytes(weight),
        flops=4 * n * fp.TILE * width,
        cores=len(work),
    )
    return fp.run_program([x, weight, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


def rope64(x, cos, sin, *, dest: str = "dest16"):
    """``rotary_embedding_hf(x, cos, sin, is_decode_mode=False)`` on ``x`` [1, n, rows, 64] bf16 TILE with the
    cos/sin row tiles [1, 1, rows, 64]; one core per tile row.  ``dest16`` = the op's own compute config (a 16-bit
    dest), ``dest32_pack`` = the same LLK calls under an fp32 dest with the packer narrowing to bf16, ``dest32_rne`` =
    fp32 dest with an SFPU RNE narrowing before every pack."""

    if dest not in ROPE_DEST_MODES:
        raise ValueError(f"dest must be one of {ROPE_DEST_MODES}, got {dest!r}")
    n = _row_tiles(x, ROPE_DIM, "rope input")
    for label, table in (("cos", cos), ("sin", sin)):
        if _row_tiles(table, ROPE_DIM, f"rope {label}") != 1:
            raise ValueError(f"rope {label} must hold one row tile")
    mesh = x.device()
    out = fp.allocate(tuple(x.shape), BF16, ttnn.TILE_LAYOUT, mesh)
    work = fp.split_work(n, mesh)
    if any(w.count != 1 for w in work):
        raise ValueError(f"rope64 takes at most one row tile per core, got {n} tiles on {len(work)} cores")
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(i, BF16, TILE_BF16, pages, cores)
        for i, pages in ((0, 2), (1, 2), (2, 2), (3, 2), (4, 1), (5, 1), (6, 1), (7, 1), (16, 2))
    ]
    reader = _reader(
        cores,
        [(x, 0), (x, 1), (cos, 2), (sin, 3)],
        [(CONST_COL_SCALAR, 4)],
        [
            (
                w.core,
                (
                    [
                        _stream(x, 1, ROPE_TILES, w.start * ROPE_TILES, 1, 0, ROPE_TILES),
                        _stream(x, 1, ROPE_TILES, w.start * ROPE_TILES + 1, WRAP_MINUS_ONE, 0, ROPE_TILES),
                        _stream(cos, 1, ROPE_TILES, 0, 1, 0, ROPE_TILES),
                        _stream(sin, 1, ROPE_TILES, 0, 1, 0, ROPE_TILES),
                    ],
                    [_bits(-1.0)],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(
        ROPE,
        cores,
        [],
        [(w.core, [1]) for w in work],
        defines=[("ROUND_RNE", "1")] if dest == "dest32_rne" else (),
        fp32_dest=dest != "dest16",
    )
    writer = _writer(cores, [(out, 16)], [(w.core, [(ROPE_TILES, w.start * ROPE_TILES, 1, ROPE_TILES)]) for w in work])
    meta = fp.program_meta(  # the tiles twice (the row and its rotated half), cos / sin once per core, the tiles out
        NAME,
        "rope64",
        1,
        reads=(x, x),
        writes=(out,),
        dram_bytes=len(work) * (fp.tensor_bytes(cos) + fp.tensor_bytes(sin)),
        flops=6 * n * fp.TILE * ROPE_DIM,
        cores=len(work),
    )
    return fp.run_program([x, cos, sin, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


# ----------------------------------------------------------------------------------------------- program A: index tail
# One program for the index projection's tail and the compressed-index write (chain programs 510-527, 18 per layer):
# index_q rms_norm + partial RoPE -> the indexer's query tile; the raw-key ring's one-hot slot update, the 0.25-scaled
# row sum, index_k rms_norm + block-start RoPE -> one row of the compressed index cache.  One core pair per lane: the
# norm core (fp32 dest: rms_norm, ring, sum) hands the RoPE tiles to its rope core (the op's 16-bit dest) over the
# NoC; the first pair also takes the index query tile.  Rows are lanes: row r of the query/raw-key tiles belongs to
# lane r, whose ring is tile row r of ``ring`` and whose compressed cache is batch r of ``compressed_cache``;
# ``positions`` holds P_r.  CB indices: index_tail_cbs.h.

INDEX_HEAD_DIM = 128
INDEX_TILES = INDEX_HEAD_DIM // fp.TILE
COMPRESS_RATIO = 4
_INDEX_TAIL_CBS = (
    (0, BF16, INDEX_TILES),
    (1, BF16, 1),
    (2, BF16, 1),
    (3, BF16, INDEX_TILES),
    (4, FP32, INDEX_TILES),
    (5, FP32, 1),
    (6, FP32, 1),
    (7, FP32, 8),
    (8, BF16, INDEX_TILES),
    (9, BF16, INDEX_TILES),
    (10, BF16, INDEX_TILES),
    (11, BF16, 1),
    (14, BF16, 1),
    (15, BF16, INDEX_TILES),
    (16, BF16, INDEX_TILES),
    (17, BF16, 2),
    (18, BF16, 2),
    (19, BF16, 2),
    (20, BF16, 2),
    (21, BF16, 1),
    (22, BF16, 1),
    (23, BF16, 1),
    (24, BF16, 1),
    (25, BF16, 2),
    (26, BF16, 2),
    (27, BF16, 2),
    (28, BF16, 2),
    (29, BF16, 2),
    (30, BF16, 2),
    (31, BF16, INDEX_TILES),
)
CB_RINGC, CB_RINGW = 12, 13
CB_POS_R, CB_POS_W, CB_POSCR, CB_POSCW, POS_PAGE = 32, 33, 34, 35, 128
POSC_BYTES = 128 + fp.ROWS_MAX * TILE_BF16  # read_positions scratch: the block-start row + one hit tile per lane
_NOC_COORDS: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}


def noc_coords(mesh, cores) -> dict[tuple[int, int], tuple[int, int]]:
    """Logical core -> NoC-0 (x, y), measured once per mesh by the probe kernel (every device of the mesh must
    agree; Python has no logical-to-NoC binding)."""

    key = id(mesh)
    wanted = {(c.x, c.y) for c in cores}
    if key in _NOC_COORDS and wanted <= set(_NOC_COORDS[key]):
        return _NOC_COORDS[key]
    grid = mesh.compute_with_storage_grid_size()
    probed = [ttnn.CoreCoord(x, y) for x in range(grid.x) for y in range(grid.y)]
    out = fp.allocate((1, 1, fp.TILE, fp.TILE * len(probed)), ttnn.uint32, ttnn.TILE_LAYOUT, mesh)
    core_set = _rect(0, 0, grid.x - 1, grid.y - 1)
    probe = fp.writer_kernel(
        NOC_PROBE, core_set, fp.accessor_args(out), [(c, [out.buffer_address(), i]) for i, c in enumerate(probed)]
    )
    fp.run_program(
        [out, out],
        fp.program_descriptor([probe], cbs=[fp.cb_descriptor(0, ttnn.uint32, fp.TILE_BYTES[ttnn.uint32], 1, core_set)]),
        meta=fp.program_meta(NAME, "noc_probe", 1, writes=(out,), cores=len(probed)),  # once per mesh
    )
    per_device = [
        ttnn.to_torch(local).reshape(fp.TILE, len(probed), fp.TILE)[0] for local in ttnn.get_device_tensors(out)
    ]
    ttnn.deallocate(out)
    for words in per_device[1:]:
        if not (words[:, :2] == per_device[0][:, :2]).all():
            raise RuntimeError("the devices of the mesh disagree on their logical-to-NoC map")
    words = per_device[0]
    _NOC_COORDS[key] = {(c.x, c.y): (int(words[i, 0]), int(words[i, 1])) for i, c in enumerate(probed)}
    return _NOC_COORDS[key]


def _lane_cores(rows: int, first_row: int, row_stride: int = 1) -> list:
    """Lane u's core: column ``u % 8``, core row ``first_row + row_stride * (u // 8)`` (one core per lane)."""

    return [ttnn.CoreCoord(u % LANE_COLUMNS, first_row + row_stride * (u // LANE_COLUMNS)) for u in range(rows)]


def _core_rows(cores) -> ttnn.CoreRangeSet:
    """The cores as one CoreRange per core row (the columns of a row are consecutive): one kernel group per row."""

    by_row: dict[int, list[int]] = {}
    for core in cores:
        by_row.setdefault(core.y, []).append(core.x)
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(min(xs), y), ttnn.CoreCoord(max(xs), y)) for y, xs in sorted(by_row.items())]
    )


def _require_grid(mesh, x: int, y: int, label: str) -> None:
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < x or grid.y < y:
        raise ValueError(f"{label} needs a {x} x {y} core grid, the device has {grid.x} x {grid.y}")


def _position_inputs(position, rows: int):
    """The chain's ``kv_block_start`` (uint32 ROW_MAJOR [1, 1, 1, >= rows], lane u's P & ~31) and ``kv_row_hit`` (bf16
    TILE [1, rows, 32, 1], lane u's one-hot column of P % 32); the kernels read P_u = start + hit row from them."""

    block_start, row_hit = position.kv_block_start, position.kv_row_hit
    bshape, hshape = tuple(block_start.shape), tuple(row_hit.shape)
    if (
        block_start.dtype != ttnn.uint32
        or block_start.layout != ttnn.ROW_MAJOR_LAYOUT
        or len(bshape) != 4
        or bshape[:3] != (1, 1, 1)
        or bshape[3] < rows
    ):
        raise ValueError(
            f"kv_block_start must be uint32 ROW_MAJOR [1, 1, 1, >= {rows}], got {block_start.layout} {block_start.dtype} {bshape}"
        )
    if row_hit.dtype != BF16 or row_hit.layout != ttnn.TILE_LAYOUT or hshape != (1, rows, fp.TILE, 1):
        raise ValueError(
            f"kv_row_hit must be bf16 TILE [1, {rows}, 32, 1], got {row_hit.layout} {row_hit.dtype} {hshape}"
        )
    return block_start, row_hit


def _expect(tensor, shape, dtype, layout, label: str) -> None:
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.layout != layout:
        raise ValueError(
            f"{label} must be {layout} {dtype} {tuple(shape)}, got {tensor.layout} {tensor.dtype} {tuple(tensor.shape)}"
        )


def _window(tensor, first_tile: int, width: int, label: str) -> None:
    """``tensor`` [1, 1, rows, W] of whole tiles holds the ``width`` columns from tile ``first_tile``: the separate
    projection shard at 0, or the projection's window of the merged shard."""

    total = fp.tile_width_of(tensor)
    if first_tile < 0 or first_tile * fp.TILE + width > total:
        raise ValueError(f"{label} window of {width} columns from tile {first_tile} is outside [1, 1, rows, {total}]")


def _io(*tensors) -> list:
    """generic_op's io list, inputs first and the output last, each tensor once (the tails take the merged projection
    shard in several projection slots)."""

    kept = []
    for tensor in tensors:
        if not any(tensor is other for other in kept):
            kept.append(tensor)
    return kept


def index_tail(
    index_q_ws,
    raw_key_ws,
    position,
    index_q_norm,
    index_k_norm,
    cos,
    sin,
    block_cos,
    block_sin,
    ring,
    compressed_cache,
    *,
    eps: float = EPS,
    index_q_first: int = 0,
    index_k_first: int = 0,
):
    """The fused index tail on ``rows`` lanes; returns the rotated index query [1, 1, rows, 128] bf16 TILE and
    updates ``ring`` [1, rows, 32, 128] and ``compressed_cache`` [rows, 1, H, 128] in place.  ``position`` carries the
    chain's ``kv_block_start`` and ``kv_row_hit`` (see ``_position_inputs``).  ``index_q_ws`` / ``raw_key_ws`` hold
    the index query / raw key from tile ``index_q_first`` / ``index_k_first``: the separate projection shards at 0,
    or both the merged projection shard with its windows' first tiles."""

    rows = fp.rows_of(index_q_ws)
    if fp.rows_of(raw_key_ws) != rows:
        raise ValueError("index_tail takes the index query and raw key as row tiles of the same rows")
    _window(index_q_ws, index_q_first, INDEX_HEAD_DIM, "index query")
    _window(raw_key_ws, index_k_first, INDEX_HEAD_DIM, "raw key")
    block_start, row_hit = _position_inputs(position, rows)
    for label, weight in (("index_q_norm", index_q_norm), ("index_k_norm", index_k_norm)):
        _expect(weight, (1, 1, 1, INDEX_HEAD_DIM), BF16, ttnn.TILE_LAYOUT, label)
    for label, table in (("cos", cos), ("sin", sin), ("block_cos", block_cos), ("block_sin", block_sin)):
        _expect(table, (1, 1, rows, ROPE_DIM), BF16, ttnn.TILE_LAYOUT, label)
    _expect(ring, (1, rows, fp.TILE, INDEX_HEAD_DIM), BF16, ttnn.TILE_LAYOUT, "raw-key ring")
    cache_shape = tuple(compressed_cache.shape)
    if (
        len(cache_shape) != 4
        or cache_shape[0] != rows
        or cache_shape[1] != 1
        or cache_shape[2] % fp.TILE
        or cache_shape[3] != INDEX_HEAD_DIM
        or compressed_cache.dtype != BF16
        or compressed_cache.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(f"compressed cache must be TILE bf16 [{rows}, 1, 32k, 128], got {cache_shape}")
    lane_tile_rows = cache_shape[2] // fp.TILE
    mesh = index_q_ws.device()
    out = fp.allocate((1, 1, rows, INDEX_HEAD_DIM), BF16, ttnn.TILE_LAYOUT, mesh)
    # one core pair per lane: lane u's norm core on row 2 * (u // 8), its rope core on the row below; the first pair also
    # takes the index query tile (its rows are all the lanes' rows)
    pair_rows = -(-rows // LANE_COLUMNS)
    _require_grid(mesh, min(rows, LANE_COLUMNS), 2 * pair_rows, "index_tail lanes")
    norm_cores, rope_cores = _lane_cores(rows, 0, 2), _lane_cores(rows, 1, 2)
    coords = noc_coords(mesh, [*norm_cores, *rope_cores])
    norm_set, rope_set = _core_rows(norm_cores), _core_rows(rope_cores)
    both = _rect(0, 0, min(rows, LANE_COLUMNS) - 1, 2 * pair_rows - 1)
    cbs = [fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, both) for index, dtype, pages in _INDEX_TAIL_CBS]
    cbs.append(
        ttnn.CBDescriptor(  # the canonical ring: the compute's reduce input and the writer's drain, one allocation
            total_size=INDEX_TILES * TILE_BF16,
            core_ranges=both,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=BF16, page_size=TILE_BF16)
                for cb in (CB_RINGC, CB_RINGW)
            ],
        )
    )
    cbs += [fp.cb_descriptor(cb, ttnn.uint32, POS_PAGE, 1, both) for cb in (CB_POS_R, CB_POS_W)]
    cbs += [fp.cb_descriptor(cb, ttnn.uint32, POSC_BYTES, 1, both) for cb in (CB_POSCR, CB_POSCW)]
    eps_bits = _bits(eps)
    acc = fp.accessor_args
    xy = lambda c: coords[(c.x, c.y)]
    lanes = range(rows)  # pair u: lane_first = u, lane_count = 1, do_query = (u == 0)
    kernels = [
        fp.reader_kernel(
            INDEX_TAIL["reader_norm"],
            norm_set,
            [
                eps_bits,
                *acc(index_q_ws),
                *acc(raw_key_ws),
                *acc(block_start),
                *acc(index_q_norm),
                *acc(index_k_norm),
                *acc(ring),
                *acc(row_hit),
            ],
            [
                (
                    norm_cores[u],
                    [
                        index_q_ws.buffer_address(),
                        raw_key_ws.buffer_address(),
                        block_start.buffer_address(),
                        index_q_norm.buffer_address(),
                        index_k_norm.buffer_address(),
                        ring.buffer_address(),
                        row_hit.buffer_address(),
                        rows,
                        index_q_first,
                        index_k_first,
                        u,
                        1,
                        int(u == 0),
                    ],
                )
                for u in lanes
            ],
        ),
        fp.compute_kernel(
            INDEX_TAIL["compute_norm"], norm_set, [], [(norm_cores[u], [1, int(u == 0)]) for u in lanes], fp32_dest=True
        ),
        fp.writer_kernel(
            INDEX_TAIL["writer_norm"],
            norm_set,
            [*acc(out), *acc(ring), *acc(compressed_cache), *acc(block_start), *acc(row_hit)],
            [
                (
                    norm_cores[u],
                    [
                        out.buffer_address(),
                        ring.buffer_address(),
                        compressed_cache.buffer_address(),
                        block_start.buffer_address(),
                        row_hit.buffer_address(),
                        rows,
                        lane_tile_rows,
                        *xy(rope_cores[u]),
                        u,
                        1,
                        int(u == 0),
                    ],
                )
                for u in lanes
            ],
        ),
        fp.reader_kernel(
            INDEX_TAIL["reader_rope"],
            rope_set,
            [*acc(cos), *acc(sin), *acc(block_cos), *acc(block_sin)],
            [
                (
                    rope_cores[u],
                    [
                        cos.buffer_address(),
                        sin.buffer_address(),
                        block_cos.buffer_address(),
                        block_sin.buffer_address(),
                        rows,
                        *xy(norm_cores[u]),
                        u,
                        1,
                        int(u == 0),
                    ],
                )
                for u in lanes
            ],
        ),
        fp.compute_kernel(
            INDEX_TAIL["compute_rope"],
            rope_set,
            [],
            [(rope_cores[u], [1, int(u == 0)]) for u in lanes],
            fp32_dest=False,
        ),
        fp.writer_kernel(
            INDEX_TAIL["writer_rope"],
            rope_set,
            [*acc(out), *acc(compressed_cache), *acc(block_start), *acc(row_hit)],
            [
                (
                    rope_cores[u],
                    [
                        out.buffer_address(),
                        compressed_cache.buffer_address(),
                        block_start.buffer_address(),
                        row_hit.buffer_address(),
                        rows,
                        lane_tile_rows,
                        u,
                        1,
                        int(u == 0),
                    ],
                )
                for u in lanes
            ],
        ),
    ]
    semaphores = [fp.semaphore_descriptor(i, both) for i in range(3)]
    io = _io(
        index_q_ws,
        raw_key_ws,
        block_start,
        row_hit,
        index_q_norm,
        index_k_norm,
        cos,
        sin,
        block_cos,
        block_sin,
        ring,
        compressed_cache,
        out,
    )
    # per lane: the index query and raw key windows (four tiles each, from the projection shard), the position
    # inputs, the two norm weights and the four RoPE tiles, the ring read and written, one compressed row written,
    # the rotated query out; the two rms_norms, the two partial RoPEs, the ring one-hot update and its 0.25 sum
    meta = fp.program_meta(
        "qsa_index_tail",
        "index_tail",
        rows,
        reads=(block_start, row_hit, index_q_norm, index_k_norm, cos, sin, block_cos, block_sin, ring),
        writes=(ring, out),
        partial=(
            (index_q_ws, INDEX_TILES * TILE_BF16),
            (raw_key_ws, INDEX_TILES * TILE_BF16),
            (compressed_cache, rows * INDEX_HEAD_DIM * 2),
        ),
        flops=rows * (8 * INDEX_HEAD_DIM + 12 * ROPE_DIM + 3 * fp.TILE * INDEX_HEAD_DIM),
        cores=2 * rows,
    )
    return fp.run_program(io, fp.program_descriptor(kernels, cbs=cbs, semaphores=semaphores), meta=meta)


def index_tail_composed(
    index_q_ws,
    raw_key_ws,
    position,
    index_q_norm,
    index_k_norm,
    cos,
    sin,
    block_cos,
    block_sin,
    ring,
    compressed_cache,
    *,
    eps: float = EPS,
):
    """Today's chain for one lane (ttnn/qsa.py ``_index_projection`` after its linears + ``_write_compressed_index_generic``):
    the same tensors, ``position`` carrying the chain's one-hots ``ring_hit`` / ``ring_keep`` [1, 1, 32, 1] bf16 TILE
    and ``block_index_i32`` int32 [1]."""

    from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_prefill

    if fp.rows_of(index_q_ws) != 1:
        raise ValueError("the composed index tail is the one-lane chain")
    mesh = index_q_ws.device()
    config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    dram = ttnn.DRAM_MEMORY_CONFIG
    index_q = ttnn.to_memory_config(index_q_ws, dram)
    raw_key = ttnn.to_memory_config(raw_key_ws, dram)
    normalized = ttnn.rms_norm(
        index_q, epsilon=eps, weight=index_q_norm, memory_config=dram, compute_kernel_config=config
    )
    rotated = apply_partial_rope_prefill(normalized, cos, sin, 1, ROPE_DIM)
    rotated = ttnn.to_memory_config(rotated, dram)
    kept = ttnn.multiply(ring, position.ring_keep, memory_config=dram)
    placed = ttnn.multiply(raw_key, position.ring_hit, memory_config=dram)
    ttnn.add(kept, placed, output_tensor=ring, fast_and_approximate_mode=False)
    pooled = ttnn.sum(
        ring, dim=2, keepdim=True, memory_config=dram, compute_kernel_config=config, scalar=1.0 / COMPRESS_RATIO
    )
    normalized_k = ttnn.rms_norm(
        pooled, epsilon=eps, weight=index_k_norm, memory_config=dram, compute_kernel_config=config
    )
    rotated_k = apply_partial_rope_prefill(normalized_k, block_cos, block_sin, 1, ROPE_DIM)
    row_config = ttnn.create_sharded_memory_config(
        (fp.TILE, INDEX_HEAD_DIM),
        ttnn.CoreGrid(y=1, x=1),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    rotated_sharded = ttnn.to_memory_config(rotated_k, row_config)
    ttnn.experimental.paged_update_cache(compressed_cache, rotated_sharded, update_idxs_tensor=position.block_index_i32)
    for t in (index_q, raw_key, normalized, kept, placed, pooled, normalized_k, rotated_k, rotated_sharded):
        ttnn.deallocate(t)
    return rotated


register(
    FusedKernel(
        name="qsa_index_tail",
        replaces="QSA index tail: 2 sharded_to_interleaved, rms_norm, partial RoPE (slice, rotary, slice, concat), ring one-hot "
        "update (2 multiply, add), 0.25 sum, rms_norm, block-start RoPE, interleaved_to_sharded, paged_update_cache (18 programs/layer)",
        tolerance=BITWISE,
        fused=index_tail,
        composed=index_tail_composed,
        gate=None,  # stateful: the dev tool fused_qsa_block_probe compares consecutive decode steps against the chain
    )
)


# ----------------------------------------------------------------------------------------------- program C: main tail
# One program for the main projection's tail, the packed-KV write and the sparse query (chain programs 548-572, 25 per
# layer): per query head rms_norm + partial RoPE (six norm cores, six rope cores), k rms_norm + RoPE (one norm core,
# one rope core), the KV staging one-hot row update, its write-back and its untilized 32-row block into the KV cache
# (one staging core per lane), and the 32-head ROW_MAJOR sparse query [zeros | q_h] with zero rows for the 26 absent
# heads.
# Rows are lanes: row r of the q/k/v tiles is lane r, whose staging is tile row r of ``staging`` and whose KV rows
# start at r * lane_rows in ``kv_cache``; ``positions`` holds P_r.  CB indices: main_tail_cbs.h.

HEAD_DIM = 256
HEAD_TILES = HEAD_DIM // fp.TILE
LOCAL_HEADS = 6
SPARSE_HEADS = 32
QG_WIDTH = 2 * LOCAL_HEADS * HEAD_DIM
KV_WIDTH = 2 * HEAD_DIM
KV_TILES = KV_WIDTH // fp.TILE
_MAIN_TAIL_CBS = (
    (0, BF16, HEAD_TILES),
    (1, BF16, 1),
    (2, BF16, 1),
    (3, BF16, HEAD_TILES),
    (4, FP32, HEAD_TILES),
    (5, FP32, 1),
    (6, FP32, 1),
    (7, FP32, 8),
    (8, BF16, HEAD_TILES),
    (9, BF16, 1),
    (10, BF16, 2),
    (11, BF16, 2),
    (12, BF16, 2),
    (13, BF16, 2),
    (14, BF16, 1),
    (15, BF16, 1),
    (16, BF16, 1),
    (17, BF16, 1),
    (18, BF16, 2),
    (19, BF16, KV_TILES),
    (20, BF16, 1),
    (23, BF16, KV_TILES),
    (24, BF16, KV_TILES),
)
CB_STGC, CB_STGW, CB_MT_POS_R, CB_MT_POS_W, CB_MT_POSCR, CB_MT_POSCW = 21, 22, 25, 26, 27, 28


def _rect(x0: int, y0: int, x1: int, y1: int) -> ttnn.CoreRangeSet:
    """One rectangular core range (one kernel group; a range per core costs ~0.25 us of dispatch each)."""

    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


class KVStage:
    """A replacement for main_tail's staging cores (the verify rows form, ttnn/fused/qsa_rows): the key cores hand the
    rotated and normalized key tiles to ``cores`` (CB_PACK slots 8..15, SEM_KROT / SEM_KNORM as for the staging cores);
    the stage brings its own CBs (``cb_descriptors(all_set)``), kernels (``kernels(ctx)`` with ``ctx`` the builder's
    accessor helper, the core sets and the noc coordinates), io tensors and program-meta parts."""

    def __init__(
        self, *, cores, cb_descriptors, kernels, io, reads=(), writes=(), partial=(), flops=0, name="main_tail"
    ):
        self.cores = list(cores)
        self.cb_descriptors = cb_descriptors
        self.kernels = kernels
        self.io = list(io)
        self.reads, self.writes, self.partial, self.flops, self.name = (
            tuple(reads),
            tuple(writes),
            tuple(partial),
            flops,
            name,
        )


def main_tail(
    qg_ws,
    k_ws,
    v_ws,
    position,
    q_norm,
    k_norm,
    cos,
    sin,
    staging,
    kv_cache,
    *,
    lane_rows: int = 0,
    eps: float = EPS,
    qg_first: int = 0,
    k_first: int = 0,
    v_first: int = 0,
    kv_stage: KVStage | None = None,
):
    """The fused main tail on ``rows`` lanes; returns the sparse query [1, 32, rows, 512] bf16 ROW_MAJOR and updates
    ``staging`` [1, rows, 32, 512] (TILE) and ``kv_cache`` [1, 1, C, 512] (ROW_MAJOR, lane r's block at
    r * lane_rows + (P_r & ~31)) in place.  The gate halves stay in ``qg_ws`` for the post-attention program.
    ``qg_ws`` / ``k_ws`` / ``v_ws`` hold their projections from tile ``qg_first`` / ``k_first`` / ``v_first``: the
    separate projection shards at 0, or the merged projection shard in all three slots with its windows' first
    tiles."""

    rows = fp.rows_of(qg_ws)
    if fp.rows_of(k_ws) != rows or fp.rows_of(v_ws) != rows:
        raise ValueError("main_tail takes qg, k and v as row tiles of the same rows")
    _window(qg_ws, qg_first, QG_WIDTH, "qg")
    _window(k_ws, k_first, HEAD_DIM, "k")
    _window(v_ws, v_first, HEAD_DIM, "v")
    block_start, row_hit = _position_inputs(position, rows) if kv_stage is None else (None, None)
    for label, weight in (("q_norm", q_norm), ("k_norm", k_norm)):
        _expect(weight, (1, 1, 1, HEAD_DIM), BF16, ttnn.TILE_LAYOUT, label)
    for label, table in (("cos", cos), ("sin", sin)):
        _expect(table, (1, 1, rows, ROPE_DIM), BF16, ttnn.TILE_LAYOUT, label)
    if kv_stage is None:
        _expect(staging, (1, rows, fp.TILE, KV_WIDTH), BF16, ttnn.TILE_LAYOUT, "KV staging")
    cache_shape = tuple(kv_cache.shape) if kv_stage is None else (1, 1, 0, KV_WIDTH)
    if kv_stage is None and (
        len(cache_shape) != 4
        or cache_shape[:2] != (1, 1)
        or cache_shape[3] != KV_WIDTH
        or kv_cache.dtype != BF16
        or kv_cache.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"KV cache must be ROW_MAJOR bf16 [1, 1, C, 512], got {kv_cache.layout} {cache_shape}")
    if kv_stage is None and rows > 1 and (lane_rows % fp.TILE or cache_shape[2] < rows * lane_rows):
        raise ValueError(
            f"{rows} lanes need lane_rows a multiple of 32 with {rows} * lane_rows <= {cache_shape[2]}, got {lane_rows}"
        )
    mesh = qg_ws.device()
    query = fp.allocate((1, SPARSE_HEADS, rows, KV_WIDTH), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh)
    q_norm_cores = [ttnn.CoreCoord(h, 0) for h in range(LOCAL_HEADS)]
    q_rope_cores = [ttnn.CoreCoord(h, 1) for h in range(LOCAL_HEADS)]
    k_norm_core, k_rope_core = ttnn.CoreCoord(LOCAL_HEADS, 0), ttnn.CoreCoord(LOCAL_HEADS, 1)
    # one staging core per lane on the core rows below the head and key cores; the key cores hand their tiles to each
    staging_cores = _lane_cores(rows, 2) if kv_stage is None else kv_stage.cores
    staging_rows = -(-rows // LANE_COLUMNS) if kv_stage is None else max(c.y for c in staging_cores) - 1
    _require_grid(mesh, LOCAL_HEADS + 2, 2 + staging_rows, "main_tail lanes")
    every = [*q_norm_cores, *q_rope_cores, k_norm_core, k_rope_core, *staging_cores]
    coords = noc_coords(mesh, every)
    xy = lambda c: coords[(c.x, c.y)]
    all_set = _rect(0, 0, LOCAL_HEADS + 1, 1 + staging_rows)  # the bounding rectangle (idle corners run no kernel)
    staging_xy = [len(staging_cores)] + [v for c in staging_cores for v in xy(c)]
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, all_set) for index, dtype, pages in _MAIN_TAIL_CBS
    ]
    if kv_stage is not None:
        cbs += kv_stage.cb_descriptors(all_set)
    cbs.append(
        ttnn.CBDescriptor(  # the canonical staging: the compute's untilize input and the writer's drain, one allocation
            total_size=KV_TILES * TILE_BF16,
            core_ranges=all_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=BF16, page_size=TILE_BF16)
                for cb in (CB_STGC, CB_STGW)
            ],
        )
    )
    cbs += [fp.cb_descriptor(cb, ttnn.uint32, POS_PAGE, 1, all_set) for cb in (CB_MT_POS_R, CB_MT_POS_W)]
    cbs += [fp.cb_descriptor(cb, ttnn.uint32, POSC_BYTES, 1, all_set) for cb in (CB_MT_POSCR, CB_MT_POSCW)]
    eps_bits = _bits(eps)
    acc = fp.accessor_args
    norm_set, rope_set = _rect(0, 0, LOCAL_HEADS, 0), _rect(0, 1, LOCAL_HEADS, 1)
    q_norm_set, q_rope_set = _rect(0, 0, LOCAL_HEADS - 1, 0), _rect(0, 1, LOCAL_HEADS - 1, 1)
    k_norm_set, k_rope_set = _rect(LOCAL_HEADS, 0, LOCAL_HEADS, 0), _rect(LOCAL_HEADS, 1, LOCAL_HEADS, 1)
    staging_set = _core_rows(staging_cores)
    kernels = [
        fp.reader_kernel(
            MAIN_TAIL["reader_norm"],
            q_norm_set,
            [eps_bits, *acc(qg_ws), *acc(q_norm)],
            [
                (c, [qg_ws.buffer_address(), q_norm.buffer_address(), qg_first + 2 * HEAD_TILES * h])
                for h, c in enumerate(q_norm_cores)
            ],
        ),
        fp.reader_kernel(
            MAIN_TAIL["reader_norm"],
            k_norm_set,
            [eps_bits, *acc(k_ws), *acc(k_norm)],
            [(k_norm_core, [k_ws.buffer_address(), k_norm.buffer_address(), k_first])],
        ),
        fp.compute_kernel(
            MAIN_TAIL["compute_norm"], norm_set, [], [(c, []) for c in [*q_norm_cores, k_norm_core]], fp32_dest=True
        ),
        fp.writer_kernel(
            MAIN_TAIL["writer_norm"],
            q_norm_set,
            [0, *acc(query)],
            [(c, [query.buffer_address(), rows, h, *xy(q_rope_cores[h]), 0]) for h, c in enumerate(q_norm_cores)],
        ),
        fp.writer_kernel(
            MAIN_TAIL["writer_norm"],
            k_norm_set,
            [1, *acc(query)],
            [(k_norm_core, [query.buffer_address(), rows, 0, *xy(k_rope_core), *staging_xy])],
        ),
        fp.reader_kernel(
            MAIN_TAIL["reader_rope"],
            rope_set,
            [*acc(cos), *acc(sin)],
            [(c, [cos.buffer_address(), sin.buffer_address()]) for c in [*q_rope_cores, k_rope_core]],
        ),
        fp.compute_kernel(
            MAIN_TAIL["compute_rope"], rope_set, [], [(c, []) for c in [*q_rope_cores, k_rope_core]], fp32_dest=False
        ),
        fp.writer_kernel(
            MAIN_TAIL["writer_rope"],
            q_rope_set,
            [0, *acc(query)],
            [(c, [query.buffer_address(), rows, h, 0]) for h, c in enumerate(q_rope_cores)],
        ),
        fp.writer_kernel(
            MAIN_TAIL["writer_rope"],
            k_rope_set,
            [1, *acc(query)],
            [(k_rope_core, [query.buffer_address(), rows, 0, *staging_xy])],
        ),
    ]

    def staging_kernels():  # the decode stage's three kernels; their tensors are None under a KV stage
        return [
            fp.reader_kernel(
                MAIN_TAIL["reader_staging"],
                staging_set,
                [*acc(staging), *acc(v_ws), *acc(block_start), *acc(row_hit)],
                [
                    (
                        core,
                        [
                            staging.buffer_address(),
                            v_ws.buffer_address(),
                            block_start.buffer_address(),
                            row_hit.buffer_address(),
                            rows,
                            v_first,
                            u,
                            1,
                        ],
                    )
                    for u, core in enumerate(staging_cores)
                ],
            ),
            fp.compute_kernel(
                MAIN_TAIL["compute_staging"], staging_set, [], [(core, [1]) for core in staging_cores], fp32_dest=True
            ),
            fp.writer_kernel(
                MAIN_TAIL["writer_staging"],
                staging_set,
                [*acc(staging), *acc(kv_cache), *acc(block_start), *acc(row_hit)],
                [
                    (
                        core,
                        [
                            staging.buffer_address(),
                            kv_cache.buffer_address(),
                            block_start.buffer_address(),
                            row_hit.buffer_address(),
                            rows,
                            lane_rows,
                            u,
                            1,
                        ],
                    )
                    for u, core in enumerate(staging_cores)
                ],
            ),
        ]

    if kv_stage is None:
        kernels += staging_kernels()
        io = _io(qg_ws, k_ws, v_ws, block_start, row_hit, q_norm, k_norm, cos, sin, staging, kv_cache, query)
    else:
        kernels += kv_stage.kernels({"acc": acc, "staging_set": staging_set, "xy": xy, "rows": rows})
        io = _io(
            qg_ws, k_ws, v_ws, q_norm, k_norm, cos, sin, *kv_stage.io, query
        )  # the query last: run_program returns it
    semaphores = [fp.semaphore_descriptor(i, all_set) for i in range(3)]
    # the qg, k and v windows (from the projection shards), the position inputs, the two norm weights and the RoPE
    # tiles in, the staging read and written, one 32-row KV block per lane written, the sparse query out; the seven
    # rms_norms and partial RoPEs, the staging one-hot update
    windows = (
        (qg_ws, 2 * LOCAL_HEADS * HEAD_TILES * TILE_BF16),
        (k_ws, HEAD_TILES * TILE_BF16),
        (v_ws, HEAD_TILES * TILE_BF16),
    )
    if kv_stage is None:
        meta = fp.program_meta(
            "qsa_main_tail",
            "main_tail",
            rows,
            reads=(block_start, row_hit, q_norm, k_norm, cos, sin, staging),
            writes=(staging, query),
            partial=(*windows, (kv_cache, rows * fp.TILE * KV_WIDTH * 2)),
            flops=rows * ((LOCAL_HEADS + 1) * (4 * HEAD_DIM + 6 * ROPE_DIM) + 3 * fp.TILE * KV_WIDTH),
            cores=2 * (LOCAL_HEADS + 1) + rows,
        )
    else:
        meta = fp.program_meta(  # the verify rows form (ttnn/fused/qsa_rows main_tail_rows), the family's one KV stage
            "qsa_rows",
            "main_tail_rows",
            rows,
            reads=(q_norm, k_norm, cos, sin, *kv_stage.reads),
            writes=(query, *kv_stage.writes),
            partial=(*windows, *kv_stage.partial),
            flops=rows * (LOCAL_HEADS + 1) * (4 * HEAD_DIM + 6 * ROPE_DIM) + kv_stage.flops,
            cores=2 * (LOCAL_HEADS + 1) + len(staging_cores),
        )
    return fp.run_program(io, fp.program_descriptor(kernels, cbs=cbs, semaphores=semaphores), meta=meta)


_ZERO_HALVES: dict[int, object] = {}


def _zero_value_half(mesh):
    key = id(mesh)
    if key not in _ZERO_HALVES:
        import torch

        _ZERO_HALVES[key] = ttnn.from_torch(
            torch.zeros(1, LOCAL_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16),
            dtype=BF16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _ZERO_HALVES[key]


def main_tail_composed(
    qg_ws, k_ws, v_ws, position, q_norm, k_norm, cos, sin, staging, kv_cache, *, lane_rows: int = 0, eps: float = EPS
):
    """Today's chain for one lane (ttnn/qsa.py ``_main_projection`` after its linears, ``_write_packed_kv_generic`` and
    the query build of ``_sparse_value_attention``); ``position`` carries the chain's ``kv_row_hit`` / ``kv_row_keep``
    [1, 1, 32, 1] bf16 TILE, ``kv_block_start`` and ``slot_zero`` uint32 ROW_MAJOR [1, 1, 1, 1].  Returns (sparse
    query [1, 32, 1, 512] ROW_MAJOR, gate [1, 6, 1, 256] TILE)."""

    import torch

    from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_prefill

    if fp.rows_of(qg_ws) != 1:
        raise ValueError("the composed main tail is the one-lane chain")
    mesh = qg_ws.device()
    config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    dram = ttnn.DRAM_MEMORY_CONFIG
    qg = ttnn.to_memory_config(qg_ws, dram)
    k = ttnn.to_memory_config(k_ws, dram)
    v = ttnn.to_memory_config(v_ws, dram)
    qg_heads = ttnn.reshape(qg, (1, LOCAL_HEADS, 1, 2 * HEAD_DIM))
    q = ttnn.slice(qg_heads, (0, 0, 0, 0), (1, LOCAL_HEADS, 1, HEAD_DIM), memory_config=dram)
    gate = ttnn.slice(qg_heads, (0, 0, 0, HEAD_DIM), (1, LOCAL_HEADS, 1, 2 * HEAD_DIM), memory_config=dram)
    q_normed = ttnn.rms_norm(q, epsilon=eps, weight=q_norm, memory_config=dram, compute_kernel_config=config)
    k_normed = ttnn.rms_norm(k, epsilon=eps, weight=k_norm, memory_config=dram, compute_kernel_config=config)
    q_rotated = apply_partial_rope_prefill(q_normed, cos, sin, LOCAL_HEADS, ROPE_DIM)
    k_rotated = apply_partial_rope_prefill(k_normed, cos, sin, 1, ROPE_DIM)
    packed_tiled = ttnn.concat([v, k_rotated], dim=3, memory_config=dram)
    kept = ttnn.multiply(staging, position.kv_row_keep, memory_config=dram)
    placed = ttnn.multiply(packed_tiled, position.kv_row_hit, memory_config=dram)
    ttnn.add(kept, placed, output_tensor=staging, fast_and_approximate_mode=False)
    stage_row_major = ttnn.to_layout(staging, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        kv_cache, stage_row_major, position.slot_zero, position.kv_block_start, 0, 1, 0
    )
    zero_half = _zero_value_half(
        mesh
    )  # the module's persistent zero V half (a host upload inside a trace capture is refused)
    sparse_query_tiled = ttnn.concat([zero_half, q_rotated], dim=3, memory_config=dram)
    sparse_query_row_major = ttnn.to_layout(sparse_query_tiled, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    sparse_query = ttnn.pad(
        sparse_query_row_major, [(0, 0), (0, SPARSE_HEADS - LOCAL_HEADS), (0, 0), (0, 0)], 0.0, memory_config=dram
    )
    for t in (
        qg,
        k,
        v,
        qg_heads,
        q,
        q_normed,
        k_normed,
        q_rotated,
        k_rotated,
        packed_tiled,
        kept,
        placed,
        stage_row_major,
        sparse_query_tiled,
        sparse_query_row_major,
    ):
        ttnn.deallocate(t)
    return sparse_query, gate


register(
    FusedKernel(
        name="qsa_main_tail",
        replaces="QSA main tail: 3 sharded_to_interleaved, head reshape, 2 slices, 2 rms_norm, q and k partial RoPE (8 programs), "
        "concat, staging one-hot update (2 multiply, add), untilize, update_padded_kv_cache, query concat, untilize, pad (25 programs/layer)",
        tolerance=BITWISE,
        fused=main_tail,
        composed=main_tail_composed,
        gate=None,
    )
)


# -------------------------------------------------------------------------------- programs D, E, B2: post-attention, widen, selection
# D replaces chain programs 574-579 (slice of the 6 local heads, tilize with zero padding, sigmoid, multiply, head
# fold, interleaved_to_sharded into the out-projection's 16-core activation shard): one core per head, the gate read
# straight from the qg L1 shard.  E replaces 581-582 (sharded_to_interleaved + typecast to fp32 before the
# reduce_scatter): 16 cores, 5 tiles each.  B2 replaces 537-544 (shift, repeat_interleave, offsets add, sentinel
# concat, keep-mask and, fill or): 8 cores of 256 columns, integer work on the data-movement RISC.

OUT_WIDTH = LOCAL_HEADS * HEAD_DIM  # 1536
PARTIAL_WIDTH = 2560
PARTIAL_TILES = PARTIAL_WIDTH // fp.TILE  # 80
WIDEN_CORES = 16
SELECTION_WIDTH, EXPANDED_WIDTH, BLOCK_IDS = 2080, 2048, 512
SELECTION_SLICES = 8
SELECTION_ROW_GROUPS = 8  # the lanes' rows spread over up to 8 core rows (row r on core row r % 8)


def post_attention(attention, qg_ws, *, memory_config=None, qg_first: int = 0):
    """sigmoid(gate) x attention for the 6 local heads -> the head-major [1, 1, rows, 1536] bf16 TILE row in
    ``memory_config`` (the out-projection's activation shard in the model).  ``attention`` is sparse_sdpa's
    ROW_MAJOR [1, 32, rows, 256]; the gates are the second 256 columns of each head's 512 in ``qg_ws`` from tile
    ``qg_first`` (0 for the separate qg shard, the qg window's first tile for the merged projection shard)."""

    rows = fp.rows_of(qg_ws)
    _window(qg_ws, qg_first, QG_WIDTH, "qg")
    if (
        tuple(attention.shape) != (1, SPARSE_HEADS, rows, HEAD_DIM)
        or attention.dtype != BF16
        or attention.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(
            f"attention must be ROW_MAJOR bf16 [1, 32, {rows}, 256], got {attention.layout} {attention.dtype} {tuple(attention.shape)}"
        )
    mesh = qg_ws.device()
    out = fp.allocate((1, 1, rows, OUT_WIDTH), BF16, ttnn.TILE_LAYOUT, mesh, memory_config or ttnn.DRAM_MEMORY_CONFIG)
    cores = _rect(0, 0, LOCAL_HEADS - 1, 0)
    head_cores = [ttnn.CoreCoord(h, 0) for h in range(LOCAL_HEADS)]
    cbs = [fp.cb_descriptor(cb, BF16, TILE_BF16, HEAD_TILES, cores) for cb in (0, 1, 2, 16)] + [
        fp.cb_descriptor(3, BF16, TILE_BF16, 1, cores)
    ]
    reader = fp.reader_kernel(
        POST_ATTENTION_READER,
        cores,
        [*fp.accessor_args(attention), *fp.accessor_args(qg_ws)],
        [
            (c, [attention.buffer_address(), qg_ws.buffer_address(), rows, h, qg_first])
            for h, c in enumerate(head_cores)
        ],
    )
    compute = fp.compute_kernel(POST_ATTENTION_COMPUTE, cores, [], [(c, []) for c in head_cores], fp32_dest=True)
    writer = _writer(
        cores, [(out, 16)], [(c, [(HEAD_TILES, h * HEAD_TILES, 1, HEAD_TILES)]) for h, c in enumerate(head_cores)]
    )
    meta = fp.program_meta(  # the six local heads' attention rows and gate tiles in, the head-major row out
        "qsa_post_attention",
        "post_attention",
        rows,
        writes=(out,),
        partial=((attention, LOCAL_HEADS * rows * HEAD_DIM * 2), (qg_ws, LOCAL_HEADS * HEAD_TILES * TILE_BF16)),
        flops=2 * rows * OUT_WIDTH,
        cores=LOCAL_HEADS,
    )
    return fp.run_program([attention, qg_ws, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


def post_attention_composed(attention, qg_ws, *, memory_config=None):
    """Today's chain for one row (ttnn/qsa.py ``_sparse_value_attention`` after sparse_sdpa, the gate from
    ``_main_projection``'s reshape + slice, then ``_project_output``'s to_memory_config)."""

    if fp.rows_of(qg_ws) != 1:
        raise ValueError("the composed post-attention is the one-row chain")
    dram = ttnn.DRAM_MEMORY_CONFIG
    qg = ttnn.to_memory_config(qg_ws, dram)
    qg_heads = ttnn.reshape(qg, (1, LOCAL_HEADS, 1, 2 * HEAD_DIM))
    gate = ttnn.slice(qg_heads, (0, 0, 0, HEAD_DIM), (1, LOCAL_HEADS, 1, 2 * HEAD_DIM), memory_config=dram)
    local = ttnn.slice(attention, (0, 0, 0, 0), (1, LOCAL_HEADS, 1, HEAD_DIM), memory_config=dram)
    local_tiled = ttnn.to_layout(local, ttnn.TILE_LAYOUT, memory_config=dram)
    activated_gate = ttnn.sigmoid(gate, memory_config=dram)
    gated = ttnn.mul(local_tiled, activated_gate, memory_config=dram)
    flat = ttnn.reshape(gated, (1, 1, 1, OUT_WIDTH))
    out = ttnn.to_memory_config(flat, memory_config or dram)
    for t in (qg, qg_heads, gate, local, local_tiled, activated_gate, gated):
        ttnn.deallocate(t)
    if flat.buffer_address() != out.buffer_address():
        ttnn.deallocate(flat)
    return out


def widen_partial(out_ws):
    """The out-projection's bf16 row tile (the matmul's 16-core L1 shard) as fp32 [1, 1, rows, 2560] TILE in DRAM, the
    reduce_scatter's input (the chain's sharded_to_interleaved + typecast)."""

    rows = fp.rows_of(out_ws)
    if fp.tile_width_of(out_ws) != PARTIAL_WIDTH:
        raise ValueError(f"widen_partial takes [1, 1, rows, {PARTIAL_WIDTH}]")
    mesh = out_ws.device()
    out = fp.allocate((1, 1, rows, PARTIAL_WIDTH), FP32, ttnn.TILE_LAYOUT, mesh)
    per_core = PARTIAL_TILES // WIDEN_CORES
    cores = _rect(0, 0, WIDEN_CORES // 2 - 1, 1)
    core_list = [ttnn.CoreCoord(x, y) for y in range(2) for x in range(WIDEN_CORES // 2)]
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, per_core, cores),
        fp.cb_descriptor(16, FP32, TILE_FP32, per_core, cores),
    ]
    reader = _reader(
        cores,
        [(out_ws, 0)],
        [],
        [(c, ([_stream(out_ws, 1, per_core, i * per_core, 1, 0, per_core)], [])) for i, c in enumerate(core_list)],
    )
    compute = fp.compute_kernel(WIDEN_COMPUTE, cores, [], [(c, [per_core]) for c in core_list], fp32_dest=True)
    writer = _writer(
        cores, [(out, 16)], [(c, [(per_core, i * per_core, 1, per_core)]) for i, c in enumerate(core_list)]
    )
    meta = fp.program_meta(  # the bf16 shard in (L1), the fp32 row out; the typecast per element
        "qsa_widen_partial",
        "widen_partial",
        rows,
        reads=(out_ws,),
        writes=(out,),
        flops=rows * PARTIAL_WIDTH,
        cores=WIDEN_CORES,
    )
    return fp.run_program([out_ws, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


def widen_partial_composed(out_ws):
    local_partial = ttnn.to_memory_config(out_ws, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.typecast(local_partial, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(local_partial)
    return out


def _rows_at_least(tensor, rows: int, width: int, label: str) -> None:
    """A uint32 ROW_MAJOR [1, 1, >= rows, width] row tensor whose first ``rows`` rows the kernel reads (the 1-row
    inputs have one row; the lanes pass their 32-row inputs and read rows 0 .. B-1)."""

    shape = tuple(tensor.shape)
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or shape[2] < rows
        or shape[3] != width
        or tensor.dtype != ttnn.uint32
        or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"{label} must be uint32 ROW_MAJOR [1, 1, >= {rows}, {width}], got {shape}")


def selection_row(block_ids, sentinel_pad, block_offsets, row_keep_bits, row_fill):
    """The sparse-attention row of token ids [1, 1, rows, 2080] uint32 ROW_MAJOR from the top-k block ids
    [1, 1, rows, 512] and the position-derived keep / fill rows (rows 0 .. rows-1 of [1, 1, >= rows, 2080] tensors)
    with the module's constant rows ``sentinel_pad`` [1, 1, 1, 32] and ``block_offsets``: one row [1, 1, 1, 2048]
    shared by every row (the decode step), or per-row offsets [1, 1, >= rows, 2048] (the lanes: row u = the offsets
    + lane u's KV region start, the chain's ``block_offsets_lanes``)."""

    shape = tuple(block_ids.shape)
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or shape[3] != BLOCK_IDS
        or block_ids.dtype != ttnn.uint32
        or block_ids.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"block ids must be uint32 ROW_MAJOR [1, 1, rows, {BLOCK_IDS}], got {shape}")
    rows = shape[2]
    for label, tensor, expected in (("sentinel_pad", sentinel_pad, (1, 1, 1, SELECTION_WIDTH - EXPANDED_WIDTH)),):
        if tuple(tensor.shape) != expected or tensor.dtype != ttnn.uint32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"{label} must be uint32 ROW_MAJOR {expected}, got {tuple(tensor.shape)}")
    _rows_at_least(block_offsets, 1, EXPANDED_WIDTH, "block_offsets")
    offset_rows = tuple(block_offsets.shape)[2]
    if offset_rows != 1 and offset_rows < rows:
        raise ValueError(f"block_offsets must hold one row or >= {rows} rows, got {offset_rows}")
    _rows_at_least(row_keep_bits, rows, SELECTION_WIDTH, "row_keep_bits")
    _rows_at_least(row_fill, rows, SELECTION_WIDTH, "row_fill")
    mesh = block_ids.device()
    out = fp.allocate((1, 1, rows, SELECTION_WIDTH), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh)
    row_groups = min(rows, SELECTION_ROW_GROUPS)  # core (slice, g) takes rows g, g + row_groups, ...
    cores = _rect(0, 0, SELECTION_SLICES - 1, row_groups - 1)
    tensors = (block_ids, block_offsets, sentinel_pad, row_keep_bits, row_fill, out)
    compile_args = [a for t in tensors for a in fp.accessor_args(t)]
    addresses = [t.buffer_address() for t in tensors]
    kernel = fp.reader_kernel(
        SELECTION_ROW,
        cores,
        compile_args,
        [
            (ttnn.CoreCoord(i, g), [*addresses, rows, i, offset_rows, g, row_groups])
            for g in range(row_groups)
            for i in range(SELECTION_SLICES)
        ],
    )
    cbs = [fp.cb_descriptor(0, ttnn.uint32, 8192, 1, cores)]
    meta = fp.program_meta(  # the block ids, offsets and sentinel rows, the rows' keep / fill rows in, the row out
        "qsa_selection_row",
        "selection_row",
        rows,
        reads=(block_ids, block_offsets, sentinel_pad),
        writes=(out,),
        partial=((row_keep_bits, rows * SELECTION_WIDTH * 4), (row_fill, rows * SELECTION_WIDTH * 4)),
        flops=4 * rows * SELECTION_WIDTH,
        cores=SELECTION_SLICES * row_groups,
    )
    return fp.run_program([*tensors[:-1], out], fp.program_descriptor([kernel], cbs=cbs), meta=meta)


def selection_row_composed(block_ids, sentinel_pad, block_offsets, row_keep_bits, row_fill):
    """Today's chain (ttnn/qsa.py ``_materialize_row_generic`` after topk_large_indices) for one row."""

    if tuple(block_ids.shape)[2] != 1:
        raise ValueError("the composed selection row is the one-row chain")
    dram = ttnn.DRAM_MEMORY_CONFIG
    starts = ttnn.bitwise_left_shift(block_ids, 2, memory_config=dram)
    repeated = ttnn.repeat_interleave(starts, repeats=COMPRESS_RATIO, dim=3, memory_config=dram)
    expanded = ttnn.add(repeated, block_offsets, memory_config=dram)
    template = ttnn.concat([expanded, sentinel_pad], dim=3, memory_config=dram)
    kept = ttnn.bitwise_and(template, row_keep_bits, memory_config=dram)
    out = ttnn.bitwise_or(kept, row_fill, memory_config=dram)
    for t in (starts, repeated, expanded, template, kept):
        ttnn.deallocate(t)
    return out


register(
    FusedKernel(
        name="qsa_post_attention",
        replaces="QSA post-attention: head slice, tilize, sigmoid, multiply, head fold, interleaved_to_sharded (6 programs/layer)",
        tolerance=BITWISE,
        fused=post_attention,
        composed=post_attention_composed,
        gate=None,
    )
)
register(
    FusedKernel(
        name="qsa_widen_partial",
        replaces="QSA out-projection sharded_to_interleaved + typecast to fp32 (2 programs/layer)",
        tolerance=BITWISE,
        fused=widen_partial,
        composed=widen_partial_composed,
        gate=None,
    )
)
register(
    FusedKernel(
        name="qsa_selection_row",
        replaces="QSA selection row: shift, repeat_interleave (3 programs), offsets add, sentinel concat, keep and, fill or (8 programs/layer)",
        tolerance=BITWISE,
        fused=selection_row,
        composed=selection_row_composed,
        gate=None,
    )
)


def lane_score_rows(local_scores, lanes: int, cache_rows: int, blocks: int):
    """Row u of the local block scores [1, 1, lanes, blocks] bf16 ROW_MAJOR = lane u's window (row u, columns
    u * cache_rows .. + blocks) of the wide indexer's scores [1, 1, 32, lanes * cache_rows]: the lane chain's B slices and
    concat as one program, one core per lane (row copies at the 64-byte grain)."""

    return lane_score_rows_of(local_scores, tuple(range(lanes)), lanes, cache_rows, blocks)


def lane_score_rows_of(local_scores, lanes_of_rows, lanes: int, cache_rows: int, blocks: int):
    """Row r of the local block scores [1, 1, len(lanes_of_rows), blocks] = lane ``lanes_of_rows[r]``'s window (row r,
    columns lane * cache_rows .. + blocks) of the wide indexer's scores [1, 1, 32, lanes * cache_rows]: the MTP lanes
    verify's lane-major rows (row u*R + j reads lane u's columns), one core per output row."""

    rows = len(lanes_of_rows)
    shape = tuple(local_scores.shape)
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or shape[2] < rows
        or shape[3] != lanes * cache_rows
        or local_scores.dtype != BF16
        or local_scores.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"lane scores must be ROW_MAJOR bf16 [1, 1, >= {rows}, {lanes} * {cache_rows}], got {shape}")
    if not 1 <= rows <= fp.ROWS_MAX or blocks > cache_rows or cache_rows % fp.TILE or blocks % fp.TILE:
        raise ValueError(
            f"lane windows need 1..32 rows and whole-tile cache_rows >= blocks, got {rows}, {cache_rows}, {blocks}"
        )
    if any(not 0 <= lane < lanes for lane in lanes_of_rows):
        raise ValueError(f"every row's lane must be in 0..{lanes - 1}, got {tuple(lanes_of_rows)}")
    mesh = local_scores.device()
    out = fp.allocate((1, 1, rows, blocks), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh)
    cores = _lane_cores(rows, 0)
    core_set = _core_rows(cores)
    kernel = fp.reader_kernel(
        LANE_SCORE_ROWS,
        core_set,
        [*fp.accessor_args(local_scores), *fp.accessor_args(out)],
        [
            (core, [local_scores.buffer_address(), out.buffer_address(), row, int(lane), cache_rows, blocks])
            for row, (core, lane) in enumerate(zip(cores, lanes_of_rows))
        ],
    )
    cbs = [fp.cb_descriptor(0, BF16, blocks * 2, 1, core_set)]
    meta = fp.program_meta(  # each row's window of the wide scores in, the rows out; data movement
        NAME, "lane_score_rows", rows, writes=(out,), partial=((local_scores, rows * blocks * 2),), cores=rows
    )
    return fp.run_program([local_scores, out], fp.program_descriptor([kernel], cbs=cbs), meta=meta)


def lane_score_rows_composed(local_scores, lanes: int, cache_rows: int, blocks: int):
    """The lane chain's windows (ttnn/qsa.py ``_score_blocks_lanes``, wide form): B slices and their concat."""

    return lane_score_rows_of_composed(local_scores, tuple(range(lanes)), lanes, cache_rows, blocks)


def lane_score_rows_of_composed(local_scores, lanes_of_rows, lanes: int, cache_rows: int, blocks: int):
    """:func:`lane_score_rows_of` as the chain: one slice per output row and their concat."""

    dram = ttnn.DRAM_MEMORY_CONFIG
    windows = [
        ttnn.slice(
            local_scores,
            (0, 0, row, lane * cache_rows),
            (1, 1, row + 1, lane * cache_rows + blocks),
            memory_config=dram,
        )
        for row, lane in enumerate(lanes_of_rows)
    ]
    if len(windows) == 1:
        return windows[0]
    out = ttnn.concat(windows, dim=2, memory_config=dram)
    for w in windows:
        ttnn.deallocate(w)
    return out


# ------------------------------------------------------------------------------------------- program B1: score merge
# Replaces chain programs 532-535 (tilize, moreh_sum over the four devices, untilize, the mask add) after the
# all-gather of the local score rows: eight cores of 1024 columns, moreh_sum's 16-bit-dest add_tiles accumulation in
# device order then binary_ng's SFPU add of the mask (NearestEven).  Elementwise work, so the ROW_MAJOR chunks are
# processed as tiles.  The collective stays a runtime op: the model gathers the rows with ttnn.all_gather (dim 2).

SCORE_CHUNK = 1024
DEVICES = 4


def score_merge(gathered, mask):
    """The gathered score rows as 2 KB pages, ``[1, 1, 4 * rows * chunks, 1024]`` bf16 ROW_MAJOR (device d's row r,
    chunk c at page ``(d * rows + r) * chunks + c``: the all_gather of the rows reshaped to ``[1, 1, rows * chunks, 1024]``,
    the page form ttnn.all_gather takes at every context), + mask ``[1, 1, >= rows, W]`` (rows 0 .. rows-1 read: the
    one-row chain passes its row, the lanes pass their 32-row mask) -> the masked block scores ``[1, 1, rows, W]``
    bf16 ROW_MAJOR for the top-k."""

    shape, mshape = tuple(gathered.shape), tuple(mask.shape)
    if (
        len(mshape) != 4
        or mshape[:2] != (1, 1)
        or mshape[3] % SCORE_CHUNK
        or mask.dtype != BF16
        or mask.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(f"mask must be ROW_MAJOR bf16 [1, 1, >= rows, k * {SCORE_CHUNK}], got {mask.layout} {mshape}")
    width = mshape[3]
    chunks = width // SCORE_CHUNK
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or shape[3] != SCORE_CHUNK
        or shape[2] % (DEVICES * chunks)
        or gathered.dtype != BF16
        or gathered.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise ValueError(
            f"gathered score pages must be ROW_MAJOR bf16 [1, 1, {DEVICES} * rows * {chunks}, {SCORE_CHUNK}], got {gathered.layout} {shape}"
        )
    rows = shape[2] // (DEVICES * chunks)
    if not 1 <= rows <= ttnn.TILE_SIZE or mshape[2] < rows:
        raise ValueError(
            f"score merge takes 1..{ttnn.TILE_SIZE} rows with a mask of at least as many rows, got {rows} rows, mask {mshape}"
        )
    mesh = gathered.device()
    out = fp.allocate((1, 1, rows, width), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh)
    work = score_merge_work(width, mesh.compute_with_storage_grid_size())
    cores = fp.core_rectangle(work, mesh)
    cbs = [
        fp.cb_descriptor(cb, BF16, TILE_BF16, pages, cores)
        for cb, pages in ((0, DEVICES), (1, 1), (2, 1), (3, 1), (16, 1))
    ]
    reader = fp.reader_kernel(
        SCORE_MERGE["reader"],
        cores,
        [*fp.accessor_args(gathered), *fp.accessor_args(mask)],
        [(w.core, [gathered.buffer_address(), mask.buffer_address(), rows, w.start, w.count, chunks]) for w in work],
    )
    compute = fp.compute_kernel(
        SCORE_MERGE["compute"], cores, [], [(w.core, [rows, w.count]) for w in work], fp32_dest=False
    )
    writer = fp.writer_kernel(
        SCORE_MERGE["writer"],
        cores,
        fp.accessor_args(out),
        [(w.core, [out.buffer_address(), rows, w.start, w.count]) for w in work],
    )
    meta = fp.program_meta(  # the four devices' score pages and the rows' mask in, the merged rows out; 3 adds + mask
        "qsa_score_merge",
        "score_merge",
        rows,
        reads=(gathered,),
        writes=(out,),
        partial=((mask, rows * width * 2),),
        flops=DEVICES * rows * width,
        cores=len(work),
    )
    return fp.run_program([gathered, mask, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


def score_merge_work(width: int, grid) -> list[fp.CoreWork]:
    """The merge's ``width // SCORE_CHUNK`` chunks (1024 columns each: one tile row of bf16 per row) over the compute
    grid in ``fp.split_work``'s order, at most one core per chunk and never more cores than the grid has; a core with
    several chunks runs them one after another.  Every output element is one chunk's row of the same four device
    tiles summed in the same order plus its mask element, whatever core carries the chunk, so the placement does not
    enter the bits.  (The first form put chunk c on core (c, 0): at 65536 tokens of context, 16 chunks, that row runs
    past the 11- or 12-column compute grid of a p150 onto a dispatch core: "Illegal kernel placement".)"""

    chunks = width // SCORE_CHUNK
    if chunks < 1:
        raise ValueError(f"the score width must hold at least one {SCORE_CHUNK}-column chunk, got {width}")
    return fp.split_work(chunks, _Grid(grid))


class _Grid:
    """A ``compute_with_storage_grid_size()`` result as the mesh argument ``fp.split_work`` reads it from."""

    def __init__(self, grid) -> None:
        self._grid = grid

    def compute_with_storage_grid_size(self):
        return self._grid


def score_merge_composed(gathered, mask):
    """The all_reduce composite's local reduce (all_reduce_async.cpp local_sum: to TILE, moreh_sum over the device
    dim, back to ROW_MAJOR) and the chain's masked add, for one row; ``gathered`` in the merge's page form."""

    width = int(mask.shape[3])
    if tuple(gathered.shape) != (1, 1, DEVICES * (width // SCORE_CHUNK), SCORE_CHUNK):
        raise ValueError("the composed score merge is the one-row chain on the gathered pages")
    dram = ttnn.DRAM_MEMORY_CONFIG
    stacked = ttnn.reshape(gathered, (DEVICES, 1, 1, width))  # the pages are device-major, so this is the row order
    tiled = ttnn.to_layout(stacked, ttnn.TILE_LAYOUT, memory_config=dram)
    summed = ttnn.moreh_sum(tiled, dim=0, keepdim=True, memory_config=dram)
    row_major = ttnn.to_layout(summed, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    row = ttnn.reshape(row_major, (1, 1, 1, width))
    out = ttnn.add(row, mask, memory_config=dram, fast_and_approximate_mode=False)
    for t in (tiled, summed, row_major):
        ttnn.deallocate(t)
    return out


register(
    FusedKernel(
        name="qsa_score_merge",
        replaces="QSA score merge after the all-gather: tilize, moreh_sum over devices, untilize, mask add (4 programs/layer)",
        tolerance=BITWISE,
        fused=score_merge,
        composed=score_merge_composed,
        gate=None,
    )
)
