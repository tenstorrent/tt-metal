# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``greedy_tail``: the greedy epilogue's candidates and resolve as three data-movement programs around one gather.

Candidates (``Qwen38TTNNLMHead.greedy_candidates``, 9 programs: untilize, argmax, pad, reshape, max, reshape, max):
``scan`` on 40 cores reads row 0 of the local bf16 logits tiles and keeps each core's maximum with its lowest id;
``merge`` on one core picks the local maximum (lowest id on ties, ttnn.argmax's rule) and writes ``local_values``
(bf16 TILE ``[1,1,1,1]``), ``local_indices`` (uint32 ``[1,1,1]``) and the packed fp32 ROW_MAJOR row ``[value | id]``.
Resolve (``resolve_greedy_on_device``, 15 programs: two all_gathers with their relayouts, tie-break, argmax, rebase,
gather, splat, tilize): ONE ``all_gather`` of the packed rows, then ``resolve`` on one core: ``value_d -
owner_tie_break[d]`` in fp32, the first maximum in owner order, ``id + lm_head_vocab_starts[owner]`` in fp32, written
as lane 0 of the fp32 TILE token row.  Integer / exact-fp32 work on the RISC (IEEE round-to-nearest, as the SFPU's
fp32 ops): tolerance class BITWISE on the token id and on both candidate tensors.  Rows = 1 is the decode step; the
batched lanes (``Qwen38TTNNLMHead.greedy_candidates_lanes`` / ``resolve_greedy_lanes_on_device``, rows = 1..32, row u =
lane u) run the same three programs over the rows: the scan reads every lane row of its tiles, the merge writes one
packed 64-byte row per lane ([1,1,rows,16]), ONE all_gather of those rows, and the resolve writes lane u of the token
row (lanes past the row count zero, as the lanes chain's zero pad).  The MTP rows path keeps the chain.
"""

from __future__ import annotations

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "greedy_tail"
CANDIDATE_ROW = "candidate_row"  # the sampling server's per-shard top-k row, folded into the scan and the merge
CANDIDATES = 32  # = embedding.SAMPLING_CANDIDATES_PER_DEVICE (pinned by the static test)
TILE = fp.TILE
KERNELS = {name: fp.kernel_source(NAME, f"{name}.cpp") for name in ("scan", "merge", "resolve")}
CB_STAGE = 0
SCAN_CORES = 40
SCAN_CORES_ENV = (
    "QWEN38_FUSED_GREEDY_TAIL_SCAN_CORES"  # dev knob: cores of the scan program (40 default; 80 / 130 to sweep)
)


LANE_SPLIT_ENV = (
    "QWEN38_FUSED_GREEDY_TAIL_LANE_SPLIT"  # dev knob: 0 runs the lanes' per-core all-rows scan (the 2026-09-25 form)
)
LANE_SPLIT_CORES = 128  # (row, tile-group) items over at most this many cores


def lane_split(environ=None) -> bool:
    import os

    return (os.environ if environ is None else environ).get(LANE_SPLIT_ENV, "1").strip() != "0"


def scan_cores(environ=None) -> int:
    import os

    value = (os.environ if environ is None else environ).get(SCAN_CORES_ENV, "")
    return int(value) if value.strip() else SCAN_CORES


SCAN_STAGE_PAGES = 4  # 49 tiles x 128 bytes of face rows + the 16-byte pair
MERGE_STAGE_PAGES = 2  # zero tile + pairs + out
LIST_BYTES = 8 * CANDIDATES  # a core's candidate list: fp32 [values | ids]
RESOLVE_STAGE_PAGES = 3  # fp32 zero tile (4 KB) + three 64-byte reads
PACKED_LANES = 2  # the two-lane packed row [value | float(id)] (the composed chain's form; the kernels' rows == 1 path)
PACKED_LANES_ROWS = (
    16  # the 64-byte packed rows (the DRAM grain), [value | float(id) | zeros]: the lanes and the decode step
)
SCAN_ARGS = (
    "logits_addr",
    "pairs_addr",
    "first_tile",
    "tile_count",
    "core_index",
    "lists_addr",
    "row",
)  # lists: candidates; row: lane_split
MERGE_ARGS = (
    "pairs_addr",
    "zero_tile_addr",
    "values_addr",
    "indices_addr",
    "packed_addr",
    "lists_addr",
    "vocab_start_addr",
    "row_addr",
)
RESOLVE_ARGS = ("gathered_addr", "tie_break_addr", "vocab_starts_addr", "zero_tile_addr", "token_row_addr", "into_addr")
_ZERO_TILES: dict[int, tuple] = {}


def _embedding():
    from models.demos.blackhole.qwen38_flash_next.ttnn import embedding

    return embedding


def prepare(mesh):
    """One zero bf16 tile and one zero fp32 tile ``[1,1,32,32]`` per mesh (the merge's value tile and the resolve's token
    tile start from them), uploaded once (before trace capture)."""

    import torch

    key = id(mesh)
    if key not in _ZERO_TILES:
        zeros = torch.zeros(1, 1, TILE, TILE)
        _ZERO_TILES[key] = tuple(
            ttnn.from_torch(
                zeros, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for dtype in (ttnn.bfloat16, ttnn.float32)
        )
    return _ZERO_TILES[key]


def _one_core(mesh):
    core = ttnn.CoreCoord(0, 0)
    return core, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _rows_of(logits) -> int:
    shape = tuple(int(v) for v in logits.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or logits.dtype != ttnn.bfloat16 or logits.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"greedy_tail logits must be bf16 TILE [1,1,rows,vocab], got {logits.dtype} {logits.layout} {shape}"
        )
    if shape[3] % TILE:
        raise ValueError(f"greedy_tail logits width must be whole tiles, got {shape[3]}")
    return shape[2]


def merge_stage_pages(rows: int, cores: int, packed_lanes: int, candidates: int = 0) -> int:
    """The merge's staging for ``rows`` rows: the zero tile, the pairs rows, the packed rows and the indices row; with
    ``candidates`` the cores' lists, the vocab-start read and the shard row after them (merge.cpp's layout)."""

    if rows == 1 and packed_lanes == PACKED_LANES and not candidates:
        return MERGE_STAGE_PAGES
    pairs_bytes = ((16 * cores) + 63) & ~63
    if not candidates:
        return -(-(2048 + rows * pairs_bytes + rows * 4 * packed_lanes + 64) // 2048)
    lists = ((2048 + rows * pairs_bytes + rows * 4 * packed_lanes + 128) + 63) & ~63
    return -(-(lists + cores * 8 * candidates + 64 + 8 * candidates) // 2048)


def greedy_candidates(
    logits,
    *,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    packed_lanes: int = PACKED_LANES,
    candidates: int = 0,
    vocab_start=None,
) -> tuple:
    """``(local_values bf16 TILE [1,1,rows,1], local_indices uint32 [1,1,rows], packed fp32 ROW_MAJOR
    [1,1,rows,packed_lanes])`` of the ``rows`` valid rows (1 = the decode step, whose packed row is [value | id];
    the lanes pass ``PACKED_LANES_ROWS`` for 64-byte rows).  Row r's lane of the value tile is (r, 0).

    ``candidates`` = ``CANDIDATES`` (rows = 1, the sampling server): the same two programs also produce the shard's
    candidate row, fp32 ROW_MAJOR ``[1,1,1,2 * candidates]`` = [values | global ids] (the ids rebased by
    ``vocab_start``, the shard's fp32 TILE ``[1,1,1,1]`` first global id), returned as a fourth tensor: each scan core
    keeps its sorted top ``candidates`` behind the argmax's compare, the merge takes the shard's from the lists."""

    rows = _rows_of(logits)
    if not 1 <= rows <= TILE:
        raise ValueError(f"greedy_tail candidates take 1..{TILE} rows (one row tile), got {rows}")
    if packed_lanes not in (PACKED_LANES, PACKED_LANES_ROWS):
        raise ValueError(f"packed_lanes must be {PACKED_LANES} or {PACKED_LANES_ROWS}, got {packed_lanes}")
    if candidates not in (0, CANDIDATES):
        raise ValueError(f"candidates is 0 or {CANDIDATES}, got {candidates}")
    if candidates and rows != 1:
        raise ValueError("the candidate row folds into the one-row scan (the decode step)")
    if candidates and vocab_start is None:
        raise ValueError("the candidate row needs the shard's vocab-start tile")
    mesh = logits.device()
    tiles = int(logits.shape[3]) // TILE
    zero_bf16, _zero_fp32 = prepare(mesh)
    grid = mesh.compute_with_storage_grid_size()
    split = rows > 1 and lane_split()
    if split:
        # (row, tile group) items: G groups of tiles, rows x G cores in linear core order (core r * G + g takes row r, group g)
        groups = max(1, min(LANE_SPLIT_CORES // rows, tiles, grid.x * grid.y // rows))
        ranges = fp.split_work(tiles, mesh, cores=groups)
        cores = fp.split_work(rows * len(ranges), mesh, cores=rows * len(ranges))
        work = [
            fp.CoreWork(core=c.core, start=ranges[i % len(ranges)].start, count=ranges[i % len(ranges)].count)
            for i, c in enumerate(cores)
        ]
        item_row = [i // len(ranges) for i in range(len(work))]
        item_group = [i % len(ranges) for i in range(len(work))]
        pairs_per_row = len(ranges)
    else:
        work = fp.split_work(tiles, mesh, cores=min(scan_cores(), tiles, grid.x * grid.y))
        item_row = [0] * len(work)
        item_group = list(range(len(work)))
        pairs_per_row = len(work)
    grid = fp.core_rectangle(work, mesh)
    pairs_lanes = -(-4 * pairs_per_row // 16) * 16  # 16 bytes per pair, the row padded to the 64-byte DRAM read grain
    pairs = fp.allocate((1, 1, rows, pairs_lanes), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    lists = row = None
    if candidates:
        if 128 * max(w.count for w in work) + 16 + 8 * candidates > 2048 * SCAN_STAGE_PAGES:
            raise ValueError("the scan's staging cannot hold its tiles' face rows, the pair and the candidate list")
        lists = fp.allocate((1, 1, len(work), 2 * candidates), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
        row = fp.allocate((1, 1, 1, 2 * candidates), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    scan_tensors = [logits, pairs] + ([lists] if candidates else [])
    defines = [("GT_CANDIDATE_ROW", "1")] if candidates else []  # the kernels' list / row branches (see scan.cpp)
    scan = fp.reader_kernel(
        KERNELS["scan"],
        grid,
        [a for t in scan_tensors for a in fp.accessor_args(t)],
        [
            (
                w.core,
                [logits.buffer_address(), pairs.buffer_address(), w.start, w.count, item_group[i]]
                + ([lists.buffer_address()] if candidates else [0])
                + ([item_row[i]] if split else []),
            )
            for i, w in enumerate(work)
        ],
        defines=defines,
        named={
            "cb_stage": CB_STAGE,
            "lanes_per_tile": TILE,
            "rows": rows,
            "candidates": candidates,
            "lane_split": int(split),
        },
    )
    if rows == 1:
        scan_pages = SCAN_STAGE_PAGES
    elif split:
        scan_pages = -(
            -(128 * max(w.count for w in work) + 16) // 2048
        )  # 128 bytes per tile (its two grains) + the pair
    else:
        scan_pages = max(w.count for w in work) + 1  # one 2 KB slot per tile + the pairs
    kernel = CANDIDATE_ROW if candidates else NAME
    # the scan reads the lane rows of its tiles (two 64-byte grains per row per tile at rows 1 and in the lane split,
    # whole tiles otherwise), writes one 16-byte pair per (row, core) and, with candidates, its sorted list; one
    # compare per element, plus the insertion into the list
    tile_read = min(2048, 128 * rows) if (rows == 1 or split) else 2048
    meta = fp.program_meta(
        kernel,
        "scan",
        rows,
        writes=(pairs, *([lists] if candidates else [])),
        dram_bytes=tiles * tile_read,
        flops=rows * tiles * TILE * (1 + (2 if candidates else 0)),
        cores=len(work),
    )
    fp.run_program(
        scan_tensors,
        fp.program_descriptor([scan], cbs=[fp.cb_descriptor(CB_STAGE, ttnn.bfloat16, 2048, scan_pages, grid)]),
        meta=meta,
    )
    values = fp.allocate((1, 1, rows, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, memory_config)
    indices = fp.allocate((1, 1, rows), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    packed = fp.allocate((1, 1, rows, packed_lanes), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, memory_config)
    core, one = _one_core(mesh)
    tensors = [pairs, zero_bf16, values, indices, packed] + ([lists, vocab_start, row] if candidates else [])
    cores = pairs_per_row  # the merge compares the pairs of each row: the tile groups (lane_split) or the scan cores
    merge = fp.reader_kernel(
        KERNELS["merge"],
        one,
        [a for t in tensors for a in fp.accessor_args(t)],
        [(core, [t.buffer_address() for t in tensors])],
        defines=defines,
        named={
            "cb_stage": CB_STAGE,
            "cores": cores,
            "rows": rows,
            "packed_lanes": packed_lanes,
            "candidates": candidates,
        },
    )
    # the pairs (and the lists, the vocab start) and the zero tile in, the value tile, the indices row, the packed rows
    # (and the shard row) out; one compare per pair, plus the merge of the cores' lists
    meta = fp.program_meta(
        kernel,
        "merge",
        rows,
        reads=(pairs, zero_bf16, *([lists, vocab_start] if candidates else [])),
        writes=(values, indices, packed, *([row] if candidates else [])),
        flops=rows * cores * (1 + (candidates if candidates else 0)),
        cores=1,
    )
    fp.run_program(
        tensors,
        fp.program_descriptor(
            [merge],
            cbs=[
                fp.cb_descriptor(
                    CB_STAGE, ttnn.bfloat16, 2048, merge_stage_pages(rows, cores, packed_lanes, candidates), one
                )
            ],
        ),
        meta=meta,
    )
    ttnn.deallocate(pairs)
    if candidates:
        ttnn.deallocate(lists)
    for tensor in (values, indices, packed) + ((row,) if candidates else ()):
        tensor.update_tensor_topology(logits.tensor_topology())  # local partials, as the chain's argmax / max outputs
    return (values, indices, packed, row) if candidates else (values, indices, packed)


def greedy_candidates_composed(logits, *, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> tuple:
    """The chain op for op (the grid-reduce form of ``Qwen38TTNNLMHead.greedy_candidates``); ``packed`` is built from
    its two outputs with the chain's own widening ops so the resolve's input can be compared too."""

    rows = _rows_of(logits)
    row_major = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    local_indices = ttnn.argmax(row_major, dim=-1, keepdim=False)
    ttnn.deallocate(row_major)
    columns = 32
    width = int(logits.shape[3])
    grid_rows = ((width + columns - 1) // columns + 31) // 32 * 32
    padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, grid_rows * columns - width)], value=-1e30)
    grid = ttnn.reshape(padded, (1, rows, grid_rows, columns))
    partial = ttnn.max(grid, dim=-1)
    partial_rows = ttnn.reshape(partial, (1, 1, rows, grid_rows))
    local_values = ttnn.max(partial_rows, dim=-1, keepdim=True)
    for tensor in (padded, grid, partial, partial_rows):
        ttnn.deallocate(tensor)
    values_fp32 = ttnn.typecast(
        ttnn.to_layout(local_values, ttnn.ROW_MAJOR_LAYOUT), ttnn.float32, memory_config=memory_config
    )
    index_fp32 = ttnn.typecast(ttnn.reshape(local_indices, (1, 1, rows, 1)), ttnn.float32, memory_config=memory_config)
    packed = ttnn.concat([values_fp32, index_fp32], dim=3, memory_config=memory_config)
    ttnn.deallocate(values_fp32)
    ttnn.deallocate(index_fp32)
    return local_values, local_indices, packed


def resolve(gathered, tie_break, vocab_starts, *, into=None, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The token row (fp32 TILE ``[1,1,1,32]``, lane r = row r's global id, the other lanes 0.0) from the gathered
    packed rows ``[1,1,rows,devices*stride]`` (stride 2: the decode step's [value | id]; 16: the lanes' 64-byte rows);
    ``into`` (a resident TOKEN_ROW) receives the same tile from the same program (the chain's ``ttnn.copy`` after it).
    """

    embedding = _embedding()
    devices = int(tie_break.shape[3])
    shape = tuple(int(v) for v in gathered.shape)
    if into is not None and (
        tuple(into.shape) != embedding.TOKEN_ROW_SHAPE or into.dtype != ttnn.float32 or into.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(f"greedy_tail resolve copies into an fp32 TILE token row {embedding.TOKEN_ROW_SHAPE}")
    if (
        gathered.dtype != ttnn.float32
        or gathered.layout != ttnn.ROW_MAJOR_LAYOUT
        or len(shape) != 4
        or shape[:2] != (1, 1)
        or not 1 <= shape[2] <= TILE
        or shape[3] % devices
        or shape[3] // devices not in (PACKED_LANES, PACKED_LANES_ROWS)
    ):
        raise ValueError(
            f"greedy_tail resolve takes the fp32 ROW_MAJOR gathered packed rows [1,1,rows,{devices}*(2|16)], got {shape}"
        )
    rows, stride = shape[2], shape[3] // devices
    mesh = gathered.device()
    _zero_bf16, zero_fp32 = prepare(mesh)
    token_row = fp.allocate(embedding.TOKEN_ROW_SHAPE, ttnn.float32, ttnn.TILE_LAYOUT, mesh, memory_config)
    core, one = _one_core(mesh)
    tensors = [gathered, tie_break, vocab_starts, zero_fp32, token_row, token_row if into is None else into]
    kernel = fp.reader_kernel(
        KERNELS["resolve"],
        one,
        [a for t in tensors for a in fp.accessor_args(t)],
        [(core, [t.buffer_address() for t in tensors])],
        named={
            "cb_stage": CB_STAGE,
            "devices": devices,
            "copy_into": 0 if into is None else 1,
            "rows": rows,
            "stride": stride,
        },
    )
    row_bytes = (devices * stride * 4 + 63) & ~63
    pages = RESOLVE_STAGE_PAGES if rows == 1 and stride == PACKED_LANES else -(-(4096 + rows * row_bytes + 128) // 4096)
    # the gathered packed rows, the tie-break and vocab-start rows and the zero tile in, the token tile out (twice
    # with ``into``); per row the tie-break subtract, the first maximum over the devices and the rebase add
    meta = fp.program_meta(
        NAME,
        "resolve",
        rows,
        reads=(gathered, tie_break, vocab_starts, zero_fp32),
        writes=(token_row, *([into] if into is not None else [])),
        flops=rows * devices * 3,
        cores=1,
    )
    fp.run_program(
        tensors,
        fp.program_descriptor([kernel], cbs=[fp.cb_descriptor(CB_STAGE, ttnn.float32, 4096, pages, one)]),
        meta=meta,
    )
    token_row.update_tensor_topology(tie_break.tensor_topology())  # replicated, as the chain's token row
    return token_row


def resolve_composed(
    gathered_values, gathered_indices, tie_break, vocab_starts, unit_column, *, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """The chain's post-gather ops of ``resolve_greedy_on_device`` on the gathered bf16 TILE values ``[1,1,1,devices]`` and
    uint32 ROW_MAJOR indices ``[1,1,1,devices]``."""

    values_row_major = ttnn.to_layout(gathered_values, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    values_fp32 = ttnn.typecast(values_row_major, ttnn.float32, memory_config=memory_config)
    ranked = ttnn.subtract(values_fp32, tie_break, memory_config=memory_config)
    owner = ttnn.argmax(ranked, dim=-1, keepdim=True)
    index_fp32 = ttnn.typecast(gathered_indices, ttnn.float32, memory_config=memory_config)
    candidate_tokens = ttnn.add(index_fp32, vocab_starts, memory_config=memory_config)
    token = ttnn.gather(candidate_tokens, 3, owner, memory_config=memory_config)
    token_wide = ttnn.multiply(token, unit_column, memory_config=memory_config)
    token_row = ttnn.to_layout(token_wide, ttnn.TILE_LAYOUT, memory_config=memory_config)
    for tensor in (values_row_major, values_fp32, ranked, owner, index_fp32, candidate_tokens, token, token_wide):
        ttnn.deallocate(tensor)
    return token_row


def resolve_lanes_composed(
    gathered_values, gathered_indices, tie_break, vocab_starts, *, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """The lanes chain's post-gather ops (``embedding.resolve_greedy_lanes_on_device``, as written) on the gathered bf16
    TILE values ``[1,1,rows,devices]`` and uint32 ROW_MAJOR indices ``[1,1,rows,devices]``: the token row with lane u =
    row u's id, the lanes past the row count zero."""

    embedding = _embedding()
    rows = int(gathered_values.shape[2])
    values_fp32 = ttnn.typecast(gathered_values, ttnn.float32, memory_config=memory_config)
    tie = ttnn.to_layout(tie_break, ttnn.TILE_LAYOUT, memory_config=memory_config)
    ranked_tile = ttnn.subtract(values_fp32, tie, memory_config=memory_config)
    ranked = ttnn.to_layout(ranked_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    owner = ttnn.argmax(ranked, dim=-1, keepdim=True)
    index_tile = ttnn.to_layout(gathered_indices, ttnn.TILE_LAYOUT, memory_config=memory_config)
    index_fp32 = ttnn.typecast(index_tile, ttnn.float32, memory_config=memory_config)
    starts = ttnn.to_layout(vocab_starts, ttnn.TILE_LAYOUT, memory_config=memory_config)
    candidate_tile = ttnn.add(index_fp32, starts, memory_config=memory_config)
    candidate_tokens = ttnn.to_layout(candidate_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
    token_column = ttnn.gather(candidate_tokens, 3, owner, memory_config=memory_config)
    transients = [
        values_fp32,
        tie,
        ranked_tile,
        ranked,
        owner,
        index_tile,
        index_fp32,
        starts,
        candidate_tile,
        candidate_tokens,
    ]
    if rows < TILE:
        padded = ttnn.pad(token_column, [(0, 0), (0, 0), (0, TILE - rows), (0, 0)], value=0.0)
        transients.append(token_column)
        token_column = padded
    token_lanes = ttnn.reshape(token_column, embedding.TOKEN_ROW_SHAPE)
    token_row = ttnn.to_layout(token_lanes, ttnn.TILE_LAYOUT, memory_config=memory_config)
    for tensor in transients:
        ttnn.deallocate(tensor)
    ttnn.deallocate(token_lanes)
    return token_row


def greedy_candidates_lanes_fused(lm_head, logits):
    """``Qwen38TTNNLMHead.greedy_candidates_lanes`` on the fused programs: the lanes' rows (1..32) in one scan / merge
    pair, the packed rows ``[1,1,rows,16]`` the lanes' resolve gathers."""

    embedding = _embedding()
    rows = lm_head._validate_logits(logits)
    values, indices, packed = greedy_candidates(logits.tensor, packed_lanes=PACKED_LANES_ROWS)
    for tensor, shape in (
        (indices, (1, 1, rows)),
        (values, (1, 1, rows, 1)),
        (packed, (1, 1, rows, PACKED_LANES_ROWS)),
    ):
        lm_head.mesh_contract.mark_local_partial(
            tensor, replicated_reference=lm_head.weights.replicated_anchor, expected_shape=shape
        )
    return embedding.Qwen38GreedyCandidates(
        local_indices=indices, local_values=values, rows=rows, vocab_ranges=lm_head.weights.vocab_ranges, packed=packed
    )


def resolve_greedy_lanes_on_device_fused(lm_head, candidates):
    """``Qwen38TTNNLMHead.resolve_greedy_lanes_on_device`` with one gather of the packed lane rows and the resolve
    program (lane u of the token row = row u's id, the lanes past the row count 0.0); candidates without a packed row
    (the chain's) take the chain."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    embedding = _embedding()
    if getattr(candidates, "packed", None) is None:
        return type(lm_head).resolve_greedy_lanes_on_device(lm_head, candidates)
    rows = embedding.require_lane_count(candidates.rows, label="greedy candidate rows")
    if candidates.vocab_ranges != lm_head.weights.vocab_ranges:
        raise ValueError("greedy candidates have different vocabulary ownership")
    if lm_head.collective_topology != ttnn.Topology.Linear:
        raise RuntimeError("on-device greedy resolve requires Linear topology")
    if tuple(int(v) for v in candidates.packed.shape) != (1, 1, rows, PACKED_LANES_ROWS):
        raise ValueError(f"lane greedy candidates need packed rows [1,1,{rows},{PACKED_LANES_ROWS}]")
    lm_head.mesh_contract.validate_tensor(candidates.packed, placement=TensorPlacement.LOCAL_PARTIAL)
    constants = lm_head.weights.token_row
    gathered = ttnn.all_gather(
        candidates.packed, dim=3, cluster_axis=embedding.TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    lm_head.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
    token_row = resolve(gathered, constants.owner_tie_break, constants.lm_head_vocab_starts)
    ttnn.deallocate(gathered)
    lm_head.mesh_contract.validate_tensor(token_row, placement=TensorPlacement.REPLICATED)
    return token_row


def greedy_candidates_fused(lm_head, logits, *, values_by_gather: bool = False, candidate_row=None):
    """``Qwen38TTNNLMHead.greedy_candidates`` on the fused programs for one row; other rows keep the chain.

    ``candidate_row`` (the sampling chain's ``Qwen38TTNNSamplingCandidateConstants``): the one-row programs also produce
    the shard's candidate row (``shard_row``, ``sampling_candidates_fused`` gathers it); other rows keep the chain and
    leave it ``None``."""

    embedding = _embedding()
    rows = lm_head._validate_logits(logits)
    if rows != 1:
        return type(lm_head).greedy_candidates(lm_head, logits, values_by_gather=values_by_gather)
    # the 64-byte packed row: an 8-byte row is padded to the 64-byte alignment and its all_gather falls back to the
    # slower composite ("input rows (8 B) are padded to the 64 B memory alignment"); the row's content is unchanged
    if candidate_row is None:
        values, indices, packed = greedy_candidates(logits.tensor, packed_lanes=PACKED_LANES_ROWS)
        row = None
    else:
        values, indices, packed, row = greedy_candidates(
            logits.tensor,
            packed_lanes=PACKED_LANES_ROWS,
            candidates=CANDIDATES,
            vocab_start=candidate_row.shard_vocab_start,
        )
    marked = [(indices, (1, 1, rows)), (values, (1, 1, rows, 1)), (packed, (1, 1, 1, PACKED_LANES_ROWS))]
    if row is not None:
        marked.append((row, (1, 1, 1, 2 * CANDIDATES)))
    for tensor, shape in marked:
        lm_head.mesh_contract.mark_local_partial(
            tensor, replicated_reference=lm_head.weights.replicated_anchor, expected_shape=shape
        )
    return embedding.Qwen38GreedyCandidates(
        local_indices=indices,
        local_values=values,
        rows=rows,
        vocab_ranges=lm_head.weights.vocab_ranges,
        packed=packed,
        shard_row=row,
    )


def sampling_candidates_fused(lm_head, logits, constants, *, into=None, candidates=None):
    """``Qwen38TTNNLMHead.sampling_candidates`` from the scan's shard row: one all_gather of the four shards' [values |
    global ids] rows into the replicated candidate row, copied into ``into`` (``constants.readback_row`` by default).
    Callers without the scan's row (``candidates`` is ``None`` or has none: the verify rows, tools that build the row
    on its own) take the chain."""

    embedding = _embedding()
    if candidates is None or getattr(candidates, "shard_row", None) is None:
        return type(lm_head).sampling_candidates(lm_head, logits, constants, into=into)
    rows = lm_head._validate_logits(logits)
    if rows != 1:
        raise ValueError(f"the scan's candidate row is the one-row step's, got {rows} rows")
    constants.validate(lm_head.mesh_contract)
    target = constants.readback_row if into is None else into
    row = ttnn.all_gather(
        candidates.shard_row, dim=3, cluster_axis=embedding.TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(candidates.shard_row)
    shape = tuple(int(v) for v in row.shape)
    if (
        shape != embedding.SAMPLING_CANDIDATE_ROW_SHAPE
        or row.dtype != ttnn.float32
        or row.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise RuntimeError(
            f"the gathered candidate row must be fp32 ROW_MAJOR {embedding.SAMPLING_CANDIDATE_ROW_SHAPE}, got {row.dtype} {row.layout} {shape}"
        )
    lm_head.mesh_contract.validate_tensor(row, placement=embedding.TensorPlacement.REPLICATED)
    ttnn.copy(row, target)
    return row


def sampling_candidates_chain(lm_head, logits, constants, *, into=None, candidates=None):
    """The chain: per shard ``ttnn.topk``, the typecasts, the rebase, the pack and its all_gather (``candidates`` unused)."""

    return type(lm_head).sampling_candidates(lm_head, logits, constants, into=into)


def resolve_greedy_on_device_fused(lm_head, candidates, *, into=None):
    """``Qwen38TTNNLMHead.resolve_greedy_on_device`` with one gather of the packed rows and the resolve program (which
    also writes ``into``, the server's persistent token row, instead of a ttnn.copy); candidates without a packed row
    (the chain's) take the chain."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    embedding = _embedding()
    if getattr(candidates, "packed", None) is None:
        return type(lm_head).resolve_greedy_on_device(lm_head, candidates, into=into)
    if candidates.rows != 1:
        raise TypeError("on-device greedy resolve requires single-token Qwen38GreedyCandidates")
    if candidates.vocab_ranges != lm_head.weights.vocab_ranges:
        raise ValueError("greedy candidates have different vocabulary ownership")
    if lm_head.collective_topology != ttnn.Topology.Linear:
        raise RuntimeError("on-device greedy resolve requires Linear topology")
    lm_head.mesh_contract.validate_tensor(candidates.packed, placement=TensorPlacement.LOCAL_PARTIAL)
    constants = lm_head.weights.token_row
    gathered = ttnn.all_gather(
        candidates.packed, dim=3, cluster_axis=embedding.TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    lm_head.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
    token_row = resolve(gathered, constants.owner_tie_break, constants.lm_head_vocab_starts, into=into)
    ttnn.deallocate(gathered)
    lm_head.mesh_contract.validate_tensor(token_row, placement=TensorPlacement.REPLICATED)
    return token_row


def greedy_candidates_chain(lm_head, logits, *, values_by_gather: bool = False):
    return type(lm_head).greedy_candidates(lm_head, logits, values_by_gather=values_by_gather)


register(
    FusedKernel(
        name=NAME,
        replaces="greedy_candidates (9 programs) and resolve_greedy_on_device (15 programs) of the decode tail; the lanes' "
        "greedy_candidates_lanes and resolve_greedy_lanes_on_device (two gathers, tie-break, argmax, rebase, gather, pad, "
        "relayouts) as the same three programs over the lane rows and one gather",
        tolerance=BITWISE,
        fused=greedy_candidates_fused,
        composed=greedy_candidates_chain,
        gate=None,  # the single-chip device test is the component gate (random logits with ties vs the chain's ops)
    )
)
register(
    FusedKernel(
        name=CANDIDATE_ROW,
        replaces="the candidate row's per-shard ttnn.topk, two typecasts, the rebase add, the concat and its relayout "
        "(9 programs before the row's all_gather), folded into greedy_tail's scan and merge",
        tolerance=BITWISE,
        fused=sampling_candidates_fused,
        composed=sampling_candidates_chain,
        gate=None,  # values bitwise, ids equal up to the boundary tie set (the candidate row's agreement rule)
    )
)
