# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``position_derive``: the decode step's position-derived tensors as one program; ``position_advance`` (this module's
second kernel, ``advance.cpp``): the step's closing ``P += 1`` as one in-place program.  From the uint32 position ``P``
(``[1,1,1,1]`` ROW_MAJOR) and the uint32 rotary shift ``S`` (``[1,1,1,1]``; 0 for text, the images' accumulated
shift otherwise, ``S <= P`` by the host's check): the two uint32 index rows (``P`` and ``P & ~3`` in 32 lanes), the
four RoPE rows (cos/sin table rows ``P - S`` and ``(P & ~3) - S``) and the nine QSA inputs of
``derive_qsa_position_inputs`` -- the chain's 42 programs (two uint32 row ops, two uint32 subtracts, four ``embedding``
gathers, 33 integer / 0-1 mask ops).

Every output is a step function of ``P``: the kernel (``kernels/derive.cpp``, one data-movement RISC, no compute)
assembles each row in L1 from constant template rows (zeros | MASK, zeros | ALL_ONES, a zero tile and a ones-column
tile, uploaded once per mesh) by NoC reads cut at the 64-byte DRAM read grain, stores the few boundary lanes, and
writes each output page once.  Integer and copy work only: tolerance class BITWISE against the chain and the torch
oracle ``emulate_qsa_position_inputs``.  Tile padding: the one-hot tiles carry zeros outside column 0 (the chain's
padding columns hold ``eq(0, P % n)``; every consumer broadcasts column 0) and the RoPE tiles carry the table row in
row 0 only (the chain repeats it over the 32 rows; every consumer is elementwise on the one logical row).
"""

from __future__ import annotations

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "position_derive"
TILE = fp.TILE
ROPE_DIM = 64
KERNEL = fp.kernel_source(NAME, "derive.cpp")
LANES_KERNEL = fp.kernel_source(NAME, "derive_lanes.cpp")
LANE_COLUMNS = 8  # lane u's core: column u % 8, core row u // 8 (32 lanes on 4 core rows)
ADVANCE_NAME = "position_advance"
ADVANCE_KERNEL = fp.kernel_source(NAME, "advance.cpp")
ADVANCE_STAGE_BYTES = 64  # the scalar page's DRAM read grain
CB_STAGE = 0
STAGE_PAGE_BYTES = 2048
STAGE_PAGES = 24  # >= the kernel's STAGE_BYTES (bf16 row 16 KB + uint32 row 8.3 KB + 2 tiles + rows + scalars)
ONE_BF16 = 0x3F80
RUNTIME_ARGS = (
    "position",
    "bf16_templates",
    "u32_templates",
    "tile_templates",
    "cos_table",
    "sin_table",
    "kv_block_start",
    "kv_row_hit",
    "kv_row_keep",
    "ring_hit",
    "ring_keep",
    "block_index_i32",
    "indexer_neg_mask",
    "row_keep_bits",
    "row_fill",
    "index_row",
    "block_start_row",
    "cos",
    "sin",
    "block_start_cos",
    "block_start_sin",
    "rope_shift",  # the last input: appended so the output addresses keep their argument numbers
)
QSA_OUTPUTS = RUNTIME_ARGS[6:15]
OUTPUT_ARGS = RUNTIME_ARGS[6:21]
LANES_RUNTIME_ARGS = (
    "position_row",
    "lane_offsets",
    "bf16_templates",
    "u32_templates",
    "tile_templates",
    "cos_table",
    "sin_table",
    "kv_block_start",
    "kv_row_hit",
    "indexer_neg_mask",
    "row_keep_bits",
    "row_fill",
    "index_row",
    "block_start_row",
    "cos",
    "sin",
    "block_start_cos",
    "block_start_sin",
    "rope_shift_row",  # the last input (lane u's shift S_u): appended as above
)
LANES_QSA_OUTPUTS = LANES_RUNTIME_ARGS[7:12]  # the fused QSA lane body's position inputs
LANES_OUTPUT_ARGS = LANES_RUNTIME_ARGS[7:18]
LANES_STAGE_PAGES = 26  # >= derive_lanes.cpp's STAGE_BYTES (bf16 row 16 KB + uint32 row 8.3 KB + a tile + rows)
_TEMPLATES: dict[tuple[int, int], tuple] = {}


def _qsa():
    from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module

    return qsa_module


def _contracts():
    from models.demos.blackhole.qwen38_flash_next.ttnn import contracts

    return contracts


def prepare(mesh, blocks: int):
    """The constant templates for ``blocks`` allocated compressed blocks, uploaded once per mesh (before trace capture):
    bf16 ``[1,1,1,2*blocks]`` = zeros | INDEXER_MASK_VALUE, uint32 ``[1,1,1,2*slots]`` = zeros | ALL_ONES, bf16 TILE
    ``[1,1,32,64]`` = a zero tile then a tile with 1.0 in column 0."""

    import torch

    key = (id(mesh), blocks)
    if key not in _TEMPLATES:
        qsa = _qsa()
        slots = qsa.SPARSE_INDEX_CAPACITY
        bf16_row = torch.cat([torch.zeros(blocks), torch.full((blocks,), qsa.INDEXER_MASK_VALUE)]).to(torch.bfloat16)
        u32_row = torch.cat(
            [torch.zeros(slots, dtype=torch.int64), torch.full((slots,), qsa.ALL_ONES_U32, dtype=torch.int64)]
        )
        tiles = torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16)
        tiles[:, TILE] = 1.0
        upload = lambda host, dtype, layout: ttnn.from_torch(
            host, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _TEMPLATES[key] = (
            upload(bf16_row.reshape(1, 1, 1, 2 * blocks), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            upload(u32_row.reshape(1, 1, 1, 2 * slots), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            upload(tiles.reshape(1, 1, TILE, 2 * TILE), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        )
    return _TEMPLATES[key]


def _expect(tensor, shape, dtype, layout, label):
    got = tuple(int(v) for v in tensor.shape)
    if got != tuple(shape) or tensor.dtype != dtype or tensor.layout != layout:
        raise ValueError(
            f"position_derive {label} must be {dtype} {layout} {list(shape)}, got {tensor.dtype} {tensor.layout} {list(got)}"
        )


def outputs(mesh, blocks: int, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> dict:
    """Fresh output tensors in the chain's shapes, dtypes and layouts."""

    slots = _qsa().SPARSE_INDEX_CAPACITY
    rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    spec = {
        "kv_block_start": ((1, 1, 1, 1), ttnn.uint32, rm),
        "kv_row_hit": ((1, 1, TILE, 1), ttnn.bfloat16, tile),
        "kv_row_keep": ((1, 1, TILE, 1), ttnn.bfloat16, tile),
        "ring_hit": ((1, 1, TILE, 1), ttnn.bfloat16, tile),
        "ring_keep": ((1, 1, TILE, 1), ttnn.bfloat16, tile),
        "block_index_i32": ((1,), ttnn.int32, rm),
        "indexer_neg_mask": ((1, 1, 1, blocks), ttnn.bfloat16, rm),
        "row_keep_bits": ((1, 1, 1, slots), ttnn.uint32, rm),
        "row_fill": ((1, 1, 1, slots), ttnn.uint32, rm),
        "index_row": ((1, 1, 1, TILE), ttnn.uint32, rm),
        "block_start_row": ((1, 1, 1, TILE), ttnn.uint32, rm),
        "cos": ((1, 1, 1, ROPE_DIM), ttnn.bfloat16, tile),
        "sin": ((1, 1, 1, ROPE_DIM), ttnn.bfloat16, tile),
        "block_start_cos": ((1, 1, 1, ROPE_DIM), ttnn.bfloat16, tile),
        "block_start_sin": ((1, 1, 1, ROPE_DIM), ttnn.bfloat16, tile),
    }
    return {
        name: fp.allocate(shape, dtype, layout, mesh, memory_config) for name, (shape, dtype, layout) in spec.items()
    }


def position_derive(
    position, cos_table, sin_table, *, blocks: int, rope_shift, memory_config=ttnn.DRAM_MEMORY_CONFIG
) -> dict:
    """All fifteen position-derived tensors of one decode step, by name (``OUTPUT_ARGS``); ``rope_shift`` is the
    uint32 ``[1,1,1,1]`` rotary shift the four RoPE table reads subtract from ``P`` and ``P & ~3``."""

    qsa, contracts = _qsa(), _contracts()
    _expect(position, (1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, "position")
    _expect(rope_shift, (1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, "rotary shift")
    for name, table in (("cos table", cos_table), ("sin table", sin_table)):
        if table.dtype != ttnn.bfloat16 or table.layout != ttnn.ROW_MAJOR_LAYOUT or int(table.shape[-1]) != ROPE_DIM:
            raise ValueError(f"position_derive {name} must be bf16 ROW_MAJOR [1,1,ctx,{ROPE_DIM}]")
    mesh = position.device()
    bf16_tpl, u32_tpl, tile_tpl = prepare(mesh, blocks)
    outs = outputs(mesh, blocks, memory_config)
    tensors = (
        [position, bf16_tpl, u32_tpl, tile_tpl, cos_table, sin_table]
        + [outs[name] for name in OUTPUT_ARGS]
        + [rope_shift]
    )
    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    named = {
        "blocks": blocks,
        "slots": qsa.SPARSE_INDEX_CAPACITY,
        "block_topk": qsa.BLOCK_TOPK,
        "kv_row_mask": qsa.KV_ROW_MASK,
        "ring_mask": qsa.COMPRESS_RATIO - 1,
        "kv_block_start_mask": qsa.KV_BLOCK_START_MASK,
        "lane_block_mask": contracts.BLOCK_START_LANE_MASK,
        "all_ones": qsa.ALL_ONES_U32,
        "one_bf16": ONE_BF16,
        "rope_dim": ROPE_DIM,
        "cb_stage": CB_STAGE,
    }
    compile_args = [a for t in tensors for a in fp.accessor_args(t)]
    reader = fp.reader_kernel(KERNEL, grid, compile_args, [(core, [t.buffer_address() for t in tensors])], named=named)
    cbs = [fp.cb_descriptor(CB_STAGE, ttnn.bfloat16, STAGE_PAGE_BYTES, STAGE_PAGES, grid)]
    # the position, the templates and the four table rows (two 128-byte rows per table) in, every output page out;
    # integer and copy work only
    meta = fp.program_meta(
        NAME,
        "derive",
        1,
        reads=(position, bf16_tpl, u32_tpl, tile_tpl, rope_shift),
        writes=tuple(outs.values()),
        dram_bytes=4 * ROPE_DIM * 2,
        cores=1,
        outputs=tuple((tensor, position) for tensor in outs.values()),  # replicated, as the position is
    )
    fp.run_program(tensors, fp.program_descriptor([reader], cbs=cbs), meta=meta)
    return outs


def lanes_outputs(mesh, blocks: int, lanes: int, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> dict:
    """Fresh output tensors of the lane derive in the lane chain's shapes, dtypes and layouts (32 lane rows; the
    kv_row_hit tiles of the ``lanes`` active lanes)."""

    slots = _qsa().SPARSE_INDEX_CAPACITY
    rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
    spec = {
        "kv_block_start": ((1, 1, 1, TILE), ttnn.uint32, rm),
        "kv_row_hit": ((1, lanes, TILE, 1), ttnn.bfloat16, tile),
        "indexer_neg_mask": ((1, 1, TILE, blocks), ttnn.bfloat16, rm),
        "row_keep_bits": ((1, 1, TILE, slots), ttnn.uint32, rm),
        "row_fill": ((1, 1, TILE, slots), ttnn.uint32, rm),
        "index_row": ((1, 1, 1, TILE), ttnn.uint32, rm),
        "block_start_row": ((1, 1, 1, TILE), ttnn.uint32, rm),
        "cos": ((1, 1, TILE, ROPE_DIM), ttnn.bfloat16, tile),
        "sin": ((1, 1, TILE, ROPE_DIM), ttnn.bfloat16, tile),
        "block_start_cos": ((1, 1, TILE, ROPE_DIM), ttnn.bfloat16, tile),
        "block_start_sin": ((1, 1, TILE, ROPE_DIM), ttnn.bfloat16, tile),
    }
    return {
        name: fp.allocate(shape, dtype, layout, mesh, memory_config) for name, (shape, dtype, layout) in spec.items()
    }


def position_derive_lanes(
    position_row,
    lane_offsets,
    cos_table,
    sin_table,
    *,
    blocks: int,
    lanes: int,
    rope_shift_row,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> dict:
    """The lane body's position-derived tensors from the uint32 position row ``[1,1,1,32]`` (lane u = P_u; the idle
    lanes at the row's residue, as the chain derives them too): one program, one core per lane row.  Row u of every
    row tensor is the 1-row derive at P_u (the fill's tail ids carry ``lane_offsets[u]``, lane u's KV region start, as
    the chain's ``block_offsets_lanes`` / ``arange_slots_lanes`` do); the four RoPE tiles hold the table rows P_u and
    P_u & ~3, each less lane u's rotary shift ``rope_shift_row[u]``, in row u; ``kv_row_hit`` is the ``lanes``
    active lanes' one-hot tiles; the index row, the block-start row and ``kv_block_start`` are the 32-lane rows.  By
    name (``LANES_OUTPUT_ARGS``)."""

    qsa, contracts = _qsa(), _contracts()
    lanes = contracts.require_lane_count(lanes, label="position derive lanes")
    _expect(position_row, (1, 1, 1, TILE), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, "position row")
    _expect(lane_offsets, (1, 1, 1, TILE), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, "lane offsets row")
    _expect(rope_shift_row, (1, 1, 1, TILE), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, "rotary shift row")
    for name, table in (("cos table", cos_table), ("sin table", sin_table)):
        if table.dtype != ttnn.bfloat16 or table.layout != ttnn.ROW_MAJOR_LAYOUT or int(table.shape[-1]) != ROPE_DIM:
            raise ValueError(f"position_derive {name} must be bf16 ROW_MAJOR [1,1,ctx,{ROPE_DIM}]")
    mesh = position_row.device()
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < LANE_COLUMNS or grid.y < TILE // LANE_COLUMNS:
        raise ValueError(f"position_derive lanes needs an {LANE_COLUMNS} x {TILE // LANE_COLUMNS} core grid")
    bf16_tpl, u32_tpl, tile_tpl = prepare(mesh, blocks)
    outs = lanes_outputs(mesh, blocks, lanes, memory_config)
    tensors = (
        [position_row, lane_offsets, bf16_tpl, u32_tpl, tile_tpl, cos_table, sin_table]
        + [outs[name] for name in LANES_OUTPUT_ARGS]
        + [rope_shift_row]
    )
    cores = [ttnn.CoreCoord(u % LANE_COLUMNS, u // LANE_COLUMNS) for u in range(TILE)]
    grid_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(LANE_COLUMNS - 1, TILE // LANE_COLUMNS - 1))]
    )
    named = {
        "blocks": blocks,
        "slots": qsa.SPARSE_INDEX_CAPACITY,
        "block_topk": qsa.BLOCK_TOPK,
        "kv_row_mask": qsa.KV_ROW_MASK,
        "ring_mask": qsa.COMPRESS_RATIO - 1,
        "kv_block_start_mask": qsa.KV_BLOCK_START_MASK,
        "lane_block_mask": contracts.BLOCK_START_LANE_MASK,
        "all_ones": qsa.ALL_ONES_U32,
        "one_bf16": ONE_BF16,
        "rope_dim": ROPE_DIM,
        "cb_stage": CB_STAGE,
    }
    compile_args = [a for t in tensors for a in fp.accessor_args(t)]
    reader = fp.reader_kernel(
        LANES_KERNEL,
        grid_set,
        compile_args,
        [(core, [t.buffer_address() for t in tensors] + [u, lanes]) for u, core in enumerate(cores)],
        named=named,
    )
    cbs = [fp.cb_descriptor(CB_STAGE, ttnn.bfloat16, STAGE_PAGE_BYTES, LANES_STAGE_PAGES, grid_set)]
    # per lane core: the position and offsets rows, the templates and its four table rows in; every output page out
    meta = fp.program_meta(
        NAME,
        "derive_lanes",
        lanes,
        writes=tuple(outs.values()),
        dram_bytes=TILE
        * (
            fp.tensor_bytes(position_row)
            + fp.tensor_bytes(lane_offsets)
            + fp.tensor_bytes(bf16_tpl)
            + fp.tensor_bytes(u32_tpl)
            + fp.tensor_bytes(tile_tpl)
            + fp.tensor_bytes(rope_shift_row)
            + 4 * ROPE_DIM * 2
        ),
        cores=TILE,
        outputs=tuple((tensor, position_row) for tensor in outs.values()),  # replicated, as the position row is
    )
    fp.run_program(tensors, fp.program_descriptor([reader], cbs=cbs), meta=meta)
    return outs


def derive_lanes_fused(model, state):
    """The lane body's position derive on the fused program: ``(Qwen38TTNNRoPEInputs, Qwen38TTNNQSAFusedLaneInputs)``
    -- the fused QSA lane body's inputs (the chain-only selection tiles and scalars are not derived)."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement
    from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNRoPEInputs

    qsa = _qsa()
    outs = position_derive_lanes(
        state.position.row,
        state.qsa_lane_constants.kv_offsets_row,
        model.rope_table.cos_table,
        model.rope_table.sin_table,
        blocks=model.qsa_position_constants.allocated_compressed_blocks,
        lanes=state.lanes,
        rope_shift_row=state.position.shift_row,
    )
    for name in LANES_OUTPUT_ARGS:
        model.mesh_contract.validate_tensor(outs[name], placement=TensorPlacement.REPLICATED)
    ttnn.deallocate(outs["index_row"])
    ttnn.deallocate(outs["block_start_row"])
    rope = Qwen38TTNNRoPEInputs(None, outs["cos"], outs["sin"], outs["block_start_cos"], outs["block_start_sin"])
    return rope, qsa.Qwen38TTNNQSAFusedLaneInputs(**{name: outs[name] for name in LANES_QSA_OUTPUTS})


def derive_lanes_composed(model, state):
    """The lane body's chain (model.py forward_decode_lanes: the index rows, the RoPE row tiles,
    ``derive_qsa_lane_inputs``)."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.model import _deallocate_unique

    index_row = state.position.index_row()
    rope_index_row = state.position.rope_index_row(index_row)
    rope_block_start_row = state.position.rope_block_start_index_row(index_row)
    rope = model.rope_table.rows_chunk(rope_index_row, rope_block_start_row)
    _deallocate_unique(index_row, rope_index_row, rope_block_start_row)
    return rope, _qsa().derive_qsa_lane_inputs(
        state.position.row, model.qsa_position_constants, state.qsa_chunk_constants, state.qsa_lane_constants
    )


_COMPOSED_CONSTANTS: dict[tuple[int, int], tuple] = {}


def composed_constants(mesh, blocks: int):
    """The chain's resident constants (``Qwen38TTNNDevicePosition`` rows, ``Qwen38TTNNQSAPositionConstants``) for one
    chip, uploaded once per mesh so the composed chain is trace-capturable like the model's."""

    import torch
    from types import SimpleNamespace

    key = (id(mesh), blocks)
    if key not in _COMPOSED_CONSTANTS:
        qsa, contracts = _qsa(), _contracts()
        dram = ttnn.DRAM_MEMORY_CONFIG

        def u32(host, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(host, dtype=ttnn.uint32, layout=layout, device=mesh, memory_config=dram)

        slots = qsa.SPARSE_INDEX_CAPACITY
        _COMPOSED_CONSTANTS[key] = (
            u32(torch.ones(1, 1, 1, TILE, dtype=torch.int64)),
            u32(torch.full((1, 1, 1, TILE), contracts.BLOCK_START_LANE_MASK, dtype=torch.int64)),
            SimpleNamespace(
                allocated_compressed_blocks=blocks,
                arange32_col=u32(torch.arange(TILE, dtype=torch.int64).reshape(1, 1, TILE, 1), ttnn.TILE_LAYOUT),
                arange_blocks=u32(torch.arange(blocks, dtype=torch.int64).reshape(1, 1, 1, blocks)),
                arange_row=u32(torch.arange(slots, dtype=torch.int64).reshape(1, 1, 1, slots)),
                all_ones=u32(torch.full((1, 1, 1, 1), qsa.ALL_ONES_U32, dtype=torch.int64)),
                high27_mask=u32(torch.full((1, 1, 1, 1), qsa.KV_BLOCK_START_MASK, dtype=torch.int64)),
            ),
        )
    return _COMPOSED_CONSTANTS[key]


def position_derive_composed(
    position, cos_table, sin_table, *, blocks: int, rope_shift, memory_config=ttnn.DRAM_MEMORY_CONFIG
) -> dict:
    """The chain op for op (``Qwen38DevicePosition.index_row`` / ``block_start_index_row`` / ``rope_index_row`` /
    ``rope_block_start_index_row``, ``Qwen38TTNNRoPETable.rows``, ``derive_qsa_position_inputs``) on the same tensors,
    with resident constants (``composed_constants``)."""

    qsa = _qsa()
    dram = ttnn.DRAM_MEMORY_CONFIG
    ones_row, mask_row, constants = composed_constants(position.device(), blocks)
    index_row = ttnn.multiply(ones_row, position, memory_config=dram)
    block_start_row = ttnn.bitwise_and(index_row, mask_row, memory_config=dram)
    rope_index_row = ttnn.subtract(index_row, rope_shift, memory_config=dram)
    rope_block_start_row = ttnn.subtract(block_start_row, rope_shift, memory_config=dram)
    indices = ttnn.reshape(rope_index_row, (1, 1, TILE))
    block_indices = ttnn.reshape(rope_block_start_row, (1, 1, TILE))
    padded = ttnn.Shape((1, 1, TILE, ROPE_DIM))

    def lookup(table_indices, table):
        rows = ttnn.embedding(table_indices, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=dram)
        rows = ttnn.unsqueeze_to_4D(rows) if len(rows.shape) == 3 else rows
        return ttnn.reshape(rows, ttnn.Shape((1, 1, 1, ROPE_DIM)), padded)

    outs = {
        "index_row": index_row,
        "block_start_row": block_start_row,
        "cos": lookup(indices, cos_table),
        "sin": lookup(indices, sin_table),
        "block_start_cos": lookup(block_indices, cos_table),
        "block_start_sin": lookup(block_indices, sin_table),
    }
    ttnn.deallocate(rope_index_row)
    ttnn.deallocate(rope_block_start_row)
    derived = qsa.derive_qsa_position_inputs(position, constants)
    outs.update({name: getattr(derived, name) for name in QSA_OUTPUTS})
    return outs


def derive_fused(model, state):
    """The decode body's position derive on the fused program: ``(Qwen38TTNNRoPEInputs, Qwen38TTNNQSAPositionInputs)``."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement
    from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNRoPEInputs

    qsa = _qsa()
    outs = position_derive(
        state.position.scalar,
        model.rope_table.cos_table,
        model.rope_table.sin_table,
        blocks=model.qsa_position_constants.allocated_compressed_blocks,
        rope_shift=state.position.shift,
    )
    for name in (
        "cos",
        "sin",
        "block_start_cos",
        "block_start_sin",
        "kv_block_start",
        "indexer_neg_mask",
        "row_keep_bits",
        "row_fill",
    ):
        model.mesh_contract.validate_tensor(outs[name], placement=TensorPlacement.REPLICATED)
    ttnn.deallocate(outs["index_row"])
    ttnn.deallocate(outs["block_start_row"])
    rope = Qwen38TTNNRoPEInputs(None, outs["cos"], outs["sin"], outs["block_start_cos"], outs["block_start_sin"])
    return rope, qsa.Qwen38TTNNQSAPositionInputs(**{name: outs[name] for name in QSA_OUTPUTS})


def derive_composed(model, state):
    """The decode body's chain (model.py: index rows, RoPE rows, ``derive_qsa_position_inputs``)."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.model import _deallocate_unique

    index_row = state.position.index_row()
    rope_index_row = state.position.rope_index_row(index_row)
    rope_block_start_row = state.position.rope_block_start_index_row(index_row)
    rope = model.rope_table.rows(rope_index_row, rope_block_start_row)
    _deallocate_unique(index_row, rope_index_row, rope_block_start_row)
    return rope, _qsa().derive_qsa_position_inputs(state.position.scalar, model.qsa_position_constants)


def advance(position, count: int = 1):
    """``Qwen38TTNNDevicePosition.advance`` / ``advance_by`` as one program: ``P += count`` in place on the resident
    uint32 scalar (the chain's ``ttnn.add`` into a fresh tensor + in-place ``ttnn.copy``).  Returns the scalar."""

    if isinstance(count, bool) or type(count) is not int or count <= 0:
        raise ValueError(f"position advance needs a positive int, got {count!r}")
    _expect(position, (1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, "position")
    mesh = position.device()
    core = ttnn.CoreCoord(0, 0)
    one = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    kernel = fp.reader_kernel(
        ADVANCE_KERNEL,
        one,
        fp.accessor_args(position),
        [(core, [position.buffer_address()])],
        named={"cb_stage": CB_STAGE, "count": count},
    )
    meta = fp.program_meta(ADVANCE_NAME, "advance", 1, reads=(position,), writes=(position,), flops=1, cores=1)
    fp.run_program(  # generic_op wants an input and an output: the resident scalar is both (read, then written)
        [position, position],
        fp.program_descriptor([kernel], cbs=[fp.cb_descriptor(CB_STAGE, ttnn.uint32, ADVANCE_STAGE_BYTES, 1, one)]),
        meta=meta,
    )
    return position


def advance_composed(position, count: int = 1):
    """The chain's two programs (contracts.py ``advance`` / ``advance_by``): add into a fresh tensor, copy back."""

    advanced = ttnn.add(position, count, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy(advanced, position)
    ttnn.deallocate(advanced)
    return position


register(
    FusedKernel(
        name=ADVANCE_NAME,
        replaces="Qwen38TTNNDevicePosition.advance / advance_by: ttnn.add + in-place ttnn.copy (2 programs per decode step "
        "and per prefill chunk)",
        tolerance=BITWISE,
        fused=advance,
        composed=advance_composed,
        gate=None,  # the device test (P sweep, counts, trace replay) against the chain's two ops is the component gate
    )
)


register(
    FusedKernel(
        name=NAME,
        replaces="index_row, block_start_index_row, the rotary-shift subtracts, the four RoPE embedding gathers and derive_qsa_position_inputs (42 programs per step); "
        "the lanes' index rows, RoPE row tiles and derive_qsa_lane_inputs (the fused QSA lane body's five inputs) as one program",
        tolerance=BITWISE,
        fused=derive_fused,
        composed=derive_composed,
        gate=None,  # the P-sweep device test against the chain and emulate_qsa_position_inputs is the component gate
    )
)
