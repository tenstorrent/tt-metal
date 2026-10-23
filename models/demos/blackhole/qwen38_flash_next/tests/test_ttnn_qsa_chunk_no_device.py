# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill chunk (32 rows) of the QSA layer: the chunk inputs, constants and RoPE rows, without a device.

The chunk inputs are the 1-row position-generic derivation on same-shape templates: the torch emulation of
row j must equal the 1-row emulation at P + j field for field.  The two bf16 select tiles (block-mean pooling
and the per-block row pick) are checked under the device's matmul arithmetic model.  The source pins hold the
derivation to exact integer ops with no host readback and the RoPE chunk rows free of the one-row view.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    ALL_ONES_U32,
    BLOCK_TOPK,
    CHUNK_BLOCKS,
    COMPRESS_RATIO,
    INDEXER_MASK_VALUE,
    SPARSE_INDEX_CAPACITY,
    TOKEN_BUDGET,
    Qwen38TTNNQSAChunkConstants,
    Qwen38TTNNQSAChunkInputs,
    emulate_qsa_chunk_inputs,
    emulate_qsa_position_inputs,
    qsa_chunk_constant_rows,
)

RESIDENT_BLOCKS = 32768 // COMPRESS_RATIO
REPOSITORY = Path(inspect.getfile(qsa_module)).resolve().parents[5]
CHUNK_POSITIONS = (0, 32, 64, 2016, 2048, 2080, 8160, 32736)
CHUNK_INPUT_FIELDS = ("kv_block_start", "block_index_i32", "indexer_neg_mask", "row_keep_bits", "row_fill")


def _ttnn_calls(function) -> list[str]:
    tree = ast.parse(_dedent(function))
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func.value) == "ttnn"
    ]


def _dedent(function) -> str:
    lines = inspect.getsource(function).splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)


# --- emulation: row j of the chunk is the 1-row derivation at P + j --------------------


@pytest.mark.parametrize("position", CHUNK_POSITIONS)
def test_chunk_emulation_rows_are_the_per_position_emulation(position: int) -> None:
    chunk = emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=RESIDENT_BLOCKS)
    assert tuple(chunk) == CHUNK_INPUT_FIELDS
    assert chunk["kv_block_start"].tolist() == [[[[position]]]]  # P % 32 == 0: the slab starts at P itself
    assert [int(index) for index in chunk["block_index_i32"]] == [position // 4 + block for block in range(8)]
    assert all(index.dtype == torch.int32 and tuple(index.shape) == (1,) for index in chunk["block_index_i32"])
    assert chunk["indexer_neg_mask"].shape == (1, 1, CHUNK_ROWS, RESIDENT_BLOCKS)
    assert chunk["row_keep_bits"].shape == chunk["row_fill"].shape == (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY)
    for row in range(CHUNK_ROWS):
        single = emulate_qsa_position_inputs(position + row, allocated_compressed_blocks=RESIDENT_BLOCKS)
        for name in ("indexer_neg_mask", "row_keep_bits", "row_fill"):
            assert torch.equal(chunk[name][:, :, row : row + 1], single[name]), (name, row)
    # The chunk's last row completes block P/4 + 7 and is the first row whose block P/4 + 7 is unmasked.
    mask = chunk["indexer_neg_mask"][0, 0]
    unmasked = (mask == 0).sum(dim=-1).tolist()
    assert unmasked == [(position + row + 1) // 4 for row in range(CHUNK_ROWS)]
    assert unmasked[-1] == position // 4 + 8 and mask.dtype == torch.bfloat16


def test_chunk_emulation_rejects_unaligned_positions(expect_error) -> None:
    for position in (1, 31, 33, -32, True, 2.0):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=RESIDENT_BLOCKS)


# --- constants: host images and the two select tiles under the matmul arithmetic model -----


def test_chunk_constant_rows_carry_the_row_index_templates_and_the_block_lanes() -> None:
    host = qsa_chunk_constant_rows(RESIDENT_BLOCKS)
    assert host["arange32_lanes"].reshape(-1).tolist() == list(range(32))
    assert host["block_start_lanes"].reshape(-1).tolist() == [4 * i for i in range(8)] + [0] * 24
    for name, width in (("row_index_blocks", RESIDENT_BLOCKS), ("row_index_slots", SPARSE_INDEX_CAPACITY)):
        assert host[name].shape == (1, 1, CHUNK_ROWS, width)
        assert torch.equal(host[name], torch.arange(CHUNK_ROWS).reshape(1, 1, CHUNK_ROWS, 1).expand_as(host[name]))
    for name, width in (("arange_blocks_rows", RESIDENT_BLOCKS), ("arange_slots_rows", SPARSE_INDEX_CAPACITY)):
        assert torch.equal(host[name], torch.arange(width).reshape(1, 1, 1, width).expand_as(host[name]))
    assert torch.equal(host["all_ones_rows"], torch.full((1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY), ALL_ONES_U32))
    for name in ("row_index_blocks", "arange_blocks_rows", "row_index_slots", "arange_slots_rows", "all_ones_rows"):
        assert host[name].dtype == torch.int64 and int(host[name].min()) >= 0 and int(host[name].max()) < 2**32
    assert host["block_offsets_rows"].shape == (1, 1, CHUNK_ROWS, TOKEN_BUDGET)
    assert torch.equal(host["block_offsets_rows"][0, 0, 5], torch.arange(4).repeat(BLOCK_TOPK))
    assert host["sentinel_pad_rows"].shape == (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY - TOKEN_BUDGET)
    assert int(host["sentinel_pad_rows"].min()) == ALL_ONES_U32
    # Memory at the resident lane: two [32, 8192], three [32, 2080], one [32, 2048] and one [32, 32] UINT32
    # templates, 3.16 MB.
    total = sum(host[name].numel() * 4 for name in host if host[name].dtype == torch.int64)
    assert total == 4 * 32 * (2 * RESIDENT_BLOCKS + 3 * SPARSE_INDEX_CAPACITY + TOKEN_BUDGET + 32 + 2)


def _select_matmul(select: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """HiFi4 model: exact bf16 x bf16 products, fp32 accumulation, one rounding to bf16."""

    products = select.to(torch.bfloat16).float().unsqueeze(-1) * rows.float().unsqueeze(-3)
    return products.sum(dim=-2).to(torch.bfloat16)


def test_pool_select_is_the_decode_ring_mean_bitwise_and_row_selects_pick_one_row() -> None:
    host = qsa_chunk_constant_rows(RESIDENT_BLOCKS)
    pool, selects = host["pool_select"][0, 0], host["row_selects"][:, 0, 0]
    assert pool.shape == (32, 32) and selects.shape == (8, 32, 32)
    assert torch.equal(pool.to(torch.bfloat16).float(), pool)  # 0.25 is exact in bf16
    assert torch.count_nonzero(pool[8:]) == 0 and (pool[:8] != 0).sum(dim=-1).eq(4).all()
    torch.manual_seed(5)
    raw_keys = (torch.randn(32, 128) * 3).to(torch.bfloat16)
    pooled = _select_matmul(pool, raw_keys)
    for block in range(CHUNK_BLOCKS):
        # The decode ring path: rows 4-31 zero, fp32 sum over 32 rows scaled by 1/4, one bf16 rounding.
        ring = torch.zeros(32, 128)
        ring[:4] = raw_keys[block * 4 : block * 4 + 4].float()
        expected = (ring.sum(dim=0) * (1.0 / COMPRESS_RATIO)).to(torch.bfloat16)
        assert torch.equal(pooled[block].view(torch.int16), expected.view(torch.int16)), block
    assert torch.count_nonzero(pooled[8:].float()) == 0
    rotated = (torch.randn(32, 128) * 2).to(torch.bfloat16)
    for block in range(CHUNK_BLOCKS):
        picked = _select_matmul(selects[block], rotated)
        assert torch.equal(picked[0].view(torch.int16), rotated[block].view(torch.int16)), block
        assert torch.count_nonzero(picked[1:].float()) == 0


def test_chunk_constants_and_inputs_declare_the_device_fields() -> None:
    assert tuple(Qwen38TTNNQSAChunkConstants.__dataclass_fields__) == (
        "allocated_compressed_blocks",
        "rows",
        "arange32_lanes",
        "block_start_lanes",
        "row_index_blocks",
        "arange_blocks_rows",
        "row_index_slots",
        "arange_slots_rows",
        "all_ones_rows",
        "block_offsets_rows",
        "sentinel_pad_rows",
        "pool_select",
        "row_selects",
        "zero_value_half_rows",
        # The slab's mask operands and page offsets (None for the chunk forms).
        "arange_blocks_row",
        "row_index_col",
        "page_offsets",
        # The slab's row index as a row: the block-shared attention kernel's positions (None for the chunk forms).
        "row_index_row",
        # The slab's hoist decision for its block masks (qsa_mask_hoist; None for the chunk forms).
        "hoist_masks",
    )
    # compressed_tile_i32: the 128-row chunk's / the slab's page table (None for the 32-row forms);
    # complete_blocks_col: the slab's mask column (None for the chunk forms); block_masks: the slab's hoisted
    # block masks under the qsa_mask_hoist glue form (empty otherwise).
    # q_positions_row: the slab's query positions P + j as a row (the block-shared attention kernel's input; None
    # for the chunk forms).
    assert tuple(Qwen38TTNNQSAChunkInputs.__dataclass_fields__) == ("rows",) + CHUNK_INPUT_FIELDS + (
        "compressed_tile_i32",
        "complete_blocks_col",
        "block_masks",
        "q_positions_row",
    )
    build = inspect.getsource(Qwen38TTNNQSAChunkConstants.build)
    assert "_upload_uint32(" in build and "replicate_tensor_2d_mesh_mapper(mesh_device)" in build
    assert "layout=ttnn.TILE_LAYOUT" in build and "dtype=ttnn.bfloat16" in build
    assert CHUNK_BLOCKS == 8 and CHUNK_ROWS == 32 and CHUNK_ROWS % COMPRESS_RATIO == 0


# --- derivation: the 1-row op set, P the only broadcast, no host readback ---------------


def test_chunk_derivation_uses_only_exact_integer_ops_and_the_scalar_broadcast() -> None:
    source = inspect.getsource(qsa_module.derive_qsa_chunk_inputs)
    for forbidden in (
        "to_torch",
        ".item()",
        "int(",
        "float(",
        "position %",
        "position //",
        "synchronize",
        "for row in",
    ):
        assert forbidden not in source, forbidden
    allowed = {
        "bitwise_and",
        "bitwise_right_shift",
        "add",
        "typecast",
        "reshape",
        "lt",
        "rsub",
        "multiply",
        "minimum",
        "bitwise_left_shift",
        "subtract",
        "ge",
        "bitwise_or",
    }
    calls = _ttnn_calls(qsa_module.derive_qsa_chunk_inputs)
    assert set(calls) <= allowed
    # The 1-row one-hots (staging row, ring slot) have no chunk form.
    assert "eq" not in calls and "one_hot" not in source and "to_layout" not in calls
    # The scalar P is added to the row-index templates; every other operand pair is same-shape.
    assert "ttnn.add(chunk.row_index_blocks, position_scalar" in source
    assert "ttnn.add(chunk.row_index_slots, position_scalar" in source
    assert "ttnn.lt(chunk.arange_blocks_rows, complete_blocks_rows" in source
    assert "ttnn.multiply(before_lo, chunk.all_ones_rows" in source
    assert "ttnn.multiply(from_hi, chunk.all_ones_rows" in source
    assert "ttnn.bitwise_or(row_tail_fill, row_sentinel_bits" in source
    assert (
        "ttnn.multiply(invalid, INDEXER_MASK_VALUE" in source and "ttnn.minimum(complete_blocks, BLOCK_TOPK" in source
    )
    for compare in ("ttnn.lt(", "ttnn.ge("):
        assert all("dtype=u32" in line for line in source.splitlines() if compare in line)
    # Casts: the block-validity mask to bf16, the eight INT32 block indices (32-row forms) and the 128-row chunk's
    # INT32 compressed tile index (its page table).
    assert source.count("ttnn.typecast(") == 3
    assert "ttnn.typecast(valid_bits, ttnn.bfloat16" in source and "ttnn.typecast(shifted, ttnn.int32" in source
    assert "ttnn.typecast(tile_index, ttnn.int32" in source and "bitwise_right_shift(position_scalar, 7" in source
    # The 1-row derivation is untouched.
    single = inspect.getsource(qsa_module.derive_qsa_position_inputs)
    assert "chunk" not in single and "CHUNK" not in single


def test_rope_chunk_rows_keep_all_32_rows_and_the_one_row_lookup_keeps_its_view() -> None:
    chunk_rows = inspect.getsource(model_module.Qwen38TTNNRoPETable.rows_chunk)
    assert (
        chunk_rows.count("self._lookup_tile(") == 1 and "for label, table_indices, table, row_count in (" in chunk_rows
    )
    assert "ttnn.Shape(" not in chunk_rows and "ttnn.slice(" not in chunk_rows and "ttnn.embedding(" not in chunk_rows
    tile = inspect.getsource(model_module.Qwen38TTNNRoPETable._lookup_tile)
    assert tile.count("ttnn.embedding(") == 1 and "_padded_shape(rows) != padded" in tile
    assert "return Qwen38TTNNRoPEInputs(None, *looked_up)" in chunk_rows
    one_row = inspect.getsource(model_module.Qwen38TTNNRoPETable._lookup)
    assert "ttnn.reshape(rows, ttnn.Shape((1, 1, 1, QSA_ROPE_DIM)), ttnn.Shape(padded))" in one_row
    assert "chunk" not in inspect.getsource(model_module.Qwen38TTNNRoPETable.rows)


def test_device_position_advance_by_mirrors_advance(expect_error) -> None:
    advance = inspect.getsource(contracts_module.Qwen38TTNNDevicePosition.advance)
    advance_by = inspect.getsource(contracts_module.Qwen38TTNNDevicePosition.advance_by)
    assert "ttnn.add(self.scalar, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)" in advance
    assert "ttnn.add(self.scalar, count, memory_config=ttnn.DRAM_MEMORY_CONFIG)" in advance_by
    for body in (advance, advance_by):
        assert "ttnn.copy(advanced, self.scalar)" in body and "ttnn.deallocate(advanced)" in body
    assert "count <= 0" in advance_by
    shell = object.__new__(contracts_module.Qwen38TTNNDevicePosition)
    for count in (0, -1, True, 2.0, "32"):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            shell.advance_by(count)


# --- pinned runtime: the chunk's multi-row admissions -----------------------------------


def _cpp(relative: str) -> str:
    return (REPOSITORY / relative).read_text(encoding="utf-8")


def test_pinned_indexer_window_admits_a_32_row_query_against_the_generic_cache() -> None:
    device = _cpp("ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_device_operation.cpp")
    # The chunk must BEGIN inside the allocated k length; its causal window may end past kv_len (pad query rows).
    assert "attrs.chunk_start_idx < T," in device
    # The generic cache carries one extra tile past the resident blocks: chunk_start = blocks, Sq = 32 fit exactly.
    chunk_start, window_rows = RESIDENT_BLOCKS, ttnn.TILE_SIZE
    assert chunk_start < RESIDENT_BLOCKS + window_rows
    assert chunk_start + CHUNK_ROWS <= RESIDENT_BLOCKS + window_rows
    assert CHUNK_ROWS == window_rows  # a wider chunk would need a wider cache window
    nanobind = _cpp("ttnn/cpp/ttnn/operations/experimental/indexer_score/indexer_score_nanobind.cpp")
    # Current upstream uses [B, 1, Sq, Hi]; TP4 gives this model one local index head.
    # Its [1, 1, Sq, 1] gate therefore has the same layout under both contracts.
    assert "weights: [B, 1, Sq, Hi] bf16 tiled learned per-head gates" in nanobind
    assert "q: [B, Hi, Sq, D] bf16 or bfp8_b tiled" in nanobind


def test_pinned_sparse_sdpa_and_topk_admit_the_chunk_rows() -> None:
    device = _cpp("ttnn/cpp/ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.cpp")
    assert "sparse_sdpa q/indices must not be padded" in device
    assert "kv must be [B,1,T,{}] (got {})" in device
    topk = _cpp(
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp"
    )
    assert "attrs.k > 0 && attrs.k <= max_supported_k && attrs.k % 16 == 0," in topk
    assert BLOCK_TOPK % 16 == 0
    assert INDEXER_MASK_VALUE < -3e38


# --- the chunk body: the decode's op walk on 32-row operands, no host ints, the 1-row body untouched -----

CHUNK_BODY = (
    "forward_chunk_generic",
    "_all_gather_hidden_rows",
    "_index_projection_rows",
    "_write_compressed_index_chunk",
    "_score_blocks_chunk",
    "_materialize_rows_chunk",
    "_main_projection_rows",
    "_write_packed_kv_chunk",
    "_sparse_value_attention_rows",
    "_project_output_rows",
)


def _method_source(name: str) -> str:
    return inspect.getsource(getattr(qsa_module.Qwen38TTNNQSA, name))


def _ttnn_op_walk(name: str) -> list[str]:
    tree = ast.parse(_dedent(getattr(qsa_module.Qwen38TTNNQSA, name)))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = ast.unparse(node.func.value)
            if owner.startswith("ttnn") and node.func.attr not in ("deallocate", "Shape"):
                calls.append((node.lineno, node.col_offset, node.func.attr))
    return [attr for _line, _col, attr in sorted(calls)]


def test_chunk_body_has_no_host_ints_no_host_io_and_the_decode_order() -> None:
    for name in CHUNK_BODY:
        source = _method_source(name)
        for forbidden in ("from_torch", "to_torch", ".item()", "copy_host_to_device_tensor", "synchronize", ".shape["):
            assert forbidden not in source, (name, forbidden)
        # Every shape is a module constant: no per-call int reaches an op.
        for argument in ast.parse(_dedent(getattr(qsa_module.Qwen38TTNNQSA, name))).body[0].args.args:
            annotation = ast.unparse(argument.annotation) if argument.annotation is not None else ""
            assert annotation != "int", (name, argument.arg)
    body = _method_source("forward_chunk_generic")
    order = (
        "self._all_gather_hidden_rows(hidden_rows, constants)",
        "self._hidden_row_tiles(full_hidden, constants)",
        "self._index_projection_rows(full_hidden, hidden_tiles, cos, sin, constants)",
        "self._write_compressed_index_chunk(",
        "self._score_blocks_chunk(index_query, state, chunk)",
        "self._materialize_rows_chunk(masked_scores, chunk, constants)",
        "self._main_projection_rows(full_hidden, hidden_tiles, cos, sin, constants)",
        "self._write_packed_kv_chunk(state, chunk_state, key, value, chunk)",
        "self._sparse_value_attention_rows(query, gate, sparse_indices, state, constants)",
        "self._project_output_rows(local_attention, full_hidden, constants)",
    )
    positions = [body.index(fragment) for fragment in order]
    assert positions == sorted(positions)
    decode = inspect.getsource(qsa_module.Qwen38TTNNQSA.forward_decode_generic)
    decode_order = [
        line.strip().split("=", 1)[-1].strip() if "=" in line else line.strip()
        for line in decode.splitlines()
        if "self._" in line and "validate" not in line
    ]
    # The long chunk's hidden row tiles (moved once for the five linears), the slab's block selection (its
    # scoring + top-k per 512-row block) and the slab's block-shared attention branch (QWEN38_FUSED=sparse_sdpa_tiled:
    # its admission on the shapes and the kernel call) are chunk-only stages.
    chunk_order = [
        line.strip().split("=", 1)[-1].strip() if "=" in line else line.strip()
        for line in body.splitlines()
        if "self._" in line
        and "validate" not in line
        and "_hidden_row_tiles" not in line
        and "_sparse_indices_slab" not in line
        and "_slab_attention_admits" not in line
        and "_block_shared_attention_rows" not in line
    ]

    # The same nine stages in the same order (the chunk names end in _rows / _chunk).
    def strip(names):
        return [
            name.split("(")[0].replace("_rows", "").replace("_row", "").replace("_chunk", "").replace("_generic", "")
            for name in names
        ]

    assert strip(chunk_order) == strip(decode_order)


def test_chunk_body_replaces_the_one_hots_with_whole_slab_writes_and_selection_matmuls() -> None:
    compressed = _ttnn_op_walk("_write_compressed_index_chunk")
    # Call sites: the pool select, then the 128-row chunk's one-tile write (the cache viewed as tile blocks, one
    # paged_fill_cache), then the 32-row form's per-block row pick / view / reshard / write inside the block loop.
    assert compressed == [
        "copy",
        "matmul",
        "rms_norm",
        "view",
        "paged_fill_cache",
        "matmul",
        "reshape",
        "to_memory_config",
        "paged_update_cache",
    ]
    assert "for block in range(len(constants.row_selects)):" in _method_source("_write_compressed_index_chunk")
    assert "if constants.rows == LONG_CHUNK_ROWS or is_slab_rows(constants.rows):" in _method_source(
        "_write_compressed_index_chunk"
    )
    assert "multiply" not in compressed and "sum" not in compressed  # no ring one-hots, no scaled sum
    kv = _ttnn_op_walk("_write_packed_kv_chunk")
    assert kv == ["concat", "copy", "to_layout", "update_padded_kv_cache"]
    score = _ttnn_op_walk("_score_blocks_chunk")
    # One indexer call per 32-row query tile (the query tile is the rows themselves at 32 rows, a row-tile slice at
    # 128), the score tiles concatenated, then the all-reduce and the per-row mask on all rows.
    assert score == ["slice", "indexer_score_dsa", "slice", "concat", "all_reduce", "add"]
    materialize = _ttnn_op_walk("_materialize_rows_chunk")
    assert materialize == [
        "topk_large_indices",
        "bitwise_left_shift",
        "repeat_interleave",
        "add",
        "concat",
        "bitwise_and",
        "bitwise_or",
    ]
    projection = _ttnn_op_walk("_main_projection_rows")
    assert _method_source("_main_projection_rows").count("self._linear_rows(") == 3
    assert _ttnn_op_walk("_linear_rows") == ["linear", "to_memory_config", "linear", "to_memory_config", "concat"]
    assert projection.count("linear") == 0 and projection.count("slice") == 2 and projection.count("concat") == 2
    assert "reshape" not in projection and "permute" not in projection  # tile-aligned head slices, no padded reshape
    attention = _ttnn_op_walk("_sparse_value_attention_rows")
    assert attention.count("sparse_sdpa") == 1 and attention.count("concat") == 2 and "reshape" not in attention
    source = _method_source("_write_compressed_index_chunk")
    assert "constants.pool_select" in source and "constants.row_selects[block]" in source
    assert "ttnn.reshape(picked, one_row, tile)" in source
    assert "update_idxs_tensor=chunk.block_index_i32[block]" in source
    assert "chunk.kv_block_start" in _method_source("_write_packed_kv_chunk")
    assert "constants.block_offsets_rows" in _method_source("_materialize_rows_chunk")
    assert "constants.sentinel_pad_rows" in _method_source("_materialize_rows_chunk")
    assert "constants.zero_value_half_rows" in _method_source("_sparse_value_attention_rows")


def test_chunk_state_is_allocated_beside_the_generic_state_and_released_by_epoch() -> None:
    assert tuple(qsa_module.Qwen38TTNNQSAChunkState.__dataclass_fields__) == (
        "layer_index",
        "epoch",
        "rows",
        "kept_kv",
        "kept_raw",
    )
    allocate = inspect.getsource(qsa_module.Qwen38TTNNQSA.allocate_chunk_state)
    assert "self._allocate_pair_grouped((1, 1, rows, 2 * HEAD_DIM), layout=ttnn.TILE_LAYOUT)" in allocate
    assert "self._allocate_replicated_tile_zeros((1, 1, rows, INDEX_HEAD_DIM))" in allocate
    assert "self._live_generic_epochs.add(epoch)" in allocate
    release = inspect.getsource(qsa_module.Qwen38TTNNQSA.release_chunk_state)
    assert "self._live_generic_epochs.remove(state.epoch)" in release
    # The 1-row generic body and its helpers are not edited by the chunk path.
    for name in (
        "forward_decode_generic",
        "_write_compressed_index_generic",
        "_score_blocks_generic",
        "_materialize_row_generic",
        "_write_packed_kv_generic",
        "_project_output",
        "_main_projection",
        "_index_projection",
        "_sparse_value_attention",
        "_all_gather_hidden",
    ):
        source = inspect.getsource(getattr(qsa_module.Qwen38TTNNQSA, name))
        assert "CHUNK_ROWS" not in source and "chunk_state" not in source and "Chunk" not in source, name


# --- P > 0: the device derivation after k chunk advances, on an exact-integer fake ------------
# The stage-1 4x p150 micro-test ran derive_qsa_chunk_inputs against the emulation at P0 in {0, 2016, 2048, 8160, 32736}
# at the layer level.  Here the same chain runs on a torch model of the integer ops after k in-trace ``advance_by(32)``
# calls on the device position, so every P % 32 == 0 the model reaches by replaying (block index crossing a tile at
# P = 128, the top-k regime at 2048, the last chunk of the resident cache) is pinned field for field.

ADVANCED_CHUNKS = (1, 2, 4, 63, 64, 127, 254, 255, 1023)


class _Tag:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


U32, I32, BF16_TAG = _Tag("uint32"), _Tag("int32"), _Tag("bfloat16")
ROW_MAJOR_TAG, TILE_TAG = _Tag("ROW_MAJOR_LAYOUT"), _Tag("TILE_LAYOUT")


class _IntTensor:
    """One device tensor of the integer fake: int64 storage for UINT32/INT32, bf16 for the mask; tagged dtype/layout."""

    _ids = iter(range(1, 10**9))

    def __init__(self, value: torch.Tensor, dtype: _Tag, layout: _Tag) -> None:
        self.value, self.dtype, self.layout = value, dtype, layout
        self.tensor_id = next(self._ids)
        self.alive = True

    @property
    def shape(self):
        return tuple(self.value.shape)

    def _read(self) -> torch.Tensor:
        assert self.alive, "read of a deallocated tensor"
        return self.value


def _operand(x):
    return x._read() if isinstance(x, _IntTensor) else x


def _int_op(torch_op):
    def op(a, b, *, memory_config=None, dtype=None):
        result = torch_op(a._read(), _operand(b))
        tag = dtype or a.dtype
        if tag is BF16_TAG:
            result = result.to(torch.bfloat16)
        elif result.dtype != torch.int64:
            result = result.to(torch.int64)
        return _IntTensor(result, tag, a.layout)

    return op


def _integer_fake() -> SimpleNamespace:
    def typecast(tensor, dtype, *, memory_config=None):
        value = tensor._read()
        return _IntTensor(value.to(torch.bfloat16) if dtype is BF16_TAG else value.clone(), dtype, tensor.layout)

    def from_torch(host, *, dtype, layout, device=None, memory_config=None, mesh_mapper=None):
        return _IntTensor(host.to(torch.int64) if dtype is U32 else host.to(torch.bfloat16), dtype, layout)

    def copy(source, target):
        target._read().copy_(source._read())
        return target

    def deallocate(tensor):
        assert tensor.alive, "double deallocation"
        tensor.alive = False

    return SimpleNamespace(
        uint32=U32,
        int32=I32,
        bfloat16=BF16_TAG,
        ROW_MAJOR_LAYOUT=ROW_MAJOR_TAG,
        TILE_LAYOUT=TILE_TAG,
        TILE_SIZE=32,
        DRAM_MEMORY_CONFIG="DRAM",
        bitwise_and=_int_op(torch.bitwise_and),
        bitwise_or=_int_op(torch.bitwise_or),
        bitwise_right_shift=_int_op(torch.bitwise_right_shift),
        bitwise_left_shift=_int_op(torch.bitwise_left_shift),
        add=_int_op(torch.add),
        subtract=_int_op(torch.sub),
        multiply=_int_op(torch.mul),
        minimum=_int_op(lambda a, b: torch.clamp(a, max=b)),
        lt=_int_op(torch.lt),
        ge=_int_op(torch.ge),
        rsub=_int_op(lambda a, b: b - a),
        typecast=typecast,
        reshape=lambda t, shape, memory_config=None: _IntTensor(t._read().reshape(tuple(shape)), t.dtype, t.layout),
        from_torch=from_torch,
        copy_host_to_device_tensor=copy,
        copy=copy,
        deallocate=deallocate,
    )


class _IntContract:
    def validate_tensor(self, tensor, *, placement, shard_dim=None, require_device=True) -> None:
        assert tensor.alive


def _u32(values: torch.Tensor) -> _IntTensor:
    return _IntTensor(values.to(torch.int64).clone(), U32, ROW_MAJOR_TAG)


@pytest.fixture
def integer_fake(monkeypatch):
    fake = _integer_fake()
    monkeypatch.setattr(qsa_module, "ttnn", fake)
    monkeypatch.setattr(contracts_module, "ttnn", fake)
    monkeypatch.setattr(contracts_module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate")
    return fake


# The 64k/128k/256k options: the chunk derivation at the long-context micro-test's positions (4x p150 2026-09-03:
# every op admitted at allocated_context 65536 at decode/chunk positions 32768 and 65504, and at 131072 and 262144)
# against the emulation at each context's top chunk (position allocated_context - 32) and its 32k/64k/128k
# boundaries, and the 32k default's top chunk unchanged.
CONTEXT_POSITIONS = (
    (32_768, 32_736),
    (65_536, 32_768),
    (65_536, 65_504),
    (131_072, 65_536),
    (131_072, 131_040),
    (262_144, 131_072),
    (262_144, 262_112),
)


@pytest.mark.parametrize(("allocated_context", "position"), CONTEXT_POSITIONS)
def test_chunk_inputs_at_the_long_context_positions_are_the_emulation(
    expect_error, integer_fake, allocated_context: int, position: int
):
    blocks = allocated_context // COMPRESS_RATIO
    scalar = _u32(torch.full((1, 1, 1, 1), position))
    host = qsa_chunk_constant_rows(blocks)
    constants = SimpleNamespace(
        allocated_compressed_blocks=blocks,
        high27_mask=_u32(torch.full((1, 1, 1, 1), qsa_module.KV_BLOCK_START_MASK)),
    )
    chunk = SimpleNamespace(
        allocated_compressed_blocks=blocks,
        rows=CHUNK_ROWS,
        **{name: _u32(host[name]) for name in CHUNK_UINT32_TEMPLATES},
    )
    inputs = qsa_module.derive_qsa_chunk_inputs(scalar, constants, chunk)
    expected = emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=blocks)
    assert inputs.kv_block_start._read().tolist() == expected["kv_block_start"].tolist() == [[[[position]]]]
    assert [t._read().tolist() for t in inputs.block_index_i32] == [e.tolist() for e in expected["block_index_i32"]]
    assert inputs.indexer_neg_mask.shape == (1, 1, CHUNK_ROWS, blocks)
    assert torch.equal(inputs.indexer_neg_mask._read(), expected["indexer_neg_mask"])
    assert torch.equal(inputs.row_keep_bits._read(), expected["row_keep_bits"])
    assert torch.equal(inputs.row_fill._read(), expected["row_fill"])
    # The last row's block count: the chunk completes blocks up to (position + 32) / 4, inside the allocation.
    unmasked = (expected["indexer_neg_mask"][0, 0] == 0).sum(dim=-1).tolist()
    assert unmasked[-1] == position // 4 + 8 <= blocks
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        qsa_module.derive_qsa_chunk_inputs(
            scalar, SimpleNamespace(allocated_compressed_blocks=blocks // 2, high27_mask=constants.high27_mask), chunk
        )


@pytest.mark.parametrize("chunks", ADVANCED_CHUNKS)
def test_chunk_inputs_after_advancing_the_device_position_are_the_emulation(integer_fake, chunks: int) -> None:
    position = contracts_module.Qwen38TTNNDevicePosition(
        scalar=_u32(torch.zeros(1, 1, 1, 1)),
        ones_row=_u32(torch.ones(1, 1, 1, CHUNK_ROWS)),
        block_start_mask_row=_u32(torch.full((1, 1, 1, CHUNK_ROWS), contracts_module.BLOCK_START_LANE_MASK)),
        mesh_device="mesh",
        mesh_contract=_IntContract(),
    )
    for _ in range(chunks):
        position.advance_by(CHUNK_ROWS)  # the chunk body's last op, once per replay
    expected_position = CHUNK_ROWS * chunks
    assert position.scalar._read().reshape(-1).tolist() == [expected_position]

    host = qsa_chunk_constant_rows(RESIDENT_BLOCKS)
    constants = SimpleNamespace(
        allocated_compressed_blocks=RESIDENT_BLOCKS,
        high27_mask=_u32(torch.full((1, 1, 1, 1), qsa_module.KV_BLOCK_START_MASK)),
    )
    chunk = SimpleNamespace(
        allocated_compressed_blocks=RESIDENT_BLOCKS,
        rows=CHUNK_ROWS,
        **{name: _u32(host[name]) for name in CHUNK_UINT32_TEMPLATES},
    )
    inputs = qsa_module.derive_qsa_chunk_inputs(position.scalar, constants, chunk)
    expected = emulate_qsa_chunk_inputs(expected_position, allocated_compressed_blocks=RESIDENT_BLOCKS)
    assert inputs.kv_block_start._read().tolist() == expected["kv_block_start"].tolist() == [[[[expected_position]]]]
    assert [t.dtype for t in inputs.block_index_i32] == [I32] * CHUNK_BLOCKS
    assert [t._read().tolist() for t in inputs.block_index_i32] == [e.tolist() for e in expected["block_index_i32"]]
    assert inputs.indexer_neg_mask.dtype is BF16_TAG and inputs.indexer_neg_mask.shape == (
        1,
        1,
        CHUNK_ROWS,
        RESIDENT_BLOCKS,
    )
    assert torch.equal(inputs.indexer_neg_mask._read(), expected["indexer_neg_mask"])
    for name in ("row_keep_bits", "row_fill"):
        actual = getattr(inputs, name)
        assert actual.dtype is U32 and actual.shape == (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY)
        assert torch.equal(actual._read(), expected[name]), name
    # The RoPE index rows of the same replay: lane j = P + j; the block-start lanes P + 4i (i < 8), P past them.
    index_row = position.index_row()
    index_rows = integer_fake.add(index_row, chunk.arange32_lanes)
    block_start_rows = integer_fake.add(index_row, chunk.block_start_lanes)
    assert index_rows._read().reshape(-1).tolist() == [expected_position + j for j in range(CHUNK_ROWS)]
    assert block_start_rows._read().reshape(-1).tolist() == [
        expected_position + (4 * i if i < CHUNK_BLOCKS else 0) for i in range(CHUNK_ROWS)
    ]
    # The hand-off's host reset lands any position, as finish_prefill does after the padded tail.
    position.reset(expected_position - 5)
    assert position.scalar._read().reshape(-1).tolist() == [expected_position - 5]
    assert inputs.kv_block_start.alive and inputs.row_fill.alive


CHUNK_UINT32_TEMPLATES = (
    "arange32_lanes",
    "block_start_lanes",
    "row_index_blocks",
    "arange_blocks_rows",
    "row_index_slots",
    "arange_slots_rows",
    "all_ones_rows",
)
