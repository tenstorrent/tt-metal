# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static and torch-emulation tests for the position-generic QSA decode path.

The device path derives every position-dependent quantity from a UINT32
position scalar with exact integer ops; these tests pin the torch emulation
against the selection geometry, the exact selects and reductions the body
relies on, the fixed op sequence of the generic body, and the pinned-runtime
C++ contracts the body depends on.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    ALL_ONES_U32,
    BLOCK_TOPK,
    CACHE_WRITE_ROWS,
    COMPRESS_RATIO,
    INDEX_HEAD_DIM,
    INDEXER_MASK_VALUE,
    KV_BLOCK_START_MASK,
    MASKED_INDEX,
    MAX_CONTEXT,
    SPARSE_INDEX_CAPACITY,
    TOKEN_BUDGET,
    Qwen38TTNNQSA,
    Qwen38TTNNQSAPositionConstants,
    Qwen38TTNNQSAPositionInputs,
    emulate_qsa_position_inputs,
    qsa_row_constants,
    qsa_selection_geometry,
)

RESIDENT_BLOCKS = 32768 // COMPRESS_RATIO
REPOSITORY = Path(inspect.getfile(Qwen38TTNNQSA)).resolve().parents[5]
POSITION_INPUT_FIELDS = (
    "kv_block_start",
    "kv_row_hit",
    "kv_row_keep",
    "ring_hit",
    "ring_keep",
    "block_index_i32",
    "indexer_neg_mask",
    "row_keep_bits",
    "row_fill",
)
GENERIC_BODY = (
    Qwen38TTNNQSA.forward_decode_generic,
    Qwen38TTNNQSA._write_compressed_index_generic,
    Qwen38TTNNQSA._score_blocks_generic,
    Qwen38TTNNQSA._materialize_row_generic,
    Qwen38TTNNQSA._write_packed_kv_generic,
    Qwen38TTNNQSA._project_output,
)


def _emulate(position: int) -> dict[str, torch.Tensor]:
    return emulate_qsa_position_inputs(position, allocated_compressed_blocks=RESIDENT_BLOCKS)


def _function_node(function) -> ast.FunctionDef:
    lines = inspect.getsource(function).splitlines(keepends=True)
    indent = len(lines[0]) - len(lines[0].lstrip())
    tree = ast.parse("".join(line[indent:] if line.strip() else line for line in lines))
    (node,) = [item for item in tree.body if isinstance(item, ast.FunctionDef)]
    return node


def _only_fail_closed_branches(function) -> None:
    """Every if/for/while in ``function`` is a validation branch that raises; none depends on the position.

    Integer ``%`` and ``//`` are allowed only between module constants (``HIDDEN_SIZE // TP_SIZE``).
    """

    node = _function_node(function)
    for binop in [n for n in ast.walk(node) if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mod, ast.FloorDiv))]:
        for name in [n for n in ast.walk(binop) if isinstance(n, ast.Name)]:
            assert name.id.isupper(), ast.dump(binop)
    for branch in [n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While))]:
        assert isinstance(branch, ast.If)
        assert not [n for n in ast.walk(branch.test) if isinstance(n, ast.Name) and n.id == "position"]
        assert any(isinstance(item, ast.Raise) for item in branch.body)


def _ttnn_calls(function) -> list[str]:
    return re.findall(
        r"\bttnn\.(?:experimental\.(?:deepseek_prefill\.)?|transformer\.)?(\w+)\(", inspect.getsource(function)
    )


def _cpp(relative: str) -> str:
    return (REPOSITORY / relative).read_text(encoding="utf-8")


# --- torch emulation ---------------------------------------------------------


def test_emulation_matches_selection_geometry_for_every_position() -> None:
    """Rows 4-17 of the position table against qsa_selection_geometry(P + 1), exhaustively."""

    slots = torch.arange(SPARSE_INDEX_CAPACITY, dtype=torch.int64)
    blocks = torch.arange(RESIDENT_BLOCKS)
    rows = torch.arange(CACHE_WRITE_ROWS)
    all_ones = torch.full_like(slots, ALL_ONES_U32)
    zeros = torch.zeros_like(slots)
    for position in range(0, 40000):
        emulated = _emulate(position)
        assert tuple(emulated) == POSITION_INPUT_FIELDS
        geometry = qsa_selection_geometry(position + 1)
        lo = COMPRESS_RATIO * geometry.selected_blocks
        hi = lo + geometry.tail_count
        assert lo == geometry.complete_token_count
        assert emulated["kv_block_start"].dtype == torch.int64
        assert emulated["kv_block_start"].item() == position & KV_BLOCK_START_MASK == position - position % 32
        assert emulated["block_index_i32"].dtype == torch.int32
        assert emulated["block_index_i32"].shape == (1,)
        assert emulated["block_index_i32"].item() == position // COMPRESS_RATIO
        assert torch.equal(emulated["row_keep_bits"].flatten(), torch.where(slots < lo, all_ones, zeros))
        # One row: the tail ids in [lo, hi), the sentinel from hi on (disjoint slots, so one OR carries both).
        tail = (slots >= lo) & (slots < hi)
        assert torch.equal(
            emulated["row_fill"].flatten(),
            torch.where(tail, geometry.tail_start + slots - lo, torch.where(slots >= hi, all_ones, zeros)),
        )
        mask = emulated["indexer_neg_mask"]
        assert mask.dtype == torch.bfloat16 and mask.shape == (1, 1, 1, RESIDENT_BLOCKS)
        visible = blocks < geometry.complete_blocks
        assert torch.equal(mask.flatten() == 0, visible)
        assert torch.all(mask.flatten()[~visible] == INDEXER_MASK_VALUE)
        for name, modulus in (("kv_row", CACHE_WRITE_ROWS), ("ring", COMPRESS_RATIO)):
            hit = emulated[f"{name}_hit"]
            keep = emulated[f"{name}_keep"]
            assert hit.dtype == keep.dtype == torch.bfloat16
            assert hit.shape == keep.shape == (1, 1, CACHE_WRITE_ROWS, 1)
            assert torch.equal(hit.flatten(), (rows == position % modulus).to(torch.bfloat16))
            assert torch.equal(keep.flatten(), (rows != position % modulus).to(torch.bfloat16))


def test_emulation_tail_ids_and_ranges_stay_inside_uint32_and_the_row() -> None:
    for position in (0, 1, 2, 3, 4, 31, 32, 2047, 2048, 2051, 2052, 32767, MAX_CONTEXT - 1):
        emulated = _emulate(position)
        geometry = qsa_selection_geometry(position + 1)
        lo = geometry.complete_token_count
        hi = lo + geometry.tail_count
        row_fill = emulated["row_fill"].flatten()
        assert row_fill[lo:hi].tolist() == list(range(geometry.tail_start, position + 1))
        assert not row_fill[:lo].any() and bool((row_fill[hi:] == ALL_ONES_U32).all())
        assert int(emulated["row_keep_bits"].max()) <= ALL_ONES_U32 and int(row_fill[:hi].max()) < 2**32
        assert 1 <= hi <= TOKEN_BUDGET + COMPRESS_RATIO - 1 < SPARSE_INDEX_CAPACITY
        assert int((row_fill == ALL_ONES_U32).sum()) == SPARSE_INDEX_CAPACITY - hi


@pytest.mark.parametrize("position", (32_768, 65_503, 65_504, 65_535))
def test_emulation_at_the_64k_positions_masks_every_block_past_the_context(position: int) -> None:
    """The 64k option's positions (the micro-test's 32768 and 65504, the top of the allocation): the one-row
    emulation at 16,384 blocks, and at 8,192 blocks the same position is refused only by the allocation, not the
    arithmetic (the mask has no room for it)."""

    blocks = 65_536 // COMPRESS_RATIO
    emulated = emulate_qsa_position_inputs(position, allocated_compressed_blocks=blocks)
    geometry = qsa_selection_geometry(position + 1)
    mask = emulated["indexer_neg_mask"]
    assert mask.shape == (1, 1, 1, blocks) and int((mask == 0).sum()) == geometry.complete_blocks
    assert geometry.complete_blocks == (position + 1) // COMPRESS_RATIO <= blocks
    assert emulated["block_index_i32"].tolist() == [position // COMPRESS_RATIO]
    assert emulated["kv_block_start"].item() == position & KV_BLOCK_START_MASK
    assert emulated["kv_row_hit"].reshape(-1).tolist().index(1.0) == position % 32
    row_fill = emulated["row_fill"].flatten()
    lo, hi = geometry.complete_token_count, geometry.complete_token_count + geometry.tail_count
    assert lo == TOKEN_BUDGET and row_fill[lo:hi].tolist() == list(range(geometry.tail_start, position + 1))
    if position >= 32_768:
        small = emulate_qsa_position_inputs(position, allocated_compressed_blocks=32_768 // COMPRESS_RATIO)
        assert int((small["indexer_neg_mask"] == 0).sum()) == 32_768 // COMPRESS_RATIO  # every resident block visible


@pytest.mark.parametrize(
    ("allocated_context", "position"),
    ((131_072, 65_536), (131_072, 131_071), (262_144, 131_072), (262_144, 262_143)),
)
def test_emulation_at_the_128k_and_256k_positions_stays_inside_the_allocation(
    expect_error, allocated_context: int, position: int
) -> None:
    """The 128k/256k options' positions (each context's 64k/128k boundary and its last position, 131,071 and
    262,143 = the model's last native position): the one-row emulation at 32,768 / 65,536 blocks."""

    blocks = allocated_context // COMPRESS_RATIO
    emulated = emulate_qsa_position_inputs(position, allocated_compressed_blocks=blocks)
    geometry = qsa_selection_geometry(position + 1)
    mask = emulated["indexer_neg_mask"]
    assert mask.shape == (1, 1, 1, blocks) and int((mask == 0).sum()) == geometry.complete_blocks
    assert geometry.complete_blocks == (position + 1) // COMPRESS_RATIO <= blocks
    assert emulated["block_index_i32"].tolist() == [position // COMPRESS_RATIO]
    assert emulated["kv_block_start"].item() == position & KV_BLOCK_START_MASK
    assert emulated["kv_row_hit"].reshape(-1).tolist().index(1.0) == position % 32
    row_fill = emulated["row_fill"].flatten()
    lo, hi = geometry.complete_token_count, geometry.complete_token_count + geometry.tail_count
    assert lo == TOKEN_BUDGET and row_fill[lo:hi].tolist() == list(range(geometry.tail_start, position + 1))
    if position + 1 == allocated_context:  # the last position completes every block of the allocation
        assert geometry.complete_blocks == blocks
    with expect_error(ValueError):  # allow-pytest.raises: the position past the model's native context
        emulate_qsa_position_inputs(MAX_CONTEXT, allocated_compressed_blocks=blocks)


def test_emulation_rejects_invalid_position_and_capacity(expect_error) -> None:
    for position in (True, -1, MAX_CONTEXT, 1.0):
        with expect_error(ValueError):
            emulate_qsa_position_inputs(position, allocated_compressed_blocks=RESIDENT_BLOCKS)
    with expect_error(ValueError):
        emulate_qsa_position_inputs(0, allocated_compressed_blocks=RESIDENT_BLOCKS + 1)


def test_emulation_indexer_mask_is_positive_zero_on_visible_blocks() -> None:
    """Visible blocks are +0.0 (bf16 0x0000, the device multiply's zero), hidden ones INDEXER_MASK_VALUE."""

    for position in (0, 3, 7, 4095, MAX_CONTEXT - 1):
        mask = _emulate(position)["indexer_neg_mask"].flatten()
        visible = (position + 1) // COMPRESS_RATIO
        assert mask.dtype == torch.bfloat16
        assert torch.all(mask[:visible].view(torch.int16) == 0), position  # +0.0, never -0.0 (0x8000)
        assert torch.all(mask[visible:] == INDEXER_MASK_VALUE), position
    mask = _emulate(7)["indexer_neg_mask"].flatten()
    scores = torch.randn(RESIDENT_BLOCKS).abs().to(torch.bfloat16)
    masked = scores + mask
    assert torch.equal(masked[:2].view(torch.int16), scores[:2].view(torch.int16))
    assert torch.all(masked[2:] == INDEXER_MASK_VALUE)
    assert not torch.isnan(masked).any() and not torch.isinf(masked).any()
    assert INDEXER_MASK_VALUE == -torch.finfo(torch.bfloat16).max


# --- exactness of the selects the body relies on -----------------------------


def _sparse_row_from_template(block_ids: torch.Tensor, emulated: dict[str, torch.Tensor]) -> torch.Tensor:
    expanded = (block_ids * COMPRESS_RATIO).repeat_interleave(COMPRESS_RATIO) + torch.arange(COMPRESS_RATIO).repeat(
        BLOCK_TOPK
    )
    template = torch.cat([expanded, torch.full((ttnn.TILE_SIZE,), MASKED_INDEX, dtype=torch.int64)])
    kept = template & emulated["row_keep_bits"].flatten()
    return kept | emulated["row_fill"].flatten()


def test_sparse_row_template_reproduces_the_materialized_selection_layout() -> None:
    generator = torch.Generator().manual_seed(2026)
    for position in range(0, 4200):
        geometry = qsa_selection_geometry(position + 1)
        selected = torch.randperm(max(geometry.complete_blocks, 1), generator=generator)[: geometry.selected_blocks]
        stale = torch.randint(0, RESIDENT_BLOCKS, (BLOCK_TOPK - geometry.selected_blocks,), generator=generator)
        block_ids = torch.cat([selected, stale])
        row = _sparse_row_from_template(block_ids, _emulate(position))
        expanded = (selected * COMPRESS_RATIO).repeat_interleave(COMPRESS_RATIO) + torch.arange(COMPRESS_RATIO).repeat(
            geometry.selected_blocks
        )
        valid = torch.cat([expanded, torch.arange(geometry.tail_start, position + 1)])
        expected = torch.cat(
            [valid, torch.full((SPARSE_INDEX_CAPACITY - len(valid),), MASKED_INDEX, dtype=torch.int64)]
        )
        assert torch.equal(row, expected), position
        assert 1 <= len(valid) <= TOKEN_BUDGET + COMPRESS_RATIO - 1
        sentinels = row == MASKED_INDEX
        assert not sentinels[: len(valid)].any() and sentinels[len(valid) :].all()
        if position >= 2047:
            assert geometry.selected_blocks == BLOCK_TOPK


def test_staging_one_hot_select_equals_slice_concat_bitwise() -> None:
    generator = torch.Generator().manual_seed(7)
    staging = torch.randn(CACHE_WRITE_ROWS, 512, generator=generator).to(torch.bfloat16)
    packed = torch.randn(1, 512, generator=generator).to(torch.bfloat16)
    for row in range(CACHE_WRITE_ROWS):
        emulated = _emulate(row)
        hit = emulated["kv_row_hit"].reshape(CACHE_WRITE_ROWS, 1)
        keep = emulated["kv_row_keep"].reshape(CACHE_WRITE_ROWS, 1)
        selected = staging * keep + packed * hit
        expected = torch.cat([staging[:row], packed, staging[row + 1 :]])
        assert torch.equal(selected.view(torch.int16), expected.view(torch.int16)), row


def test_ring_one_hot_select_keeps_rows_four_to_thirty_one_exactly_zero() -> None:
    generator = torch.Generator().manual_seed(11)
    ring = torch.zeros(CACHE_WRITE_ROWS, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    keys = torch.randn(8, INDEX_HEAD_DIM, generator=generator).to(torch.bfloat16)
    for position in range(8):
        emulated = _emulate(position)
        hit = emulated["ring_hit"].reshape(CACHE_WRITE_ROWS, 1)
        keep = emulated["ring_keep"].reshape(CACHE_WRITE_ROWS, 1)
        ring = ring * keep + keys[position : position + 1] * hit
        slot = position % COMPRESS_RATIO
        assert torch.equal(ring[slot].view(torch.int16), keys[position].view(torch.int16))
        assert torch.equal(ring[COMPRESS_RATIO:].view(torch.int16), torch.zeros(28, INDEX_HEAD_DIM, dtype=torch.int16))
        if position >= COMPRESS_RATIO - 1:
            block = position // COMPRESS_RATIO
            expected = keys[block * COMPRESS_RATIO : block * COMPRESS_RATIO + slot + 1]
            assert torch.equal(ring[: slot + 1].view(torch.int16), expected.view(torch.int16))


def test_ring_quarter_scaled_sum_equals_four_row_mean_in_bf16() -> None:
    generator = torch.Generator().manual_seed(13)
    for _ in range(64):
        keys = torch.randn(COMPRESS_RATIO, INDEX_HEAD_DIM, generator=generator).to(torch.bfloat16)
        ring = torch.cat([keys, torch.zeros(CACHE_WRITE_ROWS - COMPRESS_RATIO, INDEX_HEAD_DIM, dtype=torch.bfloat16)])
        # Sequential fp32 accumulation, as the reduce kernel does per output column.
        scaled_sum = torch.zeros(INDEX_HEAD_DIM)
        for row in ring:
            scaled_sum = scaled_sum + row.float() * (1.0 / COMPRESS_RATIO)
        total = torch.zeros(INDEX_HEAD_DIM)
        for row in keys:
            total = total + row.float()
        mean = total / COMPRESS_RATIO
        assert torch.equal(scaled_sum.to(torch.bfloat16).view(torch.int16), mean.to(torch.bfloat16).view(torch.int16))


# --- constants ---------------------------------------------------------------


def test_module_constants_pin_the_fixed_window_and_masks() -> None:
    assert not hasattr(qsa_module, "INDEXER_QUERY_ROW")  # the query is row 0 of its own tile (buffer view)
    assert KV_BLOCK_START_MASK == 0xFFFFFFE0 and ALL_ONES_U32 == 0xFFFFFFFF == MASKED_INDEX
    init = inspect.getsource(Qwen38TTNNQSA.__init__)
    assert "self.indexer_chunk_start = self.allocated_compressed_blocks\n" in init
    # Both generic-path constants are rows of the module's one UINT32 table.
    assert 'self.slot_zero = self.uint32_rows["slot_zero"]' in init
    assert 'self.sentinel_pad = self.uint32_rows["sentinel_pad"]' in init
    rows = qsa_row_constants()
    assert torch.equal(rows["slot_zero"].reshape(-1), torch.zeros(1, dtype=torch.int64))
    assert torch.equal(rows["sentinel_pad"].reshape(-1), torch.full((ttnn.TILE_SIZE,), MASKED_INDEX))
    sharded = init.split("self.compressed_row_memory_config = ttnn.create_sharded_memory_config(", 1)[1]
    sharded = sharded.split("use_height_and_width_as_shard_shape=True", 1)[0]
    assert re.sub(r"\s+", "", sharded) == (
        "(ttnn.TILE_SIZE,INDEX_HEAD_DIM),ttnn.CoreGrid(y=1,x=1),ttnn.ShardStrategy.HEIGHT,ttnn.ShardOrientation.ROW_MAJOR,"
    )


def test_position_constants_are_uint32_replicated_with_a_tile_column() -> None:
    build = inspect.getsource(Qwen38TTNNQSAPositionConstants.build)
    assert build.count("ttnn.TILE_LAYOUT") == 1
    assert (
        "arange32_col=upload(torch.arange(ttnn.TILE_SIZE).reshape(1, 1, ttnn.TILE_SIZE, 1), ttnn.TILE_LAYOUT)" in build
    )
    assert "all_ones=upload(torch.full((1, 1, 1, 1), ALL_ONES_U32))" in build
    assert "high27_mask=upload(torch.full((1, 1, 1, 1), KV_BLOCK_START_MASK))" in build
    upload = inspect.getsource(qsa_module._upload_uint32)
    assert "dtype=ttnn.uint32" in upload and "replicate_tensor_2d_mesh_mapper(mesh_device)" in upload
    assert "validate_tensor(tensor, placement=TensorPlacement.REPLICATED)" in upload
    fields = tuple(Qwen38TTNNQSAPositionInputs.__dataclass_fields__)
    assert fields == POSITION_INPUT_FIELDS
    assert set(re.findall(r"self\.(\w+),", inspect.getsource(Qwen38TTNNQSAPositionInputs.deallocate))) == set(fields)


# --- op-sequence pins ----------------------------------------------------------


def test_derivation_uses_only_exact_integer_ops_and_no_host_readback() -> None:
    source = inspect.getsource(qsa_module.derive_qsa_position_inputs)
    for forbidden in ("to_torch", ".item()", "int(", "float(", "position %", "position //", "synchronize"):
        assert forbidden not in source, forbidden
    allowed = {
        "to_layout",
        "bitwise_and",
        "eq",
        "typecast",
        "rsub",
        "bitwise_right_shift",
        "reshape",
        "add",
        "lt",
        "multiply",
        "minimum",
        "bitwise_left_shift",
        "subtract",
        "ge",
        "bitwise_or",
    }
    calls = _ttnn_calls(qsa_module.derive_qsa_position_inputs)
    assert set(calls) <= allowed
    # one_hot() is called twice; each call issues four device ops; reshape is a zero-cost view.
    assert source.count("= one_hot(position_tiled, ") == 2
    one_hot = source.split("def one_hot(", 1)[1].split("position_tiled = ttnn.to_layout(", 1)[0]
    assert re.findall(r"\bttnn\.(\w+)\(", one_hot) == ["bitwise_and", "eq", "typecast", "rsub"]
    device_ops = len(calls) - calls.count("reshape") + 4
    assert device_ops == 34  # 33 at 450403c0 plus the one OR that merges the tail ids and the sentinels per token
    assert "ttnn.bitwise_or(row_tail_fill, row_sentinel_bits" in source
    # Every integer value stays UINT32; the only casts are the 0/1 masks and the INT32 block index.
    assert source.count("ttnn.typecast(") == 3
    assert "ttnn.typecast(hit_bits, ttnn.bfloat16" in source
    assert "ttnn.typecast(valid_bits, ttnn.bfloat16" in source
    assert "ttnn.typecast(block_index, ttnn.int32" in source
    assert "ttnn.reshape(ttnn.typecast(block_index, ttnn.int32, memory_config=dram), (1,))" in source
    for compare in ("ttnn.eq(", "ttnn.lt(", "ttnn.ge("):
        assert all("dtype=u32" in line for line in source.splitlines() if compare in line)
    assert "ttnn.multiply(invalid, INDEXER_MASK_VALUE" in source
    assert "ttnn.minimum(complete_blocks, BLOCK_TOPK" in source
    assert "ttnn.bitwise_and(position_scalar, constants.high27_mask" in source
    assert "ttnn.multiply(before_lo, constants.all_ones" in source
    assert "ttnn.multiply(from_hi, constants.all_ones" in source
    assert "ttnn.multiply(tail_ids, tail_bits" in source
    _only_fail_closed_branches(qsa_module.derive_qsa_position_inputs)


def test_generic_body_has_no_host_ints_and_a_fixed_op_sequence() -> None:
    sources = {function.__name__: inspect.getsource(function) for function in GENERIC_BODY}
    joined = re.sub(r"#[^\n]*", "", "\n".join(sources.values()))
    for forbidden in (
        "next_position",
        "raw_tail_count",
        "state.compressed_blocks",
        "next_compressed_blocks",
        "complete_blocks",
        "tail_count",
        "valid_padded",
        "query_row =",
        "selected_blocks",
        "_take_next_kv_staging",
        "_allocate_view_id",
        "_consume_view",
        "Qwen38TTNNQSAState(",
        "to_torch",
        ".item()",
        "synchronize",
        "kv_len",
        "valid_length",
    ):
        assert forbidden not in joined, forbidden
    for function in GENERIC_BODY:
        _only_fail_closed_branches(function)

    # The only slice has constant bounds (slice start/end are hashed into program identity); no pad: the
    # query is scored from a view of its own tile.
    assert joined.count("ttnn.slice(") == 1 and joined.count("ttnn.pad(") == 0
    score = sources["_score_blocks_generic"]
    assert re.findall(r"\bttnn\.(?:experimental\.)?(\w+)\(", score) == [
        "Shape",
        "reshape",
        "indexer_score_dsa",
        "slice",
        "all_reduce",
        "add",
    ]
    assert "tile_shape = ttnn.Shape((1, INDEX_QUERY_HEADS_PER_DEVICE, ttnn.TILE_SIZE, INDEX_HEAD_DIM))" in score
    assert "query_tile = ttnn.reshape(index_query, tile_shape, tile_shape)" in score
    assert "_deallocate(query_tile)" not in score  # the view shares index_query's buffer
    assert "(0, 0, 0, 0),\n            (1, 1, 1, self.allocated_compressed_blocks)," in score
    indexer = score.split("ttnn.experimental.indexer_score_dsa(", 1)[1].split("\n        )", 1)[0]
    assert "chunk_start_idx=self.indexer_chunk_start" in indexer and "seq_shard_axes=[STAGING_AXIS]" in indexer
    assert "ttnn.add(\n            scores,\n            position.indexer_neg_mask," in score
    assert score.index("ttnn.all_reduce(") < score.index("position.indexer_neg_mask")

    row = sources["_materialize_row_generic"]
    assert "ttnn.experimental.topk_large_indices(masked_scores, k=BLOCK_TOPK)" in row
    assert re.findall(r"\bttnn\.(?:experimental\.)?(\w+)\(", row) == [
        "topk_large_indices",
        "bitwise_left_shift",
        "repeat_interleave",
        "add",
        "concat",
        "bitwise_and",
        "bitwise_or",
    ]
    assert "ttnn.concat([expanded, self.sentinel_pad], dim=3" in row
    assert "ttnn.bitwise_and(template, position.row_keep_bits" in row
    assert "ttnn.bitwise_or(kept, position.row_fill" in row

    kv = sources["_write_packed_kv_generic"]
    assert re.findall(r"\bttnn\.(?:experimental\.deepseek_prefill\.)?(\w+)\(", kv) == [
        "concat",
        "multiply",
        "multiply",
        "add",
        "to_layout",
        "update_padded_kv_cache",
    ]
    assert "ttnn.multiply(state.kv_staging, position.kv_row_keep" in kv
    assert "ttnn.multiply(packed_tiled, position.kv_row_hit" in kv
    assert "ttnn.add(kept, placed, output_tensor=state.kv_staging, fast_and_approximate_mode=False)" in kv
    assert 'raise RuntimeError("QSA KV staging update was not in place")' in kv
    assert "ttnn.to_layout(state.kv_staging, ttnn.ROW_MAJOR_LAYOUT" in kv
    update = kv.split("update_padded_kv_cache(", 1)[1].split("\n        )", 1)[0]
    assert re.sub(r"\s+|#[^\n]*", "", update).startswith(
        "state.packed_kv_cache,stage_row_major,self.slot_zero,position.kv_block_start,0,1,STAGING_AXIS,"
    )

    ring = sources["_write_compressed_index_generic"]
    assert re.findall(r"\bttnn\.(?:experimental\.)?(\w+)\(", ring) == [
        "multiply",
        "multiply",
        "add",
        "sum",
        "rms_norm",
        "to_memory_config",
        "paged_update_cache",
    ]
    assert "ttnn.multiply(state.raw_key_ring, position.ring_keep" in ring
    assert "ttnn.multiply(raw_key, position.ring_hit" in ring
    assert "ttnn.add(kept, placed, output_tensor=state.raw_key_ring, fast_and_approximate_mode=False)" in ring
    assert "dim=2,\n            keepdim=True," in ring and "scalar=1.0 / COMPRESS_RATIO," in ring
    assert "apply_partial_rope_prefill(normalized, block_start_cos, block_start_sin, 1, ROPE_DIM)" in ring
    assert "ttnn.to_memory_config(rotated, self.compressed_row_memory_config)" in ring
    assert "update_idxs_tensor=position.block_index_i32" in ring
    assert ring.index("output_tensor=state.raw_key_ring") < ring.index("ttnn.sum(") < ring.index("paged_update_cache(")

    body = sources["forward_decode_generic"]
    order = (
        "self._validate_generic_state(state)",
        'self._validate_rope(cos, sin, "QSA current RoPE")',
        'self._validate_rope(block_start_cos, block_start_sin, "QSA block-start RoPE")',
        "self._validate_position_inputs(position)",
        "self._all_gather_hidden(hidden_sharded)",
        "self._index_projection(full_hidden, cos, sin)",
        "self._write_compressed_index_generic(state, raw_key, position, block_start_cos, block_start_sin)",
        "self._score_blocks_generic(index_query, state, position)",
        "self._materialize_row_generic(masked_scores, position)",
        "self._main_projection(full_hidden, cos, sin)",
        "self._write_packed_kv_generic(state, key, value, position)",
        "self._sparse_value_attention(query, gate, sparse_indices, state)",
        "self._project_output(local_attention, full_hidden)",
        "_deallocate(full_hidden)",
        "return output",
    )
    positions = [body.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "position: Qwen38TTNNQSAPositionInputs" in body and "position: int" not in body


def test_generic_state_is_fixed_address_with_tile_staging_and_in_place_reset() -> None:
    allocate = inspect.getsource(Qwen38TTNNQSA.allocate_generic_state)
    assert (
        "kv_staging=self._allocate_pair_grouped((1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM), layout=ttnn.TILE_LAYOUT)"
        in allocate
    )
    assert "raw_key_ring=self._allocate_replicated_tile_zeros((1, 1, CACHE_WRITE_ROWS, INDEX_HEAD_DIM))" in allocate
    assert (
        "layout=ttnn.ROW_MAJOR_LAYOUT"
        in allocate.split("packed_kv_cache=", 1)[1].split("compressed_index_cache=", 1)[0]
    )
    # One tile past the resident blocks: the indexer's fixed query window, never written (block_index < allocated).
    compressed = "(1, 1, self.allocated_compressed_blocks + ttnn.TILE_SIZE, INDEX_HEAD_DIM)"
    assert compressed in allocate.split("compressed_index_cache=", 1)[1].split("kv_staging=", 1)[0]
    assert compressed in inspect.getsource(Qwen38TTNNQSA._validate_generic_state)
    reset = inspect.getsource(Qwen38TTNNQSA.reset_generic_state_inplace)
    assert "ttnn.fill(tensor, 0.0, output_tensor=tensor)" in reset
    assert 'raise RuntimeError(f"{label} reset was not in place")' in reset
    assert "state.kv_staging), (" in reset and "state.raw_key_ring)" in reset
    for forbidden in ("allocate_state(", "reset_state(", "packed_kv_cache", "compressed_index_cache"):
        assert forbidden not in reset, forbidden
    release = inspect.getsource(Qwen38TTNNQSA.release_generic_state)
    assert (
        "_deallocate(state.packed_kv_cache, state.compressed_index_cache, state.kv_staging, state.raw_key_ring)"
        in release
    )
    validate = inspect.getsource(Qwen38TTNNQSA._validate_generic_state)
    assert (
        'raise RuntimeError(f"packed QSA cache must be ROW_MAJOR, got {tensor_metadata(state.packed_kv_cache)}")'
        in validate
    )
    assert "TensorPlacement.KV_PAIR_GROUPED, shard_dim=1" in validate
    assert tuple(qsa_module.Qwen38TTNNQSAGenericState.__dataclass_fields__) == (
        "layer_index",
        "epoch",
        "packed_kv_cache",
        "compressed_index_cache",
        "kv_staging",
        "raw_key_ring",
    )
    assert "self._live_generic_epochs" in inspect.getsource(Qwen38TTNNQSA.deallocate)


def test_per_position_path_is_unchanged_apart_from_the_sparse_row_argument() -> None:
    decode = inspect.getsource(Qwen38TTNNQSA.forward_decode)
    assert "self._append_raw_index_key(" in decode and "self._append_packed_kv(state, key, value)" in decode
    assert "self._select(index_query, selection_state, next_compressed_blocks, reuse_selection)" in decode
    assert "self._sparse_value_attention(query, gate, selection.sparse_indices, state)" in decode
    sparse = inspect.getsource(Qwen38TTNNQSA._sparse_value_attention)
    assert "def _sparse_value_attention(self, query, gate, sparse_indices, state):" in sparse
    assert "state.packed_kv_cache,\n            sparse_indices,\n            HEAD_DIM," in sparse
    assert "selection" not in sparse


# --- pinned-runtime contracts (repo-relative C++ reads) -------------------------


def test_pinned_update_padded_kv_cache_has_the_metadata_tensor_overload() -> None:
    root = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/"
    nanobind = _cpp(root + "update_padded_kv_cache_nanobind.cpp")
    tensor_form = nanobind.split("// Per-element-tensor form (traceable).", 1)[1]
    # The metadata overload took optional cluster_axis / valid_global / tp_axis upstream; the model's four call sites
    # pass the first seven arguments and take the defaults.
    assert re.sub(r"\s+", " ", tensor_form).startswith(
        " ttnn::overload_t( nb::overload_cast< const Tensor&, const Tensor&, const Tensor&, const Tensor&, uint32_t, "
        "uint32_t, std::optional<uint32_t>, const std::optional<Tensor>&, std::optional<uint32_t>>"
        "(&update_padded_kv_cache),"
    )
    assert re.findall(r'nb::arg\("(\w+)"\)', tensor_form) == [
        "cache",
        "input",
        "slot_idx",
        "kv_actual_global",
        "layer_idx",
        "num_layers",
        "cluster_axis",
        "valid_global",
        "tp_axis",
    ]
    assert 'nb::arg("valid_global").noconvert() = nb::none(),' in tensor_form
    assert 'nb::arg("tp_axis") = nb::none()));' in tensor_form
    device = _cpp(root + "device/update_padded_kv_cache_device_operation.cpp")
    for check in (
        'TT_FATAL(meta.dtype() == DataType::UINT32, "metadata tensor {} must be UINT32", name);',
        'TT_FATAL(meta.layout() == Layout::ROW_MAJOR, "metadata tensor {} must be ROW_MAJOR", name);',
        "meta.logical_volume() == 1,",
        'TT_FATAL(!meta.is_sharded(), "metadata tensor {} must not be sharded", name);',
        # The program hash keys on the metadata path, the valid_global presence and the layer.
        "tensor_args.slot_idx.has_value(),\n        tensor_args.valid_global.has_value() || args.valid_global.has_value(),"
        "\n        args.layer_idx,",
        "writer_tile_height = 1;",
    ):
        assert check in device, check
    # The scalar path's tile-alignment check does not run on the metadata path: the writer divides by 1 for ROW_MAJOR.
    assert "if (!tensor_args.slot_idx.has_value()) {" in device
    writer = _cpp(root + "device/kernels/dataflow/writer_update_padded_kv_cache.cpp")
    assert "kv_actual_global_t = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0] / tile_height;" in writer
    assert "invalidate_l1_cache();" in writer


def test_pinned_paged_update_cache_takes_an_int32_index_tensor_and_one_sharded_user() -> None:
    root = "ttnn/cpp/ttnn/operations/experimental/paged_cache/"
    assert 'nb::arg("update_idxs_tensor").noconvert() = nb::none(),' in _cpp(root + "paged_cache_nanobind.cpp")
    device = _cpp(root + "device/update_cache/paged_update_cache_device_operation.cpp")
    for check in (
        'TT_FATAL(update_idxs_tensor_val.dtype() == DataType::INT32, "Expected update_idxs to have datatype INT32");',
        'TT_FATAL(cache_tensor.layout() == Layout::TILE, "Cache tensor in update_cache must be tilized");',
        'TT_FATAL(input_tensor.is_sharded(), "Expect input_tensor to be sharded");',
        "input_num_shards == num_users,",
        "input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,",
    ):
        assert check in device, check
    writer = _cpp(root + "device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp")
    assert "cache_tile_offset_B = update_idx % TILE_HEIGHT * Wbytes;" in writer
    assert "(update_idx / TILE_HEIGHT) * Wt" in writer


def test_pinned_indexer_score_hashes_only_kv_len_presence_and_bounds_the_fixed_window() -> None:
    device = _cpp("ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_device_operation.cpp")
    assert "chunk_start_idx is EXCLUDED" in device
    assert "attrs.has_runtime_kv_len()," in device
    validate = device.split("void validate_chunk_start(", 1)[1].split("\n}\n", 1)[0]
    assert "attrs.chunk_start_idx % tt::constants::TILE_WIDTH == 0," in validate
    # The chunk must begin inside T and inside kv_len; the causal window may end past kv_len (pad query rows).
    assert "attrs.chunk_start_idx < T," in validate
    assert "if (attrs.kv_len.has_value()) {" in validate
    assert "attrs.chunk_start_idx < kv_len," in validate
    # Without kv_len the causal window ends at T, the generic cache's last row: chunk_start + Sq == T.
    assert (
        "const uint32_t kv_len_tiles = attrs.kv_len.has_value() ? attrs.kv_len.value() / tt::constants::TILE_WIDTH : Tt;"
        in device
    )


def test_pinned_topk_large_indices_valid_length_is_optional() -> None:
    device = _cpp(
        "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp"
    )
    assert "if (attrs.valid_length.has_value()) {" in device
    assert "attrs.k > 0 && attrs.k <= max_supported_k && attrs.k % 16 == 0," in device
    assert BLOCK_TOPK % 16 == 0 and 16 <= BLOCK_TOPK <= 2048


def test_pinned_sparse_sdpa_reader_needs_a_contiguous_sentinel_tail() -> None:
    reader = _cpp("ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_reader.cpp")
    assert "Sentinels are a contiguous tail (producer contract)" in reader
    assert "contract guarantees >=1 valid" in reader
    assert SPARSE_INDEX_CAPACITY % ttnn.TILE_SIZE == 0


def test_pinned_binary_layout_rules_force_a_tile_staging_and_homogeneous_operands() -> None:
    """binary.cpp rejects a preallocated output for two ROW_MAJOR operands and tilizes mixed layouts implicitly."""

    wrapper = _cpp("ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp")
    assert "!(output_preallocated && input_a_rm && input_b_rm)," in wrapper
    assert (
        "Optional output tensor with Row Major input is not supported right now for Elementwise operations" in wrapper
    )
    assert "const auto input_a = operations::binary::detail::to_layout(lhs_eff, Layout::TILE);" in wrapper
    device = _cpp("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp")
    assert (
        "input_tensor_a.layout() == Layout::ROW_MAJOR &&\n        input_tensor_b.layout() == Layout::ROW_MAJOR) {\n"
        "        output_layout = Layout::ROW_MAJOR;" in device
    )
    utils = _cpp("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp")
    assert 'fmt::format("mul_int_tile<DataFormat::{}>", *int_data_format)' in utils
    assert "if (dtype == DataType::UINT32) {\n                return static_cast<uint32_t>(v);" in utils
    tilize = _cpp(
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/"
        "tilize_with_val_padding_device_operation.cpp"
    )
    assert "input_tensor.dtype() == DataType::UINT32" in tilize
    uint32_tests = _cpp("tests/ttnn/unit_tests/operations/eltwise/test_binary_uint32.py")
    upper = uint32_tests.split("def test_binary_mul_uint32_upper_edge_cases", 1)[1]
    assert "4294967295, 1, 4294967295" in upper
