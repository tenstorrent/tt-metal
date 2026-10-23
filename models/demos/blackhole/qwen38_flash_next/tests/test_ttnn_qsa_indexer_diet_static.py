# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static and torch-emulation tests for the per-position QSA indexer-path diet.

The complete-block selection path of ``Qwen38TTNNQSA.forward_decode`` is
trimmed op by op; every trim is pinned here against the bytes the old path
produced and against the pinned-runtime C++ it relies on.
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.reference import qsa_selected_token_mask
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    BLOCK_TOPK,
    COMPRESS_RATIO,
    EXPANSION_GATHER_WIDTH,
    MASKED_INDEX,
    MAX_SELECTED_TOKENS,
    MAX_SPECULATIVE_STEPS,
    QSA_ROW_WIDTHS,
    SHARED_ROW_CACHE,
    SPARSE_INDEX_CAPACITY,
    TOKEN_BUDGET,
    Qwen38TTNNQSA,
    emulate_block_expansion,
    emulate_sparse_row,
    emulate_step_window,
    qsa_natural_row_regime,
    qsa_row_constants,
    qsa_selection_geometry,
)

REPOSITORY = Path(inspect.getfile(Qwen38TTNNQSA)).resolve().parents[5]
UINT32_LIMIT = 2**32


def _cpp(relative: str) -> str:
    return (REPOSITORY / relative).read_text(encoding="utf-8")


def _ttnn_calls(source: str) -> list[str]:
    return re.findall(r"\bttnn\.(?:experimental\.(?:deepseek_prefill\.)?|transformer\.)?(\w+)\(", source)


# --- T2: two half-width gathers and a concat replace the eight-op UINT32 repeat_interleave ---


def test_row_constants_are_uint32_rows_of_the_pinned_widths() -> None:
    rows = qsa_row_constants()
    assert set(rows) == set(QSA_ROW_WIDTHS)
    for name, values in rows.items():
        assert values.dtype == torch.int64 and tuple(values.shape) == (1, 1, 1, QSA_ROW_WIDTHS[name]), name
        assert int(values.min()) >= 0 and int(values.max()) < UINT32_LIMIT, name
    assert QSA_ROW_WIDTHS["block_offsets"] == TOKEN_BUDGET == 2 * EXPANSION_GATHER_WIDTH
    assert QSA_ROW_WIDTHS["rep_index_lo"] == QSA_ROW_WIDTHS["rep_index_hi"] == EXPANSION_GATHER_WIDTH
    assert torch.equal(rows["block_offsets"].reshape(-1), torch.arange(COMPRESS_RATIO).repeat(BLOCK_TOPK))


def test_block_expansion_emulation_reproduces_repeat_interleave_for_random_uint32_rows(expect_error) -> None:
    rows = {name: values.reshape(-1) for name, values in qsa_row_constants().items()}
    rep_index = torch.cat([rows["rep_index_lo"], rows["rep_index_hi"]])
    assert torch.equal(rep_index, torch.arange(BLOCK_TOPK).repeat_interleave(COMPRESS_RATIO))
    generator = torch.Generator().manual_seed(20260902)
    offsets = torch.arange(COMPRESS_RATIO).repeat(BLOCK_TOPK)
    for _ in range(64):
        starts = torch.randint(0, UINT32_LIMIT, (BLOCK_TOPK,), generator=generator, dtype=torch.int64)
        expanded = emulate_block_expansion(starts)
        assert torch.equal(expanded, (starts.repeat_interleave(COMPRESS_RATIO) + offsets) & (UINT32_LIMIT - 1))
        selected = int(torch.randint(1, BLOCK_TOPK + 1, (1,), generator=generator))
        assert torch.equal(
            expanded[: selected * COMPRESS_RATIO],
            (starts[:selected].repeat_interleave(COMPRESS_RATIO) + offsets[: selected * COMPRESS_RATIO])
            & (UINT32_LIMIT - 1),
        )
    # The expansion the old path produced: 4 * block + offset, in top-k order.
    block_ids = torch.randperm(BLOCK_TOPK, generator=generator)
    assert torch.equal(
        emulate_block_expansion(block_ids << 2),
        (block_ids * COMPRESS_RATIO).repeat_interleave(COMPRESS_RATIO) + torch.arange(4).repeat(512),
    )
    with expect_error(ValueError):
        emulate_block_expansion(block_ids[:511])


def test_score_complete_blocks_gathers_both_half_rows_and_joins_them() -> None:
    source = inspect.getsource(Qwen38TTNNQSA._score_complete_blocks)
    assert "ttnn.repeat_interleave(" not in source and "self.rep_index," not in source
    assert "ttnn.gather(starts, 3, rep_index, memory_config=ttnn.DRAM_MEMORY_CONFIG)" in source
    assert "for rep_index in (self.rep_index_lo, self.rep_index_hi)" in source
    assert "ttnn.concat(halves, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)" in source
    assert (
        source.index("ttnn.bitwise_left_shift(block_ids_rm, 2")
        < source.index("ttnn.gather(")
        < source.index("ttnn.concat(halves")
        < source.index("ttnn.add(repeated, self.block_offsets")
    )


def test_pinned_runtime_routes_uint32_last_dim_repeat_interleave_to_eight_ops_and_gather_to_one() -> None:
    codegen = _cpp(
        "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.cpp"
    )
    assert "if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32 && dtype != DataType::INT32) {" in codegen
    assert "if (nd == ndim - 1) {\n            return false;" in codegen
    native = _cpp("ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/repeat_interleave.cpp")
    last_dim = native.split("if (normalized_dim == input_rank - 1) {", 1)[1].split("\n    }\n", 1)[0]
    assert "ttnn::transpose(transpose_input, -1, -2, mem_config)" in last_dim
    assert "repeat_interleave_native(transposed_input, repeat, -2, mem_config)" in last_dim
    concat = _cpp("ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp")
    assert "tensor.padded_shape()[dim] * tensor.element_size() % tensor.buffer()->alignment() == 0" in concat

    gather = _cpp("ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp")
    assert "tensor_args.input_index_tensor.dtype() == DataType::UINT32 ||" in gather
    assert "tensor_args.input_tensor.layout() == tensor_args.input_index_tensor.layout()," in gather
    assert "constexpr uint32_t GATHER_WT_THRESHOLD = 60;" in gather
    assert "if (W_index > rm_w_threshold) {\n            return RmSingleRowMultiCore{};" in gather
    router = _cpp("ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp")
    assert "const bool input_tensor_is_dim_last_idx = (normalized_dim == input_tensor_rank - 1);" in router
    nanobind = _cpp("ttnn/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp")
    assert 'nb::arg("input").noconvert(),\n        nb::arg("dim"),\n        nb::arg("index"),' in nanobind
    # The full 2048-wide row would take RmSingleRowMultiCore, which the 2026-09-02 op repro on 4x p150
    # showed wrong for every dtype (76% of the elements); each half stays on the exact single-core factory
    # and its 4 KiB stick keeps the joining concat on the aligned path.
    assert EXPANSION_GATHER_WIDTH <= 60 * ttnn.TILE_SIZE < TOKEN_BUDGET
    assert (EXPANSION_GATHER_WIDTH * 4) % 64 == 0 and TOKEN_BUDGET % EXPANSION_GATHER_WIDTH == 0


# --- T1: the query tile as a zero-cost view, causal window one tile later ---------


def _indexer_scores(query_tile: torch.Tensor, keys: torch.Tensor, gate: float) -> torch.Tensor:
    """Per element: relu(q_row . k_col) * w[row]; the kernel's matmul, packer ReLU and gate multiply never mix rows."""

    return torch.stack([torch.relu((row[None, :] * keys).float().sum(-1)) * gate for row in query_tile])


def test_view_row_zero_scores_equal_the_padded_row_scores_and_see_every_complete_block() -> None:
    generator = torch.Generator().manual_seed(2026)
    resident_blocks = 32768 // COMPRESS_RATIO
    gate = 128**-0.5
    for complete_blocks in (1, 2, 3, 8, 31, 32, 33, 512, 513, 1024, resident_blocks - 32, resident_blocks - 31):
        valid_padded = math.ceil(complete_blocks / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        view_possible = valid_padded + ttnn.TILE_SIZE <= resident_blocks
        query = torch.randn(128, generator=generator).to(torch.bfloat16)
        keys = torch.randn(valid_padded + ttnn.TILE_SIZE, 128, generator=generator).to(torch.bfloat16)
        # Padded placement (the old path and the last-tile-row fallback): the query in its causal row, zeros around.
        chunk_start = valid_padded - ttnn.TILE_SIZE
        query_row = complete_blocks - 1 - chunk_start
        padded = torch.zeros(ttnn.TILE_SIZE, 128, dtype=torch.bfloat16)
        padded[query_row] = query
        padded_scores = _indexer_scores(padded, keys[:valid_padded], gate)[query_row]
        # View placement: the query in row 0, arbitrary finite bytes (even NaN) in rows 1-31.
        view = torch.randn(ttnn.TILE_SIZE, 128, generator=generator).to(torch.bfloat16)
        view[1] = float("nan")
        view[0] = query
        view_scores = _indexer_scores(view, keys[: valid_padded + ttnn.TILE_SIZE], gate)[0]
        assert torch.equal(view_scores[:complete_blocks], padded_scores[:complete_blocks])
        assert view_scores.isfinite().all()

        # Causal geometry (indexer_score_work_split.hpp): row 0's diagonal tile is chunk_start_tiles, tiles
        # before it are unmasked; with chunk_start = valid_padded every complete block is unmasked and the
        # only extra visible column, key valid_padded itself, lies past valid_length for the top-k.
        if view_possible:
            diag_tile = valid_padded // ttnn.TILE_SIZE
            assert diag_tile * ttnn.TILE_SIZE >= complete_blocks
            assert valid_padded % ttnn.TILE_SIZE == 0 and (valid_padded + ttnn.TILE_SIZE) % ttnn.TILE_SIZE == 0
            assert valid_padded + ttnn.TILE_SIZE <= resident_blocks  # max_cs + Sq <= kv_len <= T
        else:
            assert 0 <= query_row < ttnn.TILE_SIZE and chunk_start + query_row == complete_blocks - 1
    # The view stops where no tile follows the window: 8160 complete blocks on the resident cache, so the
    # front-padded fallback serves positions 32643..32767 only.
    assert math.ceil(8160 / ttnn.TILE_SIZE) * ttnn.TILE_SIZE + ttnn.TILE_SIZE == resident_blocks
    assert math.ceil(8161 / ttnn.TILE_SIZE) * ttnn.TILE_SIZE + ttnn.TILE_SIZE > resident_blocks
    assert (
        qsa_selection_geometry(32643 + 1).complete_blocks == 8161
        and qsa_selection_geometry(32643).complete_blocks == 8160
    )


def test_pinned_runtime_view_reshape_is_a_buffer_alias_and_the_indexer_never_mixes_query_rows() -> None:
    reshape = _cpp("ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp")
    branch = reshape.split("bool tile_tensor_view_reshape_possible =", 1)[1].split("TT_FATAL(false", 1)[0]
    assert "layout == ttnn::Layout::TILE and padded_shape.rank() >= 2" in branch
    assert "tensor.padded_shape()[-1] == padded_shape[-1]);" in branch
    assert "return ttnn::experimental::view(tensor, logical_shape, padded_shape);" in branch
    nanobind = _cpp("ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_nanobind.cpp")
    assert (
        'nb::arg("input_tensor"),\n            nb::arg("logical_shape"),\n            nb::arg("padded_shape"),'
        in nanobind
    )
    ops = _cpp("ttnn/core/tensor/tensor_ops.cpp")
    assert (
        "Tensor view(const Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape) {"
        in ops
    )
    assert "input_tensor.mesh_buffer().address());" in ops  # same bytes, new shape metadata

    kernel = _cpp("ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/compute_indexer_score.cpp")
    # q.kT per DEST pass is a matmul over the head dim (per output element), ReLU in the packer, then a
    # per-row gate multiply broadcast along columns; the causal mask is stamped per tile; the output is untilized.
    assert "matmul_block(\n            q_cb,\n            k_cb,\n            q_base + dim_tile," in kernel
    assert "pack_relu_config(ReluConfig::zero());" in kernel
    assert "mul_tiles_bcast_cols(qk_cb, w_cb, head, w_base + head, 0);" in kernel
    assert "mul_tiles_bcast_cols_custom(cb_qk, cb_w, hl * cols + sub_base, w_base + hl, 0, n_cols);" in kernel
    assert "diagonal (k_tile == diag_tile): L1-ACCUMULATE the strict-upper -inf tile, keeping the lower tri." in kernel
    assert "stamp_mask_tile<cb_acc_strip, cb_mask>(slot_base + k_col, k_tile0 + k_col, diag_tile);" in kernel
    assert "compute_kernel_lib::untilize<k_tiles_per_unit, cb_acc_strip, cb_out_strip>(q_tiles_per_unit);" in kernel
    assert "if (k_tiles_in_unit == 0) {" in kernel  # cells past kv_len are skipped, not computed
    split = _cpp("ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/indexer_score_work_split.hpp")
    assert "return chunk_start_tiles + q_row_abs + (q_row_abs >= straddle_q_tile ? straddle_jump_tiles : 0);" in split
    assert "const uint32_t v = diag_tile > k_tile_start ? diag_tile - k_tile_start : 0;" in split
    device = _cpp("ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_device_operation.cpp")
    assert '"indexer_score kv_len {} must be tile-aligned"' in device
    assert "attrs.chunk_start_idx < T," in device and "attrs.chunk_start_idx < kv_len," in device


# --- T3: template row with per-token masks instead of the unaligned concat -------------

MAX_TAIL = COMPRESS_RATIO - 1 + MAX_SPECULATIVE_STEPS


def _old_sparse_row(block_ids: torch.Tensor, complete_token_count: int, tail_start: int, context_length: int):
    """The f1a465e3 row: [4 * block + offset for the selected top-k slots | tail_start .. P | sentinels]."""

    selected = block_ids[: complete_token_count // COMPRESS_RATIO]
    expanded = (selected * COMPRESS_RATIO).repeat_interleave(COMPRESS_RATIO) + torch.arange(COMPRESS_RATIO).repeat(
        len(selected)
    )
    valid = torch.cat([expanded, torch.arange(tail_start, context_length)])
    return torch.cat([valid, torch.full((SPARSE_INDEX_CAPACITY - len(valid),), MASKED_INDEX, dtype=torch.int64)])


def _device_expansion(block_ids: torch.Tensor) -> torch.Tensor:
    """What _score_complete_blocks hands on: every top-k slot expanded, stale slots included."""

    return emulate_block_expansion(block_ids << 2)


def test_step_windows_are_the_mask_rows_for_every_count() -> None:
    rows = {name: values.reshape(-1) for name, values in qsa_row_constants().items()}
    assert QSA_ROW_WIDTHS["keep_step"] == QSA_ROW_WIDTHS["sentinel_step"] == 2 * SPARSE_INDEX_CAPACITY
    assert QSA_ROW_WIDTHS["arange_row"] == SPARSE_INDEX_CAPACITY
    assert QSA_ROW_WIDTHS["sentinel_pad"] == SPARSE_INDEX_CAPACITY - TOKEN_BUDGET == ttnn.TILE_SIZE
    slots = torch.arange(SPARSE_INDEX_CAPACITY)
    for count in range(SPARSE_INDEX_CAPACITY + 1):
        keep = emulate_step_window(rows["keep_step"], count)
        sentinel = emulate_step_window(rows["sentinel_step"], count)
        assert torch.equal(keep, (slots < count).to(torch.int64) * MASKED_INDEX)
        assert torch.equal(sentinel, (slots >= count).to(torch.int64) * MASKED_INDEX)
        assert keep.shape == sentinel.shape == (SPARSE_INDEX_CAPACITY,)
    assert torch.equal(
        rows["arange_row"] | emulate_step_window(rows["sentinel_step"], 3), _old_sparse_row(slots, 0, 0, 3)
    )


def test_template_row_bytes_equal_the_old_concat_row_for_every_position_and_reuse_tail() -> None:
    generator = torch.Generator().manual_seed(2026)
    resident_blocks = 32768 // COMPRESS_RATIO
    closing = 0
    for position in range(0, 4200):
        geometry = qsa_selection_geometry(position + 1)
        selected = torch.randperm(max(geometry.complete_blocks, 1), generator=generator)[: geometry.selected_blocks]
        stale = torch.randint(0, resident_blocks, (BLOCK_TOPK - geometry.selected_blocks,), generator=generator)
        block_ids = torch.cat([selected, stale])
        lo, tail_start = geometry.complete_token_count, geometry.tail_start
        closing += geometry.tail_count == 0
        # Fresh selection at this position (block-closing positions have an empty tail).
        expanded = None if lo == 0 else _device_expansion(block_ids)
        for reuse_distance in range(0, MAX_SPECULATIVE_STEPS + 1):
            context_length = position + 1 + reuse_distance
            if context_length - tail_start > MAX_TAIL:
                continue
            new = emulate_sparse_row(
                expanded, complete_token_count=lo, tail_start=tail_start, context_length=context_length
            )
            old = _old_sparse_row(block_ids, lo, tail_start, context_length)
            assert torch.equal(new, old), (position, reuse_distance)
            valid = lo + context_length - tail_start
            assert (
                1 <= valid <= MAX_SELECTED_TOKENS
                and (new[valid:] == MASKED_INDEX).all()
                and (new[:valid] < MASKED_INDEX).all()
            )
    assert closing == 4200 // COMPRESS_RATIO


def test_natural_row_is_the_old_row_when_nothing_was_scored(expect_error) -> None:
    for context_length in (1, 2, 3, 4, 7, 2051, MAX_SELECTED_TOKENS):
        row = emulate_sparse_row(None, complete_token_count=0, tail_start=0, context_length=context_length)
        assert torch.equal(row[:context_length], torch.arange(context_length))
        assert (row[context_length:] == MASKED_INDEX).all()
    with expect_error(ValueError):
        emulate_sparse_row(None, complete_token_count=0, tail_start=4, context_length=8)


def test_materialization_and_scoring_have_the_fixed_op_sequences() -> None:
    score = inspect.getsource(Qwen38TTNNQSA._score_complete_blocks)
    assert _ttnn_calls(score) == [
        "Shape",
        "reshape",
        "indexer_score_dsa",
        "slice",
        "all_reduce",
        "topk_large_indices",
        "bitwise_left_shift",
        "gather",
        "concat",
        "add",
    ]
    assert "ttnn.gather(starts, 3, rep_index" in score and "ttnn.add(repeated, self.block_offsets" in score
    assert "selected_blocks" not in score and "token_count" not in score
    assert "return expanded, min(BLOCK_TOPK, complete_blocks) * COMPRESS_RATIO" in score
    assert _ttnn_calls(inspect.getsource(Qwen38TTNNQSA._front_padded_query)) == ["to_layout", "pad", "to_layout"]

    materialize = inspect.getsource(Qwen38TTNNQSA._materialize_selection)
    assert _ttnn_calls(materialize) == ["concat", "bitwise_and", "bitwise_or"]
    assert "ttnn.concat([complete_indices, self.sentinel_pad], dim=3" in materialize
    assert "self._natural_row(context_length)" in materialize
    assert "self._template_masks(complete_token_count, tail_start, valid_count)" in materialize
    assert "owns_sparse_indices=complete_indices is not None" in materialize
    assert _ttnn_calls(inspect.getsource(Qwen38TTNNQSA._step_window)) == ["slice"]
    assert "(0, 0, 0, SPARSE_INDEX_CAPACITY - count),\n            (1, 1, 1, 2 * SPARSE_INDEX_CAPACITY - count)," in (
        inspect.getsource(Qwen38TTNNQSA._step_window)
    )
    natural = inspect.getsource(Qwen38TTNNQSA._natural_row)
    assert _ttnn_calls(natural) == ["bitwise_or"] and natural.count("self._step_window(") == 1
    masks = inspect.getsource(Qwen38TTNNQSA._template_masks)
    assert _ttnn_calls(masks) == ["add", "bitwise_and", "bitwise_or"] and masks.count("self._step_window(") == 3
    assert "ids = ( self.arange_row if shift == 0 else ttnn.add(self.arange_row, shift," in re.sub(r"\s+", " ", masks)
    transients = inspect.getsource(Qwen38TTNNQSA._release_view_transients)
    assert "if selection.owns_sparse_indices:\n                owned.append(selection.sparse_indices)" in transients


def test_device_op_counts_per_layer_and_per_token() -> None:
    """Device ops on the selection path per QSA layer at a position with a tail (all_reduce is five launches)."""

    old_scoring = (
        3 + 1 + 1 + 5 + 1 + 1 + 1 + 8 + 1 + 1
    )  # pad triple, indexer, slice, all_reduce, topk, slice, shift, repeat_interleave, slice, add
    old_materialization = 1 + 1 + 1 + 5  # tail slice, add, sentinel slice, three-piece unaligned concat
    # view, indexer, slice, all_reduce, topk, shift, two half gathers, concat, add
    new_scoring = 0 + 1 + 1 + 5 + 1 + 1 + 2 + 1 + 1
    new_materialization = 3  # concat, and, or
    per_token_masks = 1 + 4  # keep window; tail: two windows, and, or (plus one add when the tail is shifted)
    assert (old_scoring, old_materialization) == (23, 8)
    assert (new_scoring, new_materialization) == (13, 3)
    assert old_scoring + old_materialization - new_scoring - new_materialization == 15
    assert 12 * 15 - per_token_masks == 175  # device ops saved per token with twelve QSA layers, exact path
    assert (
        12 * (1 + 1 + 1 + 4) == 84
    )  # positions 0-2: the seven-op old row (slice, add, slice, four-op concat) is now shared


class _FakeTensor:
    _next = iter(range(1, 1 << 20))

    def __init__(self, label: str) -> None:
        self.label = label
        self.tensor_id = next(self._next)


def _bare_module(mesh) -> Qwen38TTNNQSA:
    module = object.__new__(Qwen38TTNNQSA)
    module.mesh_device = mesh
    module._trace_input_retention_active = False
    module._trace_retained_inputs = {}
    module._trace_rollover_staging = []
    return module


def test_shared_rows_are_built_once_per_key_released_fifo_and_retained_with_the_trace(monkeypatch) -> None:
    freed: list[str] = []
    monkeypatch.setattr(qsa_module.ttnn, "deallocate", lambda tensor: freed.append(tensor.label))
    monkeypatch.setattr(Qwen38TTNNQSA, "_shared_rows", {})
    mesh, other_mesh = object(), object()
    first, second, elsewhere = _bare_module(mesh), _bare_module(mesh), _bare_module(other_mesh)
    builds: list[str] = []

    def build(label: str):
        def _build():
            builds.append(label)
            return _FakeTensor(label)

        return _build

    row = first._shared_row(("natural", 5), build("n5"))
    assert second._shared_row(("natural", 5), build("n5-again")) is row  # the second layer reuses the first's row
    assert elsewhere._shared_row(("natural", 5), build("other-n5")) is not row  # another mesh builds its own
    assert builds == ["n5", "other-n5"] and freed == []

    for count in range(SHARED_ROW_CACHE + 3):
        first._shared_row(("natural", 100 + count), build(f"n{100 + count}"))
    assert freed == ["n5", "n100", "n101", "n102"]  # FIFO beyond the cache, only this mesh's rows
    assert len([key for key in Qwen38TTNNQSA._shared_rows if key[0] == id(mesh)]) == SHARED_ROW_CACHE

    # A layer capturing a trace adopts the row it reads; adopted rows are never evicted.
    second._trace_input_retention_active = True
    kept = second._shared_row(("natural", 103), build("unused"))
    assert second._trace_retained_inputs == {("ttnn", kept.tensor_id): kept}
    for count in range(SHARED_ROW_CACHE + 2):
        first._shared_row(("keep", count), build(f"k{count}"))
    assert "n103" not in freed and all(label in freed for label in ("n104", "n105"))
    second._trace_input_retention_active = False
    second.release_trace_retained_inputs()
    assert freed[-1] == "n103" and ("natural", 103) not in {key[1:] for key in Qwen38TTNNQSA._shared_rows}
    assert second._trace_retained_inputs == {}

    # Module teardown releases the eager rows of its own mesh only.
    first._release_shared_rows(keep=0)
    assert not [key for key in Qwen38TTNNQSA._shared_rows if key[0] == id(mesh)]
    assert [key for key in Qwen38TTNNQSA._shared_rows if key[0] == id(other_mesh)] == [(id(other_mesh), "natural", 5)]


# --- R: no scoring while the budget covers every complete block ---------------------


def _oracle_selected_tokens(position: int, generator: torch.Generator) -> set[int]:
    """The CPU oracle's selected set at ``position`` for random keys and queries (reference.py, read-only)."""

    context_length = position + 1
    query = torch.randn(1, 1, 4, 128, generator=generator).to(torch.bfloat16)
    raw_keys = torch.randn(1, context_length, 128, generator=generator).to(torch.bfloat16)
    angles = torch.arange(context_length, dtype=torch.float32).view(1, -1, 1) * torch.linspace(0.001, 0.05, 32).view(
        1, 1, 32
    )
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(torch.bfloat16)
    mask = torch.ones(1, 1, 1, context_length, dtype=torch.bool)
    visible = qsa_selected_token_mask(
        query,
        raw_keys,
        cos,
        sin,
        mask,
        q_norm_weight=torch.ones(128, dtype=torch.bfloat16),
        k_norm_weight=torch.ones(128, dtype=torch.bfloat16),
        token_budget=TOKEN_BUDGET,
        compress_ratio=COMPRESS_RATIO,
        eps=1e-6,
    )
    assert visible.shape == (1, 1, 1, context_length)
    return set(visible.reshape(-1).nonzero().reshape(-1).tolist())


def test_natural_row_regime_selects_the_oracle_set_up_to_the_budget_and_scores_past_it(expect_error) -> None:
    generator = torch.Generator().manual_seed(2050)
    sampled = [3, 4, 7, 31, 32, 33, 127, 2047, 2050] + torch.randint(0, 2051, (6,), generator=generator).tolist()
    for position in sampled:
        complete_blocks = qsa_selection_geometry(position + 1).complete_blocks
        assert complete_blocks <= BLOCK_TOPK and qsa_natural_row_regime(complete_blocks, regime_split=True)
        row = emulate_sparse_row(None, complete_token_count=0, tail_start=0, context_length=position + 1)
        valid = row[row != MASKED_INDEX]
        assert set(valid.tolist()) == _oracle_selected_tokens(position, generator) == set(range(position + 1))
        assert torch.equal(valid, torch.arange(position + 1))  # natural order is the canonical order
    # First position of regime 2: 513 complete blocks, the oracle drops one of them.
    assert qsa_selection_geometry(2051 + 1).complete_blocks == BLOCK_TOPK + 1
    assert not qsa_natural_row_regime(BLOCK_TOPK + 1, regime_split=True)
    assert len(_oracle_selected_tokens(2051, generator)) == TOKEN_BUDGET  # 512 of 513 blocks, no tail
    assert qsa_natural_row_regime(BLOCK_TOPK, regime_split=True) and not qsa_natural_row_regime(BLOCK_TOPK)
    assert qsa_natural_row_regime(0, regime_split=False) and not qsa_natural_row_regime(1, regime_split=False)
    for bad in (-1, True, 2.0):
        with expect_error(ValueError):
            qsa_natural_row_regime(bad)


def test_regime_split_is_off_by_default_and_a_constructor_flag() -> None:
    # Off on numerics, not on the selected set: the natural-order row hands
    # sparse_sdpa the same keys in another order, its online-softmax
    # accumulation order changes, and 4x p150 micro-test (2026-09-02, positions
    # 127-2050) saw only 20-38% of the bf16 attention output within one ulp of
    # the scored path.  The path and the constructor flag stay for an A/B on a
    # later runtime.
    assert qsa_module.QSA_INDEXER_REGIME_SPLIT is False
    assert not qsa_natural_row_regime(1) and not qsa_natural_row_regime(BLOCK_TOPK)  # default: score every block
    assert qsa_natural_row_regime(0)  # no complete block: the natural row is the only row
    assert inspect.signature(Qwen38TTNNQSA.__init__).parameters["regime_split"].default is None
    init = inspect.getsource(Qwen38TTNNQSA.__init__)
    assert "self.regime_split = QSA_INDEXER_REGIME_SPLIT if regime_split is None else regime_split" in init
    select = inspect.getsource(Qwen38TTNNQSA._select)
    assert "if qsa_natural_row_regime(complete_blocks, regime_split=self.regime_split):" in select
    assert "complete_indices, complete_token_count, tail_start = None, 0, 0" in select
    assert select.index("qsa_natural_row_regime(") < select.index("self._score_complete_blocks(")
    assert "tail_start = complete_blocks * COMPRESS_RATIO" in select
    assert _ttnn_calls(select) == []


def test_device_op_counts_by_regime() -> None:
    """Selection-path device ops per QSA layer with the split on: regime 1 (P <= 2050) has none, regime 2 keeps
    the exact path.  With the split off (the default) every position takes the regime-2 counts."""

    old_per_layer = 23 + 8
    regime_1_per_layer, regime_1_per_token = 0, 2  # the shared natural row: one window slice, one OR
    regime_2_per_layer, regime_2_per_token = 13 + 3, 1 + 5  # exact path; the tail row carries the shift add
    assert old_per_layer - regime_1_per_layer == 31 and 12 * 31 - regime_1_per_token == 370
    assert old_per_layer - regime_2_per_layer == 15
    assert regime_2_per_token == 6


# --- exact data-movement diet: same arithmetic, fewer layout programs ------------------


def test_main_projection_splits_heads_with_one_reshape_and_two_slices_and_no_transpose() -> None:
    source = inspect.getsource(Qwen38TTNNQSA._main_projection)
    assert _ttnn_calls(source) == [
        "linear",
        "linear",
        "linear",
        "to_memory_config",
        "to_memory_config",
        "to_memory_config",
        "reshape",
        "slice",
        "slice",
        "rms_norm",
        "rms_norm",
    ]
    assert "ttnn.reshape(qg, (1, QUERY_HEADS_PER_DEVICE, 1, 2 * HEAD_DIM))" in source
    assert "(0, 0, 0, 0),\n            (1, QUERY_HEADS_PER_DEVICE, 1, HEAD_DIM)," in source
    assert "(0, 0, 0, HEAD_DIM),\n            (1, QUERY_HEADS_PER_DEVICE, 1, 2 * HEAD_DIM)," in source
    assert "ttnn.transpose(" not in source and "ttnn.chunk(" not in source
    # The half slices are copies; the alias guard runs before the pair tensor is released.
    assert source.index("if _tensor_key(qg_heads) in (_tensor_key(q), _tensor_key(gate)):") < source.index(
        "_deallocate(qg_heads)"
    )
    assert source.count("_retag_tensor(qg_heads, reference=full_hidden, shard_dim=1)") == 1


def test_pinned_runtime_tiled_reshape_is_one_program_without_an_explicit_pad_value() -> None:
    """[1,1,1,3072] -> [1,6,1,512] and [1,6,1,256] -> [1,1,1,1536] are neither views nor row-major: one prim::reshape_view."""

    reshape = _cpp("ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp")
    dispatch = reshape.split("bool this_is_view =", 1)[1]
    assert "(tensor_shape_last_dim == shape_last_dim) &&" in dispatch  # both diet reshapes change the last dim
    assert (
        "if (tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {\n        return operations::data_movement::detail::reshape_rm("
        in dispatch
    )
    assert "return operations::data_movement::reshape_tiled(" in dispatch
    tiled = reshape.split("ttnn::Tensor reshape_tiled(", 1)[1]
    interleaved = tiled.split("// Interleaved (DRAM / L1) tensors: call prim::reshape_view directly.", 1)[1]
    interleaved = interleaved.split("\n}\n", 1)[0]  # to the end of reshape_tiled
    assert interleaved.count("ttnn::prim::reshape_view(") == 1
    assert (
        "const bool should_fill = is_block_float_output || (pad_value_explicit && !skip_padding_fill);" in interleaved
    )
    assert interleaved.count("fill_implicit_tile_padding(") == 1  # gated on should_fill: no fill program for BF16
    assert "ttnn.reshape(qg, (1, QUERY_HEADS_PER_DEVICE, 1, 2 * HEAD_DIM))" in inspect.getsource(
        Qwen38TTNNQSA._main_projection
    )


def test_sparse_query_is_built_from_the_module_zero_half_with_one_row_major_pad() -> None:
    source = inspect.getsource(Qwen38TTNNQSA._sparse_value_attention)
    assert _ttnn_calls(source) == [
        "concat",
        "to_layout",
        "pad",
        "sparse_sdpa",
        "slice",
        "to_layout",
        "sigmoid",
        "mul",
        "reshape",
    ]
    assert "ttnn.zeros_like(" not in source and "ttnn.transpose(" not in source
    assert "local_flat = ttnn.reshape(gated, (1, 1, 1, LOCAL_QUERY_WIDTH))" in source
    assert "if not same_buffer(local_flat, gated):" in source
    assert "ttnn.concat([self.zero_value_half, query], dim=3" in source
    assert "ttnn.to_layout(\n            sparse_query_tiled, ttnn.ROW_MAJOR_LAYOUT" in source
    pad = source.split("sparse_query = ttnn.pad(", 1)[1].split("\n        )", 1)[0]
    assert (
        "sparse_query_row_major,\n            [(0, 0), (0, 32 - QUERY_HEADS_PER_DEVICE), (0, 0), (0, 0)],\n            0.0,"
        in pad
    )
    assert source.index("ttnn.to_layout(") < source.index("sparse_query = ttnn.pad(") < source.index("sparse_sdpa(")
    init = inspect.getsource(Qwen38TTNNQSA.__init__)
    assert (
        "self.zero_value_half = self._allocate_replicated_tile_zeros((1, QUERY_HEADS_PER_DEVICE, 1, HEAD_DIM))" in init
    )
    assert "_deallocate(self.index_gate, self.zero_value_half, *self.uint32_rows.values())" in inspect.getsource(
        Qwen38TTNNQSA.deallocate
    )
    # The row-major pad path takes any padded dim in one prim::pad program; the tile path rejects front padding
    # and needs a FillPad before the head-dim pad, which is why the pad moved after the untilize.
    pad_source = _cpp("ttnn/cpp/ttnn/operations/data_movement/pad/pad.cpp")
    row_major = pad_source[pad_source.index("ttnn::Tensor invoke_rm(") : pad_source.index("ttnn::Tensor invoke_tile(")]
    assert "pad_impl(input_tensor, padding_vec, value, use_multicore, memory_config_arg, sub_core_grids);" in row_major
    impl = pad_source.split("ttnn::Tensor pad_impl(", 1)[1].split("\n}\n", 1)[0]
    assert impl.count("return ttnn::prim::pad(") == 1
    assert "return !input_tensor.memory_config().is_l1();  // DRAM-sharded edge case only" in pad_source
    assert "input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&" in impl


def test_exact_diet_device_op_deltas_per_token() -> None:
    """Device ops saved per token against 450403c0 by lever (twelve QSA layers); only layout programs left."""

    layers = 12
    generic = {
        "row_fill OR-merge (one OR per layer, one more at derivation)": layers * 1 - 1,
        "all-gather writes the activation shard (no I2S copy)": layers * 1,
        "head split without transposes": layers * 2,
        "sparse query: module zero half, one row-major pad": layers * 2,
        "index query scored from its tile view (no untilize/pad/tilize)": layers * 3,
        "gated heads flattened without the transpose": layers * 1,
    }
    per_position = {
        label: saved for label, saved in generic.items() if "OR-merge" not in label and "tile view" not in label
    }
    assert list(generic.values()) == [11, 12, 24, 24, 36, 12]
    assert sum(generic.values()) == 119  # single-trace and residue-class traces (the generic body)
    assert sum(per_position.values()) == 72  # the per-position path (forward_decode)
    # QSA block ops per layer: generic body 90 -> 80 at every position; per-position residue 0 72 -> 66.
    assert 90 - (1 + 1 + 2 + 2 + 3 + 1) == 80 and 72 - (1 + 2 + 2 + 1) == 66
    # Dropped, not exact or not one program on the pinned runtime: Q4 permuted-head RoPE (changes the q.k tile
    # accumulation order), ttnn.split for the 128-wide index head (split.cpp takes the native TILE kernel only when
    # the second-last dim spans >= 2 tiles; a [1,1,1,128] row falls back to two slices), the sharded FP32 typecast +
    # reduce_scatter (unverified on this runtime), and the one-hot where-select (no two-way-broadcast ternary).
    split = _cpp("ttnn/cpp/ttnn/operations/data_movement/split/split.cpp")
    assert "input_shape[-2] / tt::constants::TILE_HEIGHT >= 2 &&" in split
