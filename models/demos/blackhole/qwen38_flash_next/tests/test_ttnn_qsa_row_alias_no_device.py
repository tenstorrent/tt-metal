# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device ownership checks: the module rows and the shared per-token rows survive the selection path.

slice.cpp returns its input unchanged when the requested range covers the
whole tensor, so a full-width slice of a shared row is that row under a new
handle.  On the f1a465e3 lineage the selected-width slice of block_offsets
was full at 512 complete blocks and its release freed the row (next layer:
"Input Tensor B is not allocated", positions 2047 and later; fixed there by
35fef4a7).  This lineage expands at a fixed width and windows the step rows
at half their width, so no slice can alias a module row; the fakes below keep
the C++ contract (a full-width slice shares the buffer, anything narrower is
a fresh buffer, one allocation flag per buffer) and torch values for every
integer op, and drive the selection path exactly where the old path failed.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    BLOCK_TOPK,
    COMPRESS_RATIO,
    EXPANSION_GATHER_WIDTH,
    INDEX_HEAD_DIM,
    MAX_REUSE_TAIL,
    SHARED_ROW_CACHE,
    SPARSE_INDEX_CAPACITY,
    TOKEN_BUDGET,
    Qwen38TTNNQSA,
    emulate_block_expansion,
    emulate_sparse_row,
    qsa_row_constants,
    qsa_selection_geometry,
)

UINT32_MASK = 0xFFFFFFFF
ALLOCATED_CONTEXT = 32768
RM_GATHER_EXACT_WIDTH = 60 * ttnn.TILE_SIZE  # gather_device_operation.cpp GATHER_WT_THRESHOLD in elements


class _Buffer:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values
        self.allocated = True


class _Tensor:
    """Device tensor stand-in: metadata plus a buffer that every alias shares."""

    _ids = itertools.count(1)

    def __init__(self, values, *, dtype, layout, buffer: _Buffer | None = None, shape=None) -> None:
        self.buffer = buffer if buffer is not None else _Buffer(values)
        self.shape = tuple(int(item) for item in (self.buffer.values.shape if shape is None else shape))
        self.dtype = dtype
        self.layout = layout
        self.tensor_id = next(self._ids)

    @property
    def values(self) -> torch.Tensor:
        if not self.buffer.allocated:
            raise RuntimeError("Input Tensor is not allocated")  # binary_ng_device_operation.cpp:610
        return self.buffer.values

    def buffer_address(self) -> int:
        return id(self.buffer)


class _FakeOps:
    """The ttnn entry points the selection path uses, with slice.cpp's no-op aliasing."""

    def __init__(self) -> None:
        self.created: list[_Tensor] = []
        self.freed: list[int] = []
        self.gather_index_widths: list[int] = []
        self.last_block_ids: torch.Tensor | None = None
        self.generator = torch.Generator().manual_seed(2047)

    def _new(self, values, *, dtype, layout, buffer=None, shape=None) -> _Tensor:
        tensor = _Tensor(values, dtype=dtype, layout=layout, buffer=buffer, shape=shape)
        self.created.append(tensor)
        return tensor

    def uint32(self, values: torch.Tensor) -> _Tensor:
        return self._new(values.to(torch.int64) & UINT32_MASK, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def live_buffers(self) -> set[int]:
        return {id(tensor.buffer) for tensor in self.created if tensor.buffer.allocated}

    def slice(self, tensor, begins, ends, *, memory_config=None):
        begins, ends = tuple(begins), tuple(ends)
        if all(begin == 0 for begin in begins) and ends == tensor.shape:
            return self._new(None, dtype=tensor.dtype, layout=tensor.layout, buffer=tensor.buffer)
        window = tuple(slice(begin, end) for begin, end in zip(begins, ends))
        return self._new(tensor.values[window].clone(), dtype=tensor.dtype, layout=tensor.layout)

    def _binary(self, left, right, operation):
        other = right.values if isinstance(right, _Tensor) else right
        values = operation(left.values, other)
        if left.dtype == ttnn.uint32:
            values = values & UINT32_MASK
        return self._new(values, dtype=left.dtype, layout=left.layout)

    def add(self, left, right, *, memory_config=None):
        return self._binary(left, right, lambda a, b: a + b)

    def bitwise_and(self, left, right, *, memory_config=None):
        return self._binary(left, right, lambda a, b: a & b)

    def bitwise_or(self, left, right, *, memory_config=None):
        return self._binary(left, right, lambda a, b: a | b)

    def bitwise_left_shift(self, tensor, shift, *, memory_config=None):
        return self._new((tensor.values << shift) & UINT32_MASK, dtype=tensor.dtype, layout=tensor.layout)

    def gather(self, tensor, dim, index, *, memory_config=None):
        assert index.dtype == ttnn.uint32 and index.layout == tensor.layout
        # gather_device_operation.cpp:24-33 routes ROW_MAJOR index rows wider than 60 tiles to
        # RmSingleRowMultiCore, which the 2026-09-02 op repro on 4x p150 showed wrong for every
        # dtype (76% of the elements misplaced); the single-core factory below that width is exact.
        if tensor.layout == ttnn.ROW_MAJOR_LAYOUT and index.shape[-1] > RM_GATHER_EXACT_WIDTH:
            raise AssertionError(f"ROW_MAJOR gather with a {index.shape[-1]}-wide index takes the wrong factory")
        self.gather_index_widths.append(index.shape[-1])
        return self._new(tensor.values.gather(dim, index.values), dtype=tensor.dtype, layout=tensor.layout)

    def concat(self, tensors, dim, *, memory_config=None):
        values = torch.cat([tensor.values for tensor in tensors], dim=dim)
        return self._new(values, dtype=tensors[0].dtype, layout=tensors[0].layout)

    def reshape(self, tensor, shape, padded_shape=None):
        # The zero-cost TILE view: the same buffer under the 32-row logical shape.
        return self._new(None, dtype=tensor.dtype, layout=tensor.layout, buffer=tensor.buffer, shape=tuple(shape))

    def to_layout(self, tensor, layout, *, memory_config=None):
        return self._new(tensor.values.clone(), dtype=tensor.dtype, layout=layout)

    def pad(self, tensor, padding, value, *, memory_config=None):
        shape = [size + front + back for size, (front, back) in zip(tensor.shape, padding)]
        values = torch.full(shape, value, dtype=tensor.values.dtype)
        window = tuple(slice(front, front + size) for size, (front, _) in zip(tensor.shape, padding))
        values[window] = tensor.values
        return self._new(values, dtype=tensor.dtype, layout=tensor.layout)

    def indexer_score_dsa(self, query, cache, gate, *, chunk_start_idx, compute_kernel_config, kv_len, seq_shard_axes):
        assert query.shape == (1, 1, ttnn.TILE_SIZE, INDEX_HEAD_DIM) and kv_len % ttnn.TILE_SIZE == 0
        scores = torch.randn(1, 1, ttnn.TILE_SIZE, cache.shape[2], generator=self.generator).to(torch.bfloat16)
        return self._new(scores, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def all_reduce(self, tensor, *, cluster_axis, memory_config=None, topology=None):
        return self._new(tensor.values.clone(), dtype=tensor.dtype, layout=tensor.layout)

    def topk_large_indices(self, scores, *, k, valid_length):
        assert scores.shape == (1, 1, 1, ALLOCATED_CONTEXT // COMPRESS_RATIO)
        row = scores.values.reshape(-1)[:valid_length].float()
        ids = torch.topk(row, k=min(k, valid_length)).indices
        self.last_block_ids = ids.clone()
        padded = torch.zeros(k, dtype=torch.int64)
        padded[: ids.numel()] = ids
        return self._new(padded.reshape(1, 1, 1, k), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def deallocate(self, tensor) -> None:
        if not tensor.buffer.allocated:
            raise RuntimeError(f"double free of buffer {id(tensor.buffer)}")
        tensor.buffer.allocated = False
        self.freed.append(id(tensor.buffer))


def _install(monkeypatch) -> _FakeOps:
    ops = _FakeOps()
    for name in (
        "slice",
        "add",
        "bitwise_and",
        "bitwise_or",
        "bitwise_left_shift",
        "gather",
        "concat",
        "reshape",
        "to_layout",
        "pad",
        "all_reduce",
        "deallocate",
    ):
        monkeypatch.setattr(qsa_module.ttnn, name, getattr(ops, name))
    monkeypatch.setattr(qsa_module.ttnn.experimental, "indexer_score_dsa", ops.indexer_score_dsa)
    monkeypatch.setattr(qsa_module.ttnn.experimental, "topk_large_indices", ops.topk_large_indices)
    monkeypatch.setattr(qsa_module, "_retag_tensor", lambda tensor, *, reference, shard_dim: None)
    monkeypatch.setattr(Qwen38TTNNQSA, "_shared_rows", {})
    return ops


def _module(ops: _FakeOps, layer_index: int, mesh) -> Qwen38TTNNQSA:
    module = object.__new__(Qwen38TTNNQSA)
    module.mesh_device = mesh
    module.layer_index = layer_index
    module.regime_split = False
    module.allocated_context = ALLOCATED_CONTEXT
    module.allocated_compressed_blocks = ALLOCATED_CONTEXT // COMPRESS_RATIO
    module.index_gate = ops._new(
        torch.full((1, 1, ttnn.TILE_SIZE, 1), INDEX_HEAD_DIM**-0.5), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    module.indexer_compute_config = None
    module.collective_topology = None
    module.mesh_contract = SimpleNamespace(
        mark_local_partial=lambda *args, **kwargs: None, validate_tensor=lambda *args, **kwargs: None
    )
    module._trace_input_retention_active = False
    module._trace_retained_inputs = {}
    module._trace_rollover_staging = []
    module.uint32_rows = {name: ops.uint32(values) for name, values in qsa_row_constants().items()}
    for name, row in module.uint32_rows.items():
        setattr(module, name, row)
    return module


def _module_rows(module: Qwen38TTNNQSA) -> list[_Tensor]:
    return [module.index_gate, *module.uint32_rows.values()]


def _state(ops: _FakeOps, *, position: int, view_id: int = 1) -> SimpleNamespace:
    cache = ops._new(
        torch.zeros(1, 1, ALLOCATED_CONTEXT // COMPRESS_RATIO, INDEX_HEAD_DIM, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    return SimpleNamespace(compressed_index_cache=cache, next_position=position, epoch=1, view_id=view_id)


def _query(ops: _FakeOps) -> _Tensor:
    return ops._new(
        torch.randn(1, 1, 1, INDEX_HEAD_DIM, generator=ops.generator).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _expected_expansion(block_ids: torch.Tensor) -> torch.Tensor:
    return (block_ids * COMPRESS_RATIO).repeat_interleave(COMPRESS_RATIO) + torch.arange(COMPRESS_RATIO).repeat(
        block_ids.numel()
    )


def test_fake_slice_aliases_a_full_width_request_like_slice_cpp(expect_error, monkeypatch) -> None:
    """The harness reproduces the old defect: freeing a full-width slice frees the row it aliases."""

    ops = _install(monkeypatch)
    row = ops.uint32(torch.arange(TOKEN_BUDGET).reshape(1, 1, 1, -1))
    alias = ops.slice(row, (0, 0, 0, 0), (1, 1, 1, TOKEN_BUDGET))
    prefix = ops.slice(row, (0, 0, 0, 0), (1, 1, 1, TOKEN_BUDGET - 4))
    assert alias.buffer is row.buffer and alias.tensor_id != row.tensor_id
    assert alias.buffer_address() == row.buffer_address()
    assert prefix.buffer is not row.buffer and prefix.shape == (1, 1, 1, TOKEN_BUDGET - 4)
    qsa_module._deallocate(alias)
    with expect_error(RuntimeError, match="not allocated"):
        ops.add(ops.uint32(torch.zeros(1, 1, 1, TOKEN_BUDGET)), row)


def test_step_windows_are_half_row_slices_that_never_alias_the_step_rows(expect_error, monkeypatch) -> None:
    ops = _install(monkeypatch)
    module = _module(ops, layer_index=3, mesh=object())
    for count in (0, 1, 7, TOKEN_BUDGET, SPARSE_INDEX_CAPACITY):
        for step in (module.keep_step, module.sentinel_step):
            window = module._step_window(step, count)
            assert window.shape == (1, 1, 1, SPARSE_INDEX_CAPACITY) and window.buffer is not step.buffer
            ops.deallocate(window)
    for count in (-1, SPARSE_INDEX_CAPACITY + 1):
        with expect_error(ValueError, match="step window count"):
            module._step_window(module.keep_step, count)
    assert all(row.buffer.allocated for row in _module_rows(module))


@pytest.mark.parametrize(
    "complete_blocks", (1, 2, 8, 9, 10, 16, 31, 32, 33, 127, 128, 480, 481, 511, 512, 513, 1024, 8160, 8161)
)
def test_expansion_matches_the_host_emulation_with_two_half_width_gathers(monkeypatch, complete_blocks) -> None:
    """From one block to the last tile row of the cache (8161 takes the front-padded query fallback)."""

    ops = _install(monkeypatch)
    module = _module(ops, layer_index=3, mesh=object())
    state = _state(ops, position=complete_blocks * COMPRESS_RATIO - 1)
    expanded, token_count = module._score_complete_blocks(_query(ops), state, complete_blocks)
    selected = min(BLOCK_TOPK, complete_blocks)
    assert token_count == selected * COMPRESS_RATIO and expanded.shape == (1, 1, 1, TOKEN_BUDGET)
    assert ops.gather_index_widths == [EXPANSION_GATHER_WIDTH, EXPANSION_GATHER_WIDTH]
    padded_ids = torch.zeros(BLOCK_TOPK, dtype=torch.int64)
    padded_ids[:selected] = ops.last_block_ids
    assert torch.equal(expanded.values.reshape(-1), emulate_block_expansion(padded_ids << 2))
    assert torch.equal(expanded.values.reshape(-1)[:token_count], _expected_expansion(ops.last_block_ids))
    assert all(row.buffer.allocated for row in _module_rows(module))


@pytest.mark.parametrize("complete_blocks", (511, 512, 513))
def test_expansion_at_the_budget_keeps_the_module_rows_allocated_across_two_calls(monkeypatch, complete_blocks):
    ops = _install(monkeypatch)
    module = _module(ops, layer_index=3, mesh=object())
    state = _state(ops, position=complete_blocks * COMPRESS_RATIO - 1)
    queries = []
    for _ in range(2):  # the second call is the next layer at the same position
        query = _query(ops)
        queries.append(query)
        expanded, token_count = module._score_complete_blocks(query, state, complete_blocks)
        selected = min(BLOCK_TOPK, complete_blocks)
        assert token_count == selected * COMPRESS_RATIO and expanded.shape == (1, 1, 1, TOKEN_BUDGET)
        assert expanded.dtype == ttnn.uint32 and expanded.layout == ttnn.ROW_MAJOR_LAYOUT
        assert torch.equal(
            expanded.values.reshape(-1)[:token_count], _expected_expansion(ops.last_block_ids[:selected])
        )
        assert all(row.buffer.allocated for row in _module_rows(module))
        ops.deallocate(expanded)
    kept = {id(row.buffer) for row in _module_rows(module)} | {id(state.compressed_index_cache.buffer)}
    kept |= {id(query.buffer) for query in queries}  # forward_decode releases the index query after _select
    assert ops.live_buffers() == kept
    assert len(ops.freed) == len(set(ops.freed))  # no buffer released twice


def test_two_layers_share_the_per_token_rows_and_keep_every_module_row(monkeypatch) -> None:
    """Both regimes, layer after layer, position after position, then the shared-row eviction."""

    ops = _install(monkeypatch)
    mesh = object()
    layers = [_module(ops, layer_index=index, mesh=mesh) for index in (3, 7)]
    for regime_split in (False, True):
        for position in (2047, 2050, 2051, 2055):  # 512, 512, 513 and 514 complete blocks
            geometry = qsa_selection_geometry(position + 1)
            for module in layers:
                module.regime_split = regime_split
                state = _state(ops, position=position)
                selection = module._select(_query(ops), state, geometry.complete_blocks, None)
                if selection.complete_indices is None:
                    expected = emulate_sparse_row(
                        None, complete_token_count=0, tail_start=0, context_length=position + 1
                    )
                    assert regime_split and geometry.complete_blocks <= BLOCK_TOPK
                else:
                    expanded = selection.complete_indices.values.reshape(-1)
                    assert torch.equal(
                        expanded[: geometry.complete_token_count],
                        _expected_expansion(ops.last_block_ids[: geometry.selected_blocks]),
                    )
                    expected = emulate_sparse_row(
                        expanded,
                        complete_token_count=geometry.complete_token_count,
                        tail_start=geometry.tail_start,
                        context_length=position + 1,
                    )
                assert torch.equal(selection.sparse_indices.values.reshape(-1), expected)
                if selection.owns_sparse_indices:
                    ops.deallocate(selection.sparse_indices)
                if selection.complete_indices is not None:
                    ops.deallocate(selection.complete_indices)
            assert all(row.buffer.allocated for module in layers for row in _module_rows(module))
    shared_before = len(Qwen38TTNNQSA._shared_rows)
    assert shared_before <= SHARED_ROW_CACHE
    layers[0]._release_shared_rows(keep=0)
    assert not Qwen38TTNNQSA._shared_rows
    assert all(row.buffer.allocated for module in layers for row in _module_rows(module))
    assert len(ops.freed) == len(set(ops.freed))


def test_longest_reuse_tail_keeps_every_module_row(expect_error, monkeypatch) -> None:
    """An MTP reuse whose tail spans MAX_REUSE_TAIL tokens builds the row from the module rows without freeing them."""

    ops = _install(monkeypatch)
    module = _module(ops, layer_index=3, mesh=object())
    block_ids = torch.randperm(BLOCK_TOPK, generator=ops.generator)
    complete = ops.uint32(_expected_expansion(block_ids).reshape(1, 1, 1, -1))
    tail_start, context_length = 8, 8 + MAX_REUSE_TAIL
    for _ in range(2):  # two layers materialize the same frozen selection
        selection = module._materialize_selection(
            state=SimpleNamespace(next_position=context_length - 1, epoch=1),
            source_view_id=1,
            source_position=7,
            tail_start=tail_start,
            complete_indices=complete,
            complete_token_count=8,
            owns_complete_indices=False,
        )
        assert selection.valid_token_count == 8 + MAX_REUSE_TAIL
        expected = emulate_sparse_row(
            complete.values.reshape(-1), complete_token_count=8, tail_start=tail_start, context_length=context_length
        )
        assert torch.equal(selection.sparse_indices.values.reshape(-1), expected)
        assert all(row.buffer.allocated for row in _module_rows(module)) and complete.buffer.allocated
        ops.deallocate(selection.sparse_indices)
    with expect_error(ValueError, match="at most"):
        module._materialize_selection(
            state=SimpleNamespace(next_position=context_length, epoch=1),
            source_view_id=1,
            source_position=7,
            tail_start=tail_start,
            complete_indices=complete,
            complete_token_count=8,
            owns_complete_indices=False,
        )
    module._release_shared_rows(keep=0)
    kept = {id(row.buffer) for row in _module_rows(module)} | {id(complete.buffer)}
    assert ops.live_buffers() == kept
    assert len(ops.freed) == len(set(ops.freed))
