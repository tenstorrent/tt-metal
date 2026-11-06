# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device tests of the generic chain's paging plumbing: the index-paging validator, the page-table width and the
host image of the chain page table."""

import dataclasses

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import qsa
from models.demos.blackhole.qwen38_flash_next.ttnn.model import generic_page_table_host
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging, paged_table_width, validate_index_paging

CONTEXT = 8192


@pytest.mark.parametrize(
    "context, page_size, width", [(32768, 128, 264), (32768, 2048, 24), (8192, 128, 72), (262144, 4096, 72)]
)
def test_paged_table_width(context, page_size, width):
    assert paged_table_width(context, page_size) == width
    assert width % 8 == 0 and width * page_size >= context + 128


@pytest.mark.parametrize("page_size", [128, 256, 1024, 4096])
def test_validate_index_paging_accepts_the_table_width(page_size):
    width = paged_table_width(CONTEXT, page_size)
    paging = Qwen38LanePaging(page_size, 4 * width, width)
    assert validate_index_paging(paging, CONTEXT) is paging


@pytest.mark.parametrize(
    "paging, message",
    [
        (Qwen38LanePaging(64, 100, 8 * 160), "power of two"),
        (Qwen38LanePaging(8192, 100, 8), "power of two"),
        (Qwen38LanePaging(192, 100, 8 * 64), "power of two"),
        (Qwen38LanePaging(128, 100, 68), "multiple of 8"),
        (Qwen38LanePaging(128, 100, 64), "cover the allocated context"),
        (Qwen38LanePaging(128, 100, 8 * 8), "cover the allocated context"),
        (Qwen38LanePaging(128, 1, 72), "num_pages"),
    ],
)
def test_validate_index_paging_rejects(paging, message, expect_error):
    with expect_error(ValueError, message):
        validate_index_paging(paging, CONTEXT)


def test_validate_index_paging_requires_the_window_rows(expect_error):
    exact = Qwen38LanePaging(128, 100, CONTEXT // 128)
    exact.validate(CONTEXT)
    with expect_error(ValueError, "window"):
        validate_index_paging(exact, CONTEXT)


def test_generic_state_paging_fields_default_to_none():
    fields = {field.name: field for field in dataclasses.fields(qsa.Qwen38TTNNQSAGenericState)}
    assert fields["paging"].default is None and fields["page_table"].default is None
    assert [field.name for field in dataclasses.fields(qsa.Qwen38TTNNQSAPagedPools)] == ["kv", "index"]


def test_generic_page_table_host_rows():
    paging = Qwen38LanePaging(128, 20, 72)
    host = generic_page_table_host(paging, [3, 1, 19])
    assert host.shape == (1, 1, 32, 72) and host.dtype == torch.int32
    expected = torch.zeros(72, dtype=torch.int32)
    expected[:3] = torch.tensor([3, 1, 19], dtype=torch.int32)
    assert all(torch.equal(host[0, 0, row], expected) for row in range(32))
    assert not generic_page_table_host(paging, []).any()
    assert generic_page_table_host(paging, list(range(1, 20)) * 3 + [1] * 15)[0, 0, 7, 71] == 1


@pytest.mark.parametrize(
    "ids, message", [([1] * 73, "holds 72"), ([0, 1], "scratch"), ([20], r"\[1, 20\)"), ([-1], "scratch")]
)
def test_generic_page_table_host_rejects(ids, message, expect_error):
    with expect_error(ValueError, message):
        generic_page_table_host(Qwen38LanePaging(128, 20, 72), ids)


def test_generic_page_table_host_rejects_an_unpaged_state(expect_error):
    with expect_error(ValueError, "not paged"):
        generic_page_table_host(None, [1])


# ---------------------------------------------------------------- paged one-row decode: the page table reaches every op

import inspect

from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38TTNNQSA, Qwen38TTNNQSAGenericState


def _state(paging=None, table="TABLE"):
    return Qwen38TTNNQSAGenericState(
        layer_index=0,
        epoch=0,
        packed_kv_cache="kv",
        compressed_index_cache="index",
        kv_staging="staging",
        raw_key_ring="ring",
        paging=paging,
        page_table=None if paging is None else table,
    )


def _module():
    module = Qwen38TTNNQSA.__new__(Qwen38TTNNQSA)
    module.allocated_compressed_blocks = CONTEXT // 4
    return module


def test_paged_decode_keywords_are_empty_for_the_flat_state():
    module = _module()
    assert module._paged_kwargs(_state()) == {}
    assert module._paged_indexer_kwargs(_state()) == {}
    assert not module._generic_paged(object())


def test_paged_decode_keywords_carry_the_chain_table():
    module = _module()
    state = _state(Qwen38LanePaging(256, 20, paged_table_width(CONTEXT, 256)))
    assert module._paged_kwargs(state) == {"page_table": "TABLE", "page_size": 256}
    assert module._paged_indexer_kwargs(state) == {
        "page_table": "TABLE",
        "paged_block_rows": 64,
        "paged_user_rows": CONTEXT // 4 + 32,
        "paged_wide": False,
        "cache_batch_idx": 0,
    }


@pytest.mark.parametrize(
    "method, needles",
    [
        (Qwen38TTNNQSA._write_compressed_index_generic, ["**self._paged_table_kwargs(state)"]),
        (Qwen38TTNNQSA._index_tail_step, ["**self._paged_kwargs(state)"]),
        (Qwen38TTNNQSA._score_blocks_fused, ["**self._paged_indexer_kwargs(state)"]),
        (Qwen38TTNNQSA._score_blocks_generic, ["**self._paged_indexer_kwargs(state)"]),
        (Qwen38TTNNQSA._main_tail_step, ["page_table", "page_size"]),
        (Qwen38TTNNQSA._materialize_row_generic, ["self._paged_selection_ids(sparse_indices, state)"]),
        (Qwen38TTNNQSA._write_packed_kv_generic, ["self._kv_write_row_start(state, position)", "row_start,"]),
        (Qwen38TTNNQSA._forward_decode_generic_fused, ["self._paged_kwargs(state), single_user"]),
    ],
)
def test_paged_decode_branches_pass_the_page_table(method, needles):
    source = inspect.getsource(method)
    for needle in needles:
        assert needle in source, needle


def test_paged_decode_helpers_use_the_chain_table():
    assert '"page_table": state.page_table' in inspect.getsource(Qwen38TTNNQSA._paged_table_kwargs)
    assert "paged_rows_composed(sparse_indices, state.page_table" in inspect.getsource(
        Qwen38TTNNQSA._paged_selection_ids
    )
    assert "paged_lane_row_starts_composed" in inspect.getsource(Qwen38TTNNQSA._kv_write_row_start)
    assert _module()._paged_table_kwargs(_state()) == {}
    assert _module()._paged_selection_ids("ids", _state()) == "ids"
    assert _module()._kv_write_row_start(_state(), type("P", (), {"kv_block_start": "start"})()) == ("start", ())


def test_paged_decode_callers_hand_the_state_to_the_composed_selection():
    for method in (Qwen38TTNNQSA.forward_decode_generic, Qwen38TTNNQSA._forward_decode_generic_fused):
        assert "_materialize_row_generic(masked_scores, position, state)" in inspect.getsource(method)


# ---------------------------------------------------------------- paged prefill: the page table reaches every op


@pytest.mark.parametrize(
    "method, needles",
    [
        (
            Qwen38TTNNQSA._write_packed_kv_chunk,
            ["self._generic_paged(state)", "kv_page_scatter(", "state.page_table", "position=chunk.kv_block_start"],
        ),
        (
            Qwen38TTNNQSA._write_compressed_index_chunk,
            ["index_page_fill(", "state.page_table", "chunk.kv_block_start", "**self._paged_table_kwargs(state)"],
        ),
        (Qwen38TTNNQSA._score_blocks_chunk, ["**self._paged_indexer_kwargs(state)"]),
        (Qwen38TTNNQSA._sparse_indices_slab, ["**self._paged_indexer_kwargs(state)", "_paged_selection_rows("]),
        (Qwen38TTNNQSA._materialize_rows_chunk, ["**self._paged_kwargs(state)", "_paged_selection_rows("]),
        (Qwen38TTNNQSA._block_shared_attention_rows, ["**self._paged_kwargs(state)"]),
        (Qwen38TTNNQSA._slab_attention_admits, ["state.page_table", "state.paging.page_size"]),
        (
            Qwen38TTNNQSA.forward_chunk_generic,
            ["_materialize_rows_chunk(masked_scores, chunk, constants, state=state)"],
        ),
    ],
)
def test_paged_prefill_branches_pass_the_page_table(method, needles):
    source = inspect.getsource(method)
    for needle in needles:
        assert needle in source, needle


def test_paged_prefill_selection_rows_translate_per_32_row_slice():
    source = inspect.getsource(Qwen38TTNNQSA._paged_selection_rows)
    assert "paged_rows_composed(" in source and "state.page_table" in source and "CHUNK_ROWS" in source
    assert _module()._paged_selection_rows("ids", _state()) == "ids"


def test_fused_selection_rows_forward_the_page_arguments():
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_rows

    source = inspect.getsource(qsa_rows.selection_rows)
    assert "page_table=page_table" in source and "page_size=page_size" in source
    parameters = inspect.signature(qsa_rows.selection_rows).parameters
    assert parameters["page_table"].default is None and parameters["page_size"].default == 0


def test_slab_attention_admits_uses_the_logical_length_when_paged(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import sparse_sdpa_tiled

    seen = {}

    def admits_shapes(heads, rows, length, width, dq, ids, config):
        seen["length"] = length
        return True

    monkeypatch.setattr(sparse_sdpa_tiled, "admits_shapes", admits_shapes)
    module = _module()
    module._slab_attention_fused = object()
    module._slab_attention_config = lambda rows, state: None
    pool = type("T", (), {"shape": (1, 1, 99 * 256, 512)})()
    table = type("T", (), {"shape": (1, 1, 32, 40)})()
    paged = dataclasses.replace(_state(Qwen38LanePaging(256, 20, 40), table), packed_kv_cache=pool)
    assert module._slab_attention_admits(512, paged)
    assert seen["length"] == 40 * 256
    flat = dataclasses.replace(_state(), packed_kv_cache=pool)
    assert module._slab_attention_admits(512, flat)
    assert seen["length"] == 99 * 256


# ---------------------------------------------------------------- paged MTP verify, draft and hand-offs


@pytest.mark.parametrize(
    "method, needles",
    [
        (
            Qwen38TTNNQSA.forward_verify_generic,
            ["_materialize_rows_chunk(", "live_rows=live_rows, state=state"],
        ),
        (Qwen38TTNNQSA._write_compressed_index_verify, ["**self._paged_table_kwargs(state)"]),
        (Qwen38TTNNQSA._main_tail_rows_step, ["**self._paged_kwargs(state)"]),
        (
            Qwen38TTNNQSA._write_packed_kv_verify,
            [
                "self._generic_paged(state)",
                "_physical_block_row_start(state, verify.chunk.kv_block_start)",
                "arange32_lanes",
            ],
        ),
        (Qwen38TTNNQSA._write_kv_block_verify, ["_physical_block_row_start(state, kv_block_start)", "row_start,"]),
        (
            Qwen38TTNNQSA.handoff_verify_state,
            ["self._generic_paged(state)", "_physical_block_row_start(state, logical_start)", "physical_start"],
        ),
        (
            Qwen38TTNNQSA.handoff_draft_state_device,
            ["_physical_block_row_start(state, kv_block_start)", "arange32_lanes, physical_start"],
        ),
    ],
)
def test_paged_verify_branches_pass_the_page_table(method, needles):
    source = inspect.getsource(method)
    for needle in needles:
        assert needle in source, needle


@pytest.mark.parametrize(
    "method",
    [
        Qwen38TTNNQSA.forward_verify_generic,
        Qwen38TTNNQSA.handoff_verify_state,
        Qwen38TTNNQSA.handoff_draft_state_device,
    ],
)
def test_verify_entry_points_no_longer_refuse_paged_pools(method):
    assert "NotImplementedError" not in inspect.getsource(method)


def test_physical_block_row_start_is_the_identity_when_flat():
    assert _module()._physical_block_row_start(_state(), "start") == ("start", ())


def test_physical_block_row_start_uses_the_chain_table_lookup():
    source = inspect.getsource(Qwen38TTNNQSA._physical_block_row_start)
    assert "paged_lane_row_starts_composed(block_start, state.page_table, state.paging.page_size, 1)" in source


def test_draft_state_inherits_the_paging_of_the_base_state():
    assert "replace(" in inspect.getsource(Qwen38TTNNQSA.allocate_draft_generic_state)
    fields = {f.name for f in dataclasses.fields(Qwen38TTNNQSAGenericState)}
    assert {"paging", "page_table"} <= fields


def test_forward_pack_accepts_paged_state():
    assert "NotImplementedError" not in inspect.getsource(Qwen38TTNNQSA.forward_pack)


class _FakeBuf:
    def __init__(self):
        self.freed = False


class _FakeTensor:
    def __init__(self, shape, buf=None, dtype="uint32", layout="rm"):
        self.shape = tuple(shape)
        self.buf = buf or _FakeBuf()
        self.dtype = dtype
        self.layout = layout


class _FakeTTNN:
    """Records deallocations per buffer; slice with full extents and reshape return views sharing the buffer."""

    DRAM_MEMORY_CONFIG = "dram"
    TILE_LAYOUT = "tile"
    ROW_MAJOR_LAYOUT = "rm"
    uint32 = "uint32"
    int32 = "int32"

    def __init__(self):
        self.freed_buffers = []

    def deallocate(self, tensor):
        tensor.buf.freed = True
        self.freed_buffers.append(tensor.buf)

    def slice(self, tensor, start, end, memory_config=None):
        shape = tuple(e - s for s, e in zip(start, end))
        return tensor if shape == tensor.shape else _FakeTensor(shape)

    def reshape(self, tensor, shape):
        return _FakeTensor(shape, tensor.buf, tensor.dtype)

    def to_layout(self, tensor, layout, memory_config=None):
        return tensor if tensor.layout == layout else _FakeTensor(tensor.shape, layout=layout)

    def _new(self, tensor, *_args, **_kwargs):
        return _FakeTensor(tensor.shape, layout=tensor.layout)

    eq = bitwise_right_shift = bitwise_left_shift = bitwise_and = add = multiply = bitwise_or = _new

    def where(self, _mask, _value, tensor, memory_config=None):
        return _FakeTensor(tensor.shape)

    def gather(self, _table, _dim, index):
        return _FakeTensor(index.shape)

    def typecast(self, tensor, _dtype, memory_config=None):
        return _FakeTensor(tensor.shape)


@pytest.mark.parametrize("lanes, width", [(1, 1), (1, 32), (4, 4), (4, 32)])
def test_paged_helpers_never_free_their_inputs(monkeypatch, lanes, width):
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

    fake = _FakeTTNN()
    monkeypatch.setattr(qsa_block, "ttnn", fake)
    monkeypatch.setattr(qsa_block, "page_table_blocks", lambda *_a, **_k: 8)
    page_table = _FakeTensor((1, 1, 32, 8), dtype="int32")
    block_start = _FakeTensor((1, 1, 1, width))
    values = _FakeTensor((1, 1, lanes, 16))
    outs = qsa_block.paged_lane_row_starts_composed(block_start, page_table, 128, lanes)
    out = qsa_block.paged_rows_composed(values, page_table, 128)
    assert not block_start.buf.freed and not page_table.buf.freed and not values.buf.freed
    assert all(not o.buf.freed for o in outs) and not out.buf.freed
    assert not any(b is block_start.buf or b is page_table.buf or b is values.buf for b in fake.freed_buffers)
