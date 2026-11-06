# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device tests of the plain lanes on the shared pools: the lane state takes the pools, and the lane decode's index
write and score pass the lane page table (the unpaged branches pass nothing new)."""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38TTNNQSA, Qwen38TTNNQSAPagedPools

BLOCKS = 1024
ROWS = BLOCKS + 32
PAGE = 128
NUM_PAGES = 40
LANES = 8
TABLE = "TABLE"


class _Tensor:
    def __init__(self, shape, layout=ttnn.TILE_LAYOUT):
        self.shape = tuple(shape)
        self.dtype = ttnn.bfloat16
        self.layout = layout


class _Contract:
    def validate_tensor(self, tensor, **kwargs):
        pass


class _Stop(Exception):
    pass


def _module():
    module = object.__new__(Qwen38TTNNQSA)
    module.layer_index = 0
    module._live_generic_epochs = set()
    module._next_epoch = 1
    module.allocated_compressed_blocks = BLOCKS
    module.allocated_context = 4096
    module.mesh_contract = _Contract()
    module.index_gate = module.indexer_compute_config = None
    module.indexer_chunk_start = 0
    return module


def _pools(num_pages=NUM_PAGES, page=PAGE):
    return Qwen38TTNNQSAPagedPools(
        kv=_Tensor((1, 1, num_pages * page, 2 * qsa_module.HEAD_DIM), ttnn.ROW_MAJOR_LAYOUT),
        index=_Tensor((num_pages, 1, page // 4, qsa_module.INDEX_HEAD_DIM)),
    )


def _constants(paged=True):
    paging = qsa_module.Qwen38LanePaging(PAGE, NUM_PAGES, 8 * ((4096 + 128 + PAGE - 1) // PAGE // 8 + 1))
    return SimpleNamespace(indexer_form="wide", paging=paging if paged else None, page_table=TABLE if paged else None)


@pytest.fixture
def allocation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Qwen38TTNNQSA,
        "_allocate_pair_grouped",
        lambda self, shape, layout: calls.append(("pair", shape)) or _Tensor(shape, layout),
    )
    monkeypatch.setattr(
        Qwen38TTNNQSA,
        "_allocate_replicated_tile_zeros",
        lambda self, shape: calls.append(("rep", shape)) or _Tensor(shape),
    )
    monkeypatch.setattr(ttnn.experimental, "view", lambda tensor, shape: _Tensor(shape))
    monkeypatch.setattr(qsa_module, "_retag_tensor", lambda *args, **kwargs: None)
    return calls


def test_lane_state_on_pools_uses_them(allocation):
    module, pools = _module(), _pools()
    state = module.allocate_lane_state(LANES, pools=pools)
    assert state.packed_kv_cache is pools.kv and state.compressed_index_cache is pools.index
    assert state.compressed_index_cache_flat is None and state.pools_shared
    assert state.pool_rows == NUM_PAGES * PAGE and not state.kv_scratch_rows
    assert [shape for kind, shape in allocation if kind == "pair"] == [(1, LANES, 32, 2 * qsa_module.HEAD_DIM)]
    assert [shape for kind, shape in allocation if kind == "rep"] == [(1, LANES, 32, qsa_module.INDEX_HEAD_DIM)]
    module._validate_lane_state(state)


def test_lane_state_without_pools_is_unchanged(allocation):
    state = _module().allocate_lane_state(LANES)
    assert not state.pools_shared and state.compressed_index_cache_flat is not None
    assert state.compressed_index_cache.shape == (LANES, 1, ROWS, qsa_module.INDEX_HEAD_DIM)


@pytest.mark.parametrize("extra", [{"kv_scratch_rows": 32}, {"pool_rows": 32}])
def test_lane_state_pools_reject_the_flat_forms(allocation, extra):
    with pytest.raises(ValueError, match="replace the state's caches"):  # allow-pytest.raises: host check
        _module().allocate_lane_state(LANES, pools=_pools(), **extra)


def test_lane_state_pools_reject_mismatched_shapes(allocation):
    pools = _pools()
    bad = Qwen38TTNNQSAPagedPools(
        kv=_Tensor((1, 1, 64, 2 * qsa_module.HEAD_DIM), ttnn.ROW_MAJOR_LAYOUT), index=pools.index
    )
    with pytest.raises(ValueError, match="K/V pool"):  # allow-pytest.raises: host check
        _module().allocate_lane_state(LANES, pools=bad)


def test_release_keeps_the_shared_pools(allocation, monkeypatch):
    module, pools = _module(), _pools()
    state = module.allocate_lane_state(LANES, pools=pools)
    freed = []
    monkeypatch.setattr(qsa_module, "_deallocate", lambda *tensors: freed.extend(tensors))
    module.release_lane_state(state)
    assert freed == [state.kv_staging, state.raw_key_ring]
    assert pools.kv not in freed and pools.index not in freed


def test_release_frees_the_per_lane_caches(allocation, monkeypatch):
    module = _module()
    state = module.allocate_lane_state(LANES)
    freed = []
    monkeypatch.setattr(qsa_module, "_deallocate", lambda *tensors: freed.extend(tensors))
    module.release_lane_state(state)
    assert state.packed_kv_cache in freed and state.compressed_index_cache in freed


# ---------------------------------------------------------------------------------- the lane decode's page arguments
def test_lane_index_tail_kwargs(allocation):
    module = _module()
    shared = module.allocate_lane_state(LANES, pools=_pools())
    assert module._lane_index_tail_kwargs(shared, _constants()) == {"page_table": TABLE, "page_size": PAGE}
    own = module.allocate_lane_state(LANES)
    assert module._lane_index_tail_kwargs(own, _constants()) == {}
    assert module._lane_index_tail_kwargs(own, _constants(paged=False)) == {}
    with pytest.raises(ValueError, match="paged lane constants"):  # allow-pytest.raises: host check
        module._lane_index_tail_kwargs(shared, _constants(paged=False))


def test_index_tail_step_passes_the_lane_keywords(monkeypatch):
    module, seen = _module(), []
    module.weights = SimpleNamespace(index_q=None, index_k=None, index_q_norm=None, index_k_norm=None)
    module.index_program_config = module.projection_compute_config = None

    def fused(*args, **kwargs):
        seen.append(kwargs)
        raise _Stop

    module._index_tail_fused = fused
    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: None)
    state = SimpleNamespace(raw_key_ring=None, compressed_index_cache=None)
    for extra, expected in (
        ({}, {}),
        ({"paged_kwargs": {"page_table": TABLE, "page_size": PAGE}}, {"page_table": TABLE, "page_size": PAGE}),
    ):
        with pytest.raises(_Stop):  # allow-pytest.raises: host check
            module._index_tail_step(None, None, None, None, None, state, None, rows=LANES, **extra)
        assert {k: v for k, v in seen[-1].items() if k in ("page_table", "page_size")} == expected


def test_lane_decode_source_passes_the_lane_table_to_the_index_tail():
    import inspect

    source = inspect.getsource(Qwen38TTNNQSA._forward_decode_lanes_fused)
    assert "paged_kwargs=self._lane_index_tail_kwargs(state, lane_constants)" in source


def _score_kwargs(monkeypatch, state, form, count=LANES, composed=False):
    captured = []

    def fake_indexer(*args, **kwargs):
        captured.append((args, kwargs))
        raise _Stop

    monkeypatch.setattr(ttnn, "reshape", lambda tensor, *shapes, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "Shape", lambda value: value)
    monkeypatch.setattr(qsa_module, "_retag_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(ttnn.experimental, "indexer_score_dsa", fake_indexer)
    module = _module()
    constants = _constants()
    constants.indexer_form = form
    method = module._score_blocks_lanes if composed else module._score_blocks_lanes_fused
    with pytest.raises(_Stop):  # allow-pytest.raises: host check
        method(None, state, SimpleNamespace(kv_block_start="kv_block_start"), constants)
    return captured[0]


@pytest.mark.parametrize("composed", [False, True])
def test_wide_indexer_scores_the_pool(allocation, monkeypatch, composed):
    module = _module()
    state = module.allocate_lane_state(LANES, pools=_pools())
    args, kwargs = _score_kwargs(monkeypatch, state, "wide", composed=composed)
    assert args[1] is state.compressed_index_cache
    assert kwargs["page_table"] == TABLE and kwargs["paged_block_rows"] == PAGE // 4
    assert kwargs["paged_user_rows"] == ROWS and kwargs["paged_wide"] is True
    assert kwargs["kv_len"] == LANES * ROWS and kwargs["chunk_start_idx"] == LANES * ROWS - ttnn.TILE_SIZE
    assert "cache_batch_idx" not in kwargs
    assert kwargs["consumed_rows"] == 1 and kwargs["valid_length_lane_rows"] == ROWS  # QWEN38_IDX_ROWWRITE
    if not composed:  # QWEN38_IDX_LEN: the fused wide form bounds each lane's keys by its kv_block_start
        assert kwargs["valid_length_tensor"] == "kv_block_start"


@pytest.mark.parametrize("composed", [False, True])
def test_per_lane_indexer_scores_the_pool(allocation, monkeypatch, composed):
    module = _module()
    state = module.allocate_lane_state(LANES, pools=_pools())
    args, kwargs = _score_kwargs(monkeypatch, state, "per_lane", composed=composed)
    assert args[1] is state.compressed_index_cache
    assert kwargs["page_table"] == TABLE and kwargs["paged_wide"] is False and kwargs["cache_batch_idx"] == 0
    assert kwargs["paged_user_rows"] == ROWS and "kv_len" not in kwargs


@pytest.mark.parametrize("composed", [False, True])
def test_unpaged_indexer_keeps_its_arguments(allocation, monkeypatch, composed):
    module = _module()
    state = module.allocate_lane_state(LANES)
    args, kwargs = _score_kwargs(monkeypatch, state, "wide", composed=composed)
    assert args[1] is state.compressed_index_cache_flat
    assert not {"page_table", "paged_block_rows", "paged_user_rows", "paged_wide", "kv_len"} & set(kwargs)
    args, kwargs = _score_kwargs(monkeypatch, state, "per_lane", composed=composed)
    assert not {"page_table", "paged_block_rows", "paged_user_rows", "paged_wide"} & set(kwargs)


def test_composed_write_passes_the_lane_table(allocation, monkeypatch):
    table = _Tensor((1, 1, 32, 8), ttnn.ROW_MAJOR_LAYOUT)
    monkeypatch.setattr(ttnn, "reshape", lambda tensor, shape, *a: _Tensor(tuple(shape), ttnn.ROW_MAJOR_LAYOUT))
    module = _module()
    ttn = ttnn
    seen = {}

    def fake_update(cache, users, **kwargs):
        seen.update(kwargs)
        raise _Stop

    for name in ("matmul", "multiply", "add", "sum", "to_layout", "pad", "rms_norm", "slice", "to_memory_config"):
        monkeypatch.setattr(ttn, name, lambda tensor=None, *a, **k: tensor)
    monkeypatch.setattr(ttn.experimental, "view", lambda tensor, shape: tensor)
    monkeypatch.setattr(qsa_module, "_deallocate", lambda *tensors: None)
    monkeypatch.setattr(qsa_module, "_tensor_key", lambda tensor: id(tensor))
    monkeypatch.setattr(qsa_module, "_require_shape", lambda *a, **k: None)
    monkeypatch.setattr(qsa_module, "apply_partial_rope_prefill", lambda tensor, *a: tensor)
    monkeypatch.setattr(ttn.experimental, "paged_update_cache", fake_update)
    module.compute_config = module.weights = None
    module.weights = SimpleNamespace(index_k_norm=None)
    module.rms_norm_eps = 1e-6
    lanes = SimpleNamespace(ring_hit_tiles=None, ring_keep_col=None, block_index_i32="IDX")
    for paged, expected in ((True, {"page_table": True}), (False, {})):
        state = module.allocate_lane_state(LANES, pools=_pools()) if paged else module.allocate_lane_state(LANES)
        constants = _constants(paged)
        constants.page_table = table if paged else None
        constants.compressed_rows_memory_config = None
        with pytest.raises(_Stop):  # allow-pytest.raises: host check
            module._write_compressed_index_lanes(state, _Tensor((1, 1, 32, 128)), None, None, lanes, constants)
        passed = seen.pop("page_table", None)
        assert (passed is not None) == bool(expected)
        if paged:
            assert passed.shape == (LANES, 8)
        assert seen == {"update_idxs_tensor": "IDX"}
        seen.clear()


class _TableTtnn:
    """Records the table ops of the lane update table; a full-extent slice returns its input like ``slice.cpp``."""

    def __init__(self):
        self.calls = []

    def slice(self, tensor, start, end, **kwargs):
        self.calls.append(("slice", tuple(end)))
        if tuple(end) == tuple(tensor.shape):
            return tensor
        return _Tensor(tuple(e - s for s, e in zip(start, end)), ttnn.ROW_MAJOR_LAYOUT)

    def reshape(self, tensor, shape, *args):
        self.calls.append(("reshape", tuple(shape)))
        return _Tensor(tuple(shape), ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("count", [1, 4, 8, 32])
def test_lane_update_table_is_two_dimensional_with_one_row_per_user_and_frees_only_its_slice(monkeypatch, count):
    module = _module()
    fake = _TableTtnn()
    monkeypatch.setattr(qsa_module.ttnn, "slice", fake.slice)
    monkeypatch.setattr(qsa_module.ttnn, "reshape", fake.reshape)
    full = _Tensor((1, 1, 32, 16), ttnn.ROW_MAJOR_LAYOUT)
    constants = SimpleNamespace(paging=object(), page_table=full)
    state = SimpleNamespace(pools_shared=True)
    table, owned = module._lane_update_page_table(state, constants, count)
    assert table.shape == (count, 16)
    assert owned is None if count == 32 else owned.shape == (1, 1, count, 16)
    assert owned is not full and table is not full
    unpaged = SimpleNamespace(pools_shared=False)
    assert module._lane_update_page_table(unpaged, constants, count) == (None, None)


# ------------------------------------------------------------------------------------- MTP width views on the pools
def _alias(tensor, *, lane_dim, per_lane=1):
    shape = list(tensor.shape)
    shape[lane_dim] = shape[lane_dim] // LANES * 2 * per_lane
    return _Tensor(shape, tensor.layout)


def test_width_view_of_a_pooled_state_keeps_the_pools_and_aliases_the_per_lane_tensors(allocation):
    module, pools = _module(), _pools()
    state = module.allocate_lane_state(LANES, pools=pools)
    view = module.lane_state_view(state, 2, _alias)
    assert view.packed_kv_cache is pools.kv and view.compressed_index_cache is pools.index
    assert view.compressed_index_cache_flat is None and view.pools_shared and view.pool_rows == state.pool_rows
    assert (view.lanes, view.stored_lanes, view.is_view, view.lane_alias, view.lane_prefix) == (
        2,
        LANES,
        True,
        True,
        True,
    )
    assert view.kv_staging is not state.kv_staging and view.kv_staging.shape[:2] == (1, 2)
    assert view.raw_key_ring.shape == (1, 2, 32, qsa_module.INDEX_HEAD_DIM)
    assert not state.lane_alias and not state.lane_prefix and view.epoch == state.epoch
    module._validate_lane_state(view)
    with pytest.raises(RuntimeError, match="bucket view"):  # allow-pytest.raises: no-device Python-level check
        module.release_lane_state(view)
    with pytest.raises(ValueError, match="flat parent state"):  # allow-pytest.raises: no-device Python-level check
        module.lane_state_view(view, 1, _alias)


def test_width_view_of_a_flat_state_sets_the_alias_flag(allocation):
    module = _module()
    state = module.allocate_lane_state(LANES, kv_scratch_rows=32)
    view = module.lane_state_view(state, 2, _alias)
    assert view.lane_alias and view.lane_prefix and view.compressed_index_cache_flat is not None
    assert view.packed_kv_cache is state.packed_kv_cache and not state.lane_alias
    module._validate_lane_state(view)


def test_constants_view_takes_a_parent_narrower_than_the_maximum(monkeypatch):
    built = []

    def build(cls, mesh_device, mesh_contract, **kwargs):
        built.append(kwargs)
        return SimpleNamespace(
            page_table=_Tensor((1,)), lanes=kwargs["lanes"], **{k: v for k, v in kwargs.items() if k != "lanes"}
        )

    monkeypatch.setattr(qsa_module.Qwen38TTNNQSALaneConstants, "build", classmethod(build))
    monkeypatch.setattr(qsa_module, "_deallocate", lambda *tensors: None)
    monkeypatch.setattr(qsa_module, "replace", lambda obj, **kw: SimpleNamespace(**{**vars(obj), **kw}))
    paging = qsa_module.Qwen38LanePaging(PAGE, NUM_PAGES, 8)
    parent = SimpleNamespace(lanes=8, paging=paging, page_table=TABLE, allocated_context=4096, indexer_form="wide")
    view = qsa_module.Qwen38TTNNQSALaneConstants.build_view("mesh", _Contract(), parent, lanes=4)
    assert view.lanes == 4 and view.page_table is TABLE and not view.owns_page_table
    with pytest.raises(ValueError):  # allow-pytest.raises: no-device Python-level check
        qsa_module.Qwen38TTNNQSALaneConstants.build_view("mesh", _Contract(), parent, lanes=16)
