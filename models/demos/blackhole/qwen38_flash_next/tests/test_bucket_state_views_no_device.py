# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Bucket views of the lane states (GDN, PLE, QSA) and of the QSA lane constants, without a device.

A view holds the parent's tensors (lane extent ``storage_lanes``) and uses lanes ``0..lanes-1``.  Not testable here:
the fused kernels' acceptance of the view and the numerics (device tests)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MAX_LANES

PARENT, VIEW = 32, 8


class _Tensor:
    _next = 0

    def __init__(self, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        type(self)._next += 1
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.tensor_id = type(self)._next


class _Contract:
    def validate_tensor(self, tensor, **kwargs):
        pass


# ---------------------------------------------------------------------------------------------------- GDN
def _gdn_state(batch: int, stored: int, *, storage_lanes: int | None = None):
    recurrent = (stored, gdn_module.VALUE_HEADS_PER_DEVICE, gdn_module.HEAD_DIM, gdn_module.HEAD_DIM)
    conv = (1, 1, stored, gdn_module.QKV_WIDTH_PER_DEVICE)
    return gdn_module.Qwen38TTNNGDNState(
        layer_index=0,
        recurrent=_Tensor(recurrent, ttnn.float32),
        conv=tuple(_Tensor(conv) for _ in range(gdn_module.CONV_KERNEL_SIZE)),
        zero_recurrent=_Tensor(recurrent, ttnn.float32),
        zero_conv=_Tensor(conv),
        mesh_contract=_Contract(),
        batch_size=batch,
        storage_lanes=stored if storage_lanes is None else storage_lanes,
    )


def test_gdn_view_validates_and_flags():
    view = _gdn_state(VIEW, PARENT)
    view.validate()
    assert view.stored_lanes == PARENT and view.is_view
    plain = _gdn_state(VIEW, VIEW, storage_lanes=0)
    plain.validate()
    assert plain.stored_lanes == VIEW and not plain.is_view
    assert not _gdn_state(VIEW, VIEW, storage_lanes=VIEW).is_view


def test_gdn_storage_below_batch_fails():
    state = _gdn_state(VIEW, VIEW, storage_lanes=4)
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        RuntimeError
    ):
        state.validate()


def test_gdn_view_shape_mismatch_fails():
    state = _gdn_state(VIEW, VIEW, storage_lanes=PARENT)  # tensors of 8 lanes claimed as a 32-lane parent's
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        RuntimeError
    ):
        state.validate()


def test_gdn_view_refuses_lifecycle_methods():
    view = _gdn_state(VIEW, PARENT)
    snapshot = SimpleNamespace(layer_index=0, captured=True)
    for call in (
        view.deallocate,
        view.reset_inplace,
        lambda: view.reset_lane_inplace(0),
        view.allocate_snapshot,
        lambda: view.capture_into(snapshot),
        lambda: view.restore_from(snapshot),
    ):
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            RuntimeError, match="bucket view; use the parent state"
        ):
            call()
    # the step body's window methods stay allowed
    view.conv_window()
    view.advance_conv_window()
    view.shift_conv_window()


def test_gdn_composed_lane_step_refuses_a_view():
    layer = object.__new__(gdn_module.Qwen38TTNNGDN)
    layer.weights = SimpleNamespace(layer_index=0)
    layer.mesh_contract = _gdn_state(VIEW, PARENT).mesh_contract
    view = _gdn_state(VIEW, PARENT)
    view.mesh_contract = layer.mesh_contract
    assert layer._validate_lane_state(view) == VIEW  # the fused path's entry accepts it
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError, match="composed GDN lane step does not take a bucket view"
    ):
        layer.forward_decode_lanes(None, view)
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError, match="composed GDN lane step does not take a bucket view"
    ):
        layer._recurrent_decode_lanes(None, None, None, None, None, (), view, VIEW)


# ---------------------------------------------------------------------------------------------------- PLE
def _ple_state(lanes: int, stored: int, *, storage_lanes: int | None = None):
    shape = ple_module._lanes_residual_shape(stored)
    return ple_module.Qwen38TTNNPLELanesState(
        lanes=lanes,
        conv=tuple(_Tensor(shape) for _ in range(ple_module.CONV_STATE_LENGTH)),
        zero_conv=_Tensor(shape),
        mesh_contract=_Contract(),
        token_contexts=(None,) * lanes,
        storage_lanes=stored if storage_lanes is None else storage_lanes,
    )


def test_ple_view_validates_and_flags():
    view = _ple_state(VIEW, PARENT)
    view.validate()
    assert view.stored_lanes == PARENT and view.is_view and len(view.token_contexts) == VIEW
    plain = _ple_state(VIEW, VIEW, storage_lanes=0)
    plain.validate()
    assert plain.stored_lanes == VIEW and not plain.is_view


def test_ple_storage_below_lanes_fails():
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        RuntimeError
    ):
        _ple_state(VIEW, VIEW, storage_lanes=4).validate()


def test_ple_view_refuses_lifecycle_methods():
    view = _ple_state(VIEW, PARENT)
    for call in (view.reset_inplace, lambda: view.reset_lane_inplace(0), view.deallocate):
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            RuntimeError, match="bucket view; use the parent state"
        ):
            call()


def test_ple_composed_lanes_step_refuses_a_view_and_fused_does_not():
    ple = object.__new__(ple_module.Qwen38TTNNPLE)
    view = _ple_state(VIEW, PARENT)
    ple._fused_forward_prepared_lanes = None
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError, match="composed PLE lanes step does not take a bucket view"
    ):
        ple.forward_prepared_lanes(None, None, view)
    seen = []
    ple._fused_forward_prepared_lanes = lambda residual, prepared, state: seen.append(state) or "fused"
    assert ple.forward_prepared_lanes(None, None, view) == "fused" and seen == [view]


# ---------------------------------------------------------------------------------------------------- QSA state
BLOCKS = 64
ROWS = BLOCKS + ttnn.TILE_SIZE
POOL = 4096


def _qsa_module(epoch: int = 1):
    qsa = object.__new__(qsa_module.Qwen38TTNNQSA)
    qsa.layer_index = 0
    qsa._live_generic_epochs = {epoch}
    qsa.allocated_compressed_blocks = BLOCKS
    qsa.allocated_context = 4096
    qsa.mesh_contract = _Contract()
    return qsa


def _qsa_state(lanes: int, stored: int, *, pool_rows: int = POOL, storage_lanes: int | None = None):
    hd = qsa_module.HEAD_DIM
    ihd = qsa_module.INDEX_HEAD_DIM
    cw = qsa_module.CACHE_WRITE_ROWS
    return qsa_module.Qwen38TTNNQSALaneState(
        layer_index=0,
        epoch=1,
        lanes=lanes,
        packed_kv_cache=_Tensor((1, 1, pool_rows or stored * 4096, 2 * hd), layout=ttnn.ROW_MAJOR_LAYOUT),
        compressed_index_cache=_Tensor((stored, 1, ROWS, ihd)),
        compressed_index_cache_flat=_Tensor((1, 1, stored * ROWS, ihd)),
        kv_staging=_Tensor((1, stored, cw, 2 * hd)),
        raw_key_ring=_Tensor((1, stored, cw, ihd)),
        pool_rows=pool_rows,
        storage_lanes=stored if storage_lanes is None else storage_lanes,
    )


def test_qsa_view_validates_and_flags():
    qsa = _qsa_module()
    view = _qsa_state(VIEW, PARENT)
    qsa._validate_lane_state(view)
    assert view.stored_lanes == PARENT and view.is_view
    plain = _qsa_state(VIEW, VIEW, storage_lanes=0)
    qsa._validate_lane_state(plain)
    assert plain.stored_lanes == VIEW and not plain.is_view


def test_qsa_storage_below_lanes_fails():
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        RuntimeError
    ):
        _qsa_module()._validate_lane_state(_qsa_state(VIEW, VIEW, storage_lanes=4))


def test_qsa_view_must_be_paged():
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError, match="paged"
    ):
        _qsa_module()._validate_lane_state(_qsa_state(VIEW, PARENT, pool_rows=0))


def test_qsa_view_refuses_lifecycle_methods():
    qsa = _qsa_module()
    view = _qsa_state(VIEW, PARENT)
    for call in (
        lambda: qsa.release_lane_state(view),
        lambda: qsa.reset_lane_state_inplace(view),
        lambda: qsa.reset_lane_inplace(view, 0),
    ):
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            RuntimeError, match="bucket view"
        ):
            call()
    assert qsa._live_generic_epochs == {1}


def test_qsa_composed_lane_paths_refuse_a_view():
    qsa = _qsa_module()
    view = _qsa_state(VIEW, PARENT)
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError, match="bucket view"
    ):
        qsa._write_compressed_index_lanes(view, None, None, None, None, None)
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError, match="bucket view"
    ):
        qsa._score_blocks_lanes(None, view, None, None)


class _Stop(Exception):
    pass


def _score_kwargs(monkeypatch, state, form: str = "wide"):
    captured = {}

    def fake_indexer(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(ttnn, "reshape", lambda tensor, *shapes, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "Shape", lambda value: value)
    monkeypatch.setattr(qsa_module, "_retag_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(ttnn.experimental, "indexer_score_dsa", fake_indexer)
    qsa = _qsa_module()
    qsa.index_gate = qsa.indexer_compute_config = None
    with pytest.raises(_Stop):  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        qsa._score_blocks_lanes_fused(None, state, None, SimpleNamespace(indexer_form=form))
    return captured


def test_qsa_wide_score_passes_kv_len_only_for_a_view(monkeypatch):
    view_kwargs = _score_kwargs(monkeypatch, _qsa_state(VIEW, PARENT))
    assert view_kwargs["kv_len"] == VIEW * ROWS
    assert view_kwargs["chunk_start_idx"] == VIEW * ROWS - ttnn.TILE_SIZE
    plain_kwargs = _score_kwargs(monkeypatch, _qsa_state(VIEW, VIEW, storage_lanes=0))
    assert "kv_len" not in plain_kwargs


def test_qsa_per_lane_indexer_refuses_a_view(monkeypatch):
    monkeypatch.setattr(ttnn, "reshape", lambda tensor, *shapes, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "Shape", lambda value: value)
    monkeypatch.setattr(qsa_module, "_retag_tensor", lambda *args, **kwargs: None)
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError, match="bucket view"
    ):
        _qsa_module()._score_blocks_lanes_fused(
            None, _qsa_state(VIEW, PARENT), None, SimpleNamespace(indexer_form="per_lane")
        )


# ---------------------------------------------------------------------------------------------------- QSA constants
def _constants(lanes: int, page_table, paging=True):
    return qsa_module.Qwen38TTNNQSALaneConstants(
        lanes=lanes,
        allocated_context=4096,
        indexer_form="wide",
        arange32_tile=_Tensor((1,)),
        lane_indicator=_Tensor((1,)),
        kv_offsets_row=_Tensor((1,)),
        block_offsets_lanes=_Tensor((1,)),
        arange_slots_lanes=_Tensor((1,)),
        compressed_rows_memory_config=None,
        paging=qsa_module.Qwen38LanePaging(page_size=64, num_pages=64, max_blocks=64) if paging else None,
        page_table=page_table,
    )


@pytest.fixture
def fake_build(monkeypatch):
    calls, freed = [], []

    def build(cls, mesh_device, mesh_contract, **kwargs):
        calls.append(kwargs)
        return _constants(kwargs["lanes"], _Tensor((1,)), paging=kwargs["paging"] is not None)

    monkeypatch.setattr(qsa_module.Qwen38TTNNQSALaneConstants, "build", classmethod(build))
    monkeypatch.setattr(qsa_module, "_deallocate", lambda *tensors: freed.extend(t for t in tensors if t is not None))
    return calls, freed


def test_build_view_shares_the_parent_page_table(fake_build):
    calls, freed = fake_build
    parent = _constants(MAX_LANES, _Tensor((1,)))
    view = qsa_module.Qwen38TTNNQSALaneConstants.build_view("mesh", _Contract(), parent, lanes=VIEW)
    assert calls == [dict(lanes=VIEW, allocated_context=4096, indexer_form="wide", paging=parent.paging)]
    assert view.page_table is parent.page_table and not view.owns_page_table and view.lanes == VIEW
    assert len(freed) == 1 and freed[0] is not parent.page_table  # only the freshly built table
    freed.clear()
    view.deallocate()
    assert parent.page_table not in freed and len(freed) == 5
    freed.clear()
    parent.deallocate()
    assert parent.page_table in freed and len(freed) == 6


def test_build_view_rejects_bad_arguments(fake_build):
    build_view = qsa_module.Qwen38TTNNQSALaneConstants.build_view
    paged = _constants(MAX_LANES, _Tensor((1,)))
    with pytest.raises(ValueError):  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        build_view("mesh", _Contract(), _constants(MAX_LANES, None, paging=False), lanes=VIEW)
    with pytest.raises(ValueError):  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        build_view("mesh", _Contract(), _constants(VIEW, _Tensor((1,))), lanes=4)
    for lanes in (0, MAX_LANES, True):
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError
        ):
            build_view("mesh", _Contract(), paged, lanes=lanes)
