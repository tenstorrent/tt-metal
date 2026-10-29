# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bucket views of the lane state, no device: a lane state of R < 32 lanes that holds the 32-lane parent's tensors.
The layer and model allocators run over fakes (the MoE class, the attention rows upload, the lane constants and the
per-layer allocators are replaced); the checks are the view's shape (lanes, storage_lanes), what it owns, what a release
frees (never a parent tensor), the validation of a view and the guards of the parent-only operations."""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import Qwen38TTNNDevicePositionRow
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN, Qwen38TTNNGDNState
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import Qwen38TTNNPLELanesState
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38TTNNQSA

Qwen38TTNNLayerNamespace = layer_module.Qwen38TTNNLayerNamespace

PARENT_LANES = 32
VIEW_LANES = 8
MAX_BLOCKS = 4


def _fake(cls, **values):
    """A dataclass instance with every field without a default None, then ``values``."""

    fields = {
        f.name: None
        for f in dataclasses.fields(cls)
        if f.init and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    fields.update(values)
    return cls(**fields)


def _bare(cls, **attributes):
    instance = object.__new__(cls)
    for name, value in attributes.items():
        object.__setattr__(instance, name, value)
    return instance


class _Tensor:
    """A device tensor stand-in; ``freed`` lists every deallocation (the shared log)."""

    def __init__(self, name, freed):
        self.name = name
        self._freed = freed

    def deallocate(self):
        self._freed.append(self.name)


class _FakeMoe:
    instances: list = []

    def __init__(self, mesh_device, mesh_contract, weights, *, rows, freed=None, **kwargs):
        self.rows = rows
        self.weights = weights
        self.kwargs = kwargs
        self.released = 0
        _FakeMoe.instances.append(self)

    def release_owned_buffers(self):
        self.released += 1


@pytest.fixture
def freed(monkeypatch):
    """The deallocation log; the layer's and the model's ``ttnn.deallocate`` and ``_deallocate_unique`` write to it."""

    log: list = []
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: log.append(tensor.name))
    monkeypatch.setattr(layer_module, "_deallocate_unique", lambda *tensors: log.extend(t.name for t in tensors))
    monkeypatch.setattr(layer_module, "_tensor_key", lambda tensor: ("fake", id(tensor)))
    return log


@pytest.fixture
def fake_uploads(monkeypatch, freed):
    """The rows uploads of the allocators (attention rows, embedding rows) as named fake tensors."""

    uploads: list = []

    def from_torch(torch_tensor, **kwargs):
        tensor = _Tensor(f"rows{tuple(torch_tensor.shape)}#{len(uploads)}", freed)
        uploads.append(tensor)
        return tensor

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "ShardTensor2dMesh", lambda *args, **kwargs: None)
    monkeypatch.setattr(layer_module, "Qwen38TTNNMoE", _FakeMoe)
    _FakeMoe.instances = []
    return uploads


# --------------------------------------------------------------------------- the layer


def _layer(kind, freed, *, with_ple=False):
    weights = object()
    mlp = SimpleNamespace(
        mesh_device="mesh",
        mesh_contract="contract",
        weights=weights,
        tt_ccl=None,
        collective_topology=None,
        synchronization_policy=None,
    )
    attention = _bare(Qwen38TTNNGDN if kind == "gdn" else Qwen38TTNNQSA, mesh_device="mesh")
    layer = _bare(
        layer_module.Qwen38TTNNDecoderLayer,
        namespace=Qwen38TTNNLayerNamespace.BACKBONE,
        layer_index=3,
        attention=attention,
        mlp=mlp,
        ple=object() if with_ple else None,
    )
    layer._validate_rows = lambda *args, **kwargs: None  # the rows' device metadata is the uploads' (faked) business
    return layer


def _parent_layer_state(layer, freed, *, with_ple=False):
    if isinstance(layer.attention, Qwen38TTNNGDN):
        attention = _fake(Qwen38TTNNGDNState, batch_size=PARENT_LANES)
        rows = None
    else:
        attention = _fake(qsa_module.Qwen38TTNNQSALaneState, lanes=PARENT_LANES)
        rows = None
    ple = (
        _fake(Qwen38TTNNPLELanesState, lanes=PARENT_LANES, token_contexts=(None,) * PARENT_LANES) if with_ple else None
    )
    moe = SimpleNamespace(
        rows=PARENT_LANES, weights=layer.mlp.weights, release_owned_buffers=lambda: freed.append("moe32")
    )
    return layer_module.Qwen38TTNNDecoderLayerLaneState(
        layer.namespace, layer.layer_index, PARENT_LANES, attention, ple, moe, rows
    )


@pytest.mark.parametrize("kind", ["gdn", "qsa"])
@pytest.mark.parametrize("with_ple", [False, True])
def test_layer_bucket_view_shares_the_parent_tensors_and_owns_its_rows(kind, with_ple, fake_uploads, freed):
    layer = _layer(kind, freed, with_ple=with_ple)
    parent = _parent_layer_state(layer, freed, with_ple=with_ple)
    view = layer.allocate_bucket_lane_state(parent, VIEW_LANES)
    assert view.lanes == VIEW_LANES and view.namespace is parent.namespace and view.layer_index == parent.layer_index
    assert view.attention.is_view and view.attention.stored_lanes == PARENT_LANES
    if kind == "gdn":
        assert view.attention.batch_size == VIEW_LANES
    else:
        assert view.attention.lanes == VIEW_LANES
        assert len(fake_uploads) == 1 and view.attention_rows is fake_uploads[0]
        assert "rows(1, 1, 8, 2560)" in view.attention_rows.name
    if kind == "gdn":
        assert view.attention_rows is None and not fake_uploads
    if with_ple:
        assert view.ple.lanes == VIEW_LANES and view.ple.stored_lanes == PARENT_LANES
        assert view.ple.token_contexts == (None,) * VIEW_LANES
        assert parent.ple.lanes == PARENT_LANES and parent.ple.token_contexts == (None,) * PARENT_LANES
    else:
        assert view.ple is None
    # a NEW rows-R MoE over the layer's weights; the parent's is untouched
    assert len(_FakeMoe.instances) == 1 and view.moe is _FakeMoe.instances[0] and view.moe is not parent.moe
    assert view.moe.rows == VIEW_LANES and view.moe.weights is layer.mlp.weights
    # the parent state itself is unchanged
    assert not parent.attention.is_view and parent.lanes == PARENT_LANES and parent.moe.rows == PARENT_LANES


@pytest.mark.parametrize("kind", ["gdn", "qsa"])
def test_layer_bucket_release_frees_only_what_the_view_owns(kind, fake_uploads, freed):
    layer = _layer(kind, freed)
    parent = _parent_layer_state(layer, freed)
    view = layer.allocate_bucket_lane_state(parent, VIEW_LANES)
    layer.release_bucket_lane_state(view)
    assert view.moe.released == 1
    assert freed == ([] if kind == "gdn" else [fake_uploads[0].name])  # the parent's moe32 and tensors never freed


def test_layer_bucket_release_and_release_lane_state_reject_the_wrong_state(fake_uploads, freed, expect_error):
    layer = _layer("qsa", freed)
    parent = _parent_layer_state(layer, freed)
    view = layer.allocate_bucket_lane_state(parent, VIEW_LANES)
    with expect_error(ValueError, "bucket view"):
        layer.release_lane_state(view)
    with expect_error(ValueError, "release_lane_state"):
        layer.release_bucket_lane_state(parent)
    assert freed == []


@pytest.mark.parametrize("lanes", [0, 32, 33, True, 8.0])
def test_layer_bucket_view_lane_count_range(lanes, fake_uploads, freed, expect_error):
    layer = _layer("gdn", freed)
    with expect_error(ValueError, "bucket lanes"):
        layer.allocate_bucket_lane_state(_parent_layer_state(layer, freed), lanes)
    assert not _FakeMoe.instances


def test_layer_bucket_view_needs_the_32_lane_parent(fake_uploads, freed, expect_error):
    layer = _layer("gdn", freed)
    parent = _parent_layer_state(layer, freed)
    small = dataclasses.replace(
        parent,
        lanes=16,
        attention=_fake(Qwen38TTNNGDNState, batch_size=16),
        moe=SimpleNamespace(rows=16, weights=layer.mlp.weights),
    )
    with expect_error(ValueError, "32-lane parent"):
        layer.allocate_bucket_lane_state(small, VIEW_LANES)


def test_layer_bucket_allocation_failure_frees_only_what_it_created(fake_uploads, freed, monkeypatch):
    layer = _layer("qsa", freed)
    parent = _parent_layer_state(layer, freed)
    real_validate = layer._validate_lane_state

    def validate(state):
        if state.lanes != PARENT_LANES:
            raise RuntimeError("boom")
        return real_validate(state)

    layer._validate_lane_state = validate
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        RuntimeError, match="boom"
    ):
        layer.allocate_bucket_lane_state(parent, VIEW_LANES)
    assert _FakeMoe.instances[0].released == 1
    assert freed == [fake_uploads[0].name]  # the new attention rows only; no parent tensor, no parent moe


# --------------------------------------------------------------------------- the model


class _Calls:
    def __init__(self):
        self.log: list = []

    def recorder(self, name):
        return lambda *args, **kwargs: self.log.append(name)


def _model(layer_states_by_lanes=None):
    layers = [
        SimpleNamespace(
            layer_index=index,
            ple=SimpleNamespace(prepare_lanes_input=None) if index == model_module.PLE_CHECKPOINT_LAYER else None,
        )
        for index in range(model_module.BACKBONE_LAYERS)
    ]
    model = _bare(
        model_module.Qwen38TTNNTextModel,
        layers=layers,
        allocated_context=4096,
        mesh_device="mesh",
        mesh_contract="contract",
        _poisoned_error=None,
        _state_owner=object(),
        _poisoned_device_owners=[],
        _active_snapshot=None,
    )
    return model


def _paging():
    return SimpleNamespace(max_blocks=MAX_BLOCKS)


def _constants(lanes, *, owns_page_table, page_table, paging=True):
    return SimpleNamespace(
        lanes=lanes,
        allocated_context=4096,
        paging=_paging() if paging else None,
        page_table=page_table if paging else None,
        owns_page_table=owns_page_table,
        deallocate=lambda: None,
    )


def _layer_states(lanes, *, view, freed):
    states = []
    for index in range(model_module.BACKBONE_LAYERS):
        attention = SimpleNamespace(is_view=view)
        states.append(
            SimpleNamespace(
                namespace=Qwen38TTNNLayerNamespace.BACKBONE,
                layer_index=index,
                lanes=lanes,
                attention=attention,
                ple=SimpleNamespace(token_contexts=(None,) * lanes)
                if index == model_module.PLE_CHECKPOINT_LAYER
                else None,
            )
        )
    return states


def _position(model, lanes, freed):
    return _bare(
        Qwen38TTNNDevicePositionRow, lanes=lanes, residue_free=True, deallocate=lambda: freed.append("position")
    )


def _page_table():
    return SimpleNamespace(shape=(1, 1, PARENT_LANES, MAX_BLOCKS), dtype=ttnn.int32)


def _parent(model, freed):
    table = _page_table()
    state = model_module.Qwen38TTNNTextModelLaneState(
        lanes=PARENT_LANES,
        position=_position(model, PARENT_LANES, freed),
        layers=tuple(_layer_states(PARENT_LANES, view=False, freed=freed)),
        token_row=_Tensor("token_row", freed),
        ple_rows=SimpleNamespace(
            active=True, source_contexts=(None,) * PARENT_LANES, release=lambda: freed.append("ple32")
        ),
        embedding_rows=None,
        qsa_chunk_constants=SimpleNamespace(deallocate=lambda: freed.append("chunk")),
        qsa_lane_constants=_constants(PARENT_LANES, owns_page_table=True, page_table=table),
        _owner=model._state_owner,
    )
    return state


def _view(model, parent, freed, *, lanes=VIEW_LANES, **overrides):
    values = dict(
        lanes=lanes,
        position=parent.position,
        layers=tuple(_layer_states(lanes, view=True, freed=freed)),
        token_row=parent.token_row,
        ple_rows=SimpleNamespace(
            active=True, source_contexts=(None,) * lanes, release=lambda: freed.append("ple_view")
        ),
        embedding_rows=_Tensor("emb_view", freed),
        qsa_chunk_constants=parent.qsa_chunk_constants,
        qsa_lane_constants=_constants(lanes, owns_page_table=False, page_table=parent.qsa_lane_constants.page_table),
        _owner=model._state_owner,
        storage_lanes=PARENT_LANES,
    )
    values.update(overrides)
    return model_module.Qwen38TTNNTextModelLaneState(**values)


def test_lane_state_view_properties(freed):
    model = _model()
    parent = _parent(model, freed)
    view = _view(model, parent, freed)
    assert (parent.stored_lanes, parent.is_view) == (PARENT_LANES, False)
    assert (view.lanes, view.stored_lanes, view.is_view) == (VIEW_LANES, PARENT_LANES, True)
    assert view.page_table is parent.page_table


def test_validate_accepts_the_parent_and_a_view(freed):
    model = _model()
    parent = _parent(model, freed)
    assert model._validate_lane_state(parent) == PARENT_LANES
    assert model._validate_lane_state(_view(model, parent, freed)) == VIEW_LANES


def test_validate_rejects_a_wrong_view(freed, expect_error):
    model = _model()
    parent = _parent(model, freed)
    # a position row of the view's own width (it must be the parent's 32-wide row)
    with expect_error(TypeError, "32-lane"):
        model._validate_lane_state(_view(model, parent, freed, position=_position(model, VIEW_LANES, freed)))
    # a view over a non-32-lane parent
    narrow = _view(model, parent, freed, storage_lanes=16, position=_position(model, 16, freed))
    with expect_error(ValueError, "32-lane parent"):
        model._validate_lane_state(narrow)
    # a view over an unpaged parent
    unpaged = _view(
        model,
        parent,
        freed,
        qsa_lane_constants=_constants(VIEW_LANES, owns_page_table=False, page_table=None, paging=False),
    )
    with expect_error(ValueError, "paged parent"):
        model._validate_lane_state(unpaged)
    # a view whose lane constants own the page table
    owning = _view(
        model,
        parent,
        freed,
        qsa_lane_constants=_constants(VIEW_LANES, owns_page_table=True, page_table=parent.page_table),
    )
    with expect_error(ValueError, "share the parent's page table"):
        model._validate_lane_state(owning)
    # a non-view 8-lane state must still hold an 8-lane position row
    plain = _view(model, parent, freed, storage_lanes=0)
    with expect_error(TypeError, "8-lane"):
        model._validate_lane_state(plain)


@pytest.mark.parametrize(
    "call",
    [
        lambda m, s: m.release_lane_state(s),
        lambda m, s: m.reset_lane_state_inplace(s, [0] * VIEW_LANES),
        lambda m, s: m.admit_lane(s, 0, 0),
        lambda m, s: m.admit_lane_position(s, 0, 0),
        lambda m, s: m.write_lane_page_table(s, [[1]]),
        lambda m, s: m.admit_lane_kv_paged(s, None, 0, 1),
        lambda m, s: m.lane_layout(s),
    ],
    ids=[
        "release_lane_state",
        "reset_lane_state_inplace",
        "admit_lane",
        "admit_lane_position",
        "write_lane_page_table",
        "admit_lane_kv_paged",
        "lane_layout",
    ],
)
def test_parent_only_operations_refuse_a_view(call, freed, expect_error):
    model = _model()
    view = _view(model, _parent(model, freed), freed)
    with expect_error(ValueError, "bucket view; use the parent lane state"):
        call(model, view)
    assert freed == []


def test_model_bucket_allocation_builds_a_view_over_the_parent(monkeypatch, fake_uploads, freed):
    model = _model()
    parent = _parent(model, freed)
    built = SimpleNamespace(constants=None, layers=[])

    def build_view(mesh_device, mesh_contract, parent_constants, *, lanes):
        assert parent_constants is parent.qsa_lane_constants and lanes == VIEW_LANES
        built.constants = _constants(lanes, owns_page_table=False, page_table=parent_constants.page_table)
        return built.constants

    monkeypatch.setattr(qsa_module.Qwen38TTNNQSALaneConstants, "build_view", staticmethod(build_view))
    for index, layer in enumerate(model.layers):

        def allocate(parent_layer_state, lanes, index=index):
            assert parent_layer_state is parent.layers[index] and lanes == VIEW_LANES
            state = _layer_states(lanes, view=True, freed=freed)[index]
            built.layers.append(state)
            return state

        layer.allocate_bucket_lane_state = allocate
    prepared = SimpleNamespace(
        active=True, source_contexts=(None,) * VIEW_LANES, release=lambda: freed.append("ple_view")
    )
    model.layers[model_module.PLE_CHECKPOINT_LAYER].ple.prepare_lanes_input = lambda tokens, ple_state: prepared

    view = model.allocate_bucket_lane_state(parent, VIEW_LANES)
    assert view.is_view and view.lanes == VIEW_LANES and view.storage_lanes == PARENT_LANES
    assert view.position is parent.position and view.token_row is parent.token_row
    assert view.qsa_chunk_constants is parent.qsa_chunk_constants and view.page_table is parent.page_table
    assert view.qsa_lane_constants is built.constants and view.ple_rows is prepared
    assert tuple(view.layers) == tuple(built.layers)
    assert view.embedding_rows is fake_uploads[0] and "(1, 1, 8, 2560)" in view.embedding_rows.name
    assert view.layers[model_module.PLE_CHECKPOINT_LAYER].ple.token_contexts == (None,) * VIEW_LANES
    assert freed == []


@pytest.mark.parametrize("lanes", [0, 32, True])
def test_model_bucket_allocation_arguments(lanes, freed, expect_error):
    model = _model()
    parent = _parent(model, freed)
    with expect_error(ValueError, "bucket lanes"):
        model.allocate_bucket_lane_state(parent, lanes)
    with expect_error(ValueError, "not over a view"):
        model.allocate_bucket_lane_state(_view(model, parent, freed), 4)
    unpaged = dataclasses.replace(
        parent, qsa_lane_constants=_constants(PARENT_LANES, owns_page_table=True, page_table=None, paging=False)
    )
    with expect_error(ValueError, "paged parent"):
        model.allocate_bucket_lane_state(unpaged, VIEW_LANES)


def test_model_bucket_allocation_failure_frees_only_the_views_objects(monkeypatch, fake_uploads, freed):
    model = _model()
    parent = _parent(model, freed)
    constants = _constants(VIEW_LANES, owns_page_table=False, page_table=parent.page_table)
    constants.deallocate = lambda: freed.append("constants_view")
    monkeypatch.setattr(
        qsa_module.Qwen38TTNNQSALaneConstants, "build_view", staticmethod(lambda *args, **kwargs: constants)
    )
    for index, layer in enumerate(model.layers):

        def allocate(parent_layer_state, lanes, index=index):
            if index == 2:
                raise RuntimeError("layer 2 failed")
            return _layer_states(lanes, view=True, freed=freed)[index]

        layer.allocate_bucket_lane_state = allocate
        layer.release_bucket_lane_state = lambda state, freed=freed: freed.append(f"layer{state.layer_index}")
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        RuntimeError, match="layer 2 failed"
    ):
        model.allocate_bucket_lane_state(parent, VIEW_LANES)
    assert freed == ["layer1", "layer0", "constants_view"]  # reverse order; nothing of the parent


def test_model_bucket_release_frees_only_the_views_objects(freed, expect_error):
    model = _model()
    parent = _parent(model, freed)
    view = _view(model, parent, freed)
    view.qsa_lane_constants.deallocate = lambda: freed.append("constants_view")
    for layer in model.layers:
        layer.release_bucket_lane_state = lambda state: freed.append(f"layer{state.layer_index}")
        layer.release_lane_state = lambda state: freed.append("PARENT-LAYER")
    model.release_bucket_lane_state(view)
    expected = [f"layer{index}" for index in reversed(range(model_module.BACKBONE_LAYERS))]
    assert freed == expected + ["ple_view", "emb_view", "constants_view"]
    for forbidden in ("token_row", "position", "chunk", "ple32", "PARENT-LAYER"):
        assert forbidden not in freed
    with expect_error(ValueError, "release_lane_state"):
        model.release_bucket_lane_state(parent)


# --------------------------------------------------------------------------- the lane constants and the sources


def test_view_constants_leave_the_shared_page_table():
    assert "owns_page_table" in inspect.getsource(qsa_module.Qwen38TTNNQSALaneConstants.deallocate)


def test_view_capable_lane_operations_size_from_the_state_lanes():
    """write_lane_tokens, write_lane_ple_rows, the embedding, forward, resolve and capture take the lane count from
    ``state.lanes`` (the validate's return), never from the parent."""

    for name in (
        "write_lane_tokens",
        "write_lane_ple_rows",
        "_embed_residual_lanes_from_device_token",
        "forward_decode_lanes",
        "resolve_lane_tokens",
        "capture_decode_lanes",
    ):
        source = inspect.getsource(getattr(model_module.Qwen38TTNNTextModel, name))
        assert "stored_lanes" not in source and "storage_lanes" not in source, name
