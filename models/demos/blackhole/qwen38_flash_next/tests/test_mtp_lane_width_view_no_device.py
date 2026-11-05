# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP lane width views: the lane-prefix aliases of the 8-lane MTP state (no device).

What is pinned, on the torch-backed ttnn of the MTP lanes tests (its ``reshape`` of a ``(logical, padded)`` shape
returns the prefix slice, which is what the device alias holds; the device identity of the buffers is the device probe
``test_lane_prefix_view_device.py``):

* the alias shape of every tensor family (TILE replicated, head-sharded fp32, ``[1,B,1,1]`` tiles, row-major, the
  lane-per-row tile) and the refusals of the helper;
* ``scratch_row = storage_lanes * C`` of the QSA lane verify constants, the rule that every width's accept constants are
  the top-left slices of the parent's;
* the view of every family (GDN state / lane rows state / lane buffers, QSA lane state / lane verify state, PLE lane
  rows state and prepared rows) keeps the parent's buffers and refuses every lifecycle op, and the builder of a whole
  view owns exactly the per-width objects: releasing the view never deallocates an alias, the parent refuses its own
  release while a view is live;
* the refusals (width not in {2, 4}, not narrower than the parent, parent a view, the GDN fold off) and the host writers
  that take the parent only;
* the k5 list (lane-per-row tile tensors) and the code facts that make each entry safe.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_v2_step6_draft_no_device as step6
from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_lanes_no_device import (  # noqa: F401 (fixtures)
    LANE_FORMS,
    _FakeLaneMoE,
    _lane_column,
    _lane_qsa_constants,
    _u32_row,
    fake,
    gdn_weights,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step4_rows_no_device import (
    BF16,
    FP32,
    ROW_MAJOR,
    TILE,
    TP,
    FakeContract,
    FakeTensor,
    _gdn_module,
    _ple_module,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step5_verify_no_device import (
    CONTEXT,
    _qsa_module,
    _replicated,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_scan
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNLayerNamespace

# The lane-per-row tile tensors of the MTP lane state and what the lane code does with each: "unused" = no MTP lane path
# reads or writes it; "scratch" = written by every pass before it is read, and kept as one own tensor per width.  There
# is no per-lane persistent tensor of this kind, so no admission write has to be checked for one.
K5_LIST = (
    ("GDN state conv slots [1,1,B,2560] (4)", "unused"),
    ("GDN state zero_conv [1,1,B,2560]", "unused"),
    ("alignment residual [1,4,B,640]", "scratch"),
    ("MoE input [1,1,moe_rows,640] (per layer and draft)", "scratch"),
)

_TILE, _ROW_MAJOR = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT  # the real constants: the helper tests patch ttnn.reshape

PARENT_LANES, DRAFTS, WIDTH = 4, 3, 2  # the fake world: a 4-lane parent, a 2-lane view (the device parent has 8 lanes)
ROWS = DRAFTS + 1


# --------------------------------------------------------------------------- the helper


class _Buf:
    """A device tensor of the helper tests: the shape the ops see, the layout, the buffer it points at."""

    def __init__(self, shape, layout, address=1000, parent=None):
        self.shape, self.layout, self.address, self.parent = tuple(shape), layout, address, parent
        self.topology = ("topology", address)

    def tensor_topology(self):
        return self.topology

    def update_tensor_topology(self, topology):
        self.topology = topology


@pytest.fixture
def reshape_log(monkeypatch):
    calls = []

    def reshape(tensor, shape, padded_shape=None):
        calls.append((tuple(shape), tuple(padded_shape)))
        return _Buf(shape, tensor.layout, tensor.address, parent=tensor)

    monkeypatch.setattr(mtp_lanes.ttnn, "reshape", reshape)
    return calls


@pytest.mark.parametrize(
    "shape,layout,lane_dim,per_lane,parent_lanes,lanes,logical,padded",
    (
        # TILE replicated [1,B,32,128]: the lane is a whole-tile axis
        ((1, 8, 32, 128), _TILE, 1, 1, 8, 2, (1, 2, 32, 128), (1, 2, 32, 128)),
        ((1, 8, 32, 128), _TILE, 1, 1, 8, 4, (1, 4, 32, 128), (1, 4, 32, 128)),
        # head-sharded fp32 recurrent state [B,12,128,128]: the lane is the outermost axis
        ((8, 12, 128, 128), _TILE, 0, 1, 8, 4, (4, 12, 128, 128), (4, 12, 128, 128)),
        # the [1,B,1,1] tiles: logical != padded (one tile per lane)
        ((1, 8, 1, 1), _TILE, 1, 1, 8, 2, (1, 2, 1, 1), (1, 2, 32, 32)),
        # row-major [1,1,B*R,640] (R = 4 rows per lane)
        ((1, 1, 32, 640), _ROW_MAJOR, 2, 4, 8, 4, (1, 1, 16, 640), (1, 1, 16, 640)),
        # the lane-per-row tile [1,4,B,640] and [1,1,B,2560]: padded to the tile
        ((1, 4, 8, 640), _TILE, 2, 1, 8, 2, (1, 4, 2, 640), (1, 4, 32, 640)),
        ((1, 1, 8, 2560), _TILE, 2, 1, 8, 4, (1, 1, 4, 2560), (1, 1, 32, 2560)),
        # a lane tile of 32 rows per lane: [1,1,B*32,2560]
        ((1, 1, 256, 2560), _TILE, 2, 32, 8, 2, (1, 1, 64, 2560), (1, 1, 64, 2560)),
    ),
)
def test_lane_prefix_view_shapes_per_family(
    reshape_log, shape, layout, lane_dim, per_lane, parent_lanes, lanes, logical, padded
) -> None:
    parent = _Buf(shape, layout)
    alias = mtp_lanes.lane_prefix_view(
        parent, lanes=lanes, parent_lanes=parent_lanes, lane_dim=lane_dim, per_lane=per_lane
    )
    assert reshape_log == [(logical, padded)]
    assert alias.shape == logical and alias.address == parent.address and alias.parent is parent
    assert alias.topology == parent.topology  # the placement of the parent: nothing moved between devices


@pytest.mark.parametrize(
    "lanes,parent_lanes,lane_dim,per_lane,match",
    (
        (8, 8, 1, 1, "takes 1 <= lanes < 8"),
        (9, 8, 1, 1, "takes 1 <= lanes < 8"),
        (0, 8, 1, 1, "takes 1 <= lanes < 8"),
        (2, 8, 0, 1, "not 8 lanes x 1"),  # dim 0 holds 1, not the 8 lanes
        (2, 8, 1, 4, "not 8 lanes x 4"),
        (2, 8, 7, 1, "do not fit a rank 4 tensor"),
    ),
)
def test_lane_prefix_view_refuses_a_bad_geometry(
    expect_error, reshape_log, lanes, parent_lanes, lane_dim, per_lane, match
) -> None:
    with expect_error(ValueError, match=match):
        mtp_lanes.lane_prefix_view(
            _Buf((1, 8, 32, 128), _TILE), lanes=lanes, parent_lanes=parent_lanes, lane_dim=lane_dim, per_lane=per_lane
        )
    assert reshape_log == []


def test_lane_prefix_view_refuses_non_int_lane_counts(expect_error, reshape_log) -> None:
    with expect_error(TypeError, match="must be ints"):
        mtp_lanes.lane_prefix_view(_Buf((1, 8, 32, 128), _TILE), lanes=True, parent_lanes=8, lane_dim=1)
    with expect_error(TypeError, match="must be ints"):
        mtp_lanes.lane_prefix_view(_Buf((1, 8, 32, 128), _TILE), lanes=2.0, parent_lanes=8, lane_dim=1)


def test_the_ledger_never_owns_an_alias(expect_error, reshape_log) -> None:
    ledger = mtp_lanes._ViewLedger(2, 8)
    alias = ledger.alias(_Buf((1, 8, 32, 128), _TILE), lane_dim=1)
    own = _Buf((1, 2, 32, 128), _TILE, address=2000)
    assert ledger.is_alias(alias) and not ledger.is_alias(own)
    assert ledger.own_tensor("scratch", own) is own
    with expect_error(ValueError, match="alias cannot be owned"):
        ledger.own_tensor("alias", alias)
    assert [label for label, _ in ledger.owned] == ["scratch"]


# --------------------------------------------------------------------------- host constants of a width


@pytest.mark.parametrize("lanes,storage", ((2, 8), (4, 8), (2, 4), (4, 4)))
def test_scratch_row_is_storage_lanes_times_capacity(expect_error, lanes: int, storage: int) -> None:
    capacity = qsa_module.validate_qsa_cache_capacity(CONTEXT)
    plain = qsa_module.qsa_lane_verify_constant_rows(lanes, ROWS, CONTEXT)
    wide = qsa_module.qsa_lane_verify_constant_rows(lanes, ROWS, CONTEXT, storage_lanes=storage)
    assert plain["scratch_row"].eq(lanes * capacity).all()
    assert wide["scratch_row"].eq(storage * capacity).all()
    for name in plain:  # every other constant depends on the width alone
        if name != "scratch_row":
            assert torch.equal(plain[name], wide[name]), name
    with expect_error(ValueError, match="fewer than its"):
        qsa_module.qsa_lane_verify_constant_rows(lanes, ROWS, CONTEXT, storage_lanes=lanes - 1)


@pytest.mark.parametrize("storage", (4, 8))
def test_inactive_lane_kv_rows_of_a_view_land_after_the_parents_last_lane(storage: int) -> None:
    blocks = CONTEXT // qsa_module.COMPRESS_RATIO
    positions, active = (5, 66), (1, 0)
    wide = qsa_module.emulate_qsa_lane_verify_inputs(
        positions, active, lanes=2, rows=ROWS, allocated_compressed_blocks=blocks, storage_lanes=storage
    )
    default = qsa_module.emulate_qsa_lane_verify_inputs(
        positions, active, lanes=2, rows=ROWS, allocated_compressed_blocks=blocks
    )
    for name, step in (("kv_row_start_lanes", 0), ("kv_row_start_next_lanes", qsa_module.CACHE_WRITE_ROWS)):
        assert wide[name].reshape(-1).tolist()[1] == storage * CONTEXT + step, name  # the scratch rows
        assert default[name].reshape(-1).tolist()[1] == 2 * CONTEXT + step, name
        assert wide[name].reshape(-1).tolist()[0] == default[name].reshape(-1).tolist()[0], name  # the active lane
    for name in ("row_positions", "stage_keep", "pool_select"):
        assert torch.equal(wide[name], default[name]), name


@pytest.mark.parametrize(
    "parent_lanes,lanes,drafts", ((8, 4, 3), (8, 2, 3), (6, 2, 4), (6, 4, 4), (4, 2, 3), (5, 4, 5), (3, 2, 5))
)
def test_every_widths_accept_constants_are_the_top_left_slices_of_the_parents(
    parent_lanes: int, lanes: int, drafts: int
) -> None:
    if parent_lanes * (drafts + 1) > 32:
        pytest.skip("the parent does not fit the tile")
    rows = drafts + 1
    parent = mtp_lanes.mtp_lane_constant_rows(parent_lanes, drafts)
    child = mtp_lanes.mtp_lane_constant_rows(lanes, drafts)
    real = lanes * rows
    assert torch.equal(child["prefix_upper_block"][:real, :real], parent["prefix_upper_block"][:real, :real])
    assert torch.count_nonzero(child["prefix_upper_block"][real:]) == 0
    assert torch.count_nonzero(child["prefix_upper_block"][:, real:]) == 0
    assert torch.equal(child["draft_plus_one_row"][:, :real], parent["draft_plus_one_row"][:, :real])
    assert torch.equal(child["draft_match_row"][:, :real], parent["draft_match_row"][:, :real])
    assert torch.equal(child["block_sum"][:real, :], parent["block_sum"][:real, :lanes])
    assert torch.equal(child["block_sum_t"][:, :real], parent["block_sum_t"][:lanes, :real])
    assert torch.equal(child["lane_base_row"], parent["lane_base_row"][:, :lanes])  # lane_base_row depends on R, not B


# --------------------------------------------------------------------------- the world: real states on the fake


class _Recorder:
    """The fake's deallocate with a log: which tensors were freed."""

    def __init__(self, fake_ttnn):
        self.freed: list = []
        self._base = fake_ttnn.deallocate
        fake_ttnn.deallocate = self

    def __call__(self, tensor, *args, **kwargs):
        self.freed.append(tensor)
        return self._base(tensor, *args, **kwargs)


def _zeros(shape, dtype=BF16, layout=TILE, shard_dim=3) -> FakeTensor:
    return FakeTensor([torch.zeros(*shape, dtype=dtype.torch) for _ in range(TP)], dtype, layout, shard_dim)


def _view_world(fake, monkeypatch, gdn_weights, *, fold: bool = True, draft: bool = True):
    """A 4-lane parent (k = 3: R = 4, B*R = 16 rows) of one QSA layer, one GDN layer with the PLE (checkpoint layer 1)
    and the alignment layer, as real states over the real modules on the fake; the draft state beside it."""

    lanes, drafts, rows = PARENT_LANES, DRAFTS, ROWS
    contract = FakeContract()
    owner = object()
    qsa, gdn, ple = _qsa_module(fake.ttnn), _gdn_module(gdn_weights[1]), _ple_module()
    scratch = mtp_lanes.kv_scratch_rows(CONTEXT)
    mlp = SimpleNamespace(
        weights=object(),
        rows=1,
        mesh_device="mesh",
        mesh_contract=contract,
        tt_ccl=None,
        collective_topology=None,
        synchronization_policy=None,
    )
    qsa_layer = SimpleNamespace(
        namespace=Qwen38TTNNLayerNamespace.BACKBONE, layer_index=0, attention=qsa, ple=None, mlp=mlp
    )
    gdn_layer = SimpleNamespace(
        namespace=Qwen38TTNNLayerNamespace.BACKBONE, layer_index=1, attention=gdn, ple=ple, mlp=mlp
    )
    mtp_layer = SimpleNamespace(namespace=Qwen38TTNNLayerNamespace.MTP, layer_index=0, attention=qsa, ple=None, mlp=mlp)
    moe = SimpleNamespace(rows=lanes * rows, release_owned_buffers=lambda: None)
    gdn_rows_constants = gdn.allocate_rows_constants(rows)
    gdn_lane_constants = gdn.allocate_lane_rows_constants(lanes, rows)
    gdn_rows = gdn.allocate_lane_rows_state(gdn_rows_constants, gdn_lane_constants)
    if fold:
        prefix = _zeros(gdn_rows_scan.lanes_prefix_shape(lanes, rows), FP32, TILE, 1)
        gdn_rows.scan_buffers = gdn_rows_scan.LaneBuffers(
            prefix=prefix, constants="constants", gated_memory_config="gated", lanes=lanes, rows=rows
        )
    qsa_state = qsa.allocate_lane_state(lanes, kv_scratch_rows=scratch)
    layers = (
        mtp_lanes.Qwen38TTNNMTPLaneLayerState(
            qsa_layer.namespace, 0, qsa_state, qsa.allocate_lane_verify_state(lanes), None, moe, None
        ),
        mtp_lanes.Qwen38TTNNMTPLaneLayerState(
            gdn_layer.namespace,
            1,
            gdn.allocate_lane_state(lanes),
            gdn_rows,
            ple.allocate_lane_rows_state(lanes, rows),
            moe,
            _zeros((1, 1, lanes * rows, step6.WIDTH)),
        ),
    )
    alignment_state = mtp_lanes.Qwen38TTNNMTPLaneLayerState(
        mtp_layer.namespace,
        0,
        qsa.allocate_lane_state(lanes, kv_scratch_rows=scratch),
        qsa.allocate_lane_verify_state(lanes),
        None,
        moe,
        _zeros((1, 1, lanes * rows, step6.WIDTH)),
    )
    alignment = mtp_lanes.Qwen38TTNNMTPLaneAlignment(
        mtp_layer,
        SimpleNamespace(name="input mixer"),
        SimpleNamespace(name="final mixer"),
        alignment_state,
        _zeros((1, 4, lanes, step6.WIDTH)),
    )
    position_constants, chunk_constants, verify_constants = _lane_qsa_constants(rows, lanes)[0]
    lane_constants = qsa_module.Qwen38TTNNQSALaneConstants.build(
        "mesh", contract, lanes=lanes, allocated_context=CONTEXT
    )
    lane_verify_constants = qsa_module.Qwen38TTNNQSALaneVerifyConstants.build(
        "mesh", contract, lanes=lanes, rows=rows, allocated_context=CONTEXT
    )
    verify = mtp_lanes.Qwen38TTNNMTPLaneVerifyState(
        lanes=lanes,
        drafts=drafts,
        rows=rows,
        moe_rows=lanes * rows,
        constants=mtp_lanes.Qwen38TTNNMTPLaneConstants.build("mesh", contract, lanes=lanes, drafts=drafts),
        gdn_rows_constants=gdn_rows_constants,
        gdn_lane_constants=gdn_lane_constants,
        qsa_chunk_constants=chunk_constants,
        qsa_verify_constants=verify_constants,
        qsa_lane_constants=lane_constants,
        qsa_lane_verify_constants=lane_verify_constants,
        layers=layers,
        alignment=alignment,
        positions=mtp_lanes.Qwen38TTNNMTPLanePositions.allocate("mesh", contract, [10, 20, 30, 40], lanes=lanes),
        token_row=_replicated(torch.zeros(1, 1, 1, 32), FP32),
        draft_lanes=_u32_row([0] * 32),
        ple_rows=ple_module.Qwen38TTNNPLERowsPreparedInput(
            _zeros((1, 1, lanes * rows, step6.WIDTH), layout=ROW_MAJOR), (0,) * (lanes * rows), ()
        ),
        accepted_lanes=_lane_column([rows - 1] * lanes),
        accepted_row=_replicated(torch.zeros(1, 1, 1, 32), FP32, ROW_MAJOR),
        active_lanes=_lane_column([1] * lanes),
        active_row=_u32_row([1] * 32),
        inactive_row=_u32_row([0] * 32),
        commit_lanes=_lane_column([1] * lanes),
        k_eff_mask_row=_replicated(torch.ones(1, 1, 1, 32), FP32, ROW_MAJOR),
        pass_contexts=[(None,) * (rows + 1) for _ in range(lanes)],
        _owner=owner,
    )
    model = SimpleNamespace(
        _state_owner=owner,
        mesh_device="mesh",
        mesh_contract=contract,
        allocated_context=CONTEXT,
        layers=(qsa_layer, gdn_layer),
        poisoned=False,
    )
    monkeypatch.setattr(mtp_lanes, "BACKBONE_LAYERS", 2)
    monkeypatch.setattr(mtp_lanes, "_allocate_lane_moe", lambda layer, moe_rows: _FakeLaneMoE(rows=moe_rows))
    monkeypatch.setattr(
        mtp_lanes,
        "_allocate_hidden_sharded_zeros",
        lambda mesh, contract_, shape, *, label: _zeros(shape),
    )
    parent_draft = None
    if draft:
        draft_qsa = qsa.allocate_lane_verify_state(lanes)
        draft_verify_constants = qsa_module.Qwen38TTNNQSAVerifyConstants.build(
            "mesh", contract, chunk_constants, rows=1
        )
        draft_lane_verify = qsa_module.Qwen38TTNNQSALaneVerifyConstants.build(
            "mesh", contract, lanes=lanes, rows=1, allocated_context=CONTEXT
        )
        zero_accepted = _lane_column([0] * lanes)
        ones = _lane_column([1] * lanes)
        parent_draft = mtp_lanes.Qwen38TTNNMTPLaneDraftState(
            lanes=lanes,
            drafts=drafts,
            rows=1,
            qsa_chunk_constants=chunk_constants,
            qsa_verify_constants=draft_verify_constants,
            qsa_lane_constants=lane_constants,
            qsa_lane_verify_constants=draft_lane_verify,
            qsa_state=draft_qsa,
            layer_state=mtp_lanes.Qwen38TTNNMTPLaneLayerState(
                mtp_layer.namespace, 0, alignment_state.attention_state, draft_qsa, None, moe, None
            ),
            zero_accepted_lanes=zero_accepted,
            ones_lanes=ones,
            advance_selectors=SimpleNamespace(deallocate=lambda: None),
            step_offset_rows=tuple(_u32_row([i] * 32) for i in range(drafts - 1)),
            sentinel_tail=_replicated(torch.full((1, 1, 1, 32 - lanes), -1.0), FP32, ROW_MAJOR),
            sentinel_lane=_replicated(torch.full((1, 1, 1, 1), -1.0), FP32, ROW_MAJOR),
            pass_row=_replicated(torch.full((1, 1, 1, mtp_lanes.lane_pass_row_width(lanes)), -1.0), FP32, ROW_MAJOR),
            _owner=owner,
        )
        monkeypatch.setattr(
            gdn_module,
            "build_rows_selectors_lanes",
            lambda accepted, active, constants: SimpleNamespace(lanes=constants.lanes, deallocate=lambda: None),
        )
        monkeypatch.setattr(mtp_lanes.gdn_module, "build_rows_selectors_lanes", gdn_module.build_rows_selectors_lanes)
    return SimpleNamespace(model=model, parent=verify, parent_draft=parent_draft, gdn=gdn, qsa=qsa, ple=ple)


@pytest.fixture
def world(fake, monkeypatch, gdn_weights):
    return _view_world(fake, monkeypatch, gdn_weights)


@pytest.fixture
def view(world):
    return mtp_lanes.allocate_lane_width_view(world.model, world.parent, WIDTH, parent_draft=world.parent_draft)


def _all_tensors(value, seen=None, visited=None):
    """Every FakeTensor reachable from a dataclass / tuple / namespace (by identity; the parent link is not followed)."""

    seen = {} if seen is None else seen
    visited = set() if visited is None else visited
    if id(value) in visited:
        return seen
    visited.add(id(value))
    if isinstance(value, FakeTensor):
        seen[id(value)] = value
    elif isinstance(value, (tuple, list)):
        for item in value:
            _all_tensors(item, seen, visited)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for name, item in vars(value).items():
            if name not in ("parent", "views"):
                _all_tensors(item, seen, visited)
    elif hasattr(value, "__slots__"):
        for name in value.__slots__:
            _all_tensors(getattr(value, name), seen, visited)
    return seen


# --------------------------------------------------------------------------- per-family views


def test_gdn_state_view_aliases_the_parents_lanes_and_refuses_every_lifecycle_op(world, expect_error) -> None:
    parent_state = world.parent.layers[1].attention_state
    alias = lambda tensor, *, lane_dim, per_lane=1: mtp_lanes.lane_prefix_view(  # noqa: E731
        tensor, lanes=WIDTH, parent_lanes=PARENT_LANES, lane_dim=lane_dim, per_lane=per_lane
    )
    state = world.gdn.lane_state_view(parent_state, WIDTH, alias)
    local = gdn_module.VALUE_HEADS_PER_DEVICE
    assert state.batch_size == WIDTH and state.lane_alias and not state.is_view
    assert state.recurrent.shape == state.zero_recurrent.shape == (WIDTH, local, 128, 128)
    assert all(slot.shape == (1, 1, WIDTH, gdn_module.QKV_WIDTH_PER_DEVICE) for slot in (*state.conv, state.zero_conv))
    for name, call in (
        ("deallocate", state.deallocate),
        ("reset_inplace", state.reset_inplace),
        ("allocate_snapshot", state.allocate_snapshot),
        ("reset_lane_inplace", lambda: state.reset_lane_inplace(0)),
    ):
        with expect_error(RuntimeError, match="MTP lane width view"):
            call()
    assert parent_state.batch_size == PARENT_LANES and not parent_state.lane_alias
    with expect_error(ValueError, match="takes 1 <= lanes < 4"):
        world.gdn.lane_state_view(parent_state, PARENT_LANES, alias)
    with expect_error(ValueError, match="not a view"):
        world.gdn.lane_state_view(state, 1, alias)


def test_gdn_lane_rows_state_view_shares_the_buffers_and_the_prefix_states(world, expect_error) -> None:
    rows_state = world.parent.layers[1].attention_rows
    lane_constants = world.gdn.allocate_lane_rows_constants(WIDTH, ROWS)
    alias = lambda tensor, *, lane_dim, per_lane=1: mtp_lanes.lane_prefix_view(  # noqa: E731
        tensor, lanes=WIDTH, parent_lanes=PARENT_LANES, lane_dim=lane_dim, per_lane=per_lane
    )
    state = rows_state.width_view(WIDTH, lane_constants, alias)
    heads, qkv, value = (
        gdn_module.VALUE_HEADS_PER_DEVICE,
        gdn_module.QKV_WIDTH_PER_DEVICE,
        gdn_module.VALUE_WIDTH_PER_DEVICE,
    )
    assert state.lanes == WIDTH and state.lane_alias and state.constants is rows_state.constants
    assert state.qkv is rows_state.qkv  # the projected tile is the parent's
    assert state.history.shape == (1, WIDTH, 32, qkv) and state.qkv_lanes.shape == (1, 1, WIDTH * 32, qkv)
    assert state.q.shape == state.k.shape == (WIDTH, 32, heads, 128)
    assert state.v.shape == (1, WIDTH, 32, value) and state.beta.shape == state.g.shape == (1, WIDTH, 32, heads)
    buffers = state.scan_buffers
    assert buffers.is_view and (buffers.lanes, buffers.rows) == (WIDTH, ROWS)
    assert buffers.prefix.shape == gdn_rows_scan.lanes_prefix_shape(WIDTH, ROWS)
    assert buffers.constants == rows_state.scan_buffers.constants
    with expect_error(RuntimeError, match="width view aliases the parent"):
        state.deallocate()
    assert state.scan_buffers is buffers and rows_state.scan_buffers.prefix.alive
    with expect_error(ValueError, match="takes 1 <= lanes < 4"):
        rows_state.width_view(PARENT_LANES, lane_constants, alias)
    with expect_error(ValueError, match="constants are 2 x 4"):
        rows_state.width_view(PARENT_LANES - 1, lane_constants, alias)
    with expect_error(ValueError, match="cannot be viewed again"):
        state.width_view(1, world.gdn.allocate_lane_rows_constants(1, ROWS), alias)


def test_lane_buffers_view_never_frees_its_prefix(fake, expect_error) -> None:
    freed = _Recorder(fake.ttnn)
    parent = gdn_rows_scan.LaneBuffers(
        prefix=_zeros(gdn_rows_scan.lanes_prefix_shape(8, 4), FP32, TILE, 1),
        constants="c",
        gated_memory_config="g",
        lanes=8,
        rows=4,
    )
    alias = _zeros(gdn_rows_scan.lanes_prefix_shape(2, 4), FP32, TILE, 1)
    child = parent.view(2, alias)
    assert child.is_view and not parent.is_view and child.prefix is alias and child.constants == "c"
    child.deallocate()
    child.deallocate()
    assert freed.freed == [] and parent.prefix.alive
    for lanes, prefix, match in (
        (8, alias, "takes 1 <= lanes < 8"),
        (2, _zeros(gdn_rows_scan.lanes_prefix_shape(3, 4), FP32, TILE, 1), "prefix must be"),
    ):
        with expect_error(ValueError, match=match):
            parent.view(lanes, prefix)
    with expect_error(ValueError, match="cannot be viewed again"):
        child.view(1, alias)


def test_a_lane_rows_state_without_the_gdn_fold_is_refused(fake, monkeypatch, gdn_weights, expect_error) -> None:
    world = _view_world(fake, monkeypatch, gdn_weights, fold=False, draft=False)
    rows_state = world.parent.layers[1].attention_rows
    assert rows_state.scan_buffers is None
    alias = lambda tensor, *, lane_dim, per_lane=1: tensor  # noqa: E731
    with expect_error(RuntimeError, match="QWEN38_FUSED_OFF=gdn_rows_scan"):
        rows_state.width_view(WIDTH, world.gdn.allocate_lane_rows_constants(WIDTH, ROWS), alias)
    with expect_error(RuntimeError, match="QWEN38_FUSED_OFF=gdn_rows_scan"):
        mtp_lanes.allocate_lane_width_view(world.model, world.parent, WIDTH)
    assert world.parent.views == []


def test_ple_lane_rows_state_and_prepared_rows_views(world, expect_error) -> None:
    ple_state = world.parent.layers[1].ple
    alias = lambda tensor, *, lane_dim, per_lane=1: mtp_lanes.lane_prefix_view(  # noqa: E731
        tensor, lanes=WIDTH, parent_lanes=PARENT_LANES, lane_dim=lane_dim, per_lane=per_lane
    )
    state = world.ple.lane_rows_state_view(ple_state, WIDTH, alias)
    assert state.lane_alias and state.lanes == WIDTH and state.rows == ROWS
    assert state.history.shape == (WIDTH, ple_module.CONV_STATE_LENGTH, 4, step6.WIDTH)
    assert state.normalized.shape == (WIDTH, ROWS, 4, step6.WIDTH)
    assert state.token_contexts == (None,) * WIDTH  # host contexts stay on the parent
    with expect_error(RuntimeError, match="width view aliases the parent"):
        state.deallocate()
    with expect_error(ValueError, match="takes 1 <= lanes < 4"):
        ple_state.width_view(PARENT_LANES, alias)
    with expect_error(ValueError, match="cannot be viewed again"):
        state.width_view(1, alias)
    prepared = ple_module.Qwen38TTNNPLERowsPreparedInput(
        world.parent.ple_rows.embedding_rows, (0,) * 16, (), True, True
    )
    with expect_error(RuntimeError, match="aliases the parent's rows"):
        prepared.release()
    assert prepared.active and world.parent.ple_rows.embedding_rows.alive


def test_qsa_lane_state_view_keeps_the_flat_kv_and_aliases_the_per_lane_caches(world, expect_error) -> None:
    parent_state = world.parent.layers[0].attention_state
    alias = lambda tensor, *, lane_dim, per_lane=1: mtp_lanes.lane_prefix_view(  # noqa: E731
        tensor, lanes=WIDTH, parent_lanes=PARENT_LANES, lane_dim=lane_dim, per_lane=per_lane
    )
    state = world.qsa.lane_state_view(parent_state, WIDTH, alias)
    blocks = world.qsa.allocated_compressed_blocks + 32
    assert (state.lanes, state.stored_lanes, state.is_view, state.lane_prefix) == (WIDTH, PARENT_LANES, True, True)
    assert not parent_state.lane_prefix
    assert state.packed_kv_cache is parent_state.packed_kv_cache  # lane u at row u*C, scratch after the last lane
    assert state.packed_kv_cache.shape == (1, 1, PARENT_LANES * CONTEXT + state.kv_scratch_rows, 512)
    assert state.kv_scratch_rows == parent_state.kv_scratch_rows and state.epoch == parent_state.epoch
    assert state.compressed_index_cache.shape == (WIDTH, 1, blocks, 128)
    assert state.compressed_index_cache_flat.shape == (1, 1, WIDTH * blocks, 128)
    assert state.kv_staging.shape[:2] == (1, WIDTH) and state.raw_key_ring.shape == (1, WIDTH, 32, 128)
    world.qsa._validate_lane_state(state)  # the validator accepts the flat non-paged view
    with expect_error(RuntimeError, match="bucket view"):
        world.qsa.release_lane_state(state)
    with expect_error(RuntimeError, match="bucket view"):
        world.qsa.reset_lane_state_inplace(state)
    with expect_error(RuntimeError, match="bucket view"):
        world.qsa.reset_lane_inplace(state, 0)
    with expect_error(ValueError, match="takes 1 <= lanes < 4"):
        world.qsa.lane_state_view(parent_state, PARENT_LANES, alias)
    with expect_error(ValueError, match="flat parent state"):
        world.qsa.lane_state_view(state, 1, alias)


def test_qsa_flat_view_without_scratch_rows_or_with_the_parents_tensors_is_still_refused(world, expect_error) -> None:
    parent_state = world.parent.layers[0].attention_state
    no_scratch = replace(parent_state, lanes=WIDTH, storage_lanes=PARENT_LANES, kv_scratch_rows=0)
    with expect_error(ValueError, match="must be paged"):  # the bucket views are paged; the raise stays for them
        world.qsa._validate_lane_state(no_scratch)
    unaliased = replace(parent_state, lanes=WIDTH, storage_lanes=PARENT_LANES)  # the parent's own 4-lane caches
    with expect_error(RuntimeError, match="compressed lane index cache shape must be"):
        world.qsa._validate_lane_state(unaliased)


def test_qsa_lane_verify_state_view(world, expect_error) -> None:
    parent_state = world.parent.layers[0].attention_rows
    alias = lambda tensor, *, lane_dim, per_lane=1: mtp_lanes.lane_prefix_view(  # noqa: E731
        tensor, lanes=WIDTH, parent_lanes=PARENT_LANES, lane_dim=lane_dim, per_lane=per_lane
    )
    state = world.qsa.lane_verify_state_view(parent_state, WIDTH, alias)
    assert state.lane_alias and state.lanes == WIDTH and state.epoch == parent_state.epoch
    assert state.raw_history.shape == (1, WIDTH, 32, 128) and state.raw_rows.shape == (1, 1, WIDTH * 32, 128)
    with expect_error(RuntimeError, match="width view aliases the parent"):
        world.qsa.release_lane_verify_state(state)
    with expect_error(ValueError, match="takes 1 <= lanes < 4"):
        world.qsa.lane_verify_state_view(parent_state, PARENT_LANES, alias)
    with expect_error(ValueError, match="cannot be viewed again"):
        world.qsa.lane_verify_state_view(state, 1, alias)


# --------------------------------------------------------------------------- the whole view


def test_width_view_shapes_per_family(view, world) -> None:
    verify, parent = view.verify, world.parent
    assert (verify.lanes, verify.stored_lanes, verify.is_view, verify.parent) == (WIDTH, PARENT_LANES, True, parent)
    assert (verify.drafts, verify.rows, verify.moe_rows) == (DRAFTS, ROWS, WIDTH * ROWS)
    mtp_lanes._validate_lane_verify_state(world.model, verify)  # the parent's own validator accepts the view
    assert view.draft is not None and view.draft.is_view and view.draft.parent is world.parent_draft
    # shared with the parent: one object (the host owner), not a copy
    for name in (
        "positions",
        "pass_contexts",
        "token_row",
        "draft_lanes",
        "accepted_row",
        "active_row",
        "inactive_row",
        "k_eff_mask_row",
        "qsa_chunk_constants",
        "qsa_verify_constants",
        "gdn_rows_constants",
    ):
        assert getattr(verify, name) is getattr(parent, name), name
    # per width: the constants, one lane-prefix alias per carried [1,B,1,1] tile
    for name in ("constants", "gdn_lane_constants", "qsa_lane_constants", "qsa_lane_verify_constants"):
        assert getattr(verify, name) is not getattr(parent, name), name
    assert (verify.constants.lanes, verify.gdn_lane_constants.lanes, verify.qsa_lane_constants.lanes) == (WIDTH,) * 3
    assert verify.qsa_lane_verify_constants.lanes == WIDTH
    assert verify.qsa_lane_verify_constants.storage_lanes == PARENT_LANES
    for name in ("accepted_lanes", "active_lanes", "commit_lanes"):
        assert getattr(verify, name).shape == (1, WIDTH, 1, 1), name
    assert verify.ple_rows.embedding_rows.shape == (1, 1, WIDTH * ROWS, step6.WIDTH) and verify.ple_rows.lane_alias
    # per layer
    qsa_layer, gdn_layer = verify.layers
    assert qsa_layer.attention_state.lane_prefix and qsa_layer.attention_state.lanes == WIDTH
    assert gdn_layer.attention_state.lane_alias and gdn_layer.attention_rows.lane_alias and gdn_layer.ple.lane_alias
    assert qsa_layer.moe.rows == gdn_layer.moe.rows == WIDTH * ROWS  # a rows-(W*R) MoE instance per width
    assert qsa_layer.moe is not parent.layers[0].moe and gdn_layer.moe is not parent.layers[1].moe
    for layer_state in (qsa_layer, gdn_layer, verify.alignment.layer_state):  # one own input slice per layer and width
        assert layer_state.moe_input.shape == (1, 1, WIDTH * ROWS, step6.WIDTH)
    # the alignment layer: the lane QSA state viewed, the residual an own per-width scratch tensor
    alignment = verify.alignment
    assert alignment.layer is parent.alignment.layer and alignment.input_mixer is parent.alignment.input_mixer
    assert alignment.layer_state.attention_state.lane_prefix
    assert (
        alignment.residual.shape == (1, 4, WIDTH, step6.WIDTH) and alignment.residual is not parent.alignment.residual
    )
    assert not any(alignment.residual is alias for alias in view.aliases)
    # the draft: pair scratch per width, caches aliased, the shared geometry the draft validator reads
    draft = view.draft
    assert draft.layer_state.attention_state is alignment.layer_state.attention_state
    assert draft.layer_state.attention_rows is draft.qsa_state and draft.qsa_state.lane_alias
    assert (
        draft.qsa_lane_constants is verify.qsa_lane_constants
        and draft.qsa_chunk_constants is verify.qsa_chunk_constants
    )
    assert draft.qsa_lane_verify_constants.rows == 1 and draft.qsa_lane_verify_constants.storage_lanes == PARENT_LANES
    assert draft.qsa_verify_constants is world.parent_draft.qsa_verify_constants
    assert draft.pass_row.shape == (1, 1, 1, mtp_lanes.lane_pass_row_width(WIDTH))
    assert draft.sentinel_tail.shape == (1, 1, 1, 32 - WIDTH) and draft.layer_state.moe.rows == WIDTH
    mtp_lanes._validate_lane_draft_state(world.model, verify, draft)
    assert parent.views == [view]


def test_releasing_a_width_view_frees_only_what_it_owns(fake, world, expect_error) -> None:
    freed = _Recorder(fake.ttnn)
    view = mtp_lanes.allocate_lane_width_view(world.model, world.parent, WIDTH, parent_draft=world.parent_draft)
    parent_tensors = _all_tensors(world.parent)
    parent_tensors.update(_all_tensors(world.parent_draft))
    aliases = {id(alias) for alias in view.aliases}
    assert aliases and not aliases & set(parent_tensors)
    owned_tensors = [
        t for t in _all_tensors(view.verify).values() if id(t) not in aliases and id(t) not in parent_tensors
    ]
    assert owned_tensors  # the per-width constants and scratch
    with expect_error(RuntimeError, match="release the lane width views"):
        mtp_lanes.release_lane_verify_state(world.model, world.parent)
    with expect_error(RuntimeError, match="release the lane width views of this lane draft state"):
        mtp_lanes.release_lane_draft_state(world.model, world.parent, world.parent_draft)
    assert freed.freed == []
    mtp_lanes.release_lane_width_view(world.model, view)
    freed_ids = {id(t) for t in freed.freed}
    assert freed_ids and not freed_ids & aliases, "an alias was deallocated (that frees the parent's buffer)"
    assert not freed_ids & set(parent_tensors), "a parent tensor was deallocated by the view release"
    assert all(t.alive for t in parent_tensors.values())
    assert not view.active and view.aliases == [] and view.owned == [] and world.parent.views == []
    with expect_error(RuntimeError, match="already released"):
        mtp_lanes.release_lane_width_view(world.model, view)
    with expect_error(ValueError, match="was not allocated by this model owner"):
        mtp_lanes.release_lane_width_view(SimpleNamespace(_state_owner=object()), view)


def test_the_view_states_refuse_the_parents_release_paths(view, world, expect_error) -> None:
    with expect_error(RuntimeError, match="width view whose tensors alias the parent's buffers"):
        mtp_lanes.release_lane_verify_state(world.model, view.verify)
    with expect_error(RuntimeError, match="width view whose tensors alias the parent's buffers"):
        mtp_lanes.release_lane_draft_state(world.model, view.verify, view.draft)


def test_a_view_of_a_released_parent_is_refused(view, world, expect_error) -> None:
    world.parent.ple_rows.release()
    with expect_error(RuntimeError, match="parent of this lane verify width view was released"):
        mtp_lanes._validate_lane_verify_state(world.model, view.verify)


@pytest.mark.parametrize("lanes", (0, 1, 3, 5, 8, 16, True, 2.0, None))
def test_widths_outside_two_and_four_are_refused(world, expect_error, lanes) -> None:
    with expect_error(ValueError, match="lane width views exist for widths"):
        mtp_lanes.allocate_lane_width_view(world.model, world.parent, lanes)
    assert world.parent.views == []


def test_a_width_not_narrower_than_the_parent_is_refused(world, expect_error) -> None:
    with expect_error(ValueError, match="narrower than its 4-lane parent"):
        mtp_lanes.allocate_lane_width_view(world.model, world.parent, 4)


def test_a_view_of_a_view_is_refused(view, world, expect_error) -> None:
    with expect_error(ValueError, match="not over a view"):
        mtp_lanes.allocate_lane_width_view(world.model, view.verify, 2)


def test_a_failed_view_allocation_releases_what_it_made_and_leaves_no_view(
    fake, world, monkeypatch, expect_error
) -> None:
    freed = _Recorder(fake.ttnn)
    calls = []

    def moe(layer, rows):
        calls.append(rows)
        if len(calls) == 2:
            raise RuntimeError("no memory for the second MoE")
        return _FakeLaneMoE(rows=rows)

    monkeypatch.setattr(mtp_lanes, "_allocate_lane_moe", moe)
    with expect_error(RuntimeError, match="no memory for the second MoE"):
        mtp_lanes.allocate_lane_width_view(world.model, world.parent, WIDTH)
    assert world.parent.views == []
    assert all(t.alive for t in _all_tensors(world.parent).values())
    assert all(t not in freed.freed for t in _all_tensors(world.parent).values())


@pytest.mark.parametrize(
    "call",
    (
        lambda m, v: mtp_lanes.write_lane_accepted(m, v, [0] * v.lanes),
        lambda m, v: mtp_lanes.write_lane_active(m, v, [1] * v.lanes, commit=[1] * v.lanes),
        lambda m, v: mtp_lanes.write_lane_tokens(m, v, [[0] * v.rows] * v.lanes),
        lambda m, v: mtp_lanes.write_lane_ple_rows(m, v, [[0] * v.rows] * v.lanes),
        lambda m, v: mtp_lanes.write_lane_verify_inputs(m, v, [[0] * v.rows] * v.lanes),
        lambda m, v: mtp_lanes.commit_lane_verify_host(m, v, [0] * v.lanes),
        lambda m, v: mtp_lanes.import_lane_state(
            m,
            v,
            0,
            position=1,
            ple_context=None,
            gdn_phases=[0],
            backbone_slot=None,
            backbone_pager=None,
            alignment_slot=None,
            alignment_pager=None,
        ),
        lambda m, v: mtp_lanes.evict_lane(m, v, 0, accepted=-1),
        lambda m, v: mtp_lanes.readmit_lane(m, v, 0, None),
    ),
)
def test_host_writers_and_lane_admission_refuse_a_view(view, world, expect_error, call) -> None:
    with expect_error(ValueError, match="host writes take the 4-lane parent state, not a width view"):
        call(world.model, view.verify)


# --------------------------------------------------------------------------- the k5 list


def _attribute_names(source: str) -> set[str]:
    return {node.attr for node in ast.walk(ast.parse(textwrap.dedent(source))) if isinstance(node, ast.Attribute)}


def test_the_k5_list_matches_the_code_constant() -> None:
    assert mtp_lanes.K5_LANE_STATE_TENSORS == K5_LIST


def test_k5_unused_tensors_are_touched_by_no_mtp_lane_path() -> None:
    """The GDN state's ring slots and zero sources (lane per row of one tile) are read or written by no MTP lane path:
    the lane rows bodies, the commit, the admission and the eviction use the recurrent state and the FIR history."""

    forbidden = {"conv", "zero_conv", "zero_recurrent", "conv_window", "advance_conv_window", "shift_conv_window"}
    module_source = inspect.getsource(mtp_lanes)
    # the builder's own docs / list text mention them; the code must not touch them
    assert not _attribute_names(module_source) & forbidden
    gdn_class = gdn_module.Qwen38TTNNGDN
    for name in (
        "forward_rows_lanes",
        "commit_rows_lanes",
        "_rows_body_lanes_chain",
        "_lane_conv_window",
        "_qkv_lanes_view",
        "_expand_lane_rows",
        "_validate_lane_rows_state",
        "allocate_lane_rows_state",
    ):
        assert not _attribute_names(inspect.getsource(getattr(gdn_class, name))) & forbidden, name
    fused_source = inspect.getsource(gdn_rows_scan)
    lane_source = fused_source[fused_source.index("def run_lanes") :]
    assert "zero_conv" not in lane_source and "state.conv" not in lane_source


def test_k5_scratch_residual_is_written_by_the_verify_and_never_by_an_admission() -> None:
    """``alignment.residual`` is rewritten by every verify pass (``_select_lane_residual_rows``) before the draft of the
    same pass reads it, and no admission / eviction path writes it: a width view owns its own copy."""

    for name in (
        "import_lane_state",
        "import_gdn_lane",
        "import_qsa_lane",
        "import_ple_lane",
        "evict_lane",
        "readmit_lane",
    ):
        assert "residual" not in inspect.getsource(getattr(mtp_lanes, name)), name
    verify_source = inspect.getsource(mtp_lanes.forward_verify_lanes) + inspect.getsource(
        mtp_lanes._forward_alignment_lanes
    )
    assert "_select_lane_residual_rows(verify, residual)" in verify_source
    assert "target = verify.alignment.residual" in inspect.getsource(mtp_lanes._select_lane_residual_rows)
    assert "_pad_rows(alignment.residual" in inspect.getsource(mtp_lanes.forward_draft_lanes)  # the draft's roots
    builder = inspect.getsource(mtp_lanes.allocate_lane_width_view)
    assert '"MTP lane residual"' in builder and "ledger.own_tensor" in builder  # own per width, never an alias


def test_k5_moe_input_is_scratch_written_by_the_slice_of_every_pass() -> None:
    source = inspect.getsource(mtp_lanes._forward_layer_verify_lanes)
    assert "output_tensor=layer_state.moe_input" in source  # written before the MoE reads it, in every pass
    assert "moe_input" not in inspect.getsource(mtp_lanes.import_lane_state)
    assert "moe_input" not in inspect.getsource(mtp_lanes.readmit_lane)


def _alias_calls(function) -> dict[str, int]:
    """``{source of the aliased tensor: lane_dim}`` of every ``alias(tensor, lane_dim=N)`` call of ``function``."""

    source = textwrap.dedent(inspect.getsource(function))
    calls = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "alias":
            dims = [k.value.value for k in node.keywords if k.arg == "lane_dim"]
            calls[ast.get_source_segment(source, node.args[0])] = dims[0]
    return calls


def test_every_alias_keeps_its_lane_on_an_outer_axis_or_in_whole_tiles() -> None:
    """The lane axes of the family builders: the outermost axis (recurrent state, prefix states, QSA caches, q / k, PLE
    rows), a whole-tile axis (history, v, beta, g, the QSA staging tiles and raw rows, ``[1,B,1,1]``), or the row axis
    of the k5 tensors (the ring slots, which no lane path touches)."""

    assert _alias_calls(gdn_module.Qwen38TTNNGDN.lane_state_view) == {
        "state.recurrent": 0,
        "slot": 2,  # k5: the ring slots
        "state.zero_recurrent": 0,
        "state.zero_conv": 2,  # k5
    }
    assert _alias_calls(gdn_module.Qwen38TTNNGDNLaneRowsState.width_view) == {
        "self.history": 1,
        "self.qkv_lanes": 2,  # 32 rows per lane: whole tiles
        "self.q": 0,
        "self.k": 0,
        "self.v": 1,
        "self.beta": 1,
        "self.g": 1,
        "buffers.prefix": 0,
    }
    assert _alias_calls(ple_module.Qwen38TTNNPLELaneRowsState.width_view) == {"self.history": 0, "self.normalized": 0}
    assert _alias_calls(qsa_module.Qwen38TTNNQSA.lane_state_view) == {
        "state.compressed_index_cache": 0,
        "state.kv_staging": 1,
        "state.raw_key_ring": 1,
    }
    assert _alias_calls(qsa_module.Qwen38TTNNQSA.lane_verify_state_view) == {
        "state.raw_history": 1,
        "state.raw_rows": 2,  # 32 rows per lane: whole tiles
    }
