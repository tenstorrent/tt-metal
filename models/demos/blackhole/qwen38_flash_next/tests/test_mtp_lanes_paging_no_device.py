# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The QSA lane verify on shared paged pools (no device): constants, the row page table, the derived page-table
starts, and the source pins of the paged compressed write / read and the scratch-row guard."""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_lanes_no_device import (  # noqa: F401  (fixtures)
    ROW_MAJOR,
    U32,
    FakeTensor,
    _derive_lane_inputs,
    _lane_qsa_constants,
    _replicated,
    fake,
    gdn_weights,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step5_verify_no_device import CONTEXT
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

PAGE_SIZE = 128
WIDTH = 24


def _lane_rows(lanes: int) -> torch.Tensor:
    table = torch.zeros(32, WIDTH, dtype=torch.int32)
    for lane in range(lanes):
        table[lane, :6] = torch.tensor([10 + 7 * lane + i for i in range(6)], dtype=torch.int32)
    return table


@pytest.mark.parametrize("lanes,rows", ((4, 5), (2, 4), (1, 5)))
def test_paged_verify_constants_have_zero_lane_offsets(lanes: int, rows: int) -> None:
    flat = qsa_module.qsa_lane_verify_constant_rows(lanes, rows, CONTEXT)
    paged = qsa_module.qsa_lane_verify_constant_rows(lanes, rows, CONTEXT, paged=True)
    base = qsa_module.qsa_row_constants()["block_offsets"].reshape(-1)
    for r in range(32):
        assert torch.equal(paged["block_offsets_lanes"][0, 0, r], base)
        assert torch.equal(paged["arange_slots_rows"][0, 0, r], torch.arange(qsa_module.SPARSE_INDEX_CAPACITY))
    assert set(paged["scratch_row"].reshape(-1).tolist()) == {0}
    for name, value in flat.items():
        if name not in ("block_offsets_lanes", "arange_slots_rows", "scratch_row"):
            assert torch.equal(paged[name], value), name


@pytest.mark.parametrize("lanes,rows", ((4, 5), (3, 4), (1, 5)))
def test_verify_row_page_table_repeats_each_lanes_row_and_pads_with_lane_zero(
    lanes: int, rows: int, expect_error
) -> None:
    table = _lane_rows(lanes)
    host = qsa_module.verify_row_page_table_host(table, rows, lanes)
    assert host.shape == (1, 1, 32, WIDTH) and host.dtype == torch.int32
    for r in range(32):
        lane = r // rows if r < lanes * rows else 0
        assert torch.equal(host[0, 0, r], table[lane]), r
    assert torch.equal(qsa_module.verify_row_page_table_host(table.reshape(1, 1, 32, WIDTH), rows, lanes), host)
    with expect_error(ValueError, "lane page table must be"):
        qsa_module.verify_row_page_table_host(table[:16], rows, lanes)


def _fake_page_starts(table: torch.Tensor, calls: list):
    def starts(block_start, page_table, page_size, lanes, active_row=None):
        calls.append((page_table, page_size, lanes, active_row))
        values = block_start.torch_shards()[0].reshape(-1)
        active = active_row.torch_shards()[0].reshape(-1)
        out = []
        for u in range(lanes):
            v = int(values[u]) * int(active[u])
            row = int(table[u, v // page_size]) * page_size + v % page_size
            out.append(_replicated(torch.tensor([row], dtype=torch.int64).reshape(1, 1, 1, 1), U32, ROW_MAJOR))
        return tuple(out)

    return starts


@pytest.mark.parametrize("single_row", (False, True))
def test_paged_derive_uses_page_table_starts_with_active_mask_and_next_block(monkeypatch, fake, single_row) -> None:
    rows, lanes = 4, 4
    positions, active = (5, 31, 100, 127), (1, 0, 1, 1)
    constants, lane_constants, lane_verify = _lane_qsa_constants(rows, lanes)
    table = _lane_rows(lanes)
    table[1] = 0  # the inactive lane's row is padding
    paging = qsa_module.Qwen38LanePaging(PAGE_SIZE, 64, WIDTH)
    lane_constants = dataclasses.replace(lane_constants, paging=paging, page_table="lane_table")
    lane_verify = dataclasses.replace(lane_verify, paged=True, page_size=PAGE_SIZE, row_page_table="row_table")
    calls: list = []
    monkeypatch.setattr(qsa_block, "paged_lane_row_starts_composed", _fake_page_starts(table, calls))
    from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_lanes_no_device as base

    position_constants, chunk_constants, verify_constants = constants
    position_row = base._u32_row(base._pad32(positions, positions[0]))
    active_row = base._u32_row(base._pad32(active, 1))
    inactive_row = base._u32_row(base._pad32([1 - a for a in active], 0))
    lane_rows = fake.ttnn.gather(position_row, 3, lane_verify.lane_of_row)
    row_positions = fake.ttnn.add(lane_rows, lane_verify.draft_of_row)
    inputs = qsa_module.derive_qsa_lane_verify_inputs(
        position_row,
        active_row,
        inactive_row,
        row_positions,
        position_constants,
        chunk_constants,
        verify_constants,
        lane_constants,
        lane_verify,
        single_row=single_row,
    )
    assert len(calls) == (1 if single_row else 2)
    assert all(c[0] == "lane_table" and c[1] == PAGE_SIZE and c[2] == lanes for c in calls)
    assert all(c[3] is active_row for c in calls)

    def expected(block_offset):
        out = []
        for u in range(lanes):
            v = ((positions[u] & ~31) + block_offset) * active[u]
            out.append(int(table[u, v // PAGE_SIZE]) * PAGE_SIZE + v % PAGE_SIZE)
        return out

    current = [int(t.torch_shards()[0].reshape(-1)[0]) for t in inputs.kv_row_start_lanes]
    assert current == expected(0)
    assert current[1] == 0
    if single_row:
        assert inputs.kv_row_start_next_lanes == ()
    else:
        assert [int(t.torch_shards()[0].reshape(-1)[0]) for t in inputs.kv_row_start_next_lanes] == expected(32)
    read = inputs.kv_read_row.torch_shards()[0].reshape(-1).tolist()
    assert read == [start + i for start in current for i in range(32)]


def test_paged_and_flat_constants_must_agree(fake, expect_error) -> None:
    from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_lanes_no_device as base

    constants, lane_constants, lane_verify = _lane_qsa_constants(4, 2)
    row = base._u32_row(base._pad32([5, 6], 5))
    with expect_error(ValueError):
        qsa_module.derive_qsa_lane_verify_inputs(
            row, row, row, row, *constants, dataclasses.replace(lane_constants, page_table="t"), lane_verify
        )


def test_flat_verify_constants_are_not_paged(fake) -> None:
    _, lane_constants, lane_verify = _lane_qsa_constants(4, 2)
    assert lane_verify.paged is False and lane_verify.row_page_table is None and lane_verify.page_size == 0
    assert lane_constants.page_table is None


def test_paged_verify_sources_carry_the_page_arguments() -> None:
    write = inspect.getsource(qsa_module.Qwen38TTNNQSA._write_compressed_index_lanes_verify)
    assert "_lane_update_page_table(state, lane_constants, count)" in write and "**table_kwargs" in write
    assert "_deallocate(rotated, owned_table)" in write
    assert write.count("page_table=update_idxs") == 0
    score = inspect.getsource(qsa_module.Qwen38TTNNQSA._local_scores_lanes_verify)
    assert "_lane_indexer_kwargs(state, lane_constants, count=count, wide=True)" in score
    assert "state.compressed_index_cache_flat" in score  # the flat path stays
    materialize = inspect.getsource(qsa_module.Qwen38TTNNQSA._materialize_rows_lanes)
    assert '"row_page_table"' in materialize and "paged_rows_composed(local, page_table, page_size)" in materialize
    forward = inspect.getsource(qsa_module.Qwen38TTNNQSA.forward_verify_lanes)
    assert "if state.pools_shared:" in forward
    assert forward.index("if state.pools_shared:") < forward.index("state.kv_scratch_rows < 2 * CACHE_WRITE_ROWS")
    assert "self._score_blocks_lanes_verify(index_query, state, inputs, lane_verify, lane_constants)" in forward
    derive = inspect.getsource(qsa_module.derive_qsa_lane_verify_inputs)
    assert "active_row=active_row" in derive and "paged_lane_row_starts_composed" in derive
    helper = inspect.getsource(qsa_block.paged_lane_row_starts_composed)
    assert "active_row=None" in helper and helper.index("active_row is not None") < helper.index("narrow =")


def test_forward_verify_guard_for_shared_pools(fake, expect_error) -> None:
    layer = object.__new__(qsa_module.Qwen38TTNNQSA)
    state = SimpleNamespace(pools_shared=True, lanes=4)
    lane_verify = SimpleNamespace(lanes=4, paged=False, row_page_table=None)
    layer._validate_lane_state = lambda s: None
    layer._validate_lane_verify_state = lambda v: 4
    layer._validate_rope_rows = lambda *a: None
    with expect_error(ValueError):
        layer.forward_verify_lanes(
            None,
            state,
            SimpleNamespace(),
            cos=None,
            sin=None,
            block_start_cos=None,
            block_start_sin=None,
            inputs=SimpleNamespace(lanes=4),
            constants=None,
            lane_constants=SimpleNamespace(lanes=4, paging=None, page_table=None),
            lane_verify=lane_verify,
        )


# --------------------------------------------------------------------------- MTP lanes on shared pools

from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes  # noqa: E402

LANE_PAGING = qsa_module.Qwen38LanePaging(PAGE_SIZE, 64, WIDTH)


class _HostTable:
    def __init__(self, host):
        self.host = host
        self.shape = tuple(host.shape)


def _install_table_fakes(monkeypatch):
    """Host-side stand-ins of the table upload: ``from_torch`` keeps the host image, ``copy_host_to_device_tensor``
    lands it in the device placeholder."""

    monkeypatch.setattr(mtp_lanes.ttnn, "from_torch", lambda host, **kwargs: _HostTable(host), raising=False)
    monkeypatch.setattr(
        mtp_lanes.ttnn,
        "copy_host_to_device_tensor",
        lambda host, device: setattr(device, "image", host.host),
        raising=False,
    )
    monkeypatch.setattr(mtp_lanes, "replicate_tensor_2d_mesh_mapper", lambda mesh: None)
    monkeypatch.setattr(qsa_module, "replicate_tensor_2d_mesh_mapper", lambda mesh: None)
    monkeypatch.setattr(mtp_lanes, "_validate_lane_verify_state", lambda model, verify: None)
    monkeypatch.setattr(mtp_lanes, "_validate_lane_draft_state", lambda model, verify, draft: None)


def _verify_constants(rows: int, lanes: int):
    constants = SimpleNamespace(rows=rows, lanes=lanes, row_page_table=SimpleNamespace(shape=(1, 1, 32, WIDTH)))
    constants.write_row_page_table = lambda mesh, host, keep=None: (
        qsa_module.Qwen38TTNNQSALaneVerifyConstants.write_row_page_table(constants, mesh, host, keep)
    )
    return constants


def _paged_verify(lanes: int = 4, rows: int = 5):
    return SimpleNamespace(
        lanes=lanes,
        rows=rows,
        views=[],
        is_view=False,
        qsa_lane_constants=SimpleNamespace(paging=LANE_PAGING, page_table=SimpleNamespace(shape=(1, 1, 32, WIDTH))),
        qsa_lane_verify_constants=_verify_constants(rows, lanes),
        parent=None,
        storage_lanes=0,
        stored_lanes=lanes,
    )


def test_write_lane_page_table_zeroes_inactive_rows_and_writes_every_table(monkeypatch) -> None:
    _install_table_fakes(monkeypatch)
    verify = _paged_verify(lanes=4, rows=5)
    draft = SimpleNamespace(is_view=False, qsa_lane_verify_constants=_verify_constants(1, 4))
    table = _lane_rows(4)
    keep: list = []
    model = SimpleNamespace(mesh_device="mesh")
    host = mtp_lanes.write_lane_page_table(model, verify, table, [0, 2, 3], keep, draft=draft)
    expected = table.clone()
    expected[1] = 0
    assert torch.equal(host, expected)
    lane_image = verify.qsa_lane_constants.page_table.image
    assert lane_image.shape == (1, 1, 32, WIDTH) and torch.equal(lane_image[0, 0], expected)
    row_image = verify.qsa_lane_verify_constants.row_page_table.image
    assert torch.equal(row_image, qsa_module.verify_row_page_table_host(expected, 5, 4))
    assert torch.count_nonzero(row_image[0, 0, 5:10]) == 0  # lane 1's five rows are the null block
    draft_image = draft.qsa_lane_verify_constants.row_page_table.image
    assert torch.equal(draft_image, qsa_module.verify_row_page_table_host(expected, 1, 4))
    assert len(keep) == 3  # lane table, verify row table, draft row table


def test_write_lane_page_table_with_no_active_lane_is_all_zeros_and_refuses_bad_input(
    monkeypatch, expect_error
) -> None:
    _install_table_fakes(monkeypatch)
    verify = _paged_verify()
    model = SimpleNamespace(mesh_device="mesh")
    host = mtp_lanes.write_lane_page_table(model, verify, _lane_rows(4).reshape(1, 1, 32, WIDTH), [])
    assert torch.count_nonzero(host) == 0 and torch.count_nonzero(verify.qsa_lane_constants.page_table.image) == 0
    rows = [[1, 2, 3], [4], [], [5, 6]]  # short and missing rows are padding
    host = mtp_lanes.write_lane_page_table(model, verify, rows, [0, 1, 3])
    assert host[0, :3].tolist() == [1, 2, 3] and host[1, 0] == 4 and torch.count_nonzero(host[2]) == 0
    assert host[3, :2].tolist() == [5, 6]
    with expect_error(ValueError):  # a lane outside the state
        mtp_lanes.write_lane_page_table(model, verify, rows, [4])
    with expect_error(ValueError):  # a block id outside the pools
        mtp_lanes.write_lane_page_table(model, verify, [[LANE_PAGING.num_pages]], [0])
    flat = SimpleNamespace(**{**vars(verify), "qsa_lane_constants": SimpleNamespace(paging=None, page_table=None)})
    with expect_error(ValueError):
        mtp_lanes.write_lane_page_table(model, flat, rows, [0])


class _FakeQSA:
    def __init__(self):
        self.calls = []
        self.allocated_context = CONTEXT

    def allocate_lane_state(self, lanes, **kwargs):
        self.calls.append(("state", lanes, kwargs))
        return SimpleNamespace(pools_shared="pools" in kwargs)

    def allocate_lane_verify_state(self, lanes):
        return "verify-state"


@pytest.mark.parametrize("paged", (False, True))
def test_layer_lane_state_allocation_passes_the_pools_or_the_scratch_rows(monkeypatch, paged) -> None:
    attention = _FakeQSA()
    layer = SimpleNamespace(attention=attention, ple=None, layer_index=3, namespace="backbone", mlp=None)
    monkeypatch.setattr(
        mtp_lanes,
        "_allocate_lane_moe",
        lambda layer_, rows: SimpleNamespace(rows=rows, release_owned_buffers=lambda: None),
    )
    pools = SimpleNamespace(kv="kv", index="index")
    state = mtp_lanes._allocate_layer_lane_state(
        layer, 4, 5, None, None, moe_rows=mtp_lanes.CHUNK_ROWS, pools=pools if paged else None
    )
    ((_, lanes, kwargs),) = attention.calls
    assert lanes == 4
    if paged:
        assert kwargs == {"pools": pools} and state.attention_state.pools_shared
    else:
        assert kwargs == {"kv_scratch_rows": mtp_lanes.kv_scratch_rows(CONTEXT)}


def test_shared_pool_arguments_are_validated(expect_error) -> None:
    qsa_layer = SimpleNamespace(attention=object(), layer_index=3)
    model = SimpleNamespace(layers=[qsa_layer])
    mtp_layer = SimpleNamespace(attention=object())
    mtp_lanes._validate_shared_pools(model, mtp_layer, None, None, None)
    mtp_lanes._validate_shared_pools(model, mtp_layer, {3: "p"}, "m", LANE_PAGING)
    for args in (
        ({3: "p"}, None, None),
        (None, None, LANE_PAGING),
        ({3: "p"}, None, LANE_PAGING),
        ({}, "m", LANE_PAGING),
    ):
        with expect_error(ValueError):
            mtp_lanes._validate_shared_pools(model, mtp_layer, *args)


def test_verify_allocation_builds_paged_constants_and_the_mtp_layer_on_its_pools() -> None:
    source = inspect.getsource(mtp_lanes.allocate_lane_verify_state)
    assert "allocated_context=model.allocated_context, paging=paging" in source
    assert source.count("paging=paging,") >= 1
    assert "layer_pools = mtp_pools if layer is mtp_layer else pools[layer.layer_index]" in source
    draft = inspect.getsource(mtp_lanes.allocate_lane_draft_state)
    assert "paging=verify.qsa_lane_constants.paging" in draft
    view = inspect.getsource(mtp_lanes.allocate_lane_width_view)
    assert "build_view" in view and "paging=parent.qsa_lane_constants.paging" in view
    assert "paging=parent.qsa_lane_constants.paging" in inspect.getsource(mtp_lanes._lane_draft_state_view)


def _real_verify_constants(row_table, **fields):
    names = [f.name for f in dataclasses.fields(qsa_module.Qwen38TTNNQSALaneVerifyConstants)]
    values = {name: f"{name}-tensor" for name in names}
    values.update(
        lanes=2,
        rows=5,
        allocated_context=CONTEXT,
        lanes_of_rows=(),
        pick_users=("a", "b"),
        storage_lanes=0,
        paged=True,
        page_size=PAGE_SIZE,
        row_page_table=row_table,
    )
    values.update(fields)
    return qsa_module.Qwen38TTNNQSALaneVerifyConstants(**values)


@pytest.mark.parametrize("paged", (False, True))
def test_width_view_verify_constants_share_the_parents_row_table_and_never_free_it(monkeypatch, paged) -> None:
    freed: list = []
    record = lambda *tensors: freed.extend(t for t in tensors if t is not None)  # noqa: E731
    monkeypatch.setattr(qsa_module, "_deallocate", record)
    monkeypatch.setattr(mtp_lanes, "_deallocate", record)
    built_args: list = []

    def build(mesh, contract, **kwargs):
        built_args.append(kwargs)
        return _real_verify_constants("built-table" if kwargs.get("paging") else None, paged=bool(kwargs.get("paging")))

    monkeypatch.setattr(qsa_module.Qwen38TTNNQSALaneVerifyConstants, "build", staticmethod(build))
    parent = _real_verify_constants("parent-table" if paged else None, paged=paged)
    paging = LANE_PAGING if paged else None
    constants, release = mtp_lanes._build_view_lane_verify_constants(
        "mesh", "contract", parent, lanes=2, rows=5, allocated_context=CONTEXT, storage_lanes=4, paging=paging
    )
    assert built_args[0]["storage_lanes"] == 4 and built_args[0].get("paging") is paging
    if not paged:
        assert constants.row_page_table is None and not constants.paged
        release()
        assert "lane_of_row-tensor" in freed
        return
    assert constants.row_page_table == "parent-table" and constants.paged
    assert "built-table" in freed  # the view's own table is dropped at build
    freed.clear()
    release()
    assert "parent-table" not in freed and "lane_of_row-tensor" in freed


def _shared_world(fake, monkeypatch, gdn_weights):
    from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_lanes_no_device as base

    return base, *base._lifecycle_world(fake, monkeypatch, gdn_weights, paged=True)


@pytest.mark.parametrize("lane,target,position", ((2, 2, 30), (1, 3, 100)))
def test_lane_evict_and_readmit_on_shared_pools_skip_the_kv_and_compressed_families(
    expect_error, fake, monkeypatch, gdn_weights, lane, target, position
) -> None:
    base, model, verify = _shared_world(fake, monkeypatch, gdn_weights)
    base._seed_lane(verify, lane, 300 + lane, position, (7, 8))
    for other in range(4):
        if other != lane:
            base._seed_lane(verify, other, 400 + other, 5 + other, (1, other))
    before = base._lane_tensors(verify, lane)
    image = mtp_lanes.evict_lane(model, verify, lane, accepted=-1)
    assert image.kv == [] and image.compressed == [] and image.kv_rows == 0
    assert image.alignment_kv is None and image.alignment_compressed is None
    assert image.position == position and image.ple_context == (7, 8)
    assert len(image.raw_history) == 1 and len(image.recurrent) == 1
    digest = image.digest()
    assert digest == mtp_lanes.evict_lane(model, verify, lane, accepted=-1).digest()
    pools = [
        verify.layers[0].attention_state.packed_kv_cache,
        verify.layers[0].attention_state.compressed_index_cache,
        verify.alignment.layer_state.attention_state.packed_kv_cache,
        verify.alignment.layer_state.attention_state.compressed_index_cache,
    ]
    pool_before = [[x.clone() for x in tensor.locals] for tensor in pools]
    base._seed_lane(verify, target, 999, 3, (9, 9))
    pool_scrambled = [[x.clone() for x in tensor.locals] for tensor in pools]
    mtp_lanes.readmit_lane(model, verify, target, image)
    for tensor, scrambled in zip(pools, pool_scrambled):  # readmit wrote no pool row
        assert all(torch.equal(x, y) for x, y in zip(tensor.locals, scrambled))
    after = base._lane_tensors(verify, target)
    for name in ("raw_history", "alignment_raw_history", "recurrent", "gdn_history", "ple_history"):
        assert all(torch.equal(a, b) for a, b in zip(after[name], before[name])), name
    assert verify.positions.positions[target] == position
    assert mtp_lanes.evict_lane(model, verify, target, accepted=-1).digest() == digest
    with expect_error(ValueError):  # a flat image (with KV rows) does not fit lanes on shared pools
        mtp_lanes.readmit_lane(model, verify, target, dataclasses.replace(image, kv=[[torch.zeros(1)]], kv_rows=32))


def test_flat_lane_image_keeps_its_kv_and_compressed_rows(fake, monkeypatch, gdn_weights, expect_error) -> None:
    from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_lanes_no_device as base

    model, verify = base._lifecycle_world(fake, monkeypatch, gdn_weights)
    base._seed_lane(verify, 1, 5, 40, (1, 2))
    image = mtp_lanes.evict_lane(model, verify, 1, accepted=-1)
    assert len(image.kv) == 1 and len(image.compressed) == 1 and image.kv_rows == mtp_lanes.kv_image_rows(40, CONTEXT)
    assert image.alignment_kv is not None and image.alignment_compressed is not None
    with expect_error(ValueError):  # an image without KV rows does not fit flat lanes
        mtp_lanes.readmit_lane(
            model,
            verify,
            1,
            dataclasses.replace(image, kv=[], compressed=[], alignment_kv=None, alignment_compressed=None),
        )


def test_qsa_lane_import_on_shared_pools_lands_only_the_raw_history(fake) -> None:
    from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_lanes_no_device as base

    base._install_admission_ops(fake)
    module = base._qsa_module(fake.ttnn)
    position, lane, lanes = 31, 2, 4
    stream = base._Stream(seed=300)
    generic, verify_b1 = module.allocate_generic_state(), module.allocate_verify_state()
    base._seed_caches(module, fake.ttnn, stream, generic, verify_b1, position)
    for target in generic.raw_key_ring.locals:
        torch.manual_seed(400)
        target.copy_(base._bf16(1, 1, 32, 128))
    module.sync_verify_raw_history_from_ring(generic, verify_b1, position=position)
    slot = SimpleNamespace(tensors={})  # no KV family in the slot
    pager = SimpleNamespace(packs={"ring": base._pack(generic.raw_key_ring.torch_shards(), base.BF16, base.TILE)})
    state = dataclasses.replace(
        module.allocate_lane_state(lanes, kv_scratch_rows=mtp_lanes.kv_scratch_rows(CONTEXT)), pools_shared=True
    )
    lane_verify = module.allocate_lane_verify_state(lanes)
    for tensor in state.packed_kv_cache.locals:
        tensor.fill_(3.0)
    mtp_lanes.import_qsa_lane(
        state, lane_verify, pager, slot, 0, position=position, lane=lane, allocated_context=CONTEXT
    )
    for tensor in state.packed_kv_cache.locals:
        assert torch.count_nonzero(tensor.float() - 3.0) == 0
    assert torch.count_nonzero(state.compressed_index_cache.torch_shards()[0].float()) == 0
    assert torch.equal(
        lane_verify.raw_history.torch_shards()[0][0, lane], verify_b1.raw_history.torch_shards()[0][0, 0]
    )


def test_import_lane_state_checks_the_pager_layouts_against_the_pools(monkeypatch, expect_error) -> None:
    monkeypatch.setattr(mtp_lanes, "_validate_lane_verify_state", lambda model, verify: None)
    monkeypatch.setattr(mtp_lanes, "_require_parent", lambda verify, what: None)
    verify = _paged_verify()
    pager = SimpleNamespace(layout=SimpleNamespace(kv=("kv",), compressed=()))
    kwargs = dict(
        position=5,
        ple_context=None,
        gdn_phases=(),
        backbone_slot=None,
        backbone_pager=pager,
        alignment_slot=None,
        alignment_pager=pager,
    )
    model = SimpleNamespace(layers=[])
    with expect_error(ValueError):  # the layout still carries the KV pool tensors
        mtp_lanes.import_lane_state(model, verify, 0, **kwargs)
    with expect_error(ValueError):  # shared=False on a paged state
        mtp_lanes.import_lane_state(model, verify, 0, shared=False, **kwargs)


def test_alignment_layout_drops_the_pools_when_shared() -> None:
    generic = SimpleNamespace(packed_kv_cache="kv", compressed_index_cache="c", kv_staging="s", raw_key_ring="r")
    flat = mtp_lanes.lane_alignment_layout(generic, CONTEXT, shared=False)
    shared = mtp_lanes.lane_alignment_layout(generic, CONTEXT, shared=True)
    assert flat.kv == ("kv",) and flat.compressed == ("c",)
    assert shared.kv == () and shared.compressed == () and shared.staging == ("s",) and shared.ring == ("r",)
