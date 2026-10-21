# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The lane lifecycle's host side, no device: the page table (session -> lane or host slot), the residue-aligned
admission scheduler, the packed lane image's families and bytes, the helper-thread mover, and the source pins of the
pager (fill_cache at the lane's batch index, the pack concats into persistent buffers, the raw non-blocking replays
behind an event fence, the residue rule before any re-admission write) and of the model glue."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import lanes as lanes_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import (
    FILL_FAMILIES,
    GDN_LAYERS,
    LANE_AXIS,
    QSA_LAYERS,
    RING_SLOTS,
    Qwen38LaneAdmissionScheduler,
    Qwen38LaneLayout,
    Qwen38LaneMover,
    Qwen38LanePageTable,
)

SOURCE = inspect.getsource(lanes_module)


# --------------------------------------------------------------------------- the page table


def test_page_table_places_parks_and_frees_with_single_ownership(expect_error):
    table = Qwen38LanePageTable(4, 2)
    assert table.free_lanes() == [0, 1, 2, 3] and table.free_slots() == [0, 1]
    a = table.register("a", position=12, committed=range(12), ple_context=(3, 4))
    table.register("b")
    table.place("a", 1)
    table.place("b", 0)
    assert table.free_lanes() == [2, 3] and a.lane == 1 and a.resident and a.residue == 0
    with expect_error(ValueError, match="owned by session 'b'"):
        table.place("a", 0)
    with expect_error(ValueError, match="already resident"):
        table.place("a", 2)
    table.park("a", 1)
    assert a.lane is None and a.slot == 1 and table.free_lanes() == [1, 2, 3] and table.free_slots() == [0]
    with expect_error(ValueError, match="not resident"):
        table.park("a", 0)
    with expect_error(ValueError, match="owned by session 'a'"):
        table.park("b", 1)
    # Re-admission from the slot frees the slot.
    table.place("a", 3)
    assert a.lane == 3 and a.slot is None and table.free_slots() == [0, 1]
    table.touch("a", position=20, committed=range(20), ple_context=(5, 6), now_ns=1_000)
    table.touch("b", position=8, committed=range(8), ple_context=None, now_ns=5_000)
    assert [s.session_id for s in table.idle_residents(now_ns=6_000, idle_ns=2_000)] == ["a"]
    assert [s.session_id for s in table.idle_residents(now_ns=9_000, idle_ns=2_000)] == ["a", "b"]
    table.drop("a")
    assert "a" not in table.sessions and table.free_lanes() == [1, 2, 3]
    with expect_error(KeyError):
        table.session("a")
    with expect_error(ValueError, match="already registered"):
        table.register("b")
    with expect_error(ValueError, match=r"lane must be an int in \[0,4\)"):
        table.place("b", 4)


def test_page_table_rejects_bad_counts(expect_error):
    with expect_error(ValueError):
        Qwen38LanePageTable(0, 1)
    with expect_error(ValueError):
        Qwen38LanePageTable(33, 1)
    with expect_error(ValueError):
        Qwen38LanePageTable(4, -1)
    with expect_error(ValueError):
        Qwen38LanePageTable(4, True)


# --------------------------------------------------------------------------- the admission scheduler


def test_scheduler_admits_only_the_aligned_residue_class_fifo():
    scheduler = Qwen38LaneAdmissionScheduler()
    scheduler.enqueue("p12", 12)  # residue 0
    scheduler.enqueue("p5", 5)  # residue 1
    scheduler.enqueue("p16", 16)  # residue 0
    scheduler.enqueue("p0", 0)  # residue 0
    assert scheduler.waits(0) == {"p12": 0, "p5": 1, "p16": 0, "p0": 0}
    assert scheduler.waits(2) == {"p12": 2, "p5": 3, "p16": 2, "p0": 2}
    admitted = scheduler.admissions(0, free_lanes=[3, 5])
    assert [(a.session_id, a.lane, a.position, a.passed_boundaries) for a in admitted] == [
        ("p12", 3, 12, 0),
        ("p16", 5, 16, 0),
    ]
    assert [item[0] for item in scheduler.waiting] == ["p5", "p0"]
    # The next boundary has residue 1: p5 joins, p0 waits on (passed twice).
    admitted = scheduler.admissions(1, free_lanes=[7])
    assert [(a.session_id, a.lane, a.passed_boundaries) for a in admitted] == [("p5", 7, 1)]
    assert scheduler.waiting == [("p0", 0, 2)]
    assert scheduler.admissions(2, free_lanes=[7]) == [] and scheduler.admissions(3, free_lanes=[7]) == []
    admitted = scheduler.admissions(0, free_lanes=[7])
    assert [(a.session_id, a.lane, a.passed_boundaries) for a in admitted] == [("p0", 7, 4)]
    assert scheduler.waiting == []


def test_scheduler_idle_batch_takes_the_first_waiters_residue_and_needs_free_lanes(expect_error):
    scheduler = Qwen38LaneAdmissionScheduler()
    scheduler.enqueue("a", 7)  # residue 3
    scheduler.enqueue("b", 3)  # residue 3
    scheduler.enqueue("c", 8)  # residue 0
    assert scheduler.waits(None) == {"a": 0, "b": 0, "c": 0}
    admitted = scheduler.admissions(None, free_lanes=[0, 1, 2])
    assert [(a.session_id, a.lane) for a in admitted] == [("a", 0), ("b", 1)]
    assert scheduler.waiting == [("c", 8, 1)]
    # No free lane: nothing joins, everyone is passed over.
    assert scheduler.admissions(0, free_lanes=[]) == [] and scheduler.waiting == [("c", 8, 2)]
    assert scheduler.admissions(None, free_lanes=[]) == [] and scheduler.waiting == [("c", 8, 3)]
    with expect_error(ValueError, match="already waiting"):
        scheduler.enqueue("c", 8)
    with expect_error(ValueError, match="non-negative int"):
        scheduler.enqueue("d", -1)
    with expect_error(ValueError, match=r"resident residue must be in \[0,4\)"):
        scheduler.admissions(4, free_lanes=[0])


def test_scheduler_wait_matches_the_contract_rule():
    for residue in range(4):
        for position in range(0, 40, 3):
            scheduler = Qwen38LaneAdmissionScheduler()
            scheduler.enqueue("s", position)
            expected = contracts_module.admission_wait_steps(residue, position)
            assert scheduler.waits(residue) == {"s": expected}
            boundary = residue
            joined = None
            for step in range(4):
                if scheduler.admissions(boundary, free_lanes=[0]):
                    joined = step
                    break
                boundary = (boundary + 1) % 4
            assert joined == expected


# --------------------------------------------------------------------------- the packed image


def _fake_layout(lanes: int, context: int, *, qsa: int = QSA_LAYERS, gdn: int = GDN_LAYERS, ple: int = 9):
    fake = lambda: SimpleNamespace()  # noqa: E731 - only the counts enter families() and the bytes
    return Qwen38LaneLayout(
        lanes,
        context,
        tuple(fake() for _ in range(qsa)),
        tuple(fake() for _ in range(qsa)),
        tuple(fake() for _ in range(qsa)),
        tuple(fake() for _ in range(qsa)),
        tuple(fake() for _ in range(gdn)),
        tuple(fake() for _ in range(4 * gdn)),
        tuple(fake() for _ in range(ple)),
    )


def test_layout_constants_and_families_follow_the_storage_plan():
    assert (QSA_LAYERS, GDN_LAYERS, RING_SLOTS) == (12, 36, 144)
    assert FILL_FAMILIES == ("compressed", "recurrent", "staging", "ring", "ple")
    assert LANE_AXIS == {"kv": 2, "compressed": 0, "recurrent": 0, "staging": 1, "ring": 1, "ple": 1, "conv": 2}
    layout = _fake_layout(8, 32768)
    assert len(layout.tensors()) == 237
    families = {f.name: f for f in layout.families()}
    assert list(families) == ["kv", "compressed", "staging", "ring", "recurrent", "conv", "ple"]
    assert (
        families["kv"].local_shape == (1, 1, 32768, 512)
        and families["kv"].count == 12
        and families["kv"].shard_dim == 1
    )
    assert families["compressed"].local_shape == (12, 1, 8192 + 32, 128) and families["compressed"].shard_dim is None
    assert families["staging"].local_shape == (12, 1, 32, 512) and families["ring"].local_shape == (12, 1, 32, 128)
    assert families["recurrent"].local_shape == (36, 12, 128, 128) and families["recurrent"].global_shape == (
        36,
        48,
        128,
        128,
    )
    assert families["conv"].local_shape == (1, 144, 1, 2560) and families["conv"].global_shape == (1, 144, 1, 10240)
    assert families["ple"].local_shape == (9, 1, 4, 640) and families["ple"].global_shape == (9, 1, 4, 2560)
    assert (
        families["conv"].layout is lanes_module.ttnn.ROW_MAJOR_LAYOUT
        and families["kv"].layout is lanes_module.ttnn.ROW_MAJOR_LAYOUT
    )
    assert families["recurrent"].dtype is lanes_module.ttnn.float32
    # 13,056 bytes per context row + the fixed per-lane bytes (the lane-state fork micro-test's 83,162,112 at 4k).
    for context in (4096, 8192, 32768):
        assert _fake_layout(8, context).image_bytes_per_device() == 13_056 * context + 29_684_736
    assert _fake_layout(8, 4096).image_bytes_per_device() == 83_162_112
    # A one-layer GDN subset: recurrent and ring slots only.
    subset = _fake_layout(8, 4096, qsa=0, gdn=1, ple=0)
    assert [f.name for f in subset.families()] == ["recurrent", "conv"]
    assert subset.image_bytes_per_device() == 786_432 + 4 * 2560 * 2


def test_layout_validate_checks_the_family_counts_before_any_tensor(expect_error):
    with expect_error(RuntimeError, match="one tensor per QSA layer"):
        Qwen38LaneLayout(8, 4096, (), (SimpleNamespace(),), (), (), (), (), ()).validate(None)
    with expect_error(RuntimeError, match=r"ring slots must be 4 per recurrent"):
        Qwen38LaneLayout(8, 4096, (), (), (), (), (SimpleNamespace(),), (SimpleNamespace(),) * 3, ()).validate(None)
    with expect_error(RuntimeError, match="PLE slots must be 0 or 9"):
        Qwen38LaneLayout(8, 4096, (), (), (), (), (), (), (SimpleNamespace(),) * 2).validate(None)
    with expect_error(RuntimeError, match="holds no tensors"):
        Qwen38LaneLayout(8, 4096, (), (), (), (), (), (), ()).validate(None)
    with expect_error(RuntimeError, match=r"lane layout lanes must be in \[1,32\]"):
        _fake_layout(33, 4096).validate(None)


# --------------------------------------------------------------------------- the mover


def test_mover_runs_one_move_at_a_time_and_reraises(expect_error):
    mover = Qwen38LaneMover()
    assert not mover.busy
    mover.submit(lambda: "moved")
    assert mover.busy
    with expect_error(RuntimeError, match="already in flight"):
        mover.submit(lambda: None)
    assert mover.wait() == "moved" and not mover.busy
    with expect_error(RuntimeError, match="no page move"):
        mover.wait()

    def failing():
        raise ValueError("slab refused")

    mover.submit(failing)
    with expect_error(ValueError, match="slab refused"):
        mover.wait()
    assert not mover.busy


# --------------------------------------------------------------------------- source pins


def _body(name: str) -> str:
    return inspect.getsource(getattr(lanes_module.Qwen38TTNNLanePager, name))


def test_pager_writes_back_by_fill_cache_at_the_lane_and_packs_into_persistent_buffers():
    unpack, pack = _body("_unpack_body"), _body("_pack_body")
    assert "ttnn.fill_cache(cache_view, filled, batch_idx=lane)" in unpack
    assert unpack.count("ttnn.matmul(") == 1 and "output_tensor=slot" in unpack  # the ring-slot selection write
    assert "self._concat_tree(parts, 0, release_parts=True, output=self.packs[name])" in pack
    assert 'output_tensor=self.packs["conv"]' in pack
    concat = _body("_concat_tree")
    # ttnn.concat takes no output tensor: the reduced tensor is copied into the pack in place.
    assert "output_tensor" not in concat and "ttnn.copy(result, output)" in concat and "CONCAT_FAN_IN" in concat
    # A lane slice without an output tensor is owned (a whole-tensor slice would alias the state or the pack).
    slice_lane = _body("_slice_lane")
    assert "return slice_owned(tensor, start, end)" in slice_lane and slice_lane.count("ttnn.slice(") == 1
    owned = inspect.getsource(lanes_module.slice_owned)
    assert "end == _shape(tensor)" in owned and "ttnn.clone(tensor" in owned
    # No host I/O inside the traced bodies.
    for text in (pack, unpack, concat, slice_lane, _body("_select_write")):
        for forbidden in ("to_torch", "from_torch", "copy_host_to_device_tensor", "copy_device_to_host_tensor"):
            assert forbidden not in text, forbidden


def test_pager_moves_replay_raw_behind_an_event_fence_and_apply_the_residue_rule_first():
    evict, readmit = _body("evict"), _body("readmit")
    # The pack / unpack run through _run_pack / _run_unpack: a raw non-blocking trace replay when captured, else the
    # eager body (the default; a replay's baked scratch aliases another lane's trace, so tracing is opt-in).
    for text in (_body("_run_pack"), _body("_run_unpack")):
        assert "ttnn._ttnn_execute_trace(self.mesh_device" in text and "blocking=False" in text
        assert "ttnn.execute_trace(" not in text
    for text in (evict, readmit):
        assert text.count("self._fence()") == 3
    assert "self._run_pack(lane)" in evict and "self._run_unpack(lane)" in readmit
    assert "position_row.admit(lane, slot.position)" in readmit
    assert readmit.index("position_row.admit(lane, slot.position)") < readmit.index("copy_host_to_device_tensor")
    assert (
        "update_padded_kv_cache(\n                        cache, staging, 0, 0, 1, lane * context, qsa_module.STAGING_AXIS"
        in readmit
    )
    assert "output_tensor=staging" in evict and "blocking=False, cq_id=self.cq_id" in evict
    # A move's device allocations are acknowledged corruptible: they are consumed before the next replay and may
    # sit in a lane trace's scratch (the tracker refused the replay after the first eviction otherwise).
    for text in (evict, readmit):
        assert text.count("with corruptible_allocation_scope(self.mesh_device)") == 1
        assert text.index("corruptible_allocation_scope") < text.index("self._fence()")
    assert "(self.evict(lane, slot), self.readmit(slot, lane)) for lane in range(self.layout.lanes)" in _body(
        "warm_moves"
    )
    assert "KV_STAGINGS = 2" in SOURCE and "index % KV_STAGINGS" in evict and "index % KV_STAGINGS" in readmit
    assert "tt-smi" not in SOURCE and "kill" not in SOURCE


def test_pager_construction_warms_up_pack_then_unpack_per_lane_before_the_captures():
    init = inspect.getsource(lanes_module.Qwen38TTNNLanePager.__init__)
    assert (
        'bodies = (("pack", self._pack_body, self.pack_traces), ("unpack", self._unpack_body, self.unpack_traces))'
        in init
    )
    assert init.index("self.warmup_ms[name].append") < init.index("ttnn.begin_trace_capture")
    assert "corruptible_allocation_scope(mesh_device)" in init
    assert "self.release()" in init  # a failed construction releases what it allocated


def test_model_lane_layout_glue_collects_every_family_and_leaves_the_position_row_with_the_model():
    glue = inspect.getsource(model_module.Qwen38TTNNTextModel.lane_layout)
    for field in (
        "packed_kv_cache",
        "compressed_index_cache",
        "kv_staging",
        "raw_key_ring",
        "recurrent",
        "s.conv",
        "ple_state.conv",
    ):
        assert field in glue, field
    assert "layout.validate(self.mesh_contract)" in glue
    assert "state.position" not in glue.split('"""')[2]  # the row stays with the model (the docstring names it)
    # The 1-row production bodies and the lane bodies of B3 are not touched by the glue.
    for name in (
        "forward_decode",
        "capture_decode_generic",
        "forward_decode_lanes",
        "capture_decode_lanes",
        "admit_lane",
    ):
        assert "lanes_module" not in inspect.getsource(getattr(model_module.Qwen38TTNNTextModel, name))
        assert "Qwen38LaneLayout" not in inspect.getsource(getattr(model_module.Qwen38TTNNTextModel, name))


def test_lane_shapes_match_the_component_states():
    shapes = lanes_module._lane_shapes(8, 4096)
    assert shapes["kv"][0] == (1, 1, 8 * 4096, 512) and shapes["compressed"][0] == (8, 1, 1024 + 32, 128)
    assert shapes["staging"][0] == (1, 8, 32, 512) and shapes["ring"][0] == (1, 8, 32, 128)
    assert shapes["recurrent"][0] == (8, 12, 128, 128) and shapes["conv"][0] == (1, 1, 8, 2560)
    assert shapes["ple"][0] == (1, 8, 4, 640)
    placements = {name: spec[1].value for name, spec in shapes.items()}
    assert placements == {
        "kv": "kv_pair_grouped",
        "compressed": "replicated",
        "staging": "kv_pair_grouped",
        "ring": "replicated",
        "recurrent": "head_sharded",
        "conv": "head_sharded",
        "ple": "hidden_sharded",
    }
    # The QSA / GDN / PLE lane-state sources allocate exactly these shapes.
    qsa = inspect.getsource(lanes_module.qsa_module.Qwen38TTNNQSA.allocate_lane_state)
    assert (
        "(1, 1, lanes * self.allocated_context + kv_scratch_rows, 2 * HEAD_DIM)" in qsa
        and "(lanes, 1, rows, INDEX_HEAD_DIM)" in qsa
    )
    assert "(1, lanes, CACHE_WRITE_ROWS, 2 * HEAD_DIM)" in qsa and "(1, lanes, CACHE_WRITE_ROWS, INDEX_HEAD_DIM)" in qsa
    gdn = inspect.getsource(lanes_module.gdn_module.Qwen38TTNNGDNState.allocate)
    assert (
        "(batch_size, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM)" in gdn
        and "(1, 1, batch_size, QKV_WIDTH_PER_DEVICE)" in gdn
    )
    assert re.search(
        r"return \(1, lanes, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE\)", inspect.getsource(lanes_module.ple_module)
    )
