# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F155 (owner 2026-10-22): the framework measures device time with the program real-time profiler: sections and op
calls are device-operation id ranges, the records are charged after the run, nothing syncs per section or per op."""

import sys
import types

import pytest

from models.demos.common.bringup.testing import profiler


class Rec:
    def __init__(self, rid, chip, start, end, f=1.0):
        self.runtime_id, self.chip_id, self.start_timestamp, self.end_timestamp, self.frequency = (
            rid,
            chip,
            start,
            end,
            f,
        )


def fake_ttnn(monkeypatch, active=True):
    """A ttnn stand-in: ops take device-operation ids, the test delivers the records; syncs are counted."""
    st = types.SimpleNamespace(next_id=100, syncs=0, cb=None, launched=[])
    tt = types.ModuleType("ttnn")
    tt.Tensor = type("Tensor", (), {})
    tt.synchronize_device = lambda mesh: setattr(st, "syncs", st.syncs + 1)
    tt._ttnn = types.SimpleNamespace(get_device_operation_id=lambda: st.next_id)

    def register(cb):
        st.cb = cb
        return 1

    tt.device = types.SimpleNamespace(
        IsProgramRealtimeProfilerActive=lambda: active,
        RegisterProgramRealtimeProfilerCallback=register,
        UnregisterProgramRealtimeProfilerCallback=lambda h: None,
    )

    class FastOperation:
        def __init__(self, name, programs):
            self.python_fully_qualified_name, self.programs = name, programs

        def __call__(self, *a, **k):
            for _ in range(self.programs):
                st.launched.append(st.next_id)
                st.next_id += 1

    dec = types.ModuleType("ttnn.decorators")
    dec.FastOperation = FastOperation
    tt.decorators = dec
    monkeypatch.setitem(sys.modules, "ttnn", tt)
    monkeypatch.setitem(sys.modules, "ttnn.decorators", dec)
    monkeypatch.setattr(profiler.time, "sleep", lambda s: None)
    return st, FastOperation


class Mesh:
    def get_device_ids(self):
        return [10, 11]


def deliver(st, ns_by_id):
    """One record per launched program per chip: ``ns_by_id[id] = (chip-10 ns, chip-11 ns)``, back to back."""
    t = {10: 0.0, 11: 0.0}
    recs = []
    for i in st.launched:
        for d, ns in zip((10, 11), ns_by_id[i]):
            recs.append(Rec(i, d, t[d], t[d] + ns))
            t[d] += ns + 5.0
    st.cb(types.SimpleNamespace(records=recs, dropped=0))


def test_sections_are_id_ranges_charged_after_the_run_without_syncs(monkeypatch):
    st, Op = fake_ttnn(monkeypatch)
    profiler.enable(Mesh(), rt=True)
    try:
        assert profiler.backend() == "rt"
        profiler.set_layer(0)
        profiler.signpost("attention")
        Op("ttnn.matmul", 2)()  # ids 100, 101
        profiler.signpost("experts")
        Op("ttnn.add", 1)()  # id 102
        profiler.set_layer(1)
        profiler.signpost("attention")
        Op("ttnn.matmul", 1)()  # id 103
        profiler.signpost("end")
        Op("ttnn.add", 1)()  # id 104: after the last signpost, charged nowhere
        syncs = st.syncs
        deliver(st, {100: (10, 12), 101: (20, 18), 102: (7, 9), 103: (40, 30), 104: (99, 99)})
        profiler.finish_rt()
        res = profiler.result()
    finally:
        profiler.disable()
    assert syncs == 1, "only enable() syncs before the run; signposts and ops never do"
    assert res["kernel_ns"] == {"attention": 30.0 + 40.0, "experts": 9.0}  # per range: the slowest chip
    assert res["kernel_ns_dev"]["attention"] == {0: 70.0, 1: 60.0}
    assert res["programs"] == {"attention": 6, "experts": 2}  # programs x chips, as the device profiler counts


def test_op_mode_rows_other_and_timeline_from_one_run(monkeypatch):
    st, Op = fake_ttnn(monkeypatch)
    profiler.enable(Mesh(), rt=True, calls=True)
    try:
        profiler.set_layer(3)
        profiler.signpost("attention")
        mm = Op("ttnn.matmul", 1)
        mm()  # 100
        mm()  # 101: back to back, same op and shape: one row
        st.launched.append(st.next_id)  # 102: a program no wrapped op launched
        st.next_id += 1
        Op("ttnn.add", 1)()  # 103
        profiler.signpost("end")
        syncs = st.syncs
        deliver(st, {100: (10, 20), 101: (10, 20), 102: (3, 3), 103: (5, 1)})
        calls = profiler.finish_rt()
        res = profiler.result()
    finally:
        profiler.disable()
    assert syncs == 1
    rows = res["op_ns"]["L3.attention"]
    assert [(r["op"], r["calls"], r["programs"]) for r in rows] == [("matmul", 2, 4), ("(other)", 1, 2), ("add", 1, 2)]
    assert rows[0]["ns"] == 40.0 and rows[0]["ns_dev"] == {0: 20.0, 1: 40.0}
    assert len(calls) == 3 and all(c["programs"] == 2 for c in calls)
    tl = res["timeline_rt"]
    assert tl["critical_chip"] == 1 and tl["calls"] == 3
    # chip 1: programs at [0,20] [25,45] [50,53] [58,59]: the add waited 5 behind the unwrapped program
    assert calls[2]["gap_ns"] == 5.0 and calls[2]["slot_ns"] == 6.0 and calls[1]["gap_ns"] == 5.0
    from models.demos.common.bringup.testing.profile import rt_ops

    ops = rt_ops(res, [3])
    assert set(ops) == {"L3.attention"} and {"gap_ms", "slot_ms", "host_ms", "timeline_kernel_ms"} <= set(
        ops["L3.attention"][0]
    )
    assert rt_ops(res, [4]) == {}


def test_inactive_realtime_profiler_fails_without_fallback(monkeypatch):
    fake_ttnn(monkeypatch, active=False)
    with pytest.raises(RuntimeError, match="does not fall back"):
        profiler.enable(Mesh(), rt=True)
    profiler.disable()


def test_dropped_records_fail(monkeypatch):
    st, Op = fake_ttnn(monkeypatch)
    profiler.enable(Mesh(), rt=True)
    try:
        profiler.signpost("attention")
        Op("ttnn.add", 1)()
        profiler.signpost("end")
        st.cb(types.SimpleNamespace(records=[], dropped=3))
        with pytest.raises(RuntimeError, match="dropped 3"):
            profiler.finish_rt()
    finally:
        profiler.disable()


def test_backend_switch(monkeypatch):
    class S:
        def __init__(self, v):
            self.v = v

        def get(self, k, d=None):
            return self.v if k == "perf.profiler" else d

    monkeypatch.delenv("BRINGUP_PROFILER", raising=False)
    assert (
        profiler.backend() == "rt" and profiler.backend(S(None)) == "rt" and profiler.backend(S("device")) == "device"
    )
    monkeypatch.setenv("BRINGUP_PROFILER", "device")
    assert profiler.backend(S("rt")) == "device"
    monkeypatch.setenv("BRINGUP_PROFILER", "tracy")
    with pytest.raises(ValueError):
        profiler.backend()
    monkeypatch.delenv("BRINGUP_PROFILER")
    for k in profiler.PROFILER_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="TT_METAL_DEVICE_PROFILER=1"):
        profiler.require_device_profiler()


def test_ledger_profile_env_follows_the_switch():
    from models.demos.common.bringup.core.gate import with_clean_wall
    from models.demos.common.bringup.plan.ledger_gen import DEVICE_PROFILE_ENV, profile_env

    class S:
        def __init__(self, v):
            self.v = v

        def get(self, k, d=None):
            return (self.v or "rt") if k == "perf.profiler" else d

    assert profile_env(S(None)) == "" and profile_env(S("device")) == f"{DEVICE_PROFILE_ENV} "
    prof = "scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py"
    old = "TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 " + prof  # a ledger written before F155
    wall, run = (x.strip() for x in with_clean_wall(old).split("&&"))
    assert wall == f"BRINGUP_PROFILE_WALL=1 {prof}" and run == f"BRINGUP_WALL_CLEAN=1 {prof}"
    wall, run = (x.strip() for x in with_clean_wall(prof).split("&&"))
    assert wall == f"BRINGUP_PROFILE_WALL=1 {prof}" and run == f"BRINGUP_WALL_CLEAN=1 {prof}"
    dev = f"{DEVICE_PROFILE_ENV} {prof}"
    wall, run = (x.strip() for x in with_clean_wall(dev).split("&&"))
    assert "TT_METAL_" not in wall and "BRINGUP_PROFILER" not in wall and run == f"BRINGUP_WALL_CLEAN=1 {dev}"
    assert with_clean_wall(with_clean_wall(prof)) == with_clean_wall(prof)
