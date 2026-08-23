# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Five attempts at per-stage fidelity were written into the run that cannot see fidelity.

THE TOOL MEASURES TWICE:

    A  tracy eager profile   PROFILER=1, TT_PERF_TRACE=0, coverage depth
       per-op records: op_code, shape, FIDELITY, cores, grid, memory, bytes, device_ms
    B  full-depth stopwatch  profiler popped, all layers, trace+1cq
       TRACE_STAGE_MS / TRACE_STAGE_BYTES -- times and totals, no per-op anything

A traced replay runs as one fused program and emits NO per-op device data. So fidelity exists only
in A, and every one of the five attempts -- the section map, the bucket reader, the byte hook, the
signposts, the profiled branch -- was written into trace_replay, which is B. B never produces a
fidelity field, so none of them could ever have worked, and each "fix" only revealed the next gate:

    section map -> bucket tag -> self-traced branch -> test guard -> TT_PERF_TRACE=0

All five were the same fact seen from five angles: measure_adapter is B's code.

SO THE MARKS GO IN A, and A's measured call is `pipe.run_head(...)` -- one opaque call. This adds a
SECOND, MARKED pass after it rather than replacing it: 94 of the 112 ops in a real capture carry no
parseable shape, so it cannot be shown by inspection that per-stage steps reach the same op set, and
an op only run_head touches would vanish from the ladder's view. The measured region keeps its exact
ops, bracketed by the conventional start/stop pair that resolve_signposts and refine() already use;
the marked pass is additive and feeds fidelity rollup only.
"""
import sys
import types
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


@pytest.fixture()
def marks(monkeypatch):
    tt = types.ModuleType("ttnn")
    tt.synchronize_device = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "ttnn", tt)
    seen = []
    tr = types.ModuleType("tracy")
    tr.signpost = lambda n: seen.append(n)
    monkeypatch.setitem(sys.modules, "tracy", tr)
    return seen


class _St:
    def __init__(self, n):
        self.name = n
        self.ran = False

    def step(self):
        self.ran = True


def test_every_declared_stage_is_bracketed(marks):
    from agent.stage_marks import mark_stages

    class _A:
        stages = [_St("encode"), _St("prefill"), _St("decode")]

    assert mark_stages(_A(), object()) == 3
    assert marks == [
        "stage:encode",
        "stage:encode:end",
        "stage:prefill",
        "stage:prefill:end",
        "stage:decode",
        "stage:decode:end",
    ]


def test_the_names_come_from_the_model_not_from_a_list(marks):
    """A fixed {encode,prefill,decode} could not mark an audio tower, a vocoder or a denoiser."""
    from agent.stage_marks import mark_stages

    class _A:
        stages = [_St("denoise"), _St("vocode")]

    assert mark_stages(_A(), object()) == 2
    assert marks == ["stage:denoise", "stage:denoise:end", "stage:vocode", "stage:vocode:end"]


def test_a_stage_that_cannot_run_alone_costs_only_its_own_boundary(marks):
    """Its window is still closed -- an unterminated one would be skipped by stage_windows anyway --
    and it comes back empty, which build_buckets drops."""
    from agent.stage_marks import mark_stages

    class _Bad(_St):
        def step(self):
            raise RuntimeError("needs prior state")

    class _A:
        stages = [_St("encode"), _Bad("prefill"), _St("decode")]

    assert mark_stages(_A(), object()) == 2
    assert "stage:prefill" in marks and "stage:prefill:end" in marks
    assert marks.count("stage:decode") == 1, "a failure stopped the stages after it"


def test_no_declared_stages_means_no_marks(marks):
    from agent.stage_marks import mark_stages

    class _A:
        stages = []

    assert mark_stages(_A(), object()) == 0
    assert marks == []


def test_it_never_opens_a_capture():
    """Synchronising inside a trace capture is fatal -- "Event Synchronization is not supported
    during trace capture" -- and the profiler attributes per-op time from EAGER dispatch anyway."""
    src = (_PA / "agent" / "stage_marks.py").read_text()
    assert "begin_trace_capture" not in src and "execute_trace" not in src


def test_the_marks_are_not_in_the_run_that_cannot_see_fidelity():
    """trace_replay is B. It emits times and totals; it has no fidelity to roll up, and it runs with
    the profiler popped, so a mark there marks nothing. Five attempts lived there."""
    tr = (_PA / "agent" / "trace_replay.py").read_text()
    assert "signpost" not in tr, "stage marks are back in the stopwatch path"
    assert "_measure_stage_profiled" not in tr


def test_the_measured_region_is_bracketed_so_the_marked_pass_is_not_counted_into_it():
    """The extra pass adds ops to the capture. Without start/stop around the measured region the
    main report would count them, and every per-op number would shift for a reason unrelated to the
    model."""
    src = (_PA / "agent" / "perf_test_gen.py").read_text()
    i = src.index('_sm.signpost("start")')
    j = src.index('_sm.signpost("stop")')
    assert "_eager_forward()" in src[i:j], "the measured call is not inside the start/stop pair"
    assert src.index("mark_stages(") > j, "the marked pass runs before the measured region closes"
