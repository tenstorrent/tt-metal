# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The token window bounds a recurring stage however the pipeline takes its horizon -- and a profile
that still outgrows the disk is stopped instead of filling it.

Mistral-Small-4-119B-2603, 2026-10-14: the tool set TT_PERF_OSL_TOKENS=2, the generated test handed it
to generate() -- which takes no such argument; the pipeline takes its horizon at construction -- and
the profiled decode ran toward its 1024-token horizon until a 97 GB device log filled the root disk.
The window is now counted on the marked stage method itself, and the run is watched for free space.

Stage names are invented on purpose: everything here reads them from the model's own PIPELINE_STAGES.
"""

import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from agent import probes
from agent import stage_marks as sm
from agent import stage_seams as _seams
from agent.layer_depth import TOKENS_ENV

_HORIZON = 50  # the pipeline's own horizon, far past any window below
_SRC_DIR = Path(tempfile.mkdtemp(prefix="window_pipes_"))


def _load_pipe(src):
    """Write `src` as a module and return its Pipe: the marks read hooks' calls from real source."""
    path = _SRC_DIR / ("pipe_%d.py" % len(list(_SRC_DIR.glob("pipe_*.py"))))
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod.Pipe


def _pipe_class(stages, recurring, repeats=_HORIZON, extra_own=None):
    """A pipeline declaring `stages`; `recurring` loops its own horizon and states it via REPEATS.

    Written as source because the marks read each step hook's own calls from its source."""
    src = "class Pipe:\n    PIPELINE_STAGES = %r\n\n    def __init__(self):\n" % list(stages)
    src += "        self.calls = {s: 0 for s in self.PIPELINE_STAGES}\n"
    for s in stages:
        src += "\n    def run_%s(self):\n        self.calls[%r] += 1\n" % (s, s)
        src += "\n    def %s(self, inputs=None):\n        pass\n" % _seams.hook(s, _seams.SETUP)
        helper = "self.%s()\n        " % extra_own if extra_own and s == recurring else ""
        src += "\n    def %s(self):\n        %sreturn self.run_%s()\n" % (_seams.hook(s, _seams.STEP), helper, s)
    src += "\n    def %s(self):\n        return %d\n" % (_seams.hook(recurring, _seams.REPEATS), repeats)
    if extra_own:
        src += "\n    def %s(self):\n        pass\n" % extra_own
    src += "\n    def generate(self):\n"
    for s in stages:
        pre = "self.%s()\n            " % extra_own if extra_own and s == recurring else ""
        src += "        for _ in range(%d):\n            %sself.run_%s()\n" % (
            _HORIZON if s == recurring else 1,
            pre,
            s,
        )
    src += "        return self.calls\n"
    return _load_pipe(src)


@pytest.fixture(autouse=True)
def profiled(monkeypatch):
    """Every test here is a profiled run unless it says otherwise."""
    for k, v in sm._profiling_env().items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def no_signposts(monkeypatch):
    monkeypatch.setattr(sm, "signpost", lambda name: None)


def _profiled_generate(pipe):
    """What the injected bracket does around the eager call: a closed window ends it, quietly."""
    sm.mark_stages_on_forward(pipe)
    sm.bound_profile_window(pipe)
    try:
        pipe.generate()
    except Exception as exc:  # noqa: BLE001
        if not sm.window_reached(exc):
            raise
    return pipe.calls


def test_the_window_bounds_the_last_recurring_stage(monkeypatch, no_signposts):
    monkeypatch.setenv(TOKENS_ENV, "3")
    calls = _profiled_generate(_pipe_class(("alpha", "beta"), "beta")())
    assert calls == {"alpha": 1, "beta": 3}


def test_no_window_leaves_the_horizon_alone(monkeypatch, no_signposts):
    monkeypatch.delenv(TOKENS_ENV, raising=False)
    calls = _profiled_generate(_pipe_class(("alpha", "beta"), "beta")())
    assert calls == {"alpha": 1, "beta": _HORIZON}


def test_a_recurring_stage_with_another_after_it_is_not_cut(monkeypatch, no_signposts):
    """Ending the forward early would leave the later stage unprofiled."""
    monkeypatch.setenv(TOKENS_ENV, "3")
    calls = _profiled_generate(_pipe_class(("alpha", "beta"), "alpha")())
    assert calls == {"alpha": _HORIZON, "beta": 1}


def test_a_stage_that_fits_its_window_is_not_counted(monkeypatch, no_signposts):
    monkeypatch.setenv(TOKENS_ENV, "3")
    pipe = _pipe_class(("alpha", "beta"), "beta", repeats=3)()
    assert sm.bound_profile_window(pipe) == 0


def test_a_step_with_several_own_methods_is_still_bounded(monkeypatch, no_signposts):
    """The stage cannot be matched to ONE method for marks, so its marks take the separate pass --
    the window must hold anyway, whichever of its methods the forward reaches first."""
    cls = _pipe_class(("alpha", "beta"), "beta", extra_own="helper_beta")
    pipe = cls()
    assert sm.stage_methods(pipe) == {}
    monkeypatch.setenv(TOKENS_ENV, "3")
    assert sm.bound_profile_window(pipe) == 3
    try:
        pipe.generate()
    except sm.ProfileWindowReached:
        pass
    assert pipe.calls["beta"] == 3


def test_the_separate_marked_pass_does_not_spend_the_window(monkeypatch, no_signposts):
    monkeypatch.setenv(TOKENS_ENV, "2")
    pipe = _pipe_class(("alpha", "beta"), "beta", extra_own="helper_beta")()
    monkeypatch.setattr(sm, "mark_stages_for", lambda p, d: [p.beta_trace_step() for _ in range(2)] and 2)
    assert sm.mark_stages_in_scope({"pipe": pipe}) == 2
    pipe.calls = {"alpha": 0, "beta": 0}
    try:
        pipe.generate()
    except sm.ProfileWindowReached:
        pass
    assert pipe.calls["beta"] == 2


def test_outside_a_profiled_run_there_is_no_window(monkeypatch, no_signposts):
    """The full-pipeline gate sets the token env too, and its eager fallback has no bracket to catch."""
    monkeypatch.setenv(TOKENS_ENV, "3")
    for k in sm._profiling_env():
        monkeypatch.delenv(k)
    calls = _profiled_generate(_pipe_class(("alpha", "beta"), "beta")())
    assert calls == {"alpha": 1, "beta": _HORIZON}


def test_an_unusable_window_is_no_window(monkeypatch):
    for v in ("", "0", "-4", "two"):
        monkeypatch.setenv(TOKENS_ENV, v)
        assert sm._profile_window() == 0


def test_each_pipeline_counts_its_own_passes(monkeypatch, no_signposts):
    monkeypatch.setenv(TOKENS_ENV, "2")
    cls = _pipe_class(("alpha", "beta"), "beta")
    assert _profiled_generate(cls())["beta"] == 2
    assert _profiled_generate(cls())["beta"] == 2


def _horizon_pipe_class(stages, recurring, draws):
    """A pipeline that takes its horizon at construction, states it from that attribute, and -- as a
    sampling loop does -- reads one per-pass input BEFORE each pass of `recurring`. `draws` is sized
    the way the generated test sizes them: to the window, not to the horizon."""
    src = "class Pipe:\n    PIPELINE_STAGES = %r\n\n    def __init__(self, horizon):\n" % list(stages)
    src += "        self.horizon = horizon\n        self.draws = list(range(%d))\n" % draws
    src += "        self.calls = {s: 0 for s in self.PIPELINE_STAGES}\n"
    for s in stages:
        src += "\n    def run_%s(self):\n        self.calls[%r] += 1\n" % (s, s)
        src += "\n    def %s(self):\n        return self.run_%s()\n" % (_seams.hook(s, _seams.STEP), s)
    src += '\n    def %s(self):\n        """stated."""\n        return self.horizon\n' % _seams.hook(
        recurring, _seams.REPEATS
    )
    src += "\n    def generate(self):\n"
    for s in stages:
        if s == recurring:
            src += "        for t in range(self.horizon):\n            self.draws[t]\n            self.run_%s()\n" % s
        else:
            src += "        self.run_%s()\n" % s
    src += "        return self.calls\n"
    return _load_pipe(src)


def test_a_horizon_taken_at_construction_is_lowered_before_the_loop_outruns_its_inputs(monkeypatch, no_signposts):
    """Counting passes alone let the loop read draw #3 of 2 before the third pass was refused."""
    monkeypatch.setenv(TOKENS_ENV, "2")
    pipe = _horizon_pipe_class(("alpha", "beta"), "beta", draws=2)(_HORIZON)
    assert _profiled_generate(pipe) == {"alpha": 1, "beta": 2}
    assert pipe.horizon == 2


def test_a_horizon_is_never_raised_or_touched_without_a_window(monkeypatch, no_signposts):
    monkeypatch.delenv(TOKENS_ENV, raising=False)
    pipe = _horizon_pipe_class(("alpha", "beta"), "beta", draws=_HORIZON)(_HORIZON)
    assert _profiled_generate(pipe)["beta"] == _HORIZON and pipe.horizon == _HORIZON
    monkeypatch.setenv(TOKENS_ENV, "3")
    small = _horizon_pipe_class(("alpha", "beta"), "beta", draws=2)(2)
    assert sm.bound_profile_window(small) == 0 and small.horizon == 2


def test_only_a_bare_attribute_read_names_the_horizon():
    """A computed count is not an attribute the window may write; the pass count still holds it."""
    cls = _pipe_class(("alpha", "beta"), "beta")
    assert sm._attr_a_hook_returns(getattr(cls, _seams.hook("beta", _seams.REPEATS))) is None
    hcls = _horizon_pipe_class(("alpha", "beta"), "beta", draws=1)
    assert sm._attr_a_hook_returns(getattr(hcls, _seams.hook("beta", _seams.REPEATS))) == "horizon"
    assert sm._attr_a_hook_returns(len) is None


# --- the bracket the injector writes around the eager call ---------------------------------------

_TEST = (
    "import os\n"
    "def test_x_perf(device):\n"
    "    def _eager_forward():\n"
    "        pipe = build_pipeline(device)\n"
    "        pipe.generate()\n"
    "    if os.environ.get('TT_METAL_DEVICE_PROFILER') == '1':\n"
    "        _eager_forward()\n"
)


def _run_guard(raised):
    def _eager_forward():
        raise raised

    ns = {"_tt_sm": sm, "_eager_forward": _eager_forward}
    exec(sm._guarded_call("", "_eager_forward()\n"), ns)


def test_the_bracket_ends_quietly_on_the_window_and_raises_anything_else():
    _run_guard(sm.ProfileWindowReached("beta: 3 of 3 passes"))
    with pytest.raises(RuntimeError, match="a real failure"):  # allow-pytest.raises: no expect_error fixture
        _run_guard(RuntimeError("a real failure"))


def test_a_fresh_injection_is_guarded_and_still_reaches_the_eager_call():
    out, why = sm.inject_stage_marks(_TEST)
    assert sm.marks_ok(why), why
    assert sm._WINDOW_GUARD_KEY in out
    compile(out, "t.py", "exec")
    assert [c for _l, _i, c in sm.reachable_bare_calls(out)] == ["_eager_forward"]
    assert sm.inject_stage_marks(out) == (out, "already injected")


def test_a_bracket_from_before_the_guard_is_given_it_in_place():
    fresh, _ = sm.inject_stage_marks(_TEST)
    lines = fresh.splitlines(keepends=True)
    at = next(j for j, ln in enumerate(lines) if ln.strip() == sm._BRACKET_BEFORE) + 1
    guard_len = len(sm._guarded_call("", "_eager_forward()").splitlines())
    ind = lines[at][: len(lines[at]) - len(lines[at].lstrip())]
    old = "".join(lines[:at] + [ind + "_eager_forward()\n"] + lines[at + guard_len :])
    assert sm._WINDOW_GUARD_KEY not in old
    out, why = sm.inject_stage_marks(old)
    assert why == sm._REFRESHED and sm.marks_ok(why), why
    assert out == fresh
    assert sm.inject_stage_marks(out) == (out, "already injected")


def test_a_reshaped_bracket_is_left_alone():
    fresh, _ = sm.inject_stage_marks(_TEST)
    hand = fresh.replace(sm._guarded_call("        ", "_eager_forward()"), "        _eager_forward(1)\n")
    assert hand != fresh
    assert sm._refresh_bracket(hand) == (hand, False)


# --- the disk a profile writes to -------------------------------------------------------------


def test_the_floor_is_half_of_a_disk_that_starts_low(monkeypatch, tmp_path):
    monkeypatch.setattr(probes, "_MIN_FREE_DISK_GB", 10.0)
    monkeypatch.setattr(probes, "_free_disk_gb", lambda paths: {p: 4.0 for p in paths})
    assert probes.disk_floor([tmp_path]) == 2.0
    monkeypatch.setattr(probes, "_free_disk_gb", lambda paths: {p: 500.0 for p in paths})
    assert probes.disk_floor([tmp_path]) == 10.0


def test_an_unmeasurable_path_never_fails_a_run(tmp_path):
    assert probes.disk_floor_breached([tmp_path / "missing" / "dir"], 1e12) == ""
    assert probes.disk_floor_breached([tmp_path], 1e12) != ""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_a_run_eating_the_disk_is_stopped_and_not_retried_as_a_hang(monkeypatch, tmp_path):
    polls = iter([500.0])  # at launch; every later poll reads the disk nearly gone

    def _free(paths):
        gb = next(polls, 0.5)
        return {p: gb for p in paths}

    monkeypatch.setattr(probes, "_free_disk_gb", _free)
    script = tmp_path / "run.py"
    script.write_text("import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(300)\n")
    log = tmp_path / "run.log"
    t0 = time.monotonic()
    full = probes.DiskFullError
    with pytest.raises(full, match="filled the disk") as err:  # allow-pytest.raises: no expect_error fixture
        probes._execute([sys.executable, str(script)], tmp_path, dict(os.environ), 120, log, stall_timeout_s=60)
    assert not isinstance(err.value, probes.TracyHangError)
    assert time.monotonic() - t0 < 60
    pid = int(log.read_text().split()[0])
    assert not _alive(pid)
