"""mark_stages read an empty stage list off an adapter nobody had built.

PipelineStageAdapter.__init__ sets `self.stages = []`; setup() is what constructs the pipeline and
binds one _Stage per declared stage. measure_adapter calls setup() before touching them. mark_stages
did not -- it read the empty list straight off the freshly constructed adapter the injected block
hands it, and returned 0 without asking the pipeline anything.

Measured on voxtral run 25's baseline capture: 2 signposts where 8 were expected -- the start/stop
bracket fired, every stage:<name> / stage:<name>:end boundary was missing -- so stage_windows found
no windows, stage_buckets came back {}, and the roofline shared ONE math-fidelity peak (702 TFLOPS,
LoFi) across encode, prefill and decode. Encode and prefill were priced at a peak 4x their own.

And the zero was silent. "Not asked", "asked and refused" and "ran and failed" all returned a bare
0, which is why this looked identical to eight earlier failures of the stage axis."""
import sys
import types

import pytest


class _Stage:
    def __init__(self, name, step):
        self.name, self.step = name, step


class _Adapter:
    """Mirrors the real contract: stages are EMPTY until setup(device) is called."""

    def __init__(self, names, fail_setup=False, fail_step=None):
        self._names, self._fail_setup, self._fail_step = names, fail_setup, fail_step
        self.stages = []
        self.setup_calls = 0

    def setup(self, device):
        self.setup_calls += 1
        if self._fail_setup:
            raise RuntimeError("pipeline would not build")
        self.stages = [_Stage(n, self._make(n)) for n in self._names]

    def _make(self, n):
        def _step():
            if self._fail_step == n:
                raise RuntimeError("cannot run alone")

        return _step


@pytest.fixture()
def sm(monkeypatch):
    """stage_marks imports ttnn inside mark_stages; stub it so no device is needed."""
    stub = types.ModuleType("ttnn")
    stub.synchronize_device = lambda d: None
    monkeypatch.setitem(sys.modules, "ttnn", stub)
    from agent import stage_marks as _sm

    emitted = []
    monkeypatch.setattr(_sm, "signpost", lambda name: emitted.append(name))
    return _sm, emitted


def test_it_sets_the_adapter_up_before_reading_its_stages(sm):
    """The regression: a fresh adapter has stages == [], and returning 0 there asks nobody."""
    m, emitted = sm
    a = _Adapter(["encode", "prefill", "decode"])
    n = m.mark_stages(a, device=object())
    assert a.setup_calls == 1, "mark_stages never asked the adapter to build its stages"
    assert n == 3
    assert emitted == [
        "stage:encode",
        "stage:encode:end",
        "stage:prefill",
        "stage:prefill:end",
        "stage:decode",
        "stage:decode:end",
    ]


def test_an_already_built_adapter_is_not_set_up_twice(sm):
    m, _ = sm
    a = _Adapter(["decode"])
    a.setup(object())
    a.setup_calls = 0
    assert m.mark_stages(a, device=object()) == 1
    assert a.setup_calls == 0, "an adapter that already has stages was rebuilt"


def test_a_setup_that_fails_says_so_instead_of_returning_a_bare_zero(sm, capsys):
    m, _ = sm
    a = _Adapter(["decode"], fail_setup=True)
    assert m.mark_stages(a, device=object()) == 0
    err = capsys.readouterr().err
    assert "NO per-stage boundaries" in err and "adapter.setup failed" in err


def test_a_pipeline_with_nothing_to_declare_says_so_too(sm, capsys):
    m, _ = sm
    assert m.mark_stages(_Adapter([]), device=object()) == 0
    assert "declares no stages after setup" in capsys.readouterr().err


def test_a_stage_that_will_not_run_alone_still_leaves_its_boundary(sm):
    """The window must close even when the step raises -- the :end is in a finally."""
    m, emitted = sm
    a = _Adapter(["encode", "decode"], fail_step="encode")
    assert m.mark_stages(a, device=object()) == 1
    assert "stage:encode" in emitted and "stage:encode:end" in emitted


def test_the_names_are_what_stage_windows_looks_for(sm):
    """stage_windows pairs 'stage:<name>' with 'stage:<name>:end'; anything else is not a window."""
    m, emitted = sm
    m.mark_stages(_Adapter(["encode"]), device=object())
    starts = [n for n in emitted if n.startswith("stage:") and not n.endswith(":end")]
    assert starts == ["stage:encode"]
    assert ["%s:end" % s for s in starts] == [n for n in emitted if n.endswith(":end")]
