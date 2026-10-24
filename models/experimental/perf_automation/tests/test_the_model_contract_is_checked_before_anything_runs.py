# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A model states it can be measured the tool's way, and that is checked by READING it.

Every requirement in agent/model_contract.py already existed as prose, in the module that consumes
it -- perf_adapter's docstring describes the per-stage hooks, emit_e2e's prompt specifies
build_pipeline's signature. Prose is checked by whoever happens to read it, which is how a model
reaches the device missing a clause and the tool finds out forty minutes later.

THE CASE. gemma-3's prefill decides its own traced-vs-eager from an allow-list inside the model,
while decode is controlled by the harness. Before profiling the tool asks for eager -- the profiler
attributes per-op device time from eager dispatch, a traced region emits none, and synchronising
inside a capture is fatal. Prefill traced anyway: 194 x "Event Synchronization is not supported
during trace capture", no profiling data, no baseline, after minutes of device time. Nothing in that
failure names its cause.

The clause is visible in the source. That is what this checks, and why it runs first.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

from agent.model_contract import CLAUSES, Finding, check, report  # noqa: E402


def _model(tmp_path, **files):
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body))
    return tmp_path


# A compliant model STATES what each stage retires. The item count is the only input to the stage's
# arithmetic ceiling (2 x params x items), and a stage that states nothing is priced at one item --
# right for a recurring step, ~1500x wrong for an encoder over 1500 frames, and indistinguishable
# from a real answer downstream. Note the recurring stage declares 1 EXPLICITLY: "one item" has to
# be a statement, or it cannot be told apart from "nobody said".
_GOOD_PIPE = """
    PIPELINE_STAGES = ["prefill", "decode"]

    def build_pipeline(device, model=None, layers=None, **kwargs):
        return object()

    def prefill_trace_setup(inputs): ...
    def prefill_trace_step(): ...
    def prefill_trace_items(): return 1024
    def decode_trace_setup(inputs): ...
    def decode_trace_step(): ...
    def decode_trace_items(): return 1
    def decode_prefill(inputs): ...
    def decode_step(): ...

    def trace_capture_selftest(device):
        return True

    def host_op_selftest():
        return True

    def can_enable_trace(seq_len, cached=0):
        import os
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            return False
        return seq_len in (128, 1024)

    LAYERS = int(__import__("os").environ.get("TT_PERF_LAYERS", "0") or 0)
"""


# ---------------------------------------------------------------- the clause that cost a run


def test_a_model_side_trace_gate_that_ignores_the_harness_is_blocking(tmp_path):
    """THE gemma-3 CASE. The gate consults its own allow-list and nothing else, so no harness
    signal can reach it -- and 'profiler on' plus 'traced' cannot both be true."""
    root = _model(
        tmp_path,
        **{
            "m.py": """
            PIPELINE_STAGES = ["decode"]
            def build_pipeline(device, model=None, layers=None, **kwargs): ...
            def decode_trace_setup(i): ...
            def decode_trace_step(): ...
            def can_enable_trace(seq_len, cached=0):
                return seq_len in self.trace_prefill_supported_seq_lens
            """
        },
    )
    f = [x for x in check(root) if x.clause == "trace-authority"]
    assert f and f[0].severity == "error", check(root)
    assert "TT_METAL_DEVICE_PROFILER" in f[0].remedy


def test_a_gate_that_consults_the_harness_passes(tmp_path):
    root = _model(tmp_path, **{"m.py": _GOOD_PIPE})
    assert [x for x in check(root) if x.clause == "trace-authority"] == []


def test_no_model_side_gate_at_all_is_fine(tmp_path):
    """If the model never decides, the harness is the only authority -- which is the goal."""
    root = _model(
        tmp_path,
        **{
            "m.py": """
            PIPELINE_STAGES = ["decode"]
            def build_pipeline(device, model=None, layers=None, **kwargs): ...
            def decode_trace_setup(i): ...
            def decode_trace_step(): ...
            LAYERS = __import__("os").environ.get("TT_PERF_LAYERS")
            """
        },
    )
    assert [x for x in check(root) if x.clause == "trace-authority"] == []


# ---------------------------------------------------------------- the structural clauses


def test_a_missing_build_pipeline_is_blocking(tmp_path):
    root = _model(tmp_path, **{"m.py": "PIPELINE_STAGES = ['decode']\n"})
    f = [x for x in check(root) if x.clause == "build-pipeline"]
    assert f and f[0].severity == "error"


def test_the_device_must_be_the_first_parameter(tmp_path):
    root = _model(tmp_path, **{"m.py": "def build_pipeline(model, device=None, **kw): ...\n"})
    f = [x for x in check(root) if x.clause == "build-pipeline"]
    assert any("first parameter" in x.detail for x in f), f


def test_layers_defaulting_to_zero_is_blocking(tmp_path):
    """0 arrives truthy from an env var and has been read as 'build zero layers' -- which measures
    nothing and reports no markers, while looking like a successful run."""
    root = _model(tmp_path, **{"m.py": "def build_pipeline(device, model=None, layers=0, **kw): ...\n"})
    f = [x for x in check(root) if x.clause == "build-pipeline"]
    assert any("layers defaults to 0" in x.detail for x in f), f


def test_a_declared_stage_without_its_hooks_is_blocking(tmp_path):
    root = _model(
        tmp_path,
        **{
            "m.py": """
            PIPELINE_STAGES = ["prefill", "decode"]
            def build_pipeline(device, model=None, layers=None, **kwargs): ...
            def decode_trace_setup(i): ...
            def decode_trace_step(): ...
            """
        },
    )
    f = [x for x in check(root) if x.clause == "stages"]
    assert any("prefill_trace_setup" in x.detail for x in f), f


def test_a_fully_compliant_model_reports_nothing(tmp_path):
    root = _model(tmp_path, **{"m.py": _GOOD_PIPE})
    assert check(root) == [], check(root)
    assert "meets all" in report([], root)


# ---------------------------------------------------------------- it must not take the run down


def test_a_file_that_does_not_parse_is_reported_not_skipped(tmp_path):
    """A clause that cannot be checked has not been met; silence would read as compliance."""
    root = _model(tmp_path, **{"m.py": _GOOD_PIPE, "broken.py": "def (((\n"})
    assert any(x.clause == "sources" for x in check(root)), check(root)


def test_the_check_never_raises(tmp_path):
    """A contract check that takes the run down with it is worse than the gap it looks for."""
    assert check(tmp_path / "does-not-exist") is not None
    assert isinstance(check(tmp_path), list)


def test_the_tools_own_generated_test_is_not_evidence(tmp_path):
    """Judging the model by a file the TOOL wrote would make the contract self-satisfying."""
    root = _model(
        tmp_path,
        **{
            "tests/e2e/test_main_perf.py": _GOOD_PIPE,  # the tool's output
            "m.py": "X = 1\n",
        },
    )
    assert any(x.clause == "build-pipeline" for x in check(root)), "the generated test satisfied a clause"


# ---------------------------------------------------------------- the remedy is the point


def test_every_finding_carries_an_actionable_remedy():
    """The remedy IS the porting task. A finding without one just relocates the confusion."""
    for f in check(Path(__file__).resolve().parent):  # this test dir: not a model, so it fails clauses
        assert f.remedy and len(f.remedy) > 20, f
        assert f.clause in {c for c, _ in CLAUSES}, f


def test_blocking_findings_sort_first():
    a = Finding("x", "d", "r", severity="warn")
    b = Finding("y", "d", "r")
    assert sorted([a, b], key=lambda f: 0 if f.severity == "error" else 1)[0] is b


# ---------------------------------------------------------------- compatibility vs porting


def test_missing_emit_e2e_shape_never_blocks_a_direct_optimize_model(tmp_path):
    """OPTIMIZE IS ALSO RUN DIRECTLY ON HAND-WRITTEN MODELS. A model EMITTED by emit-e2e satisfies
    PIPELINE_STAGES, the per-stage hooks and the self-tests by construction -- they are its output.
    gemma-3 and llama3_1_8b_p150 never went through it and legitimately lack that shape. Refusing
    them for not resembling emit-e2e's output would refuse the entire direct path.

    So the porting clauses are reported and stepped over; they are the porting TASK, not a defect."""
    root = _model(
        tmp_path, **{"m.py": "def build_pipeline(device, model=None, layers=None, **kw):\n    return object()\n"}
    )
    f = check(root)
    assert f, "a bare model should still report the porting gaps"
    assert all(not x.blocking for x in f), [str(x) for x in f]
    assert any(x.kind == "porting" for x in f)


def test_fighting_the_harness_blocks_however_the_model_was_written(tmp_path):
    """The other half of the rule. Not looking like emit-e2e's output is fine; a trace gate the
    harness cannot reach is not -- that is what produced 194 fatals and no baseline."""
    root = _model(
        tmp_path,
        **{
            "m.py": """
            def build_pipeline(device, model=None, layers=None, **kw):
                return object()
            def can_enable_trace(seq_len, cached=0):
                return seq_len in self.trace_prefill_supported_seq_lens
            """
        },
    )
    blk = [x for x in check(root) if x.blocking]
    assert [x.clause for x in blk] == ["trace-authority"], [str(x) for x in check(root)]
    assert blk[0].kind == "compatibility"


def test_a_factory_that_runs_the_model_blocks(tmp_path):
    """A one-shot result exposes none of the hooks, so the trace engine skips the model entirely --
    while appearing to succeed."""
    root = _model(
        tmp_path,
        **{"m.py": "def build_pipeline(device, model=None, layers=None, **kw):\n    return model.generate()\n"},
    )
    assert any(x.blocking and "not the pipeline" in x.detail for x in check(root)), [str(x) for x in check(root)]
