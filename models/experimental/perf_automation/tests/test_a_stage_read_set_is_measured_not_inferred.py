# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The roofline's memory floor divides by bytes nobody ever measured.

WHAT THE DIVISOR HAD TO COME FROM, before this. All three are inferences from the checkpoint:

    params x bytes_per_param        the whole model, including towers a token never reads
    _TOWER_ONLY name list           `audio_tower|vision_tower|...` -- a model can spell it otherwise
    stage_roots section map         tried, reverted: drops lm_head on any untied model (12.5%)

And the thing that looks like a measurement is not one. summary._bytes_for reads

    _b = <per-stage bytes off the profile> or _bytes_for(name, toks)

and that reader could never work: buckets carry no `bytes` key, and the regime tag it keyed on reads
"na" for every one of them -- three ways of returning 0, so the line always took its estimate branch
while appearing to prefer a measurement. It has since been deleted. _stage_roofs says why it could
not have worked: "_top_ops keys on (op_code, shape, memory) and records nothing about which phase an
op ran in".

WHERE THE MEASUREMENT WAS ALREADY SITTING. trace_replay runs each stage in isolation and, to decide
whether the step was traced or eager, wraps ttnn's Operation.__call__ to count dispatches -- so it
already sees every op a stage runs, with its arguments. The read set is those ops' distinct DEVICE
tensors. It rides the warmup call that was happening anyway: no second pass, no eager run.

    TRACE_STAGE_MS[decode]=16.6779     already emitted
    TRACE_STAGE_BYTES[decode]=...      now emitted beside it

DISTINCT BY IDENTITY, because the quantity it replaces (params x width) counts each weight once.
Host tensors excluded by asking the tensor, via the census's own _on_device -- voxtral keeps an fp32
host copy alive while the device holds bf16, and counting it reported 29.96 GB for an 11.3 GB chip.
"""
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


class _DevT:
    """A ttnn-shaped device tensor."""

    def __init__(self, numel, dt="BFLOAT16"):
        self.shape = (numel,)

        class _D:
            name = dt

        self.dtype = _D()

    def storage_type(self):
        return "DEVICE"


class _HostT:
    """Same duck type, no device -- voxtral's fp32 shadow copy."""

    def __init__(self, numel):
        self.shape = (numel,)
        self.dtype = "torch.float32"


def _hooked(fn):
    """Drive trace_replay's dispatch hook over a fake ttnn Operation."""
    import types

    dec = types.ModuleType("ttnn.decorators")

    class Operation:
        def __call__(self, *a, **kw):
            return None

    dec.Operation = Operation
    ttnn = types.ModuleType("ttnn")
    ttnn.decorators = dec
    sys.modules["ttnn"] = ttnn
    sys.modules["ttnn.decorators"] = dec
    # AS A PACKAGE MODULE. trace_replay reaches the census with a RELATIVE import, which raises when
    # the file is loaded by path -- and the byte hook catches that and disables itself, so every
    # count comes back 0 and the test measures the fallback instead of the feature.
    try:
        import agent.trace_replay as tr
    except Exception as exc:  # noqa: BLE001
        pytest.skip("trace_replay needs a real ttnn to import: %s" % exc)
    op = Operation()
    return tr._count_op_dispatches(lambda: fn(op))


def test_the_read_set_is_the_bytes_of_the_tensors_the_ops_touched():
    w = _DevT(1_000_000)
    n, b = _hooked(lambda op: op(w))
    assert n == 1
    assert b == 2_000_000, b  # bf16


def test_a_weight_used_twice_counts_once():
    """THE WORKING SET, not the traffic. params x width counts each weight once, so this must too or
    the two numbers are not comparable and the ceiling moves for the wrong reason."""
    w = _DevT(1_000_000)

    def run(op):
        op(w)
        op(w)

    n, b = _hooked(run)
    assert n == 2 and b == 2_000_000, (n, b)


def test_two_distinct_weights_both_count():
    a, c = _DevT(1_000_000), _DevT(500_000)
    _n, b = _hooked(lambda op: op(a, c))
    assert b == 3_000_000, b


def test_a_host_tensor_is_not_counted():
    """voxtral holds an fp32 host copy while the device holds bf16; counting it reported 29.96 GB for
    an 11.3 GB device footprint."""
    _n, b = _hooked(lambda op: op(_HostT(1_000_000)))
    assert b == 0, b


def test_kwargs_are_seen_too():
    w = _DevT(1_000_000)
    _n, b = _hooked(lambda op: op(x=w))
    assert b == 2_000_000, b


def test_a_tensor_that_raises_is_skipped_not_fatal():
    """Instrumentation riding a measurement may under-report; it may never take the run down."""

    class _Angry:
        shape = (8,)

        def storage_type(self):
            raise RuntimeError("nope")

    n, b = _hooked(lambda op: op(_Angry(), _DevT(1000)))
    assert n == 1 and b == 2000, (n, b)


def test_nothing_on_device_reports_zero_and_zero_is_a_refusal():
    """Zero must not become a ceiling of infinity: the consumer treats it as 'no measurement' and
    keeps the estimate."""
    _n, b = _hooked(lambda op: op(123, "x", None))
    assert b == 0


def _is_docstring(node):
    import ast as _a

    return isinstance(node, _a.Expr) and isinstance(node.value, _a.Constant) and isinstance(node.value.value, str)


def test_the_consumer_prefers_the_measurement_over_the_estimate():
    """Order of preference: the pinned baseline, then this build's reading, then the estimate.

    Asserted against stage_read_bytes, which is now the ONE place the question is answered. It used
    to be a chain inline in _stage_roofs, and that is precisely why a renderer outside that function
    reached for the model-level figure and printed a second answer in the same row.

    The pin comes first for the reason the floor is pinned: measuring the bytes made them right and
    did nothing about them moving, and the dtype rung halves them by construction.
    """
    import ast

    import cc_optimize.summary as S

    src = (_PA / "cc_optimize" / "summary.py").read_text()
    body = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "stage_read_bytes":
            # STATEMENTS ONLY -- not the signature, not the docstring. `measured=None` in the
            # signature and the word "measured" in the prose both sort before the code, so ordering
            # asserted over the whole unparse tests where the words appear, not what runs.
            stmts = node.body[1:] if _is_docstring(node.body[0]) else node.body
            body = "\n".join(ast.unparse(x) for x in stmts)
    assert body, "the owner is gone"
    for nxt in ("_pinned_stage_bytes", "measured", "estimate"):
        assert nxt in body, body
    assert body.index("_pinned_stage_bytes") < body.index("measured") < body.index("estimate")
    assert "_stage_measured_bytes" not in body, "the dead reader is back in the chain"
    assert callable(S.stage_read_bytes) and callable(S._measured_stage_bytes)


def test_the_read_set_is_pinned_so_a_dtype_win_cannot_move_the_ceiling(tmp_path, monkeypatch):
    """THE MISTAKE THIS ALMOST REPEATED. Measuring the bytes fixes accuracy, not stability: bf16 ->
    bf8_b halves a weight, the observed read set halves, and an unpinned ceiling follows the build
    down so the target is never reached. Same defect as the modelled floor, same fix."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    import cc_optimize.summary as S

    led = S._ledger()
    monkeypatch.setattr(
        led, "ledger_path", lambda model="", task="": tmp_path / ("%s_%s.jsonl" % (model or "m", task or "main"))
    )
    led.anchor(led.KIND_STAGE_BYTES, 4786.0, depth="decode", mode="bytes_mb", source="t", model="m")
    # a later, smaller reading must not move the pin
    led.anchor(led.KIND_STAGE_BYTES, 2393.0, depth="decode", mode="bytes_mb", source="t", model="m")
    assert S._pinned_stage_bytes("decode", "m", "main") == 4_786_000_000
    assert S._pinned_stage_bytes("prefill", "m", "main") is None  # keyed per stage


def test_the_marker_is_emitted_only_when_something_was_observed(capsys):
    """A stage with no reported bytes keeps its estimate rather than being handed an empty read set.

    Asserted over BEHAVIOUR, not over the source line that used to implement it. The old form checked
    for the literal `if _ws_bytes > 0:`, which passed while the hook watched a class the build never
    instantiates -- the marker was correctly withheld, for the wrong reason, on every run. What must
    hold is that a zero read set emits no measurement marker; a zero must now also SAY so, which the
    parser ignores because it keys on "TRACE_STAGE_BYTES[" and the failure line is
    "TRACE_STAGE_BYTES_NONE[".
    """
    import sys
    import types

    _stub = types.ModuleType("ttnn")
    _stub.decorators = types.ModuleType("ttnn.decorators")
    sys.modules.setdefault("ttnn", _stub)
    sys.modules.setdefault("ttnn.decorators", _stub.decorators)
    from agent import trace_replay as _tr

    _tr._report_read_set("decode", 0, 0)
    _tr._report_read_set("encode", 12, 4_786_000_000)
    out = capsys.readouterr().out
    assert "TRACE_STAGE_BYTES[decode]" not in out, "an empty read set was emitted as a measurement"
    assert "TRACE_STAGE_BYTES_NONE[decode]" in out, "a zero said nothing at all -- the original defect"
    assert "TRACE_STAGE_BYTES[encode]=4786000000" in out


def test_the_marker_survives_the_round_trip():
    """Emitted in the workload process, parsed in the harness, persisted, read by the renderer --
    four hops, and a marker parsed at none of them is why the width once never arrived."""
    line = "TRACE_STAGE_BYTES[decode]=4786000000"
    nm = line.split("TRACE_STAGE_BYTES[", 1)[1].split("]", 1)[0]
    val = int(float(line.split("]=", 1)[1].split()[0]))
    assert nm == "decode" and val == 4_786_000_000
    mcp = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert 'line.split("TRACE_STAGE_BYTES[", 1)' in mcp, "emitted but never parsed"
    assert '"bytes": stage_bytes or {},' in mcp, "parsed but never persisted"
    assert "def read_stage_bytes(" in mcp, "persisted but never read back"


def test_it_uses_the_census_definition_of_a_device_tensor():
    """ONE DEFINITION of "how big is this device tensor". The first version re-implemented
    weight_census._tensor_entry inline and got it subtly wrong -- it read `.dtype` only, missing the
    `get_dtype()` fallback, so a tensor exposing the latter counted as zero bytes. Two definitions of
    one quantity is how the ceiling's divisor came to disagree with itself before."""
    src = (_PA / "agent" / "trace_replay.py").read_text()
    assert "from .weight_census import _tensor_entry as _entry" in src
    i = src.index("def _note(x, depth=0):")
    body = src[i : src.index("def _make_counting", i)]
    assert "_entry(x)" in body
    for reimplemented in ("_on_device", "padded_shape", "for d in tuple(shape)"):
        assert reimplemented not in body, "still re-implementing the census: %s" % reimplemented


def test_a_tensor_that_only_exposes_get_dtype_is_counted():
    """The bug the duplication carried: `.dtype` absent, `get_dtype()` present."""

    class _GetDtypeOnly:
        shape = (1_000_000,)

        def get_dtype(self):
            class _D:
                name = "BFLOAT16"

            return _D()

        def storage_type(self):
            return "DEVICE"

    _n, b = _hooked(lambda op: op(_GetDtypeOnly()))
    assert b == 2_000_000, b


# --- and it has to run on the path the model actually takes ---------------------------------------


def test_the_measurement_is_on_the_path_every_stage_takes():
    """WIRED IS NOT FIRED. The hook was placed in _measure_native, which _measure_stage reaches only
    for a SELF-TRACED pipeline. voxtral declares no self_traced, so all three of its stages took the
    capture branch and the measurement never ran once -- the stage doc recorded `bytes: {}` on every
    run while the roofline printed per-stage figures that were the checkpoint estimate.

    Four separate mechanisms were built on top of that measurement -- the pin, the central owner, the
    registry entry, the bandwidth cell -- before anyone checked whether it executed.
    """
    import ast

    src = (_PA / "agent" / "trace_replay.py").read_text()
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_measure_stage":
            fn = node
    assert fn, "_measure_stage is gone"
    body = ast.unparse(fn)
    assert "_count_op_dispatches" in body, "the read set is not measured on the general path"
    assert "TRACE_STAGE_BYTES" in body, "the general path measures it and does not report it"
    # and it must sit BEFORE the self-traced early return does not apply -- i.e. after that guard,
    # so a self-traced stage still gets its own measurement inside _measure_native
    assert body.index("self_traced") < body.index("_count_op_dispatches")


def test_a_self_traced_stage_still_measures_its_own():
    """The two paths are exclusive; neither may lose the measurement."""
    import ast

    src = (_PA / "agent" / "trace_replay.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_measure_native":
            assert "_count_op_dispatches" in ast.unparse(node)
            return
    raise AssertionError("_measure_native is gone")


# The two tests that stood here asserted the stage marks live in measure_adapter and that the
# skeleton's profiler guard was gone. Both described a design that has been removed: measure_adapter
# is the STOPWATCH's machinery, it runs with the profiler popped, and a mark there marks nothing.
# Per-stage marks now belong to the tracy run and are covered by
# tests/test_stage_marks_go_where_the_per_op_data_is.py. What remains here is the READ SET, which is
# genuinely the stopwatch's to measure -- a byte total needs full depth.


def test_an_unmarked_capture_still_falls_back_and_says_so():
    """Additive: no marks -> no windows -> no stage_buckets -> the whole-profile peak, plus a line
    saying which footing the reader is on."""
    tt = (_PA / "agent" / "tracy_tool.py").read_text()
    assert "def stage_windows(" in tt and "_per_stage_buckets" in tt
    assert "no stage signposts in the capture" in tt
