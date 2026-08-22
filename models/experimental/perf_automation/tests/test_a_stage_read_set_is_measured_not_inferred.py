# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The roofline's memory floor divides by bytes nobody ever measured.

WHAT THE DIVISOR HAD TO COME FROM, before this. All three are inferences from the checkpoint:

    params x bytes_per_param        the whole model, including towers a token never reads
    _TOWER_ONLY name list           `audio_tower|vision_tower|...` -- a model can spell it otherwise
    stage_roots section map         tried, reverted: drops lm_head on any untied model (12.5%)

And the thing that looks like a measurement is not one. summary._bytes_for reads

    _b = _stage_measured_bytes(profile, name) or _bytes_for(name, toks)

but _stage_measured_bytes filters profile buckets by a `stage` tag the profiler never writes, so it
returns 0 for every stage on every real profile and that line has always taken its estimate branch.
_stage_roofs says why: "_top_ops keys on (op_code, shape, memory) and records nothing about which
phase an op ran in".

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


def test_the_consumer_prefers_the_measurement_over_the_estimate():
    import cc_optimize.summary as S

    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index("_b = int((_meas_bytes or {}).get(name) or 0)")
    line = src[i : src.index("\n", i)]
    assert "_stage_measured_bytes(profile, name)" in line, line
    assert "_bytes_for(name, toks)" in line, line
    assert line.index("_meas_bytes") < line.index("_bytes_for"), "the estimate must be last"
    assert callable(S._measured_stage_bytes)


def test_the_marker_is_emitted_only_when_something_was_observed():
    """A stage with no reported bytes keeps its estimate rather than being handed an empty read set."""
    src = (_PA / "agent" / "trace_replay.py").read_text()
    assert "if _ws_bytes > 0:" in src
    assert 'print("TRACE_STAGE_BYTES[%s]=%d"' in src


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
