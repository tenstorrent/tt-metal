# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every stack divided by the same peak, because there was only ever one number to divide by.

    peak_flops, _dom = ...          resolved from the WHOLE profile
    ...
    for name, toks in stages:       42 lines later
        comp_ms = flops / peak_flops

One variable, used three times. Not three values that happened to agree.

It was the only number obtainable: _fidelity_breakdown aggregates FLOPs across the whole capture and
the ops carry no phase, which _stage_roofs states outright -- "_top_ops keys on (op_code, shape,
memory) and records nothing about which phase an op ran in".

WHY IT WAS NOT OBTAINABLE, AND NOW IS. tt-perf-report has always been able to slice a capture between
two signposts; refine() passes --start-signpost/--end-signpost on every profile; resolve_signposts
scans the model's tests for them. Nothing ever emitted any. Run 18's raw capture:

    17,786 rows, every one OP TYPE `tt_dnn_device`
    signpost rows: 0

so the slice was the documented no-op -- "No signposts found in the file. Using the entire file for
analysis." -- and every bucket carried regime "na". trace_replay already runs each stage alone, so a
mark around that call is all that was missing.

WHAT IT IS WORTH ON VOXTRAL, from run 18's own op table:

    encode      hifi4 0.946e12                     true peak 175.5   the shared value is right
    projector   hifi4 0.155e12                     true peak 175.5   right
    prefill     hifi4 0.329e12                     true peak 175.5   right
    decode      hifi2 3.299e12 + hifi4 2.608e12    true peak 351.0   the shared value is WRONG

Harmless today only because decode binds on memory by ~230x. It stops being harmless the moment the
fidelity rung lands on encode or prefill, which bind on compute -- which is what the ladder is for.
"""
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


_FIXTURE = _PA / "tests" / "profiles" / "run18_baseline_profile.json"


def _profile_with_stages():
    """Run 18's REAL ops, split into stages the way run 18 actually ran them.

    Real op dicts, not hand-written ones: residual_report prices an op from a set of annotator fields
    that is easy to get subtly wrong, and an unpriced op yields an empty breakdown -- which reads
    exactly like "this stage has no peak of its own".

    Attributed by the LEADING dim against the model's declared capacities, which is how the same
    split was verified against the run:

        1504, 384 -> encode (audio hidden 1280, ~1500 mel frames, then the projector)
        512       -> prefill (PREFILL_C = 512)
        32        -> decode  (DECODE_CAP = 32), and that is where the hifi2 lm_head lives

    Getting this wrong is not academic: a first version dumped every hifi4 op into decode, so hifi4
    outweighed the lm_head and decode came back 175.5 -- the fixture asserting the very behaviour the
    test exists to reject.
    """
    import json

    base = json.loads(_FIXTURE.read_text())
    CAP = {1504: "encode", 384: "encode", 512: "prefill", 32: "decode"}
    by_stage: dict = {}
    for b in base.get("buckets", []):
        for o in b.get("top_ops") or []:
            try:
                lead = int(str(o.get("op_code")).split()[1])
            except Exception:  # noqa: BLE001
                continue
            st = CAP.get(lead)
            if st:
                by_stage.setdefault(st, []).append(o)
    return {
        "buckets": base.get("buckets", []),
        "stage_buckets": {st: [{"id": "matmul", "device_ms": 5.0, "top_ops": ops}] for st, ops in by_stage.items()},
    }


def test_each_stage_resolves_its_own_peak():
    """THE POINT. Same profile, two stages, two different hardware constants."""
    import cc_optimize.summary as S

    prof = _profile_with_stages()
    enc, enc_dom = S._peak_for_stage("encode", prof, "m", "main")
    dec, dec_dom = S._peak_for_stage("decode", prof, "m", "main")
    assert abs(enc / 1e12 - 175.5) < 0.1, (enc, enc_dom)
    assert abs(dec / 1e12 - 351.0) < 0.1, (dec, dec_dom)
    assert enc != dec, "both stacks still share one peak"


def test_an_unmarked_capture_changes_nothing():
    """No signposts -> no stage_buckets -> 0.0, and the caller keeps the whole-profile figure it
    already had. Every older model and every run without a profiler is byte for byte as before."""
    import cc_optimize.summary as S

    assert S._peak_for_stage("decode", {"buckets": []}, "m", "main") == (0.0, "")
    assert S._peak_for_stage("decode", None, "m", "main") == (0.0, "")


def test_a_stage_absent_from_the_capture_is_refused():
    """A stage that did not run leaves no marks; pricing it from a window that is not there would
    invent a measurement."""
    import cc_optimize.summary as S

    assert S._peak_for_stage("vocoder", _profile_with_stages(), "m", "main") == (0.0, "")


def test_the_pinned_peak_outranks_the_capture(tmp_path, monkeypatch):
    """Same rule as every other roof input: the fidelity rung moves the mode a stage runs at, and a
    ceiling recomputed from it retreats ahead of the measurement."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    import cc_optimize.summary as S

    led = S._ledger()
    monkeypatch.setattr(
        led, "ledger_path", lambda model="", task="": tmp_path / ("%s_%s.jsonl" % (model or "m", task or "main"))
    )
    led.anchor(led.KIND_PEAK_FLOPS, 175.5e12, depth="decode", mode="roofline", source="t", model="m")
    pk, _dom = S._peak_for_stage("decode", _profile_with_stages(), "m", "main")
    assert abs(pk / 1e12 - 175.5) < 0.1, "the capture overrode the pin"


def test_the_peak_is_keyed_per_stage_not_per_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    import cc_optimize.summary as S

    led = S._ledger()
    monkeypatch.setattr(
        led, "ledger_path", lambda model="", task="": tmp_path / ("%s_%s.jsonl" % (model or "m", task or "main"))
    )
    led.anchor(led.KIND_PEAK_FLOPS, 175.5e12, depth="encode", mode="roofline", source="t", model="m")
    led.anchor(led.KIND_PEAK_FLOPS, 702.0e12, depth="decode", mode="roofline", source="t", model="m")
    assert abs(S._peak_for_stage("encode", {}, "m", "main")[0] / 1e12 - 175.5) < 0.1
    assert abs(S._peak_for_stage("decode", {}, "m", "main")[0] / 1e12 - 702.0) < 0.1


# --- the emission and the slicing ---------------------------------------------------------------


def test_the_stage_marks_are_emitted_around_the_isolated_run():
    """OUTSIDE the trace capture: _measure_stage captures and replays within itself, and host work
    inside a capture raises 'Writes/Reads are not supported during trace capture'."""
    src = (_PA / "agent" / "trace_replay.py").read_text()
    i = src.index('_signpost("stage:%s" % st.name)')
    j = src.index('_signpost("stage:%s:end" % st.name)')
    mid = src[i:j]
    assert "_measure_stage(device, st)" in mid, "the marks do not bracket the stage"
    assert "_capture_step_trace" not in mid, "a mark landed inside the trace capture"


def test_a_missing_profiler_costs_the_split_not_the_measurement():
    """A normal trace_replay has no tracy module at all -- the marker is only useful under a profiled
    run, and its absence must cost the split, never the measurement.

    ttnn is stubbed rather than skipped: skipping here would leave the emission path untested on any
    machine without a device, which is every machine that runs this suite in CI."""
    import types

    if "ttnn" not in sys.modules:
        dec = types.ModuleType("ttnn.decorators")

        class Operation:
            def __call__(self, *a, **kw):
                return None

        dec.Operation = Operation
        ttnn = types.ModuleType("ttnn")
        ttnn.decorators = dec
        sys.modules["ttnn"] = ttnn
        sys.modules["ttnn.decorators"] = dec
    try:
        import agent.trace_replay as tr
    except Exception as exc:  # noqa: BLE001
        pytest.fail("trace_replay must import against a stubbed ttnn: %s" % exc)
    assert tr._signpost("stage:decode") is None  # no tracy module -> no raise, no effect


def test_windows_are_read_from_the_capture_not_from_the_expected_stage_list(tmp_path):
    """A stage that failed to run leaves no marks, and must not be given a window it does not have."""
    from agent.tracy_tool import stage_windows

    csvp = tmp_path / "raw.csv"
    csvp.write_text(
        "OP CODE,OP TYPE\n"
        "stage:encode,signpost\n"
        "MatmulDeviceOperation,tt_dnn_device\n"
        "stage:encode:end,signpost\n"
        "stage:decode,signpost\n"
        "MatmulDeviceOperation,tt_dnn_device\n"
        "stage:decode:end,signpost\n"
    )
    assert stage_windows(csvp) == [
        ("encode", "stage:encode", "stage:encode:end"),
        ("decode", "stage:decode", "stage:decode:end"),
    ]


def test_an_unterminated_window_is_skipped(tmp_path):
    """A capture cut short mid-stage has a start and no end; half a window is not a window."""
    from agent.tracy_tool import stage_windows

    csvp = tmp_path / "raw.csv"
    csvp.write_text("OP CODE,OP TYPE\nstage:decode,signpost\nMatmulDeviceOperation,tt_dnn_device\n")
    assert stage_windows(csvp) == []


def test_an_unmarked_capture_yields_no_windows(tmp_path):
    """Run 18's shape: 17,786 rows, all tt_dnn_device, no signposts."""
    from agent.tracy_tool import stage_windows

    csvp = tmp_path / "raw.csv"
    csvp.write_text("OP CODE,OP TYPE\n" + "MatmulDeviceOperation,tt_dnn_device\n" * 50)
    assert stage_windows(csvp) == []


def test_the_marks_agree_with_how_tt_perf_report_finds_them():
    """tt-perf-report selects `df[df["OP TYPE"] == "signpost"]`, so reading the same column the same
    way is what keeps the slicer and the window list from disagreeing."""
    import tt_perf_report.perf_report as pr
    import inspect

    assert 'df["OP TYPE"] == "signpost"' in inspect.getsource(pr)
