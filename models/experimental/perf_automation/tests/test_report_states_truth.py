# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The report must not assert things that are not true. It goes out for confirmation.

Two false statements were being printed for llama3_1_8b_p150:

  "(tok/s/u — N/A: not an LLM decode pipeline)"   -- it IS one; it runs a traced KV-cache decode.
      The ms branch is taken because active_bytes is 0, i.e. the physics numerator was never
      computed. Explaining a missing input by inventing a property of the model is worse than
      saying nothing.

  "modeled floor : 341.47 ms (Σ per-op roofline floors)"   -- reads as complete and arbitrary. It is
      physics (bytes/bandwidth dominates) but sums each bucket's top_ops only, so it is a LOWER
      bound over ~86% of device time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _sm():
    spec = importlib.util.spec_from_file_location("sm_truth_ut", _ROOT / "cc_optimize" / "summary.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sm_truth_ut"] = m
    spec.loader.exec_module(m)
    return m


_PROFILE = {
    "device_ms": 664.2,
    "buckets": [
        {"id": "matmul", "device_ms": 560.3, "top_ops": [{"device_ms": 469.0}]},
        {"id": "reduction", "device_ms": 53.4, "top_ops": [{"device_ms": 53.4}]},
        {"id": "host_overhead", "device_ms": 46.6, "top_ops": []},
    ],
}


def test_it_does_not_claim_a_decode_model_is_not_a_decode_model():
    m = _sm()
    out = m._roofline_lines({"modeled_floor_ms": 341.47, "active_bytes": 0, "peak_bw_gbps": 512.0}, 615.69)
    txt = "\n".join(out)
    assert "not an LLM decode pipeline" not in txt, txt
    assert "no weight-bytes input" in txt, txt


def test_it_still_says_not_decode_when_that_is_actually_why():
    m = _sm()
    out = m._roofline_lines(
        {"modeled_floor_ms": 341.47, "active_bytes": 12345, "peak_bw_gbps": 512.0, "is_llm_decode": False}, 615.69
    )
    assert "not an LLM decode pipeline" in "\n".join(out)


def test_the_floor_states_the_physics_and_its_coverage():
    m = _sm()
    basis = m._floor_basis(_PROFILE)
    assert "bytes/BW" in basis, basis
    assert "covers" in basis and "%" in basis, basis


def test_the_coverage_excludes_host_overhead_and_uncounted_ops():
    """522.4 of 664.2 device_ms is covered by top_ops (host_overhead excluded) -> 79%."""
    m = _sm()
    basis = m._floor_basis(_PROFILE)
    assert "79%" in basis, basis


def test_a_profile_without_buckets_degrades_to_the_bare_basis():
    m = _sm()
    assert m._floor_basis({}) == "Σ per-op max(FLOPs/peak, bytes/BW, dispatch)"
    assert m._floor_basis(None).startswith("Σ per-op max")


def test_the_rendered_floor_line_carries_the_basis_not_an_opaque_label():
    """Testing _floor_basis alone let a mutation revert the render line to the opaque
    "(Σ per-op roofline floors)" while every test still passed."""
    m = _sm()
    out = m._roofline_lines({"modeled_floor_ms": 341.47, "active_bytes": 0, "peak_bw_gbps": 512.0}, 615.69, _PROFILE)
    line = next(l for l in out if "modeled floor" in l)
    assert "bytes/BW" in line, line
    assert "covers" in line and "%" in line, line
    assert line.strip().endswith(")"), line


def _sm_and_led():
    import importlib.util
    import sys as _s
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1]
    out = []
    for name, rel in (("sm_trace_ut", "cc_optimize/summary.py"), ("led_trace_ut", "cc_optimize/measurements.py")):
        spec = importlib.util.spec_from_file_location(name, root / rel)
        mod = importlib.util.module_from_spec(spec)
        _s.modules[name] = mod
        spec.loader.exec_module(mod)
        out.append(mod)
    return out


def test_the_trace_pass_baseline_comes_from_the_ledger_not_the_profile_file(tmp_path, monkeypatch):
    """THE DEFECT: this line read per_token_ms out of the per-profile JSON, which EVERY profile
    overwrites -- so an optimized reading carried the word BASELINE and the number changed run to run
    (11.93 -> 9.34 on llama3_1_8b_p150). The durable row could never be written either: the writer's
    call was guarded on a name that was never defined.
    """
    sm, led = _sm_and_led()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    led.record(led.KIND_TRACE_PASS, led.PHASE_BEFORE, 11.93, depth="16", mode="tracy-trace")
    led.record(led.KIND_TRACE_PASS, led.PHASE_AFTER, 9.34, depth="16", mode="tracy-trace")
    line = sm._ledger_line(led.KIND_TRACE_PASS, "tracy trace pass", "", "")
    assert line and "11.93" in line and "9.34" in line, line


def test_an_optimized_profile_cannot_relabel_itself_as_the_trace_baseline(tmp_path, monkeypatch):
    """With a before-row already held, a later profile is an AFTER -- it cannot overwrite the anchor."""
    sm, led = _sm_and_led()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    led.record(led.KIND_TRACE_PASS, led.PHASE_BEFORE, 11.93, depth="16", mode="tracy-trace")
    for later in (9.34, 8.10, 7.02):
        led.record(led.KIND_TRACE_PASS, led.PHASE_AFTER, later, depth="16", mode="tracy-trace")
    first = led.first(led.KIND_TRACE_PASS, led.PHASE_BEFORE)
    assert first["value_ms"] == 11.93
    assert led.last(led.KIND_TRACE_PASS, led.PHASE_AFTER)["value_ms"] == 7.02


def test_one_extractor_reads_the_trace_ms_from_a_profile():
    sm, led = _sm_and_led()
    assert led.trace_ms_from_profile({"per_token_ms": 9.3393}) == 9.3393
    assert led.trace_ms_from_profile({"trace_per_token_ms": 4.0}) == 4.0
    assert led.trace_ms_from_profile({"trace_ms": 2.5}) == 2.5
    assert led.trace_ms_from_profile({"per_token_ms": None, "trace_ms": 2.5}) == 2.5
    for bad in ({}, None, {"per_token_ms": 0}, {"per_token_ms": -1}, {"per_token_ms": float("nan")}, "x"):
        assert led.trace_ms_from_profile(bad) is None, bad
    assert sm._baseline_trace_ms({"per_token_ms": 9.3393}) == 9.3393


def test_the_writers_guard_is_gone(tmp_path):
    """It was `if "_baseline_trace_ms_from" in globals()` against a name never defined, so the branch
    was permanently dead and no trace_pass row was ever recorded."""
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py").read_text()
    assert "_baseline_trace_ms_from" not in src
    assert "led.trace_ms_from_profile(prof)" in src


def test_no_report_is_written_when_no_model_root_was_configured(tmp_path, monkeypatch, capsys):
    """A report belongs to a MODEL. The model root falls back to "." when nothing configured one, so
    an unconfigured import wrote RUN_REPORT.md into whatever directory the process started in -- once
    into the repo, where a broad `git add` committed a generated artifact as tool code. The tool
    refuses, rather than every caller and every test having to remember to point it somewhere safe.
    """
    import importlib.util
    import sys as _s
    from pathlib import Path as _P

    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    monkeypatch.delenv("PERF_MCP_MODEL_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("pm_unrooted_ut", root / "cc_optimize" / "perf_mcp.py")
    pm = importlib.util.module_from_spec(spec)
    _s.modules["pm_unrooted_ut"] = pm
    spec.loader.exec_module(pm)

    assert pm._MODEL_ROOT_CONFIGURED is False
    assert not (tmp_path / "RUN_REPORT.md").exists()

    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "mymodel"))
    spec2 = importlib.util.spec_from_file_location("pm_rooted_ut", root / "cc_optimize" / "perf_mcp.py")
    pm2 = importlib.util.module_from_spec(spec2)
    _s.modules["pm_rooted_ut"] = pm2
    spec2.loader.exec_module(pm2)
    assert pm2._MODEL_ROOT_CONFIGURED is True
