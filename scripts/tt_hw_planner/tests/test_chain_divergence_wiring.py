"""Unit tests for the chain-divergence wiring helpers in cli.py.

These cover the cli-side glue introduced by Item 1b of the 2026-06-02
audit: probe auto-enable, the all-in-one diagnostic runner, the
logger, and the persister. The pure comparator
(:func:`compare_hf_tt_probes`) is covered separately by
``test_chain_divergence_comparator.py``.

The wiring helpers are best-effort — every failure mode degrades to
"no diagnostic" rather than raising, so the e2e PCC failure path can
escalate uninterrupted whether the diagnostic ran or not.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts.tt_hw_planner.agentic.probe import (
    HFModuleStats,
    HFProbeResult,
    ChainDivergenceResult,
    ModuleDivergence,
    compare_hf_tt_probes,
)
from scripts.tt_hw_planner.cli import (
    _auto_enable_tt_probe,
    _find_demo_dir_safe,
    _log_chain_divergence,
    _persist_chain_divergence,
    _run_chain_divergence_diagnostic,
)


@contextmanager
def _env_var(name: str, value):
    """Save/restore an env var across a test."""
    sentinel = object()
    prev = os.environ.get(name, sentinel)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if prev is sentinel:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev  # type: ignore[arg-type]


def _make_tt_probe_json(tmp_path: Path, records: list) -> Path:
    """Write a TT-probe-shaped JSON file to tmp_path and return path."""
    p = tmp_path / "tt_probe.json"
    p.write_text(json.dumps({"version": 1, "captured_at": 0, "count": len(records), "records": records}))
    return p


def _hf_stats(name: str, step: int, mean: float = 1.0) -> HFModuleStats:
    return HFModuleStats(
        qualified_name=name,
        class_name="X",
        step=step,
        shape=(1, 4),
        dtype="float32",
        mean=mean,
        std=0.1,
        l2=1.0,
        abs_max=0.7,
    )


# ─── _auto_enable_tt_probe ───────────────────────────────────────────


def test_auto_enable_sets_env_var_to_deterministic_path() -> None:
    """The env var must be set so subprocess pytest runs install the
    probe. The path must be deterministic for a given model_id so the
    diagnostic can find it on rerun without re-discovering."""
    with _env_var("TT_PLANNER_PROBE_OUTPUT", None):
        p1 = _auto_enable_tt_probe("org/some-model")
        p2 = _auto_enable_tt_probe("org/some-model")
        assert p1 == p2  # idempotent / deterministic
        assert os.environ.get("TT_PLANNER_PROBE_OUTPUT") == p1
        assert "some-model" in p1.lower() or "some_model" in p1.lower()


def test_auto_enable_uses_distinct_paths_per_model() -> None:
    """Two models in the same session must not share the probe file."""
    with _env_var("TT_PLANNER_PROBE_OUTPUT", None):
        p_a = _auto_enable_tt_probe("org/model-a")
        p_b = _auto_enable_tt_probe("org/model-b")
        assert p_a != p_b


# ─── _run_chain_divergence_diagnostic: setup-failure paths ──────────


def test_diagnostic_returns_none_when_env_var_not_set(capsys) -> None:
    """If TT_PLANNER_PROBE_OUTPUT was never set, the probe didn't run
    and there's nothing to compare — degrade silently to None."""
    with _env_var("TT_PLANNER_PROBE_OUTPUT", None):
        result = _run_chain_divergence_diagnostic("org/m")
    assert result is None
    err = capsys.readouterr().err
    assert "TT_PLANNER_PROBE_OUTPUT not set" in err


def test_diagnostic_returns_none_when_probe_file_missing(capsys, tmp_path) -> None:
    """Env var set but file doesn't exist (probe install failed in
    subprocess?) → degrade to None."""
    missing = str(tmp_path / "nope.json")
    result = _run_chain_divergence_diagnostic("org/m", probe_output_path=missing)
    assert result is None
    err = capsys.readouterr().err
    assert "does not exist" in err


def test_diagnostic_returns_none_when_probe_file_malformed(capsys, tmp_path) -> None:
    """Malformed JSON in probe file → degrade to None, log warning."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    result = _run_chain_divergence_diagnostic("org/m", probe_output_path=str(bad))
    assert result is None
    err = capsys.readouterr().err
    assert "could not read TT probe records" in err


def test_diagnostic_returns_none_when_records_list_empty(capsys, tmp_path) -> None:
    """Empty records → degrade to None."""
    p = _make_tt_probe_json(tmp_path, [])
    result = _run_chain_divergence_diagnostic("org/m", probe_output_path=str(p))
    assert result is None
    err = capsys.readouterr().err
    assert "empty or malformed" in err


# ─── _run_chain_divergence_diagnostic: HF probe paths ──────────────


def test_diagnostic_returns_none_when_hf_probe_returns_none(tmp_path) -> None:
    """HF probe couldn't load model (transformers missing, bad id, etc.)
    → diagnostic returns None, never raises."""
    p = _make_tt_probe_json(
        tmp_path, [{"qualified_name": "a", "step": 0, "mean": 1, "std": 0.1, "l2": 1, "abs_max": 0.7}]
    )
    with patch("scripts.tt_hw_planner.agentic.probe.probe_hf_modules", return_value=None):
        result = _run_chain_divergence_diagnostic("org/m", probe_output_path=str(p))
    assert result is None


def test_diagnostic_returns_none_when_hf_probe_raises(capsys, tmp_path) -> None:
    """HF probe raised (model load OOM, etc.) → caught + logged, None
    returned. The e2e escalation must not be blocked."""
    p = _make_tt_probe_json(
        tmp_path, [{"qualified_name": "a", "step": 0, "mean": 1, "std": 0.1, "l2": 1, "abs_max": 0.7}]
    )
    with patch("scripts.tt_hw_planner.agentic.probe.probe_hf_modules", side_effect=RuntimeError("OOM")):
        result = _run_chain_divergence_diagnostic("org/m", probe_output_path=str(p))
    assert result is None
    err = capsys.readouterr().err
    assert "HF probe raised" in err


def test_diagnostic_returns_result_on_happy_path(tmp_path) -> None:
    """Valid TT records + working HF probe → real ChainDivergenceResult.
    The result is what the comparator computed; the diagnostic just
    glues the inputs together and forwards. Asserts the wiring is
    correct, not the comparator's logic (which is tested separately)."""
    p = _make_tt_probe_json(
        tmp_path,
        [{"qualified_name": "layer0", "step": 0, "mean": 0.5, "std": 0.1, "l2": 1.0, "abs_max": 0.7}],
    )
    fake_hf = HFProbeResult(
        model_id="org/m",
        records=[_hf_stats("layer0", 0, mean=1.0)],  # mean drifted vs TT's 0.5
        num_modules_hooked=1,
        decode_steps=[0],
        note="ok",
        prompt_text="hi",
        elapsed_s=0.1,
    )
    with patch("scripts.tt_hw_planner.agentic.probe.probe_hf_modules", return_value=fake_hf):
        result = _run_chain_divergence_diagnostic("org/m", probe_output_path=str(p), threshold=0.05)
    assert isinstance(result, ChainDivergenceResult)
    assert result.first_divergence is not None
    assert result.first_divergence.qualified_name == "layer0"


# ─── _log_chain_divergence ──────────────────────────────────────────


def test_log_does_nothing_when_result_is_none(capsys) -> None:
    """None result → silent. (The diagnostic-skipped messages come
    from inside _run_chain_divergence_diagnostic, not here.)"""
    _log_chain_divergence(None, model_id="org/m")
    out = capsys.readouterr().out
    assert out == ""


def test_log_prints_banner_with_first_divergence(capsys) -> None:
    """A populated result must render a grep-friendly banner with
    the first-divergence module name, max drift, and per-stat breakdown."""
    div = ModuleDivergence(
        qualified_name="vision_encoder.layer_0",
        class_name="EncoderLayer",
        step=0,
        hf_stats={"mean": 1.0, "std": 0.1, "l2": 1.0, "abs_max": 0.7},
        tt_stats={"mean": 0.5, "std": 0.1, "l2": 1.0, "abs_max": 0.7},
        relative_drift={"mean": 0.5, "std": 0.0, "l2": 0.0, "abs_max": 0.0},
        max_drift=0.5,
    )
    result = ChainDivergenceResult(
        first_divergence=div,
        table=[div],
        paired_modules=1,
        unpaired_hf_modules=[],
        unpaired_tt_modules=[],
        threshold=0.05,
        note="ok",
    )
    _log_chain_divergence(result, model_id="org/m")
    out = capsys.readouterr().out
    assert "CHAIN-DIVERGENCE DIAGNOSTIC for org/m" in out
    assert "vision_encoder.layer_0" in out
    assert "FIRST DIVERGENCE" in out
    assert "max drift" in out


# ─── _persist_chain_divergence ──────────────────────────────────────


def test_persist_returns_none_for_none_result() -> None:
    """No diagnostic to persist → return None, no file written."""
    assert _persist_chain_divergence(None, demo_dir=Path("/tmp")) is None


def test_persist_returns_none_for_none_demo_dir() -> None:
    """No demo dir to persist into → return None, no file written.
    The diagnostic stays in the log only — better than raising."""
    div_result = ChainDivergenceResult(
        first_divergence=None,
        table=[],
        paired_modules=0,
        unpaired_hf_modules=[],
        unpaired_tt_modules=[],
        threshold=0.05,
        note="ok",
    )
    assert _persist_chain_divergence(div_result, demo_dir=None) is None


def test_persist_writes_json_to_demo_dir(tmp_path) -> None:
    """Happy path: result serialized as chain_divergence.json under
    the model's demo dir. Post-mortem tooling reads this back."""
    div_result = ChainDivergenceResult(
        first_divergence=None,
        table=[],
        paired_modules=0,
        unpaired_hf_modules=["a@0"],
        unpaired_tt_modules=[],
        threshold=0.05,
        note="ok",
    )
    out_path = _persist_chain_divergence(div_result, demo_dir=tmp_path)
    assert out_path is not None
    assert out_path.is_file()
    blob = json.loads(out_path.read_text())
    assert blob["paired_modules"] == 0
    assert blob["unpaired_hf_modules"] == ["a@0"]
    assert blob["threshold"] == 0.05


# ─── _find_demo_dir_safe ─────────────────────────────────────────────


def test_find_demo_dir_safe_returns_find_result_on_normal_path() -> None:
    """Happy path: wrapper just forwards find_demo_dir's return value."""
    expected = Path("/tmp/fake-demo")
    with patch("scripts.tt_hw_planner.bringup_loop.find_demo_dir", return_value=expected):
        assert _find_demo_dir_safe("org/m") == expected


def test_find_demo_dir_safe_returns_none_when_find_returns_none() -> None:
    """Forwards None too — find_demo_dir returns None when no demo dir
    matches, and the wrapper preserves that signal."""
    with patch("scripts.tt_hw_planner.bringup_loop.find_demo_dir", return_value=None):
        assert _find_demo_dir_safe("org/m") is None


def test_find_demo_dir_safe_swallows_exceptions() -> None:
    """Defensive: if find_demo_dir raises (e.g. discovery.BRINGUP_ROOT()
    misconfigured), the wrapper returns None instead of propagating.
    The diagnostic site needs an Optional[Path], not an exception."""
    with patch("scripts.tt_hw_planner.bringup_loop.find_demo_dir", side_effect=RuntimeError("boom")):
        assert _find_demo_dir_safe("org/m") is None


def test_persist_handles_unwriteable_dir_gracefully(tmp_path, capsys) -> None:
    """Caller passes a non-existent dir → log warning, return None,
    never raise (so escalation isn't blocked by a disk issue)."""
    div_result = ChainDivergenceResult(
        first_divergence=None,
        table=[],
        paired_modules=0,
        unpaired_hf_modules=[],
        unpaired_tt_modules=[],
        threshold=0.05,
        note="ok",
    )
    bad = tmp_path / "does" / "not" / "exist"
    result = _persist_chain_divergence(div_result, demo_dir=bad)
    assert result is None
    err = capsys.readouterr().err
    assert "could not persist" in err
