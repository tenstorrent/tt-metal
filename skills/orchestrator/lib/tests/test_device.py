"""Tests for lib/device.py — device registry, defaults extractor, tt-smi reset."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from skills.orchestrator.lib import device
from skills.orchestrator.lib.device import (
    DEVICE_REGISTRY,
    UnknownDeviceError,
    device_info,
    extract_defaults,
    tt_smi_reset,
)


# ---------------------------------------------------------------------------
# DEVICE_REGISTRY / device_info
# ---------------------------------------------------------------------------


def test_registry_has_expected_devices():
    assert set(DEVICE_REGISTRY) == {"n150", "n300", "p150", "t3k", "tg"}


def test_device_info_returns_copy():
    info = device_info("n150")
    info["arch"] = "mutated"
    assert device_info("n150")["arch"] == "wormhole_b0"


def test_device_info_unknown_raises():
    with pytest.raises(UnknownDeviceError):
        device_info("doesnotexist")


def test_device_info_blackhole_arch():
    assert device_info("p150")["arch"] == "blackhole"


def test_device_info_t3k_mesh_shape():
    assert device_info("t3k")["mesh_shape"] == (1, 8)


# ---------------------------------------------------------------------------
# extract_defaults — subprocess mocked
# ---------------------------------------------------------------------------


def _fake_completed(stdout: str, returncode: int = 0):
    """Build a CompletedProcess with the given stdout."""
    return subprocess.CompletedProcess(args=["grep"], returncode=returncode, stdout=stdout, stderr="")


def test_extract_defaults_parses_decimal_int(monkeypatch):
    stdout = (
        "path/to/foo.py:42:    l1_small_size = 16384,\n"
        "path/to/foo.py:43:    trace_region_size = 50_000_000,\n"
        "path/to/foo.py:44:    mesh_shape = (1, 8),\n"
    )
    monkeypatch.setattr(device.subprocess, "run", lambda *a, **kw: _fake_completed(stdout))
    result = extract_defaults("/dummy")
    assert result == {
        "l1_small_size": 16384,
        "trace_region_size": 50_000_000,
        "mesh_shape": (1, 8),
    }


def test_extract_defaults_parses_hex(monkeypatch):
    stdout = "path/foo.py:1:    l1_small_size = 0x4000\n"
    monkeypatch.setattr(device.subprocess, "run", lambda *a, **kw: _fake_completed(stdout))
    result = extract_defaults("/dummy")
    assert result["l1_small_size"] == 0x4000  # == 16384


def test_extract_defaults_first_hit_wins(monkeypatch):
    stdout = "a.py:1:    l1_small_size = 16384\n" "b.py:2:    l1_small_size = 99999\n"
    monkeypatch.setattr(device.subprocess, "run", lambda *a, **kw: _fake_completed(stdout))
    result = extract_defaults("/dummy")
    assert result["l1_small_size"] == 16384


def test_extract_defaults_missing_keys_are_None(monkeypatch):
    stdout = "a.py:1:    mesh_shape = (8, 4)\n"
    monkeypatch.setattr(device.subprocess, "run", lambda *a, **kw: _fake_completed(stdout))
    result = extract_defaults("/dummy")
    assert result == {
        "l1_small_size": None,
        "trace_region_size": None,
        "mesh_shape": (8, 4),
    }


def test_extract_defaults_grep_no_hits(monkeypatch):
    monkeypatch.setattr(
        device.subprocess,
        "run",
        lambda *a, **kw: _fake_completed("", returncode=1),
    )
    result = extract_defaults("/dummy")
    assert result == {
        "l1_small_size": None,
        "trace_region_size": None,
        "mesh_shape": None,
    }


def test_extract_defaults_subprocess_called_with_recursive_extended_regex(
    monkeypatch,
):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _fake_completed("", returncode=1)

    monkeypatch.setattr(device.subprocess, "run", fake_run)
    extract_defaults("/some/path")

    args = captured["args"]
    # First arg is the executable name "grep"
    assert args[0] == "grep"
    # Must include -r AND -E (either separate or combined as -rE / -rEn etc.)
    joined = " ".join(args[1:])
    assert "r" in joined and "E" in joined, f"missing -r/-E flags in {args!r}"
    # The path passed must appear as one of the args.
    assert "/some/path" in args
    # check=False is required.
    assert captured["kwargs"].get("check") is False


def test_extract_defaults_smoke_against_llama3_70b():
    """Named RED test from the plan — real grep against llama3_70b_galaxy."""
    repo_root = Path(__file__).resolve().parents[4]
    target = repo_root / "models" / "demos" / "llama3_70b_galaxy"
    if not target.exists():
        pytest.skip(f"{target} does not exist")
    result = extract_defaults(str(target))
    assert set(result.keys()) == {"l1_small_size", "trace_region_size", "mesh_shape"}


# ---------------------------------------------------------------------------
# tt_smi_reset
# ---------------------------------------------------------------------------


def test_tt_smi_reset_runs_with_correct_args(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(device.subprocess, "run", fake_run)
    rc = tt_smi_reset()
    assert captured["args"] == ["tt-smi", "-r"]
    assert captured["kwargs"].get("check") is False
    assert captured["kwargs"].get("capture_output") is True
    assert rc == 0


def test_tt_smi_reset_returns_nonzero_without_raising(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=2, stdout=b"", stderr=b"")

    monkeypatch.setattr(device.subprocess, "run", fake_run)
    rc = tt_smi_reset()
    assert rc == 2
