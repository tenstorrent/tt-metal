# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One board, one clamp point -- not one per model, and not thrown away by --fresh.

The temperature at which the driver drops AICLK from 1350 MHz to 800 is a property of the board and
its cooling. The tool learns it instead of hardcoding it, which is right. It then filed what it
learned under the MODEL, which is not: state_dir() is ~/.perf_mcp/<model>, so every new model met the
same hardware knowing nothing about it.

Measured 2026-08-14, one host, one p300c, two answers to "what is too hot":

    gemma3   clean_at 140 samples (63.9 .. 75.3), clamped_at 27 samples (68.2 .. 79.9)
    voxtral  clean_at none,                       clamped_at 4  samples (76.0 .. 86.5)

Voxtral was re-learning, from four failures, something the box next door already knew from 167
observations. And --fresh deleted even those, so the next run started blinder still.

Re-learning is not free. A sample is only produced by RUNNING a measurement and seeing whether the
clock held, so the cost of an empty profile is paid in clamped runs -- each one minutes of device
time and a reading that gets discarded.
"""

import json
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _mcp(monkeypatch, state):
    from cc_optimize import perf_mcp

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(state))
    # The autouse sandbox pins the board dir to its own box; these tests are ABOUT the board dir, so
    # they take it back and point it at the layout under test.
    monkeypatch.delenv("PERF_MCP_BOARD_STATE_DIR", raising=False)
    monkeypatch.delenv("PERF_MCP_MAX_START_TEMP_C", raising=False)
    return perf_mcp


def _board(tmp_path, **models):
    """A ~/.perf_mcp with one directory per model, each holding its own profile."""
    root = tmp_path / ".perf_mcp"
    for name, doc in models.items():
        d = root / name
        d.mkdir(parents=True)
        (d / "perf_mcp_thermal_profile.json").write_text(json.dumps(doc))
    (root / "current").mkdir(parents=True, exist_ok=True)
    return root


def test_the_profile_sits_beside_the_board_not_inside_a_model(tmp_path, monkeypatch):
    root = _board(tmp_path)
    mcp = _mcp(monkeypatch, root / "current")
    assert mcp._thermal_profile_path().parent == root, mcp._thermal_profile_path()


def test_a_new_model_inherits_what_the_board_already_taught_another_one(tmp_path, monkeypatch):
    """THE POINT. Voxtral should not re-learn this board from four clamped runs."""
    root = _board(
        tmp_path,
        gemma3={"clean_at": [63.9, 68.0], "clamped_at": [68.2, 79.9]},
    )
    mcp = _mcp(monkeypatch, root / "current")
    doc = mcp._load_thermal_profile()
    assert 68.2 in doc["clamped_at"] and 63.9 in doc["clean_at"], doc


def test_observations_from_every_model_are_pooled(tmp_path, monkeypatch):
    root = _board(
        tmp_path,
        gemma3={"clamped_at": [68.2]},
        voxtral={"clamped_at": [76.0, 86.5]},
    )
    mcp = _mcp(monkeypatch, root / "current")
    assert mcp._load_thermal_profile()["clamped_at"] == [68.2, 76.0, 86.5]


def test_the_board_file_wins_once_it_exists(tmp_path, monkeypatch):
    """Adoption is a one-time migration, not a merge on every read -- otherwise a stale per-model
    file would keep resurrecting observations the board file has already aged out."""
    root = _board(tmp_path, gemma3={"clamped_at": [68.2]})
    (root / "perf_mcp_thermal_profile.json").write_text(json.dumps({"clamped_at": [71.0]}))
    mcp = _mcp(monkeypatch, root / "current")
    assert mcp._load_thermal_profile()["clamped_at"] == [71.0]


def test_pooling_makes_the_threshold_more_careful_never_less(tmp_path, monkeypatch):
    """A heavier model reaches the same start temperature with more heat already in the package, so
    mixing models can only pull the threshold DOWN. Waiting slightly too long is the safe direction."""
    alone = _board(tmp_path / "a", voxtral={"clamped_at": [76.0, 86.5]})
    mcp = _mcp(monkeypatch, alone / "current")
    solo = mcp._clamp_threshold_c()

    pooled = _board(tmp_path / "b", voxtral={"clamped_at": [76.0, 86.5]}, gemma3={"clamped_at": [68.2]})
    mcp = _mcp(monkeypatch, pooled / "current")
    assert mcp._clamp_threshold_c() <= solo, "pooling raised the temperature the tool is willing to start at"


def test_a_writable_parent_is_required_before_climbing(monkeypatch):
    """With no explicit state dir the tool uses the tempdir, whose parent is / -- not somewhere to
    write, and not a board directory either."""
    from cc_optimize import perf_mcp

    monkeypatch.delenv("PERF_MCP_BOARD_STATE_DIR", raising=False)
    monkeypatch.delenv("PERF_MCP_STATE_DIR", raising=False)
    assert perf_mcp._board_state_dir() == perf_mcp.state_dir()


def test_an_explicit_board_dir_wins_over_climbing(tmp_path, monkeypatch):
    """CLIMBING OUT OF A SANDBOX DEFEATS IT. The suite gives each test a private state dir; a blind
    .parent lands in the shared pytest root, and tests inherit each other's clamp observations."""
    from cc_optimize import perf_mcp

    box = tmp_path / "box"
    box.mkdir()
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(box))
    monkeypatch.setenv("PERF_MCP_BOARD_STATE_DIR", str(box))
    assert perf_mcp._board_state_dir() == box, "the sandbox pin was ignored and it climbed anyway"


def test_the_suite_sandbox_pins_the_board_dir():
    """Belt and braces: the fixture must set it, or every test escapes its box again."""
    src = (_PA / "tests" / "conftest.py").read_text()
    assert 'monkeypatch.setenv("PERF_MCP_BOARD_STATE_DIR", str(box))' in src


def test_a_mocked_test_never_waits_on_a_real_thermometer(tmp_path, monkeypatch):
    """The hang, at its root -- and it is no longer an empty box that prevents it.

    This used to assert the threshold was None: an isolated box has no observations, so nothing was
    learned, so the gate returned at once. That made suite speed a SIDE EFFECT of the profile being
    empty, and the moment the threshold became a stated 65C the side effect vanished and the suite
    hung on a real thermometer. The suite now states the gate off (conftest), which is what it always
    meant, and this asserts the property that matters: no wait, whatever the threshold is."""
    import time as _time

    from cc_optimize import perf_mcp

    box = tmp_path / "box"
    box.mkdir()
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(box))
    monkeypatch.setenv("PERF_MCP_BOARD_STATE_DIR", str(box))
    monkeypatch.setattr(perf_mcp, "_read_die_temp_c", lambda: 95.0)
    t0 = _time.time()
    ok, _t = perf_mcp._wait_for_thermal_headroom()
    assert ok is True and _time.time() - t0 < 1.0, "a mocked test polled a thermometer"


def test_fresh_no_longer_deletes_it():
    from agent import fresh_start

    assert "perf_mcp_thermal_profile.json" in fresh_start.KEEP
    assert "perf_mcp_thermal_profile.json" not in fresh_start._STATE_GLOBS


def test_fresh_leaves_the_board_profile_on_disk(tmp_path):
    from agent import fresh_start

    sd = tmp_path / "voxtral"
    sd.mkdir()
    keep = sd / "perf_mcp_thermal_profile.json"
    keep.write_text("{}")
    (sd / "perf_mcp_knob_cache.json").write_text("{}")
    removed = fresh_start.wipe(sd)
    assert keep.exists(), "--fresh still deletes the board's clamp history"
    assert len(removed) == 1
