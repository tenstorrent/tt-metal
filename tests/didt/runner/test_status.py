# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.didt.runner import status
from tests.didt.runner.checks import Status
from tests.didt.runner.status import Attempt, StepState, write

RUN = "LOG_2026_11_27_09_10_bh-glx-110-a06u02_d9006726d27"
HEADER = {
    "tool": "didt_runner",
    "host": "bh-glx-110-a06u02",
    "run": RUN,
    "mode": "real",
    "commit": "d9006726d27",
    "run_dir": f"/data/x/{RUN}",
    "config": {"path": "tests/didt/didt_runner_config.json", "sha256": "ab"},
    "chips": [{"dev": 19, "bdf": "0000:84:00.0", "tray": 4, "asic": 4, "asic_id": "0x81caba39768f9912", "kmd": 27}],
}


def _steps() -> list[StepState]:
    hang = Attempt(
        name=None,
        env={},
        opts=[],
        status=Status.HANG,
        reason="triage x/nd/minimal/crash_triage.csv",
        details=["device 27 bdf=0000:c4:00.0 tray=3 asic=4: hung in MinimalMatmulDeviceOperation"],
        devices=[{"dev": 27, "bdf": "0000:c4:00.0", "hung": True}],
        log="x/nd/minimal/log",
        start=1.0,
        end=2.0,
    )
    throttle = Attempt("throttle", {"TT_MM_THROTTLE_PERF": "5"}, [], Status.PASS, "1 passed", [], [], "y", 3.0, 4.0)
    pending = Attempt(None, {}, [], Status.PENDING, "", [], [], "", None, None)
    return [StepState("nd:minimal", hang, [throttle]), StepState("nd:ff1", pending, [])]


def test_shape_matches_spec(tmp_path):
    path = tmp_path / "status.json"
    write(path, HEADER, _steps(), "nd:ff1")
    doc = json.loads(path.read_text())
    assert list(doc) == ["tool", "host", "run", "mode", "commit", "run_dir", "config", "chips", "current", "steps"]
    assert doc["current"] == "nd:ff1"
    step = doc["steps"][0]
    assert list(step) == ["key", "status", "reason", "details", "log", "start", "end", "devices", "fallbacks"]
    assert step["status"] == "HANG" and step["devices"][0]["dev"] == 27
    fb = step["fallbacks"][0]
    assert list(fb) == ["name", "env", "opts", "status", "reason", "details", "devices", "log", "start", "end"]
    assert fb["name"] == "throttle" and fb["status"] == "PASS" and fb["env"] == {"TT_MM_THROTTLE_PERF": "5"}
    assert doc["steps"][1] == {
        "key": "nd:ff1",
        "status": "PENDING",
        "reason": "",
        "details": [],
        "log": "",
        "start": None,
        "end": None,
        "devices": [],
        "fallbacks": [],
    }


def test_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    write(path, HEADER, _steps(), None)
    before = path.read_text()

    def boom(obj, f, **kwargs):
        f.write('{"tool": "didt_ru')
        raise KeyboardInterrupt

    monkeypatch.setattr(status.json, "dump", boom)
    try:
        write(path, HEADER, _steps(), "nd:ff1")
    except KeyboardInterrupt:
        pass
    else:
        pytest.fail("write swallowed the interrupt")
    assert path.read_text() == before and json.loads(before)["current"] is None
    assert os.listdir(tmp_path) == ["status.json"]


def _sweep_dir() -> Path | None:
    candidates = [os.environ.get("GWDT_SWEEP_DIR", "")]
    candidates += [str(Path.home() / ".claude/skills/build-gwdt-sweep")]
    candidates += [f"/data/{os.environ.get('USER', '')}/.claude/skills/build-gwdt-sweep"]
    return next((Path(c) for c in candidates if c and (Path(c) / "sweep.py").is_file()), None)


def test_renders_with_sweep_page(tmp_path):
    sweep = _sweep_dir()
    if sweep is None:
        pytest.skip("build-gwdt-sweep skill not found; set GWDT_SWEEP_DIR")
    run_dir = tmp_path / RUN
    run_dir.mkdir()
    write(run_dir / "status.json", HEADER, _steps(), None)
    scan = subprocess.run(
        [sys.executable, str(sweep / "scan.py"), str(run_dir)], capture_output=True, text=True, check=True
    )
    (tmp_path / "scan.json").write_text(scan.stdout)
    (tmp_path / "labels.json").write_text("{}")
    out = tmp_path / "out.html"
    r = subprocess.run(
        [
            sys.executable,
            str(sweep / "sweep.py"),
            "render",
            str(tmp_path / "scan.json"),
            str(tmp_path / "labels.json"),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    page = out.read_text()
    assert "HANG" in page and "fallback 1: PASS 1 passed" in page
