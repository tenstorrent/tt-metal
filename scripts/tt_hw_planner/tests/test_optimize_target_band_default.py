"""Pin: optimize's --target-band is ON by default (stop when the DRAM-bandwidth band is reached),
with --no-target-band to opt out."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_target_band_defaults_on() -> None:
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "cli.py").read_text()
    i = src.find('"--target-band"')
    assert i >= 0, "optimize must register --target-band"
    region = src[i : i + 400]
    assert "BooleanOptionalAction" in region, "must allow --no-target-band opt-out"
    assert "default=True" in region, "--target-band must default ON"


def test_help_shows_opt_out() -> None:
    import subprocess

    p = subprocess.run(
        [sys.executable, "-m", "scripts.tt_hw_planner", "optimize", "--help"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=60,
    )
    assert p.returncode == 0
    assert "--no-target-band" in p.stdout
