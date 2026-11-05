#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for llk-fetch-test-durations.sh, with a fake gh on PATH."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "llk-fetch-test-durations.sh"

FAKE_GH = """#!/usr/bin/env bash
# gh run list prints the run ids; gh run download copies $FIXTURE/<run>/* into -D.
if [[ "$2" == list ]]; then cat "$FIXTURE/runs"; exit 0; fi
run="$3"; dir="${@: -1}"
[[ -d "$FIXTURE/$run" ]] || { echo "no artifact matches" >&2; exit 1; }
cp -R "$FIXTURE/$run/." "$dir"
"""


def _fetch(tmp_path, runs, artifacts):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "runs").write_text("".join(f"{run}\n" for run in runs))
    for run, legs in artifacts.items():
        for name, durations in legs.items():
            leg = fixture / str(run) / name
            leg.mkdir(parents=True)
            (leg / ".test_durations").write_text(json.dumps(durations))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "gh").chmod(0o755)
    output = tmp_path / "github_output"
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FIXTURE=str(fixture), GITHUB_OUTPUT=str(output))
    out_dir = tmp_path / "out"
    subprocess.run(["bash", str(SCRIPT), "wf.yaml", "main", "push", str(out_dir)], env=env, check=True)
    tables = {p.stem: json.loads(p.read_text()) for p in out_dir.glob("*.json")}
    return output.read_text().strip(), tables


def test_nothing_found(tmp_path):
    assert _fetch(tmp_path, [11, 10], {}) == ("found=false", {})


def test_newest_run_with_durations_wins_and_parts_are_summed(tmp_path):
    found, tables = _fetch(
        tmp_path,
        [12, 11, 10],
        {
            11: {
                "test-durations-blackhole-1": {"a.py::t[1]": 1.0},
                "test-durations-blackhole-1-compile": {"a.py::t[1]": 0.5, "a.py::t[2]": 2.0},
                "test-durations-wormhole-2": {"b.py::t": 3.0},
            },
            10: {"test-durations-blackhole-1": {"old.py::t": 9.0}},
        },
    )
    assert found == "found=true"
    assert tables == {
        "blackhole": {"a.py::t[1]": 1.5, "a.py::t[2]": 2.0},
        "wormhole": {"b.py::t": 3.0},
    }
