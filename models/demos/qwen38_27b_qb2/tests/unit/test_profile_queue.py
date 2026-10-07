# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the persistent queue without device access or real systemd calls."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("qualified,diagnostic_status", [(False, 0), (True, 0), (True, 124)])
@pytest.mark.parametrize("skip_profile", [False, True])
def test_bounded_diagnostics_never_bypass_qualification(tmp_path, qualified, diagnostic_status, skip_profile):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    state = "inactive" if qualified else "failed"
    scripts = {
        "systemctl": f'#!/bin/bash\ncase "$*" in *ActiveState*) echo {state};; *) echo 0;; esac\n',
        "timeout": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$QUEUE_LOG"\nexit ' + str(diagnostic_status) + "\n",
    }
    for name, body in scripts.items():
        path = binaries / name
        path.write_text(body)
        path.chmod(0o755)
    demo = tmp_path / "metal-galaxy/models/demos/qwen38_27b_qb2/demo"
    demo.mkdir(parents=True)
    (demo / "run_galaxy_serving.sh").write_text('#!/bin/bash\nprintf "SERVE %s\\n" "$*" >> "$QUEUE_LOG"\n')
    log = tmp_path / "queue.log"
    script = Path(__file__).resolve().parents[2] / "demo/run_profile_then_serving.sh"
    result = subprocess.run(
        [
            "/bin/bash",
            str(script),
            str(tmp_path),
            str(tmp_path / "profile"),
            str(tmp_path / "attention"),
            str(tmp_path / "serving"),
            "qualification.service",
            str(tmp_path / "qualification.json"),
        ],
        env={
            **os.environ,
            "PATH": f"{binaries}:/usr/bin:/bin",
            "QUEUE_LOG": str(log),
            "QWEN_SKIP_PROFILE": "1" if skip_profile else "0",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    if not qualified:
        assert result.returncode == 3
        assert not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        commands = log.read_text().splitlines()
        assert len(commands) == (2 if skip_profile else 3)
        assert all(line.startswith("--signal=TERM --kill-after=300 900 ") for line in commands[:-1])
        if skip_profile:
            assert "P0 gate remains incomplete" in result.stdout
            assert not (tmp_path / "profile.log").exists()
        else:
            assert "run_galaxy_layer_profile.sh" in commands[0]
        assert "run_long_context_attention.sh" in commands[-2]
        assert commands[-1].startswith("SERVE ") and commands[-1].endswith("qualification.json")
