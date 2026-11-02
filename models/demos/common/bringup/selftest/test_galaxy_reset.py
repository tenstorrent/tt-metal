# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A Galaxy mesh is reset with the cluster's recover-hosts, never tt-smi -r alone (owner 2026-11-02, Exabox Slurm
User Guide); other boxes keep tt-smi -r. Runs scripts/lib/tt_device_pool.sh in bash with the card count, tt-smi and
the recover command stubbed: no device is touched."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
POOL = REPO / "scripts" / "lib" / "tt_device_pool.sh"


def reset(tmp_path, cards: int, mesh: bool = True, recover: str = "echo Reset completed successfully", env=None):
    """(rc of ttpool_reset, what ran) with `cards` cards, tt-smi a stub that records its arguments."""
    calls = tmp_path / "calls"
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    (stub / "tt-smi").write_text(f'#!/bin/bash\necho "tt-smi $*" >> {calls}\n')
    (stub / "tt-smi").chmod(0o755)
    script = f"""
        source {POOL}
        ttpool_card_count() {{ echo {cards}; }}
        TTPOOL_IS_MESH={"true" if mesh else "false"}; TTPOOL_CARDS="0 1"
        ttpool_reset
    """
    run_env = {
        "PATH": f"{stub}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "TTPOOL_RECOVER_CMD": f'echo "recover-hosts $(hostname)" >> {calls}; {recover}',
        **(env or {}),
    }
    p = subprocess.run(["bash", "-c", script], env=run_env, capture_output=True, text=True)
    return p.returncode, (calls.read_text() if calls.exists() else ""), p.stdout


def test_a_galaxy_mesh_runs_recover_hosts_not_tt_smi(tmp_path):
    rc, calls, out = reset(tmp_path, 32)
    assert rc == 0, out
    assert "recover-hosts" in calls and "tt-smi" not in calls, calls


def test_a_failed_recovery_fails_the_reset(tmp_path):
    rc, calls, out = reset(tmp_path, 32, recover="echo something broke; exit 3")
    assert rc != 0 and "recover-hosts failed (rc 3)" in out, out
    rc, _, out = reset(tmp_path, 32, recover="echo no success line")  # rc 0 without its success line is a failure
    assert rc != 0, out


def test_other_boxes_and_the_override_keep_tt_smi(tmp_path):
    for cards, mesh, env, want in (
        (4, True, None, "tt-smi -r\n"),  # a QuietBox mesh: the whole box
        (32, False, None, "tt-smi -r 0 1\n"),  # single cards on a Galaxy: per card, as before
        (32, True, {"TTPOOL_GALAXY_RESET": "tt-smi"}, "tt-smi -r\n"),
    ):
        calls = tmp_path / "calls"
        calls.unlink(missing_ok=True)
        rc, got, out = reset(tmp_path, cards, mesh=mesh, env=env)
        assert rc == 0 and got == want, (cards, mesh, env, got, out)
