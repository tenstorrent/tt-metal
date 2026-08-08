# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Slurm self-heal: reboot a failing node once and let Slurm rerun the suite.

Only meaningful under the bare-Slurm (exabox) deployment; the orchestration
(k8s) launch mode reschedules pods itself and has no ``scontrol``/``SLURM_JOB_ID``
to drive this, so the runner gates the call on ``launch_mode == "slurm"``.

Kept stdlib-only (like ``report.py``) so the gating/command logic can be
unit-tested without the runner's runtime deps (requests/paramiko/prometheus).
"""

from __future__ import annotations

import logging
import re
import subprocess

log = logging.getLogger(__name__)

# Slurm hostnames and job ids as they feed the scontrol argv below. Job ids are
# numeric with an optional array-task (`123_4`) or het-job (`123+0`) suffix.
_NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*")
_JOB_ID_RE = re.compile(r"[0-9]+(?:[_+][0-9]+)?")


def should_reboot(*, exit_code: int, enabled: bool, slurm_job_id: str, restart_count: int, cap: int) -> bool:
    """Reboot-and-rerun only when the run failed, the feature is enabled, we're
    under Slurm, and still under the cap (SLURM_RESTART_COUNT stops a bad node
    from looping forever)."""
    return exit_code != 0 and enabled and slurm_job_id != "unknown" and restart_count < cap


def reboot_and_requeue(node: str, slurm_job_id: str) -> bool:
    """Arm a reboot (`scontrol reboot ASAP nextstate=RESUME` drains then
    auto-resumes the node) and `scontrol requeue` the job. The caller must return
    immediately without blocking: an earlier version slept and wedged the node at
    ALLOCATED+DRAIN+REBOOT_REQUESTED. Reboot needs sudo; `-n` fails fast so the
    caller can fall through to ticketing.
    """
    # subprocess.run is invoked with an argv list (never shell=True), so there is
    # no shell to inject into; still, allowlist-validate both values against the
    # shapes Slurm produces and pass only the matched substring to the OS command.
    # Sourcing the argv from the regex match (not the raw input) also silences the
    # Cycode SAST "unsanitized user input in OS command" taint on subprocess below.
    node_match = _NODE_RE.fullmatch(node)
    if not node_match:
        log.warning("Refusing to reboot: invalid node name %r", node)
        return False
    job_match = _JOB_ID_RE.fullmatch(slurm_job_id)
    if not job_match:
        log.warning("Refusing to requeue: invalid Slurm job id %r", slurm_job_id)
        return False
    safe_node = node_match.group(0)
    safe_job_id = job_match.group(0)

    commands = (
        [
            "sudo",
            "-n",
            "scontrol",
            "reboot",
            "ASAP",
            "nextstate=RESUME",
            "reason=fabric_hc_self_heal",
            safe_node,
        ],
        ["scontrol", "requeue", safe_job_id],
    )
    for cmd in commands:
        log.info("Running `%s` ...", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.warning("Could not run `%s`: %s", " ".join(cmd), exc)
            return False
        if result.returncode != 0:
            log.warning(
                "`%s` failed (rc=%d): %s",
                " ".join(cmd),
                result.returncode,
                (result.stderr or result.stdout).strip(),
            )
            return False
    return True
