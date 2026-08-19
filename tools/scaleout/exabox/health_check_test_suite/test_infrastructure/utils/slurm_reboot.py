# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Slurm self-heal: reboot a failing node once and let Slurm rerun the suite.

Only meaningful under the bare-Slurm (exabox) deployment; the orchestration
(k8s) launch mode reschedules pods itself and has no ``scontrol``/``SLURM_JOB_ID``
to drive this, so the runner gates the call on ``launch_mode == "slurm"``.
Stdlib-only (like ``report.py``) to stay importable without the runner's deps.
"""

from __future__ import annotations

import logging
import re
import subprocess

log = logging.getLogger(__name__)

# Allowlists for the values that reach the scontrol argv (job ids may carry an
# array-task `123_4` or het-job `123+0` suffix).
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
    # Allowlist-validate and pass only the matched substring into the argv. No
    # shell=True, so this is defense-in-depth; it also clears the Cycode taint.
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
