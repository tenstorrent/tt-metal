# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reboot-and-rerun self-heal for a failing health check.

Bare-Slurm only: many fabric failures are transient link/training faults that a
power cycle clears, so a failed run reboots the node once and requeues the job
so the suite reruns on a clean boot. Everything here goes through Slurm
(``scontrol``), which the Kubernetes/orchestration deployment has no equivalent
for — a pod restart doesn't power-cycle the host — so that deployment skips the
reboot path entirely and goes straight to ticketing.
"""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger(__name__)

# One reboot per job. A node that fails again after its power cycle has a real
# fault, so it gets a ticket instead of looping through reboots forever.
REBOOT_CAP = 1


def slurm_restart_count() -> int:
    """How many times Slurm has already requeued this job (0 outside Slurm)."""
    try:
        return int(os.environ.get("SLURM_RESTART_COUNT", "0") or "0")
    except ValueError:
        return 0


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
    commands = (
        [
            "sudo",
            "-n",
            "scontrol",
            "reboot",
            "ASAP",
            "nextstate=RESUME",
            "reason=fabric_hc_self_heal",
            node,
        ],
        ["scontrol", "requeue", slurm_job_id],
    )
    for cmd in commands:
        log.info("Running `%s` ...", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Any spawn failure (scontrol absent, sudo not executable) has to stay
        # non-fatal: crashing here would lose the JIRA ticket for a real failure.
        except (OSError, subprocess.TimeoutExpired) as exc:
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
