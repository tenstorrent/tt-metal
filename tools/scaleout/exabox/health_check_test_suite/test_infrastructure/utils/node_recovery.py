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

Under Slurm the runner itself now executes inside the tt-metal image, which
carries no Slurm client and no route to the node's munge socket, so it cannot
call ``scontrol`` at all. The job's *host* side can, and does: the launcher
(``launch-health-check.sh`` in exabox-infra) starts a broker before ``docker
run`` and passes the paths of a request and a result file, both inside the run
directory the container bind-mounts. Asking for a reboot is then writing the
request file and waiting for the result, which carries the same "armed, or here
is why not" answer the direct ``scontrol`` calls used to return. When a usable
``scontrol`` *is* on PATH — the runner invoked straight on a node, outside the
image — the calls are still made here.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

# One reboot per job. A node that fails again after its power cycle has a real
# fault, so it gets a ticket instead of looping through reboots forever.
REBOOT_CAP = 1

# Set by the host-side launcher, but only once it has confirmed it can drive
# Slurm (client on PATH, passwordless sudo for it, controller reachable). Both
# paths are inside the bind-mounted run directory, so they name the same file on
# either side of the container boundary.
REQUEST_FILE_ENV = "HC_SELF_HEAL_REQUEST_FILE"
RESULT_FILE_ENV = "HC_SELF_HEAL_RESULT_FILE"

# The broker answers as soon as scontrol returns, so this budget only has to
# cover a slow or backed-up controller. Running out is reported as a self-heal
# failure, which tickets the node rather than dropping the failure silently.
BROKER_TIMEOUT_SECONDS = 120
_BROKER_POLL_SECONDS = 1.0


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


def reboot_and_requeue(node: str, slurm_job_id: str) -> str | None:
    """Arm a reboot (`scontrol reboot ASAP nextstate=RESUME` drains then
    auto-resumes the node) and `scontrol requeue` the job. The caller must return
    immediately without blocking: an earlier version slept and wedged the node at
    ALLOCATED+DRAIN+REBOOT_REQUESTED.

    Runs through the host-side broker when the launcher provided one (the normal
    containerized path, where there is no Slurm client in here), otherwise
    directly.

    Returns ``None`` on success, else a short failure reason (e.g.
    ``scontrol: command not found``) so a failed reboot isn't a silent no-op.
    """
    request = os.environ.get(REQUEST_FILE_ENV, "")
    result = os.environ.get(RESULT_FILE_ENV, "")
    if request and result:
        return _self_heal_via_host_broker(Path(request), Path(result), node, slurm_job_id)
    if shutil.which("scontrol") is None:
        return (
            "no Slurm client here and the launcher offered no host-side self-heal "
            f"broker (${REQUEST_FILE_ENV} unset); the node was not rebooted"
        )
    return _self_heal_via_scontrol(node, slurm_job_id)


def _self_heal_via_host_broker(
    request_path: Path,
    result_path: Path,
    node: str,
    slurm_job_id: str,
) -> str | None:
    """Hand the reboot to the broker running on the job's host side and wait for
    its verdict.

    The request file is a trigger, not an instruction: the broker only ever
    reboots the node it runs on and requeues its own job, so what is written here
    is for the operator reading the run dir, not for the broker to act on. That
    matters because the scheduled run directory is one shared NFS path for the
    whole fleet.
    """
    log.info("Requesting host-side self-heal for %s (job %s) via %s ...", node, slurm_job_id, request_path)
    # Rename into place so the broker can never poll a half-written request.
    staging_path = request_path.parent / f"{request_path.name}.tmp"
    try:
        staging_path.write_text(f"node={node}\nslurm_job_id={slurm_job_id}\n")
        os.replace(staging_path, request_path)
    except OSError as exc:
        return f"could not write the self-heal request to {request_path}: {exc}"

    deadline = time.monotonic() + BROKER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            answer = result_path.read_text().strip()
        except FileNotFoundError:
            answer = ""
        except OSError as exc:
            return f"could not read the self-heal result from {result_path}: {exc}"
        if answer:
            status, _, detail = answer.partition(" ")
            if status == "OK":
                return None
            return detail.strip() or f"host-side self-heal answered {answer!r}"
        time.sleep(_BROKER_POLL_SECONDS)

    # A successful requeue kills this job, so losing the race to read the answer
    # is possible and harmless: the rerun happens either way and the ticket this
    # reason produces is closed by the next passing run.
    return f"host-side self-heal broker did not answer within {BROKER_TIMEOUT_SECONDS}s"


def _self_heal_via_scontrol(node: str, slurm_job_id: str) -> str | None:
    """Drive Slurm directly, for a runner executing on the node rather than in
    the image. Reboot needs sudo; `-n` fails fast so the caller can fall through
    to ticketing."""
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
        printable = " ".join(cmd)
        log.info("Running `%s` ...", printable)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Any spawn failure (scontrol absent, sudo not executable) has to stay
        # non-fatal: crashing here would lose the JIRA ticket for a real failure.
        except (OSError, subprocess.TimeoutExpired) as exc:
            reason = f"`{printable}` could not run: {exc}"
            log.warning(reason)
            return reason
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            reason = f"`{printable}` failed (rc={result.returncode}): {detail}"
            log.warning(reason)
            return reason
    return None
