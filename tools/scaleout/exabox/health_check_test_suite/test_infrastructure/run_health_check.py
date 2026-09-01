#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fabric System Health Check runner.

Runs the TT fabric system health check (``diag_runner.py``) as a subprocess,
writes the JSON report and per-test logs to a results directory, creates a JIRA
ticket on failure, produces CSV analysis, and uploads results via SFTP.

Designed to run *inside* the tt-metal upstream image (the image is the runtime),
so the diag suite is invoked directly rather than in a nested container. The
orchestration's supporting logic lives in the ``utils`` package:

    utils/diag_execution.py  run the diag suite as a subprocess (timeout/kill aware)
    utils/system_info.py     tt-smi / kmd / fw version discovery
    utils/telemetry.py       tt-telemetry (Prometheus) collection + formatting
    utils/report.py          post-reset normalization + actionable-failure verdict
    utils/node_recovery.py   Slurm reboot-and-requeue self-heal on failure
    utils/jira_client.py     JIRA ticket create/update/close/attach
    utils/sftp_upload.py     CSV upload to the Data-team SFTP endpoint
    utils/secrets_loader.py  JIRA/SFTP credential file parsing

The sibling ``analyze_health_check_results.py`` turns diag_report.json into the
runs/checks CSVs.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from utils.diag_execution import run_diag_subprocess
from utils.jira_client import (
    _build_failure_body,
    _build_recovery_body,
    add_comment_to_jira,
    artifact_upload_name,
    attach_files_to_jira,
    attach_log_to_jira,
    create_jira_ticket,
    find_open_ticket_for_node,
    transition_jira_ticket,
    update_jira_versions,
)
from utils.node_recovery import REBOOT_CAP, reboot_and_requeue, should_reboot, slurm_restart_count
from utils.report import has_actionable_failure, normalize_health_report
from utils.secrets_loader import load_jira_secrets, load_sftp_secrets
from utils.sftp_upload import upload_csv_sftp
from utils.system_info import collect_version_info
from utils.telemetry import (
    aggregate_telemetry_for_csv,
    collect_prometheus_metrics,
    format_prometheus_metrics,
    telemetry_port_for_launch_mode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Deployment this runner is executing under. The runner is shared across the
# bare-Slurm (exabox) and Kubernetes (tt-orchestration) health checks; the two
# spots that genuinely differ — how the diag suite is launched and how the SFTP
# key is loaded — dispatch on this. Default preserves the Slurm behavior.
LAUNCH_MODES = ("slurm", "orchestration")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fabric system health check on a node.")

    p.add_argument("node", help="Target node hostname")
    p.add_argument(
        "--launch-mode",
        choices=LAUNCH_MODES,
        default="slurm",
        help=(
            "Deployment mode. 'slurm' (default) runs diag_runner.py directly and "
            "loads creds from env + files; 'orchestration' (k8s) runs run_diag.sh "
            "and loads the SFTP key from SFTP_KEY_PATH."
        ),
    )
    p.add_argument("--log-dir", required=True, help="Directory for logs and credentials")
    p.add_argument(
        "--tier",
        default="light",
        choices=["light", "medium", "deploy"],
        help="Diagnostic tier passed to the diag suite",
    )
    p.add_argument(
        "--results-dir",
        default="",
        help=(
            "Directory for the JSON report and per-test logs. " "Defaults to <log-dir>/<node>-<slurm_job_id>-results."
        ),
    )
    p.add_argument(
        "--diag-runner",
        default="",
        help="Path to diag_runner.py (slurm mode; defaults to the sibling health-check suite script).",
    )
    p.add_argument(
        "--tt-metal-path",
        default="",
        help="tt-metal repo root (defaults to $TT_METAL_HOME / the diag suite's own default).",
    )
    p.add_argument("--timeout-minutes", type=int, default=30, help="Timeout for the diag run")
    p.add_argument(
        "--telemetry-port",
        type=int,
        default=None,
        help=(
            "Port of the local tt-telemetry /metrics endpoint. Defaults to the "
            "launch mode's port (slurm: 8080, orchestration: 18080); set this when "
            "the endpoint has been moved to avoid a host-port clash."
        ),
    )

    jira = p.add_argument_group("JIRA integration")
    jira.add_argument("--jira-base-url", default="", help="JIRA REST API base URL")
    jira.add_argument("--jira-site-url", default="", help="JIRA site URL for browse links")
    jira.add_argument("--jira-project-key", default="", help="JIRA project key")
    jira.add_argument("--jira-issue-type", default="Bug", help="JIRA issue type")
    jira.add_argument(
        "--jira-resolve-transition",
        default="Done",
        help="Transition used to close a node's open ticket when it recovers "
        "(falls back to any done-category transition if not found)",
    )
    jira.add_argument(
        "--grafana-base-url",
        default="https://grafana.it.aws.tenstorrent.com",
        help="Grafana base URL for the node's tt-telemetry dashboard link in failure "
        "tickets; empty string omits the link",
    )

    p.add_argument(
        "--create-jira",
        choices=("true", "false"),
        default="true",
        help="Create/update a JIRA ticket on failure",
    )
    p.add_argument(
        "--upload-sftp",
        choices=("true", "false"),
        default="true",
        help="Upload runs/checks CSVs to the Data-team SFTP endpoint",
    )
    p.add_argument(
        "--cleanup",
        choices=("true", "false"),
        default="true",
        help="Delete the log and results directory after the run",
    )
    p.add_argument(
        "--reboot-on-failure",
        choices=("true", "false"),
        default="true",
        help="Reboot the node once and let Slurm rerun the suite on failure "
        "(slurm launch mode only; ignored under orchestration)",
    )
    p.add_argument(
        "--exclude",
        choices=("true", "false"),
        default="false",
        help="Run the full workflow (JIRA + SFTP upload) but flag the results discard=1 so "
        "they are excluded from the production dashboards. For dev end-to-end testing.",
    )
    p.add_argument(
        "--exclude-reason",
        default="",
        help="Reason recorded in discard_reason when --exclude is true (defaults to 'dev-run').",
    )

    args = p.parse_args()
    # node is used to build filesystem paths (log/results dirs); constrain it to a
    # hostname to prevent path traversal.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", args.node):
        p.error(f"invalid node name: {args.node!r}")
    args.create_jira = args.create_jira == "true"
    args.upload_sftp = args.upload_sftp == "true"
    args.cleanup = args.cleanup == "true"
    args.reboot_on_failure = args.reboot_on_failure == "true"
    args.exclude = args.exclude == "true"
    return args


# ---------------------------------------------------------------------------
# CSV analysis
# ---------------------------------------------------------------------------


def run_csv_analysis(
    json_report: Path | None,
    node: str,
    slurm_job_id: str,
    ticket_key: str | None,
    versions: dict[str, str],
    telemetry: dict | None = None,
    discard: int | None = None,
    discard_reason: str | None = None,
    run_id_suffix: str = "",
) -> str | None:
    """Translate the diag_report.json into the runs/checks CSVs for Superset.

    The report is normalized (same normalize_health_report used for the verdict)
    so the CSV verdict reflects post-reset state. Fail-closed: if the report is
    missing/unparseable the analyzer still emits an ERROR runs.csv. Returns the
    path to the CSV output directory.
    """

    csv_output_dir = tempfile.mkdtemp(prefix="health-check-csv-")

    log.info("Producing CSV analysis results ...")
    try:
        from analyze_health_check_results import analyze

        report = None
        if json_report and json_report.is_file():
            try:
                report = normalize_health_report(json.loads(json_report.read_text()))
            except (OSError, ValueError) as exc:
                log.warning("Could not read %s for CSV analysis: %s", json_report, exc)

        analyze(
            report=report,
            csv_output_dir=csv_output_dir,
            hostname=node,
            slurm_job_id=slurm_job_id,
            jira_ticket=ticket_key or "",
            versions={
                "tt_smi_version": _clean_version(versions.get("tt_smi")),
                "tt_kmd_version": _clean_version(versions.get("tt_kmd")),
                "fw_bundle_version": _clean_version(versions.get("fw_bundle")),
            },
            telemetry=telemetry,
            discard=discard,
            discard_reason=discard_reason,
            run_id_suffix=run_id_suffix,
        )
    except Exception as exc:
        log.warning("CSV analysis failed: %s", exc)

    return csv_output_dir


def _clean_version(value: str | None) -> str:
    """Map the collector's "N/A" sentinel (and None) to empty string so it
    doesn't surface as a real version bucket in the dashboards."""
    return "" if value in (None, "", "N/A") else value


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def remove_path(base: str, target: str) -> None:
    """Delete target only if it canonicalizes to a strict child of base.

    Rejecting ``target_real == base_real`` is deliberate: commonpath() of two
    identical paths is that path, so without the equality guard a caller passing
    target == base (e.g. --results-dir equal to --log-dir) would recursively
    delete the entire base directory, including credentials and unrelated runs.
    """
    base_real = os.path.realpath(base)
    target_real = os.path.realpath(target)
    if target_real == base_real or os.path.commonpath([base_real, target_real]) != base_real:
        log.warning("Refusing to delete %s: not a child of %s", target_real, base_real)
        return
    if os.path.isdir(target_real):
        shutil.rmtree(target_real, ignore_errors=True)
    else:
        Path(target_real).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()

    node = args.node
    log_dir = args.log_dir
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    log_file = Path(log_dir) / f"{node}-{slurm_job_id}.log"
    timeout_seconds = args.timeout_minutes * 60

    launch_mode = args.launch_mode
    jira_bearer_token = load_jira_secrets(log_dir, launch_mode)
    sftp_user, sftp_host = load_sftp_secrets(log_dir, launch_mode)

    # Dev end-to-end runs: keep the full workflow but flag the results so they
    # never land in the fleet dashboards.
    exclude_reason = (args.exclude_reason or "dev-run") if args.exclude else None
    if args.exclude:
        log.info(
            "Run marked --exclude: results will upload but are flagged discard=1 (reason: %s)",
            exclude_reason,
        )

    # Collect version info
    versions = collect_version_info()
    version_header = (
        "--- version info ---\n"
        f"tt-smi version: {versions['tt_smi']}\n"
        f"tt-kmd version: {versions['tt_kmd']}\n"
        f"fw_bundle_version={versions['fw_bundle']}\n"
        "--- fabric system health check ---\n"
    )
    print(version_header, flush=True)

    # Collect Prometheus metrics from local telemetry endpoint. The port defaults
    # per deployment and --telemetry-port overrides it (see
    # telemetry_port_for_launch_mode).
    prom_metrics = collect_prometheus_metrics(port=telemetry_port_for_launch_mode(launch_mode, args.telemetry_port))
    prom_output = ""
    if prom_metrics:
        prom_output = format_prometheus_metrics(prom_metrics)
        print(prom_output, flush=True)
    else:
        log.info("No Prometheus metrics collected")
    # Flatten the collected families for the CSV verdict. Without this the
    # analyzer only gets the log-formatted string and every run reports
    # telemetry_available=0 to Superset despite a successful collection.
    telemetry_summary = aggregate_telemetry_for_csv(prom_metrics)

    # Run the diag suite as a subprocess. It writes its JSON report + per-test
    # logs straight into results_dir on the host filesystem.
    results_dir = Path(args.results_dir or (Path(log_dir) / f"{node}-{slurm_job_id}-results"))
    exit_code, test_output, artifacts_dir = run_diag_subprocess(
        tier=args.tier,
        timeout_seconds=timeout_seconds,
        results_dir=results_dir,
        launch_mode=launch_mode,
        diag_runner=Path(args.diag_runner) if args.diag_runner else None,
        tt_metal_path=Path(args.tt_metal_path) if args.tt_metal_path else None,
    )

    full_output = version_header + prom_output + "\n" + test_output
    print(test_output, flush=True)

    # Write console output to log file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(full_output)

    # Surface the structured artifacts (JSON report + gtest logs) for downstream
    # JIRA attachment / SFTP upload (handled by the analysis step, to follow).
    json_report = None
    logs_subdir = None
    if artifacts_dir and artifacts_dir.is_dir():
        json_report = artifacts_dir / "diag_report.json"
        logs_subdir = artifacts_dir / "logs"
        log.info(
            "JSON report: %s",
            json_report if json_report.is_file() else "MISSING (not produced)",
        )
        if logs_subdir.is_dir():
            log.info(
                "Per-test logs (%d file(s)) in %s",
                len(list(logs_subdir.glob("*.log"))),
                logs_subdir,
            )
    else:
        log.warning("No results directory present (diag run may have failed to start)")

    # Don't fail/ticket if nothing actionable remains post-reset (pre-reset and
    # excluded-check FAILs cleared after tt-smi reset).
    effective_code = exit_code
    if exit_code != 0 and json_report and json_report.is_file():
        try:
            report = json.loads(json_report.read_text())
        except (OSError, ValueError) as exc:
            report = None
            log.warning("Could not read %s to classify failure: %s", json_report, exc)
        if report is not None and not has_actionable_failure(normalize_health_report(report)):
            log.info(
                "No actionable failure after dropping pre-reset snapshot / excluded "
                "checks; treating run as non-failing and skipping JIRA ticket."
            )
            effective_code = 0

    # Reboot-and-rerun self-heal. This is the one behavior that can't be shared
    # between the deployments: it drives Slurm (scontrol reboot + requeue), which
    # orchestration has no equivalent for, so there we say so and ticket instead.
    restart_count = slurm_restart_count()
    # Set only when self-heal tried but couldn't reboot/requeue; surfaced on the ticket.
    reboot_failure: str | None = None
    if effective_code != 0 and args.reboot_on_failure and launch_mode != "slurm":
        log.warning(
            "--reboot-on-failure is not supported in %s launch mode (reboot-and-requeue "
            "needs Slurm); proceeding to JIRA ticketing",
            launch_mode,
        )
    elif should_reboot(
        exit_code=effective_code,
        enabled=args.reboot_on_failure,
        slurm_job_id=slurm_job_id,
        restart_count=restart_count,
        cap=REBOOT_CAP,
    ):
        log.info(
            "Test failed (exit %d); rebooting and requeuing for a clean rerun (restart_count=%d, cap=%d)",
            effective_code,
            restart_count,
            REBOOT_CAP,
        )
        # Upload the failing run as discard=1 (visible but out of fleet stats)
        # before rebooting; suffix avoids colliding with the post-reboot row.
        pre_reboot_csv = run_csv_analysis(
            json_report=json_report,
            node=node,
            slurm_job_id=slurm_job_id,
            ticket_key=None,
            versions=versions,
            telemetry=telemetry_summary,
            discard=1,
            discard_reason="reboot_pending",
            run_id_suffix="-pre-reboot",
        )
        if pre_reboot_csv and args.upload_sftp and sftp_user and sftp_host:
            upload_csv_sftp(pre_reboot_csv, sftp_user, sftp_host, log_dir=log_dir, launch_mode=launch_mode)
        if args.cleanup and pre_reboot_csv:
            remove_path(tempfile.gettempdir(), pre_reboot_csv)
        reboot_failure = reboot_and_requeue(node, slurm_job_id)
        if reboot_failure is None:
            log.info("Reboot armed and job requeued; exiting so the node reboots and reruns")
            return effective_code
        log.error(
            "Self-heal reboot FAILED (%s); node was NOT rebooted or requeued, " "proceeding to JIRA ticketing",
            reboot_failure,
        )

    # JIRA ticket creation (failure only)
    ticket_key = None
    if effective_code != 0:
        log.info("Test failed with exit code %d", exit_code)

        if not args.create_jira:
            log.info("JIRA ticket creation disabled by config")
        elif not jira_bearer_token:
            log.warning("JIRA credentials not configured, skipping ticket creation")
        elif not args.jira_base_url:
            log.warning("JIRA base URL not configured, skipping ticket creation")
        else:
            # Files this run will reference with [^name] so JIRA renders them
            # inline with the comment/description.
            result_files = []
            if artifacts_dir and artifacts_dir.is_dir():
                result_files = sorted(p for p in artifacts_dir.rglob("*") if p.is_file())
            attachment_names = [f"{node}-{slurm_job_id}.log"] + [
                artifact_upload_name(p, slurm_job_id) for p in result_files
            ]

            existing_key = find_open_ticket_for_node(
                node=node,
                jira_base_url=args.jira_base_url,
                jira_project_key=args.jira_project_key,
                jira_bearer_token=jira_bearer_token,
            )

            if existing_key:
                # Recurring failure while a ticket is still open: append the new
                # failure as a comment instead of opening a duplicate ticket.
                comment_body = (
                    f"Fabric System Health Check failed again on node {node} "
                    f"(recurring failure).\n\n"
                    + _build_failure_body(
                        node=node,
                        slurm_job_id=slurm_job_id,
                        exit_code=exit_code,
                        versions=versions,
                        telemetry_summary=prom_output,
                        test_output=full_output,
                        attachment_names=attachment_names,
                        restart_count=restart_count,
                        reboot_failure=reboot_failure,
                        grafana_base_url=args.grafana_base_url,
                    )
                )
                add_comment_to_jira(
                    ticket_key=existing_key,
                    body=comment_body,
                    jira_base_url=args.jira_base_url,
                    jira_bearer_token=jira_bearer_token,
                )
                # Keep the mandatory version fields current with this run.
                update_jira_versions(
                    ticket_key=existing_key,
                    versions=versions,
                    jira_base_url=args.jira_base_url,
                    jira_bearer_token=jira_bearer_token,
                )
                ticket_key = existing_key
            else:
                ticket_key = create_jira_ticket(
                    node=node,
                    slurm_job_id=slurm_job_id,
                    exit_code=exit_code,
                    test_output=full_output,
                    jira_base_url=args.jira_base_url,
                    jira_site_url=args.jira_site_url,
                    jira_project_key=args.jira_project_key,
                    jira_issue_type=args.jira_issue_type,
                    jira_bearer_token=jira_bearer_token,
                    versions=versions,
                    telemetry_summary=prom_output,
                    attachment_names=attachment_names,
                    restart_count=restart_count,
                    reboot_failure=reboot_failure,
                    grafana_base_url=args.grafana_base_url,
                )
                if ticket_key:
                    transition_jira_ticket(
                        ticket_key=ticket_key,
                        jira_base_url=args.jira_base_url,
                        jira_bearer_token=jira_bearer_token,
                    )

            # Upload under the names referenced above.
            if ticket_key:
                attached = attach_log_to_jira(
                    ticket_key=ticket_key,
                    test_output=full_output,
                    node=node,
                    slurm_job_id=slurm_job_id,
                    jira_base_url=args.jira_base_url,
                    jira_bearer_token=jira_bearer_token,
                )
                if attached and args.cleanup:
                    log.info("Cleaned up log file: %s", log_file)
                    remove_path(log_dir, str(log_file))

                if result_files:
                    attach_files_to_jira(
                        ticket_key=ticket_key,
                        files=result_files,
                        slurm_job_id=slurm_job_id,
                        jira_base_url=args.jira_base_url,
                        jira_bearer_token=jira_bearer_token,
                    )
    else:
        # Clean run: if this node has an open failure ticket, it just recovered
        # — close it so machines that are fine don't accumulate stale tickets.
        if args.create_jira and jira_bearer_token and args.jira_base_url:
            recovered_key = find_open_ticket_for_node(
                node=node,
                jira_base_url=args.jira_base_url,
                jira_project_key=args.jira_project_key,
                jira_bearer_token=jira_bearer_token,
            )
            if recovered_key:
                log.info(
                    "Node %s passed with open ticket %s; closing it (recovered)",
                    node,
                    recovered_key,
                )
                # Attach this passing run's artifacts so the closed ticket carries
                # the recovery evidence (telemetry, Grafana, CSVs) next to the logs.
                result_files = []
                if artifacts_dir and artifacts_dir.is_dir():
                    result_files = sorted(p for p in artifacts_dir.rglob("*") if p.is_file())
                attachment_names = [f"{node}-{slurm_job_id}.log"] + [
                    artifact_upload_name(p, slurm_job_id) for p in result_files
                ]
                add_comment_to_jira(
                    ticket_key=recovered_key,
                    body=(
                        f"Fabric System Health Check passed on node {node} "
                        f"(Slurm job {slurm_job_id}); the node has recovered. "
                        f"Closing this ticket automatically.\n\n"
                        + _build_recovery_body(
                            node=node,
                            slurm_job_id=slurm_job_id,
                            versions=versions,
                            telemetry_summary=prom_output,
                            test_output=full_output,
                            attachment_names=attachment_names,
                            restart_count=restart_count,
                            grafana_base_url=args.grafana_base_url,
                        )
                    ),
                    jira_base_url=args.jira_base_url,
                    jira_bearer_token=jira_bearer_token,
                )
                attach_log_to_jira(
                    ticket_key=recovered_key,
                    test_output=full_output,
                    node=node,
                    slurm_job_id=slurm_job_id,
                    jira_base_url=args.jira_base_url,
                    jira_bearer_token=jira_bearer_token,
                )
                if result_files:
                    attach_files_to_jira(
                        ticket_key=recovered_key,
                        files=result_files,
                        slurm_job_id=slurm_job_id,
                        jira_base_url=args.jira_base_url,
                        jira_bearer_token=jira_bearer_token,
                    )
                transition_jira_ticket(
                    ticket_key=recovered_key,
                    jira_base_url=args.jira_base_url,
                    jira_bearer_token=jira_bearer_token,
                    target_transition=args.jira_resolve_transition,
                    fallback_to_done=True,
                )

    # CSV analysis (always)
    csv_dir = run_csv_analysis(
        json_report=json_report,
        node=node,
        slurm_job_id=slurm_job_id,
        ticket_key=ticket_key,
        versions=versions,
        telemetry=telemetry_summary,
        discard=1 if args.exclude else None,
        discard_reason=exclude_reason,
    )

    # SFTP upload
    if csv_dir and args.upload_sftp:
        if sftp_user and sftp_host:
            upload_csv_sftp(csv_dir, sftp_user, sftp_host, log_dir=log_dir, launch_mode=launch_mode)
        else:
            log.warning("SFTP credentials not configured, skipping CSV upload")
    elif csv_dir:
        log.info("SFTP upload disabled by config")

    if args.cleanup:
        if csv_dir:
            remove_path(tempfile.gettempdir(), csv_dir)
        if effective_code == 0:
            remove_path(log_dir, str(log_file))
        remove_path(log_dir, str(results_dir))

    return effective_code


if __name__ == "__main__":
    sys.exit(main())
