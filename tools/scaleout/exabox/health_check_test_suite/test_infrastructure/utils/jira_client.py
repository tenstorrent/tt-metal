# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""JIRA integration for the fabric system health check.

Creates/updates a Bug ticket per node on failure, appends recurring-failure
comments while a ticket stays open, closes tickets when a node recovers, and
attaches the run log + result artifacts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

from .grafana import telemetry_dashboard_url

log = logging.getLogger(__name__)


def _telemetry_section(telemetry_summary: str) -> str:
    """The ``{noformat}`` telemetry block, shared by failure and recovery bodies."""
    if not telemetry_summary:
        return ""
    return f"\n*Telemetry Metrics:*\n{{noformat}}\n{telemetry_summary}\n{{noformat}}\n"


def _grafana_section(*, node: str, when: datetime, grafana_base_url: str) -> str:
    """A Grafana deep link time-boxed around ``when`` (fail or pass time)."""
    if not grafana_base_url:
        return ""
    url = telemetry_dashboard_url(base_url=grafana_base_url, node=node, fail_time=when)
    return f"\n*Telemetry dashboard:* [Grafana {node}|{url}]\n"


def _attachment_section(attachment_names: list[str] | None) -> str:
    """Inline ``[^name]`` links so JIRA renders the attached artifacts."""
    if not attachment_names:
        return ""
    links = "\n".join(f"[^{name}]" for name in attachment_names)
    return f"\n*Attachments:*\n{links}\n"


def _build_failure_body(
    *,
    node: str,
    slurm_job_id: str,
    exit_code: int,
    versions: dict[str, str],
    telemetry_summary: str,
    test_output: str,
    attachment_names: list[str] | None = None,
    restart_count: int = 0,
    reboot_failure: str | None = None,
    grafana_base_url: str = "",
) -> str:
    """Build the shared failure detail block used by both a new ticket's
    description and a recurring-failure comment, so the two never drift."""
    fail_time = datetime.now(timezone.utc)
    fail_date = fail_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    log_tail = test_output[-4096:]

    reboot_line = ""
    if restart_count > 0:
        reboot_line = f"*Reboot recovery:* failure persisted after {restart_count} reboot(s)\n"

    # A failed self-heal reboot means the run never got its clean rerun; flag it
    # with the error so it isn't read as a normal single-run failure.
    reboot_failure_line = ""
    if reboot_failure:
        reboot_failure_line = (
            f"*Self-heal reboot: FAILED* - node was NOT rebooted or requeued. " f"Reason: {{{{{reboot_failure}}}}}\n"
        )

    return (
        f"*Node:* {node}\n"
        f"*Date:* {fail_date}\n"
        f"*Slurm Job ID:* {slurm_job_id}\n"
        f"*Exit Code:* {exit_code}\n"
        f"{reboot_line}"
        f"{reboot_failure_line}"
        f"*TT-SMI Version:* {versions['tt_smi']}\n"
        f"*TT-KMD Version:* {versions['tt_kmd']}\n"
        f"*Firmware Version:* {versions['fw_bundle']}\n"
        f"{_telemetry_section(telemetry_summary)}"
        f"{_grafana_section(node=node, when=fail_time, grafana_base_url=grafana_base_url)}"
        f"{_attachment_section(attachment_names)}\n"
        f"*Last lines of output:*\n"
        f"{{noformat}}\n{log_tail}\n{{noformat}}"
    )


def _build_recovery_body(
    *,
    node: str,
    slurm_job_id: str,
    versions: dict[str, str],
    telemetry_summary: str,
    test_output: str,
    attachment_names: list[str] | None = None,
    restart_count: int = 0,
    grafana_base_url: str = "",
) -> str:
    """Detail block for the auto-close comment when a node recovers.

    Same telemetry / Grafana / attachment / log sections as a failure so the
    closed ticket carries the passing evidence, framed as a PASS instead of a
    failure.
    """
    pass_time = datetime.now(timezone.utc)
    pass_date = pass_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    log_tail = test_output[-4096:]

    recovery_line = ""
    if restart_count > 0:
        recovery_line = f"*Recovered after:* {restart_count} reboot(s)\n"

    return (
        f"*Node:* {node}\n"
        f"*Date:* {pass_date}\n"
        f"*Slurm Job ID:* {slurm_job_id}\n"
        f"*Result:* PASS\n"
        f"{recovery_line}"
        f"*TT-SMI Version:* {versions['tt_smi']}\n"
        f"*TT-KMD Version:* {versions['tt_kmd']}\n"
        f"*Firmware Version:* {versions['fw_bundle']}\n"
        f"{_telemetry_section(telemetry_summary)}"
        f"{_grafana_section(node=node, when=pass_time, grafana_base_url=grafana_base_url)}"
        f"{_attachment_section(attachment_names)}\n"
        f"*Last lines of output:*\n"
        f"{{noformat}}\n{log_tail}\n{{noformat}}"
    )


def find_open_ticket_for_node(
    *,
    node: str,
    jira_base_url: str,
    jira_project_key: str,
    jira_bearer_token: str,
) -> str | None:
    """Return the key of the most recent OPEN JIRA ticket for this node, or None.

    Subsequent failures append to this ticket instead of spawning duplicates.
    The JQL narrows on the failure-ticket summary phrase and open status; the
    exact summary is then re-checked in Python because JIRA text search tokenizes
    hostnames on hyphens (so bh-glx-b09u08 could otherwise fuzzy-match ...u02).
    Any API/parse error returns None so we fall back to creating a ticket and
    never lose the failure signal.
    """
    expected_summary = f"[Fabric Health Check] Test failed on {node}"
    jql = (
        f"project = {jira_project_key} "
        f'AND labels = "fabric-health-check" '
        f'AND summary ~ "\\"Test failed on {node}\\"" '
        f"AND statusCategory != Done "
        f"ORDER BY created DESC"
    )
    # Enhanced search endpoint; the legacy GET /rest/api/2/search was removed by
    # Atlassian (returns 410 Gone). /search/jql takes the query in a POST body.
    url = f"{jira_base_url}/rest/api/2/search/jql"
    headers = {
        "Authorization": f"Bearer {jira_bearer_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {"jql": jql, "fields": ["summary"], "maxResults": 20}

    log.info("Searching for an open JIRA ticket for %s ...", node)
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        issues = resp.json().get("issues", [])
    except (requests.RequestException, ValueError) as exc:
        log.warning("JIRA search failed (%s); will create a new ticket", exc)
        return None

    for issue in issues:
        if issue.get("fields", {}).get("summary", "") == expected_summary:
            key = issue.get("key")
            log.info("Found open ticket %s for %s", key, node)
            return key
    log.info("No open ticket found for %s", node)
    return None


def add_comment_to_jira(
    *,
    ticket_key: str,
    body: str,
    jira_base_url: str,
    jira_bearer_token: str,
) -> bool:
    """Add a comment to an existing JIRA ticket. Returns True on success."""
    url = f"{jira_base_url}/rest/api/2/issue/{ticket_key}/comment"
    headers = {
        "Authorization": f"Bearer {jira_bearer_token}",
        "Content-Type": "application/json",
    }

    log.info("Adding recurring-failure comment to %s ...", ticket_key)
    try:
        resp = requests.post(url, headers=headers, json={"body": body}, timeout=30)
    except requests.RequestException as exc:
        log.warning("Failed to comment on %s: %s", ticket_key, exc)
        return False

    if not resp.ok:
        log.warning(
            "Failed to comment on %s (HTTP %d): %s",
            ticket_key,
            resp.status_code,
            resp.text,
        )
        return False

    log.info("Comment added to %s", ticket_key)
    return True


def _version_fields(versions: dict[str, str]) -> dict:
    """The mandatory version custom fields, shared by ticket create and update so
    the two never drift: Firmware / TT-SMI / KMD versions."""
    return {
        "customfield_16168": versions["fw_bundle"],  # Firmware Version
        "customfield_16169": versions["tt_smi"],  # TT-SMI Version
        "customfield_16170": versions["tt_kmd"],  # KMD Version
    }


def update_jira_versions(
    *,
    ticket_key: str,
    versions: dict[str, str],
    jira_base_url: str,
    jira_bearer_token: str,
) -> bool:
    """Refresh the mandatory version fields on an existing ticket so they reflect
    the latest run's firmware/tt-smi/kmd versions. Returns True on success."""
    url = f"{jira_base_url}/rest/api/2/issue/{ticket_key}"
    headers = {
        "Authorization": f"Bearer {jira_bearer_token}",
        "Content-Type": "application/json",
    }

    log.info("Updating version fields on %s ...", ticket_key)
    try:
        resp = requests.put(url, headers=headers, json={"fields": _version_fields(versions)}, timeout=30)
    except requests.RequestException as exc:
        log.warning("Failed to update version fields on %s: %s", ticket_key, exc)
        return False

    if not resp.ok:
        log.warning(
            "Failed to update version fields on %s (HTTP %d): %s",
            ticket_key,
            resp.status_code,
            resp.text,
        )
        return False

    log.info("Updated version fields on %s", ticket_key)
    return True


def create_jira_ticket(
    *,
    node: str,
    slurm_job_id: str,
    exit_code: int,
    test_output: str,
    jira_base_url: str,
    jira_site_url: str,
    jira_project_key: str,
    jira_issue_type: str,
    jira_bearer_token: str,
    versions: dict[str, str],
    telemetry_summary: str = "",
    attachment_names: list[str] | None = None,
    restart_count: int = 0,
    reboot_failure: str | None = None,
    grafana_base_url: str = "",
) -> str | None:
    """Create a JIRA ticket for a failed health check. Returns ticket key or None."""

    description = f"Fabric System Health Check failed on node {node}.\n\n" + _build_failure_body(
        node=node,
        slurm_job_id=slurm_job_id,
        exit_code=exit_code,
        versions=versions,
        telemetry_summary=telemetry_summary,
        test_output=test_output,
        attachment_names=attachment_names,
        restart_count=restart_count,
        reboot_failure=reboot_failure,
        grafana_base_url=grafana_base_url,
    )

    payload = {
        "fields": {
            "project": {"key": jira_project_key},
            "summary": f"[Fabric Health Check] Test failed on {node}",
            "description": description,
            "issuetype": {"name": jira_issue_type},
            "labels": ["fabric-health-check"],
            "customfield_13389": {"id": "17418"},
            **_version_fields(versions),
        }
    }

    url = f"{jira_base_url}/rest/api/2/issue"
    headers = {
        "Authorization": f"Bearer {jira_bearer_token}",
        "Content-Type": "application/json",
    }

    log.info("Creating JIRA ticket at %s ...", url)
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        log.warning("Failed to create JIRA ticket: %s", exc)
        return None

    log.info("JIRA API response: HTTP %d", resp.status_code)
    if not resp.ok:
        log.warning("Failed to create JIRA ticket (HTTP %d): %s", resp.status_code, resp.text)
        return None

    try:
        ticket_key = resp.json().get("key")
    except (ValueError, KeyError):
        log.warning("JIRA response is not valid JSON: %s", resp.text[:256])
        return None

    if ticket_key:
        log.info("JIRA ticket created: %s/browse/%s", jira_site_url, ticket_key)
    return ticket_key


def transition_jira_ticket(
    ticket_key: str,
    jira_base_url: str,
    jira_bearer_token: str,
    target_transition: str = "Health Status",
    fallback_to_done: bool = False,
) -> None:
    """Transition a JIRA ticket to the target status.

    Matches the transition by name. When ``fallback_to_done`` is set and no
    transition matches ``target_transition``, any available transition whose
    destination is in the "done" status category is used instead — this keeps
    ticket-closing working across workflow variants where the closing
    transition isn't named exactly as configured.
    """
    headers = {"Authorization": f"Bearer {jira_bearer_token}"}

    url = f"{jira_base_url}/rest/api/2/issue/{ticket_key}/transitions"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Could not fetch transitions for %s: %s", ticket_key, exc)
        return

    try:
        transitions = resp.json().get("transitions", [])
    except ValueError as exc:
        log.warning("Could not parse transitions for %s: %s", ticket_key, exc)
        return
    transition_id = next((t["id"] for t in transitions if t["name"] == target_transition), None)
    if not transition_id and fallback_to_done:
        transition_id = next(
            (t["id"] for t in transitions if t.get("to", {}).get("statusCategory", {}).get("key") == "done"),
            None,
        )
        if transition_id:
            log.info(
                "Transition '%s' not found for %s; falling back to a done-category " "transition",
                target_transition,
                ticket_key,
            )
    if not transition_id:
        log.warning("Could not find '%s' transition for %s", target_transition, ticket_key)
        return

    log.info("Transitioning %s to '%s' ...", ticket_key, target_transition)
    try:
        resp = requests.post(
            url,
            headers={**headers, "Content-Type": "application/json"},
            json={"transition": {"id": transition_id}},
            timeout=15,
        )
        if resp.ok:
            log.info("Transitioned %s to '%s'", ticket_key, target_transition)
        else:
            log.warning("Failed to transition ticket (HTTP %d)", resp.status_code)
    except requests.RequestException as exc:
        log.warning("Failed to transition ticket: %s", exc)


def attach_log_to_jira(
    ticket_key: str,
    test_output: str,
    node: str,
    slurm_job_id: str,
    jira_base_url: str,
    jira_bearer_token: str,
) -> str | None:
    """Attach the test log to a JIRA ticket. Returns the filename, or None."""
    if not test_output:
        return None

    url = f"{jira_base_url}/rest/api/2/issue/{ticket_key}/attachments"
    headers = {
        "Authorization": f"Bearer {jira_bearer_token}",
        "X-Atlassian-Token": "no-check",
    }
    filename = f"{node}-{slurm_job_id}.log"

    log.info("Attaching log file to %s ...", ticket_key)
    try:
        resp = requests.post(
            url,
            headers=headers,
            files={"file": (filename, test_output.encode("utf-8"))},
            timeout=30,
        )
        if resp.ok:
            log.info("Log file attached to %s", ticket_key)
            return filename
        else:
            log.warning("Failed to attach log file (HTTP %d): %s", resp.status_code, resp.text)
            return None
    except requests.RequestException as exc:
        log.warning("Failed to attach log file: %s", exc)
        return None


def artifact_upload_name(path: Path, slurm_job_id: str) -> str:
    """Suffix an artifact's name with the Slurm job id to keep it unique per run."""
    return f"{path.stem}-{slurm_job_id}{path.suffix}"


def attach_files_to_jira(
    ticket_key: str,
    files: list[Path],
    slurm_job_id: str,
    jira_base_url: str,
    jira_bearer_token: str,
) -> list[str]:
    """Attach files under job-id-unique names. Returns the uploaded names."""
    if not files:
        return []

    url = f"{jira_base_url}/rest/api/2/issue/{ticket_key}/attachments"
    headers = {
        "Authorization": f"Bearer {jira_bearer_token}",
        "X-Atlassian-Token": "no-check",
    }

    uploaded = []
    for path in files:
        name = artifact_upload_name(path, slurm_job_id)
        log.info("Attaching %s to %s ...", name, ticket_key)
        try:
            with path.open("rb") as fh:
                resp = requests.post(
                    url,
                    headers=headers,
                    files={"file": (name, fh)},
                    timeout=60,
                )
            if resp.ok:
                uploaded.append(name)
            else:
                log.warning(
                    "Failed to attach %s (HTTP %d): %s",
                    name,
                    resp.status_code,
                    resp.text,
                )
        except (OSError, requests.RequestException) as exc:
            log.warning("Failed to attach %s: %s", name, exc)

    log.info("Attached %d/%d result file(s) to %s", len(uploaded), len(files), ticket_key)
    return uploaded
