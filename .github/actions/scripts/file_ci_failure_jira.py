#!/usr/bin/env python3
"""File a Jira issue for a CI failure described by a normalized payload.

The AI Summary action analyses logs and produces structured failure detail; this
adapter does routing, de-duplication, ticket creation and owner assignment. It
knows nothing about any particular workflow -- a monitored workflow publishes the
payload below and this decides where it goes (MINFRA-1611).

AI/IP release-test failures do NOT come through here: those are filed per test,
routed to their AIIPSW requirement, by file_rtl_sim_jira.py onto RELEASE.

Payload (JSON, one failure per object; a list files one issue per entry):
    repository, workflow, job          identity of the failing leg
    run_url, job_url                   links back to CI
    commit                             sha under test            (optional)
    test                               failing test path         (optional)
    error_message                      verbatim, quoted in the issue
    category, subcategory, error_layer AI classification         (optional)
    confidence                         0..1; below min_confidence is not filed
    root_cause, suggested_action       AI narrative              (optional)
    log_urls                           list of links             (optional)
    actionable                         explicit override         (optional)
    owner                              Jira accountId if known   (optional)

Environment:
  CI_FAILURE_PAYLOAD   path to the payload JSON                  (required)
  CI_FAILURE_ROUTING   routing config (default: ./ci_failure_routing.json)
  JIRA_BASE_URL / JIRA_USER_EMAIL / JIRA_API_TOKEN               (required)
  JIRA_DRY_RUN         print instead of calling Jira

Exits 0 when nothing is actionable -- reporting must never fail a pipeline.
"""
import hashlib
import json
import os
import re
import sys

from jira_client import _env, _truthy, file_issue

HERE = os.path.dirname(os.path.abspath(__file__))


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()


def load_payload(path):
    """Return a list of failure dicts; a bare object is treated as one failure."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("failures", [data])
    return [d for d in data if isinstance(d, dict)]


def is_actionable(failure, min_confidence):
    """Explicit actionable wins; otherwise require confidence at or above the floor.

    A missing confidence is treated as actionable: the AI declining to score is
    not evidence that a real failure should be dropped.
    """
    if "actionable" in failure:
        return bool(failure["actionable"])
    conf = failure.get("confidence")
    if conf is None:
        return True
    try:
        return float(conf) >= float(min_confidence)
    except (TypeError, ValueError):
        return True


def route_for(failure, routing):
    """First route whose stated fields all match; an omitted field is a wildcard."""
    for route in routing.get("routes", []):
        fields = [k for k in ("category", "subcategory", "error_layer", "workflow") if k in route]
        if all(str(failure.get(k, "")).lower() == str(route[k]).lower() for k in fields):
            merged = dict(routing.get("default", {}))
            merged.update({k: v for k, v in route.items() if not k.startswith("_")})
            return merged
    return dict(routing.get("default", {}))


def dedup_key(failure):
    """Stable identity for one recurring failure.

    Deliberately excludes run/job URLs and the commit: the same test failing the
    same way next week is the same problem, and must comment rather than open a
    duplicate.
    """
    parts = [
        failure.get("repository", ""),
        failure.get("workflow", ""),
        failure.get("job", ""),
        failure.get("test", ""),
        failure.get("category", ""),
        failure.get("subcategory", ""),
    ]
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    label = _slug(failure.get("test") or failure.get("job") or "ci-failure")[:60]
    return f"ci-failure:{label}:{digest}"


def summary_for(failure):
    what = failure.get("test") or failure.get("job") or "CI failure"
    where = failure.get("workflow") or failure.get("repository") or "CI"
    return f"{where}: {what} failed"[:250]


def description_for(failure):
    lines = ["An AI Summary classified this CI failure as actionable.\n"]
    for label, key in [
        ("Repository", "repository"),
        ("Workflow", "workflow"),
        ("Job", "job"),
        ("Test", "test"),
        ("Commit", "commit"),
        ("Category", "category"),
        ("Subcategory", "subcategory"),
        ("Error layer", "error_layer"),
        ("Confidence", "confidence"),
    ]:
        if failure.get(key) not in (None, ""):
            lines.append(f"{label + ':':14}{failure[key]}")
    for label, key in [("Root cause", "root_cause"), ("Suggested action", "suggested_action")]:
        if failure.get(key):
            lines.append(f"\n{label}:\n{failure[key]}")
    if failure.get("error_message"):
        lines.append(f"\nError:\n{failure['error_message']}")
    for label, key in [("Run", "run_url"), ("Job", "job_url")]:
        if failure.get(key):
            lines.append(f"{label + ':':14}{failure[key]}")
    for url in failure.get("log_urls") or []:
        lines.append(f"Log:          {url}")
    return "\n".join(lines) + "\n"


def main():
    payload_path = _env("CI_FAILURE_PAYLOAD", required=True)
    routing_path = _env("CI_FAILURE_ROUTING", os.path.join(HERE, "ci_failure_routing.json"))
    dry = _truthy(_env("JIRA_DRY_RUN"))

    if not os.path.isfile(payload_path):
        print(f"no payload at {payload_path}; nothing to file")
        return 0
    with open(routing_path) as f:
        routing = json.load(f)
    failures = load_payload(payload_path)
    print(f"payload carries {len(failures)} failure(s)")

    base = _env("JIRA_BASE_URL", required=True)
    email = _env("JIRA_USER_EMAIL", required=True)
    token = _env("JIRA_API_TOKEN", required=True)
    min_conf = routing.get("min_confidence", 0)

    filed = 0
    for failure in failures:
        what = failure.get("test") or failure.get("job") or "?"
        if not is_actionable(failure, min_conf):
            print(f"skip (not actionable, confidence={failure.get('confidence')}): {what}")
            continue
        route = route_for(failure, routing)
        labels = list(route.get("labels") or [])
        assignee = failure.get("owner") or route.get("triage_assignee")
        if not assignee:
            # Better an unassigned issue someone triages than a wrong assignee.
            labels.append("needs-owner")
        print(
            file_issue(
                base=base,
                email=email,
                token=token,
                project=route["project"],
                summary=summary_for(failure),
                issue_type=route.get("issue_type", "Bug"),
                description=description_for(failure),
                labels=labels,
                dedup_label=dedup_key(failure),
                assignee=assignee,
                dry_run=dry,
            )
        )
        filed += 1

    print(f"filed/updated {filed} issue(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
