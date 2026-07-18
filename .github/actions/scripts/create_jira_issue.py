#!/usr/bin/env python3
"""Create (or de-duplicate onto) a Jira issue to record a CI failure.

Generic building block: reads connection and issue details from the environment
and files an issue in a Jira Cloud project via the REST API. Idempotent -- if an
open issue already carries the dedup label, a comment is added instead of opening
a duplicate, so a persistently-failing pipeline does not spawn a new issue per run.

Environment:
  JIRA_BASE_URL     e.g. https://tenstorrent.atlassian.net   (required)
  JIRA_USER_EMAIL   Atlassian account email for the token     (required)
  JIRA_API_TOKEN    Atlassian API token, used as basic auth   (required)
  JIRA_PROJECT_KEY  project/board key to file under           (required)
  JIRA_SUMMARY      issue summary/title                       (required)
  JIRA_ISSUE_TYPE   issue type name                           (default: Bug)
  JIRA_DESCRIPTION  issue body, plain text (newlines kept)    (optional)
  JIRA_LABELS       comma-separated labels                    (optional)
  JIRA_DEDUP_LABEL  label used to detect an existing open issue for this failure
                    (optional; when set, comment-instead-of-create is enabled)
  JIRA_DRY_RUN      when truthy, print the payload and exit without calling Jira

Prints the resulting issue key and URL. Exit non-zero on API/config error.
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        sys.exit(f"error: {name} is required")
    return val


def _truthy(val):
    """Interpret an env-style string as a boolean (so "false"/"0"/"no" are false)."""
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def _api(base, email, token, method, path, body=None):
    url = f"{base.rstrip('/')}{path}"
    auth = base64.b64encode(f"{email}:{token}".encode()).decode()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Basic {auth}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"error: Jira {method} {path} -> {e.code} {e.reason}\n{e.read().decode(errors='replace')}")


def _adf(text):
    """Wrap plain text (newline-separated) in a minimal Atlassian Document Format doc."""
    paragraphs = [
        {"type": "paragraph", "content": [{"type": "text", "text": line}]}
        for line in (text or "").splitlines()
        if line.strip()
    ] or [{"type": "paragraph", "content": []}]
    return {"type": "doc", "version": 1, "content": paragraphs}


def _find_open_dupe(base, email, token, project, dedup_label):
    jql = f'project = "{project}" AND labels = "{dedup_label}" AND statusCategory != Done ORDER BY created DESC'
    path = "/rest/api/3/search/jql?" + urllib.parse.urlencode({"jql": jql, "maxResults": 1, "fields": "key"})
    issues = _api(base, email, token, "GET", path).get("issues", [])
    return issues[0]["key"] if issues else None


def file_issue(
    base, email, token, project, summary, issue_type="Bug", description="", labels=None, dedup_label="", dry_run=False
):
    """Create (or comment onto a de-duped) Jira issue.

    Returns a human-readable result string. When dedup_label is set and an open
    issue already carries it, a comment is added instead of opening a duplicate.
    """
    labels = list(labels or [])
    if dedup_label and dedup_label not in labels:
        labels.append(dedup_label)

    fields = {
        "project": {"key": project},
        "issuetype": {"name": issue_type},
        "summary": summary,
        "description": _adf(description),
    }
    if labels:
        fields["labels"] = labels

    if dry_run:
        return "DRY RUN -- would POST /rest/api/3/issue with fields:\n" + json.dumps({"fields": fields}, indent=2)

    if dedup_label:
        existing = _find_open_dupe(base, email, token, project, dedup_label)
        if existing:
            _api(
                base,
                email,
                token,
                "POST",
                f"/rest/api/3/issue/{existing}/comment",
                {"body": _adf(f"Recurred.\n{summary}\n{description}")},
            )
            return f"commented on existing {existing}: {base.rstrip('/')}/browse/{existing}"

    created = _api(base, email, token, "POST", "/rest/api/3/issue", {"fields": fields})
    key = created["key"]
    return f"created {key}: {base.rstrip('/')}/browse/{key}"


def main():
    labels = [l.strip() for l in (_env("JIRA_LABELS", "") or "").split(",") if l.strip()]
    print(
        file_issue(
            base=_env("JIRA_BASE_URL", required=True),
            email=_env("JIRA_USER_EMAIL", required=True),
            token=_env("JIRA_API_TOKEN", required=True),
            project=_env("JIRA_PROJECT_KEY", required=True),
            summary=_env("JIRA_SUMMARY", required=True),
            issue_type=_env("JIRA_ISSUE_TYPE", "Bug"),
            description=_env("JIRA_DESCRIPTION", ""),
            labels=labels,
            dedup_label=_env("JIRA_DEDUP_LABEL", ""),
            dry_run=_truthy(_env("JIRA_DRY_RUN")),
        )
    )


if __name__ == "__main__":
    main()
