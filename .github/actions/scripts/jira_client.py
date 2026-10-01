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

With JIRA_ACTION=close the script closes instead of filing: every open issue
carrying JIRA_DEDUP_LABEL is commented with JIRA_COMMENT (optional) and
transitioned to Done. JIRA_SUMMARY/JIRA_ISSUE_TYPE/etc. are not read.

Prints the resulting issue key and URL. Exit non-zero on API/config error.
"""
import base64
import json
import os
import re
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
    except urllib.error.URLError as e:
        # HTTPError subclasses URLError, so this is network-level only.
        sys.exit(f"error: cannot reach Jira ({method} {path}): {e.reason}")


# [label](url), or a bare URL. Producers write plain text; this is the one
# place that knows how Jira spells a hyperlink.
# Lazy label: job names carry their own brackets ("... [bh_sc16]"), so stop at
# the first "](" rather than the first "]".
_LINK_RE = re.compile(r"\[([^\n]+?)\]\((https?://[^\s)]+)\)|(https?://[^\s<>)\]]+)")


def _line_nodes(line):
    """Text nodes for one line, with links marked up so Jira renders them."""
    nodes, pos = [], 0
    for m in _LINK_RE.finditer(line):
        if m.start() > pos:
            nodes.append({"type": "text", "text": line[pos : m.start()]})
        end = m.end()
        if m.group(2):
            label, href = m.group(1), m.group(2)
        else:
            href = m.group(3)
            trimmed = href.rstrip(".,;:")  # trailing punctuation is prose, not URL
            end -= len(href) - len(trimmed)
            label = href = trimmed
        nodes.append({"type": "text", "text": label, "marks": [{"type": "link", "attrs": {"href": href}}]})
        pos = end
    if pos < len(line):
        nodes.append({"type": "text", "text": line[pos:]})
    return [n for n in nodes if n["text"]]


def _commit_link(sha, repo=None):
    """[short sha](commit url) when the repo is known, else the bare sha."""
    repo = repo if repo is not None else os.environ.get("GITHUB_REPOSITORY", "")
    if not sha or not repo:
        return sha or "unknown"
    return f"[{sha[:12]}](https://github.com/{repo}/commit/{sha})"


def _adf(text):
    """Wrap plain text (newline-separated) in a minimal Atlassian Document Format doc.

    Two markdown-ish forms are recognised so producers can stay plain-text:
    a "### " prefix renders as a level-3 heading, and runs of "- " lines as a
    bullet list. Anything else is a paragraph. Links render everywhere.
    """
    blocks = []
    bullets = None  # the open bulletList, while consecutive "- " lines continue it
    for line in (text or "").splitlines():
        if not line.strip():
            bullets = None
            continue
        if line.startswith("- "):
            if bullets is None:
                bullets = {"type": "bulletList", "content": []}
                blocks.append(bullets)
            bullets["content"].append(
                {"type": "listItem", "content": [{"type": "paragraph", "content": _line_nodes(line[2:])}]}
            )
            continue
        bullets = None
        if line.startswith("### "):
            blocks.append({"type": "heading", "attrs": {"level": 3}, "content": _line_nodes(line[4:])})
        else:
            blocks.append({"type": "paragraph", "content": _line_nodes(line)})
    return {"type": "doc", "version": 1, "content": blocks or [{"type": "paragraph", "content": []}]}


def _find_open_issues(base, email, token, project, label, max_results=20):
    jql = f'project = "{project}" AND labels = "{label}" AND statusCategory != Done ORDER BY created DESC'
    path = "/rest/api/3/search/jql?" + urllib.parse.urlencode({"jql": jql, "maxResults": max_results, "fields": "key"})
    issues = _api(base, email, token, "GET", path).get("issues", [])
    return [i["key"] for i in issues]


def _find_open_dupe(base, email, token, project, dedup_label):
    keys = _find_open_issues(base, email, token, project, dedup_label, max_results=1)
    return keys[0] if keys else None


def _pick_done_transition(transitions):
    """The transition that lands in the Done status category, if any."""
    done = [t for t in transitions if (t.get("to") or {}).get("statusCategory", {}).get("key") == "done"]
    if not done:
        return None
    # Prefer the conventional names so boards with several done-ish states
    # (e.g. Done and Won't Do) resolve rather than discard.
    for name in ("done", "closed", "resolved", "resolve"):
        for t in done:
            if t.get("name", "").strip().lower() == name:
                return t
    return done[0]


def close_issues(base, email, token, project, label, comment="", dry_run=False):
    """Close every open issue carrying `label`, commenting `comment` first.

    Returns a list of human-readable result strings, one per issue. An issue
    whose workflow offers no transition into the Done category keeps the
    comment but stays open -- better a stale-open ticket than a lost record.
    """
    keys = _find_open_issues(base, email, token, project, label)
    if dry_run:
        return [f"DRY RUN -- would close {k} with comment: {comment!r}" for k in keys] or [
            f"DRY RUN -- no open issue carries label {label!r}"
        ]
    results = []
    for key in keys:
        if comment:
            _api(base, email, token, "POST", f"/rest/api/3/issue/{key}/comment", {"body": _adf(comment)})
        transitions = _api(base, email, token, "GET", f"/rest/api/3/issue/{key}/transitions").get("transitions", [])
        chosen = _pick_done_transition(transitions)
        if not chosen:
            results.append(f"commented on {key} but found no Done transition; left open")
            continue
        _api(base, email, token, "POST", f"/rest/api/3/issue/{key}/transitions", {"transition": {"id": chosen["id"]}})
        results.append(f"closed {key} ({chosen['name']}): {base.rstrip('/')}/browse/{key}")
    return results or [f"no open issue carries label {label!r}; nothing to close"]


def file_issue(
    base,
    email,
    token,
    project,
    summary,
    issue_type="Bug",
    description="",
    labels=None,
    dedup_label="",
    assignee=None,
    dry_run=False,
):
    """Create (or comment onto a de-duped) Jira issue.

    Returns a human-readable result string. When dedup_label is set and an open
    issue already carries it, a comment is added instead of opening a duplicate.
    assignee, when set, is a Jira accountId the new issue is assigned to.
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
    if assignee:
        fields["assignee"] = {"accountId": assignee}

    if dry_run:
        return "DRY RUN -- would POST /rest/api/3/issue with fields:\n" + json.dumps({"fields": fields}, indent=2)

    if dedup_label:
        existing = _find_open_dupe(base, email, token, project, dedup_label)
        if existing:
            # Merge this filing's labels onto the existing ticket (adding a
            # label an issue already has is a no-op). Notably the per-ref
            # label close-on-green searches by, which older tickets predate.
            if labels:
                _api(
                    base,
                    email,
                    token,
                    "PUT",
                    f"/rest/api/3/issue/{existing}",
                    {"update": {"labels": [{"add": label} for label in labels]}},
                )
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
    if _env("JIRA_ACTION", "file").strip().lower() == "close":
        for line in close_issues(
            base=_env("JIRA_BASE_URL", required=True),
            email=_env("JIRA_USER_EMAIL", required=True),
            token=_env("JIRA_API_TOKEN", required=True),
            project=_env("JIRA_PROJECT_KEY", required=True),
            label=_env("JIRA_DEDUP_LABEL", required=True),
            comment=_env("JIRA_COMMENT", ""),
            dry_run=_truthy(_env("JIRA_DRY_RUN")),
        ):
            print(line)
        return
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
            assignee=_env("JIRA_ASSIGNEE_ACCOUNT_ID", "") or None,
            dry_run=_truthy(_env("JIRA_DRY_RUN")),
        )
    )


if __name__ == "__main__":
    main()
