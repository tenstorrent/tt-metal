#!/usr/bin/env python3
"""M4: Create issues from deterministic aggregate failures and notify Slack."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PRIMARY_REPO = "tenstorrent/tt-metal"
ISSUE_REPO_TEST = "ebanerjeeTT/issue_dump"
SLACK_CHANNEL_TEST = "C0APK6215B5"
MARKER = "===FINAL_M4_REVIEW_DECISION==="
PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_.-])((?:tt_metal|ttnn|models|tests|\\.github)/[A-Za-z0-9_./-]+)")


def log(message: str) -> None:
    print(f"[m4] {message}", flush=True)


def run(
    cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    proc_env = None
    if env:
        proc_env = os.environ.copy()
        proc_env.update(env)
    proc = subprocess.run(cmd, text=True, capture_output=capture, env=proc_env, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


def run_guarded_gh(tokens: list[str], *, github_token: str) -> subprocess.CompletedProcess[str]:
    command = " ".join(shlex.quote(tok) for tok in tokens)
    return run(
        [sys.executable, "tools/ci/guarded_gh.py", "--command", command],
        env={"GITHUB_TOKEN": github_token},
    )


def parse_job_url(url: str) -> tuple[int, int]:
    m = re.search(r"/actions/runs/(\d+)/job/(\d+)", url)
    if not m:
        raise ValueError(f"unsupported job URL format: {url}")
    return int(m.group(1)), int(m.group(2))


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        return []
    return [job for job in jobs if isinstance(job, dict)]


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def write_job_log_file(*, job_url: str, read_token: str, out_path: Path) -> Path:
    run_id, job_id = parse_job_url(job_url)
    proc = run_guarded_gh(
        ["gh", "run", "view", str(run_id), "--repo", PRIMARY_REPO, "--job", str(job_id), "--log-failed"],
        github_token=read_token,
    )
    text = (proc.stdout or "").strip()
    if not text:
        proc = run_guarded_gh(
            ["gh", "run", "view", str(run_id), "--repo", PRIMARY_REPO, "--job", str(job_id), "--log"],
            github_token=read_token,
        )
        text = (proc.stdout or "").strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def parse_agent_json(text: str) -> dict[str, Any]:
    parsed = _parse_agent_json_payload(text, marker=MARKER)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object for single decision payload, got {type(parsed).__name__}")
    return parsed


def _strip_json_fence(payload: str) -> str:
    stripped = payload.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
    return stripped


def _parse_agent_json_payload(text: str, *, marker: str) -> Any:
    idx = text.rfind(marker)
    if idx >= 0:
        payload = _strip_json_fence(text[idx + len(marker) :])
        return json.loads(payload)

    stripped = text.strip()
    if not stripped:
        raise ValueError(f"marker not found: {marker}. output excerpt: <empty>")

    # Fallback 1: raw JSON in stdout without marker.
    try:
        return json.loads(_strip_json_fence(stripped))
    except Exception:
        pass

    # Fallback 2: parse last fenced block.
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    for block in reversed(fenced):
        candidate = block.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # Fallback 3: scan for trailing JSON object/array start.
    for match in re.finditer(r"[\{\[]", stripped):
        candidate = stripped[match.start() :].strip()
        try:
            return json.loads(candidate)
        except Exception:
            continue

    excerpt = stripped[-600:].replace("\n", "\\n")
    raise ValueError(f"marker not found: {marker}. Could not parse fallback JSON. output excerpt: {excerpt}")


def load_command_spec(path: str, fallback: str) -> str:
    p = Path(path)
    if not p.exists():
        return fallback
    return p.read_text(encoding="utf-8")


def fetch_recent_slack_messages(*, slack_token: str, channel_id: str, limit: int = 20) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"channel": channel_id, "limit": str(limit)})
    req = urllib.request.Request(
        f"https://slack.com/api/conversations.history?{query}",
        headers={"Authorization": f"Bearer {slack_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Slack history fetch failed: http_{exc.code}") from exc
    if not payload.get("ok", False):
        raise RuntimeError(f"Slack history fetch failed: {payload.get('error', 'unknown_error')}")
    messages = payload.get("messages", [])
    return messages if isinstance(messages, list) else []


def build_open_issue_context(open_issues: list[dict[str, Any]], cap: int = 20) -> str:
    lines: list[str] = []
    for issue in open_issues[:cap]:
        number = issue.get("number")
        title = str(issue.get("title", "")).strip()
        url = str(issue.get("url", "")).strip()
        body = str(issue.get("body", "")).strip()
        marker_match = re.search(r"Auto-triage-fingerprint:\s*([A-Za-z0-9_-]+)", body)
        fp = marker_match.group(1) if marker_match else ""
        lines.append(f"- #{number} {title} ({url}) fingerprint={fp}")
    return "\n".join(lines) if lines else "(none)"


def build_recent_slack_context(messages: list[dict[str, Any]], cap: int = 10) -> str:
    lines: list[str] = []
    for msg in messages[:cap]:
        ts = str(msg.get("ts", "")).strip()
        user = str(msg.get("user", "") or msg.get("bot_id", "")).strip()
        text = str(msg.get("text", "")).strip().replace("\n", " ")
        if len(text) > 220:
            text = text[:220] + "..."
        lines.append(f"- ts={ts} user={user} text={text}")
    return "\n".join(lines) if lines else "(none)"


def parse_codeowners(path: Path) -> list[tuple[str, list[str]]]:
    if not path.exists():
        return []
    rules: list[tuple[str, list[str]]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pattern = parts[0].lstrip("/")
        owners = [p[1:] for p in parts[1:] if p.startswith("@")]
        if owners:
            rules.append((pattern, owners))
    return rules


def codeowners_match(path: str, pattern: str) -> bool:
    p = path.lstrip("/")
    pat = pattern.lstrip("/")
    if fnmatch.fnmatch(p, pat):
        return True
    if pat.endswith("/") and p.startswith(pat):
        return True
    if "/" not in pat and fnmatch.fnmatch(Path(p).name, pat):
        return True
    return False


def owners_for_paths(paths: list[str], rules: list[tuple[str, list[str]]]) -> set[str]:
    owners: set[str] = set()
    for path in paths:
        matched: list[str] = []
        for pattern, rule_owners in rules:
            if codeowners_match(path, pattern):
                matched = rule_owners
        owners.update(matched)
    return owners


def _tokenize_owner_hint(value: str) -> list[str]:
    tokens = re.split(r"[^a-z0-9]+", value.lower())
    return [t for t in tokens if len(t) >= 3]


def owners_from_workflow_name(
    workflow_name: str, *, rules: list[tuple[str, list[str]]], workflow_root: Path
) -> set[str]:
    if not workflow_name.strip():
        return set()
    hint_tokens = set(_tokenize_owner_hint(workflow_name))
    if not hint_tokens:
        return set()
    matched_paths: list[str] = []
    for ext in ("*.yaml", "*.yml"):
        for wf in workflow_root.glob(ext):
            rel = wf.as_posix()
            canonical_rel = f".github/workflows/{wf.name}"
            file_tokens = set(_tokenize_owner_hint(wf.stem))
            overlap = len(hint_tokens.intersection(file_tokens))
            # Require at least 2 shared tokens (or 1 when workflow name itself
            # has only one meaningful token), to avoid broad false matches.
            required = 1 if len(hint_tokens) == 1 else 2
            if overlap >= required:
                matched_paths.append(rel)
                matched_paths.append(canonical_rel)
    if not matched_paths:
        return set()
    owners: set[str] = set()
    for rel_path in matched_paths:
        owners.update(owners_for_paths([rel_path], rules))
    return owners


def extract_repo_paths(text: str) -> list[str]:
    out: list[str] = []
    for m in PATH_PATTERN.finditer(text):
        value = m.group(1).strip()
        if value and value not in out:
            out.append(value)
    return out


def github_user_info(token: str, username: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"https://api.github.com/users/{urllib.parse.quote(username)}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "tt-metal-m4-owner-resolver",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def slack_api_form(token: str, endpoint: str, fields: dict[str, str]) -> dict[str, Any]:
    data = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(
        f"https://slack.com/api/{endpoint}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload


def slack_lookup_by_email(token: str, email: str) -> str | None:
    try:
        payload = slack_api_form(token, "users.lookupByEmail", {"email": email})
    except Exception:
        return None
    if not payload.get("ok"):
        return None
    user = payload.get("user", {})
    user_id = str(user.get("id", "")).strip()
    return user_id or None


def slack_list_members(token: str) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    cursor = ""
    while True:
        params = {"limit": "200"}
        if cursor:
            params["cursor"] = cursor
        payload = slack_api_form(token, "users.list", params)
        if not payload.get("ok"):
            break
        batch = payload.get("members", [])
        if isinstance(batch, list):
            members.extend([m for m in batch if isinstance(m, dict)])
        cursor = str(payload.get("response_metadata", {}).get("next_cursor", "")).strip()
        if not cursor:
            break
    return members


def slack_lookup_by_username(token: str, username: str, members_cache: list[dict[str, Any]] | None) -> str | None:
    if members_cache is None:
        return None
    target = username.strip().lower()
    if not target:
        return None
    targets = {target}
    if target.endswith("tt") and len(target) > 2:
        targets.add(target[:-2])
    for member in members_cache:
        if member.get("deleted") or member.get("is_bot"):
            continue
        profile = member.get("profile", {}) if isinstance(member.get("profile"), dict) else {}
        candidates = [
            str(member.get("name", "")),
            str(profile.get("display_name", "")),
            str(profile.get("real_name", "")),
        ]
        if any(c.strip().lower() in targets for c in candidates if c.strip()):
            uid = str(member.get("id", "")).strip()
            if uid:
                return uid
    return None


def slack_lookup_by_full_name(full_name: str, members_cache: list[dict[str, Any]] | None) -> str | None:
    if members_cache is None:
        return None
    name = full_name.strip()
    if not name:
        return None
    name_l = name.lower()
    candidate_rows: list[tuple[str, str, str, str]] = []
    for member in members_cache:
        if member.get("deleted") or member.get("is_bot"):
            continue
        uid = str(member.get("id", "")).strip()
        if not uid:
            continue
        profile = member.get("profile", {}) if isinstance(member.get("profile"), dict) else {}
        row = (
            uid,
            str(member.get("name", "")).strip(),
            str(profile.get("display_name", "")).strip(),
            str(profile.get("real_name", "")).strip(),
        )
        candidate_rows.append(row)

    # Exact match by full name first.
    for uid, name_field, display_name, real_name in candidate_rows:
        values = [name_field, display_name, real_name]
        if any(v and v.lower() == name_l for v in values):
            return uid

    # Fuzzy fallback from working GH->Slack mapping pattern:
    # if a >=3-char name token uniquely matches one Slack member, use it.
    words = [w.lower() for w in re.split(r"\s+", name) if len(w) >= 3]
    seen_words: set[str] = set()
    for word in words:
        if word in seen_words:
            continue
        seen_words.add(word)
        matches: set[str] = set()
        for uid, name_field, display_name, real_name in candidate_rows:
            values_l = [name_field.lower(), display_name.lower(), real_name.lower()]
            if any(word in v for v in values_l if v):
                matches.add(uid)
        if len(matches) == 1:
            return next(iter(matches))
    return None


def recent_author_emails_for_paths(paths: list[str], *, limit_per_path: int = 5) -> set[str]:
    emails: set[str] = set()
    for path in paths:
        try:
            proc = run(
                ["git", "log", f"-n{limit_per_path}", "--format=%ae", "--", path],
                check=False,
                capture=True,
            )
        except Exception:
            continue
        for line in (proc.stdout or "").splitlines():
            email = line.strip()
            if email and "@" in email:
                emails.add(email)
    return emails


def render_owner_mentions(
    *,
    issue_token: str,
    slack_token: str,
    workflow_name: str,
    job_name: str,
    text_sources: list[str],
    members_cache: list[dict[str, Any]] | None,
) -> tuple[str, list[str]]:
    combined = "\n".join(text_sources + [workflow_name, job_name])
    paths = extract_repo_paths(combined)
    codeowners = parse_codeowners(Path(".github/CODEOWNERS"))
    gh_usernames = owners_for_paths(paths, codeowners)
    if not gh_usernames:
        gh_usernames = owners_from_workflow_name(
            workflow_name,
            rules=codeowners,
            workflow_root=Path(".github/workflows"),
        )
    resolved_ids: set[str] = set()
    unresolved_handles: list[str] = []

    for username in sorted(gh_usernames):
        # CODEOWNERS teams (org/team) are not direct users and cannot map
        # to a single Slack user id.
        if "/" in username:
            unresolved_handles.append(username)
            continue
        info = github_user_info(issue_token, username)
        email = str(info.get("email", "")).strip()
        gh_name = str(info.get("name", "")).strip()
        gh_login = str(info.get("login", username)).strip()
        user_id = slack_lookup_by_email(slack_token, email) if email else None
        if not user_id:
            user_id = slack_lookup_by_username(slack_token, gh_login, members_cache)
        if not user_id and gh_name:
            user_id = slack_lookup_by_full_name(gh_name, members_cache)
        if user_id:
            resolved_ids.add(user_id)
        else:
            unresolved_handles.append(username)

    # Supplement with recent path authors when CODEOWNERS is sparse.
    for email in sorted(recent_author_emails_for_paths(paths)):
        user_id = slack_lookup_by_email(slack_token, email)
        if user_id:
            resolved_ids.add(user_id)

    mentions = " ".join(sorted(f"<@{uid}>" for uid in resolved_ids))
    return mentions, unresolved_handles


def normalize_slack_text(
    *,
    slack_text: str,
    issue_url: str,
    owner_mentions: str,
) -> str:
    text = slack_text
    text = re.sub(
        r"Issue:\s*https://github\.com/tenstorrent/tt-metal/issues/TBD",
        f"Issue: {issue_url}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"Issue:\s*TBD", f"Issue: {issue_url}", text, flags=re.IGNORECASE)
    text = re.sub(r"Likely owners:\s*.+", f"Likely owners: {owner_mentions}", text, flags=re.IGNORECASE)
    if issue_url not in text:
        text = text.rstrip() + f"\nIssue: {issue_url}"
    if "Likely owners:" not in text:
        owner_text = owner_mentions if owner_mentions else "(owner resolution pending)"
        text = text.rstrip() + f"\nLikely owners: {owner_text}"
    return text


def find_exact_previous_decision(
    review_entries: list[dict[str, Any]],
    *,
    workflow_name: str,
    job_name: str,
    job_urls: list[str],
) -> dict[str, Any] | None:
    for entry in review_entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("workflow_name", "")).strip() != workflow_name:
            continue
        if str(entry.get("job_name", "")).strip() != job_name:
            continue
        prev_decision = entry.get("decision")
        if not isinstance(prev_decision, dict):
            continue
        cached_reason = str(prev_decision.get("reason", "")).lower()
        if "only log head excerpts provided" in cached_reason:
            continue
        if "prompt exceeded os argument length" in cached_reason:
            continue
        prev_urls = entry.get("job_urls")
        if isinstance(prev_urls, list) and [str(u).strip() for u in prev_urls] == job_urls:
            return prev_decision
    return None


def decision_key(workflow_name: str, job_name: str, job_urls: list[str]) -> str:
    return json.dumps(
        {"workflow_name": workflow_name, "job_name": job_name, "job_urls": job_urls},
        sort_keys=True,
    )


def sanitize_for_path(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "item"


def agent_review_and_prepare_outputs(
    *,
    workflow_name: str,
    job_name: str,
    job_urls: list[str],
    log_paths: list[str],
    open_issue_context: str,
    recent_slack_context: str,
    model: str,
) -> dict[str, Any]:
    create_ticket_spec = load_command_spec(
        ".cursor/commands/ci/ci-create-tickets.md",
        "Create deterministic tickets only when the same terminal failure appears in all 3 logs.",
    )
    slack_draft_spec = load_command_spec(
        ".cursor/commands/ci/ci-draft-slack-ci-issue.md",
        "Draft concise Slack incident updates with direct links.",
    )
    log_sections: list[str] = []
    for idx, (url, path) in enumerate(zip(job_urls, log_paths, strict=False), start=1):
        log_sections.append(f"Run {idx} URL: {url}\nRun {idx} local log path: {path}")
    prompt = (
        "You are the M4 reviewer for deterministic CI failures.\n"
        "You MUST manually compare all three logs and determine if terminal failure signatures are the same.\n"
        "You MUST also review existing open issues and recent Slack messages before drafting outputs.\n"
        "CRITICAL: logs are available as local files below; inspect those files directly instead of relying on excerpts in this prompt.\n"
        "Use shell/read tools to inspect those paths and determine the terminal failure signature.\n"
        "Criteria: deterministic only if the same job failed 3 times in a row with semantically identical terminal failure.\n\n"
        f"Workflow: {workflow_name}\n"
        f"Job: {job_name}\n"
        "Open issues context:\n"
        f"{open_issue_context}\n\n"
        "Recent Slack messages context:\n"
        f"{recent_slack_context}\n\n"
        "Ticket command guidance:\n"
        f"{create_ticket_spec}\n\n"
        "Slack draft guidance:\n"
        f"{slack_draft_spec}\n\n"
        "Decide if all three runs share one identical terminal failure signature.\n"
        "If yes, draft issue title/body and draft initial Slack message text.\n"
        "If uncertain, set create_issue=false and draft_slack=false.\n\n"
        + "Log file references:\n"
        + "\n".join(f"- {section}" for section in log_sections)
        + "\n\nOutput marker exactly on its own line:\n"
        + MARKER
        + "\nThen output compact JSON only:\n"
        + (
            '{"deterministic":false,"confidence":"low|medium|high","signature":"","error_excerpt":"","reason":"",'
            '"create_issue":false,"draft_slack":false,"issue_title":"","issue_body":"","slack_text":""}'
        )
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model != "auto":
        cmd[1:1] = ["--model", model]
    proc = run(cmd)
    return parse_agent_json(proc.stdout or "")


def parse_batch_agent_json(text: str) -> list[dict[str, Any]]:
    parsed = _parse_agent_json_payload(text, marker=MARKER)
    decisions = parsed.get("decisions", []) if isinstance(parsed, dict) else []
    return decisions if isinstance(decisions, list) else []


def agent_review_batch_prepare_outputs(
    *,
    jobs: list[dict[str, Any]],
    open_issue_context: str,
    recent_slack_context: str,
    model: str,
) -> list[dict[str, Any]]:
    create_ticket_spec = load_command_spec(
        ".cursor/commands/ci/ci-create-tickets.md",
        "Create deterministic tickets only when the same terminal failure appears in all 3 logs.",
    )
    slack_draft_spec = load_command_spec(
        ".cursor/commands/ci/ci-draft-slack-ci-issue.md",
        "Draft concise Slack incident updates with direct links.",
    )
    sections: list[str] = []
    for idx, job in enumerate(jobs, start=1):
        workflow_name = str(job.get("workflow_name", "")).strip()
        job_name = str(job.get("job_name", "")).strip()
        job_urls = job.get("job_urls", [])
        log_paths = job.get("log_paths", [])
        if not isinstance(job_urls, list) or not isinstance(log_paths, list):
            continue
        lines = [f"[JOB {idx}]", f"Workflow: {workflow_name}", f"Job: {job_name}"]
        for run_idx, (url, path) in enumerate(zip(job_urls, log_paths, strict=False), start=1):
            lines.append(f"Run {run_idx} URL: {url}")
            lines.append(f"Run {run_idx} local log path: {path}")
        sections.append("\n".join(lines))
    prompt = (
        "You are the M4 reviewer for deterministic CI failures.\n"
        "For EACH provided job, manually compare all three logs and determine if terminal failure signatures are the same.\n"
        "You MUST review existing open issues and recent Slack messages before drafting outputs.\n"
        "CRITICAL: logs are available as local files below; inspect those files directly.\n"
        "Use shell/read tools to inspect those paths and determine terminal failure signatures.\n"
        "Criteria: deterministic only if the same job failed 3 times in a row with semantically identical terminal failure.\n\n"
        "Open issues context:\n"
        f"{open_issue_context}\n\n"
        "Recent Slack messages context:\n"
        f"{recent_slack_context}\n\n"
        "Ticket command guidance:\n"
        f"{create_ticket_spec}\n\n"
        "Slack draft guidance:\n"
        f"{slack_draft_spec}\n\n"
        "Jobs to review:\n"
        + "\n\n".join(sections)
        + "\n\nOutput marker exactly on its own line:\n"
        + MARKER
        + "\nThen output compact JSON only:\n"
        + '{"decisions":[{"workflow_name":"","job_name":"","job_urls":[],"deterministic":false,"confidence":"low|medium|high","signature":"","error_excerpt":"","reason":"","create_issue":false,"draft_slack":false,"issue_title":"","issue_body":"","slack_text":""}]}'
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model != "auto":
        cmd[1:1] = ["--model", model]
    proc = run(cmd)
    return parse_batch_agent_json(proc.stdout or "")


def fingerprint_for(workflow_name: str, job_name: str, signature: str) -> str:
    raw = f"{workflow_name}\n{job_name}\n{signature.strip().lower()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def job_identity_key(workflow_name: str, job_name: str) -> str:
    raw = f"{workflow_name}\n{job_name}".strip().lower().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def issue_marker(fingerprint: str) -> str:
    return f"Auto-triage-fingerprint: {fingerprint}"


def issue_job_identity_marker(job_key: str) -> str:
    return f"Auto-triage-job-key: {job_key}"


def list_open_issue_bodies(*, issue_token: str) -> list[dict[str, Any]]:
    proc = run_guarded_gh(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            ISSUE_REPO_TEST,
            "--state",
            "open",
            "--limit",
            "200",
            "--json",
            "number,title,body,url",
        ],
        github_token=issue_token,
    )
    parsed = json.loads(proc.stdout or "[]")
    return parsed if isinstance(parsed, list) else []


def find_existing_issue_for_fingerprint(open_issues: list[dict[str, Any]], fingerprint: str) -> str | None:
    marker = issue_marker(fingerprint)
    for issue in open_issues:
        body = str(issue.get("body", ""))
        if marker in body:
            return str(issue.get("url", "")).strip() or None
    return None


def find_existing_issue_for_job_identity(
    open_issues: list[dict[str, Any]],
    *,
    workflow_name: str,
    job_name: str,
) -> str | None:
    marker = issue_job_identity_marker(job_identity_key(workflow_name, job_name))
    workflow_l = workflow_name.lower()
    job_l = job_name.lower()
    for issue in open_issues:
        body = str(issue.get("body", ""))
        title = str(issue.get("title", ""))
        url = str(issue.get("url", "")).strip() or None
        if marker in body:
            return url
        combined = f"{title}\n{body}".lower()
        # Legacy fallback for older issues that predate job-key marker.
        if workflow_l in combined and job_l in combined:
            return url
    return None


def find_existing_issue_for_title(open_issues: list[dict[str, Any]], title: str) -> str | None:
    wanted = title.strip().lower()
    if not wanted:
        return None
    for issue in open_issues:
        existing_title = str(issue.get("title", "")).strip().lower()
        if existing_title == wanted:
            return str(issue.get("url", "")).strip() or None
    return None


def create_issue(
    *,
    issue_token: str,
    title: str,
    body: str,
) -> str:
    proc = run_guarded_gh(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            ISSUE_REPO_TEST,
            "--title",
            title,
            "--body",
            body,
            "--label",
            "CI auto triage",
        ],
        github_token=issue_token,
    )
    return (proc.stdout or "").strip().splitlines()[-1].strip()


def post_slack_message(*, slack_token: str, channel_id: str, text: str) -> str:
    data = urllib.parse.urlencode({"channel": channel_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Authorization": f"Bearer {slack_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Slack post failed: http_{exc.code}") from exc
    if not payload.get("ok", False):
        raise RuntimeError(f"Slack post failed: {payload.get('error', 'unknown_error')}")
    return str(payload.get("ts", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 deterministic issue + Slack bootstrap automation.")
    parser.add_argument(
        "--failing-jobs-json",
        default="build_ci/ci_ticketing/create_tickets/failing_jobs.json",
    )
    parser.add_argument(
        "--output-json",
        default="build_ci/ci_ticketing/create_tickets/m4_issue_and_slack_result.json",
    )
    parser.add_argument(
        "--review-cache-json",
        default="build_ci/ci_ticketing/create_tickets/m4_review_cache.json",
    )
    parser.add_argument(
        "--previous-artifact-dir",
        default="build_ci/m4_cache_prev",
    )
    parser.add_argument(
        "--downloaded-logs-dir",
        default="build_ci/ci_ticketing/create_tickets/downloaded_logs",
    )
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-new-issues", type=int, default=1)
    parser.add_argument(
        "--max-agent-reviews",
        type=int,
        default=0,
        help="Deprecated: review count is no longer capped; retained for compatibility.",
    )
    parser.add_argument("--model", default="auto")
    args = parser.parse_args()

    read_token = os.environ.get("AGGREGATE_READ_TOKEN", "").strip()
    issue_token = os.environ.get("ISSUE_WRITE_TOKEN", "").strip()
    slack_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not read_token:
        print("AGGREGATE_READ_TOKEN is required", file=sys.stderr)
        return 2
    if not issue_token:
        print("ISSUE_WRITE_TOKEN is required", file=sys.stderr)
        return 2
    if not slack_token:
        print("SLACK_BOT_TOKEN is required", file=sys.stderr)
        return 2
    if not os.environ.get("CURSOR_API_KEY", "").strip():
        print("CURSOR_API_KEY is required", file=sys.stderr)
        return 2

    jobs_path = Path(args.failing_jobs_json)
    if not jobs_path.exists():
        print(f"Missing failing jobs input: {jobs_path}", file=sys.stderr)
        return 2
    candidates = load_candidates(jobs_path)
    if not candidates:
        log("No failing jobs found; nothing to do.")
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps({"created": [], "skipped": []}, indent=2), encoding="utf-8")
        return 0

    previous_dir = Path(args.previous_artifact_dir)
    prev_review_path = previous_dir / "m4_review_cache.json"
    if not prev_review_path.exists():
        matches = list(previous_dir.rglob("m4_review_cache.json"))
        if matches:
            prev_review_path = matches[0]
    prev_review_payload = load_optional_json(prev_review_path) or {}
    prev_review_version = int(prev_review_payload.get("version", 1)) if isinstance(prev_review_payload, dict) else 1
    prev_review_entries = prev_review_payload.get("entries", [])
    if not isinstance(prev_review_entries, list):
        prev_review_entries = []
    if prev_review_version < 2:
        prev_review_entries = []
        log("Ignoring prior m4 review cache (version < 2).")
    open_issues = list_open_issue_bodies(issue_token=issue_token)
    recent_slack_messages = fetch_recent_slack_messages(
        slack_token=slack_token, channel_id=SLACK_CHANNEL_TEST, limit=20
    )
    slack_members_cache = slack_list_members(slack_token)
    open_issue_context = build_open_issue_context(open_issues)
    recent_slack_context = build_recent_slack_context(recent_slack_messages)
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    review_cache_entries: list[dict[str, Any]] = []
    max_new_issues = max(args.max_new_issues, 0)
    max_agent_reviews = max(args.max_agent_reviews, 0)
    fresh_agent_reviews = 0
    fresh_agent_calls = 0
    prepared: list[dict[str, Any]] = []
    for candidate in candidates[: max(args.max_candidates, 0)]:
        job_name = str(candidate.get("job_name", "")).strip()
        workflow_name = str(candidate.get("workflow_name", "")).strip()
        urls_raw = candidate.get("failing_job_urls", [])
        if not job_name or not workflow_name or not isinstance(urls_raw, list):
            continue
        job_urls = [str(u).strip() for u in urls_raw if str(u).strip()][:3]
        if len(job_urls) < 3:
            skipped.append({"job_name": job_name, "reason": "requires_3_failing_runs"})
            continue
        prepared.append(
            {
                "workflow_name": workflow_name,
                "job_name": job_name,
                "job_urls": job_urls,
            }
        )

    logs_root = Path(args.downloaded_logs_dir)
    if logs_root.exists():
        shutil.rmtree(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)
    decisions_by_key: dict[str, dict[str, Any]] = {}
    reused_cache_keys: set[str] = set()
    to_review: list[dict[str, Any]] = []

    for item in prepared:
        workflow_name = item["workflow_name"]
        job_name = item["job_name"]
        job_urls = item["job_urls"]
        existing_by_identity = find_existing_issue_for_job_identity(
            open_issues,
            workflow_name=workflow_name,
            job_name=job_name,
        )
        if existing_by_identity:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": f"already_tracked_job_identity:{existing_by_identity}",
                }
            )
            continue
        key = decision_key(workflow_name, job_name, job_urls)
        prior = find_exact_previous_decision(
            prev_review_entries,
            workflow_name=workflow_name,
            job_name=job_name,
            job_urls=job_urls,
        )
        if prior:
            decisions_by_key[key] = prior
            reused_cache_keys.add(key)
            log(f"Reused cached review decision for workflow={workflow_name} job={job_name}")
            continue
        job_slug = sanitize_for_path(f"{workflow_name}-{job_name}")[:120]
        log_paths: list[str] = []
        for idx, url in enumerate(job_urls, start=1):
            log_path = logs_root / job_slug / f"run{idx}.log"
            write_job_log_file(job_url=url, read_token=read_token, out_path=log_path)
            log_paths.append(str(log_path))
        to_review.append(
            {
                "workflow_name": workflow_name,
                "job_name": job_name,
                "job_urls": job_urls,
                "log_paths": log_paths,
            }
        )

    if to_review:
        log(f"Running batched agent review for {len(to_review)} jobs")
        try:
            batch_decisions = agent_review_batch_prepare_outputs(
                jobs=to_review,
                open_issue_context=open_issue_context,
                recent_slack_context=recent_slack_context,
                model=args.model,
            )
            fresh_agent_calls += 1
            batch_map: dict[str, dict[str, Any]] = {}
            for decision in batch_decisions:
                if not isinstance(decision, dict):
                    continue
                workflow_name = str(decision.get("workflow_name", "")).strip()
                job_name = str(decision.get("job_name", "")).strip()
                job_urls = decision.get("job_urls", [])
                if not workflow_name or not job_name or not isinstance(job_urls, list):
                    continue
                normalized_urls = [str(u).strip() for u in job_urls if str(u).strip()][:3]
                batch_map[decision_key(workflow_name, job_name, normalized_urls)] = decision
            for review_item in to_review:
                key = decision_key(
                    review_item["workflow_name"],
                    review_item["job_name"],
                    review_item["job_urls"],
                )
                decision = batch_map.get(key)
                if not decision:
                    decision = {
                        "deterministic": False,
                        "confidence": "low",
                        "signature": "",
                        "error_excerpt": "",
                        "reason": "Batch review did not return a decision for this job.",
                        "create_issue": False,
                        "draft_slack": False,
                        "issue_title": "",
                        "issue_body": "",
                        "slack_text": "",
                    }
                decisions_by_key[key] = decision
                fresh_agent_reviews += 1
        except OSError as exc:
            if exc.errno != 7:
                raise
            for review_item in to_review:
                key = decision_key(
                    review_item["workflow_name"],
                    review_item["job_name"],
                    review_item["job_urls"],
                )
                decisions_by_key[key] = {
                    "deterministic": False,
                    "confidence": "low",
                    "signature": "",
                    "error_excerpt": "",
                    "reason": "Prompt exceeded OS argument length while sending batched file references to agent.",
                    "create_issue": False,
                    "draft_slack": False,
                    "issue_title": "",
                    "issue_body": "",
                    "slack_text": "",
                }
                fresh_agent_reviews += 1
            fresh_agent_calls += 1

    for item in prepared:
        if len(created) >= max_new_issues:
            skipped.append({"reason": f"max_new_issues_reached:{max_new_issues}"})
            break
        workflow_name = item["workflow_name"]
        job_name = item["job_name"]
        job_urls = item["job_urls"]
        key = decision_key(workflow_name, job_name, job_urls)
        decision = decisions_by_key.get(key)
        if not decision:
            continue
        reused_cache = key in reused_cache_keys
        review_cache_entries.append(
            {
                "workflow_name": workflow_name,
                "job_name": job_name,
                "job_urls": job_urls,
                "decision": decision,
                "reused_cache": reused_cache,
            }
        )
        deterministic = bool(decision.get("deterministic", False))
        confidence = str(decision.get("confidence", "low")).strip().lower()
        signature = str(decision.get("signature", "")).strip()
        excerpt = str(decision.get("error_excerpt", "")).strip()
        reason = str(decision.get("reason", "")).strip()
        create_issue_flag = bool(decision.get("create_issue", False))
        draft_slack_flag = bool(decision.get("draft_slack", False))
        issue_title = str(decision.get("issue_title", "")).strip()
        issue_body_from_agent = str(decision.get("issue_body", "")).strip()
        slack_text = str(decision.get("slack_text", "")).strip()
        if not deterministic or confidence != "high" or not signature or not create_issue_flag:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": "agent_rejected_or_low_confidence",
                    "decision": decision,
                }
            )
            log(
                "Skipped candidate "
                f"workflow={workflow_name} job={job_name}: {reason[:220] if reason else 'rejected_or_low_confidence'}"
            )
            continue

        # Re-check identity here because open_issues mutates within this run
        # after each created issue, and prepared candidates may contain repeats.
        existing_by_identity = find_existing_issue_for_job_identity(
            open_issues,
            workflow_name=workflow_name,
            job_name=job_name,
        )
        if existing_by_identity:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": f"already_tracked_job_identity_postcreate:{existing_by_identity}",
                }
            )
            continue

        fp = fingerprint_for(workflow_name, job_name, signature)
        existing = find_existing_issue_for_fingerprint(open_issues, fp)
        if existing:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": f"already_tracked:{existing}",
                    "fingerprint": fp,
                }
            )
            continue

        if not issue_title:
            issue_title = f"CI auto triage: deterministic failure in {job_name}"
        existing_by_title = find_existing_issue_for_title(open_issues, issue_title)
        if existing_by_title:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": f"already_tracked_title:{existing_by_title}",
                    "issue_title": issue_title,
                }
            )
            continue
        if not issue_body_from_agent:
            issue_body_from_agent = (
                f"Workflow: `{workflow_name}`\n"
                f"Job: `{job_name}`\n\n"
                f"Failure signature: `{signature}`\n\n"
                f"Error excerpt:\n"
                f"```\n{excerpt or reason or 'No excerpt captured'}\n```\n\n"
                f"Failing job URLs (last 3):\n" + "\n".join(f"- {url}" for url in job_urls)
            )
        marker = issue_marker(fp)
        job_marker = issue_job_identity_marker(job_identity_key(workflow_name, job_name))
        issue_body = issue_body_from_agent
        if marker not in issue_body:
            issue_body = issue_body.rstrip() + "\n\n" + marker + "\n"
        if job_marker not in issue_body:
            issue_body = issue_body.rstrip() + "\n" + job_marker + "\n"
        if not draft_slack_flag:
            skipped.append(
                {
                    "job_name": job_name,
                    "workflow_name": workflow_name,
                    "reason": "agent_declined_slack_draft",
                    "fingerprint": fp,
                    "decision": decision,
                }
            )
            continue
        issue_url = create_issue(issue_token=issue_token, title=issue_title, body=issue_body)
        open_issues.append({"url": issue_url, "title": issue_title, "body": issue_body})
        if not slack_text:
            slack_text = (
                f"CI auto triage detected a deterministic failure (3x in a row) for `{job_name}` in `{workflow_name}`.\n"
                f"Issue: {issue_url}\n"
                f"Fingerprint: `{fp}`\n"
                f"Latest run: {job_urls[0]}"
            )
        owner_mentions, unresolved_handles = render_owner_mentions(
            issue_token=issue_token,
            slack_token=slack_token,
            workflow_name=workflow_name,
            job_name=job_name,
            text_sources=[issue_body, excerpt, reason, signature],
            members_cache=slack_members_cache,
        )
        slack_text = normalize_slack_text(
            slack_text=slack_text,
            issue_url=issue_url,
            owner_mentions=owner_mentions,
        )
        if unresolved_handles:
            slack_text = (
                slack_text.rstrip()
                + "\nUnresolved GitHub owner handles: "
                + ", ".join(f"@{h}" for h in unresolved_handles)
            )
        slack_ts = post_slack_message(slack_token=slack_token, channel_id=SLACK_CHANNEL_TEST, text=slack_text)
        recent_slack_messages.insert(0, {"ts": slack_ts, "user": "m4-auto-triage", "text": slack_text})
        recent_slack_context = build_recent_slack_context(recent_slack_messages)
        open_issue_context = build_open_issue_context(open_issues)
        created.append(
            {
                "workflow_name": workflow_name,
                "job_name": job_name,
                "fingerprint": fp,
                "issue_url": issue_url,
                "slack_channel": SLACK_CHANNEL_TEST,
                "slack_ts": slack_ts,
                "job_urls": job_urls,
                "agent_decision": decision,
                "owner_mentions": owner_mentions,
                "unresolved_owner_handles": unresolved_handles,
            }
        )
        log(f"Created issue and posted Slack bootstrap: {issue_url} (ts={slack_ts})")

    output = {
        "created": created,
        "skipped": skipped,
        "candidate_count": len(candidates),
        "processed_count": min(len(candidates), max(args.max_candidates, 0)),
        "max_new_issues": max_new_issues,
        "max_agent_reviews": max_agent_reviews,
        "fresh_agent_reviews": fresh_agent_reviews,
        "fresh_agent_calls": fresh_agent_calls,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    review_cache_payload = {
        "version": 2,
        "entries": review_cache_entries,
    }
    review_cache_path = Path(args.review_cache_json)
    review_cache_path.parent.mkdir(parents=True, exist_ok=True)
    review_cache_path.write_text(json.dumps(review_cache_payload, indent=2), encoding="utf-8")
    print(json.dumps({"created_count": len(created), "skipped_count": len(skipped)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
