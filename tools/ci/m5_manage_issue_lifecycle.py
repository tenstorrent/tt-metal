#!/usr/bin/env python3
"""M5 lifecycle manager: thread-aware follow-ups, assignment, and post-disable messaging."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import hashlib
import re
import shlex
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Ensure repository root is importable when running as `python tools/ci/<script>.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ci.thread_signal_analysis import classify_thread_progress, detect_dev_fix_request

ISSUE_REPO_TEST = "ebanerjeeTT/issue_dump"
SLACK_CHANNEL_TEST = "C0APK6215B5"
ISSUE_URL_RE = re.compile(r"https://github\.com/ebanerjeeTT/issue_dump/issues/(\d+)")


def now_unix() -> float:
    import time

    return time.time()


def now_utc() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = None
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=merged_env)
    if proc.returncode != 0:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M5 lifecycle automation over Slack + triage state.")
    p.add_argument("--slack-json", required=True)
    p.add_argument("--state-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--summary-md", required=True)
    p.add_argument("--channel-id", default=SLACK_CHANNEL_TEST)
    p.add_argument("--warning-hours", type=float, default=24.0)
    p.add_argument("--final-warning-hours", type=float, default=40.0)
    p.add_argument("--disable-hours", type=float, default=48.0)
    p.add_argument("--progress-hold-hours", type=float, default=24.0)
    return p.parse_args()


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
        owners = [tok[1:] for tok in parts[1:] if tok.startswith("@") and "/" not in tok[1:]]
        if owners:
            rules.append((pattern, owners))
    return rules


def codeowners_match(path: str, pattern: str) -> bool:
    p = path.lstrip("/")
    pat = pattern.lstrip("/")
    return fnmatch.fnmatch(p, pat) or (pat.endswith("/") and p.startswith(pat))


def extract_repo_paths(text: str) -> list[str]:
    path_re = re.compile(r"(?<![A-Za-z0-9_.-])((?:tt_metal|ttnn|models|tests|\\.github)/[A-Za-z0-9_./-]+)")
    out: list[str] = []
    for m in path_re.finditer(text):
        path = m.group(1).strip()
        if path and path not in out:
            out.append(path)
    return out


def candidate_github_owners_from_text(text: str, rules: list[tuple[str, list[str]]]) -> list[str]:
    paths = extract_repo_paths(text)
    owners: list[str] = []
    seen: set[str] = set()
    for path in paths:
        matched: list[str] = []
        for pattern, ows in rules:
            if codeowners_match(path, pattern):
                matched = ows
        for ow in matched:
            low = ow.lower()
            if low in seen:
                continue
            seen.add(low)
            owners.append(ow)
    return owners


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
        return json.loads(resp.read().decode("utf-8"))


def post_slack_thread_message(*, slack_token: str, channel: str, thread_ts: str, text: str) -> str:
    payload = slack_api_form(
        slack_token,
        "chat.postMessage",
        {"channel": channel, "thread_ts": thread_ts, "text": text},
    )
    if not payload.get("ok"):
        raise RuntimeError(f"chat.postMessage failed: {payload.get('error', 'unknown_error')}")
    return str(payload.get("ts", "")).strip()


def github_user_info(token: str, username: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"https://api.github.com/users/{urllib.parse.quote(username)}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "tt-metal-m5-lifecycle",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def slack_lookup_by_email(token: str, email: str) -> str | None:
    if not email:
        return None
    try:
        payload = slack_api_form(token, "users.lookupByEmail", {"email": email})
    except Exception:
        return None
    if not payload.get("ok"):
        return None
    uid = str(payload.get("user", {}).get("id", "")).strip()
    return uid or None


def parse_json_after_marker(text: str, marker: str) -> dict[str, Any]:
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"marker not found: {marker}")
    payload = text[idx + len(marker) :].strip()
    if not payload:
        raise ValueError("empty json payload after marker")
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("payload after marker is not a JSON object")
    return obj


def slack_auth_user_id(token: str) -> str | None:
    try:
        payload = slack_api_form(token, "auth.test", {})
    except Exception:
        return None
    if not payload.get("ok"):
        return None
    user_id = str(payload.get("user_id", "")).strip()
    return user_id or None


def slack_user_profile(token: str, user_id: str) -> dict[str, Any]:
    if not user_id:
        return {}
    try:
        payload = slack_api_form(token, "users.info", {"user": user_id})
    except Exception:
        return {}
    if not payload.get("ok"):
        return {}
    user = payload.get("user", {})
    if not isinstance(user, dict):
        return {}
    profile = user.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}
    return {
        "user_id": str(user.get("id", "")).strip(),
        "real_name": str(profile.get("real_name", "")).strip(),
        "display_name": str(profile.get("display_name", "")).strip(),
        "email": str(profile.get("email", "")).strip(),
    }


def thread_owner_claim_via_agent(
    *,
    top_level_text: str,
    thread_replies: list[dict[str, Any]],
    bot_user_ids: set[str],
    model: str,
) -> dict[str, Any]:
    curated_replies = []
    for reply in thread_replies[-16:]:
        if not isinstance(reply, dict):
            continue
        curated_replies.append(
            {
                "ts": str(reply.get("ts", "")).strip(),
                "user": str(reply.get("user", "")).strip(),
                "text": str(reply.get("text", "")).strip(),
            }
        )
    prompt_payload = {
        "top_level_text": top_level_text[:1600],
        "thread_replies": curated_replies,
        "bot_user_ids": sorted(bot_user_ids),
        "task": (
            "Determine whether a non-bot thread reply contains a clear ownership claim "
            "(e.g., the person says they are working on it / will fix)."
        ),
    }
    prompt = (
        "You are classifying CI triage thread ownership claims.\n"
        "Token-efficiency rule: do one short pass and output only final JSON contract.\n"
        "Rules:\n"
        "- Only consider non-bot replies (exclude user ids in bot_user_ids).\n"
        "- Return claimed=true only for explicit self-ownership/fix intent.\n"
        "- Do not infer from vague or generic text.\n"
        "- Prefer precision over recall.\n"
        "At end print marker exactly on its own line:\n"
        "===FINAL_OWNER_CLAIM_JSON===\n"
        "Then print only JSON object:\n"
        "{\n"
        '  "claimed": true|false,\n'
        '  "slack_user_id": "U123" or "",\n'
        '  "confidence": "high|medium|low",\n'
        '  "reason": "short reason"\n'
        "}\n\n"
        f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=True)}\n"
    )
    cmd = ["agent", "--trust", "-p", prompt]
    model_name = model.strip()
    if model_name and model_name != "auto":
        cmd.extend(["--model", model_name])
    proc = run(cmd)
    parsed = parse_json_after_marker(proc.stdout or "", "===FINAL_OWNER_CLAIM_JSON===")
    claimed = bool(parsed.get("claimed", False))
    user_id = str(parsed.get("slack_user_id", "")).strip()
    if claimed and not user_id:
        return {"claimed": False, "slack_user_id": "", "confidence": "low", "reason": "claimed_without_user_id"}
    return {
        "claimed": claimed,
        "slack_user_id": user_id,
        "confidence": str(parsed.get("confidence", "low")).strip() or "low",
        "reason": str(parsed.get("reason", "unspecified")).strip() or "unspecified",
    }


def github_search_login(token: str, query: str) -> str | None:
    q = query.strip()
    if not q:
        return None
    req = urllib.request.Request(
        f"https://api.github.com/search/users?q={urllib.parse.quote(q)}&per_page=1",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "tt-metal-m5-lifecycle",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return None
    first = items[0]
    if not isinstance(first, dict):
        return None
    login = str(first.get("login", "")).strip()
    return login or None


def github_login_from_slack_profile(token: str, profile: dict[str, Any]) -> str | None:
    email = str(profile.get("email", "")).strip()
    real_name = str(profile.get("real_name", "")).strip()
    display_name = str(profile.get("display_name", "")).strip()
    if email:
        login = github_search_login(token, f"{email} in:email")
        if login:
            return login
    if real_name:
        login = github_search_login(token, f'"{real_name}" in:fullname')
        if login:
            return login
    if display_name:
        login = github_search_login(token, f"{display_name} in:login")
        if login:
            return login
    return None


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "updated_at_utc": now_utc(), "items": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("state must be object")
    if not isinstance(payload.get("items"), list):
        payload["items"] = []
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = now_utc()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def issue_assignees(issue_token: str, issue_number: int) -> list[str]:
    proc = run_guarded_gh(
        [
            "gh",
            "issue",
            "view",
            "--repo",
            ISSUE_REPO_TEST,
            str(issue_number),
            "--json",
            "assignees",
        ],
        github_token=issue_token,
    )
    payload = json.loads(proc.stdout or "{}")
    rows = payload.get("assignees", [])
    out: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            login = str(row.get("login", "")).strip()
            if login:
                out.append(login)
    return out


def assign_issue(issue_token: str, issue_number: int, github_login: str) -> bool:
    run_guarded_gh(
        [
            "gh",
            "issue",
            "edit",
            "--repo",
            ISSUE_REPO_TEST,
            str(issue_number),
            "--add-assignee",
            github_login,
        ],
        github_token=issue_token,
    )
    return True


def pr_merge_info(issue_token: str, pr_url: str) -> dict[str, Any]:
    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", pr_url)
    if not m:
        return {"merged": False, "reason": "bad_pr_url"}
    repo = m.group(1)
    number = m.group(2)
    proc = run_guarded_gh(
        [
            "gh",
            "pr",
            "view",
            "--repo",
            repo,
            number,
            "--json",
            "state,mergedAt",
        ],
        github_token=issue_token,
    )
    payload = json.loads(proc.stdout or "{}")
    merged = bool(payload.get("mergedAt"))
    state = str(payload.get("state", "")).strip().upper()
    return {"merged": merged, "state": state, "merged_at": payload.get("mergedAt")}


def message_by_ts(slack_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    messages = slack_payload.get("messages", [])
    if not isinstance(messages, list):
        return out
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        ts = str(msg.get("ts", "")).strip()
        if ts:
            out[ts] = msg
    return out


def message_replies(msg: dict[str, Any]) -> list[dict[str, Any]]:
    replies = msg.get("thread_replies")
    if not isinstance(replies, list):
        replies = msg.get("replies")
    if not isinstance(replies, list):
        return []
    return [row for row in replies if isinstance(row, dict)]


def issue_numbers_from_text(text: str) -> list[int]:
    nums: list[int] = []
    for m in ISSUE_URL_RE.finditer(text):
        value = int(m.group(1))
        if value not in nums:
            nums.append(value)
    return nums


def summary_md(data: dict[str, Any]) -> str:
    lines = [
        "## M5 Lifecycle Summary",
        "",
        f"- Warning messages posted: {len(data.get('warnings_posted', []))}",
        f"- Final warnings posted: {len(data.get('final_warnings_posted', []))}",
        f"- Post-disable follow-ups posted: {len(data.get('post_disable_followups', []))}",
        f"- Issue assignments added: {len(data.get('assignments_added', []))}",
        f"- Fix requests detected: {len(data.get('fix_requests_detected', []))}",
        "",
        "## Notes",
    ]
    for note in data.get("notes", [])[:20]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    issue_token = os.environ.get("ISSUE_WRITE_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    slack_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not issue_token:
        print("ISSUE_WRITE_TOKEN or GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    if not slack_token:
        print("SLACK_BOT_TOKEN is required", file=sys.stderr)
        return 2
    if args.channel_id != SLACK_CHANNEL_TEST:
        print(f"Refusing non-test Slack channel for bootstrap: {args.channel_id}", file=sys.stderr)
        return 2

    slack_payload = json.loads(Path(args.slack_json).read_text(encoding="utf-8"))
    state_path = Path(args.state_json)
    state = load_state(state_path)
    by_ts = message_by_ts(slack_payload)
    rules = parse_codeowners(Path(".github/CODEOWNERS"))
    now = now_unix()
    bot_user_id = slack_auth_user_id(slack_token)
    bot_user_ids: set[str] = {bot_user_id} if bot_user_id else set()
    slack_profile_cache: dict[str, dict[str, Any]] = {}
    slack_to_github_cache: dict[str, str | None] = {}
    owner_claim_cache: dict[str, dict[str, Any]] = {}
    m5_agent_model = os.environ.get("M5_AGENT_MODEL", "").strip() or "auto"

    output: dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "warnings_posted": [],
        "final_warnings_posted": [],
        "post_disable_followups": [],
        "assignments_added": [],
        "fix_requests_detected": [],
        "notes": [],
    }

    for item in state.get("items", []):
        if not isinstance(item, dict):
            continue
        ts = str(item.get("slack_ts", "")).strip()
        if not ts:
            continue
        msg = by_ts.get(ts)
        if not msg:
            output["notes"].append(f"missing_source_message_for_ts:{ts}")
            continue
        issue_numbers = item.get("issue_numbers", [])
        issue_number = int(issue_numbers[0]) if isinstance(issue_numbers, list) and issue_numbers else 0
        if issue_number <= 0:
            output["notes"].append(f"missing_issue_number_for_ts:{ts}")
            continue

        top_text = str(msg.get("text", ""))
        thread_replies = message_replies(msg)
        progress = classify_thread_progress(
            top_level_text=top_text,
            thread_replies=thread_replies,
            now_unix=now,
            hold_hours_after_progress=args.progress_hold_hours,
        )
        fix_request = detect_dev_fix_request(top_level_text=top_text, thread_replies=thread_replies)
        if fix_request.get("requested"):
            output["fix_requests_detected"].append(
                {"source_slack_ts": ts, "issue_number": issue_number, "reason": fix_request.get("reason", "")}
            )

        age_hours = max(0.0, (now - float(ts)) / 3600.0)
        notif = item.setdefault("notification", {})
        if not isinstance(notif, dict):
            notif = {}
            item["notification"] = notif

        issue_closed = bool(msg.get("issue_closed", False))
        if issue_closed:
            output["notes"].append(f"skip_closed_issue:{issue_number}")
            continue

        assignees = issue_assignees(issue_token, issue_number)
        if not assignees:
            thread_fingerprint = hashlib.sha256(
                json.dumps(
                    [{"ts": r.get("ts"), "user": r.get("user"), "text": r.get("text")} for r in thread_replies[-16:]],
                    sort_keys=True,
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            claim_cache_key = f"{ts}:{thread_fingerprint}"
            claim = owner_claim_cache.get(claim_cache_key)
            if claim is None:
                try:
                    claim = thread_owner_claim_via_agent(
                        top_level_text=top_text,
                        thread_replies=thread_replies,
                        bot_user_ids=bot_user_ids,
                        model=m5_agent_model,
                    )
                except Exception as exc:
                    reason = f"agent_owner_claim_failed:{exc}"
                    claim = {"claimed": False, "reason": reason}
                    output["notes"].append(f"thread_owner_claim_agent_error_issue_{issue_number}:{exc}")
                owner_claim_cache[claim_cache_key] = claim
            if claim.get("claimed"):
                claim_user_id = str(claim.get("slack_user_id", "")).strip()
                github_login = slack_to_github_cache.get(claim_user_id)
                if github_login is None and claim_user_id not in slack_to_github_cache:
                    profile = slack_profile_cache.get(claim_user_id)
                    if profile is None:
                        profile = slack_user_profile(slack_token, claim_user_id)
                        slack_profile_cache[claim_user_id] = profile
                    github_login = github_login_from_slack_profile(issue_token, profile)
                    slack_to_github_cache[claim_user_id] = github_login
                if github_login:
                    try:
                        assign_issue(issue_token, issue_number, github_login)
                        output["assignments_added"].append(
                            {
                                "issue_number": issue_number,
                                "assigned_github_login": github_login,
                                "source": "thread_owner_claim",
                                "source_slack_user_id": claim_user_id,
                            }
                        )
                        post_slack_thread_message(
                            slack_token=slack_token,
                            channel=args.channel_id,
                            thread_ts=ts,
                            text=(
                                f"Acknowledged — assigning this incident to <@{claim_user_id}> "
                                f"(GitHub: @{github_login}) based on thread ownership signal."
                            ),
                        )
                        assignees = [github_login]
                    except Exception as exc:
                        output["notes"].append(f"thread_claim_assignment_failed_issue_{issue_number}:{exc}")
                else:
                    output["notes"].append(
                        f"thread_claim_no_github_match_issue_{issue_number}:slack_user={claim_user_id}"
                    )

        # Stage warnings unless thread has active high-confidence progress.
        if not bool(progress.get("defer_disable", False)):
            if age_hours >= args.warning_hours and not notif.get("warning_24h_sent_ts"):
                warning_text = (
                    f"CI auto triage follow-up: this issue is still unresolved after {int(args.warning_hours)}h. "
                    f"If unresolved by {int(args.disable_hours)}h, a disable PR may be proposed."
                )
                posted_ts = post_slack_thread_message(
                    slack_token=slack_token,
                    channel=args.channel_id,
                    thread_ts=ts,
                    text=warning_text,
                )
                notif["warning_24h_sent_ts"] = posted_ts
                output["warnings_posted"].append(
                    {"source_slack_ts": ts, "issue_number": issue_number, "posted_ts": posted_ts}
                )
            if age_hours >= args.final_warning_hours and not notif.get("warning_final_sent_ts"):
                warning_text = (
                    "CI auto triage final warning: issue remains unresolved near disable SLA threshold. "
                    "Please post status or fix PR link to defer disable."
                )
                posted_ts = post_slack_thread_message(
                    slack_token=slack_token,
                    channel=args.channel_id,
                    thread_ts=ts,
                    text=warning_text,
                )
                notif["warning_final_sent_ts"] = posted_ts
                output["final_warnings_posted"].append(
                    {"source_slack_ts": ts, "issue_number": issue_number, "posted_ts": posted_ts}
                )

        # Post-disable follow-up + assignment when disable PR merged.
        disable_pr = item.get("disable_pr", {})
        pr_url = str(disable_pr.get("url", "")).strip() if isinstance(disable_pr, dict) else ""
        if pr_url and not notif.get("post_disable_followup_sent_ts"):
            info = pr_merge_info(issue_token, pr_url)
            if info.get("merged"):
                assigned_login = assignees[0] if assignees else ""
                if not assigned_login:
                    owners = candidate_github_owners_from_text(top_text, rules)
                    if owners:
                        try:
                            assign_issue(issue_token, issue_number, owners[0])
                            assigned_login = owners[0]
                            output["assignments_added"].append(
                                {"issue_number": issue_number, "assigned_github_login": assigned_login}
                            )
                        except Exception as exc:
                            output["notes"].append(f"assignment_failed_issue_{issue_number}:{exc}")
                mention = ""
                if assigned_login:
                    info_user = github_user_info(issue_token, assigned_login)
                    slack_uid = slack_lookup_by_email(slack_token, str(info_user.get("email", "")).strip())
                    if slack_uid:
                        mention = f"<@{slack_uid}> "
                followup = (
                    f"{mention}disable PR merged for this incident ({pr_url}). "
                    "Please follow through on root-cause fix and post fix PR link; "
                    "we should re-enable once stable."
                ).strip()
                posted_ts = post_slack_thread_message(
                    slack_token=slack_token,
                    channel=args.channel_id,
                    thread_ts=ts,
                    text=followup,
                )
                notif["post_disable_followup_sent_ts"] = posted_ts
                output["post_disable_followups"].append(
                    {"source_slack_ts": ts, "issue_number": issue_number, "posted_ts": posted_ts, "pr_url": pr_url}
                )

    # Ad-hoc assignment path for unresolved Slack issues not yet represented in triage state.
    tracked_issue_numbers = {
        int(row.get("issue_number", 0))
        for row in output.get("assignments_added", [])
        if isinstance(row, dict) and int(row.get("issue_number", 0) or 0) > 0
    }
    for msg in slack_payload.get("messages", []):
        if not isinstance(msg, dict):
            continue
        if bool(msg.get("issue_closed", False)):
            continue
        top_text = str(msg.get("text", "")).strip()
        if not top_text:
            continue
        issue_numbers = issue_numbers_from_text(top_text)
        if not issue_numbers:
            continue
        issue_number = issue_numbers[0]
        if issue_number in tracked_issue_numbers:
            continue
        assignees = issue_assignees(issue_token, issue_number)
        if assignees:
            continue
        ts = str(msg.get("ts", "")).strip()
        if not ts:
            continue
        thread_replies = message_replies(msg)
        if not thread_replies:
            continue
        thread_fingerprint = hashlib.sha256(
            json.dumps(
                [{"ts": r.get("ts"), "user": r.get("user"), "text": r.get("text")} for r in thread_replies[-16:]],
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        claim_cache_key = f"{ts}:{thread_fingerprint}"
        claim = owner_claim_cache.get(claim_cache_key)
        if claim is None:
            try:
                claim = thread_owner_claim_via_agent(
                    top_level_text=top_text,
                    thread_replies=thread_replies,
                    bot_user_ids=bot_user_ids,
                    model=m5_agent_model,
                )
            except Exception as exc:
                reason = f"agent_owner_claim_failed:{exc}"
                claim = {"claimed": False, "reason": reason}
                output["notes"].append(f"ad_hoc_owner_claim_agent_error_issue_{issue_number}:{exc}")
            owner_claim_cache[claim_cache_key] = claim
        if not claim.get("claimed"):
            continue
        claim_user_id = str(claim.get("slack_user_id", "")).strip()
        github_login = slack_to_github_cache.get(claim_user_id)
        if github_login is None and claim_user_id not in slack_to_github_cache:
            profile = slack_profile_cache.get(claim_user_id)
            if profile is None:
                profile = slack_user_profile(slack_token, claim_user_id)
                slack_profile_cache[claim_user_id] = profile
            github_login = github_login_from_slack_profile(issue_token, profile)
            slack_to_github_cache[claim_user_id] = github_login
        if not github_login:
            output["notes"].append(
                f"ad_hoc_thread_claim_no_github_match_issue_{issue_number}:slack_user={claim_user_id}"
            )
            continue
        try:
            assign_issue(issue_token, issue_number, github_login)
            output["assignments_added"].append(
                {
                    "issue_number": issue_number,
                    "assigned_github_login": github_login,
                    "source": "ad_hoc_thread_owner_claim",
                    "source_slack_user_id": claim_user_id,
                }
            )
            post_slack_thread_message(
                slack_token=slack_token,
                channel=args.channel_id,
                thread_ts=ts,
                text=(
                    f"Acknowledged — assigning this incident to <@{claim_user_id}> "
                    f"(GitHub: @{github_login}) based on thread ownership signal."
                ),
            )
            tracked_issue_numbers.add(issue_number)
        except Exception as exc:
            output["notes"].append(f"ad_hoc_thread_claim_assignment_failed_issue_{issue_number}:{exc}")

    save_state(state_path, state)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    sm = Path(args.summary_md)
    sm.parent.mkdir(parents=True, exist_ok=True)
    sm.write_text(summary_md(output), encoding="utf-8")
    print(
        json.dumps(
            {
                "warnings_posted": len(output["warnings_posted"]),
                "final_warnings_posted": len(output["final_warnings_posted"]),
                "post_disable_followups": len(output["post_disable_followups"]),
                "fix_requests_detected": len(output["fix_requests_detected"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
