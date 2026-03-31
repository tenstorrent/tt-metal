#!/usr/bin/env python3
"""Execute structured stale-disable actions with deterministic git/gh steps."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import base64
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_PR_REPO = "ebanerjeeTT/tt-metal"
DEFAULT_PR_BASE = "main"
ISSUE_REPO_TEST = "tenstorrent/temporary-issue-dump"
GUARDED_GH = [sys.executable, "tools/ci/guarded_gh.py"]
PROTECTED_AGENT_PATHS = {
    "tools/ci/guarded_gh.py",
    "tools/ci/execute_disable_actions.py",
}
DEFAULT_REQUIRED_PR_CHECK_WORKFLOWS = [
    "all-static-checks.yaml",
]
OPTIONAL_EARLY_WORKFLOWS = {
    "pr-gate.yaml",
    "merge-gate.yaml",
}
ALLOWED_STATUS = {
    "new",
    "planned",
    "pr_open",
    "kickoff_running",
    "kickoff_failed_new_failure",
    "completed",
    "needs_human",
    "paused",
}


def redact_secrets(text: str) -> str:
    if not text:
        return text
    redacted = text
    redacted = re.sub(r"https://x-access-token:[^@]+@github\.com/", "https://x-access-token:***@github.com/", redacted)
    redacted = re.sub(r"AUTHORIZATION:\s*basic\s+[A-Za-z0-9+/=]+", "AUTHORIZATION: basic ***", redacted)
    redacted = re.sub(
        r"http\.https://github\.com/\.extraheader=AUTHORIZATION:\s*basic\s+[A-Za-z0-9+/=]+",
        "http.https://github.com/.extraheader=AUTHORIZATION: basic ***",
        redacted,
    )
    redacted = re.sub(r"\bghp_[A-Za-z0-9]+\b", "ghp_***", redacted)
    secret_keys = (
        "TARGET_PR_PUSH_TOKEN",
        "GITHUB_TOKEN",
        "ISSUE_REPO_GITHUB_TOKEN",
        "CURSOR_API_KEY",
    )
    for key in secret_keys:
        value = os.environ.get(key, "")
        if value:
            redacted = redacted.replace(value, "***")
            encoded_basic = base64.b64encode(f"x-access-token:{value}".encode("utf-8")).decode("ascii")
            redacted = redacted.replace(encoded_basic, "***")
            encoded_raw = base64.b64encode(value.encode("utf-8")).decode("ascii")
            redacted = redacted.replace(encoded_raw, "***")
    return redacted


def run(
    cmd: list[str], *, check: bool = True, capture: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    proc_env = None
    if env:
        proc_env = os.environ.copy()
        proc_env.update(env)
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture,
        env=proc_env,
    )
    if check and proc.returncode != 0:
        stdout = redact_secrets(proc.stdout.strip() if proc.stdout else "")
        stderr = redact_secrets(proc.stderr.strip() if proc.stderr else "")
        cmd_string = redact_secrets(" ".join(shlex.quote(c) for c in cmd))
        raise RuntimeError(
            "Command failed with non-zero exit status "
            f"{proc.returncode}: {cmd_string}\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    return proc


def run_streaming(
    cmd: list[str],
    *,
    check: bool = True,
    heartbeat_label: str = "",
    heartbeat_interval_sec: int = 30,
    timeout_sec: int | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    out_parts: list[str] = []
    err_parts: list[str] = []

    def _pump(stream: Any, sink: list[str], *, to_stderr: bool) -> None:
        if stream is None:
            return
        for line in stream:
            sink.append(line)
            if to_stderr:
                print(line, file=sys.stderr, end="", flush=True)
            else:
                print(line, end="", flush=True)

    t_out = threading.Thread(target=_pump, args=(proc.stdout, out_parts), kwargs={"to_stderr": False})
    t_err = threading.Thread(target=_pump, args=(proc.stderr, err_parts), kwargs={"to_stderr": True})
    t_out.start()
    t_err.start()
    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        if not heartbeat_label:
            return
        started = time.monotonic()
        while not stop_heartbeat.wait(heartbeat_interval_sec):
            elapsed = int(time.monotonic() - started)
            log(f"{heartbeat_label}: still running ({elapsed}s elapsed)")

    t_hb = threading.Thread(target=_heartbeat)
    t_hb.start()
    timed_out = False
    if timeout_sec is None:
        returncode = proc.wait()
    else:
        try:
            returncode = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            returncode = 124
    stop_heartbeat.set()
    t_hb.join()
    t_out.join()
    t_err.join()

    stdout = "".join(out_parts)
    stderr = "".join(err_parts)
    if timed_out:
        timeout_msg = f"Command timed out after {timeout_sec}s: {' '.join(shlex.quote(c) for c in cmd)}"
        if stderr:
            stderr = f"{stderr.rstrip()}\n{timeout_msg}\n"
        else:
            stderr = timeout_msg + "\n"
    if check and returncode != 0:
        cmd_string = redact_secrets(" ".join(shlex.quote(c) for c in cmd))
        raise RuntimeError(
            "Command failed with non-zero exit status "
            f"{returncode}: {cmd_string}\n"
            f"stdout:\n{redact_secrets(stdout.strip())}\n\nstderr:\n{redact_secrets(stderr.strip())}"
        )
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def run_guarded_gh(tokens: list[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
    command_str = " ".join(shlex.quote(tok) for tok in tokens)
    return run([*GUARDED_GH, "--command", command_str], capture=capture, check=check)


def prepare_guarded_gh_runtime_copy() -> None:
    source = Path("tools/ci/guarded_gh.py")
    if not source.exists():
        raise RuntimeError(f"Missing guarded gh wrapper at {source}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="guarded-gh-"))
    runtime_script = tmp_dir / "guarded_gh_runtime.py"
    shutil.copy2(source, runtime_script)
    GUARDED_GH[:] = [sys.executable, str(runtime_script)]


def slugify(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9-]+", "-", text).strip("-").lower()
    return clean[:48] if clean else "auto"


def branch_name(action: dict[str, Any]) -> str:
    hint = str(action.get("branch_name_hint", "")).strip()
    if hint:
        return slugify(hint)
    issue = action.get("issue_number")
    ts = str(action.get("source_slack_ts", "")).replace(".", "")
    return f"ci-disable-test-{issue}-{ts[-8:]}"


def branch_exists_remote(branch: str, repo_slug: str) -> bool:
    remote_url = f"https://github.com/{repo_slug}.git"
    check = run(["git", "ls-remote", "--heads", remote_url, f"refs/heads/{branch}"], check=False, capture=True)
    return bool((check.stdout or "").strip())


def choose_branch_name(base: str, source_ts: str, attempt: int, repo_slug: str) -> str:
    if not branch_exists_remote(base, repo_slug):
        return base
    ts = source_ts.replace(".", "")
    suffix_seed = ts[-6:] if ts else "retry"
    for idx in range(1, 50):
        candidate = f"{base}-r{attempt}-{suffix_seed}-{idx}"
        if not branch_exists_remote(candidate, repo_slug):
            return candidate
    raise RuntimeError(f"unable to allocate unique branch name from base {base!r}")


def parse_repo_from_pr_url(pr_url: str) -> str:
    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/\d+", pr_url)
    return m.group(1) if m else ""


def ensure_no_duplicate_open_pr(source_ts: str, pr_repo: str) -> str | None:
    marker = f"Auto-disable-source-ts: {source_ts}"
    prs = run_guarded_gh(["gh", "pr", "list", "--repo", pr_repo, "--state", "open", "--json", "number,url,body"])
    items = json.loads(prs.stdout)
    for pr in items:
        if marker in (pr.get("body") or ""):
            return pr.get("url")
    return None


def is_pr_open(pr_url: str, default_repo: str) -> bool:
    pr_number = parse_pr_number(pr_url)
    if pr_number <= 0:
        return False
    pr_repo = parse_repo_from_pr_url(pr_url) or default_repo
    viewed = run_guarded_gh(
        ["gh", "pr", "view", "--repo", pr_repo, str(pr_number), "--json", "state"],
        check=False,
    )
    if viewed.returncode != 0:
        return False
    try:
        payload = json.loads(viewed.stdout or "{}")
    except json.JSONDecodeError:
        return False
    return str(payload.get("state", "")).upper() == "OPEN"


def parse_agent_json_after_marker(text: str, marker: str) -> dict[str, Any]:
    idx = text.rfind(marker)
    marker_end = idx + len(marker) if idx >= 0 else -1
    if idx < 0:
        marker_re = re.compile(rf"`?\s*{re.escape(marker)}\s*`?")
        matches = list(marker_re.finditer(text))
        if matches:
            marker_end = matches[-1].end()
    if marker_end < 0:
        raise ValueError(f"marker not found: {marker}")

    payload = text[marker_end:].strip()
    if payload.startswith("```"):
        # Accept fenced JSON payloads while still requiring JSON parseability.
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
        payload = payload.strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Fall back to parsing from the first JSON object start after the marker.
        brace_idx = payload.find("{")
        if brace_idx < 0:
            raise
        return json.loads(payload[brace_idx:])


def load_disable_command_spec() -> str:
    fallback_spec = """# CI Disable Test (Automation Mode)

## Purpose
Apply the smallest safe CI-only disable for a failing signal, but do not perform git/gh operations.

## Input
- Required: one GitHub issue URL/number in tenstorrent/temporary-issue-dump during testing, or in tenstorrent/tt-metal after production promotion.
- Optional: one or more job URLs for stronger evidence.

## Hard Constraints
- You may read files and edit files in this repository.
- You may use gh only for read operations needed to inspect issue/run context and logs.
- Do not create branches.
- Do not run git add, git commit, git push, or gh pr create.
- Do not dispatch workflows.

## Procedure
1. Resolve issue/job context from provided input.
2. Download and inspect failed logs in build_ci/disabling.
3. Identify the narrowest disable scope that is supported by evidence.
4. Apply only minimal code/workflow edits needed to disable the failing target.
5. Add a TODO comment near the disable with non-closing issue reference.
6. Clean transient logs from build_ci/disabling.

## Required Final Output
At the end of your response, print this exact marker on its own line:
===FINAL_DISABLE_EDIT_SUMMARY===

After the marker, print only compact JSON with this schema:
{
  "issue_number": 0,
  "disable_scope": "short description",
  "files_modified": ["path1", "path2"],
  "notes": "optional short note"
}
"""
    command_path = Path(".cursor/commands/ci/ci-disable-test-ci.md")
    if not command_path.exists():
        return fallback_spec
    return command_path.read_text(encoding="utf-8")


def safe_slug(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-")
    return clean or "item"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_disable_editor(action: dict[str, Any], issue_url: str, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    issue_number = int(action["issue_number"])
    log(f"disable_editor: start issue #{issue_number} with model={model}")
    job_urls = action.get("job_urls", [])
    if not isinstance(job_urls, list):
        job_urls = []
    job_urls = [str(u).strip() for u in job_urls if str(u).strip()]
    scope_hint = str(action.get("disable_scope_hint", "")).strip()
    slack_link = str(action.get("source_slack_permalink", "")).strip()

    command_spec = load_disable_command_spec()
    prompt = (
        "Use the following command specification exactly:\n\n"
        f"{command_spec}\n\n"
        f"Input issue: {issue_url}\n"
        + (f"Evidence job URLs: {', '.join(job_urls)}\n" if job_urls else "")
        + (f"Disable scope hint: {scope_hint}\n" if scope_hint else "")
        + (f"Source Slack: {slack_link}\n" if slack_link else "")
        + "If evidence is weak, make no code edits and explain in JSON summary.\n"
        + "You must emit the marker and JSON contract from the command specification."
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model != "auto":
        cmd[1:1] = ["--model", model]
    result = run_streaming(cmd, heartbeat_label=f"disable_editor issue #{issue_number}")
    debug: dict[str, Any] = {
        "used_retry": False,
        "primary_stdout": result.stdout,
        "primary_stderr": result.stderr,
    }
    try:
        summary = parse_agent_json_after_marker(result.stdout, "===FINAL_DISABLE_EDIT_SUMMARY===")
        log(f"disable_editor: parsed summary on primary response for issue #{issue_number}")
        return summary, debug
    except Exception as exc:
        log(f"disable_editor: primary parse failed for issue #{issue_number}; retrying summary-only response")
        debug["primary_parse_error"] = str(exc)
        # Retry with a strict summary-only prompt in case the main run omitted marker formatting.
        retry_prompt = (
            "Output only the required marker and compact JSON summary.\n"
            "Do not run tools and do not edit files.\n"
            "Use marker: ===FINAL_DISABLE_EDIT_SUMMARY===\n"
            f"Set issue_number to {issue_number} and provide a concise disable_scope/notes."
        )
        retry_cmd = ["agent", "--trust", "-p", retry_prompt]
        if model != "auto":
            retry_cmd[1:1] = ["--model", model]
        retry = run_streaming(retry_cmd, heartbeat_label=f"disable_editor retry issue #{issue_number}")
        debug["used_retry"] = True
        debug["retry_stdout"] = retry.stdout
        debug["retry_stderr"] = retry.stderr
        summary = parse_agent_json_after_marker(retry.stdout, "===FINAL_DISABLE_EDIT_SUMMARY===")
        log(f"disable_editor: parsed summary on retry for issue #{issue_number}")
        return summary, debug


def invoke_kickoff_agent(pr_url: str, model: str) -> str:
    prompt = (
        "Follow .cursor/commands/ci/ci-kickoff-workflows.md exactly.\n"
        f"Input PR URL: {pr_url}\n"
        "Automatically proceed end-to-end without asking for additional confirmation."
    )
    cmd = ["agent", "--trust", "-p", prompt]
    if model != "auto":
        cmd[1:1] = ["--model", model]
    result = run_streaming(
        cmd,
        check=False,
        heartbeat_label=f"kickoff_agent for {pr_url}",
        timeout_sec=300,
    )
    if result.returncode == 124:
        log(f"kickoff_agent: timed out after 300s for {pr_url}; continuing without blocking")
    elif result.returncode != 0:
        log(f"kickoff_agent: exited non-zero ({result.returncode}) for {pr_url}; continuing")
    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    return combined[-2000:]


def parse_first_url(text: str) -> str:
    for line in (text or "").splitlines():
        candidate = line.strip()
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate
    return ""


def parse_extra_workflows(raw: str) -> list[str]:
    workflows: list[str] = []
    for entry in (raw or "").split(","):
        workflow = entry.strip()
        if not workflow:
            continue
        if workflow not in OPTIONAL_EARLY_WORKFLOWS:
            raise ValueError(f"unsupported extra workflow {workflow!r}; allowed: {sorted(OPTIONAL_EARLY_WORKFLOWS)}")
        workflows.append(workflow)
    return workflows


def dispatch_required_pr_check_workflows(branch: str, workflows: list[str], pr_repo: str) -> dict[str, str]:
    runs: dict[str, str] = {}
    for workflow in workflows:
        log(f"dispatch: triggering workflow {workflow} for branch {branch}")
        dispatched = run_guarded_gh(["gh", "workflow", "run", workflow, "--repo", pr_repo, "--ref", branch])
        runs[workflow] = parse_first_url(dispatched.stdout)
        if runs[workflow]:
            log(f"dispatch: workflow {workflow} run URL {runs[workflow]}")
        else:
            log(f"dispatch: workflow {workflow} dispatched (run URL not returned)")
    return runs


def git_changed_files() -> list[str]:
    out = run(["git", "status", "--porcelain"], capture=True).stdout.strip().splitlines()
    files: list[str] = []
    for line in out:
        if len(line) > 3:
            files.append(line[3:])
    return files


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[{now_utc()}] {message}", flush=True)


def state_key_for_ts(source_ts: str) -> str:
    return f"slack_ts:{source_ts}"


def parse_pr_number(pr_url: str) -> int:
    m = re.search(r"/pull/(\d+)", pr_url)
    if not m:
        return 0
    return int(m.group(1))


def post_triggered_workflows_comment(pr_url: str, runs: dict[str, str], pr_repo: str) -> None:
    pr_number = parse_pr_number(pr_url)
    if pr_number <= 0 or not runs:
        return
    lines = [
        "Auto-triage triggered these workflows:",
        "",
    ]
    for workflow, run_url in sorted(runs.items()):
        if run_url:
            lines.append(f"- `{workflow}`: {run_url}")
        else:
            lines.append(f"- `{workflow}`: dispatched (run URL unavailable)")
    body = "\n".join(lines).strip()
    run_guarded_gh(
        [
            "gh",
            "pr",
            "comment",
            "--repo",
            pr_repo,
            str(pr_number),
            "--body",
            body,
        ]
    )


def pr_label(pr_url: str) -> str:
    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", pr_url)
    if not m:
        return pr_url
    return f"{m.group(1)}#{m.group(2)}"


def empty_state() -> dict[str, Any]:
    return {"version": 1, "updated_at_utc": now_utc(), "items": []}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("state must be a JSON object")
    if not isinstance(data.get("items"), list):
        raise ValueError("state.items must be a list")
    seen: set[str] = set()
    for item in data["items"]:
        if not isinstance(item, dict):
            raise ValueError("state items must be objects")
        key = str(item.get("key", ""))
        if not key:
            raise ValueError("state item missing key")
        if key in seen:
            raise ValueError(f"duplicate state key: {key}")
        seen.add(key)
        status = str(item.get("status", ""))
        if status not in ALLOWED_STATUS:
            raise ValueError(f"invalid state status for {key}: {status}")
        attempts = item.get("attempts", 0)
        if not isinstance(attempts, int) or attempts < 0:
            raise ValueError(f"invalid attempts for {key}")
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = now_utc()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def state_index(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["key"]): item for item in state.get("items", []) if isinstance(item, dict) and "key" in item}


def append_history(item: dict[str, Any], event: str, details: str) -> None:
    history = item.setdefault("history", [])
    if not isinstance(history, list):
        item["history"] = []
        history = item["history"]
    history.append({"ts_utc": now_utc(), "event": event, "details": details})


def ensure_state_item(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    source_ts = str(action["source_slack_ts"])
    key = state_key_for_ts(source_ts)
    idx = state_index(state)
    if key in idx:
        return idx[key]
    issue = int(action["issue_number"])
    item = {
        "key": key,
        "slack_ts": source_ts,
        "issue_numbers": [issue],
        "status": "new",
        "disable_pr": {"number": 0, "url": "", "branch": "", "head_sha": ""},
        "attempts": 0,
        "last_kickoff_runs": [],
        "notification": {"terminal_notified": False, "last_error": ""},
        "terminal_reason": "",
        "history": [],
    }
    append_history(item, "state_created", f"Initialized from issue #{issue}")
    state["items"].append(item)
    return item


def set_status(item: dict[str, Any], status: str, *, event: str, details: str) -> None:
    if status not in ALLOWED_STATUS:
        raise ValueError(f"invalid status transition target: {status}")
    item["status"] = status
    append_history(item, event, details)


def state_has_active_pr(item: dict[str, Any]) -> str | None:
    status = str(item.get("status", ""))
    if status not in {"pr_open", "kickoff_running", "kickoff_failed_new_failure"}:
        return None
    disable_pr = item.get("disable_pr")
    if not isinstance(disable_pr, dict):
        return None
    pr_url = str(disable_pr.get("url", "")).strip()
    return pr_url or None


def tracked_pr_url(item: dict[str, Any]) -> str:
    disable_pr = item.get("disable_pr")
    if not isinstance(disable_pr, dict):
        return ""
    return str(disable_pr.get("url", "")).strip()


def ensure_git_identity() -> None:
    name = run(["git", "config", "--get", "user.name"], check=False, capture=True)
    email = run(["git", "config", "--get", "user.email"], check=False, capture=True)
    if name.returncode != 0 or not (name.stdout or "").strip():
        run(["git", "config", "user.name", "CI Auto Disable Bot"], capture=True)
    if email.returncode != 0 or not (email.stdout or "").strip():
        run(["git", "config", "user.email", "ci-auto-disable-bot@tenstorrent.invalid"], capture=True)


def push_branch_with_token(branch: str, target_pr_repo: str, token: str) -> None:
    push_url = f"https://github.com/{target_pr_repo}.git"
    askpass_script = Path(tempfile.mkdtemp(prefix="git-askpass-")) / "askpass.sh"
    askpass_script.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  *Username*) echo "x-access-token" ;;\n'
        '  *) echo "$GIT_PUSH_TOKEN" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    askpass_script.chmod(0o700)
    run(
        ["git", "push", "-u", push_url, f"HEAD:refs/heads/{branch}"],
        capture=True,
        env={
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": str(askpass_script),
            "GIT_PUSH_TOKEN": token,
        },
    )


def write_summary(path: Path, data: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Auto Disable Actions")
    lines.append("")
    lines.append(f"- Planned actions: {data.get('planned_actions', 0)}")
    lines.append(f"- Executed actions: {len(data.get('executed', []))}")
    lines.append(f"- Skipped actions: {len(data.get('skipped', []))}")
    lines.append(f"- State updates: {data.get('state_updates', 0)}")
    lines.append("")
    lines.append("## State Status Counts")
    for status, count in sorted((data.get("state_status_counts") or {}).items()):
        lines.append(f"- {status}: {count}")
    lines.append("")
    lines.append("## Executed")
    for item in data.get("executed", []):
        issue_number = item.get("issue_number")
        pr_url = str(item.get("pr_url", "")).strip()
        lines.append(f"- issue #{issue_number} -> PR: {pr_label(pr_url)}")
    lines.append("")
    lines.append("## Skipped")
    for item in data.get("skipped", []):
        lines.append(f"- ts {item.get('source_slack_ts')}: {item.get('reason')}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute auto-disable action JSON.")
    parser.add_argument("--actions-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--debug-dir", default="")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--max-attempts-per-item", type=int, default=10)
    parser.add_argument("--target-pr-repo", default=DEFAULT_PR_REPO)
    parser.add_argument("--target-pr-base", default=DEFAULT_PR_BASE)
    parser.add_argument(
        "--extra-pr-check-workflows",
        default="",
        help="Comma-separated optional early workflows to run after PR create (allowed: pr-gate.yaml, merge-gate.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target_pr_repo = args.target_pr_repo.strip() or DEFAULT_PR_REPO
    target_pr_base = args.target_pr_base.strip() or DEFAULT_PR_BASE
    extra_pr_check_workflows = parse_extra_workflows(args.extra_pr_check_workflows)
    log(
        "execute_disable_actions: start "
        f"dry_run={args.dry_run} model={args.model} max_attempts_per_item={args.max_attempts_per_item} "
        f"target_pr_repo={target_pr_repo}"
    )

    if not args.dry_run:
        if not os.environ.get("GITHUB_TOKEN"):
            print("GITHUB_TOKEN is required", file=sys.stderr)
            return 2
        if not os.environ.get("CURSOR_API_KEY"):
            print("CURSOR_API_KEY is required", file=sys.stderr)
            return 2

    actions_doc = json.loads(Path(args.actions_json).read_text(encoding="utf-8"))
    actions = actions_doc.get("actions", [])
    if not isinstance(actions, list):
        print("Invalid actions JSON: actions must be a list", file=sys.stderr)
        return 2

    # Validate and dedupe by source ts.
    dedup: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        source_ts = str(action.get("source_slack_ts", "")).strip()
        issue = action.get("issue_number")
        if not source_ts or not isinstance(issue, int):
            continue
        dedup[source_ts] = action
    validated_actions = list(dedup.values())
    log(f"execute_disable_actions: validated {len(validated_actions)} actions after input filtering")
    state_path = Path(args.state_json)
    state = load_state(state_path)
    log(f"execute_disable_actions: loaded state from {state_path}")

    result: dict[str, Any] = {
        "planned_actions": len(validated_actions),
        "executed": [],
        "skipped": [],
        "dry_run": args.dry_run,
        "state_path": str(state_path),
        "state_updates": 0,
    }
    debug_root = Path(args.debug_dir) if args.debug_dir else None
    if debug_root is not None:
        debug_root.mkdir(parents=True, exist_ok=True)
        result["debug_dir"] = str(debug_root)

    if not args.dry_run:
        log("execute_disable_actions: preparing guarded gh runtime copy and auth checks")
        prepare_guarded_gh_runtime_copy()
        run_guarded_gh(["gh", "auth", "status"])
        run(["git", "fetch", "origin", "main"], capture=True)
        log("execute_disable_actions: git fetch origin/main complete")

    for action in validated_actions:
        source_ts = str(action["source_slack_ts"])
        issue_number = int(action["issue_number"])
        log(f"action: evaluating issue #{issue_number} source_ts={source_ts}")
        issue_repo = str(action.get("issue_repo", ISSUE_REPO_TEST)).strip() or ISSUE_REPO_TEST
        issue_url = str(action.get("issue_url", "")).strip() or f"https://github.com/{issue_repo}/issues/{issue_number}"
        item = ensure_state_item(state, action)

        if item["status"] in {"completed", "paused"}:
            result["skipped"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "reason": f"terminal_state:{item['status']}",
                }
            )
            append_history(item, "skip_terminal_state", f"Skipped because status is {item['status']}")
            continue

        active_pr_url = tracked_pr_url(item)
        if active_pr_url and not args.dry_run:
            active_pr_repo = parse_repo_from_pr_url(active_pr_url)
            if active_pr_repo and active_pr_repo != target_pr_repo:
                log(
                    f"action: tracked PR repo mismatch for issue #{issue_number}; "
                    f"expected {target_pr_repo}, got {active_pr_repo}; resetting state for retarget"
                )
                item["disable_pr"] = {"number": 0, "url": "", "branch": "", "head_sha": ""}
                item["attempts"] = 0
                set_status(
                    item,
                    "new",
                    event="tracked_pr_repo_mismatch_reset",
                    details=f"Reset PR metadata and attempts from repo {active_pr_repo} to target {target_pr_repo}",
                )
                result["state_updates"] += 1
                active_pr_url = ""
        if active_pr_url and not args.dry_run and not is_pr_open(active_pr_url, target_pr_repo):
            log(f"action: tracked PR is no longer open for issue #{issue_number}: {active_pr_url}")
            item["disable_pr"] = {"number": 0, "url": "", "branch": "", "head_sha": ""}
            set_status(
                item,
                "new",
                event="active_pr_not_open_anymore",
                details=f"Previously tracked PR is no longer open: {active_pr_url}",
            )
            result["state_updates"] += 1
            active_pr_url = ""
        if active_pr_url:
            log(f"action: skipping issue #{issue_number} because tracked open PR exists: {active_pr_url}")
            result["skipped"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "reason": f"active_tracked_pr:{active_pr_url}",
                }
            )
            append_history(item, "skip_active_state_pr", f"Skipped action because active PR exists: {active_pr_url}")
            continue

        if int(item.get("attempts", 0)) >= args.max_attempts_per_item:
            set_status(
                item,
                "needs_human",
                event="max_attempts_exceeded",
                details=f"attempts={item.get('attempts', 0)} threshold={args.max_attempts_per_item}",
            )
            result["state_updates"] += 1
            result["skipped"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "reason": "max_attempts_exceeded",
                }
            )
            continue

        pr_title = str(action.get("pr_title", "")).strip() or f"ci: disable failing test for #{issue_number}"
        pr_body = str(action.get("pr_body", "")).strip()
        if not pr_body:
            pr_body = (
                f"Refs #{issue_number}\n\n"
                f"Source Issue: {issue_url}\n"
                f"Auto-disable-source-ts: {source_ts}\n"
                f"Source Slack: {action.get('source_slack_permalink', '')}\n"
            )
        if f"Auto-disable-source-ts: {source_ts}" not in pr_body:
            pr_body += f"\n\nAuto-disable-source-ts: {source_ts}\n"

        existing = None if args.dry_run else ensure_no_duplicate_open_pr(source_ts, target_pr_repo)
        if existing:
            log(f"action: found matching open PR marker for issue #{issue_number}: {existing}")
            item["disable_pr"]["url"] = existing
            item["disable_pr"]["number"] = parse_pr_number(existing)
            set_status(item, "pr_open", event="existing_pr_detected", details=f"Found open PR {existing}")
            result["state_updates"] += 1
            result["skipped"].append(
                {"source_slack_ts": source_ts, "issue_number": issue_number, "reason": f"already_open_pr:{existing}"}
            )
            continue

        branch = branch_name(action)
        if not args.dry_run:
            branch = choose_branch_name(branch, source_ts, int(item.get("attempts", 0)) + 1, target_pr_repo)
            if branch != branch_name(action):
                log(f"action: adjusted branch name for issue #{issue_number} to avoid remote collision: {branch}")
        if args.dry_run:
            log(f"action: dry-run planned for issue #{issue_number} on branch {branch}")
            set_status(
                item,
                "planned",
                event="dry_run_planned_disable",
                details=f"Would create or update branch {branch}",
            )
            result["state_updates"] += 1
            result["executed"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "branch": branch,
                    "pr_url": "(dry-run)",
                    "kickoff_output_tail": "(dry-run)",
                }
            )
            continue

        try:
            log(f"action: checking out branch {branch} for issue #{issue_number}")
            run(["git", "checkout", "-B", branch, "origin/main"], capture=True)
            before = set(git_changed_files())
            if before:
                result["skipped"].append(
                    {
                        "source_slack_ts": source_ts,
                        "issue_number": issue_number,
                        "reason": "working_tree_not_clean_before_action",
                    }
                )
                append_history(item, "skip_dirty_tree", "Working tree not clean before action")
                continue

            item["attempts"] = int(item.get("attempts", 0)) + 1
            set_status(
                item, "planned", event="attempt_started", details=f"Attempt {item['attempts']} on branch {branch}"
            )
            result["state_updates"] += 1

            edit_summary, editor_debug = run_disable_editor(action, issue_url, args.model)
            if debug_root is not None:
                item_dir = debug_root / f"{safe_slug(source_ts)}_issue_{issue_number}"
                write_text(item_dir / "disable_editor_primary.stdout.txt", str(editor_debug.get("primary_stdout", "")))
                write_text(item_dir / "disable_editor_primary.stderr.txt", str(editor_debug.get("primary_stderr", "")))
                if editor_debug.get("used_retry"):
                    write_text(item_dir / "disable_editor_retry.stdout.txt", str(editor_debug.get("retry_stdout", "")))
                    write_text(item_dir / "disable_editor_retry.stderr.txt", str(editor_debug.get("retry_stderr", "")))
                write_text(item_dir / "disable_edit_summary.json", json.dumps(edit_summary, indent=2))
            changed = git_changed_files()
            if not changed:
                log(f"action: no code changes produced by disable editor for issue #{issue_number}")
                summary_note = str(edit_summary.get("notes", "")).strip() if isinstance(edit_summary, dict) else ""
                skip_reason = "no_code_changes_from_agent"
                if summary_note:
                    skip_reason += f":{summary_note}"
                set_status(
                    item,
                    "needs_human",
                    event="no_changes_from_disable_editor",
                    details=f"Attempt {item['attempts']} produced no code changes. {summary_note}".strip(),
                )
                result["state_updates"] += 1
                result["skipped"].append(
                    {
                        "source_slack_ts": source_ts,
                        "issue_number": issue_number,
                        "reason": skip_reason,
                        "disable_edit_summary": edit_summary,
                    }
                )
                continue
            protected = sorted({p for p in changed if p in PROTECTED_AGENT_PATHS})
            if protected:
                log(f"action: protected paths modified for issue #{issue_number}; escalating to needs_human")
                set_status(
                    item,
                    "needs_human",
                    event="protected_paths_modified",
                    details=f"Attempt {item['attempts']} modified protected paths: {', '.join(protected)}",
                )
                result["state_updates"] += 1
                result["skipped"].append(
                    {
                        "source_slack_ts": source_ts,
                        "issue_number": issue_number,
                        "reason": f"protected_paths_modified:{','.join(protected)}",
                        "disable_edit_summary": edit_summary,
                    }
                )
                continue

            run(["git", "add", "."], capture=True)
            commit_msg = f"ci: disable failing test for #{issue_number}"
            ensure_git_identity()
            log(f"action: committing disable changes for issue #{issue_number}")
            run(["git", "commit", "-m", commit_msg], capture=True)
            log(f"action: pushing branch {branch} for issue #{issue_number}")
            push_token = os.environ.get("TARGET_PR_PUSH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
            if not push_token:
                raise RuntimeError("TARGET_PR_PUSH_TOKEN or GITHUB_TOKEN is required for PR branch push")
            push_branch_with_token(branch, target_pr_repo, push_token)

            log(f"action: creating draft PR for issue #{issue_number}")
            pr = run_guarded_gh(
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    target_pr_repo,
                    "--draft",
                    "--base",
                    target_pr_base,
                    "--head",
                    branch,
                    "--title",
                    pr_title,
                    "--body",
                    pr_body,
                    "--label",
                    "CI auto triage",
                ]
            )
            pr_url = pr.stdout.strip().splitlines()[-1].strip()
            log(f"action: created PR {pr_url} for issue #{issue_number}")
            required_check_runs = dispatch_required_pr_check_workflows(
                branch, [*DEFAULT_REQUIRED_PR_CHECK_WORKFLOWS, *extra_pr_check_workflows], target_pr_repo
            )
            post_triggered_workflows_comment(pr_url, required_check_runs, target_pr_repo)
            log(f"action: invoking kickoff workflow agent for PR {pr_url}")
            kickoff_tail = invoke_kickoff_agent(pr_url, args.model)
            item["disable_pr"] = {
                "number": parse_pr_number(pr_url),
                "url": pr_url,
                "branch": branch,
                "head_sha": "",
            }
            set_status(item, "kickoff_running", event="pr_created_and_kickoff_started", details=pr_url)
            result["state_updates"] += 1

            result["executed"].append(
                {
                    "source_slack_ts": source_ts,
                    "issue_number": issue_number,
                    "branch": branch,
                    "pr_url": pr_url,
                    "required_check_runs": required_check_runs,
                    "disable_edit_summary": edit_summary,
                    "kickoff_output_tail": kickoff_tail,
                    "attempts": item["attempts"],
                }
            )
        except Exception as exc:
            safe_exc = redact_secrets(str(exc))
            log(f"action: failed issue #{issue_number}: {safe_exc}")
            set_status(item, "needs_human", event="action_failed", details=safe_exc)
            result["state_updates"] += 1
            result["skipped"].append(
                {"source_slack_ts": source_ts, "issue_number": issue_number, "reason": f"action_failed:{safe_exc}"}
            )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for item in state.get("items", []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    result["state_status_counts"] = counts
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_state(state_path, state)
    write_summary(Path(args.summary_md), result)
    log(f"execute_disable_actions: wrote outputs to {out_path} and {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
