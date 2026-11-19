#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The fixed steps of the LLK PR review workflow, on both sides of the AI step.

The AI step reads pull-request text that anyone can write, so it gets no shell
and no GitHub access. These two commands do what it used to do with them:

    llk_pr_review.py inputs --repo OWNER/REPO --pr N --base-sha S --head-sha S \
        --base-tip-sha S --out DIR
        Write everything the AI may read about the pull request as files:
        metadata, the exact-SHA diff (whole and per file), the discussion so
        far, the PR's commits and the recent history of each changed file.

    llk_pr_review.py post --input DIR [--findings FILE] [--dry-run]
        Publish the AI's findings as review comments on that pull request, and
        nowhere else. Refuses output that looks like it holds a credential.

Both use `git` in the current directory and `gh` with the token in GH_TOKEN.
Every GitHub call is one of the fixed endpoints of the one pull request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MARKER = "<!-- llk-pr-reviewer -->"
SIGNATURE = "_Posted by the **LLK PR Reviewer**._"
FOOTER = f"{MARKER}\n{SIGNATURE}"

# What the AI step returns (the workflow passes the same schema to --json-schema).
FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer"},
                    "start_line": {"type": "integer"},
                    "body": {"type": "string"},
                },
                "required": ["path", "line", "body"],
            },
        },
    },
    "required": ["findings"],
}

MAX_FINDINGS = 50
# The review procedure delegates to at least this many agents (the CLAUDE.md
# lookup, the intake and four reviewers). A session that ended with fewer did not
# review the pull request, whatever it returned.
MIN_AGENT_CALLS = 6
MAX_BODY_CHARS = 6000
MAX_TEXT_CHARS = 20000          # one PR body or comment, as given to the AI
MAX_DISCUSSION_ITEMS = 400
HISTORY_COMMITS = 10

# Text that looks like a credential. A review comment never needs one, so output
# that holds one is not posted at all.
CREDENTIAL_PATTERNS = {
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "GitHub fine-grained token": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "Anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"),
    "JSON web token": re.compile(r"eyJ[A-Za-z0-9_-]{30,}"),
    "private key": re.compile(r"-----BEGIN"),
}
_URL = re.compile(r"https?://[^\s<>()\[\]\"'`]+")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ReviewError(RuntimeError):
    """A step could not do its work; the message says why."""


def _run(argv: list[str], *, timeout: int = 300) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise ReviewError(f"{' '.join(argv[:3])} failed: {result.stderr.strip()[-400:]}")
    return result.stdout


def _git(*args: str) -> str:
    return _run(["git", *args])


def _gh_json(*args: str):
    return json.loads(_run(["gh", "api", *args], timeout=120) or "null")


def _gh_pages(endpoint: str) -> list:
    """Every page of a list endpoint, as one list."""
    pages = _gh_json("--paginate", "--slurp", endpoint) or []
    return [item for page in pages for item in page]


def _cut(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= MAX_TEXT_CHARS else text[:MAX_TEXT_CHARS] + "\n[cut]"


# ── inputs ────────────────────────────────────────────────────────────────────
def changed_files(base: str, head: str) -> list[dict]:
    """The files of the exact-SHA diff, renames followed."""
    files = []
    for line in _git("diff", "--name-status", "-M", base, head).splitlines():
        status, *paths = line.split("\t")
        entry = {"path": paths[-1], "status": status[0]}
        if len(paths) == 2:
            entry["previous_path"] = paths[0]
        files.append(entry)
    return files


def shallow_commits() -> set:
    """The commits where a depth-limited fetch cut the history off."""
    try:
        return set((Path(_git("rev-parse", "--git-common-dir").strip()) / "shallow")
                   .read_text(encoding="utf-8").split())
    except OSError:
        return set()


def file_history(head: str, paths: list[str], cut: set) -> str:
    """The latest commits that touched ``paths``, as far as the fetched history goes.

    A commit at the cut has no parent here, so git shows it as adding every file:
    it is left out, and the text says where the history stops.
    """
    lines = []
    for line in _git("log", f"-n{HISTORY_COMMITS}", "--date=short",
                     "--format=%H %h %ad %an: %s", head, "--", *paths).splitlines():
        sha, _, rest = line.partition(" ")
        if sha in cut:
            lines.append("(older history was not fetched)")
            break
        lines.append(rest)
    return "\n".join(lines) + "\n"


def write_inputs(repo: str, pr: int, base: str, head: str, base_tip: str, out: Path) -> dict:
    """Write the review inputs under ``out`` and return the pull request's metadata."""
    data = _gh_json(f"repos/{repo}/pulls/{pr}")
    cut = shallow_commits()
    (out / "diff" / "files").mkdir(parents=True)
    (out / "history").mkdir()
    (out / "diff" / "full.diff").write_text(_git("diff", "--no-color", "-M", base, head),
                                            encoding="utf-8")
    files = changed_files(base, head)
    for index, entry in enumerate(files, 1):
        paths = [entry[key] for key in ("previous_path", "path") if key in entry]
        entry["diff"] = f"diff/files/{index:04d}.diff"
        entry["history"] = f"history/{index:04d}.txt"
        (out / entry["diff"]).write_text(
            _git("diff", "--no-color", "-M", base, head, "--", *paths), encoding="utf-8")
        (out / entry["history"]).write_text(file_history(head, paths, cut), encoding="utf-8")
    (out / "commits.txt").write_text(
        _git("log", "--date=short", "--stat", "--format=commit %h %ad %an%n%s%n%b",
             f"{base}..{head}"), encoding="utf-8")
    meta = {
        "repository": repo, "number": pr, "url": data.get("html_url"),
        "title": data.get("title") or "", "body": _cut(data.get("body")),
        "author": (data.get("user") or {}).get("login"),
        "state": data.get("state"), "draft": bool(data.get("draft")),
        "labels": [label.get("name") for label in data.get("labels") or []],
        "base_branch": (data.get("base") or {}).get("ref"),
        "base_tip_sha": base_tip, "comparison_base_sha": base, "reviewed_head_sha": head,
        "head_sha_on_github": (data.get("head") or {}).get("sha"),
        "changed_files": files,
    }
    (out / "pr.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def comment(item: dict, **more) -> dict:
        return {"author": (item.get("user") or {}).get("login"),
                "created_at": item.get("created_at") or item.get("submitted_at"),
                "body": _cut(item.get("body")), **more}

    discussion = {
        "issue_comments": [comment(item) for item in
                           _gh_pages(f"repos/{repo}/issues/{pr}/comments")],
        "review_comments": [comment(item, path=item.get("path"),
                                    line=item.get("line") or item.get("original_line"),
                                    start_line=item.get("start_line"))
                            for item in _gh_pages(f"repos/{repo}/pulls/{pr}/comments")],
        "reviews": [comment(item, state=item.get("state")) for item in
                    _gh_pages(f"repos/{repo}/pulls/{pr}/reviews")
                    if (item.get("body") or "").strip() or item.get("state") != "COMMENTED"],
    }
    for name, items in discussion.items():
        discussion[name] = items[-MAX_DISCUSSION_ITEMS:]
    (out / "discussion.json").write_text(json.dumps(discussion, indent=2) + "\n",
                                         encoding="utf-8")
    return meta


# ── post ──────────────────────────────────────────────────────────────────────
def right_side_lines(diff_text: str) -> dict:
    """For each file of a diff, its hunks as lists of new-side line numbers."""
    files: dict = {}
    path, hunk, line, previous = None, None, 0, ""
    for text in diff_text.splitlines():
        # A file header is "--- a/x" then "+++ b/x"; an added line can also start "+++ ".
        is_header = text.startswith("+++ ") and previous.startswith("--- ")
        previous = text
        if is_header:
            path = text[6:] if text.startswith("+++ b/") else None
            hunk = None
            continue
        if text.startswith("diff --git "):
            hunk = None
            continue
        header = _HUNK.match(text)
        if header:
            line = int(header.group(1))
            hunk = []
            if path is not None:
                files.setdefault(path, []).append(hunk)
            continue
        if hunk is None or text.startswith("\\"):
            continue
        if text.startswith("-"):
            continue
        if text.startswith(("+", " ")) or text == "":
            hunk.append(line)
            line += 1
    return files


def anchored(finding: dict, hunks_by_file: dict) -> bool:
    """Whether the finding's lines are new-side lines of one hunk of its file."""
    start = finding.get("start_line") or finding["line"]
    return any(start in hunk and finding["line"] in hunk
               for hunk in hunks_by_file.get(finding["path"], []))


def credential_in(text: str) -> str | None:
    for name, pattern in CREDENTIAL_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def defang_links(text: str, repo: str) -> str:
    """Keep links into this repository; make every other URL plain text.

    A comment renders links and images, and a rendered image is a request to
    its host. Review comments only ever need to point into the repository.
    """
    own = f"https://github.com/{repo}/"

    def fix(match: re.Match) -> str:
        url = match.group(0)
        return url if url.startswith(own) else url.replace("://", "[:]//", 1)

    return _URL.sub(fix, text)


def checked_findings(document, repo: str) -> list[dict]:
    """The findings of the AI's output, or ReviewError when it is not fit to post."""
    if not isinstance(document, dict) or not isinstance(document.get("findings"), list):
        raise ReviewError("the review output has no list of findings")
    if len(document["findings"]) > MAX_FINDINGS:
        raise ReviewError(f"the review output has more than {MAX_FINDINGS} findings")
    findings = []
    for entry in document["findings"]:
        if not isinstance(entry, dict):
            raise ReviewError("a finding is not an object")
        path, line, body = entry.get("path"), entry.get("line"), entry.get("body")
        start = entry.get("start_line")
        if (not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/")
                or not isinstance(line, int) or isinstance(line, bool) or line < 1
                or not isinstance(body, str) or not body.strip()):
            raise ReviewError("a finding lacks a valid path, line or body")
        if start is not None and (not isinstance(start, int) or isinstance(start, bool)
                                  or start < 1 or start > line):
            raise ReviewError("a finding has an invalid start_line")
        if len(body) > MAX_BODY_CHARS:
            raise ReviewError(f"a finding's body is longer than {MAX_BODY_CHARS} characters")
        found = credential_in(body) or credential_in(path)
        if found:
            raise ReviewError(f"the review output holds text that looks like a credential "
                              f"({found}); nothing was posted")
        body = body.replace(MARKER, "").replace(SIGNATURE, "").strip()
        findings.append({"path": path, "line": line,
                         "start_line": start if start and start < line else None,
                         "body": defang_links(body, repo)})
    return findings


def agent_calls(execution_file: Path) -> int:
    """How many agents the lead session of a review launched, from its record."""
    events = json.loads(execution_file.read_text(encoding="utf-8"))
    if not isinstance(events, list):       # one JSON object per line
        raise ValueError("the execution file is not a list of events")
    count = 0
    for event in events:
        if (not isinstance(event, dict) or event.get("type") != "assistant"
                or event.get("parent_tool_use_id")):
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if (isinstance(block, dict) and block.get("type") == "tool_use"
                    and block.get("name") in ("Agent", "Task")):
                count += 1
    return count


class Poster:
    """The writes to the one pull request. ``dry_run`` prints them instead."""

    def __init__(self, repo: str, pr: int, *, dry_run: bool = False):
        self.repo, self.pr, self.dry_run = repo, pr, dry_run
        self.urls: list[str] = []

    def head_sha(self) -> str:
        return _gh_json(f"repos/{self.repo}/pulls/{self.pr}")["head"]["sha"]

    def paths_changed_since(self, reviewed: str, current: str) -> set | None:
        """Paths changed between the reviewed and the current head; None when the
        history was rewritten or the list is not complete, so every path counts."""
        try:
            compare = _gh_json(f"repos/{self.repo}/compare/{reviewed}...{current}")
        except (ReviewError, ValueError):
            return None
        files = compare.get("files") or []
        if compare.get("status") != "ahead" or len(files) >= 300:
            return None
        return ({item["filename"] for item in files}
                | {item["previous_filename"] for item in files if item.get("previous_filename")})

    def _write(self, endpoint: str, fields: list[str], what: str) -> str | None:
        if self.dry_run:
            print(f"[dry run] {what}: POST {endpoint} {json.dumps(fields)}")
            self.urls.append(f"dry-run:{what}")
            return self.urls[-1]
        try:
            url = _run(["gh", "api", "--method", "POST", endpoint, *fields, "--jq", ".html_url"],
                       timeout=120).strip()
        except ReviewError as exc:
            print(f"{what}: not accepted ({str(exc)[-200:]})")
            return None
        if url:
            self.urls.append(url)
        return url or None

    def issue_comment(self, body: str) -> str | None:
        return self._write(f"repos/{self.repo}/issues/{self.pr}/comments",
                           ["--raw-field", f"body={body}\n\n{FOOTER}"], "issue comment")

    def inline_comment(self, finding: dict, body: str, commit: str) -> str | None:
        fields = ["--raw-field", f"body={body}\n\n{FOOTER}", "--raw-field", f"commit_id={commit}",
                  "--raw-field", f"path={finding['path']}", "--field", f"line={finding['line']}",
                  "--raw-field", "side=RIGHT"]
        if finding["start_line"]:
            fields += ["--field", f"start_line={finding['start_line']}",
                       "--raw-field", "start_side=RIGHT"]
        return self._write(f"repos/{self.repo}/pulls/{self.pr}/comments", fields,
                           f"inline comment on {finding['path']}:{finding['line']}")


def _location(finding: dict) -> str:
    if finding["start_line"]:
        return f"{finding['path']}:{finding['start_line']}-{finding['line']}"
    return f"{finding['path']}:{finding['line']}"


def post_review(meta: dict, findings: list[dict], hunks_by_file: dict, poster: Poster) -> int:
    """Post the findings, or the no-findings note. Returns how many writes failed."""
    reviewed = meta["reviewed_head_sha"]
    short = reviewed[:11]
    current = poster.head_sha()
    moved = current != reviewed
    failed = 0
    if not findings:
        note = (f"\n\n_Reviewed at `{short}`; the PR has since moved to `{current[:11]}`. "
                "Dispatch the review again to cover the latest commit._" if moved else "")
        body = ("## Code review" + note + "\n\nNo issues found. Checked LLK correctness, hazards, "
                "team rules, and applicable architecture/test context.")
        failed += poster.issue_comment(body) is None
    else:
        changed = None
        if moved:
            changed = poster.paths_changed_since(reviewed, current)
            failed += poster.issue_comment(
                f"## Code review\n\nThis review covers `{short}`. The PR head moved to "
                f"`{current[:11]}` while the review was running. Inline comments are anchored "
                "to the reviewed commit, and GitHub marks them outdated where the lines have "
                "since changed. Dispatch the review again to cover the latest commit.") is None
        for finding in findings:
            body = finding["body"]
            if moved and (changed is None or finding["path"] in changed):
                body = (f"_Reviewed at `{short}`; the PR has since moved to `{current[:11]}`, "
                        f"so this may already be addressed._\n\n{body}")
            url = (poster.inline_comment(finding, body, reviewed)
                   if anchored(finding, hunks_by_file) else None)
            if url is None:
                # The location is kept in the text when it cannot be an inline anchor.
                where = f"`{_location(finding)}`" + (f" at `{short}`" if moved else "")
                failed += poster.issue_comment(f"{where}\n\n{body}") is None
    if not moved and not poster.dry_run:
        latest = poster.head_sha()
        if latest != reviewed:
            failed += poster.issue_comment(
                f"## Code review\n\nThis review covers `{short}`. The PR head moved to "
                f"`{latest[:11]}` while the review was being posted. Dispatch the review again "
                "to cover the latest commit.") is None
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs", help="write the files the AI step reads")
    inputs.add_argument("--repo", required=True)
    inputs.add_argument("--pr", required=True, type=int)
    inputs.add_argument("--base-sha", required=True)
    inputs.add_argument("--head-sha", required=True)
    inputs.add_argument("--base-tip-sha", required=True)
    inputs.add_argument("--out", required=True, type=Path)
    post = commands.add_parser("post", help="publish the AI step's findings on the pull request")
    post.add_argument("--input", required=True, type=Path, help="the directory `inputs` wrote")
    post.add_argument("--findings", type=Path,
                      help="file with the AI step's output (default: $LLK_REVIEW_FINDINGS)")
    post.add_argument("--execution-file", type=Path,
                      help="the AI step's record; refuse a session that skipped the procedure")
    post.add_argument("--closed", action="store_true",
                      help="the pull request is closed: post the skip note instead")
    post.add_argument("--dry-run", action="store_true", help="print the writes, make none")
    commands.add_parser("schema", help="print the JSON schema of the AI step's output")
    args = parser.parse_args(argv)

    try:
        if args.command == "schema":
            print(json.dumps(FINDINGS_SCHEMA, separators=(",", ":")))
            return 0
        if args.command == "inputs":
            for sha in (args.base_sha, args.head_sha, args.base_tip_sha):
                if not _SHA.match(sha):
                    raise ReviewError("the commits must be full SHAs")
            if not _REPO.match(args.repo) or args.pr < 1:
                raise ReviewError("invalid repository or pull request number")
            meta = write_inputs(args.repo, args.pr, args.base_sha, args.head_sha,
                                args.base_tip_sha, args.out)
            print(f"Wrote the review inputs: {len(meta['changed_files'])} changed file(s), "
                  f"state {meta['state']}")
            if os.environ.get("GITHUB_OUTPUT"):
                with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
                    output.write(f"state={meta['state']}\n")
            return 0

        meta = json.loads((args.input / "pr.json").read_text(encoding="utf-8"))
        poster = Poster(meta["repository"], int(meta["number"]), dry_run=args.dry_run)
        if args.closed:
            failed = poster.issue_comment(
                "## Code review\n\nSkipped: the pull request is closed.") is None
        else:
            if args.execution_file:
                launched = agent_calls(args.execution_file)
                print(f"The review session launched {launched} agent(s)")
                if launched < MIN_AGENT_CALLS:
                    raise ReviewError(
                        f"the review session launched {launched} agent(s); the procedure "
                        f"needs at least {MIN_AGENT_CALLS}. Nothing was posted")
            text = (args.findings.read_text(encoding="utf-8") if args.findings
                    else os.environ.get("LLK_REVIEW_FINDINGS", ""))
            if not text.strip():
                raise ReviewError("the AI step returned no output")
            try:
                document = json.loads(text)
            except ValueError as exc:
                raise ReviewError(f"the AI step's output is not JSON: {exc}") from exc
            findings = checked_findings(document, meta["repository"])
            hunks = right_side_lines((args.input / "diff" / "full.diff").read_text(encoding="utf-8"))
            failed = post_review(meta, findings, hunks, poster)
            print(f"{len(findings)} finding(s)")
        print(f"LLK review posted {len(poster.urls)} comment(s)")
        if failed or not poster.urls:
            raise ReviewError(f"{failed} comment(s) could not be posted")
        return 0
    except (ReviewError, OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        print(f"::error::{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
