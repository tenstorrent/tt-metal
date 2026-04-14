#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Review changes from a BH issue solver run using a separate Claude instance.

Spawns a reviewer Claude that reads the diff and produces structured comments.
If there are actionable comments, spawns a fixer Claude to address them.

Usage:
    # Review only
    python scripts/review_changes.py --repo-root /path/to/tt-llk --issue 1153 --title "Fix unpack"

    # Review and auto-fix
    python scripts/review_changes.py --repo-root /path/to/tt-llk --issue 1153 --title "Fix unpack" --auto-fix

    # Output review results to file
    python scripts/review_changes.py --repo-root /path/to/tt-llk --issue 1153 --title "Fix unpack" --output /tmp/review.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_diff(repo_root: Path) -> str:
    """Get full diff against origin/main."""
    proc = subprocess.run(
        ["git", "diff", "origin/main...HEAD"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=30,
    )
    return proc.stdout


def get_diff_stat(repo_root: Path) -> str:
    """Get diff --stat summary."""
    proc = subprocess.run(
        ["git", "diff", "--stat", "origin/main...HEAD"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=10,
    )
    return proc.stdout.strip()


def _extract_review_json(cli_stdout: str) -> dict | None:
    """Extract structured review JSON from Claude CLI output.

    The CLI ``--output-format json`` may produce:
    - A JSON array of conversation events
    - A single JSON result object (with ``result``, ``session_id``, etc.)
    - Rarely, a raw JSON dict with ``verdict`` directly

    The review JSON (``{"verdict": …}``) can appear in any of:
    ``result``, ``content`` (string or blocks), or ``message.content``.
    As a last resort we regex-search the raw stdout.
    """

    try:
        data = json.loads(cli_stdout)
    except json.JSONDecodeError:
        # Unparsable JSON — fall through to regex search
        return _find_verdict_json(cli_stdout)

    # Direct dict with "verdict" — simplest case
    if isinstance(data, dict) and "verdict" in data:
        return data

    # Normalise: treat a single dict the same as a one-element list
    entries = data if isinstance(data, list) else [data]

    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue

        # Check message.content (standard format)
        content = entry.get("content") or entry.get("message", {}).get("content", [])
        if isinstance(content, str):
            found = _try_parse_verdict(content)
            if found:
                return found
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    found = _try_parse_verdict(block.get("text", ""))
                    if found:
                        return found

        # Check "result" field (prompt-mode CLI output)
        result_text = entry.get("result", "")
        if result_text:
            found = _try_parse_verdict(result_text)
            if found:
                return found

    # Last resort: regex search through the raw stdout
    return _find_verdict_json(cli_stdout)


def _try_parse_verdict(text: str) -> dict | None:
    """Try to parse a verdict JSON from text, handling fences and escaping."""
    for candidate in [text, text.strip("` \n"), _strip_json_fences(text)]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    # The text may itself be JSON-escaped (double-encoded)
    try:
        unescaped = json.loads(f'"{text}"') if text.startswith("{") else None
        if isinstance(unescaped, str):
            parsed = json.loads(unescaped)
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _find_verdict_json(text: str) -> dict | None:
    """Last-resort: find a JSON object containing ``"verdict"`` via brace matching."""
    import re

    # Try raw text and one level of unescaping
    variants = [text]
    unescaped = text.replace('\\"', '"').replace("\\\\", "\\")
    if unescaped != text:
        variants.append(unescaped)

    for t in variants:
        for m in re.finditer(r'"verdict"', t):
            # Walk backwards to find the opening brace
            start = None
            for i in range(m.start() - 1, -1, -1):
                if t[i] == "{":
                    start = i
                    break
            if start is None:
                continue
            # Walk forward to find matching closing brace
            depth = 0
            for i in range(start, len(t)):
                if t[i] == "{":
                    depth += 1
                elif t[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(t[start : i + 1])
                            if isinstance(parsed, dict) and "verdict" in parsed:
                                return parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
    return None


def _strip_json_fences(text: str) -> str:
    """Strip markdown ```json ... ``` fences."""
    import re

    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else text


def run_reviewer(
    repo_root: Path,
    issue_num: int,
    issue_title: str,
    model: str = "claude-opus-4-6",
    codegen_dir: Path | None = None,
    timeout: int = 600,
) -> dict:
    """Spawn a reviewer Claude instance."""
    diff = get_diff(repo_root)
    if not diff.strip():
        return {"status": "skipped", "reason": "no changes to review", "comments": []}

    stat = get_diff_stat(repo_root)

    # Truncate very large diffs
    if len(diff) > 100_000:
        diff = diff[:100_000] + "\n\n... (diff truncated at 100KB)"

    prompt = f"""You are reviewing code changes for GitHub issue #{issue_num}: "{issue_title}".
This is a Blackhole LLK kernel repository (Tenstorrent hardware).

Focus on REAL PROBLEMS that would cause a human reviewer to reject this PR:
1. **Correctness**: Does the change fix the described issue?
2. **Compilation**: Wrong includes, namespaces, undefined symbols, type mismatches.
3. **Logic errors**: Wrong register, wrong template parameter, off-by-one, missing init/uninit symmetry.
4. **Missing changes**: Files that should have been updated but weren't.
5. **Regressions**: Changes that could break existing tests.

Do NOT flag style, formatting, missing comments, or minor warnings.

Diff stat:
{stat}

Full diff:
```diff
{diff}
```

Output ONLY a JSON object (no markdown fences, no extra text):
{{"verdict": "approve" or "request_changes", "summary": "one-line finding", "comments": [{{"file": "path", "line": 42, "severity": "error" or "warning", "message": "problem and fix"}}]}}"""

    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--tools",
        "",
    ]

    # Run from /tmp to avoid loading project .claude/ config (superpowers,
    # hooks, etc.) which details the reviewer into full agent mode instead
    # of producing a clean JSON verdict.
    cwd = "/tmp"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )

        if proc.returncode != 0:
            return {
                "status": "error",
                "reason": f"reviewer exited {proc.returncode}",
                "stderr": proc.stderr[-500:],
                "comments": [],
            }

        review_data = _extract_review_json(proc.stdout)
        if review_data:
            return {
                "status": "completed",
                "verdict": review_data.get("verdict", "unknown"),
                "summary": review_data.get("summary", ""),
                "comments": review_data.get("comments", []),
                "model": model,
            }

        # Save full stdout for debugging parse failures
        debug_path = Path("/tmp") / f"review_debug_{issue_num}.json"
        try:
            debug_path.write_text(proc.stdout)
        except OSError:
            pass

        return {
            "status": "parse_error",
            "reason": "could not extract structured review from output",
            "raw_output": proc.stdout[-1000:],
            "debug_file": str(debug_path),
            "comments": [],
        }

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "comments": []}
    except Exception as e:
        return {"status": "error", "reason": str(e), "comments": []}


def run_fixer(
    repo_root: Path,
    issue_num: int,
    comments: list[dict],
    model: str = "claude-opus-4-6",
    codegen_dir: Path | None = None,
    timeout: int = 1800,
) -> dict:
    """Spawn a fixer Claude instance to address review comments."""
    if not comments:
        return {"status": "skipped", "reason": "no comments to fix"}

    error_comments = [c for c in comments if c.get("severity") == "error"]
    if not error_comments:
        return {"status": "skipped", "reason": "only warnings, no errors to fix"}

    comments_text = json.dumps(error_comments, indent=2)

    prompt = f"""A code reviewer found issues with changes for GitHub issue #{issue_num}.

Review comments to address (errors only):
{comments_text}

For each comment:
1. Read the file mentioned
2. Understand the issue
3. Fix it
4. Verify the fix compiles

After fixing all comments, commit with message:
"fix: address review comments for issue #{issue_num}"

Work autonomously, do not ask questions."""

    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
    ]

    cwd = str(codegen_dir or repo_root)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "status": "completed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "comments_addressed": len(error_comments),
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Review changes from BH issue solver run"
    )
    parser.add_argument(
        "--repo-root", type=Path, required=True, help="tt-llk repo root"
    )
    parser.add_argument("--issue", type=int, required=True, help="GitHub issue number")
    parser.add_argument("--title", required=True, help="Issue title")
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Reviewer model (default: opus)",
    )
    parser.add_argument(
        "--fixer-model",
        default="claude-opus-4-6",
        help="Fixer model (default: opus for quality)",
    )
    parser.add_argument(
        "--auto-fix", action="store_true", help="Auto-fix error-severity comments"
    )
    parser.add_argument("--output", "-o", type=Path, help="Write review JSON to file")
    parser.add_argument(
        "--codegen-dir",
        type=Path,
        default=None,
        help="codegen-bh directory (default: repo-root/codegen-bh)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Reviewer timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    codegen_dir = args.codegen_dir or (args.repo_root / "codegen-bh")

    print(f"Reviewing changes for issue #{args.issue}: {args.title}")
    print(f"  Reviewer model: {args.model}")

    result = run_reviewer(
        repo_root=args.repo_root,
        issue_num=args.issue,
        issue_title=args.title,
        model=args.model,
        codegen_dir=codegen_dir,
        timeout=args.timeout,
    )

    print(f"  Review status: {result.get('status')}")
    print(f"  Verdict: {result.get('verdict', 'N/A')}")
    print(f"  Comments: {len(result.get('comments', []))}")

    if result.get("summary"):
        print(f"  Summary: {result['summary']}")

    for c in result.get("comments", []):
        sev = c.get("severity", "?").upper()
        print(
            f"    [{sev}] {c.get('file', '?')}:{c.get('line', '?')} — {c.get('message', '')}"
        )

    # Auto-fix if requested and there are error-severity comments
    fix_result = None
    if args.auto_fix and result.get("verdict") == "request_changes":
        error_comments = [
            c for c in result.get("comments", []) if c.get("severity") == "error"
        ]
        if error_comments:
            print(f"\n  Auto-fixing {len(error_comments)} error(s)...")
            fix_result = run_fixer(
                repo_root=args.repo_root,
                issue_num=args.issue,
                comments=error_comments,
                model=args.fixer_model,
                codegen_dir=codegen_dir,
            )
            result["fix_result"] = fix_result
            print(f"  Fix status: {fix_result.get('status')}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    return 0 if result.get("verdict") == "approve" else 1


if __name__ == "__main__":
    sys.exit(main())
