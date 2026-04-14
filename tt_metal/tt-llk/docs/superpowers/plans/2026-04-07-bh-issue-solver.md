# BH Issue Solver Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a dedicated `nstamatovic/bh_issue_solver` branch with script-driven logging (not agent-driven), post-run evaluation, automated code review, and dashboard integration.

**Architecture:** The batch runner (`batch_generate_bh.sh`) orchestrates the full pipeline: fetch issues -> solve with Claude -> evaluate results -> review changes -> log everything. Logging is owned by the runner scripts, not by agents. A standalone `review_changes.py` spawns a second Claude instance as a reviewer, then a third as a fixer if needed. The dashboard at `llk_code_gen/` is updated to display evaluation and review data without touching quasar.

**Tech Stack:** Bash, Python 3.10+, Claude CLI (`claude -p`), Flask dashboard (existing at `/proj_sw/user_dev/llk_code_gen/dashboard/`)

**Note on monitoring:** Claude Code offers OpenTelemetry-based monitoring, but it requires external OTLP collector infrastructure. For our use case, parsing CLI `--output-format json` output is simpler, sufficient, and already partially implemented. We fix the existing parsing bugs rather than adding OTel infrastructure.

**Pending decision:** User will specify which local agent(s) from `codegen-bh/agents/` to keep. Marked as `[TBD]` in Task 1.

---

### Task 1: Branch Setup & `.claude` Configuration

**Files:**
- Copy: `.claude/` from `nstamatovic/claude-code-team-setup` branch
- Remove: `codegen-bh/agents/*.md` (except `[TBD]` — user will specify)
- Remove: `codegen-bh/scripts/ci_weekly_bh.py`
- Remove: `codegen-bh/scripts/install-agents.sh`

- [ ] **Step 1: Create the branch**

```bash
cd /proj_sw/user_dev/nstamatovic/tt-llk
git checkout -b nstamatovic/bh_issue_solver nstamatovic/bh_code_gen_v1
```

- [ ] **Step 2: Copy `.claude/` from team-setup branch**

```bash
# Remove current .claude if it exists on this branch
rm -rf .claude

# Checkout .claude from the team-setup branch
git checkout nstamatovic/claude-code-team-setup -- .claude/
```

Expected: `.claude/` now contains `CLAUDE.md`, `agents/` (sage-blackhole, sage-wormhole, sage-quasar, llk-debugger, llk-test-runner), `skills/` (arch-lookup, debug-kernel, port-kernel, run-test), `references/`, `scripts/`.

- [ ] **Step 3: Remove codegen-bh agents (except TBD exception)**

```bash
# Remove all codegen-bh agents — these are kernel-gen specific
rm -f codegen-bh/agents/llk-analyzer.md
rm -f codegen-bh/agents/llk-planner.md
rm -f codegen-bh/agents/llk-kernel-writer.md
rm -f codegen-bh/agents/llk-tester.md
rm -f codegen-bh/agents/llk-debugger.md
rm -f codegen-bh/agents/llk-arch-lookup.md
# [TBD] User may want to keep one — leave it if specified
```

- [ ] **Step 4: Remove cron job and install script**

```bash
rm -f codegen-bh/scripts/ci_weekly_bh.py
rm -f codegen-bh/scripts/install-agents.sh
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: set up bh_issue_solver branch with team .claude config

- Copy .claude/ from nstamatovic/claude-code-team-setup (sages, debugger, test-runner, skills)
- Remove codegen-bh kernel-gen agents (not needed for issue solving)
- Remove ci_weekly_bh.py cron job (will be separate branch)
- Remove install-agents.sh (using .claude/ directly)"
```

---

### Task 2: Extract and Fix Logging — `log_run.py`

**Files:**
- Create: `codegen-bh/scripts/log_run.py`
- Modify: `codegen-bh/scripts/batch_generate_bh.sh` (remove embedded Python `log_run()`)

**Why:** The current `log_run()` is ~160 lines of Python embedded in bash. It has a critical bug (missing `import glob` at line 219) and multiple silent exception handlers that swallow data loss. Extracting it makes it testable, debuggable, and fixable.

- [ ] **Step 1: Create `codegen-bh/scripts/log_run.py`**

```python
#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Log a Blackhole issue solver run to the shared runs directory.

Creates a versioned run directory, parses CLI JSON for tokens/cost,
captures git state, and writes run.json + appends to runs.jsonl.

Usage:
    python scripts/log_run.py \
        --issue 1153 --title "Fix unpack_reduce" \
        --branch nstamatovic/issue-1153-codegen-v1 \
        --status completed --start 2026-04-07T10:00:00Z --end 2026-04-07T11:00:00Z \
        --log-dir /tmp/codegen_bh_logs_20260407/ \
        --model claude-opus-4-6 --repo-root /proj_sw/user_dev/nstamatovic/tt-llk \
        [--batch-id 2026-04-07_bh_batch] \
        [--issues-json artifacts/bh_p2_issues.json] \
        [--evaluation /tmp/eval.json] \
        [--review /tmp/review.json]
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODEGEN_DIR = SCRIPT_DIR.parent
REPO_ROOT_DEFAULT = CODEGEN_DIR.parent
RUNS_BASE_DEFAULT = (REPO_ROOT_DEFAULT / "../../llk_code_gen/blackhole_issue_solver").resolve()
EXTRACT_SCRIPT = SCRIPT_DIR / "extract_conversation.py"


def parse_cli_json(cli_json_path: Path) -> dict:
    """Parse Claude CLI JSON output for tokens, cost, turns, duration.

    The CLI outputs a JSON array of conversation events. The last entry
    carries aggregated modelUsage and total_cost_usd.
    """
    empty = {"cost_usd": 0, "tokens": {}, "num_turns": 0, "duration_seconds": 0}
    if not cli_json_path.exists():
        return empty

    try:
        data = json.loads(cli_json_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: could not read CLI JSON {cli_json_path}: {e}", file=sys.stderr)
        return empty

    if not isinstance(data, list) or len(data) == 0:
        print(f"  Warning: CLI JSON is empty or not a list", file=sys.stderr)
        return empty

    last = data[-1]
    if not isinstance(last, dict):
        return empty

    result = {
        "num_turns": last.get("num_turns", 0),
        "cost_usd": round(last.get("total_cost_usd", 0), 4),
        "duration_seconds": int(last.get("duration_ms", 0) / 1000),
    }

    model_usage = last.get("modelUsage", {})
    if model_usage:
        tokens = {}
        for model, usage in model_usage.items():
            tokens[model] = {
                "input": usage.get("inputTokens", 0),
                "output": usage.get("outputTokens", 0),
                "cache_read": usage.get("cacheReadInputTokens", 0),
                "cache_creation": usage.get("cacheCreationInputTokens", 0),
                "cost_usd": round(usage.get("costUSD", 0), 4),
            }
        # Compute totals across all models
        model_entries = [v for v in tokens.values() if isinstance(v, dict)]
        tokens["total"] = {
            "input": sum(t.get("input", 0) for t in model_entries),
            "output": sum(t.get("output", 0) for t in model_entries),
            "cache_read": sum(t.get("cache_read", 0) for t in model_entries),
            "cache_creation": sum(t.get("cache_creation", 0) for t in model_entries),
            "cost_usd": result["cost_usd"],
        }
        result["tokens"] = tokens
    else:
        result["tokens"] = {}

    return result


def get_changed_files(repo_root: Path) -> list[str]:
    """Get files changed between origin/main and HEAD."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        return [f.strip() for f in proc.stdout.strip().splitlines() if f.strip()]
    except Exception as e:
        print(f"  Warning: git diff failed: {e}", file=sys.stderr)
        return []


def get_git_commit(repo_root: Path) -> str:
    """Get short HEAD commit hash."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=5,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def create_run_dir(issue_num: int, runs_base: Path) -> tuple[str, Path]:
    """Create versioned run directory: blackhole_issue_{num}_v{N}."""
    existing = sorted(glob(str(runs_base / f"blackhole_issue_{issue_num}_v*")))
    version = len(existing) + 1
    run_id = f"blackhole_issue_{issue_num}_v{version}"
    run_dir = runs_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def log_run(
    issue_num: int,
    title: str,
    branch: str,
    status: str,
    start_time: str,
    end_time: str,
    tmp_log_dir: Path,
    model: str,
    repo_root: Path,
    runs_base: Path,
    batch_id: str | None = None,
    issues_json: Path | None = None,
    evaluation: dict | None = None,
    review: dict | None = None,
) -> dict:
    """Create run directory, parse metrics, write run.json + runs.jsonl.

    Returns the run entry dict.
    """
    runs_jsonl = runs_base / "runs.jsonl"
    run_id, run_dir = create_run_dir(issue_num, runs_base)

    # Copy logs from temp dir into run dir
    for pattern in [f"issue_{issue_num}.json", f"issue_{issue_num}.log"]:
        src = tmp_log_dir / pattern
        if src.is_file():
            shutil.copy2(src, run_dir)

    # Parse CLI JSON for tokens/cost/turns
    cli_json = run_dir / f"issue_{issue_num}.json"
    metrics = parse_cli_json(cli_json)

    # Validate token capture
    if not metrics["tokens"]:
        print("  Warning: NO TOKEN DATA CAPTURED — CLI JSON may be missing or malformed", file=sys.stderr)

    # Get git info
    changed_files = get_changed_files(repo_root)
    git_commit = get_git_commit(repo_root)

    # Copy changed files as snapshots (path separators flattened to underscores)
    for fpath in changed_files:
        full = repo_root / fpath
        if full.is_file():
            flat_name = fpath.replace("/", "_")
            try:
                shutil.copy2(full, run_dir / flat_name)
            except OSError:
                pass

    # Fetch issue metadata from cached issues JSON
    issue_meta = {
        "number": issue_num,
        "title": title,
        "url": f"https://github.com/tenstorrent/tt-llk/issues/{issue_num}",
        "labels": [],
    }
    if issues_json and issues_json.exists():
        try:
            data = json.loads(issues_json.read_text())
            for iss in data.get("issues", []):
                if iss["number"] == issue_num:
                    issue_meta["labels"] = iss.get("labels", [])
                    issue_meta["url"] = iss.get("url", issue_meta["url"])
                    break
        except (json.JSONDecodeError, KeyError):
            pass

    # Build the run entry
    entry = {
        "run_id": run_id,
        "arch": "blackhole",
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": metrics["duration_seconds"],
        "num_turns": metrics["num_turns"],
        "status": status,
        "model": model,
        "run_type": "ci" if batch_id else "manual",
        "cost_usd": metrics["cost_usd"],
        "tokens": metrics["tokens"],
        "issue": issue_meta,
        "changed_files": changed_files,
        "git_branch": branch,
        "git_commit": git_commit,
        "batch_id": batch_id or None,
        "log_dir": run_id,
    }

    if evaluation:
        entry["evaluation"] = evaluation

    if review:
        entry["review"] = review

    # Write run.json into run directory
    (run_dir / "run.json").write_text(json.dumps(entry, indent=2) + "\n")

    # Append to runs.jsonl (file-locked for parallel safety)
    runs_base.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with open(runs_jsonl, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)

    # Extract readable conversation from CLI JSON
    if EXTRACT_SCRIPT.exists() and cli_json.exists():
        try:
            proc = subprocess.run(
                [sys.executable, str(EXTRACT_SCRIPT), str(run_dir)],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0:
                print(proc.stdout.strip())
            else:
                print(f"  Warning: extract_conversation failed: {proc.stderr[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: could not extract conversation: {e}", file=sys.stderr)

    # Print summary
    cost = metrics["cost_usd"]
    turns = metrics["num_turns"]
    n_files = len(changed_files)
    print(f"  Logged to {run_dir}/")
    print(f"  cost=${cost}  turns={turns}  changed={n_files} files")
    if evaluation:
        print(f"  evaluation: {evaluation.get('overall', '?')}")
    if review:
        print(f"  review: {review.get('verdict', '?')} ({len(review.get('comments', []))} comments)")

    return entry


def main():
    parser = argparse.ArgumentParser(description="Log a BH issue solver run")
    parser.add_argument("--issue", type=int, required=True, help="GitHub issue number")
    parser.add_argument("--title", required=True, help="Issue title")
    parser.add_argument("--branch", required=True, help="Git branch name")
    parser.add_argument("--status", required=True, choices=["completed", "crashed"],
                        help="Run exit status")
    parser.add_argument("--start", required=True, help="Start time (ISO8601)")
    parser.add_argument("--end", required=True, help="End time (ISO8601)")
    parser.add_argument("--log-dir", type=Path, required=True, help="Temp log directory")
    parser.add_argument("--model", required=True, help="Model used")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--runs-base", type=Path, default=RUNS_BASE_DEFAULT)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--issues-json", type=Path, default=None)
    parser.add_argument("--evaluation", type=Path, default=None,
                        help="Path to evaluation JSON file")
    parser.add_argument("--review", type=Path, default=None,
                        help="Path to review JSON file")
    args = parser.parse_args()

    eval_data = None
    if args.evaluation and args.evaluation.exists():
        try:
            eval_data = json.loads(args.evaluation.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    review_data = None
    if args.review and args.review.exists():
        try:
            review_data = json.loads(args.review.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    entry = log_run(
        issue_num=args.issue,
        title=args.title,
        branch=args.branch,
        status=args.status,
        start_time=args.start,
        end_time=args.end,
        tmp_log_dir=args.log_dir,
        model=args.model,
        repo_root=args.repo_root,
        runs_base=args.runs_base,
        batch_id=args.batch_id,
        issues_json=args.issues_json,
        evaluation=eval_data,
        review=review_data,
    )

    # Print the entry for the caller
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify log_run.py syntax**

Run: `python3 -c "import py_compile; py_compile.compile('codegen-bh/scripts/log_run.py', doraise=True)"`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add codegen-bh/scripts/log_run.py
git commit -m "feat: extract log_run.py from batch script

Standalone Python script for run logging. Fixes:
- Missing 'import glob' bug from embedded version
- Silent exception swallowing (now warns to stderr)
- Token capture validation
- Accepts evaluation and review data from new pipeline steps"
```

---

### Task 3: Post-Run Evaluation — `evaluate_run.py`

**Files:**
- Create: `codegen-bh/scripts/evaluate_run.py`

**Why:** The current runner only knows exit code (0 = completed, non-zero = crashed). It cannot distinguish "Claude ran but changed nothing" from "Claude fixed the issue and tests pass." This script runs post-hoc checks: compilation, tests, diff analysis.

- [ ] **Step 1: Create `codegen-bh/scripts/evaluate_run.py`**

```python
#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate the results of a Blackhole issue solver run.

Checks:
1. Were any files actually changed?
2. Do changed header files compile?
3. Do relevant functional tests pass?
4. How meaningful is the diff?

Usage:
    python scripts/evaluate_run.py --repo-root /path/to/tt-llk
    python scripts/evaluate_run.py --repo-root /path/to/tt-llk --output /tmp/eval.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def get_changed_files(repo_root: Path) -> list[str]:
    """Get files changed between origin/main and HEAD."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        return [f.strip() for f in proc.stdout.strip().splitlines() if f.strip()]
    except Exception:
        return []


def check_compilation(changed_files: list[str], repo_root: Path) -> dict:
    """Run check_compile.py on changed Blackhole header files."""
    headers = [
        f for f in changed_files
        if f.endswith(".h") and "tt_llk_blackhole" in f
    ]
    if not headers:
        return {"status": "skipped", "reason": "no BH header files changed", "files": []}

    results = []
    codegen_dir = repo_root / "codegen-bh"
    env = {**os.environ, "PYTHONPATH": str(repo_root)}

    for header in headers:
        full_path = repo_root / header
        if not full_path.exists():
            results.append({"file": header, "status": "missing"})
            continue

        try:
            proc = subprocess.run(
                [sys.executable, "scripts/check_compile.py", str(full_path), "-v"],
                capture_output=True, text=True, timeout=120,
                cwd=codegen_dir, env=env,
            )
            results.append({
                "file": header,
                "status": "passed" if proc.returncode == 0 else "failed",
                "output": proc.stderr[-500:] if proc.returncode != 0 else "",
            })
        except subprocess.TimeoutExpired:
            results.append({"file": header, "status": "timeout"})
        except Exception as e:
            results.append({"file": header, "status": "error", "message": str(e)})

    passed = all(r["status"] in ("passed", "skipped") for r in results)
    return {"status": "passed" if passed else "failed", "files": results}


def check_tests(changed_files: list[str], repo_root: Path) -> dict:
    """Run quick functional tests for kernels whose files changed."""
    # Infer kernel names from changed paths
    kernels = set()
    for f in changed_files:
        stem = Path(f).stem
        if "ckernel_sfpu_" in stem:
            kernels.add(stem.replace("ckernel_sfpu_", ""))
        elif stem.startswith("llk_"):
            kernels.add(stem.replace("llk_", ""))

    if not kernels:
        return {"status": "skipped", "reason": "no kernel files changed", "kernels": []}

    results = []
    codegen_dir = repo_root / "codegen-bh"
    env = {**os.environ, "PYTHONPATH": str(repo_root)}

    for kernel in kernels:
        try:
            proc = subprocess.run(
                [sys.executable, "scripts/run_functional_test.py", kernel,
                 "--arch", "blackhole", "--quick"],
                capture_output=True, text=True, timeout=600,
                cwd=codegen_dir, env=env,
            )
            results.append({
                "kernel": kernel,
                "status": "passed" if proc.returncode == 0 else "failed",
                "output": (proc.stdout + proc.stderr)[-500:],
            })
        except subprocess.TimeoutExpired:
            results.append({"kernel": kernel, "status": "timeout"})
        except Exception as e:
            results.append({"kernel": kernel, "status": "error", "message": str(e)})

    passed = all(r["status"] in ("passed", "skipped") for r in results)
    return {"status": "passed" if passed else "failed", "kernels": results}


def analyze_diff(changed_files: list[str], repo_root: Path) -> dict:
    """Analyze the diff for meaningfulness."""
    if not changed_files:
        return {"status": "no_changes", "score": 0, "summary": "No files changed"}

    try:
        proc = subprocess.run(
            ["git", "diff", "--stat", "origin/main...HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=10,
        )
        stat = proc.stdout.strip()
    except Exception:
        stat = ""

    has_kernel = any("tt_llk_" in f for f in changed_files)
    has_test = any("test" in f.lower() for f in changed_files)

    score = 0
    if has_kernel:
        score += 50
    if has_test:
        score += 30
    if has_kernel and has_test:
        score += 20

    parts = [f"{len(changed_files)} files changed"]
    if has_kernel:
        parts.append("includes kernel modifications")
    if has_test:
        parts.append("includes test changes")
    if not has_kernel and not has_test:
        parts.append("no kernel or test files")
        score = max(score, 10)

    return {
        "status": "evaluated",
        "score": min(score, 100),
        "file_count": len(changed_files),
        "has_kernel_changes": has_kernel,
        "has_test_changes": has_test,
        "summary": ", ".join(parts),
        "stat": stat,
    }


def evaluate(repo_root: Path) -> dict:
    """Run the full evaluation suite."""
    changed_files = get_changed_files(repo_root)

    compilation = check_compilation(changed_files, repo_root)
    tests = check_tests(changed_files, repo_root)
    diff_analysis = analyze_diff(changed_files, repo_root)

    if not changed_files:
        overall = "no_changes"
    elif compilation["status"] == "failed":
        overall = "compile_failed"
    elif tests["status"] == "failed":
        overall = "tests_failed"
    elif compilation["status"] == "passed" and tests["status"] == "passed":
        overall = "success"
    elif compilation["status"] == "passed":
        overall = "compiled"
    else:
        overall = "partial"

    return {
        "overall": overall,
        "compilation": compilation,
        "tests": tests,
        "diff_analysis": diff_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BH issue solver run results")
    parser.add_argument("--repo-root", type=Path, required=True, help="tt-llk repo root")
    parser.add_argument("--output", "-o", type=Path, help="Write JSON result to file")
    args = parser.parse_args()

    result = evaluate(args.repo_root)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    return 0 if result["overall"] in ("success", "compiled") else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import py_compile; py_compile.compile('codegen-bh/scripts/evaluate_run.py', doraise=True)"`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add codegen-bh/scripts/evaluate_run.py
git commit -m "feat: add post-run evaluation script

Checks compilation, runs quick functional tests, analyzes diff
meaningfulness. Produces structured JSON with overall status."
```

---

### Task 4: Changes Reviewer — `review_changes.py`

**Files:**
- Create: `codegen-bh/scripts/review_changes.py`

**Why:** Human reviewers waste time on basic AI mistakes (wrong register, missing include, broken symmetry). A second Claude instance catches these before the PR, and a third fixes them — so the human reviewer sees cleaner code.

- [ ] **Step 1: Create `codegen-bh/scripts/review_changes.py`**

```python
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
        capture_output=True, text=True, cwd=repo_root, timeout=30,
    )
    return proc.stdout


def get_diff_stat(repo_root: Path) -> str:
    """Get diff --stat summary."""
    proc = subprocess.run(
        ["git", "diff", "--stat", "origin/main...HEAD"],
        capture_output=True, text=True, cwd=repo_root, timeout=10,
    )
    return proc.stdout.strip()


def _extract_review_json(cli_stdout: str) -> dict | None:
    """Extract structured review JSON from Claude CLI output.

    CLI --output-format json produces a JSON array. The assistant's response
    text is inside content blocks. We search for the JSON review object.
    """
    try:
        data = json.loads(cli_stdout)
    except json.JSONDecodeError:
        return None

    # Direct dict with "verdict" — simplest case
    if isinstance(data, dict) and "verdict" in data:
        return data

    # CLI JSON array — search through entries for assistant text
    if isinstance(data, list):
        for entry in reversed(data):
            if not isinstance(entry, dict):
                continue
            # Check message.content (standard format)
            content = entry.get("content") or entry.get("message", {}).get("content", [])
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "verdict" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        # Try to find JSON in the text (may have markdown fences)
                        for candidate in [text, text.strip("` \n"), _strip_json_fences(text)]:
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict) and "verdict" in parsed:
                                    return parsed
                            except json.JSONDecodeError:
                                continue
            # Check "result" field (some CLI versions)
            result_text = entry.get("result", "")
            if result_text:
                try:
                    parsed = json.loads(result_text)
                    if isinstance(parsed, dict) and "verdict" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass

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
    model: str = "claude-sonnet-4-6",
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
        "claude", "-p", prompt,
        "--model", model,
        "--dangerously-skip-permissions",
        "--output-format", "json",
    ]

    cwd = str(codegen_dir or repo_root)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd,
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

        return {
            "status": "parse_error",
            "reason": "could not extract structured review from output",
            "raw_output": proc.stdout[-1000:],
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
4. Verify the fix compiles (run: PYTHONPATH=.. python scripts/check_compile.py <path> -v)

After fixing all comments, commit with message:
"fix: address review comments for issue #{issue_num}"

Work autonomously, do not ask questions."""

    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--dangerously-skip-permissions",
        "--output-format", "json",
    ]

    cwd = str(codegen_dir or repo_root)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd,
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
    parser = argparse.ArgumentParser(description="Review changes from BH issue solver run")
    parser.add_argument("--repo-root", type=Path, required=True, help="tt-llk repo root")
    parser.add_argument("--issue", type=int, required=True, help="GitHub issue number")
    parser.add_argument("--title", required=True, help="Issue title")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Reviewer model (default: sonnet for speed/cost)")
    parser.add_argument("--fixer-model", default="claude-opus-4-6",
                        help="Fixer model (default: opus for quality)")
    parser.add_argument("--auto-fix", action="store_true",
                        help="Auto-fix error-severity comments")
    parser.add_argument("--output", "-o", type=Path, help="Write review JSON to file")
    parser.add_argument("--codegen-dir", type=Path, default=None,
                        help="codegen-bh directory (default: repo-root/codegen-bh)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Reviewer timeout in seconds (default: 600)")
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
        print(f"    [{sev}] {c.get('file', '?')}:{c.get('line', '?')} — {c.get('message', '')}")

    # Auto-fix if requested and there are error-severity comments
    fix_result = None
    if args.auto_fix and result.get("verdict") == "request_changes":
        error_comments = [c for c in result.get("comments", []) if c.get("severity") == "error"]
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
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import py_compile; py_compile.compile('codegen-bh/scripts/review_changes.py', doraise=True)"`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add codegen-bh/scripts/review_changes.py
git commit -m "feat: add automated changes reviewer

Spawns a reviewer Claude (sonnet for speed) to inspect the diff.
If errors found, optionally spawns a fixer Claude (opus for quality).
Produces structured JSON with verdict, comments, fix results."
```

---

### Task 5: Update Batch Runner — `batch_generate_bh.sh`

**Files:**
- Modify: `codegen-bh/scripts/batch_generate_bh.sh`

**Changes:**
1. Remove embedded `log_run()` Python function (~160 lines)
2. Add calls to `evaluate_run.py`, `review_changes.py`, `log_run.py`
3. Remove cron-related references
4. Add `--no-review` flag to skip review step
5. Add `--auto-fix` flag to enable auto-fix on review comments

- [ ] **Step 1: Replace `log_run()` function with call to `log_run.py`**

Remove lines 197-357 (the entire `log_run()` function) and replace with:

```bash
# --- Log run result ---
log_run() {
  local issue_num="$1" title="$2" branch="$3" status="$4" start_time="$5" end_time="$6" \
        tmp_log_dir="$7" eval_json="${8:-}" review_json="${9:-}"

  local log_args=(
    --issue "$issue_num" --title "$title" --branch "$branch"
    --status "$status" --start "$start_time" --end "$end_time"
    --log-dir "$tmp_log_dir" --model "$MODEL"
    --repo-root "$REPO_ROOT" --runs-base "$RUNS_BASE"
  )

  [[ -n "$CODEGEN_BATCH_ID" ]] && log_args+=(--batch-id "$CODEGEN_BATCH_ID")
  [[ -f "$ISSUES_JSON" ]] && log_args+=(--issues-json "$ISSUES_JSON")
  [[ -n "$eval_json" && -f "$eval_json" ]] && log_args+=(--evaluation "$eval_json")
  [[ -n "$review_json" && -f "$review_json" ]] && log_args+=(--review "$review_json")

  python3 "${SCRIPT_DIR}/log_run.py" "${log_args[@]}"
}
```

- [ ] **Step 2: Add evaluate + review steps to `run_sequential()`**

In `run_sequential()`, after the `claude -p` call and before `log_run`, add:

```bash
    # --- Evaluate results ---
    local eval_json="${LOG_DIR}/eval_${num}.json"
    echo "  Evaluating results..."
    python3 "${SCRIPT_DIR}/evaluate_run.py" --repo-root "$REPO_ROOT" --output "$eval_json" || true

    # --- Review changes (unless --no-review) ---
    local review_json="${LOG_DIR}/review_${num}.json"
    if [[ "$NO_REVIEW" == false ]]; then
      echo "  Reviewing changes..."
      local review_args=(
        --repo-root "$REPO_ROOT" --issue "$num" --title "$title"
        --output "$review_json" --codegen-dir "$CODEGEN_DIR"
      )
      $AUTO_FIX && review_args+=(--auto-fix)
      python3 "${SCRIPT_DIR}/review_changes.py" "${review_args[@]}" || true
    fi
```

- [ ] **Step 3: Add `--no-review` and `--auto-fix` flags to arg parsing**

Add after the existing `--model` case:

```bash
    --no-review) NO_REVIEW=true; shift ;;
    --auto-fix)  AUTO_FIX=true; shift ;;
```

And add defaults after existing defaults:

```bash
NO_REVIEW=false
AUTO_FIX=false
```

- [ ] **Step 4: Update `log_run` calls in sequential and parallel modes**

Replace existing `log_run` calls to pass eval and review paths:

```bash
    log_run "$num" "$title" "$branch" "$status" "$start_time" "$end_time" "$LOG_DIR" "$eval_json" "$review_json"
```

- [ ] **Step 5: Update help text**

Add to the help output:

```bash
      echo "  --no-review   Skip the automated code review step"
      echo "  --auto-fix    Auto-fix errors found by the reviewer"
```

- [ ] **Step 6: Verify the script parses correctly**

Run: `bash -n codegen-bh/scripts/batch_generate_bh.sh`
Expected: No syntax errors.

- [ ] **Step 7: Dry-run test**

Run: `cd codegen-bh && bash scripts/batch_generate_bh.sh --issue 1153 --dry-run`
Expected: Shows prompt and branch info without executing.

- [ ] **Step 8: Commit**

```bash
git add codegen-bh/scripts/batch_generate_bh.sh
git commit -m "refactor: integrate evaluation and review into batch runner

- Replace embedded Python log_run() with call to log_run.py
- Add evaluation step (evaluate_run.py) after each claude run
- Add review step (review_changes.py) with --no-review/--auto-fix flags
- Remove ci_weekly references"
```

---

### Task 6: Update `codegen-bh/CLAUDE.md`

**Files:**
- Modify: `codegen-bh/CLAUDE.md`

**Why:** The current CLAUDE.md is the kernel-gen orchestrator (phase-based agent spawning). This branch is for issue solving, which has a different workflow.

- [ ] **Step 1: Replace CLAUDE.md content**

```markdown
# LLK CodeGen — Blackhole Issue Solver

This branch is focused on solving Blackhole P2 issues from `tenstorrent/tt-llk`.

## Quick Start

```bash
cd codegen-bh

# List all open Blackhole P2 issues
bash scripts/batch_generate_bh.sh

# Solve a single issue
bash scripts/batch_generate_bh.sh --issue 1153

# Solve with automated review and auto-fix
bash scripts/batch_generate_bh.sh --issue 1153 --auto-fix

# Dry-run (preview prompt, don't execute)
bash scripts/batch_generate_bh.sh --issue 1153 --dry-run

# Run all issues sequentially
bash scripts/batch_generate_bh.sh --run

# Run all issues in parallel (git worktrees)
bash scripts/batch_generate_bh.sh --run --parallel -j 4
```

## Pipeline Flow

For each issue, the batch runner executes:

1. **Branch** — Creates `{user}/issue-{num}-codegen-v{N}` from `origin/main`
2. **Solve** — Runs `claude -p` with issue context (model: opus by default)
3. **Evaluate** — `evaluate_run.py` checks compilation, tests, diff quality
4. **Review** — `review_changes.py` spawns a reviewer Claude (sonnet) to find mistakes
5. **Fix** — If `--auto-fix`, spawns a fixer Claude (opus) to address review errors
6. **Log** — `log_run.py` writes run.json, appends to runs.jsonl, extracts conversation

## Scripts

| Script | Purpose |
|--------|---------|
| `batch_generate_bh.sh` | Main runner — orchestrates the full pipeline |
| `log_run.py` | Logs run data (tokens, cost, git state, evaluation, review) |
| `evaluate_run.py` | Post-run evaluation (compile, test, diff analysis) |
| `review_changes.py` | Automated code review + optional auto-fix |
| `fetch_bh_issues.py` | Fetch Blackhole P2 issues from GitHub |
| `extract_conversation.py` | Parse CLI JSON into readable markdown |
| `check_compile.py` | Compilation checker for Blackhole kernels |
| `run_functional_test.py` | Functional test runner |
| `compiler.py` | SFPI compiler wrapper library |

## Logging

All logging is **script-driven** — the batch runner captures everything, not agents.

**Shared data directory:** `/proj_sw/user_dev/llk_code_gen/blackhole_issue_solver/`

```
blackhole_issue_solver/
├── runs.jsonl                           # Append-only run registry
└── blackhole_issue_{num}_v{N}/          # Per-run directory
    ├── run.json                         # Structured metadata
    ├── issue_{num}.json                 # Full CLI conversation (JSON)
    ├── issue_{num}.log                  # stderr log
    ├── conversation.md                  # Readable main thread
    ├── summary.md                       # Run metrics summary
    ├── agent_*.md                       # Per-subagent transcripts
    └── {flattened_file_names}           # Snapshots of changed files
```

## Dashboard

View runs at the LLK CodeGen Dashboard (see `/proj_sw/user_dev/llk_code_gen/dashboard/`).

The dashboard reads `runs.jsonl` and displays:
- Run status, duration, cost, token usage
- Issue metadata and labels
- Changed files
- Evaluation results (compile, test, diff score)
- Review verdict and comments

## Key Paths

| Path | Purpose |
|------|---------|
| `tt_llk_blackhole/` | Blackhole LLK implementations |
| `tt_llk_wormhole_b0/` | Wormhole reference implementations |
| `tt_llk_quasar/` | Quasar implementations |
| `tests/` | Test infrastructure (sources + python_tests) |

## Skills and Agents

This branch uses the team `.claude/` configuration:
- **Sages**: `sage-blackhole`, `sage-wormhole`, `sage-quasar` (architecture specialists)
- **Skills**: `/arch-lookup`, `/debug-kernel`, `/port-kernel`, `/run-test`
- **Agents**: `llk-debugger`, `llk-test-runner`
```

- [ ] **Step 2: Commit**

```bash
git add codegen-bh/CLAUDE.md
git commit -m "docs: replace kernel-gen orchestrator CLAUDE.md with issue solver docs"
```

---

### Task 7: Dashboard Updates

**Files:**
- Modify: `/proj_sw/user_dev/llk_code_gen/dashboard/core/utils.py`
- Modify: `/proj_sw/user_dev/llk_code_gen/dashboard/static/dashboard.js`

**What changes:**
1. `compute_display_status()` uses evaluation data when available
2. Detail view shows evaluation results (compile, test, diff score)
3. Detail view shows review verdict and comments

**Important:** Only modify the blackhole_issue_solver parts. Do NOT touch quasar rendering.

- [ ] **Step 1: Update `compute_display_status()` to use evaluation data**

In `/proj_sw/user_dev/llk_code_gen/dashboard/core/utils.py`, replace the function:

```python
def compute_display_status(run):
    """Compute display status: success / compiled / failed.

    Uses evaluation data if available for more accurate status.
    """
    status = run.get("status", "failed")

    # If we have evaluation data, use it for more precise status
    evaluation = run.get("evaluation")
    if evaluation:
        overall = evaluation.get("overall", "")
        if overall == "success":
            return "success"
        elif overall == "compile_failed":
            return "failed"
        elif overall == "tests_failed":
            return "compiled"
        elif overall == "no_changes":
            return "failed"
        elif overall == "compiled":
            return "compiled"

    # Fallback to existing logic
    if status == "failed":
        return "failed"
    tests_total = run.get("tests_total", 0)
    tests_passed = run.get("tests_passed", 0)
    if tests_total > 0 and tests_passed == tests_total:
        return "success"
    if status == "success" and tests_total > 0 and tests_passed == tests_total:
        return "success"
    # Issue solver runs may report success without formal test counts
    if status == "success" and run.get("issue"):
        return "success"
    return "compiled"
```

- [ ] **Step 2: Add evaluation + review rendering to `dashboard.js`**

In `/proj_sw/user_dev/llk_code_gen/dashboard/static/dashboard.js`, find the `renderIssueDetailHeader` function. After the `Changed Files` section (around the end of the function), append:

```javascript
      ${_renderEvaluation(r.evaluation)}

      ${_renderReview(r.review)}
```

And add these two helper functions (add them before `renderIssueDetailHeader`):

```javascript
function _renderEvaluation(evaluation) {
  if (!evaluation) return '';

  const overall = evaluation.overall || 'unknown';
  const statusColors = {
    success: 'var(--green)', compiled: 'var(--blue)',
    compile_failed: 'var(--red)', tests_failed: 'var(--yellow)',
    no_changes: 'var(--text-muted)', partial: 'var(--yellow)',
  };
  const color = statusColors[overall] || 'var(--text-muted)';

  let html = `
    <h3 style="margin-bottom:8px;margin-top:16px">Evaluation</h3>
    <div style="background:var(--bg-secondary);border-radius:6px;padding:12px;margin-bottom:16px">
      <div style="display:flex;gap:16px;margin-bottom:8px">
        <span><strong>Overall:</strong> <span style="color:${color}">${overall.toUpperCase()}</span></span>`;

  const diffAnalysis = evaluation.diff_analysis;
  if (diffAnalysis && diffAnalysis.score !== undefined) {
    html += `<span><strong>Score:</strong> ${diffAnalysis.score}/100</span>`;
  }
  if (diffAnalysis && diffAnalysis.summary) {
    html += `<span style="color:var(--text-muted)">${escapeHtml(diffAnalysis.summary)}</span>`;
  }

  html += '</div>';

  // Compilation results
  const comp = evaluation.compilation;
  if (comp && comp.files && comp.files.length > 0) {
    html += '<div style="margin-top:8px"><strong>Compilation:</strong>';
    for (const f of comp.files) {
      const icon = f.status === 'passed' ? '<span style="color:var(--green)">PASS</span>'
                 : f.status === 'skipped' ? '<span style="color:var(--text-muted)">SKIP</span>'
                 : '<span style="color:var(--red)">FAIL</span>';
      html += `<div style="font-family:monospace;font-size:12px;padding:2px 0">${icon} ${escapeHtml(f.file)}</div>`;
    }
    html += '</div>';
  }

  // Test results
  const tests = evaluation.tests;
  if (tests && tests.kernels && tests.kernels.length > 0) {
    html += '<div style="margin-top:8px"><strong>Tests:</strong>';
    for (const t of tests.kernels) {
      const icon = t.status === 'passed' ? '<span style="color:var(--green)">PASS</span>'
                 : t.status === 'skipped' ? '<span style="color:var(--text-muted)">SKIP</span>'
                 : '<span style="color:var(--red)">FAIL</span>';
      html += `<div style="font-family:monospace;font-size:12px;padding:2px 0">${icon} ${escapeHtml(t.kernel)}</div>`;
    }
    html += '</div>';
  }

  html += '</div>';
  return html;
}


function _renderReview(review) {
  if (!review) return '';
  if (review.status === 'skipped') return '';

  const verdict = review.verdict || 'unknown';
  const verdictColor = verdict === 'approve' ? 'var(--green)' : 'var(--yellow)';
  const comments = review.comments || [];

  let html = `
    <h3 style="margin-bottom:8px;margin-top:16px">Code Review</h3>
    <div style="background:var(--bg-secondary);border-radius:6px;padding:12px;margin-bottom:16px">
      <div style="display:flex;gap:16px;margin-bottom:8px">
        <span><strong>Verdict:</strong> <span style="color:${verdictColor}">${verdict.toUpperCase()}</span></span>
        <span><strong>Comments:</strong> ${comments.length}</span>`;

  if (review.model) {
    html += `<span style="color:var(--text-muted)">Reviewer: ${escapeHtml(review.model)}</span>`;
  }

  html += '</div>';

  if (review.summary) {
    html += `<div style="color:var(--text-muted);margin-bottom:8px">${escapeHtml(review.summary)}</div>`;
  }

  for (const c of comments) {
    const sev = (c.severity || 'info').toUpperCase();
    const sevColor = c.severity === 'error' ? 'var(--red)' : 'var(--yellow)';
    html += `<div style="border-left:3px solid ${sevColor};padding:4px 8px;margin:4px 0;font-size:12px">
      <strong style="color:${sevColor}">[${sev}]</strong>
      <span style="font-family:monospace">${escapeHtml(c.file || '?')}:${c.line || '?'}</span>
      — ${escapeHtml(c.message || '')}
    </div>`;
  }

  // Fix result if available
  const fix = review.fix_result;
  if (fix) {
    const fixColor = fix.status === 'completed' ? 'var(--green)' : 'var(--red)';
    html += `<div style="margin-top:8px"><strong>Auto-fix:</strong>
      <span style="color:${fixColor}">${fix.status}</span>
      ${fix.comments_addressed ? `(${fix.comments_addressed} errors addressed)` : ''}
    </div>`;
  }

  html += '</div>';
  return html;
}
```

- [ ] **Step 3: Add evaluation summary cards to overview**

In `dashboard.js`, find the issue solver summary cards section (search for `"Solved"` or `isIssueSolver()`). After the existing cards, add an evaluation breakdown. In the `renderSummaryCards` function, for the issue solver project, add:

```javascript
  // After existing cards for issue solver
  if (isIssueSolver()) {
    const evaluated = runs.filter(r => r.evaluation);
    const evalSuccess = evaluated.filter(r => r.evaluation.overall === 'success').length;
    const reviewed = runs.filter(r => r.review && r.review.verdict);
    const approved = reviewed.filter(r => r.review.verdict === 'approve').length;

    // Add these to the existing cards array
    cards += `
      <div class="card">
        <div class="label">Eval Passed</div>
        <div class="value green">${evalSuccess}/${evaluated.length}</div>
      </div>
      <div class="card">
        <div class="label">Review Approved</div>
        <div class="value green">${approved}/${reviewed.length}</div>
      </div>`;
  }
```

- [ ] **Step 4: Verify dashboard loads without errors**

Run:
```bash
cd /proj_sw/user_dev/llk_code_gen
python -c "from dashboard.core.utils import compute_display_status; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit dashboard changes**

```bash
cd /proj_sw/user_dev/llk_code_gen
git add dashboard/core/utils.py dashboard/static/dashboard.js
git commit -m "feat(dashboard): show evaluation + review results for BH issue solver

- compute_display_status() uses evaluation.overall when available
- Detail view shows compilation/test/diff evaluation results
- Detail view shows review verdict and comments
- Overview shows eval/review summary cards for issue solver
- No changes to quasar or kernel gen rendering"
```

---

### Task 8: Final Cleanup and Verification

**Files:**
- Modify: `codegen-bh/scripts/README.md`

- [ ] **Step 1: Update scripts README**

```markdown
# Blackhole Issue Solver Scripts

Scripts for fetching, solving, evaluating, and reviewing Blackhole P2 issues.

## Prerequisites

- `gh` CLI authenticated (`gh auth status`)
- `claude` CLI available on PATH
- Python 3.10+

## Quick Start

```bash
cd codegen-bh

# List all open Blackhole P2 issues
bash scripts/batch_generate_bh.sh

# Solve a single issue (with review)
bash scripts/batch_generate_bh.sh --issue 1153

# Solve with auto-fix for review errors
bash scripts/batch_generate_bh.sh --issue 1153 --auto-fix

# Skip the review step
bash scripts/batch_generate_bh.sh --issue 1153 --no-review

# Dry-run (preview prompt without executing)
bash scripts/batch_generate_bh.sh --issue 1153 --dry-run
```

## Pipeline

```
fetch_bh_issues.py → batch_generate_bh.sh → evaluate_run.py → review_changes.py → log_run.py
                                                                                      ↓
                                                          llk_code_gen/blackhole_issue_solver/runs.jsonl
```

## Scripts

| Script | Purpose |
|--------|---------|
| `batch_generate_bh.sh` | Main runner — fetch, solve, evaluate, review, log |
| `log_run.py` | Create run directory, parse tokens/cost, write runs.jsonl |
| `evaluate_run.py` | Post-run: compile changed files, run tests, score diff |
| `review_changes.py` | Spawn reviewer Claude, optionally spawn fixer Claude |
| `fetch_bh_issues.py` | Fetch BH P2 issues from GitHub via `gh` CLI |
| `extract_conversation.py` | Parse CLI JSON into readable markdown logs |
| `check_compile.py` | Compile-check a Blackhole kernel header |
| `run_functional_test.py` | Run functional tests for a kernel |
| `compiler.py` | SFPI compiler wrapper (used by check_compile.py) |

## Outputs

| Path | Content |
|------|---------|
| `artifacts/bh_p2_issues.json` | Cached issue data from GitHub |
| `/tmp/codegen_bh_logs_*/` | Temp logs during batch runs |
| `../../llk_code_gen/blackhole_issue_solver/runs.jsonl` | Run history |
| `../../llk_code_gen/blackhole_issue_solver/blackhole_issue_*_v*/` | Per-run data |
```

- [ ] **Step 2: Full dry-run test**

```bash
cd /proj_sw/user_dev/nstamatovic/tt-llk/codegen-bh
bash scripts/batch_generate_bh.sh --dry-run
```

Expected: Lists issues, shows prompts, no execution.

- [ ] **Step 3: Verify Python scripts have no import errors**

```bash
cd /proj_sw/user_dev/nstamatovic/tt-llk
python3 -c "import py_compile; py_compile.compile('codegen-bh/scripts/log_run.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('codegen-bh/scripts/evaluate_run.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('codegen-bh/scripts/review_changes.py', doraise=True)"
echo "All scripts compile OK"
```

- [ ] **Step 4: Final commit (scripts README)**

```bash
cd /proj_sw/user_dev/nstamatovic/tt-llk
git add codegen-bh/scripts/README.md
git commit -m "docs: update scripts README for issue solver pipeline"
```

---

## Summary of Changes

| What | Before | After |
|------|--------|-------|
| **Logging** | Agent-driven (unreliable, often missing) | Script-driven via `log_run.py` (always captures) |
| **Token tracking** | Embedded Python with missing `import glob` bug | Standalone `parse_cli_json()` with validation |
| **Result evaluation** | Exit code only (0=ok, else=crash) | `evaluate_run.py`: compile + test + diff analysis |
| **Code review** | None | `review_changes.py`: reviewer Claude + fixer Claude |
| **Cron job** | `ci_weekly_bh.py` | Removed (separate branch later) |
| **Agents** | 6 kernel-gen agents in `codegen-bh/agents/` | Team `.claude/` (sages, debugger, test-runner) |
| **CLAUDE.md** | Kernel-gen orchestrator (phase-based) | Issue solver docs (pipeline description) |
| **Dashboard** | Shows basic run data, missing evaluation | Shows evaluation results + review feedback |

## Cross-Run Learning

**Not implemented** per user request. `runs.jsonl` accumulates history but is not read by subsequent runs. This can be added later if needed.

## Monitoring Decision

Claude Code offers OpenTelemetry-based monitoring, but it requires external OTLP collector infrastructure (Prometheus/Jaeger/etc). For this use case, parsing the CLI `--output-format json` output is:
- Simpler (no external services)
- Sufficient (captures tokens, cost, turns, duration per model)
- Already partially implemented (just needed bug fixes)

If we later need real-time monitoring or multi-user dashboards, OTel can be added without changing the scripts — it runs alongside.
