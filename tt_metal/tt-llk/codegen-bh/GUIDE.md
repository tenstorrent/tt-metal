# Blackhole Issue Solver -- User Guide

## What This Is

An automated pipeline that takes open Blackhole P2 issues from `tenstorrent/tt-llk`, spawns Claude to solve them, evaluates the results, reviews the code, and logs everything to a shared dashboard.

The pipeline runs entirely from scripts -- no manual agent configuration needed. Each issue gets its own git branch, and every run is logged with token usage, cost, evaluation results, and review feedback.

## Prerequisites

| Requirement | Check |
|------------|-------|
| `gh` CLI authenticated | `gh auth status` |
| `claude` CLI on PATH | `claude --version` |
| Python 3.10+ | `python3 --version` |
| On the right branch | `git branch --show-current` should show `$(whoami)/bh_issue_solver` |

## Quick Start

```bash
cd /proj_sw/user_dev/$(whoami)/tt-llk/codegen-bh

# See what issues are available
bash scripts/batch_generate_bh.sh

# Solve one issue (full pipeline: solve + evaluate + review)
bash scripts/batch_generate_bh.sh --issue 1153

# Solve one issue and auto-fix review errors
bash scripts/batch_generate_bh.sh --issue 1153 --auto-fix

# Preview what would happen without running anything
bash scripts/batch_generate_bh.sh --issue 1153 --dry-run
```

---

## How the Pipeline Works

When you run `bash scripts/batch_generate_bh.sh --issue 1153`, this is what happens:

### Step 1: Fetch Issues

The script checks if `artifacts/bh_p2_issues.json` exists. If not, it calls `fetch_bh_issues.py` to pull all open issues from GitHub with labels `blackhole` + `P2`. Use `--refresh` to force re-fetch.

### Step 2: Create Branch

For each issue, the script creates a versioned branch:

```
{user}/issue-1153-codegen-v1
{user}/issue-1153-codegen-v2   (if v1 already exists)
```

Branched from `origin/main`, so every run starts clean.

### Step 3: Solve (Claude)

Spawns `claude -p` with a prompt like:

> Investigate and fix Blackhole issue #1153: {title}. Work autonomously -- use superpowers skills, do not ask questions. Test your changes thoroughly before committing.

Claude has access to the full repo, the team `.claude/` config (sages, debugger, test-runner skills), and runs with `--dangerously-skip-permissions --effort max`.

Default model: `claude-opus-4-6`. Override with `--model sonnet` or `--model haiku`.

### Step 4: Evaluate (`evaluate_run.py`)

After Claude finishes, the evaluator checks what actually happened:

| Check | What it does |
|-------|-------------|
| **Changed files** | `git diff --name-only origin/main...HEAD` |
| **Compilation** | Runs `check_compile.py` on every changed `.h` file under `tt_llk_blackhole/` |
| **Tests** | Runs `run_functional_test.py --quick` for kernels whose files changed |
| **Diff score** | 0-100 score: kernel changes (+50), test changes (+30), both (+20) |

The overall verdict is one of: `success`, `compiled`, `compile_failed`, `tests_failed`, `no_changes`, `partial`.

### Step 5: Review (`review_changes.py`)

A **second** Claude instance (Sonnet, for speed) reviews the diff. It looks for:

- Correctness issues (does the change actually fix the issue?)
- Compilation problems (wrong includes, undefined symbols, type mismatches)
- Logic errors (wrong register, off-by-one, broken init/uninit symmetry)
- Missing changes (files that should have been updated but weren't)
- Regressions (changes that could break existing tests)

It does NOT flag style, formatting, or missing comments.

The reviewer outputs a structured verdict:
```json
{
  "verdict": "approve" or "request_changes",
  "summary": "one-line finding",
  "comments": [
    {"file": "path/to/file.h", "line": 42, "severity": "error", "message": "..."}
  ]
}
```

### Step 5b: Auto-Fix (optional)

If you passed `--auto-fix` and the reviewer found `error`-severity comments, a **third** Claude instance (Opus, for quality) reads the review comments and fixes them. It then commits with:

```
fix: address review comments for issue #1153
```

Skip the review entirely with `--no-review`.

### Step 6: Log (`log_run.py`)

Everything gets logged to the shared directory:

```
/proj_sw/user_dev/llk_code_gen/blackhole_issue_solver/
├── runs.jsonl                              # One JSON line per run (append-only)
└── blackhole_issue_1153_v1/                # This run's directory
    ├── run.json                            # Structured metadata
    ├── issue_1153.json                     # Full Claude conversation (raw JSON)
    ├── issue_1153.log                      # stderr log
    ├── cli_output.json                     # Copy for dashboard backfill
    ├── conversation.md                     # Readable main thread
    ├── summary.md                          # Token/cost/tool usage summary
    ├── agent_analyze_1153.md               # Per-subagent transcripts
    ├── tt_llk_blackhole_llk_lib_llk_foo.h  # Snapshots of changed files
    └── ...
```

The `runs.jsonl` entry includes:
- Token usage per model (opus, haiku, sonnet) with cost breakdown
- Evaluation results (compile, test, diff score)
- Review verdict and comments
- Git branch, commit hash, changed files list
- Issue metadata (number, title, labels, URL)

---

## Running Multiple Issues

### Sequential (default)

```bash
# All open BH P2 issues, one at a time (stops on first crash)
bash scripts/batch_generate_bh.sh --run

# With review and auto-fix
bash scripts/batch_generate_bh.sh --run --auto-fix
```

### Parallel (git worktrees)

Each issue gets an isolated git worktree under `/tmp/codegen_bh_worktree_{num}/`. Worktrees are cleaned up after completion.

```bash
# All issues in parallel (unlimited concurrency)
bash scripts/batch_generate_bh.sh --run --parallel

# Max 4 concurrent
bash scripts/batch_generate_bh.sh --run --parallel -j 4
```

### Filtering

```bash
# Only issues with an additional label
bash scripts/batch_generate_bh.sh --label LLK --run

# Single issue by number
bash scripts/batch_generate_bh.sh --issue 960

# Re-fetch issues from GitHub before running
bash scripts/batch_generate_bh.sh --refresh --run
```

---

## All CLI Flags

```
bash scripts/batch_generate_bh.sh [OPTIONS]

Options:
  --run           Run codegen (without this, just lists issues)
  --issue NUM     Run a single issue by number
  --label LABEL   Filter by additional GitHub label (e.g., LLK)
  --refresh       Re-fetch issues from GitHub
  --parallel      Run issues in parallel (git worktrees)
  -j N            Max concurrent parallel jobs
  --model MODEL   Claude model (default: claude-opus-4-6)
  --no-review     Skip the automated code review step
  --auto-fix      Auto-fix errors found by the reviewer
  --dry-run       Show prompts without executing
```

---

## Individual Scripts

You can run each script independently for debugging or manual workflows.

### Fetch issues

```bash
# Fetch and display summary table
python scripts/fetch_bh_issues.py --summary

# Include closed issues
python scripts/fetch_bh_issues.py --state all --summary

# JSON to stdout (for piping)
python scripts/fetch_bh_issues.py --stdout | jq '.issues[] | {number, title}'
```

### Evaluate a run (after Claude has made changes)

```bash
python scripts/evaluate_run.py --repo-root /proj_sw/user_dev/$(whoami)/tt-llk
python scripts/evaluate_run.py --repo-root /proj_sw/user_dev/$(whoami)/tt-llk --output /tmp/eval.json
```

### Review changes (after Claude has made changes)

```bash
# Review only
python scripts/review_changes.py \
  --repo-root /proj_sw/user_dev/$(whoami)/tt-llk \
  --issue 1153 --title "Fix unpack_reduce"

# Review + auto-fix
python scripts/review_changes.py \
  --repo-root /proj_sw/user_dev/$(whoami)/tt-llk \
  --issue 1153 --title "Fix unpack_reduce" \
  --auto-fix

# Use a specific model for review
python scripts/review_changes.py \
  --repo-root /proj_sw/user_dev/$(whoami)/tt-llk \
  --issue 1153 --title "Fix unpack_reduce" \
  --model claude-opus-4-6
```

### Log a run manually

```bash
python scripts/log_run.py \
  --issue 1153 --title "Fix unpack_reduce" \
  --branch $(whoami)/issue-1153-codegen-v1 \
  --status completed \
  --start 2026-04-07T10:00:00Z --end 2026-04-07T11:00:00Z \
  --log-dir /tmp/codegen_bh_logs_20260407/ \
  --model claude-opus-4-6 \
  --repo-root /proj_sw/user_dev/$(whoami)/tt-llk
```

### Compile-check a file

```bash
cd codegen-bh
source ../tests/.venv/bin/activate
PYTHONPATH=.. python scripts/check_compile.py ../tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sigmoid.h -v
```

### Run functional tests

```bash
cd codegen-bh
source ../tests/.venv/bin/activate
python scripts/run_functional_test.py sigmoid --arch blackhole --quick
python scripts/run_functional_test.py --list --arch blackhole
```

---

## Dashboard

The LLK CodeGen Dashboard at `/proj_sw/user_dev/llk_code_gen/dashboard/` reads `runs.jsonl` and displays everything.

### What it shows

**Overview tab:**
- Run list with status, issue number, description, duration, date
- Summary cards: Solved, In Progress, Failed, Issues Touched, Files Changed, Eval Passed, Review OK

**Detail view (click a run):**
- Issue metadata with GitHub link
- Changed files list
- Evaluation section: compilation pass/fail per file, test pass/fail per kernel, diff score
- Code review section: verdict, reviewer comments with severity/file/line, auto-fix result
- Token usage and cost breakdown
- Agent logs (conversation transcripts)

**Coverage tab:**
- GitHub issues with P2+blackhole labels
- PR status: merged, in review, active, todo
- Changed files aggregated across PRs

### How data flows to the dashboard

```
batch_generate_bh.sh
  └─> log_run.py
        ├─> blackhole_issue_{num}_v{N}/run.json     (structured metadata)
        ├─> blackhole_issue_{num}_v{N}/cli_output.json (for dashboard backfill)
        └─> runs.jsonl                                (append-only registry)
                 │
                 └─> dashboard/core/runs.py reads this
                       └─> dashboard/static/dashboard.js renders it
```

The dashboard also has a backfill mechanism: if `runs.jsonl` has zero tokens, it tries to read `cli_output.json` from the run directory to extract the data. Our `log_run.py` writes both the parsed tokens AND the `cli_output.json` file, so the dashboard always has the data.

---

## Skills and Agents Available

The team `.claude/` config gives Claude access to architecture specialists and utilities when solving issues.

### Agents (spawned as subagents by Claude)

| Agent | What it does |
|-------|-------------|
| `sage-blackhole` | Blackhole architecture specialist -- reads only `tt_llk_blackhole/` |
| `sage-wormhole` | Wormhole reference specialist -- reads only `tt_llk_wormhole_b0/` |
| `sage-quasar` | Quasar architecture specialist -- reads only `tt_llk_quasar/` |
| `llk-debugger` | Fixes compilation and runtime errors with structured investigation |
| `llk-test-runner` | Runs tests via `run_test.sh` wrapper (never raw pytest) |

### Skills (user-invocable slash commands)

| Skill | Usage | What it does |
|-------|-------|-------------|
| `/arch-lookup` | `/arch-lookup "How does SFPMAD work?"` | Dispatches relevant sages in parallel |
| `/debug-kernel` | `/debug-kernel path/to/kernel.h` | Spawns debugger with error context |
| `/port-kernel` | `/port-kernel reduce --from wormhole --to blackhole` | Coordinates source + target sages |
| `/run-test` | `/run-test test_pack_untilize.py` | Spawns test runner with proper env |

---

## Logging Design

### Why script-driven, not agent-driven

Previously, agents were responsible for logging their own data (writing to `LOG_DIR/agent_*.md`, appending to `runs.jsonl`). This was unreliable -- agents often skipped logging, wrote inconsistent schemas, or crashed before logging.

Now, **all logging happens in the runner scripts**:
- `batch_generate_bh.sh` captures the CLI JSON output
- `log_run.py` parses it and writes structured data
- The agent doesn't need to know about logging at all

This means every run gets logged, every time, with consistent data.

### What gets captured

| Data | Source | Reliability |
|------|--------|-------------|
| Token usage (per model) | CLI `--output-format json` last entry `modelUsage` | High -- always present if Claude runs |
| Cost (USD) | CLI `total_cost_usd` field | High |
| Duration | CLI `duration_ms` field | High |
| Turn count | CLI `num_turns` field | High |
| Changed files | `git diff --name-only` | High |
| Commit hash | `git rev-parse --short HEAD` | High |
| Compilation result | `evaluate_run.py` -> `check_compile.py` | High |
| Test result | `evaluate_run.py` -> `run_functional_test.py` | Medium (tests may not exist) |
| Review verdict | `review_changes.py` -> Claude sonnet | Medium (parsing may fail) |
| Conversation transcript | `extract_conversation.py` | High |

---

## Troubleshooting

### "No matching Blackhole P2 issues found"

The issue cache may be stale. Re-fetch:
```bash
bash scripts/batch_generate_bh.sh --refresh
```

Or check if the issue has the right labels:
```bash
gh issue view 1153 -R tenstorrent/tt-llk --json labels
```

### Claude crashes or times out

Check the stderr log:
```bash
cat /tmp/codegen_bh_logs_*/issue_1153.log
```

The default timeout is 4 hours. Claude runs with `--effort max` which uses extended thinking.

### Review parsing fails

The reviewer outputs JSON, but sometimes Claude wraps it in markdown fences. The parser handles this, but if it fails, the review result will show `status: "parse_error"` with the raw output. The run still gets logged -- just without structured review data.

### Dashboard not showing new runs

The dashboard reads `runs.jsonl` on each page load. Check the file:
```bash
tail -1 /proj_sw/user_dev/llk_code_gen/blackhole_issue_solver/runs.jsonl | python3 -m json.tool
```

If the entry is there but tokens are zero, the dashboard will try to backfill from `cli_output.json` in the run directory.

### Branch already exists

The script auto-increments: `v1`, `v2`, `v3`, etc. If a branch exists (locally or on remote), it picks the next version.

---

## Directory Structure

```
tt-llk/
├── .claude/                         # Team config (from claude-code-team-setup)
│   ├── CLAUDE.md                    # Repo overview, architecture, skills reference
│   ├── agents/                      # 5 agents (3 sages + debugger + test-runner)
│   ├── skills/                      # 4 skills (arch-lookup, debug, port, run-test)
│   ├── references/                  # common-errors.md, porting-guide.md
│   └── scripts/                     # run_test.sh wrapper
│
├── codegen-bh/                      # Issue solver workspace
│   ├── CLAUDE.md                    # Issue solver documentation
│   ├── GUIDE.md                     # This file
│   ├── agents/                      # Empty (agents live in .claude/)
│   ├── artifacts/                   # Cached issues JSON, generated artifacts
│   └── scripts/                     # All pipeline scripts
│       ├── batch_generate_bh.sh     # Main runner
│       ├── log_run.py               # Run logging
│       ├── evaluate_run.py          # Post-run evaluation
│       ├── review_changes.py        # Code reviewer + fixer
│       ├── fetch_bh_issues.py       # GitHub issue fetcher
│       ├── extract_conversation.py  # CLI JSON -> markdown
│       ├── check_compile.py         # Compilation checker
│       ├── run_functional_test.py   # Functional test runner
│       └── compiler.py              # SFPI compiler wrapper
│
├── tt_llk_blackhole/                # Blackhole LLK implementations
├── tt_llk_wormhole_b0/              # Wormhole reference implementations
├── tt_llk_quasar/                   # Quasar implementations
└── tests/                           # Test infrastructure

/proj_sw/user_dev/llk_code_gen/
├── blackhole_issue_solver/          # Shared run data
│   ├── runs.jsonl                   # Run registry
│   └── blackhole_issue_*_v*/        # Per-run directories
└── dashboard/                       # Flask web dashboard
```
