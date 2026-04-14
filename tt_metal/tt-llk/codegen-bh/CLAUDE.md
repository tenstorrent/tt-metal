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
3. **Evaluate** — `evaluate_run.py` checks diff quality
4. **Review** — `review_changes.py` spawns a reviewer Claude (sonnet) to find mistakes
5. **Fix** — If `--auto-fix`, spawns a fixer Claude (opus) to address review errors
6. **Log** — `log_run.py` writes run.json, appends to runs.jsonl, extracts conversation

## Scripts

| Script | Purpose |
|--------|---------|
| `batch_generate_bh.sh` | Main runner — orchestrates the full pipeline |
| `log_run.py` | Logs run data (tokens, cost, git state, evaluation, review) |
| `evaluate_run.py` | Post-run evaluation (diff analysis) |
| `review_changes.py` | Automated code review + optional auto-fix |
| `fetch_bh_issues.py` | Fetch Blackhole P2 issues from GitHub |
| `extract_conversation.py` | Parse CLI JSON into readable markdown |

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
    ├── cli_output.json                  # Copy for dashboard backfill
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
