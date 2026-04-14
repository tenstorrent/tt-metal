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
fetch_bh_issues.py -> batch_generate_bh.sh -> evaluate_run.py -> review_changes.py -> log_run.py
                                                                                         |
                                                          llk_code_gen/blackhole_issue_solver/runs.jsonl
```

## Scripts

| Script | Purpose |
|--------|---------|
| `batch_generate_bh.sh` | Main runner -- fetch, solve, evaluate, review, log |
| `log_run.py` | Create run directory, parse tokens/cost, write runs.jsonl |
| `evaluate_run.py` | Post-run: score diff quality |
| `review_changes.py` | Spawn reviewer Claude, optionally spawn fixer Claude |
| `fetch_bh_issues.py` | Fetch BH P2 issues from GitHub via `gh` CLI |
| `extract_conversation.py` | Parse CLI JSON into readable markdown logs |

## Outputs

| Path | Content |
|------|---------|
| `artifacts/bh_p2_issues.json` | Cached issue data from GitHub |
| `/tmp/codegen_bh_logs_*/` | Temp logs during batch runs |
| `../../llk_code_gen/blackhole_issue_solver/runs.jsonl` | Run history |
| `../../llk_code_gen/blackhole_issue_solver/blackhole_issue_*_v*/` | Per-run data |
