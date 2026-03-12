---
name: eval-dev
description: Orient yourself for developing the eval system. Loads architecture, file map, data flow, testing patterns, and current working state. Use before any eval/ work.
---

# Eval System Development Context

Load this context, then handle the user's request. Do NOT dump this back at the user — internalize it and get to work.

## Step 1: Read Current State

Run these in parallel:

1. `git diff --name-only HEAD -- eval/ .claude/skills/ .claude/agents/ .claude/scripts/tdd-pipeline/` — what's already modified
2. `git log --oneline -5 -- eval/ .claude/skills/ .claude/agents/ .claude/scripts/tdd-pipeline/` — recent changes for context
3. `ls eval/prompts/*.txt` and `ls eval/golden_tests/` — current prompts and golden suites

Internalize the results. Only mention to the user if something is directly relevant to their request (e.g., "heads up, you have uncommitted changes in that file").

## Step 2: Architecture Reference

Internalize this map. Do not print it unless asked.

### Data Flow

```
run_eval.sh (orchestrator)
  ├── clones repo, builds, launches claude -p
  ├── calls eval_test_runner.sh after claude exits
  │     ├── validates API contract (validate_contract.py)
  │     ├── runs pytest with hang_plugin.py
  │     ├── classifies failures (classify_failures.py → test_results.json)
  │     └── outputs: junit.xml, test_results.json, golden_results.txt
  ├── runs score.py → score.json
  └── ingests into DB (ingest.py → db.py → eval_runs.db)

dashboard.py reads DB → generates static HTML
list_runs.py reads DB → prints run paths
annotate.py writes to DB (manual quality scores)
quick_ingest.py = shortcut: run golden tests + ingest in one command
```

### File Map

| File | Purpose | Key functions/classes |
|------|---------|---------------------|
| `eval/run_eval.sh` | Top-level eval orchestrator. Clones, builds, runs claude, collects results. All runs parallel. | `run_single()`, `run_golden_tests()`, `monitor_progress()`, `detect_phase()` |
| `eval/eval_test_runner.sh` | Runs golden tests with device lock, hang detection, failure classification. | Wraps pytest → junit.xml → classify → summary |
| `eval/db.py` | SQLite schema + CRUD. Tables: `runs`, `test_results`, `score_criteria`, `kernels`, `host_code`, `artifacts`. | `connect()`, `insert_run()`, `insert_test_results_batch()`, `get_stats()` |
| `eval/ingest.py` | Post-run ingestion: reads score.json + test_results.json + source files → DB. | `ingest_run()`, `_collect_kernels()`, `_collect_host_code()` |
| `eval/classify_failures.py` | Regex-based failure categorization from JUnit XML. Categories: hang, OOM, compilation, signature, numerical, other. | `classify()`, `parse_junit_xml()`, `PATTERNS` |
| `eval/validate_contract.py` | Offline check: does generated op match api_contract.md signature? | `parse_contract()`, `validate_operation()` |
| `eval/dashboard.py` | Generates self-contained HTML dashboard from DB. All HTML in Python strings. | `generate_html()`, `serve()`, `_html_runs_table()`, `_html_run_detail()` |
| `eval/list_runs.py` | Query runs with filters, resolve clone filesystem paths. | `query_runs()`, `resolve_clone_path()`, `print_runs()` |
| `eval/quick_ingest.py` | One-command: run golden tests + ingest results for a local op. | `quick_ingest()` |
| `eval/annotate.py` | CLI to add manual 1-5 star ratings to runs. | Thin wrapper around `db.annotate_run()` |
| `eval/hang_plugin.py` | Pytest plugin: skips remaining parametrizations after a hang. | `pytest_runtest_makereport`, `pytest_runtest_setup` |
| `eval/generality_problem.md` | Design doc / roadmap: difficulty tiers, iterative reprompting, test visibility. Open questions for future work. | Not code — read for strategic context |

### DB Schema (eval/db.py)

```
runs: id, timestamp, prompt_name, run_number, starting_branch, starting_commit,
      created_branch, score_total, score_grade, golden_passed, golden_total,
      annotation_score, annotation_notes, golden_name, duration_seconds

test_results: id, run_id→runs, test_name, test_file, shape, status,
              failure_category, failure_message

score_criteria: id, run_id→runs, criterion, raw_score, weight, weighted_score

kernels: id, run_id→runs, filename, source_code       (C++ kernel files)
host_code: id, run_id→runs, filename, source_code     (Python program descriptor + entry point)
artifacts: id, run_id→runs, name, content              (self_reflection.md etc.)
```

Migrations live in `db.py:MIGRATIONS` — list of `ALTER TABLE` statements, applied idempotently.

### Golden Tests Structure

Each op in `eval/golden_tests/{op_name}/`:
```
__init__.py, conftest.py, helpers.py, api_contract.md,
test_golden_shapes.py (~70 shape parametrizations),
test_golden_modes.py (param variations + data distributions),
test_golden_validation.py (input rejection tests, no device)
```

Created by `/golden-tests` skill. Prompts in `eval/prompts/{op_name}.txt` have `# golden: {op_name}` tags linking them.

### Skills ↔ Eval Files

| Skill | Creates/Reads | Eval files involved |
|-------|--------------|-------------------|
| `/golden-tests` | Creates | `eval/golden_tests/{op}/`, `eval/prompts/{op}.txt` |
| `/list-runs` | Reads | `eval/list_runs.py` → DB |
| `/create-op` | Triggers | Prompt consumed by `run_eval.sh`; golden tests run post-claude |
| `/nuke-op` | Unrelated | Removes op code, not eval infra |
| `/spec-op` | Creates | `op_spec.md` (consumed by `/create-op`, not eval directly) |

### Dashboard Specifics

`dashboard.py` is a single-file HTML generator. No templates, no JS framework. Structure:

- `_html_head()` — CSS + highlight.js CDN links
- `_html_stats()` — summary cards (total runs, avg score, pass rate)
- `_html_failure_bars()` — colored bar chart of failure categories
- `_html_runs_table()` — main table with expandable detail rows
- `_html_run_detail()` — tabbed detail view (Tests, Score, Kernels, Host-Side, Self-Reflection)
- `_html_foot()` — JavaScript for toggling, tabs, code highlighting

Color maps: `GRADE_COLORS`, `STATUS_COLORS`, `CATEGORY_COLORS` — all at top of file.

Serve: `python3 -m eval.dashboard [--port 8080]`. Refresh endpoint: `/refresh`.
Static export: `python3 -m eval.dashboard --generate-only -o /tmp/dashboard.html`.

## Step 3: Testing Your Changes

### Unit tests (no device, fast)
```bash
pytest eval/tests/ -v
```
Tests exist for: `db.py`, `classify_failures.py`, `dashboard.py`, `ingest.py`. All use in-memory SQLite or temp files.

### Preview dashboard
```bash
source python_env/bin/activate && python3 -m eval.dashboard --generate-only -o /tmp/dashboard.html
# Then open /tmp/dashboard.html in browser
```

### Validate a contract
```bash
source python_env/bin/activate && python3 -m eval.validate_contract eval/golden_tests/<op>/
```

### Preview list_runs
```bash
source python_env/bin/activate && python3 -m eval.list_runs --last 5
```

### Test golden test syntax (no device)
```bash
python3 -c "import ast; ast.parse(open('eval/golden_tests/<op>/test_golden_shapes.py').read())"
```

### Full eval run (heavyweight, use sparingly)
```bash
./eval/run_eval.sh eval/prompts/<op>.txt --runs 1
```

## Step 4: Common Change Patterns

**New DB column**: `db.py` MIGRATIONS list → `insert_run()` params → `ingest.py` population → `dashboard.py` display → `eval/tests/test_db.py`

**New failure category**: `classify_failures.py` PATTERNS → `dashboard.py` CATEGORY_COLORS → `eval/tests/test_classify.py`

**Dashboard UI change**: Edit the `_html_*` functions in `dashboard.py`. CSS is in `_html_head()`. JS is in `_html_foot()`. Preview with `--generate-only`.

**New golden test suite**: Use `/golden-tests <op>` or manually create under `eval/golden_tests/<op>/`. Must have `api_contract.md` + `helpers.py` + test files. Prompt needs `# golden: <op>` tag.

**Modify run_eval.sh**: This runs inside clones, not the source repo. Test with `./eval/run_eval.sh eval/prompts/<op>.txt --runs 1`. Changes must be committed+pushed to take effect in eval runs (clone pulls from origin).

**New eval CLI tool**: Add `eval/<tool>.py` with `if __name__ == "__main__": main()`. Run as `python3 -m eval.<tool>`. Optionally add a skill in `.claude/skills/<name>/SKILL.md`.

## Step 5: Handle the User's Request

You now have full context. Proceed with whatever the user asked for. If they haven't specified a task yet, briefly acknowledge you're oriented and ask what they need.
