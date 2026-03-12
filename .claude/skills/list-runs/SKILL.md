---
name: list-runs
description: List eval runs with clickable clone paths. Understands natural language like "last 10 runs", "yesterday's runs", "softmax runs from last night". Args = natural language query.
argument-hint: "<query, e.g. 'last night', 'last 5', 'softmax runs yesterday'>"
---

# List Eval Runs

Show eval run paths from the SQLite database at `/localdev/$USER/eval_runs.db`.

## Tool

```
python3 -m eval.list_runs [OPTIONS]
```

Options:
- `--last N` — show last N runs
- `--date DATE` — filter by date: `today`, `yesterday`, or `YYYY_MM_DD`
- `--prompt NAME` — filter by prompt name (substring)
- `--branch NAME` — filter by starting branch (substring)
- `--session HHMM` — filter by session time
- `--grade LETTER` — filter by score grade (A/B/C/D/F)
- `--paths-only` — print only clone paths (one per line)
- `--json` — output as JSON

## Your Job

Translate the user's natural language query into the right flags, run the command, and show results.

### Translation Examples

| User says | Command |
|-----------|---------|
| "last 10 runs" | `--last 10` |
| "last night" / "yesterday" / "runs from yesterday" | `--date yesterday` |
| "today's runs" | `--date today` |
| "the 8 runs from last night" | `--date yesterday` (shows all, grouped by session) |
| "softmax runs" | `--prompt softmax` |
| "softmax from yesterday" | `--date yesterday --prompt softmax` |
| "NewOrchestrator runs" | `--branch NewOrchestrator` |
| "session 1701" | `--session 1701` |
| "just the paths" / "paths only" | add `--paths-only` |
| "all runs" | `--last 999` |
| "A-grade runs" | `--grade A` |
| (no args / empty) | (default: last 10) |

### Date handling

- "last night", "yesterday", "last evening" → `--date yesterday`
- "today", "this morning" → `--date today`
- "March 9", "3/9", "03_09" → `--date 2026_03_09`
- "two days ago" → compute the date, use `--date YYYY_MM_DD`

## Execution

1. Parse the user's query into flags
2. Run: `source python_env/bin/activate && python3 -m eval.list_runs <flags>`
3. **REQUIRED**: After the command runs, you MUST re-print every clone path as a clickable link in your response text. Format each path on its own line so the user can click it directly in the terminal. Example output:

```
Session: 2026_03_11_1701 (mstaletovic/AgentsTestingInfra) — 4 runs

  [18] rms_norm (run 1) — Grade: B, Golden: 183/204
       /localdev/mstaletovic/2026_03_11/1701_mstaletovic_AgentsTestingInfra/clones/rms_norm_run1/tt-metal

  [17] layer_norm_rm (run 1) — Grade: B
       /localdev/mstaletovic/2026_03_11/1701_mstaletovic_AgentsTestingInfra/clones/layer_norm_rm_run1/tt-metal
```

The paths MUST appear as plain text in your message (not inside code blocks), so the terminal renders them as clickable file links.

4. If the user asked for "paths only" or similar, add `--paths-only` and just list the bare paths.

Do NOT use agents. Just run the bash command directly.
