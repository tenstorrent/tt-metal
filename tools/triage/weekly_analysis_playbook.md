# Weekly tt-triage CI Failure Analysis Playbook

This document contains end-to-end instructions for a Claude instance to perform the weekly triage analysis. Follow each step in order.

## Prerequisites

### 1. Superset CSV

Go to the TT Superset instance and run the "Hang Jobs" query for the target week:
- Filter: `test_start_ts >= '{MONDAY_DATE}'` AND `test_start_ts < '{NEXT_MONDAY_DATE}'`
- Export as CSV to `triage-runs.csv` in the tt-metal repo root
- Required columns: `test_start_ts`, `test_function`, `host_name`, `job_name`, `github_job_link`, `error_message`
- The `github_job_link` column contains HTML anchor tags with GitHub Actions job URLs

### 2. GitHub CLI

`gh auth status` must succeed with access to `tenstorrent/tt-metal`. The token needs `repo` scope. If `gh` is not installed:
```bash
sudo apt install gh
gh auth login
```

### 3. Python Environment

Python 3.10+. Only standard library modules are used (`csv`, `json`, `re`, `subprocess`, `pathlib`, `time`). No pip installs needed.

### 4. Working Directory

All commands run from the tt-metal repo root. The scripts use `Path(__file__).resolve().parents[2]` to find the repo root.

---

## Step 1: Extract Triage Logs from CI

**Script**: `tools/triage/ci_log_extractor.py`

```bash
python3 tools/triage/ci_log_extractor.py
```

**What it does**:
1. Parses `triage-runs.csv`, deduplicates to unique job URLs
2. For each job, downloads the raw CI log via `gh api repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs`
3. Extracts the triage section (from `dump_configuration.py:` to CI cleanup markers)
4. Strips ANSI escape codes and GitHub Actions timestamps
5. Saves to `triage_outputs/{job_id}.txt`
6. Writes `triage_outputs/index.json` with metadata per job

**Resumable**: Already-downloaded jobs are skipped. Safe to re-run after interruption.

**Rate limiting**: 0.5s between API calls. ~5 minutes per 100 jobs.

**Expected output**: ~200-400 `.txt` files in `triage_outputs/`, plus `index.json`.

### Post-extraction check

```python
python3 -c "
import json
from pathlib import Path
d = Path('triage_outputs')
index = json.load(open(d / 'index.json'))
has = sum(1 for v in index.values() if v['has_output'])
no = sum(1 for v in index.values() if not v['has_output'])
print(f'Total: {len(index)}, With triage: {has}, Without: {no}')
"
```

### Post-extraction trim

The extraction may capture a few trailing CI lines (docker cleanup, artifact upload). Trim them:

```python
python3 -c "
from pathlib import Path
d = Path('triage_outputs')
end_patterns = ['##[group]Run', '##[endgroup]', '[UPLOAD-ARTIFACT-UUID]']
trimmed = 0
for f in d.glob('*.txt'):
    if f.stem == 'index': continue
    lines = f.read_text().split('\n')
    cut = len(lines)
    for i, line in enumerate(lines):
        for pat in end_patterns:
            if pat in line:
                cut = i; break
        if cut < len(lines): break
    if cut < len(lines):
        f.write_text('\n'.join(lines[:cut]))
        trimmed += 1
print(f'Trimmed {trimmed} files')
"
```

---

## Step 2: Split Multi-Invocation Files

**Script**: `tools/triage/split_triage_runs.py`

```bash
python3 tools/triage/split_triage_runs.py
```

**What it does**:
- Splits each triage output at `dump_configuration.py:` boundaries
- A single CI job can trigger triage 1-96 times (test hangs, retries, each re-triggers triage)
- Saves to `triage_outputs_split/{job_id}_run{N}.txt`
- Writes `triage_outputs_split/index.json` with `run_number`, `total_runs`, `original_job_id`

**Why this matters**: First runs are on fresh device state. Subsequent runs operate on devices corrupted by previous hangs AND previous triage interventions. Analyzing them together inflates error counts by up to 29x.

### Post-split check

```python
python3 -c "
import json
index = json.load(open('triage_outputs_split/index.json'))
first = sum(1 for v in index.values() if v.get('run_number') == 1)
nth = sum(1 for v in index.values() if v.get('run_number', 1) > 1)
print(f'Total: {len(index)}, First runs: {first}, Subsequent: {nth}')
"
```

---

## Step 3: Baseline Regex Scan (Optional)

**Script**: `tools/triage/build_split_analysis_csv.py`

```bash
python3 tools/triage/build_split_analysis_csv.py
```

Quick baseline numbers using regex pattern matching. Produces `triage_analysis_results.csv`. This is a sanity check — the agent analysis in Step 4 is more thorough and produces richer data.

---

## Step 4: Dispatch Analysis Agents

This is the core analysis step. Agents read the extracted triage files and produce structured JSON findings.

### 4.1 Prepare batches

```python
import json
from pathlib import Path

d = Path('triage_outputs_split')
index = json.load(open(d / 'index.json'))

# Separate cohorts
cohort_a = {k: v for k, v in index.items() if v.get('run_number') == 1}
cohort_b = {k: v for k, v in index.items() if v.get('run_number', 1) > 1}

# Batch into groups of 15
BATCH_SIZE = 15
def make_batches(cohort):
    keys = sorted(cohort.keys())
    return [keys[i:i+BATCH_SIZE] for i in range(0, len(keys), BATCH_SIZE)]

batches_a = make_batches(cohort_a)
batches_b = make_batches(cohort_b)
```

### 4.2 Agent prompt construction

For each batch, construct the agent prompt by combining:

1. **The full contents of `tools/triage/agent_analysis_instructions.md`** (Document 2)
2. **The file list**: "Your batch of files (in `triage_outputs_split/`): {comma-separated file keys}"
3. **Cohort tag**: "These are Cohort A (first runs)" or "Cohort B (subsequent runs)"
4. **Metadata context**: for each file key, include `test_function`, `host_name`, `arch` from the index

### 4.3 Dispatch

**Rate limit warning**: Launching too many agents in parallel will hit the Anthropic API rate limit (2M input tokens/minute). In our first run, 19 parallel agents caused 5 to fail with 429 errors. **Recommended approach:**

- **Wave 1**: Launch 10 agents in parallel (not all 19)
- **Wait** for Wave 1 to complete (~5-10 minutes)
- **Wave 2**: Launch the remaining agents
- **Retry** any that hit 429 errors, 2-3 at a time with spacing between waves

If you have ~20 batches, plan for 2-3 waves with a few minutes between each. The rate limit resets per minute, so spacing waves by 2-3 minutes is usually enough.

Use `run_in_background=true` for all agents.

**For Cohort A** (priority — these are the accurate numbers):
```
For each batch in batches_a:
    Agent(
        description=f"Analyze triage Cohort A batch {i}",
        prompt=f"""
{contents of agent_analysis_instructions.md}

---

You are analyzing **Cohort A (first runs on fresh device)**. These represent accurate diagnostic data.

Your batch of files (in triage_outputs_split/):
{', '.join(batch)}

File metadata:
{json.dumps({k: {
    'test_function': index[k]['test_function'],
    'host_name': index[k]['host_name'],
    'run_number': index[k]['run_number'],
    'total_runs': index[k]['total_runs']
} for k in batch}, indent=2)}

Return your analysis as a JSON array per the schema in Section 8.
""",
        run_in_background=True
    )
```

**For Cohort B** (secondary — quantifies degradation):
Same pattern but with `cohort_b` batches and "Cohort B (subsequent runs on contaminated device)" tag.

**Sampling rule**:
- **If total Cohort B runs < 200**: analyze ALL of them. Sampling a small cohort just throws away signal — the runs are cheap to process and the per-script statistics become unreliable below a few hundred observations.
- **If total Cohort B runs ≥ 200**: sample per `COHORT_B_SAMPLING`. Default is `first_and_last` — take run 1 and the final run of each job. Alternatively, restrict to jobs that had >5 triage invocations to focus on the most degraded states.

### 4.4 Collect agent results

As agents complete, collect their JSON outputs. Each agent returns a JSON array of per-file findings. Merge all arrays into:
- `cohort_a_findings.json` — all Cohort A agent results merged
- `cohort_b_findings.json` — all Cohort B agent results merged

---

## Step 5: Consolidation

Build the 5 output files from the merged agent findings. See `tools/triage/report_definitions.md` (Document 3) for exact schemas and build logic.

### 5.0 Load previous week's data (for trends)

Check if previous week's CSVs exist. `PREV_YYYYMMDD` is the Monday 7 days before `YYYYMMDD`.

```python
from pathlib import Path
prev_files = {
    'script_reliability': Path(f'triage_script_reliability_{PREV_YYYYMMDD}.csv'),
    'error_patterns': Path(f'triage_error_patterns_{PREV_YYYYMMDD}.csv'),
    'per_test': Path(f'triage_per_test_breakdown_{PREV_YYYYMMDD}.csv'),
    'new_errors': Path(f'triage_new_errors_{PREV_YYYYMMDD}.csv'),
}
has_prev_week = all(f.exists() for f in prev_files.values())
```

If `has_prev_week` is True, load these CSVs and use them in Step 5.5 to compute week-over-week deltas per the rules in `report_definitions.md` Section 7.

### 5.1 Build Script Reliability CSV

```
For each cohort (A, B):
  For each script_name across all findings:
    Count PASS, EXPECTED, UNEXPECTED, ABSENT
    Compute percentages
    Find top 3 unexpected pattern IDs
  → Write to triage_script_reliability_{YYYYMMDD}.csv
```

### 5.2 Build Error Patterns CSV

```
For each cohort:
  For each pattern_id in known_patterns_found across all findings:
    Count unique job_ids, sum occurrences
    Compute averages
  → Write to triage_error_patterns_{YYYYMMDD}.csv
```

### 5.3 Build Per-Test Breakdown CSV

```
For each cohort:
  For each (test_function, script_name) pair:
    Count PASS, EXPECTED, UNEXPECTED
    Find dominant pattern
  → Write to triage_per_test_breakdown_{YYYYMMDD}.csv
```

### 5.4 Build New Errors CSV

```
Collect all new_errors from all findings
Group by similar suggested_name
Count unique jobs per group
→ Write to triage_new_errors_{YYYYMMDD}.csv
```

### 5.5 Build Weekly Report

Follow the template in `tools/triage/report_definitions.md` Section 7. Fill in:
- Summary stats from the index
- Tables from the CSVs above
- Narrative analysis of key findings
- Recommendations

#### 5.5.0 Why Triage Did Not Complete

Immediately after the Executive Summary, include a `### Why Triage Did Not Complete (Cohort A)` table when any file has `init_failure=true` **or** `triage_outcome ∈ {ABORTED, FAILED_TO_START}`. Columns:

| Column | Source |
|--------|--------|
| `file_key` | agent JSON `file_key` |
| `test_function` | agent JSON `test_function` (or split index fallback) |
| `Outcome` | `FAILED_TO_START` if init_failure, else `triage_outcome` |
| `Last script run` | last script in execution order whose status is not ABSENT |
| `Reason` | mapped from `known_patterns_found` — E14 (CI timeout), E15 (no triage section), E19–E22 (init failures), E23 (mid-script crash), E24 (output truncated) |

Omit the section entirely when every file COMPLETED. This answers the "why don't all triage runs complete?" question before the reader reaches the per-script tables — so a high ABORTED rate is explained up front, not inferred from missing data in later tables.

#### 5.5.1 Week-over-week comparison (weekly report)

When `triage_script_reliability_{PREV_YYYYMMDD}.csv` **and** `triage_error_patterns_{PREV_YYYYMMDD}.csv` both exist, the weekly report **must** include a `### Week-over-Week Trends (Cohort A)` section with two sub-tables:

1. **Script UNEXPECTED% Trends** — one row per script. Columns: Script | Last Week % | This Week % | Δ pp | Status.
2. **Error Pattern Trends (jobs%)** — one row per pattern (E01–E24 union of both weeks). Columns: Pattern | Last Week % | This Week % | Δ pp | Status.

Use the trend tags defined in `report_definitions.md` §7 (REGRESSION, IMPROVEMENT, STABLE, NEW, CLEARED). Apply the same thresholds as the drill-down (§5.6.1): |Δ pp| > 5 flips to REGRESSION/IMPROVEMENT, 0→>0 is NEW, >0→0 is CLEARED.

Only compare Cohort A. Cohort B is sampled and its denominators shift week-to-week; Cohort B WoW would mislead.

Omit the WoW section entirely when either previous-week CSV is missing (first-ever run).

→ Write to `triage_weekly_report_{YYYYMMDD}.md`

### 5.6 Build Script Reliability Drill-Down

```bash
python3 tools/triage/drilldown_script_reliability.py
```

Produces a per-script breakdown of **specific error types** — not just PASS/EXPECTED/UNEXPECTED totals, but what exact error caused each UNEXPECTED outcome (e.g., "FD exhaustion — Errno 24 (E01)" vs "Missing fabric ERISC router ELF (E03)").

**Cohort A and Cohort B must be reported separately in the drill-down.** Keep them in distinct sections — never merge their counts. Reasons:

1. **Different denominators.** Cohort A's denominator is "first runs on fresh devices"; Cohort B's is "later runs on devices already contaminated by previous triage/hangs." Combining them produces a weighted average that is neither an accurate measure of triage's baseline quality nor of its degradation-under-contamination behavior.
2. **Different purpose.** Cohort A drives recommendations and WoW trends. Cohort B quantifies how much worse triage gets after repeated invocation — useful for deciding which failure modes cascade. Mixing them hides both signals.
3. **Sampling parity.** Cohort B is frequently sampled (see Step 4.3); its per-script percentages aren't comparable to Cohort A's unless they are labelled as a separate, sampled population.

Output structure (both sections use the same format — summary line, non-PASS reason table):

```
# tt-triage Script Reliability Drill-Down

## Cohort A (First Runs) — {N_A} jobs analyzed
### `check_arc.py`
  Summary: ... PASS: ... EXPECTED: ... UNEXPECTED: ...
  | Status | Reason | Count | % of runs |
  ...

## Cohort B (Subsequent Runs, sampled) — {N_B} runs analyzed
### `check_arc.py`
  Summary: ... PASS: ... EXPECTED: ... UNEXPECTED: ...
  | Status | Reason | Count | % of runs |
  ...
```

The CSV gains a leading `cohort` column (`A` or `B`) so rows from the two populations are easy to filter separately.

#### 5.6.1 Week-over-week comparison (drill-down)

If `triage_script_drilldown_{PREV_YYYYMMDD}.csv` exists, append a `## Week-over-Week — Cohort {A|B}` section to the drill-down report. Emit one section per cohort that existed in both weeks. Do **not** merge cohorts when diffing — Cohort A rows compare to Cohort A rows only.

For each script (only render if it had any present runs this week OR any rows in last week's CSV):

1. **Status-level summary table** — PASS / EXPECTED / UNEXPECTED % last week vs. this week vs. Δ pp.
2. **Reason-level delta table** — union of the non-PASS reason labels seen in either week. For each row, show: status, reason, last-week %, this-week %, Δ pp, trend tag. Sort rows by |Δ pp| descending so the biggest movers appear first.

Trend tags (match the weekly-report convention in §5.5):
- **NEW** — reason appeared at 0% last week, >0% this week.
- **CLEARED** — reason was >0% last week, is 0% this week.
- **REGRESSION** — Δ pp > +5.
- **IMPROVEMENT** — Δ pp < -5.
- **STABLE** — otherwise.

When last week's drilldown CSV is missing (first-ever run or previous week's analysis was skipped), omit the whole WoW section rather than emitting an empty table.

Implementation note: last week's CSV may not have a `cohort` column (added this week). Treat `cohort`-less rows as Cohort A for backward compatibility.

→ Write to `triage_script_drilldown_{YYYYMMDD}.md` + `.csv`

---

## Step 6: Drilling into Specific Failures

After the weekly report is built, you'll commonly want to investigate a single cell — e.g. "which 35 Cohort A files hit `check_noc_status.py` E06?" Two equivalent entry points:

### 6.1 From the drill-down CSV

`triage_script_drilldown_{YYYYMMDD}.csv` includes a `file_keys` column (semicolon-joined) per (cohort, script, status, reason) row. Filter in your spreadsheet / `awk` / `csvkit`:

```bash
awk -F, '$1=="A" && $2=="check_noc_status.py" && $4 ~ /E06/ {print $7}' \
    triage_script_drilldown_{YYYYMMDD}.csv | tr ";" "\n"
```

### 6.2 Using `query_findings.py`

```bash
# List file_keys, test_functions, hosts, arch for a (script, pattern) combo
python3 tools/triage/query_findings.py --script check_noc_status.py --pattern E06

# Add --urls to also print the GitHub Actions job URL for each match
python3 tools/triage/query_findings.py --script dump_callstacks.py --pattern E01 --urls

# Add --grep to print the matching error lines from each file with 2 lines of context
python3 tools/triage/query_findings.py --script check_noc_status.py --pattern E06 --grep --limit 3

# Cohort defaults to A; switch to B or any:
python3 tools/triage/query_findings.py --pattern E04 --cohort any

# By pattern category (triage_bug / environment / diagnostic / informational / init_failure)
python3 tools/triage/query_findings.py --pattern-category triage_bug
```

The script reads `triage_agent_findings/cohort_{a,b}_batch_*.json`, so it works on the current week's data without regenerating anything. Each matched `file_key` maps to `triage_outputs_split/{file_key}.txt`, which is the raw triage section for that run.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WEEK_START` | Previous Monday (YYYY-MM-DD) | Superset query start date |
| `WEEK_END` | Current Monday (YYYY-MM-DD) | Superset query end date |
| `BATCH_SIZE` | 15 | Files per analysis agent |
| `MAX_PARALLEL_AGENTS` | 10 | Max concurrent agents per wave (keep under API rate limit) |
| `COHORT_B_SAMPLING` | `first_and_last` when Cohort B ≥ 200 runs, else `all` | How to sample Cohort B: `all`, `first_and_last`, `first_only`. Skip sampling when total Cohort B runs < 200 (see Step 4.3). |
| `OUTPUT_PREFIX` | `triage` | Prefix for output files |
| `YYYYMMDD` | Monday of analysis week | Date suffix for output files |

---

## Troubleshooting

### Anthropic API rate limits (agent dispatch)
The Anthropic API has a 2M input tokens/minute rate limit. Each analysis agent consumes ~100-170K tokens reading triage files. Launching 19 agents simultaneously will exceed the limit — expect 429 errors on ~25% of agents. **Solution**: dispatch in waves of 10, wait for completion, then dispatch the next wave. Retry failed agents 2-3 at a time.

### GitHub API rate limits (log extraction)
Authenticated requests: 5,000/hour. For 300 jobs at 0.5s spacing, that's ~150 requests in 2.5 minutes. Well within limits. If you hit limits, increase the sleep in `ci_log_extractor.py`.

### Jobs with no triage section
9 of 287 jobs in our initial analysis had `[NO TRIAGE SECTION FOUND]`. These are jobs where triage was configured but never triggered, or the output was captured as a CI artifact only. Count them but exclude from script-level analysis.

### Very large files
Some triage outputs are 15+ MB (multi-device runs with 50+ triage invocations). The split step breaks these into manageable per-run files. If agents hit context limits, reduce `BATCH_SIZE`.

### Missing `dump_configuration.py:` marker
If a file has no marker, triage may have crashed during initialization (exalens failure, UMD failure). These should be flagged as `E15` (No Triage Section) and excluded from per-script analysis.

---

## Reference: Existing Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `ci_log_extractor.py` | `tools/triage/ci_log_extractor.py` | Download CI logs, extract triage sections |
| `split_triage_runs.py` | `tools/triage/split_triage_runs.py` | Split multi-invocation files |
| `build_split_analysis_csv.py` | `tools/triage/build_split_analysis_csv.py` | Regex baseline scan |
| `build_analysis_csv.py` | `tools/triage/build_analysis_csv.py` | Original (non-split) regex scan |
| `consolidate_report.py` | `tools/triage/consolidate_report.py` | Build weekly report + 4 CSVs from agent JSON findings in `triage_agent_findings/` |
| `drilldown_script_reliability.py` | `tools/triage/drilldown_script_reliability.py` | Per-script error type drill-down (includes `file_keys` column for follow-up queries) |
| `query_findings.py` | `tools/triage/query_findings.py` | Filter agent findings by (script, pattern, cohort); optionally grep matching error context |

## Reference: Other Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Agent Analysis Instructions | `tools/triage/agent_analysis_instructions.md` | Detailed instructions for analysis agents (embedded in prompts) |
| Report Definitions | `tools/triage/report_definitions.md` | CSV schemas, report template, consolidation logic |
