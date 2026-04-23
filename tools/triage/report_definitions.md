# Report Definitions & Output Schemas

## 1. Cohort Definitions

All analysis is split into two cohorts:

| Cohort | Filter | Description |
|--------|--------|-------------|
| **A (First Runs)** | `run_number == 1` | Exactly 1 per job. Device was in whatever state the hanging test left it. These provide accurate diagnostic data. |
| **B (Subsequent Runs)** | `run_number > 1` | Device has been through N-1 previous triage cycles. Contaminated by: cores broken by previous triage, leaked FDs, changed device state. |

**Rule**: Primary analysis and recommendations use Cohort A only. Cohort B is analyzed separately to quantify triage-induced degradation.

## 2. Output Files

All output files are written to the repo root, prefixed with the analysis week:

| File | Purpose |
|------|---------|
| `triage_weekly_report_{YYYYMMDD}.md` | Narrative summary for humans |
| `triage_script_reliability_{YYYYMMDD}.csv` | Per-script PASS/EXPECTED/UNEXPECTED breakdown |
| `triage_per_test_breakdown_{YYYYMMDD}.csv` | Per-test x per-script reliability matrix |
| `triage_error_patterns_{YYYYMMDD}.csv` | Known pattern frequency data |
| `triage_new_errors_{YYYYMMDD}.csv` | Newly discovered errors not in catalog |
| `triage_script_drilldown_{YYYYMMDD}.md` | Per-script drill-down: specific error types behind each outcome |
| `triage_script_drilldown_{YYYYMMDD}.csv` | Same drill-down data in CSV format |

`YYYYMMDD` is the Monday of the analysis week.

---

## 3. Analysis Dimension 1: Script Reliability Report

**Purpose**: For each triage script, how often does it PASS vs find legitimate issues vs hit its own bugs?

> **Note**: For a drill-down showing the specific error types behind each script's UNEXPECTED/UNEXPECTED outcomes, see `triage_script_drilldown_{YYYYMMDD}.md/.csv` produced by `tools/triage/drilldown_script_reliability.py`.

### CSV Schema: `triage_script_reliability_{YYYYMMDD}.csv`

| Column | Type | Description |
|--------|------|-------------|
| `cohort` | string | `A` or `B` |
| `script_name` | string | e.g., `check_noc_status.py` |
| `total_runs` | int | Files where this script appeared |
| `pass_count` | int | PASS classifications |
| `pass_pct` | float | `pass_count / total_runs * 100` |
| `expected_count` | int | EXPECTED classifications |
| `expected_pct` | float | percentage |
| `unexpected_count` | int | UNEXPECTED classifications |
| `unexpected_pct` | float | percentage |
| `errored_count` | int | UNEXPECTED classifications (triage died during this script) |
| `errored_pct` | float | percentage |
| `absent_count` | int | Files where script was missing (includes post-crash ABSENT) |
| `absent_post_crash_count` | int | Subset of absent_count where reason was a crash or truncation in an earlier script |
| `top_unexpected_errors` | string | Semicolon-separated top 3 unexpected/crash error pattern IDs (e.g., `E01; E23`) |

### Init failure tracking

In addition to per-script rows, include a synthetic row:

| cohort | script_name | total_runs | ... | unexpected_count | ... |
|--------|-------------|-----------|-----|------------------|-----|
| A | `_triage_init` | {total files} | 0 | {count of init_failure=true} | ... |

This tracks how often triage crashes before producing ANY output. `init_failure_reason` values from agent output are aggregated into `top_unexpected_errors` (e.g., `E19; E20; E21`).

### How to build from agent output

```
For each cohort:
  Count files where init_failure == true -> init_failure_count
  Add row: script_name="_triage_init", unexpected_count=init_failure_count

  For each script_name across all agent JSON outputs (excluding init failures):
    Count files where scripts[script_name].status == "PASS"
    Count files where scripts[script_name].status == "EXPECTED"
    Count files where scripts[script_name].status == "UNEXPECTED"
    Count files where scripts[script_name].status == "UNEXPECTED"
    Count files where scripts[script_name].status == "ABSENT"
    Count ABSENT where absent_reason contains "errored" or "truncated" -> absent_post_crash_count
    Collect pattern_ids from known_patterns_found where script matches
    Rank by frequency -> top_unexpected_errors
```

### Narrative section in weekly report

- Render as markdown table
- **Highlight** scripts with `unexpected_pct > 10%` in Cohort A (these are priority fixes)
- **Highlight** scripts with any `errored_count > 0` — these are the scripts where triage died mid-execution
- **Compare** Cohort A vs B: if a script has much higher unexpected rate in B, that quantifies triage-induced degradation
- **Call out** scripts with high `absent_post_crash_count` — these scripts never ran because an earlier script errored. Distinguish from scripts that are normally conditional (like `dump_risc_debug_signals.py`)
- **Triage completion rate**: `(total_files - init_failures - crash_truncations) / total_files` — what fraction of triage runs complete all scripts

---

## 4. Analysis Dimension 2: Error Pattern Report

**Purpose**: How frequently does each known pattern appear? Grouped by root cause category.

### CSV Schema: `triage_error_patterns_{YYYYMMDD}.csv`

| Column | Type | Description |
|--------|------|-------------|
| `cohort` | string | `A` or `B` |
| `pattern_id` | string | `E01`-`E18` (or new IDs) |
| `pattern_name` | string | Human-readable name |
| `category` | string | `triage_bug`, `environment`, `diagnostic`, `informational` |
| `affected_scripts` | string | Semicolon-separated script names |
| `is_triage_bug` | bool | Whether this is a bug in triage itself |
| `jobs_affected` | int | Unique jobs with this pattern |
| `jobs_pct` | float | `jobs_affected / total_jobs * 100` |
| `total_occurrences` | int | Sum of occurrence counts across all files |
| `avg_per_job` | float | `total_occurrences / jobs_affected` |

### How to build from agent output

```
For each cohort:
  For each pattern_id across all agent JSON known_patterns_found:
    Count unique job_ids
    Sum counts
    Compute averages
```

### Narrative section

Group by category in this order:
1. **Triage bugs** (actionable) — sorted by `jobs_affected` desc
2. **Environment issues** — sorted by `jobs_affected` desc
3. **Diagnostic findings** — sorted by `jobs_affected` desc
4. **Informational** — brief mention

For triage bugs: include root cause and recommended fix.

---

## 5. Analysis Dimension 3: Per-Test Reliability Matrix

**Purpose**: Which tests trigger which triage script failures? Identifies problematic tests vs problematic scripts.

### CSV Schema: `triage_per_test_breakdown_{YYYYMMDD}.csv`

| Column | Type | Description |
|--------|------|-------------|
| `cohort` | string | `A` or `B` |
| `test_function` | string | e.g., `test_all_gather_linear_2D_nightly` |
| `arch` | string | `blackhole`, `wormhole_b0`, or `unknown` |
| `total_jobs` | int | Jobs running this test |
| `script_name` | string | Each triage script |
| `pass_count` | int | PASS for this test+script |
| `expected_count` | int | EXPECTED for this test+script |
| `unexpected_count` | int | UNEXPECTED for this test+script |
| `dominant_pattern` | string | Most common known pattern for this test+script (pattern ID) |

### How to build from agent output

```
For each cohort:
  For each unique test_function:
    For each script_name:
      Count PASS/EXPECTED/UNEXPECTED across files for this test
      Find most frequent pattern_id in known_patterns_found for this test+script
```

### Narrative section

- **Top 10 worst tests** by total `unexpected_count` across all scripts
- For each: which scripts fail and with which patterns
- **Pivot insight**: does the test cause the failure, or does the script have a bug that manifests on many tests?
  - If one script fails UNEXPECTED on 50 different tests → script bug
  - If one test causes UNEXPECTED on 10 different scripts → test-specific device state issue

---

## 6. Analysis Dimension 4: New/Unknown Error Report

**Purpose**: Capture errors that don't match any known pattern. Candidates for next week's catalog.

### CSV Schema: `triage_new_errors_{YYYYMMDD}.csv`

| Column | Type | Description |
|--------|------|-------------|
| `cohort` | string | `A` or `B` |
| `script_name` | string | Where the error appeared |
| `error_text` | string | First 500 chars of error |
| `suggested_name` | string | Agent's proposed pattern name |
| `suggested_regex` | string | Agent's proposed regex |
| `jobs_affected` | int | How many jobs showed this |
| `file_keys` | string | Semicolon-separated list of file keys |

### How to build from agent output

```
Collect all new_errors from all agents
Group by similar error_text (fuzzy dedup — same suggested_name)
Count unique jobs per group
Sort by jobs_affected desc
```

### Narrative section

- List each new error with full context
- **Errors seen in 3+ jobs**: Recommend adding to known pattern catalog
  - Propose pattern ID (E19, E20, ...)
  - Provide exact regex to add to `agent_analysis_instructions.md` Section 5
- **Errors seen in 1-2 jobs**: Note for monitoring, don't add to catalog yet

---

## 7. Analysis Dimension 5: Week-over-Week Comparison

**Purpose**: Track trends across weeks. Detect regressions (triage getting worse), improvements (fixes landing), and persistent new errors.

### Prerequisites

The consolidation step checks for the previous week's output CSVs:
- `triage_script_reliability_{PREV_YYYYMMDD}.csv`
- `triage_error_patterns_{PREV_YYYYMMDD}.csv`
- `triage_per_test_breakdown_{PREV_YYYYMMDD}.csv`
- `triage_new_errors_{PREV_YYYYMMDD}.csv`

Where `PREV_YYYYMMDD` is the Monday 7 days before the current analysis week. If these files don't exist (first week of analysis), skip all week-over-week sections.

### 7.1 Script Reliability Trends

For each script in Cohort A, compare `unexpected_pct` between this week and last week:

```
For each script_name:
  delta_pp = this_week.unexpected_pct - last_week.unexpected_pct
  If delta_pp > 5:  → REGRESSION (flag red)
  If delta_pp < -5: → IMPROVEMENT (flag green)
  Else:             → STABLE
```

### 7.2 Error Pattern Trends

For each known pattern (E01-E18+) in Cohort A, compare `jobs_pct`:

```
For each pattern_id:
  delta_pp = this_week.jobs_pct - last_week.jobs_pct
  If delta_pp > 5:  → REGRESSION
  If delta_pp < -5: → IMPROVEMENT
  Else:             → STABLE
```

Also track:
- **Patterns that dropped to 0%**: Something was fixed — note what changed
- **Patterns that appeared from 0%**: New issue introduced this week

### 7.3 New Error Persistence

Cross-reference this week's `new_errors` against last week's `new_errors`:

| Case | Action |
|------|--------|
| New error appeared last week AND this week (3+ jobs both times) | **Promote to known catalog**: assign next pattern ID (E19, E20...), add regex to `agent_analysis_instructions.md` Section 5 |
| New error appeared last week but NOT this week | **Transient**: remove from monitoring, likely a one-off CI issue |
| New error appeared this week for the first time | **Monitor**: keep in new_errors, check again next week |

### 7.4 Per-Test Trends

For the top 10 worst tests from each week, compare:
- Tests that improved (lower unexpected count)
- Tests that regressed (higher unexpected count)
- Tests that are new to the top 10 this week (newly problematic)
- Tests that dropped off the top 10 (fixed or no longer running)

### 7.5 High-Level Metrics Trend

Track these week-over-week:

| Metric | How to compute |
|--------|----------------|
| Total hang jobs | Count of unique jobs in CSV |
| Multi-run job % | Jobs with `total_runs > 1` / total jobs |
| Triage completion rate | Files where all expected scripts ran (no UNEXPECTED, no suspicious ABSENT) / total files |
| Init failure rate | Files with `init_failure=true` / total files |
| Overall unexpected rate | Across all scripts in Cohort A, total (UNEXPECTED + UNEXPECTED) / total classifications |
| Triage bug hit rate | Jobs hitting any `is_triage_bug=true` pattern / total jobs |
| Clean run rate | Jobs where NO script had UNEXPECTED or UNEXPECTED / total jobs |

---

## 8. Weekly Report Template

```markdown
# tt-triage Weekly Analysis Report
## Week of {WEEK_START} to {WEEK_END}

### Executive Summary
- Total hang jobs: {N_JOBS}
- Jobs with triage output: {N_WITH_OUTPUT}
- Jobs with multiple triage runs: {N_MULTI} ({N_MULTI/N_JOBS*100:.0f}%)
- Total triage invocations: {TOTAL_RUNS} (Cohort A: {N_A}, Cohort B: {N_B})
- Architecture breakdown: {N_BH} Blackhole, {N_WH} Wormhole

### Key Findings
1. {Top finding — usually the most impactful triage bug}
2. {Second finding}
3. {Third finding}

---

### Triage Init Failures (Cohort A)

{Count and percentage of first runs where triage errored before producing any script output}

| Failure Reason | Count | % | Pattern |
|----------------|-------|---|---------|
{Breakdown by init_failure_reason, mapped to E19-E22}

{If init failures > 5%: flag as critical — these jobs get ZERO diagnostic data}

---

### Script Reliability — Cohort A (First Runs on Fresh Device)

{Markdown table from triage_script_reliability.csv, cohort=A}

**Scripts with >10% unexpected failure rate:**
- {script}: {unexpected_pct}% — {top_unexpected_errors}

### Script Reliability — Cohort A vs B Comparison

{Table showing unexpected_pct for A and B side by side}

**Biggest degradation in subsequent runs:**
- {script}: {A_pct}% → {B_pct}% (+{delta}pp)

---

### Error Patterns

#### Triage Bugs (Actionable)

| Pattern | Jobs Affected | % | Avg/Job | Root Cause |
|---------|--------------|---|---------|------------|
{rows from triage_error_patterns.csv where category=triage_bug, cohort=A}

#### Environment Issues

| Pattern | Jobs Affected | % | Note |
|---------|--------------|---|------|
{rows where category=environment, cohort=A}

#### Diagnostic Findings (Triage Working Correctly)

| Pattern | Jobs Affected | % | What It Found |
|---------|--------------|---|---------------|
{rows where category=diagnostic, cohort=A}

---

### Per-Test Analysis (Cohort A)

**Top 10 tests by unexpected failure count:**

| Test | Total Jobs | Worst Script | Unexpected Count | Dominant Pattern |
|------|-----------|-------------|-----------------|-----------------|
{top 10 from triage_per_test_breakdown.csv}

---

### New Errors Discovered

{For each new error seen in 3+ jobs:}
#### {Suggested Name}
- **Script**: {script_name}
- **Jobs affected**: {count}
- **Error text**: `{abbreviated error text}`
- **Recommended action**: Add to catalog as E{next_id}

---

### Week-over-Week Trends
{Only include this section if previous week's CSVs exist}

#### High-Level Metrics

| Metric | Last Week | This Week | Delta |
|--------|-----------|-----------|-------|
| Total hang jobs | {prev_N_JOBS} | {N_JOBS} | {delta} |
| Multi-run job % | {prev_%} | {this_%} | {delta}pp |
| Overall unexpected rate | {prev_%} | {this_%} | {delta}pp |
| Triage bug hit rate | {prev_%} | {this_%} | {delta}pp |
| Clean run rate | {prev_%} | {this_%} | {delta}pp |

#### Error Pattern Trends (Cohort A)

| Pattern | Last Week | This Week | Delta | Status |
|---------|-----------|-----------|-------|--------|
{For each pattern, show jobs_pct last vs this, flag REGRESSION/IMPROVEMENT/STABLE}

#### Regressions (>5pp increase in unexpected rate)
{List scripts or patterns that got significantly worse}

#### Improvements (>5pp decrease)
{List scripts or patterns that got better — call out if a fix was deployed}

#### Persistent New Errors (seen last week AND this week)
{List with recommendation to promote to known catalog with assigned pattern IDs}

---

### Recommendations (Priority Order)
1. {Highest priority — usually fixing the most frequent triage bug}
2. {Second priority}
3. {Third priority}

---

### Appendix: Data Files
- [Script Reliability](triage_script_reliability_{YYYYMMDD}.csv)
- [Error Patterns](triage_error_patterns_{YYYYMMDD}.csv)
- [Per-Test Breakdown](triage_per_test_breakdown_{YYYYMMDD}.csv)
- [New Errors](triage_new_errors_{YYYYMMDD}.csv)
```

---

## 8. Consolidation Logic Summary

The consolidation step receives JSON arrays from all analysis agents and the split index metadata. It produces all 5 output files by:

1. **Merge** all agent JSON arrays into one list
2. **Split** by cohort: `run_number == 1` → A, else → B
3. **Aggregate** per the "How to build" instructions in each section above
4. **Render** markdown report using the template in Section 7
5. **Write** all 5 files with the `{YYYYMMDD}` suffix
