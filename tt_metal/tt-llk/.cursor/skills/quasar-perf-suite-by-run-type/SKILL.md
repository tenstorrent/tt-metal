---
name: quasar-perf-suite-by-run-type
description: >-
  Run the Quasar LLK perf suite one PerfRunType at a time, analyze pytest
  outputs, and rename perf_data CSV reports to append the PerfRunType.
  Use when collecting Quasar perf CSVs per L1_TO_L1 / UNPACK_ISOLATE /
  MATH_ISOLATE / PACK_ISOLATE / L1_CONGESTION, or when the user asks to run
  run_perf_suite_and_report.sh by run type and tag perf_data CSVs.
---

# Quasar Perf Suite By Run Type

## Goal

For each `PerfRunType`, run the suite with that type only, analyze outputs, then rename CSV reports under `tt-llk/perf_data/<perf_test_name>/` so the filename appends `_<PerfRunType>`.

## Paths

| Item | Path |
|------|------|
| Suite script | `tests/python_tests/quasar/run_perf_suite_and_report.sh` |
| Suite cwd | `tests/python_tests/quasar/` |
| Outputs | `perf_output_<PerfRunType>_<suite_id>.txt` |
| Report | `perf_suite_results_<PerfRunType>.md` (single-type) or `perf_suite_results.md` |
| CSV root | `<LLK_ROOT>/perf_data/<perf_test_name>/` |
| Rename helper | `.cursor/skills/quasar-perf-suite-by-run-type/scripts/rename_perf_csvs.sh` |

`LLK_ROOT` = `tt-metal/tt_metal/tt-llk` (absolute path in this workspace).

## PerfRunType order

Run **one type at a time**, in this order:

1. `L1_TO_L1`
2. `UNPACK_ISOLATE`
3. `MATH_ISOLATE`
4. `PACK_ISOLATE`
5. `L1_CONGESTION`

Do not start the next type until rename for the current type finishes (otherwise CSVs would be overwritten).

## Suite test map

| suite_id | test file / perf_data dir |
|----------|---------------------------|
| 02 | `perf_eltwise_unary_datacopy_quasar` |
| 03 | `perf_eltwise_binary_broadcast_quasar` |
| 04 | `perf_eltwise_binary_quasar` |
| 05 | `perf_unpack_tilize_quasar` |
| 06 | `perf_unpack_unary_operand_quasar` |
| 07 | `perf_transpose_dest_quasar` |
| 08 | `perf_pack_quasar` |
| 09 | `perf_pack_untilize_quasar` |
| 10 | `perf_unary_broadcast_quasar` |
| 11 | `perf_pack_l1_acc_quasar` |
| 12 | `perf_reduce_quasar` |
| 13 | `perf_eltwise_binary_reuse_dest_quasar` |
| 14 | `perf_unpack_reduce_col_tilizeA_strided_quasar` |

Matmul is excluded.

## Workflow

Copy and track:

```
Perf suite by run type:
- [ ] L1_TO_L1: run → analyze → rename
- [ ] UNPACK_ISOLATE: run → analyze → rename
- [ ] MATH_ISOLATE: run → analyze → rename
- [ ] PACK_ISOLATE: run → analyze → rename
- [ ] L1_CONGESTION: run → analyze → rename
- [ ] Final summary
```

### Step A — Run suite for one type

```bash
cd "$LLK_ROOT/tests/python_tests/quasar"
bash ./run_perf_suite_and_report.sh --run-type <PerfRunType>
```

- Long-running (hours). Use a high Shell `block_until_ms` / background + monitor.
- Script retries **only** tt-exalens 600s timeout; **does not** retry 5‑min hangs or other failures.
- Expect outputs: `perf_output_<PerfRunType>_02.txt` … `_14.txt`.

### Step B — Analyze outputs

For each `perf_output_<PerfRunType>_<suite_id>.txt`:

1. Record status: `PASSED` / `FAILED` / `HANG` / `EXALENS_TIMEOUT` / `NO_TESTS` (from file + `perf_suite_results_<PerfRunType>.md`).
2. Note first failure / hang nodeid if present.
3. Build the list of **successful** (or CSV-producing) `perf_test_name`s from the suite map.

Still attempt rename when the `perf_data/<name>/` dir has fresh untagged CSVs even if the pytest log is partial — but call out mismatches in the summary.

### Step C — Rename CSVs

Untagged files look like:

- `perf_data/<name>/<name>.csv`
- `perf_data/<name>/<name>.post.csv`

Rename to append `_<PerfRunType>` before the suffix:

- `<name>.csv` → `<name>_<PerfRunType>.csv`
- `<name>.post.csv` → `<name>_<PerfRunType>.post.csv`

Prefer the helper (pass only names that should be tagged for this type):

```bash
bash "$LLK_ROOT/.cursor/skills/quasar-perf-suite-by-run-type/scripts/rename_perf_csvs.sh" \
  --llk-root "$LLK_ROOT" \
  --run-type <PerfRunType> \
  --test-name perf_eltwise_unary_datacopy_quasar \
  --test-name perf_eltwise_binary_quasar
  # ... one --test-name per suite entry that produced CSVs
```

Or rename all untagged `perf_*_quasar` dirs (use only when this type’s suite just finished and no other untagged CSVs are expected):

```bash
bash "$LLK_ROOT/.cursor/skills/quasar-perf-suite-by-run-type/scripts/rename_perf_csvs.sh" \
  --llk-root "$LLK_ROOT" \
  --run-type <PerfRunType>
```

Helper skips files already tagged with any PerfRunType.

### Step D — Next type

Repeat A→C for the next PerfRunType. Do not run multiple types in parallel.

### Step E — Final summary

Report a table:

| PerfRunType | suite PASSED | FAILED/HANG | CSVs renamed |
|-------------|-------------:|------------:|-------------:|
| … | … | … | … |

List any missing `perf_data` dirs or untagged CSVs left behind.

## Rules

- One PerfRunType per suite invocation (`--run-type`).
- Rename **before** starting the next type.
- Do not delete CSVs; only `mv` untagged → tagged.
- Do not retry hangs; only the suite script’s exalens retry applies.
- Keep matmul out of this flow unless the user explicitly asks.
