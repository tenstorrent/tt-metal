# Perf infra & scripts changes

Scope: `ndivnic/add_remaining_perf_tests_quasar` (vs `origin/main`) for Quasar **perf infrastructure and orchestration scripts**, plus local **tt-umd-simulators** launcher script changes used by those runners.

Excluded: individual `perf_*.py` / `test_*.py` test bodies, CSV/log artifacts, and suite result dumps (except where they are produced by the scripts below).

Status legend:

- **Committed** — on the branch tip
- **WIP** — present in the working tree, not committed yet
- **Local (umd)** — uncommitted changes in `tt-umd-simulators` (on `main`, not a named perf branch)

---

## 1. Suite orchestration scripts (`tt-llk/.../quasar/`)

### Committed

| File | What changed / why |
|------|--------------------|
| `run_perf_suite_and_report.sh` | Main per-`PerfRunType` suite runner. Sets Quasar emu env, installs the instrumented 1x3 launcher into the build tree, runs each suite file with `--speed-of-light` and `-k PerfRunType.<name>`, retries tt-exalens bring-up timeouts (~610s, up to 5×), detects single-test hangs after exalens is ready (~300s), writes `perf_suite_results_*.md`, supports `--run-type` / `--suite-ids`. |
| `run_perf_suite_orchestrator.sh` | Sequences run types (`UNPACK_ISOLATE` → `MATH_ISOLATE` → `PACK_ISOLATE` → `L1_CONGESTION` → `L1_TO_L1` last), invoking the report script then `rename_perf_csvs.sh` so CSVs are not overwritten by the next type. Resumable via `RUN_ORDER`. |
| `run_perf_suite_rerun_incomplete.sh` | After orchestrator `DONE_ALL_RUN_TYPES`, re-runs suite entries that never finished / had no usable result; skips `PASSED` and `HANG`. |
| `run_perf_tests.sh` | Earlier/simpler long-run helper: can patch tests to L1-only, watches for exalens fail / stall hang, retries. Largely superseded by the suite+report / all-modes runners but still on the branch. |
| `preserve_exalens_logs.sh` | Archives `tt-exalens.log` / `emu_*.log` before the next pytest (exalens opens the log in `'w'`). Optionally installs `quasar-1x3_run_dev.instrumented.sh` into `$TT_METAL_SIMULATOR`. |

### WIP (uncommitted)

| File | What changed / why |
|------|--------------------|
| `run_perf_suite_all_modes.sh` | All-modes runner: one pytest per suite file with **all** `PerfRunType`s in one parametrization (no `-k`, no pytest `--timeout`). Hang-monitors open tests / output stalls (~600s), kills local+Zebu, records a hang-skip, retries. Writes `perf_output_all_<id>.txt` and updates `perf_suite_progress_summary.md`. Installs the instrumented emu launcher like the report script. |
| `watch_perf_suite_hangs.sh` | External watchdog for the all-modes orchestrator (focused on suites 12–14). Detects hang or dead orch, derives `PerfRunType` from newest `build.h`, calls hang-skip API, restarts from incomplete suite; emits `AGENT_LOOP_TICK_perfhang` lines for Cursor notify. |
| `run_perf_retry_02_03_after_main.sh` | Waits for `DONE_ALL_MODES` (or RESULT 14 + orch exit), then retries suites 02/03 with existing hang-skips and refreshes the progress summary. |
| `run_perf_suite_and_report.sh` (tiny edit) | Report notes now document `-k PerfRunType.<name>` (bare `PACK_ISOLATE` also matches `UNPACK_ISOLATE`). |

---

## 2. Agent / Cursor helper scripts

### Committed

| File | What changed / why |
|------|--------------------|
| `.cursor/scripts/run_test.sh` | Multi-file `--test` support; trailing bare test paths; `--speed-of-light` flag threaded into compile/simulate/run; safer arg quoting / labels for multi-test runs. |
| `.claude/scripts/run_test.sh` | Same `--speed-of-light` plumbing + timestamped test-id logging on simulate/run. |
| `.cursor/skills/quasar-perf-test/SKILL.md` and `.claude/skills/quasar-perf-test/SKILL.md` | Unified guidance for creating perf tests and implementing or debugging all Quasar `PerfRunType` kernel paths. |
| `.cursor/skills/quasar-perf-suite-by-run-type/SKILL.md` and `.claude/skills/quasar-perf-suite-by-run-type/SKILL.md` | Documents the by-run-type workflow (order, suite map, rename step). |
| Each `quasar-perf-suite-by-run-type/scripts/rename_perf_csvs.sh` | Renames combined CSVs under `perf_data/<test>/` to append `_<PerfRunType>` before `.csv` / `.post.csv`, so per-type suite passes do not clobber each other. |

---

## 3. Python perf infrastructure

### Committed

| File | What changed / why |
|------|--------------------|
| `helpers/perf.py` | Perf report schema hardening: `PerfSchemaError`, refuse mixing incompatible column schemas in one CSV; `_collapse_duplicate_keys` when combine sees duplicate (sweep, marker) keys; `_assert_combined_schema` across worker files; `PerfConfig` improvements for multi-run-type / SOL / skip-report paths (shared functional+perf entrypoints). |
| `helpers/exalens_server.py` | Treat emu log line containing `FATAL: SSH` as early bring-up failure so pytest does not wait the full ready timeout when the Aether host is unreachable. |
| `helpers/llk_params.py` | Adds `PERF_RUN_TYPES_QUASAR` — committed form is **one `PerfRunType` per pytest case** (five nested singleton lists) for independent isolate reporting. |
| `conftest.py` | More robust runtime-only variant collapse for compile-producer (`_make_hashable`, cleaner compile keys); quasar collect ignore kept/adjusted for arch filtering. |
| `tests/helpers/include/perf.h` | Quasar enabled on the BH-style single-source-bank matmul mock path (`ARCH_QUASAR`); removes “fixme: add quasar support” on math mock. Needed for isolate/SOL matmul-style perf kernels. |
| `compare_test_and_perf.py` | Offline diagnostic: compare `@parametrize` sweeps of `test_*` vs `perf_*` modules (no simulator). |
| `quasar/debug_perf_isolate.py` | Focused repro that runs each `PerfRunType` as its own pytest case to surface isolate deadlocks outside a multi-mode sweep. |

### WIP (uncommitted)

| File | What changed / why |
|------|--------------------|
| `helpers/llk_params.py` | `PERF_RUN_TYPES_QUASAR` switched to **one nested list of all modes** (`L1_TO_L1`+isolates; `L1_CONGESTION` dropped) so a single pytest case fills one homogeneous CSV schema for the all-modes runner. |
| `helpers/param_config.py` | Node-id formatting for enum lists: `run_types:PerfRunType.A+PerfRunType.B` so `-k PerfRunType.PACK_ISOLATE` does not substring-match `UNPACK_ISOLATE`. |
| `helpers/perf.py` | Schema-guard message improved: if schemas differ only by `mean(`/`std(`/`TEXT_SIZE(` columns, hint that the test is emitting different PerfRunType metric sets across cases (points authors at the all-modes / homogeneous-CSV pattern). |
| `helpers/perf_hang_skips.py` + `quasar/perf_hang_skips.json` | Persistent registry of `{test_module: [PerfRunType, ...]}` hang skips; `add_skip` / `filter_run_types` used by all-modes runner and perf tests on retry. |

---

## 4. Design shift: by-run-type → all-modes

Committed scripts optimize for **one PerfRunType per pytest invocation** (`-k`, rename CSVs, orchestrator order).

WIP scripts/infra optimize for **all modes in one case**:

1. Homogeneous CSV columns (`mean`/`TEXT_SIZE` for every mode on every row).
2. Hang on a mode → record skip → retry same file without that mode.
3. Drop `L1_CONGESTION` from the default Quasar list (known hangy).

Both approaches remain on disk; all-modes is the newer operational path used for `perf_output_all_*.txt`.

---

## 5. tt-umd-simulators scripts (local / uncommitted)

Repo: `/proj_sw/user_dev/ndivnic/tt-umd-simulators` on `main`. These are what the Quasar suite runners copy/install into `build/emu-quasar-1x3/`.

| File | Status | What changed / why |
|------|--------|--------------------|
| `emu/quasar-1x3/quasar-1x3_run_dev.sh` | Modified | `EMULATOR_TIMEOUT` 180 → **36000** (long Zebu runs). SSH host discovery over `soc-l-01`…`12` (prefer `SSH_MACHINE_NAME`, else first reachable). Stricter `SSH_OPTS` (BatchMode, ConnectTimeout, ServerAlive*). Distinguish SSH connection failure (rc 255) from “workspace missing” so a dead host does not trigger a bogus Aether clone path. Emit `FATAL: SSH ...` for exalens early-abort. |
| `emu/quasar-1x3/quasar-1x3_run_dev.instrumented.sh` | New (untracked) | Fail-fast / instrumented launcher: phase timestamps, discovery vs per-host retries, buffered `ssh_script` for retries, same FATAL marker. Installed into the build tree by `preserve_exalens_logs.sh` / suite runners as `quasar-1x3_run_dev.sh`. |
| `emu/quasar-2x3/quasar-2x3_run.sh` | Modified | Timeout 180 → 36000 only. |
| `emu/quasar-2x3_DISPATCH/quasar-2x3_DISPATCH_run.sh` | Modified | Timeout 180 → 36000 only. |
| `emu/quasar-9x4_DM/quasar-9x4_DM_run.sh` | Modified | Timeout 180 → 36000 only. |

---

## 6. How the pieces connect

```text
tt-umd-simulators
  quasar-1x3_run_dev[.instrumented].sh  --SSH-->  soc-l-* (Aether/Zebu)
        ^
        | cp/install
quasar suite runners
  run_perf_suite_and_report.sh   (per PerfRunType + -k)
  run_perf_suite_orchestrator.sh -> rename_perf_csvs.sh
  run_perf_suite_all_modes.sh    (all modes + hang skips)  [WIP]
        |
        v
helpers/exalens_server.py  (FATAL: SSH early fail)
helpers/perf.py            (CSV schema / combine / collapse)
helpers/perf_hang_skips.py [WIP]  <->  perf_hang_skips.json
```

---

## 7. Intentionally omitted from this list

- New/updated Quasar `perf_*.py` tests and functional test edits (product tests, not infra).
- Generated outputs: `perf_output*.txt`, `exalens_log_archive/`, `perf_suite_results_*.md(.rows)`, progress/summary noise from runs.
- Docs that only describe matmul sweep strategy without being executable infra.
