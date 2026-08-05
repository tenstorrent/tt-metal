# Quasar performance commit extraction plan

## Scope

This document analyzes commit `47a66006894` (`Consolidate Quasar performance
infrastructure`) relative to its `origin/main` parent, `014e247ddd9`.

The commit contains 22 changed files and combines unrelated concerns:

- Quasar LLK correctness fixes
- pytest and performance-harness reliability fixes
- performance metadata and run-type selection
- full-suite shell automation
- operational recovery tools
- documentation and agent guidance

The changes should not be proposed as one PR. Kernel correctness fixes deserve
the fastest and most focused review, while workflow conveniences can be merged
later or omitted.

## Priority summary

### Must extract

1. Quasar tilize MOP correctness for `block_ct_dim == 1`
2. Quasar reuse-destination dvalid correctness
3. Compile-producer key stability and counter-report teardown fix
4. Parallel compile-directory race prevention

### Meaningful, but workflow-dependent

5. `PerfConfig` boot-mode passthrough
6. Register-format metadata in performance CSVs
7. Precise `PerfRunType` selection and the core suite runner
8. Exalens SSH fail-fast handling

### Nice to have

9. Full-suite orchestration, rerun recovery, and log archiving
10. Functional/performance sweep comparison tool
11. Runbook and mirrored agent skills
12. Test-runner timestamp logging and small cleanups

## Recommended PR stack

### PR 1: Fix Quasar unpack tilize for one-tile MOPs

**Priority:** P0 correctness

**File:**

- `tt_llk_quasar/llk_lib/llk_unpack_tilize.h`

**Keep:**

- Guard both runtime `set_last_outer_loop_instr` calls with
  `block_ct_dim > 1`.
- Guard the compile-time block variant with
  `if constexpr (BLOCK_CT_DIM > 1)`.

**Why it matters:**

With an inner loop length of one, `set_last_outer_loop_instr` replaces the
only `UNPACR_TILIZE` operation with the reset instruction. The default Quasar
metal tilize path uses `block_ct_dim == 1`, so the bug can suppress actual data
movement and produce stale or incorrect data.

**Do not combine with:** performance scripts or Python harness changes.

**Validation:**

- Run `test_unpack_tilize_quasar.py` with `block_ct_dim` values 1 and 2.
- Cover the default metal API path where `block_ct_dim` is omitted.
- Cover the 32-bit destination path with `BLOCK_CT_DIM == 1`.

### PR 2: Fix dvalid for Quasar reuse-destination unpack

**Priority:** P1 correctness

**File:**

- `tt_llk_quasar/llk_lib/llk_unpack_unary_operand.h`

**Keep:**

- Change the dummy unpacker NOP from `Set_Dvalid=0` to `Set_Dvalid=1`.

**Why it matters:**

The reuse-destination path fills SrcA or SrcB through `MOVD2A/B`. The dummy
unpacker still has to signal dvalid. Without it, math can stall waiting for
`SRCA_VLD` or `SRCB_VLD`, or consume stale validity.

**Do not combine with:** the tilize fix unless maintainers explicitly prefer a
single small "Quasar unpack MOP correctness" PR. The fixes are thematically
related but causally independent.

**Validation:**

- Run `test_eltwise_binary_reuse_dest_quasar.py`.
- Exercise both `DEST_TO_SRCA` and `DEST_TO_SRCB`.
- Add or run a perf-isolate case that checks the dvalid handshake.

### PR 3: Harden compile-producer and parallel builds

**Priority:** High reliability

**Files and hunks:**

- `tests/python_tests/conftest.py`
  - Add `_make_hashable`.
  - Use a hashable compile key in `_collapse_runtime_only_variants`.
  - Safely retain variants without `compile_key_fn`.
  - Replace the stale `TestConfig.MODE == TestMode.PRODUCE` check with
    `TestConfig.BUILD_MODE == BuildMode.PRODUCE`.
- `tests/python_tests/helpers/test_config.py`
  - Create `VARIANT_OBJ_DIR` and `VARIANT_ELF_DIR` inside
    `build_kernel_part` with `exist_ok=True`.

**Why it matters:**

- Nested runtime parameters need stable compile keys to avoid crashes,
  duplicate builds, or incorrect deduplication.
- The old counter-report condition references nonexistent symbols and can
  raise `NameError` during fixture teardown.
- Parallel workers can reach compilation or linking before another worker
  creates the variant directories.

**Leave out:**

- Duplicate `NUM_TILES_IN_BLOCK` removal. It is valid cleanup, but Python was
  already using the later class definition and behavior is unchanged.

**Validation:**

- Run a large `--compile-producer -n auto` collection/build twice and compare
  selected variant and ELF counts.
- Run the counter-report path in produce and consume modes.
- Stress parallel linking and confirm there are no missing-directory errors.

### PR 4: Preserve performance configuration metadata

**Priority:** Medium-high correctness and report fidelity

**Files and hunks:**

- `tests/python_tests/helpers/perf.py`
  - Accept and forward `boot_mode`.
  - Add `register_format_hint` to performance CSV rows.
- `tests/python_tests/helpers/test_config.py`
  - Store `register_format_hint` from the input format.

**Why it matters:**

- Quasar performance tests that request a non-default boot mode should not be
  silently run with `BootMode.DEFAULT`.
- MxFp4 sweeps can use different register-format hints while otherwise sharing
  the same visible format tuple. Without the CSV column, those measurements are
  ambiguous.

**Do not split:**

The `test_config.py` storage and `perf.py` CSV header/value changes form one
contract and must land together.

**Leave out:**

- `return TestOutcome()` from `PerfConfig.run`. Current callers ignore the
  return value, so it is API consistency rather than required behavior.

**Validation:**

- Compare default and TRISC boot-mode performance variants.
- Run MxFp4 reduce or matmul sweeps and confirm distinct
  `register_format_hint` values appear in the CSV.

### PR 5: Add precise PerfRunType selection and the core suite runner

**Priority:** Meaningful workflow functionality

**Files:**

- `tests/python_tests/helpers/param_config.py`
- The run-type-specific schema guidance in
  `tests/python_tests/helpers/perf.py`
- `tests/python_tests/quasar/perf_suite_common.sh`
- `tests/python_tests/quasar/run_perf_suite_and_report.sh`
- `tests/python_tests/quasar/rename_perf_csvs.sh`

**Why it matters:**

pytest `-k` performs substring matching. A bare `PACK_ISOLATE` selector also
matches `UNPACK_ISOLATE`. Qualified IDs such as
`PerfRunType.PACK_ISOLATE` make one-run-type-per-session execution reliable.

The three shell scripts provide the minimum useful suite workflow:

- shared paths, environment checks, and suite inventory
- scoped process-group cleanup instead of broad `pkill`
- timeout and hang classification
- per-run-type report generation
- CSV suffixing before the next run type overwrites the same filenames

**Do not split:**

- Qualified enum-list IDs and the runner's qualified `-k` selector
- `perf_suite_common.sh` and its consumers
- The runner and CSV renamer for multi-run-type usage

**Validation:**

- Confirm `-k PerfRunType.PACK_ISOLATE --collect-only` selects no
  `UNPACK_ISOLATE` variants.
- Run one suite ID for one run type.
- Confirm targeted process cleanup leaves unrelated simulator sessions alone.
- Confirm generated CSVs receive the selected run-type suffix.

### PR 6: Fail fast on Quasar simulator SSH errors

**Priority:** Meaningful operational reliability

**File:**

- `tests/python_tests/helpers/exalens_server.py`

**Keep:**

- Change the single emulator error marker into a tuple.
- Add `FATAL: SSH`.

**Why it matters:**

An instrumented external launcher emits `FATAL: SSH` when it cannot reach the
Aether host. Recognizing it avoids waiting for the full Exalens ready timeout.

**Dependency:**

The improvement is effective only when the launcher in the external
`tt-umd-simulators` checkout emits the matching marker. This dependency should
be explicit in the PR description.

**Validation:**

- Induce an SSH failure and confirm pytest exits promptly.
- Confirm existing `zServer : ERROR` detection still works.

## Optional follow-up PRs

### PR 7: Full-suite orchestration and recovery

**Files:**

- `run_perf_suite_orchestrator.sh`
- `run_perf_suite_rerun_incomplete.sh`
- `preserve_exalens_logs.sh`

**Classification:** Useful for overnight and interrupted runs, but not needed
to execute one run type or one suite subset.

The orchestrator should land before the rerun script because the rerun script
parses its reports and `DONE_ALL_RUN_TYPES` marker. Log preservation is
independent apart from the shared environment helper.

### PR 8: Runbook

**File:**

- `docs/tests/running_quasar_llk_tests_from_tt_metal.md`

**Classification:** Valuable onboarding material, but it does not alter
runtime behavior.

Land it with or after the scripts it documents. Validate commands on a fresh
reservation and clearly identify instructions that depend on changes in the
external `tt-umd-simulators` repository.

### PR 9: Agent guidance

**Files:**

- `.claude/skills/perf-report/SKILL.md`
- `.claude/skills/quasar-perf-test/SKILL.md`
- `.cursor/skills/perf-report/SKILL.md`
- `.cursor/skills/quasar-perf-test/SKILL.md`

**Classification:** Agent discoverability and guidance only.

Keep Claude and Cursor copies in one PR so their instructions do not drift.
Land after the suite scripts and runbook exist.

### PR 10: Sweep-alignment diagnostic

**File:**

- `tests/python_tests/compare_test_and_perf.py`

**Classification:** Independent developer convenience.

The script helps authors compare functional and performance parameter sweeps,
but it is not used by pytest, CI, or the suite scripts. It can be omitted
unless maintainers want to standardize this review step.

## Changes that can be omitted

The following changes are reasonable but do not make a meaningful runtime
difference:

- Removing the shadowed, earlier `NUM_TILES_IN_BLOCK` class
- Returning an empty `TestOutcome` from `PerfConfig.run`
- Timestamp and test-ID logging in `.claude/scripts/run_test.sh` and
  `.cursor/scripts/run_test.sh`
- Comment and example-path corrections in `.cursor/scripts/run_test.sh`
- Skill description wording and placeholder formatting

These are best included only when they support an accepted documentation or
agent-tooling PR.

## Dependency and merge order

Recommended merge order:

1. PR 1: tilize correctness
2. PR 2: reuse-destination dvalid
3. PR 3: compile-producer and parallel-build hardening
4. PR 4: boot mode and register-format metadata
5. PR 6: Exalens SSH fail-fast
6. PR 5: precise run-type selection and core runner
7. Optional PRs 7 through 10

PRs 1, 2, 3, 4, and 6 are independently reviewable. PR 5 should be based on
the qualified `PerfRunType` ID change. Documentation and agent guidance should
not merge before the scripts and behavior they reference.

## Extraction strategy

For each new branch:

1. Branch from the latest `origin/main`.
2. Restore only the listed files from `47a66006894`.
3. Interactively stage only the hunks assigned to that PR where a file contains
   mixed concerns.
4. Add focused regression coverage when practical.
5. Validate the PR independently rather than relying on the original combined
   commit's checks.
6. Avoid stacking correctness PRs beneath optional tooling unless a real code
   dependency requires it.

The original combined commit should remain available as a reference until all
selected PRs have been extracted and their resulting trees have been compared
against the intended subset.
