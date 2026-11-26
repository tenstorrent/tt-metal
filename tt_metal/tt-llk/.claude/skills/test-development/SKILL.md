---
name: test-development
description: Guides planning, adding, changing, or reviewing tests under tt-llk/tests while preserving the functional, performance, architecture, and harness contracts. Use before editing test_*.py, perf_*.py, tests/sources/*_test.cpp, fuser YAML, accuracy tests, hardware-free tests, or the out-of-tree harness; routes the work to the focused test skills and validation path.
user_invocable: true
---

# Test Development

Use this skill first for any change under `tests/`. It classifies the work and
routes to the focused skill; it does not replace those skills.

## 1. Classify the change

| Change | Read next | Contract |
|---|---|---|
| Add or edit `test_*.py` or `sources/*_test.cpp` | `functional-test` | Correct result against a golden; explicitly preserve, improve, or exclude perf coverage |
| Add, edit, repair, or review `perf_*.py` or measured kernel paths | `perf-test`, then `perf-coverage` | Perf measures the functional program |
| Unify a legacy `*_perf.cpp` or retain a phase microbenchmark while adding paired perf | `functional-test`, `perf-test`, `perf-coverage` | Paired path shares runner/kernel; phase path uses explicit `_phases` names and is not comparer coverage |
| Choose a Quasar functional sweep | `functional-test`, then `quasar-test-coverage` | Representative shapes, Dest overflow, formats, and fidelity |
| Run a device or simulator test | `run-test` | Never invoke `pytest` directly |
| Add a fuser case | `.claude/CLAUDE.md`, "The Fuser Is the Reference" | Add YAML or a new op; do not weaken existing reference data or validation |
| Add an SFPU accuracy case | `tests/SFPU_ULP.md` and `docs/tests/accuracy_tests.md` | Accuracy/perf mode is separate from the normal `test_*.py`/`perf_*.py` pairing |
| Add a host-only harness/tool test | Nearby `tests/hardware_free/` tests | Must not open a card; validate with `tests/run_hardware_free.sh` |
| Change the external harness API | `docs/tests/getting_started.md`, "Out-of-tree / proprietary kernels" | Preserve the versioned `tt_llk_harness` contract |

If one change spans rows, follow every applicable contract. For example, a
new Quasar operation normally needs `functional-test`,
`quasar-test-coverage`, `perf-test`, `perf-coverage`, and `run-test`.

## 2. Inventory before editing

1. Identify every target architecture:
   - Wormhole and Blackhole share top-level Python and C++ files; import-time
     constraints can still make their collected sweeps differ.
   - Quasar normally uses `python_tests/quasar/` and `sources/quasar/`, has a
     fourth `sfpu` kernel component (only some operations give it a real
     fourth-thread stage), and uses Quasar-specific markers and sweep rules.
2. Find the complete program:
   - functional Python module and each `test_*` function;
   - perf module and each normalized `test_perf_*` function, if present;
   - every C++ source selected by either side;
   - golden, stimuli, templates, runtimes, constraints, and schema entry.
3. Check discovery, not a registry. Pytest collects `test_*.py`,
   `perf_*.py`, and `fused_*.py`; callable names must start with `test`.
   Device kernels are referenced by `TestConfig`/`PerfConfig` strings. There
   is no CMake registration step for this suite.
4. Read values that tools cannot infer. `compare_test_and_perf.py` sees
   `@parametrize` axes, but not values fixed in function bodies, runtimes,
   `PerfConfig` calls, skips, or kernel literals.

## 3. Record the functional/perf outcome

Every functional change has exactly one outcome:

- **Paired:** it changes an operation or MOP that is, or should be, measured.
  Update or add the corresponding perf function in the same change. Share the
  runner, kernel, and sweep source. Perf may add measurement axes and use only
  documented narrowings; it must not add unsupported values or perf-only
  rejections.
- **Intentionally functional-only:** it is harness/debug/accuracy coverage,
  or a narrow regression that should not become a published measurement.
  Keep it out of the shared perf sweep and state the reason in the PR.

An existing perf pair makes the change paired unless the new function is an
explicit regression-only exception. Do not silently reduce existing perf
coverage.

A phase microbenchmark is a separate published family, not a third functional
outcome. It has no functional golden, uses explicit `_phases` naming, is
excluded from sweep pairing, and does not replace normal paired perf coverage.

## 4. Validate through the owning skills

1. Use `run-test` for focused functional and perf (`--perf`) runs; use its
   architecture inference and compile/simulator flow.
2. Use `perf-coverage` once per applicable architecture. Wormhole and
   Blackhole need separate comparisons even when they share files.
3. Manually compare body-fixed values, runtimes, skips, sources, and measured
   loop structure because the comparer cannot see them.
4. Run `tests/run_hardware_free.sh` when parameters, perf columns, schemas,
   comparer behavior, or harness code changes.
5. Update perf schema versions when columns change, and split-cost files only
   when the build-count change is material; follow `perf-test`.
6. If the change adds an op, changes an LLK signature, or changes unpack/pack
   behavior, also follow `.claude/references/metal-integration.md`.

Report which architectures and modes were run, expected coverage differences,
and anything not validated.
