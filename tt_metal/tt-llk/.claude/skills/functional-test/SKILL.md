---
name: functional-test
description: Guides adding, extending, repairing, or reviewing an LLK functional test while keeping its perf coverage aligned. Use when creating test_[op].py or sources/[op]_test.cpp, adding a test_* function or parametrized case to an existing functional module, changing a functional sweep, golden, stimuli, template, runtime, skip, or kernel path, or deciding whether a functional-only regression needs a test_perf_* counterpart.
user_invocable: true
---

# Functional Test

## Goal

A functional test proves one LLK program against a golden. When that program
is performance-relevant, its perf test measures the same program: one C++
source, one Python runner, and one source of sweep values.

Read `test-development` first. Detailed worked patterns are in
[references/examples.md](references/examples.md); architecture and exception
handling is in [references/edge-cases.md](references/edge-cases.md).

Related skills, in the order work usually needs them:

- `quasar-test-coverage`: choose a Quasar sweep before perf inherits it.
- `perf-test`: add or change the measured extension of the functional test.
- `perf-coverage`: compare functional/perf axes and cross-architecture
  coverage.
- `run-test`: count, compile, and run device or simulator tests.

## Choose the workflow

1. **New functional module:** follow "New test", then make the perf decision.
2. **New behavior in an existing module:** follow "Add a case or function".
3. **Edit a sweep, skip, parameter, golden, runner, or kernel:** inventory
   every functional and perf caller, then apply the coverage contract.
4. **Review:** use the final checklist.

## 1. Inventory the test family

Read, do not infer:

1. Every `test_*` function in the functional module.
2. The matching `perf_*.py` and its `test_perf_*` functions, including
   cross-imports from differently named modules.
3. Every C++ source selected by either side.
4. Shared sweep constants/generators, skips, goldens, stimuli, templates,
   runtimes, and the perf schema entry.

Pairing strips `test_perf_`, `perf_test_`, `test_` or `perf_`, then Quasar's
trailing `_quasar` and a trailing `_perf` or `_test`. Therefore
`test_op_special` pairs with `test_perf_op_special`, not with a generic
`test_perf_op`.

Do not treat `compare_test_and_perf.py` as a complete inventory. It cannot see
body-fixed values, runtimes, `PerfConfig` arguments, skips, selected source
strings, or kernel literals.

## 2. New test

### Files and discovery

- Wormhole/Blackhole:
  - `tests/python_tests/test_<op>.py`
  - `tests/sources/<op>_test.cpp`
- Quasar:
  - `tests/python_tests/quasar/test_<op>_quasar.py`
  - `tests/sources/quasar/<op>_quasar_test.cpp`
  - `@pytest.mark.quasar`

Pytest discovers the file and callable names; there is no test registry or
CMake entry to add. Name every collected callable `test_*`.

### Python

1. Start from a current test with the same pipeline and architecture, not a
   legacy test with a separate perf kernel unless it is an explicitly named
   phase microbenchmark. Good paired patterns:
   `test_pack_l1_acc.py` / `perf_pack_l1_acc.py` and
   `quasar/test_pack_quasar.py` / `quasar/perf_pack_quasar.py`.
2. Define an exported `OP_SWEEP` or one shared generator in the functional
   module. Use canonical axis names from `perf-test`.
3. Put stimuli, golden inputs, block/tile derivation, templates, runtimes,
   `StimuliConfig`, and config construction in one `_run_<op>_test` helper
   (or a functional `test_*` callable intentionally reused by perf).
4. Give that runner keyword-only perf controls:
   `is_perf=False`, `perf_report=None`, `run_types=None`, and
   `loop_factor=1`. Reject `is_perf=True` without `perf_report`.
5. Build through `execute_llk` for the standard shape, or
   `create_test_or_perf_config` when the test needs custom control. Default
   functional runs to `PerfRunType.L1_TO_L1` and loop factor 1.
6. Keep result collection and the golden assertion after the perf return.
   Values needed to configure the kernel must still be derived identically in
   both modes.
7. Use existing `TemplateParameter`, `RuntimeParameter`, constraint, golden,
   and result helpers before adding new ones. Template/runtime struct order
   must match the generated C++ fields exactly.

### C++

1. Use one source for the functional/perf pair. Do not add `<op>_perf.cpp`.
2. Include `params.h` in each active kernel component and define
   `run_kernel(RUNTIME_PARAMETERS params)`.
3. Wormhole/Blackhole normally implement
   `LLK_TRISC_UNPACK`, `LLK_TRISC_MATH`, and `LLK_TRISC_PACK`. Quasar can add
   `LLK_TRISC_ISOLATE_SFPU` when the operation has a real fourth-thread stage.
4. Derive loop bounds, tile shapes, formats, and addresses from the generated
   parameters. Keep functional and perf `L1_TO_L1` on the same buffers and
   loop body.
5. If the test is paired with perf, structure measurement zones and isolates
   with `perf-test`; do not invent a second measured implementation.
6. Preserve the operation's real synchronization and release each Dest block
   at the block boundary. Follow `quasar-test-coverage` for Quasar bank-switch
   coverage.

## 3. Add a case or function

Choose the smallest representation that expresses new behavior:

1. **Another value of an existing behavior:** extend the existing shared
   sweep or generator. If perf intentionally narrows that axis, update and
   document the filtered perf copy.
2. **A distinct operation family or configuration with its own sweep:** add a
   separately named `test_<family>` that calls the existing shared runner.
   If performance-relevant, add the normalized
   `test_perf_<family>` wrapper to the perf module.
3. **A narrow regression:** add a separately named functional function and
   keep it out of perf. State why it is functional-only in the PR; do not
   merge it into the shared sweep merely to make pairing look complete.
4. **A different kernel program:** it may select another `*_test.cpp`, but
   that function and its perf counterpart still share that one source and
   runner.

When a shared runner or C++ source changes, audit every functional function
and perf wrapper that calls it. A local-looking change can alter several
published perf families.

## 4. Functional/perf coverage contract

Classify every functional change:

### Paired coverage

Use for operation/MOP coverage and whenever a perf counterpart already
measures the changed path.

- Functional owns the sweep values and runner.
- Perf imports those and adds only measurement axes (`run_types`,
  `loop_factor`, `is_perf`, and `iterations` where that measurement axis is
  used, commonly in SFPU sweeps).
- Perf has no value, skip, runtime, init path, or kernel source that functional
  lacks.
- Functional-only values are allowed only for documented runtime/edge
  coverage or a documented perf narrowing. Perf-only values are not allowed:
  add a functional golden first or remove them.
- Architecture constraints and invalid-combination rejection apply to both
  modes. Perf does not add its own rejection.
- Follow `perf-test` for run types, measured loops, schemas, and split costs.

### Phase microbenchmark

A phase microbenchmark times internal subroutines or stages rather than the
functional L1-to-L1 program. It is not paired perf coverage and is exceptional:

- Name the module `perf_<op>_phases.py`, the callable
  `test_perf_<op>_phases`, and the schema `perf_<op>_phases`.
- Keep it out of functional/perf comparison and list it as intentionally
  unmatched in the PR.
- Add normal paired coverage separately when the operation itself needs perf
  coverage; the paired path still shares its runner and `*_test.cpp`.
- Use a dedicated `*_perf.cpp` only for an existing, documented phase
  benchmark. Do not copy this pattern for a new operation without explicit
  reviewer agreement.

Follow `perf-test`, "Phase microbenchmarks", for schema and baseline handling.

### Intentionally functional-only

Typical cases are harness/debug behavior, accuracy tests, state-machine
regressions, hang recovery, profiler tests, and a forced block/layout
regression such as `test_pack_l1_acc_two_blocked_packs_of_4`.

Do not create a meaningless perf row for these. Keep the case separately
named and record the rationale.

## 5. Validate

1. Count/compile through `run-test` for every applicable architecture.
2. Run the focused functional function and relevant edge variants.
3. Run the corresponding perf function with `run-test --perf`.
4. Run `perf-coverage` for each applicable architecture; Wormhole and
   Blackhole are separate imports even though they share files.
5. Manually compare sources, body-fixed values, runtimes, skips, config
   construction, and kernel loops.
6. Run `tests/run_hardware_free.sh` when parameters, schema columns, comparer
   logic, or harness code changes.
7. If columns changed, update the schema/version through `perf-test`. If
   variant count changed materially, evaluate split costs.
8. Report before/after ELF and runtime-variant counts for Quasar sweep changes
   and list all intentionally unmatched functions.

## Review checklist

- [ ] Correct file/function names and architecture markers
- [ ] Golden covers every functional value without weakened tolerances
- [ ] One runner and C++ source per functional/perf pair
- [ ] Any phase microbenchmark is separately named `_phases` and documented
- [ ] Shared sweep/generator; canonical axis names
- [ ] New function has a normalized perf peer or an explicit functional-only reason
- [ ] Same skips, parameters, buffers, and `L1_TO_L1` loop in both modes
- [ ] Quasar shape, MX, fidelity, Dest blocking, and fourth-thread rules checked
- [ ] `perf-coverage` plus manual blind-spot inventory completed
- [ ] Functional, perf, host-only, schema, and cost validation applied as needed
