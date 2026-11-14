---
name: perf-coverage
description: Check and fix that LLK test sweeps line up - functional test_*.py versus perf_*.py on one architecture, and Blackhole versus Quasar (or Wormhole versus Blackhole) for functional and perf - using compare_test_and_perf.py. Use when a perf sweep may have drifted from its functional sweep, when narrowing a perf sweep, when running compare_test_and_perf.py --arch / --dir / --cross-arch, when pairing BH and QSR families, unbundling composite parametrize axes, splitting lumped perf wrappers, or equalizing approx_mode / mathop / formats without faking ISA.
user_invocable: true
---

# Perf Coverage

## Goal

Make sweeps comparable, in this priority order:

1. **Same-architecture test versus perf**: the perf sweep is the functional
   sweep, plus measurement axes, minus documented narrowings.
2. **Cross-architecture perf versus perf** and **functional versus
   functional**: shared families and axes use the same names, and the
   remaining differences are real ISA or datapath differences.

Never edit one architecture's sweep only to make another's comparison look
identical. Never invent an axis or op the other ISA cannot run.

Related: `perf-test` (the one-kernel, one-runner contract this skill checks),
`quasar-test-coverage` (choose the Quasar functional sweep),
`perf-report` (run a sweep). Cross-architecture specifics are in
[references/bh-vs-qsr.md](references/bh-vs-qsr.md), with a worked example in
[references/bh-vs-qsr-binary-sfpu.md](references/bh-vs-qsr-binary-sfpu.md).

## How the comparer works

Script: `tests/python_tests/compare_test_and_perf.py`. Run it from
`tests/python_tests` with `tests/.venv` active (created by
`tests/setup_external_testing_env.sh`; a git worktree has none, so use the
main checkout's). It reads `@parametrize`; it does not run tests or touch a
device. It cannot see values set inside a test body, runtimes, or run types
fixed inside a `PerfConfig` call: read the code for those (`perf-test`,
"Inventory the drift").

**Pairing**

- Same-architecture files: `test_<op>.py` with `perf_<op>.py`. Quasar files
  are in `quasar/` and end in `_quasar.py`.
- Cross-architecture files: `--cross-arch LEFT RIGHT --kind perf` or
  `--kind func`. Quasar stems drop a trailing `_quasar`. Wormhole and
  Blackhole share files, which are imported once per `CHIP_ARCH`.
- Functions: strip `test_perf_`, `test_` or `perf_`, then a trailing
  `_quasar`. One lumped wrapper cannot pair with several functional families;
  split it.
- `CROSS_ARCH_FUNCTION_EXCEPTIONS` pairs a function that lives in a different
  module on the other side.

**Axes**

- Verdicts are per axis value set, not per combination. Yes+No on both sides
  is `identical` even if one side uses Yes for one op only.
- Flattening does not rename axes. A dataclass on axis `formats` becomes
  `formats.input`; the same fields packed as `formats_dest_acc` become
  `formats_dest_acc.formats.input` and never meet. Tuples flatten to type
  names (`str`, `MathOperation`), which collide across ops. Unbundle the test
  axes; do not special-case a dataclass in the comparer.
- The comparer joins `math_op` with `mathop`, and splits a tuple axis whose
  name is exactly the other side's axes (`dest_sync_dest_acc`) when each slot
  has the same type. It does not read schema `aliases`.
- Ignored: only the measurement axes `run_types`, `loop_factor`,
  `iterations`, `is_perf`. Everything else is coverage.

**Import.** `--arch` sets one `CHIP_ARCH`. `--cross-arch` sets it per side
and reloads already-imported modules. Pass Quasar paths as
`quasar/test_*_quasar.py`. A mixed `test_` plus `perf_` pair under
`--cross-arch` is an error; compare those with `--arch`.

## Same-architecture workflow

1. Run the comparer for the pair (commands below), once per architecture the
   test runs on: Wormhole and Blackhole import the same file differently.
2. Classify each difference:

   | Class | Example | Action |
   |---|---|---|
   | Perf-only value or axis | perf sweeps a format functional lacks | remove it from perf, or add it to functional with a golden |
   | Functional-only value | perf has no tilize | share the sweep (`perf-test`, "Python") |
   | Functional-only axis that changes the MOP | `num_faces` swept only in functional | share it; perf must measure each MOP functional checks |
   | Perf-only packaging | a shared sweep multiplied by the run types is too big | narrow in the perf module, as below |
   | Documented perf generator | Quasar `PERF_TILE_SIZES`, `DestSync.Half` | keep; it is the architecture's documented narrowing |
   | Functional-only axis | `runtime(...)` axes such as `tile_indices`, edge stimuli | keep; not a separate MOP |

3. Fix the tests, not the comparer, and rerun until only expected
   differences remain.

### Perf-only packaging

A functional sweep shared into perf is multiplied by the run-type count. To
keep a point functional-only, copy the sweep in the perf module and narrow
that axis. Do not edit the functional sweep:

```python
_FLOAT_PERF_SWEEP = {
    **_func.FLOAT_SWEEP,
    "approx_mode": _perf_approx_modes,  # Yes/No only where the kernel reads it
}
```

A keyword after `**FLOAT_SWEEP` for a key the dict already holds raises
`TypeError`; replace the key in a copy instead. Do not add a declared-count
assertion. Pin a perf axis that the kernel ignores (for example
`approx_mode=No` for kernels that never read it) instead of multiplying
builds by it.

## Commands

```bash
cd tests/python_tests && source ../.venv/bin/activate   # or the main checkout's tests/.venv

# Same architecture, one pair
python3 compare_test_and_perf.py --arch wormhole  test_<op>.py perf_<op>.py
python3 compare_test_and_perf.py --arch blackhole test_<op>.py perf_<op>.py
python3 compare_test_and_perf.py --arch quasar \
  quasar/test_<op>_quasar.py quasar/perf_<op>_quasar.py

# Same architecture, whole folder
python3 compare_test_and_perf.py --arch blackhole
python3 compare_test_and_perf.py --dir quasar --arch quasar

# Cross architecture
python3 compare_test_and_perf.py --cross-arch blackhole quasar --kind perf
python3 compare_test_and_perf.py --cross-arch blackhole quasar --kind func
python3 compare_test_and_perf.py --cross-arch wormhole blackhole --kind func
```

## Reading a report

- `[=]` identical value set
- `[~]` one side is a subset
- `[x]` both sides have the axis, and the sets differ
- `[T]` / `[P]` functional-only or perf-only, on a same-architecture compare
- `[B]` / `[Q]` / `[W]` one architecture only, on a cross-architecture compare
- `[i]` ignored measurement axis

Success for test versus perf: every coverage axis identical, except named
functional-only axes and the architecture's documented perf narrowing.

Success across architectures: shared packaging axes identical or an honest
subset, and every remaining difference named as ISA or datapath in the PR.

## After sweep edits

A sweep edit can change CSV columns. Run the host-only suite, which includes
the schema header gate: `tests/run_hardware_free.sh`. Bump the schema version
only when columns change ([perf-facts](../perf-test/references/perf-facts.md),
"Accounting").

## Do not

- patch `compare_test_and_perf.py` to hide a packaging difference;
- take the Cartesian product of format, `dest_acc` or `approx_mode` values
  the kernel cannot run together;
- override a shared sweep with a keyword after `**SWEEP` (`TypeError`);
- shrink both sides to their intersection just to print `identical`;
- declare success from a green report after dropping real ops.
