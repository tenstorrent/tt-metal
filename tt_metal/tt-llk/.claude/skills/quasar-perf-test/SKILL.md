---
name: quasar-perf-test
description: Entry point for Quasar LLK perf tests - creating or repairing perf_[op]_quasar.py, wiring PerfConfig, implementing UNPACK_ISOLATE / MATH_ISOLATE / PACK_ISOLATE / SFPU_ISOLATE, choosing PERF_TILE_SIZES or dest-fill matrices, or investigating implausible Quasar perf metrics and dvalid handshakes. Routes to perf-test and its Quasar reference.
user_invocable: true
---

# Quasar Perf Test

The perf-test contract is the same on every architecture, so it lives in one
skill. For Quasar work:

1. Read `perf-test` (`.claude/skills/perf-test/SKILL.md`) and follow it. It
   covers the shared sweep and runner, the kernel structure, the run-type
   table, source and Dest dvalid accounting, the repair workflow, and
   validation.
2. Read its Quasar reference,
   `.claude/skills/perf-test/references/quasar.md`, for what differs on
   Quasar: the `_PERF_AXES` harness, `PERF_RUN_TYPES_QUASAR`,
   `select_perf_tile_sizes()` and `generate_perf_input_dimensions()`, wait
   masks, `set_up_dest_dvalid_per_thread`, and the known Quasar stand-in
   counts.

Related: `quasar-test-coverage` chooses the functional sweep that perf
inherits; `perf-coverage` compares Quasar test versus perf, and Blackhole
versus Quasar.
