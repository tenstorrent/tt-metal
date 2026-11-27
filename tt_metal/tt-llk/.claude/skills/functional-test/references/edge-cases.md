# Functional test edge cases

Use this inventory while authoring or reviewing. Apply only the rows relevant
to the changed family, but do not skip the architecture and perf-pairing
checks.

## Architecture

| Case | Required handling |
|---|---|
| Wormhole and Blackhole | They share top-level Python/C++ files, but `CHIP_ARCH` can change constraints, APIs, formats, skips, and collected variants. Compile and compare both. Put signature differences behind existing `tests/helpers/include/llk_lib_*_wrappers.h` patterns rather than scattering `#ifdef ARCH_BLACKHOLE` through test kernels. |
| Quasar | Use `python_tests/quasar/`, `sources/quasar/`, `_quasar` names, and `@pytest.mark.quasar`. Quasar normally boots from TRISC, uses semantic LLK header names, and always compiles an `sfpu` component. Include `sfpu_stub.h` when there is no real fourth-thread stage; a real `LLK_TRISC_ISOLATE_SFPU` section must open the shared `INIT`/`TILE_LOOP` profiler zones. Use `PERF_RUN_TYPES_QUASAR_4_TRISC`/`SFPU_ISOLATE` only when that stage does real work. |
| Quasar sweep | Follow `quasar-test-coverage`: representative tile shapes, at least one Dest-overflow shape, MX smoke rather than an accidental full product, valid fidelity, and per-block Dest release. Perf may use only its documented tile/dimension narrowing. |
| Architecture-only test | Use the established skip/only marker or a shared constraint. Apply the same restriction to the perf peer. A Blackhole-only value in a shared file must not compile on Wormhole. |
| Different APIs by architecture | Copy calls from a current test for the target architecture. Do not copy a WH/BH call into Quasar or an old perf source into a current functional source. |

`BootMode.DEFAULT` resolves to BRISC on Wormhole/Blackhole and TRISC on
Quasar. Override it only when the behavior under test is the boot mode.

## Test type boundaries

| Type | Do this | Do not do this |
|---|---|---|
| Normal functional | Python test + C++ driver + golden | Add a central registry or CMake entry; none exists |
| Fuser | Add a new YAML case, or register a genuinely new op under the rules in `.claude/CLAUDE.md` | Add functions to `test_fused.py`, edit existing YAML/goldens, loosen validation, or change the fuser to make a kernel pass |
| Hardware-free | Put harness/tool checks under `tests/hardware_free/`; run `tests/run_hardware_free.sh` | Put them in `python_tests/`, where card jobs collect them |
| SFPU accuracy | Follow `tests/SFPU_ULP.md` and the accuracy budgets | Treat accuracy `--mode perf` as a normal same-stem `perf_*.py` pair |
| Phase microbenchmark | Name it `perf_<op>_phases.py` / `test_perf_<op>_phases`, keep a dedicated phase source only for an existing documented benchmark, and list it as intentionally unmatched | Present it as paired coverage, share the functional stem without paired L1-to-L1 perf, or copy the pattern for a new operation |
| Out-of-tree | Import only from `tt_llk_harness`, version exported surface changes, and keep the contract fixture green | Reach into `helpers` from an external suite |
| Debug/hang/profiler regression | Keep it separately named and functional-only when it has no useful published measurement | Add a meaningless perf row merely to satisfy filename symmetry |

## Discovery and pairing

- Collected files are `test_*.py`, `perf_*.py`, and `fused_*.py`; collected
  callables start with `test`. A helper accidentally named `test_*` is a
  test, and a callable without that prefix is not collected.
- Normal file pairing is `test_<stem>.py` with `perf_<stem>.py`; Quasar drops
  the trailing `_quasar` for comparison.
- Function pairing strips `test_perf_`, `perf_test_`, `test_`, or `perf_`,
  then the Quasar suffix and trailing `_perf`/`_test`. A module containing
  several functional families needs several paired perf functions, not one
  lumped wrapper.
- `test_block_max8.py` / `perf_block_max8.py` are the concrete paired-plus-
  phases example: both functional families have normal paired perf functions,
  while `perf_block_max8_phases.py` is an explicitly unmatched custom-zone
  microbenchmark.
- When renaming a module, function, axis, or perf column, follow the rename
  checklist in `perf-test`; pairing, schemas, baselines, split costs, and
  hard-coded IDs can all depend on it.

## Sweeps, constraints, and generated parameters

| Edge | Rule |
|---|---|
| Invalid Cartesian combinations | Prefer dependent callables or a shared generator that never emits them. If runtime validation is necessary, skip/reject the same combination in functional and perf. |
| Composite/tuple axis | `compare_test_and_perf.py` can flatten it under type-derived or nested names. Prefer `flat_axes` or explicit axes; do not patch the comparer to hide a mismatch. |
| `runtime(...)` axis | It can intentionally cover runtime indexing without creating a distinct MOP. Document a functional-only runtime axis rather than multiplying perf blindly. |
| Body-fixed value | The comparer cannot see it. If it affects measured behavior, promote it to a single-value axis/recorded parameter and pass it through both modes. |
| Template/runtime field | Emit it in both modes. Keep dataclass field order, serialization order, generated struct order, and C++ reads aligned. Runtime fields change the L1 struct layout. |
| Format inference | Effective Dest accumulation and unpack-to-Dest can differ from the requested enum. Use shared inference/constraint helpers and test the effective path, not just the axis label. |
| Tile shape | Derive face count, face rows, tile count, pack size, and block split from the shape. Do not assume 32x32/four faces for tiny or two-face tiles. |
| Partial Dest block | The common block helper requires uniform divisibility and raises on invalid shapes. If a test computes its own split, prove and handle the tail instead of silently overrunning Dest. |
| Build deduplication | Two IDs that generate the same kernel are not extra coverage. Conversely, changed compile-time values must produce distinct affected ELFs. |
| Golden tolerance | Use the established format-aware checker or a stricter operation-specific assertion. Never loosen a tolerance simply to land a kernel change. |

## Functional/perf relationship

- Perf can be a documented subset of functional; it cannot contain a value
  functional never checks.
- Measurement axes (`run_types`, `loop_factor`, `is_perf`, and `iterations`
  where used) are not coverage differences.
- Quasar `DestSync.Half` and selected perf tile/dimension generators are
  documented narrowings. Other narrowings need an explicit reason.
- A functional-only regression stays outside the shared sweep. If inserted
  into `OP_SWEEP`, every importing perf function inherits it.
- If a new functional path removes a pipeline stage, omit or report the
  corresponding isolate as empty according to `perf-test`; do not fabricate
  work so every run type has a number.
- Shared stimuli/golden computation may be optimized in perf mode, but
  formats, geometry, runtime values, buffers, init, and the real
  `L1_TO_L1` loop remain equivalent.
- A new parameter or runtime can change perf CSV columns. Update
  `helpers/perf/test_schemas.py` and the correct wide schema only when the
  column set changes; bump the module version once relative to main.
- A materially larger paired sweep multiplies by perf run types. Reassess
  `perf_split_costs.json` and `perf_split_costs_sol.json`; do not shrink
  correctness coverage solely to reduce CI cost.

## Kernel and runtime failures

| Symptom/risk | Check |
|---|---|
| Compile-only passes, run cannot find ELF | Functional/perf filters, architecture, assert mode, speed-of-light, and variant IDs must match between producer and consumer. Use `run-test`. |
| Hang | Map SrcA/SrcB and Dest producer/consumer handshakes. Perf stand-in counts must equal real pulses. Do not auto-retry or reset blindly; use `run-test` triage and `debug-kernel`. |
| Reconfiguration-only failure | Tests can leak hardware state. Do not reset the device or add end-of-call cleanup to hide it. Reproduce the ordering, identify the leaked state, make the consuming operation's init program every state it depends on at entry, and keep the ordering as a regression. |
| Bit-exact rerun failure | Result L1 and mutable state must be restored between runs. |
| Functional passes, isolate hangs | The shared real loop can be correct while a perf partner stand-in, wait mask, or Dest release is wrong. Follow the architecture reference in `perf-test`. |
| Speed-of-light mismatch | Every compile-time constant must mean the same thing as its normal-build runtime field. Never select semantic values on `LLK_PROFILER`. |
| Quasar SFPU path missing | Confirm whether the operation is 3- or 4-TRISC and use the matching Quasar run-type set, including `SFPU_ISOLATE` only when the stage exists. |

## Validation traps

- Do not invoke raw `pytest` for kernel/device/simulator tests. Use
  `run-test`, including count and compile-only flows.
- Run `compare_test_and_perf.py` once per supported architecture. The same
  top-level module can produce different Wormhole and Blackhole sweeps.
- A green comparer is insufficient: inspect sources, body values, skips,
  runtimes, config kwargs, speed-of-light values, and C++ loops manually.
- Plain collection can initialize a device; use the `run-test` count path
  rather than ad hoc collection commands.
- Run all perf run types together after focused isolate debugging; persistent
  state bugs can appear only when one type follows another.
- If an LLK signature, operation, unpack path, or pack path changed, test
  work is not the whole integration. Follow
  `.claude/references/metal-integration.md`.
