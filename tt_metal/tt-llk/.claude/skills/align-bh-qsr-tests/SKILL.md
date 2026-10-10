---
name: align-bh-qsr-tests
description: Align Blackhole and Quasar LLK functional and performance tests so family names and sweep axes compare. Use when pairing BH vs QSR coverage, unbundling composite parametrize axes, splitting lumped perf wrappers, running compare_test_and_perf.py --arch or --cross-arch, matching test-vs-perf or cross-arch func/func and perf/perf reports, or equalizing approx_mode / mathop / formats without faking ISA.
user_invocable: true
---

# Align BH vs QSR Functional and Perf Tests

## Goal

Make one LLK op comparable across:

1. **Same-arch test vs perf** (functional `test_*.py` vs `perf_*.py`)
2. **Cross-arch perf vs perf** (Blackhole `perf_*.py` vs Quasar `quasar/perf_*_quasar.py`)
3. **Cross-arch func vs func** (Blackhole `test_*.py` vs Quasar `quasar/test_*_quasar.py`)

Priority is always (1), then (2) and (3). Do not edit Blackhole sweeps just to green a Quasar compare. Do not invent axes or mathops the other ISA cannot run.

Related: `quasar-perf-test` (create/repair a Quasar perf harness), `perf-report` (run a sweep), `run-test` (execute tests). This skill is the coverage-alignment layer on top of those.

## How the comparer works

Script: `tests/python_tests/compare_test_and_perf.py`. Run from `tests/python_tests` with `tests/.venv`. It introspects `@parametrize`; it does not execute tests.

**Pairing**

- Same-arch files: `test_<op>.py` ↔ `perf_<op>.py`. Quasar lives in `quasar/` and uses `*_quasar.py`.
- Cross-arch files: `--cross-arch LEFT RIGHT --kind perf` or `--kind func`. Quasar stems drop a trailing `_quasar`. Wormhole and Blackhole share files and are imported twice, once per `CHIP_ARCH`.
- Functions: strip `test_perf_` / `test_` / `perf_`, then a trailing `_quasar`. `test_perf_eltwise_binary_sfpu_float` pairs with `test_perf_eltwise_binary_sfpu_float_quasar`. One lumped QSR wrapper cannot pair with six functional families.
- `CROSS_ARCH_FUNCTION_EXCEPTIONS` pairs a function that lives in a different module. The one entry is Blackhole `eltwise_binary_dest_reuse` inside `perf_eltwise_binary.py` with Quasar `perf_eltwise_binary_reuse_dest_quasar.py`.
- The comparer joins parametrize axis names. It does not read schema `aliases`. `math_op` is joined to `mathop` when the two sides use different names. A tuple axis whose name is exactly the other side's axes (`dest_sync_dest_acc` is `dest_sync` + `dest_acc`) is split when each slot is the same enum or dataclass. An axis that exists on only one architecture stays one-sided. `mathop` versus `op` is still fixed by unbundling the tests, not by a CSV alias.

**Verdicts are axis-level sets**, not combinations. `approx_mode` Yes+No on both sides is `identical` even if QSR only uses Yes for ATAN2. Combinations still differ.

**Flattening does not rename axes.** A dataclass on axis `formats` becomes `formats.input`. The same fields packed as `formats_dest_acc` become `formats_dest_acc.formats.input` and never meet. Tuples flatten to type names (`str`, `MathOperation`), which collides across ops. Unbundle the test axes; do not patch the comparer to special-case one dataclass.

**Ignored axes:** measurement controls `run_types`, `loop_factor`, `iterations`, `is_perf`, plus `implied_math_format` on every compare. The format axis prints `[i]` and is not a measurement control. Everything else is coverage, including perf-only `approx_mode`. Do not add `implied_math_format` to BH or WH to make the axis exist on both sides.

**Import:** `--arch` sets one `CHIP_ARCH`. `--cross-arch` sets it per side and clears `helpers.chip_architecture._cached_chip_architecture` before each import, reloading a module that was already loaded. Pass Quasar paths as `quasar/test_*_quasar.py` (bare `test_*_quasar.py` fails import). A mixed `test_` + `perf_` pair with `--cross-arch` is an error; that compare stays on `--arch`.

## Workflow

Copy and track:

```
Alignment:
- [ ] Inventory families and axes on BH functional, BH perf, QSR functional, QSR perf
- [ ] Split QSR perf 1:1 with QSR functional families
- [ ] Unbundle QSR composites onto BH axis names (constrained callables, not full cartesian)
- [ ] Use shared MathOperation aliases where the kernel is the same
- [ ] Pin perf approx_mode to No except kernels that read it (atan2)
- [ ] Leave real ISA / datapath diffs visible; leave implied_math_format ignored
- [ ] QSR test vs perf
- [ ] BH vs QSR perf, and BH vs QSR functional
- [ ] Header gate; optional LLK perf YAML
```

### 1. Inventory

Read all four modules. For each family record: function name, `@parametrize` axis names, value sets, functional-only `runtime(...)` axes, perf-only `_PERF_AXES`, C++ `BinaryOp` / kernel path.

Classify every mismatch:

| Class | Examples | Action |
|---|---|---|
| Name pairing | One QSR perf vs six functional names | Split wrappers |
| Axis packaging | `formats_dest_acc` vs `formats`+`dest_acc`; tuple `binary_op_mathop_approx` vs `mathop`+`approx_mode` | Unbundle to BH names |
| Enum alias | QSR `SfpuElwmulInt` vs BH `SfpuMulInt32` | Keep the QSR member; the domain test maps it onto the WH/BH kernel |
| Perf-only packaging | A functional sweep shared into perf grows by the run-type count | Opt out with a perf-only override such as `_PERF_EXCLUDED_MATHOPS` |
| Real ISA | Quasar int sweep does not include shifts; BH int SUB; BH `bcast_dim` | Keep the diff. `RSHFT` / `LSHFT` / `LOGICAL_RSHFT` exist on Quasar. `implied_math_format` is ignored, not a gap |
| Family split | BH `div`/`atan2` are separate tests; QSR lumps them into `float` | Split QSR only if the kernel exists; do not invent BH families |

### 2. Align QSR test vs perf first

Shared Python helper + shared C++ kernel via `create_test_or_perf_config`. Perf wrappers import the functional module and pass `**FLOAT_SWEEP, **_PERF_AXES` (or `INT_SWEEP`, …).

```python
# tests/python_tests/quasar/perf_<op>_quasar.py
_PERF_AXES = dict(
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
```

Do **not** put `approx_mode` on that global dict if any family already has a constrained `approx_mode` axis. A later keyword after `**FLOAT_SWEEP` raises `TypeError` because that key is already in the dict. Copy the sweep and replace the key:

```python
_FLOAT_PERF_SWEEP = {
    **_func.FLOAT_SWEEP,
    "implied_math_format": [ImpliedMathFormat.Yes],
    "approx_mode": _perf_approx_modes,  # Yes/No only for atan2; No otherwise
}
```

Perf `approx_mode` is No except kernels that read it (atan2). Functional QSR DIV and atan2 still sweep Yes/No. A functional sweep dict shared into perf becomes perf cases, times the run types. A perf-only override such as `_PERF_EXCLUDED_MATHOPS` is the expected opt-out. Do not add a declared-count assertion.

Functional-only axes use `runtime(...)` (e.g. `tile_indices`) and stay off the perf wrapper. Expected test-vs-perf: coverage identical except those functional-only axes and ignored axes. `implied_math_format` is ignored, so a functional No+Yes versus a perf pin of Yes is not a coverage mismatch.

### 3. Unbundle onto BH names

BH typically sweeps flat `formats`, `dest_acc`, `mathop`, `approx_mode`. QSR often packs `QuasarSfpuVariant` or `(binary_op, mathop, approx)` tuples.

Explode representatives into named axes. Keep the **same combinations** with callable constraints (`helpers.param_config._params_solve_dependencies` resolves callables from argument names):

```python
def _dest_acc_for_float_formats(formats):
    return [v.dest_acc for v in _VARIANTS
            if v.formats.input_format == formats.input_format
            and v.formats.output_format == formats.output_format]

def _approx_modes_for_mathop(mathop):
    # Functional. Perf copies the sweep and pins DIV back to No.
    if mathop in (MathOperation.SfpuAtan2, MathOperation.SfpuElwdiv):
        return [ApproximationMode.No, ApproximationMode.Yes]
    return [ApproximationMode.No]

FLOAT_SWEEP = dict(
    formats=[v.formats for v in _VARIANTS],
    dest_acc=_dest_acc_for_float_formats,
    mathop=_MATHOPS,
    approx_mode=_approx_modes_for_mathop,
    implied_math_format=[ImpliedMathFormat.No, ImpliedMathFormat.Yes],  # QSR-only; comparer ignores it
)
```

Do not cartesian every in/out × dest_acc pair. Resolve `QuasarSfpuVariant` / `binary_op = mathop.cpp_enum_value` **inside** the test, not as a sweep axis.

### 4. Shared aliases, not QSR-only enum members

`SfpuGtInt` / `SfpuLtInt` / `SfpuLeInt` / `SfpuGeInt` remain in `helpers/llk_params.py` as aliases. They are not sweep values; sweeps use `SfpuElwLt/Gt/Le/Ge` on every architecture. `SfpuElwmulInt` stays Quasar-only (`BinaryOp::MUL`); the domain test maps it to `SfpuMulInt32` (`MUL_INT32`) and must not pretend they are the same sweep value. `SfpuCopyDest` (`COPY_DEST`) is declared only on Quasar. Leave `MathOpType.SFPU_BINARY` as it is, and keep a guard that `COPY_DEST` is absent from the WH/BH `BinaryOp` enums.

### 5. Then BH vs QSR perf

Equalize only packaging. Typical remaining diffs after a good unbundle:

- **identical**: shared coverage axes whose value sets match (`dest_acc`, Int32 `formats`)
- **subset**: QSR formats without Bfp8 / MX-only extras. Float perf `approx_mode` is a Blackhole subset of Quasar (`[No]` versus `[No, Yes]`) because atan2 stays in the Quasar float family
- **DIFFERENT mathop**: intersection is the portable ops; extras are ISA
- **one architecture only**: `bcast_dim`, `dest_sync`. The report tags these `[B]` or `[Q]`. `iterations` is `[i]`, an ignored measurement axis

`implied_math_format` is `[i]`, not a QSR-only gap. Do not add a dummy `bcast_dim=[None_]` or a fake BH `implied_math_format`. Do not shrink both mathop lists to the intersection just to print `identical`. Do not add Bfp8_b to DIV perf: unpack expands it to BF16 before the SFPU, so the extra rows do not change the math kernel.

Optional packaging: split a QSR lumped family so it pairs with an existing BH family name (`div`, `atan2`) when both have the kernel. That turns `mathop DIFFERENT` into `QSR subset` plus new matched pairs.

### 6. Schema and header gate

Homogeneous CSV schema is per **module** name (`perf_eltwise_binary_sfpu`, `perf_eltwise_binary_sfpu_quasar`), not per family wrapper. Do not bump `helpers/perf/test_schemas.py` unless templates/runtimes grow a new column. After sweep-axis edits:

```bash
cd tests && python3 -m pytest python_tests/test_perf_header_gate.py -q
```

## Commands

Always `source tests/.venv/bin/activate`. Cwd `tests/python_tests` for the comparer.

```bash
# QSR test vs perf (focused)
python3 compare_test_and_perf.py --arch quasar \
  quasar/test_<op>_quasar.py quasar/perf_<op>_quasar.py

# BH test vs perf (focused)
python3 compare_test_and_perf.py --arch blackhole \
  test_<op>.py perf_<op>.py

# BH vs QSR perf (focused, or the whole folder)
python3 compare_test_and_perf.py --cross-arch blackhole quasar \
  perf_<op>.py quasar/perf_<op>_quasar.py
python3 compare_test_and_perf.py --cross-arch blackhole quasar --kind perf

# BH vs QSR functional
python3 compare_test_and_perf.py --cross-arch blackhole quasar \
  test_<op>.py quasar/test_<op>_quasar.py
python3 compare_test_and_perf.py --cross-arch blackhole quasar --kind func

# Wormhole vs Blackhole reimports the same files
python3 compare_test_and_perf.py --cross-arch wormhole blackhole --kind func

# Same-arch folder sweeps
python3 compare_test_and_perf.py --dir quasar --arch quasar
python3 compare_test_and_perf.py --arch blackhole
```

Hardware: `run-test` / `perf-report`, not ad-hoc pytest. CI: workflow **LLK perf** (`llk-perf.yaml`) with `architecture=all|blackhole|quasar` and `speed-of-light=false` when comparing against a non-SoL baseline. `--speed-of-light` changes measured cycles; never mix SoL and non-SoL rows.

```bash
gh workflow run "LLK perf" --ref <branch> -f architecture=all -f speed-of-light=false
```

## Reading a report

- `[=]` identical value set
- `[~]` one side is a subset
- `[x]` both have the axis, sets differ
- `[T]` / `[P]` functional-only / perf-only, on a same-arch compare
- `[B]` / `[Q]` / `[W]` one architecture only, on a cross-arch compare
- `[i]` ignored. Measurement controls are labeled as such. `implied_math_format` is ignored and is not a measurement control

Success for test vs perf: coverage axes identical except documented functional-only (`tile_indices`, edge stimuli) and ignored axes. Perf `approx_mode` is No except atan2.

Success for BH vs QSR: shared packaging axes identical or an honest subset; remaining diffs named as ISA or datapath in the PR.

## Do not

- Flatten `QuasarSfpuVariant` in `compare_test_and_perf.py` instead of unbundling tests
- Cartesian illegal format × dest_acc / approx_mode combinations
- Add QSR mathops missing from `tt_llk_quasar/.../ckernel_defs.h` `BinaryOp`
- Add BH `implied_math_format` or QSR `bcast_dim` without a kernel path
- Rename QSR-only families (`quant`, `bf16_rne`, `max_min_*`) to BH names
- Put `approx_mode` on QSR global `_PERF_AXES` when float already constrains it
- Override a shared sweep with a keyword after `**FLOAT_SWEEP` (`TypeError`)
- Cartesian perf `approx_mode` Yes/No onto kernels that ignore it
- Halve `TILE_COUNT` to make per-tile MATH_ISOLATE match an older one-operand kernel. `tile_cnt` also bounds unpack and dvalids
- Drop `#1230` from the Wormhole/Blackhole NONE `_perf_unpack_loop_set_valid` mocks
- Declare success from pytest or a green `[x] mathop` after dropping real ops

## Measurement notes from the binary SFPU perf alignment

- The perf schema version is `PERF_TEST_SCHEMAS[...]["version"]` in `helpers/perf/test_schemas.py`, not a CSV column. It changes when columns change (binary SFPU perf went v3 to v5: different kernel, two real operands). A `tile_cnt` accounting change can make rows incomparable without a version bump: when `tile_cnt` counts both operand tiles, per-tile MATH_ISOLATE is about half a result tile. At 16-bit, `dest_acc=Yes` also doubles dest handshakes; leave that measurement as-is.
- Datacopy and SFPU init are hoisted out of the per-block loop on purpose. Production still reconfigs per tile.
- Keep `#1230` on the Wormhole/Blackhole NONE dvalid mocks: SrcA plus a SrcB zerosrc dvalid every face, including `dest_acc=No`. Quasar posts per tile, and SrcB only when dest is 32-bit (`<true, is_fp32_dest_acc_en>(LOOP_FACTOR * TILE_CNT)`).

Worked example (eltwise binary SFPU): [examples.md](examples.md)
