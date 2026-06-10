# SFPU Binary-Op Accuracy Sweep — Design

**Date:** 2026-06-10
**Status:** Approved (design); pending implementation plan
**Author:** Jaksa Macanovic
**Builds on:** `2026-06-05-sfpu-accuracy-csv-design.md` (the unary accuracy harness)

## Summary

Extend the SFPU accuracy harness to cover **binary** SFPU ops. A binary op is a
function of two inputs `f(a, b)`, so the sweep traverses a **2-D grid** of
(a, b) values and exports per-element golden-vs-hardware data to **one CSV per
op**, in a separate `_csv_output/binary/` subfolder. The unary work is reused
where possible (metrics module, shard→merge plumbing) and left behavior-stable.

## Goals

- Sweep binary SFPU ops over a deterministic 2-D grid and record per-point
  accuracy (same metrics as unary: abs/signed/rel error, ULP, finite flags).
- One CSV per op under `_csv_output/binary/`, feeding the same dashboard.
- Reuse the shared metrics and the shard→merge collection; keep the unary path
  behavior-stable (only a backward-compatible generalization).

## Non-goals (YAGNI)

- **pow, rsub** — the kernel supports them but `BinarySFPUGolden` has no golden;
  excluded (would require adding `_pow`/`_rsub` goldens first).
- **add / sub / mul** — supported but essentially bit-exact; low accuracy
  interest; excluded.
- Multi-tile grids (N > 32), broadcast variants, block/integer formats.

## Key decisions (from brainstorming)

| # | Decision |
|---|----------|
| Sweep shape | **2-D grid** — full N×N Cartesian product of per-operand value sweeps → a surface/heatmap per variant. |
| Grid size | N = 32 (1024 points = exactly one 32×32 tile per operand; tunable constant). |
| Ops | **div + xlogy** only (both have kernel + golden; the two approximated binary ops). |
| Formats | Match unary: Float32, Float16, Float16_b (via `input_output_formats` = 9 pairs). |
| Configs | `approx_mode {No,Yes}` × `dest_acc {No,Yes}`. Binary has **no** fast_mode; broadcast fixed to None. |
| Output | Separate `_csv_output/binary/<op>.csv`. Unary relocates to `_csv_output/unary/<op>.csv` for symmetry. |
| Structure | New `accuracy/binary_harness.py`; generalize `clear_shards`/`merge_shards` by path so both reuse them. |
| Gating | Sanity-assert only (length, not-all-NaN, not-all-zero). No ULP thresholds. |

## Background: how binary SFPU tests work (verified)

- C++ source: `sources/sfpu_binary_test.cpp`.
- Golden: `BinarySFPUGolden`. Its call form is
  `golden(op, src, src1_idx=0, src2_idx=1, dst_idx=0, num_iterations=32, input_dimensions=[64,32], stimuli_format)`.
  It reads the **two operands as two tiles of a single tensor**: tile 0 is
  operand A, tile 1 is operand B. So `input_dimensions=[64,32]` = 2 stacked
  32×32 tiles, and the op computes `f(tile0[i], tile1[i])` element-wise.
- Op support (kernel `helpers/include/sfpu_operations.h` + `BinarySFPUGolden`):
  ADD, SUB, MUL, **DIV**, **XLOGY** have both; POW/RSUB have kernel but no
  golden. → div + xlogy are the testable accuracy-interesting ops.

## Architecture

```
helpers/accuracy_metrics.py            REUSED, unchanged (operand-agnostic)

accuracy/accuracy_harness.py           MODIFY (backward-compatible generalization):
    OUTPUT_DIR = _csv_output/unary      # was _csv_output  (unary relocated)
    SHARD_DIR  = _csv_output/unary/_shards
    write_shard(df, variant, shard_dir=SHARD_DIR)
    clear_shards(shard_dir=SHARD_DIR)
    merge_shards(shard_dir=SHARD_DIR, output_dir=OUTPUT_DIR, sort_cols=MERGE_SORT_COLS)
    # unary code (CSV_COLUMNS, rows_dataframe, run_case, build_sweep_spec)
    # is otherwise unchanged (run_case still calls write_shard(df, vname)
    # with the default unary SHARD_DIR).

accuracy/binary_harness.py             NEW
    OUTPUT_DIR_BINARY = _csv_output/binary
    SHARD_DIR_BINARY  = _csv_output/binary/_shards
    BINARY_CSV_COLUMNS, BINARY_MERGE_SORT_COLS, BINARY_GRID_N = 32
    build_binary_grid(op, in_fmt, n=BINARY_GRID_N) -> (a_flat, b_flat)
    rows_dataframe_binary(...) -> pd.DataFrame
    run_case_binary(op, formats, approx, dest, *, n=BINARY_GRID_N) -> Path

accuracy/conftest.py                   MODIFY: clear + merge BOTH dirs
    pytest_configure       -> clear_shards(SHARD_DIR); clear_shards(SHARD_DIR_BINARY)
    pytest_sessionfinish   -> merge_shards(unary defaults); merge_shards(binary paths/sort)

accuracy/test_sfpu_binary_accuracy.py  NEW: the mega grid sweep
accuracy/tests/test_binary_harness_units.py  NEW: off-device unit tests
accuracy/README.md                     MODIFY: document the binary sweep + new layout
```

### Output layout (symmetric)

```
_csv_output/                  (gitignored)
├── unary/
│   ├── _shards/              per-variant intermediates
│   └── exp.csv, log.csv, …
└── binary/
    ├── _shards/
    └── sfpuelwdiv.csv, sfpuxlogy.csv
```

Each category owns its `_shards/`; the unary merge reads only `unary/_shards/`
and the binary merge only `binary/_shards/`, so the two never interfere
(a unary-only run no-ops the binary merge and vice versa).

## The 2-D grid → tile mapping

1. `a_vals` = `n` deterministic ramp points over the op's clipped **operand-A**
   domain (`for_op(op, in_fmt).spec_A` → `exclude_undefined(op, ·, Operand.A)`),
   piecewise across intervals when present.
2. `b_vals` = `n` points over the clipped **operand-B** domain (e.g. div's
   divisor: two bands `[(-10,-0.1),(0.1,10)]`, so `b` never hits 0).
3. `A, B = np.meshgrid(a_vals, b_vals)`; `a_flat = A.ravel()`, `b_flat = B.ravel()`
   → `n²` element-aligned points enumerating every (a, b) combo.
4. Build the `[64, 32]` source tensor: tile 0 = `a_flat` (reshaped 32×32 in the
   layout the golden/kernel expect), tile 1 = `b_flat`. With n = 32, each
   operand is exactly one 1024-element tile.

`BinarySFPUGolden(op, src, src1_idx=0, src2_idx=1, dst_idx=0, num_iterations=32,
[64,32], golden_fmt)` then computes `f(a_flat[k], b_flat[k])` for all k.

## CSV schema (binary)

```
identity : op, input_format, output_format, chip_arch, distribution,
           intervals_a, intervals_b, seed
data     : sample_index, x_a, x_b, golden, hw
config   : approx_mode, dest_acc                      (0/1; no fast_mode)
metrics  : signed_error, rel_error, signed_ulp_error,
           is_finite_hw, is_finite_golden
```

Same compact conventions as unary: formats/arch abbreviated (`fp32`/`fp16`/
`bf16`, `wh`/`bh`/`qsr`), finite flags `T`/`F`, floats `%.7g`, `abs_*` derived
by the consumer. `distribution = "grid"`. `intervals_a`/`intervals_b` record the
two swept domains. `BINARY_MERGE_SORT_COLS` ends with `x_a, x_b` for byte-stable
reruns. ULP semantics are identical to unary (NaN for fp32-output / non-finite /
golden==0 / block-float).

## `run_case_binary` data flow

```
build_binary_grid(op, in_fmt)            -> a_flat, b_flat (n² each)
  -> assemble [64,32] src (tile0=a, tile1=b)
  -> BinarySFPUGolden(op, src, 0, 1, 0, 32, [64,32], golden_fmt)   # golden
  -> TestConfig("sources/sfpu_binary_test.cpp",
        templates=[generate_input_dim, MATH_OP(op), APPROX_MODE(approx),
                   BROADCAST_TYPE(None_)],
        variant_stimuli=StimuliConfig(src, …), dest_acc=dest,
        unpack_to_dest=input_format.is_32_bit())
  -> .run().result  (hardware)
  -> rows_dataframe_binary(a=a_flat, b=b_flat, golden, hw, …)
        # sorts by (x_a, x_b), calls compute_pointwise_metrics internally
        # (mirrors unary rows_dataframe), assigns sample_index
  -> write_shard(df, variant, SHARD_DIR_BINARY)
```

Note: `write_shard` also gains a `shard_dir` argument (defaulting to the unary
`SHARD_DIR`) as part of the same backward-compatible generalization, so the
binary path can target `SHARD_DIR_BINARY`.

Sanity-asserts only: result length matches golden, HW not all-non-finite, and
not all-zero when golden has signal. No ULP/error gating.

`golden_fmt` follows the existing binary test rule (`Float16_b` when input is a
block format, else the input format). The bf16/fp16/fp32 set we sweep needs no
special-casing, but the rule is preserved for correctness.

## Mega test

`accuracy/test_sfpu_binary_accuracy.py`:
- `BINARY_OPS = [SfpuElwdiv, SfpuXlogy]`
- `FORMATS = input_output_formats([Float32, Float16, Float16_b])`
- params = `product(FORMATS, [approx No,Yes], BINARY_OPS, [dest No,Yes])`
- `@pytest.mark.nightly`, body calls `run_case_binary`.
- Skip guards mirrored from `test_zzz_sfpu_binary.py`:
  - `fp32 input + dest_acc=No` → skip (unsupported).
  - Blackhole + `Float16` input + `dest_acc=No` → skip.
  - (Blackhole auto-upgrades Float16/Float32 to dest_acc=Yes, as the existing
    test does.)

## Shared-plumbing change (impact on unary)

Only `clear_shards`/`merge_shards` change, gaining parameters whose **defaults
equal today's values**, so existing unary callers behave identically. Unary
`CSV_COLUMNS`, `rows_dataframe`, `run_case`, `build_sweep_spec` are untouched.
The unary output **relocates** from `_csv_output/*.csv` to
`_csv_output/unary/*.csv` (constants change) — a cosmetic move of gitignored
output, the only behavior change to the unary flow. `conftest.py` keeps its
unary clear+merge and adds the binary ones.

## Testing

Off-device unit tests (`accuracy/tests/test_binary_harness_units.py`):
- `build_binary_grid` returns `n²` element-aligned points; every point lies in
  the clipped per-operand domains; for div, no `x_b` equals 0 (divisor band).
- `rows_dataframe_binary` emits exactly `BINARY_CSV_COLUMNS`, with `x_a`/`x_b`
  present and a stable `sample_index`.
- `merge_shards` with the binary paths/sort writes `binary/<op>.csv`, overwrites
  from current-run shards only, sorted by `BINARY_MERGE_SORT_COLS`.
- Re-run the existing unary unit tests to confirm the shard/merge generalization
  is behavior-preserving.

On-device validation: run a small slice (e.g. `div` in Float16_b) via the
run_test.sh wrapper / `CHIP_ARCH=… pytest`, confirm `binary/sfpuelwdiv.csv` is
produced with the schema above and sorted by `x_a, x_b`.

## Files changed / added

- `accuracy/binary_harness.py` — NEW.
- `accuracy/test_sfpu_binary_accuracy.py` — NEW.
- `accuracy/tests/test_binary_harness_units.py` — NEW.
- `accuracy/accuracy_harness.py` — MODIFY (relocate unary dirs; parameterize
  clear/merge).
- `accuracy/conftest.py` — MODIFY (clear+merge both categories).
- `accuracy/README.md` — MODIFY (binary sweep + symmetric layout).

## Open questions / future work

- Add `_pow` / `_rsub` goldens to `BinarySFPUGolden` to extend coverage to pow
  and rsub.
- Higher-resolution grids (N > 32, multi-tile) if the dashboard wants finer
  heatmaps.
- Block/integer-format binary ops (shifts, comparisons).
