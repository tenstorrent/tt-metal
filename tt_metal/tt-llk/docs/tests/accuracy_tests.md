# SFPU Accuracy Tests

The accuracy suite sweeps SFPU ops across their input domain on hardware
(Wormhole and Blackhole only), compares each result against a torch golden
reference, and writes a per-op file of golden-vs-hardware error data. Use it to
see how accurate an op is across its range, in absolute, relative, and ULP
error.

The tests live in `tests/python_tests/accuracy/`.

## Running

These are ordinary pytest tests. Just set the device arch via `CHIP_ARCH`.
The examples below use paths relative to `tests/python_tests/` and `CHIP_ARCH = wormhole`.

```bash
# one op (--op selects by MathOperation name; repeatable for several)
CHIP_ARCH=wormhole pytest accuracy/test_sfpu_accuracy.py --op=Reciprocal -s

# runs every op in the given test file
CHIP_ARCH=wormhole pytest accuracy/test_examples_accuracy.py -s
```

`--op` takes a `MathOperation` name (case-insensitive, exact), is repeatable
(`--op=Exp --op=Log`), and composes with `-k`/`-m`.

## Run modes

The same op/format matrix can run in three modes, selected with `--mode`. All
three run the one merged kernel (`sources/eltwise_unary_sfpu_perf.cpp`); they
differ only in what they measure:

- `accuracy` (default) — run on hardware and compare against the torch golden,
  writing the per-op error files described below.
- `perf` — profile every scenario for timings only (no golden compare, no file).
- `both` — do both in one pass: profile *and* write the accuracy files.

```bash
# accuracy only (default — --mode can be omitted)
CHIP_ARCH=wormhole pytest accuracy/test_sfpu_accuracy.py

# perf only
CHIP_ARCH=wormhole pytest accuracy/test_sfpu_accuracy.py --mode=perf

# accuracy + perf in one run
CHIP_ARCH=wormhole pytest accuracy/test_sfpu_accuracy.py --mode=both
```

`--mode` keeps only the matching sweep and deselects the other two. It defaults
to `accuracy`, and can be combined with `--op`.

## Output

Results go to `accuracy/_csv_output/<arch>/` (e.g. `wh/`, `bh/`), one file per op
(`exp.parquet`, `reciprocal.parquet`, ...). Each run clears the intermediate
shards, then rewrites only the ops in that run — so a partial run leaves other
ops' files untouched and you can build up the results incrementally. To start
fresh, delete `accuracy/_csv_output/`.

### File format: Parquet by default, CSV on request

The default output is **Parquet** — much smaller to store than CSV, and the
source of truth. Parquet isn't human-readable, so there are two ways to get CSV:

```bash
# 1. Write CSV directly instead of Parquet:
CHIP_ARCH=wormhole pytest accuracy/test_sfpu_accuracy.py --op=Reciprocal --csv

# 2. Convert an existing Parquet file to CSV (no hardware run needed):
python -m accuracy.to_csv accuracy/_csv_output/wh/exp.parquet   # one file
python -m accuracy.to_csv accuracy/_csv_output/wh               # whole dir
```

Both CSV paths produce identical output: floats use `%.9g` (lossless for the
float32-or-smaller data), and booleans render as `T`/`F`.

### Columns

| group | columns |
|-------|---------|
| identity | `op`, `input_format`, `output_format`, `chip_arch`, `distribution`, `intervals`, `seed` |
| data | `sample_index`, `test_value`, `golden_result`, `hardware_result` |
| config | `approx_mode`, `fast_mode`, `dest_acc` (0/1) |
| metrics | `signed_error`, `rel_error`, `signed_ulp_error`, `is_finite_hw`, `is_finite_golden` |

Rows are sorted ascending by `test_value` within each (format, config) block.
