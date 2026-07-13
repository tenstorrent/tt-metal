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
# one op
CHIP_ARCH=wormhole pytest accuracy/test_sfpu_accuracy.py -k Reciprocal -s

# runs every op in the given test file
CHIP_ARCH=wormhole pytest accuracy/test_examples_accuracy.py -s
```

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
CHIP_ARCH=wormhole pytest accuracy/test_sfpu_accuracy.py -k Reciprocal --csv

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
