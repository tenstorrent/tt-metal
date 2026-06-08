# SFPU Accuracy Harness — How to use

Sweeps SFPU ops across input domains/formats/configs on hardware and exports
**one CSV per op** of per-element golden-vs-hardware accuracy data (for a
downstream dashboard). Spec: `docs/superpowers/specs/2026-06-05-sfpu-accuracy-csv-design.md`.

## Layout

```
helpers/accuracy_metrics.py        pure error math (abs/signed/rel/ULP, finite flags). No hardware.
accuracy/
    accuracy_harness.py            run_case() + build_sweep_spec + shard/merge + CSV schema
    conftest.py                    clears shards at start, merges into per-op CSVs at end
    test_sfpu_accuracy.py          the mega sweep (18 transcendental ops x formats x configs)
    test_examples_accuracy.py      copy-me single-op examples
    tests/                         off-device unit tests for the pure parts
    _csv_output/                   OUTPUT (gitignored): {op}.csv + _shards/
```

## Running on hardware

These are ordinary pytest tests. You can run them two ways.

### Plain pytest (simplest; good for one op / local poking)

From `tests/python_tests/`, set the device arch via the `CHIP_ARCH` env var and
run like any other test. Default build mode is compile+execute, so no special
flags are needed:

```bash
CHIP_ARCH=wormhole pytest accuracy/test_sfpu_accuracy.py -k "Exp and not Exp2" -s
CHIP_ARCH=wormhole pytest accuracy/test_examples_accuracy.py -s
```

The conftest still clears shards and merges per-op CSVs at the end. (`-k` filters
by op name in the test id; use boolean form to disambiguate `Exp`/`Exp2`/`Log`/`Log1p`.)

### The run_test.sh wrapper (preferred for big sweeps / CI)

The wrapper is NOT required — it's a convenience/safety layer that adds: (a) a
two-phase split (compile all variants in parallel, then run) — much faster for
large sweeps; (b) a device lock so two runs don't fight over the chip; (c) hang
detection + auto-reset. Worth it for the full ~1400-variant corpus. Run from the
tt-llk root (`tt_metal/tt-llk/`):

```bash
# Full sweep — all ops, all variants (~a few min; large CSVs)
./.claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test accuracy/test_sfpu_accuracy.py --maxfail 5000

# One op
./.claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test accuracy/test_sfpu_accuracy.py --k "Exp and not Exp2"
```

Equivalent via skill: `/run-test accuracy/test_sfpu_accuracy.py --k "Exp"`.
`--maxfail 5000` keeps a corpus run going if one variant fails.

## Output

After a run, `conftest.py` merges shards into one CSV per op:

```
accuracy/_csv_output/exp.csv          one row per swept input x, all formats/configs stacked
accuracy/_csv_output/_shards/         per-variant intermediates (ignore)
```

Each run **clears shards first** and rewrites per-op CSVs for **only the ops in
that run** (run `--k Exp` → only `exp.csv` is (re)written; other op CSVs are
left untouched).

### CSV columns (22)

```
identity : op, input_format, output_format, chip_arch, distribution, intervals, variant_name, seed
data     : sample_index, x, golden, hw
config   : approx_mode, fast_mode, dest_acc          (0/1)
metrics  : abs_error, signed_error, rel_error, signed_ulp_error, abs_ulp_error, is_finite_hw, is_finite_golden
```

Rows are sorted ascending by `x` within each (format, config) block. `intervals`
records the swept domain (e.g. `[(-100.0, 80.0)]` or multi-band for reciprocal).

### Reading the metrics — important caveats

- **ULP is meaningful only for same-precision 16-bit output** (`Float16_b`,
  `Float16`). For `Float32` output, ULP explodes by construction (SFPU computes
  at ~16-bit internal precision, ULP measured on the fp32 grid) — use
  `abs_error`/`rel_error` there.
- **Block-float output (`Bfp8_b`/`Bfp4_b`): ULP columns are `NaN`** (no torch
  ULP). `abs_error`/`rel_error`/finite flags are still populated.
- **Big ULP near where golden≈0** (e.g. log at x≈1, atanh at x≈±1) is usually a
  normalization artifact (tiny error ÷ tiny local ULP), not a real defect —
  check `abs_error` to tell artifact from genuine error.
- `signed_ulp_error`/`abs_ulp_error`/`rel_error` are `NaN` (empty in CSV) wherever
  golden==0 or a value is non-finite.

## Off-device unit tests (no hardware)

The pure logic (metrics, sweep-spec, shard/merge) is unit-tested. From
`tests/python_tests/`:

```bash
/opt/venv/bin/pytest accuracy/tests/ helpers/tests/test_accuracy_metrics.py -q
```

## Add your own focused test

Copy a block in `test_examples_accuracy.py`:

```python
@pytest.mark.nightly
def test_accuracy_tanh_f16():
    run_case(
        MathOperation.Tanh,
        InputOutputFormat(DataFormat.Float16, DataFormat.Float16),
        ApproximationMode.No, FastMode.No, DestAccumulation.Yes,
    )
```

It auto-contributes rows to `tanh.csv` via the same shard→merge pipeline.
`run_case` sweeps the op's safe domain (from `sfpu_domains.for_op`, clipped by
`exclude_undefined`) as a deterministic ramp, runs golden + hardware, computes
metrics, and writes the shard. It sanity-asserts only (length, not-all-NaN,
not-all-zero) — no ULP thresholds.

## Analyzing a CSV (pandas, in /opt/venv)

```python
import pandas as pd, numpy as np
df = pd.read_csv("accuracy/_csv_output/silu.csv")

# canonical bf16 view, worst/mean ULP + exact-match %
v = df[(df.input_format=="Float16_b")&(df.output_format=="Float16_b")&(df.approx_mode==0)]
print(v.abs_ulp_error.max(), v.abs_ulp_error.mean(), 100*(v.abs_error==0).mean())

# where does the error live? bin |ULP| over x
v = v.assign(xbin=pd.cut(v.x, 12))
print(v.groupby("xbin", observed=True).abs_ulp_error.mean())
```

## Choosing which formats / ops / configs to test

All three are plain lists in `test_sfpu_accuracy.py`:

- **Formats:** `FORMATS = input_output_formats([...])` builds the cartesian
  input×output of whatever you list (e.g. 3 formats → 9 pairs). Add/remove a
  `DataFormat` to widen/narrow. `input_output_formats([...], same=True)` gives
  only same-in/out pairs.
- **Ops:** `TRANSCENDENTAL_OPS`.
- **Configs:** the `approx`/`fast`/`dest` lists in the `ACCURACY_PARAMS`
  comprehension (`fast` varies only for the fast-mode-capable ops).

## Tuning size

- **Sweep density:** `DEFAULT_SWEEP_POINTS` in `accuracy_harness.py` (default
  2048 → `input_dimensions = [32, 64]`). Lower it to shrink CSVs.

## Gotchas

- A full corpus is large (2048 pts × many variants × ops) — gitignored. Cursor's
  CSV preview extension chokes on tens-of-MB files; preview a small slice or use
  pandas / Rainbow CSV.
- `test_sfpu_plot.py` (the rich 6-panel plotter) needs `matplotlib`, which is in
  `/opt/venv` but **not** the test-runner venv — run/plot it with
  `/opt/venv/bin/python`, or plot straight from the CSVs.
- The mega test is `@pytest.mark.nightly`; running the file directly still
  executes it (no `-m "not nightly"` default in `pytest.ini`).
```
