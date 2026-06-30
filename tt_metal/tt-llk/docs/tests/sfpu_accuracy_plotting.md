# SFPU Accuracy Plotting Harness

`test_sfpu_plot.py` is a table-driven harness for checking the accuracy of a single SFPU op on hardware. For each configured op, it sweeps an input range, runs it on the device, compares the result against a golden torch reference, and writes a multi-panel accuracy plot plus a text stats summary.

Use it when you want to see how an SFPU op behaves across its input domain: where it loses precision, whether it stays monotonic, and how big the worst-case error is.

## Quick start

Run from the `tests/` directory.

```bash
# Run one op
pytest python_tests/test_sfpu_plot.py -k Log -s

# Run every configured op
pytest python_tests/test_sfpu_plot.py -s
```

- `-k <Op>` selects a single case by op name, for example `Log`, `Sqrt`, or `Reciprocal`.
- `-s` lets the stats summary print to your terminal.
- `CHIP_ARCH` (`blackhole`, `wormhole`, or `quasar`) selects the target device.

Each case writes its plot to `_plot_output/sfpu_<id>.png` and asserts that the hardware result matches golden.

## Adding a test

Append one `Case(...)` to the `CASES` list near the bottom of the file. That is the whole workflow.

Example cases:

```python
Case(op=MathOperation.Log, spec=StimuliSpec.ramp(low=0.5, high=10.0))
Case(op=MathOperation.Sqrt, spec=StimuliSpec.ramp(low=0.0, high=100.0), fmt=FP32)
```

Two fields are required:

- `op` — the SFPU op to test, a `MathOperation` such as `MathOperation.Exp`.
- `spec` — the input domain to sweep, a `StimuliSpec` such as `StimuliSpec.ramp(low=-10, high=10)`.

## Picking the format

`fmt` chooses the numeric format. The default is `BF16`.

- `BF16` — bfloat16 input and output.
- `FP16` — float16 input and output.
- `FP32` — float32 input and output.

## Choosing the input domain

`spec` sets the input range to sweep — `StimuliSpec.ramp(low, high)` for an evenly spaced sweep, or `StimuliSpec.uniform(low, high)` for random sampling.

To sweep disjoint bands instead of a single range, for example either side of a singularity, pass `intervals`:

```python
Case(op=MathOperation.Reciprocal, spec=StimuliSpec.uniform(intervals=[(-10.0, -0.01), (0.01, 10.0)]))
```

### Exhaustive sweep (`ulp_sweep`)

For BF16 or FP16, `StimuliSpec.ulp_sweep(low, high)` tests *every* representable value in the range instead of sampling it.

```python
Case(op=MathOperation.Reciprocal, spec=StimuliSpec.ulp_sweep(low=0.01, high=10.0))
```

`input_dimensions` is set automatically and should be left unset. Coverage is limited to 8 tiles (8192 values); if the range exceeds this, only the lowest values are swept and a warning is logged. FP32 is not currently supported.

## Optional Case fields

Sensible defaults cover the common cases, so override only when needed.

- `expect_pass` — set to `False` to keep the run green while exploring a known-inaccurate op.
- `name` — custom test ID or output filename. The default is `<Op>-<fmt>`.
- `approx_mode`
- `clamp_negative` — enable the kernel's negative-input clamp.
- `dest_acc` and `unpack_to_dest` — override the format-derived accumulator defaults.
- `input_dimensions` — set how many points sample the domain (see below).
- `extra_undefined_ranges` — override the red undefined-domain shading on the plot.

### How many points to sample (`input_dimensions`)

`input_dimensions` is the input tensor shape; its element count is how many points sample the domain. The default `[32, 32]` is one tile = 1024 points.

1024 points cover BF16/FP16 well, but sample FP32's much finer grid sparsely. To sample FP32 more densely, add points using whole tiles — each dimension must be a multiple of 32, so use `[32, 32*K]` for `1024*K` points.

```python
# Denser FP32 sweep: 4 tiles = 4096 points.
Case(op=MathOperation.Sqrt, spec=StimuliSpec.ramp(low=0.0, high=100.0), fmt=FP32,
     input_dimensions=[32, 32 * 4])
```

## Reading the output

For details on how to read each plot, see SFPU Accuracy Plots & Metrics.

## Logging

Console and file output go through the shared loguru logger as described in [LOGGING.md](../../tests/LOGGING.md).

The summary always lands in `test_run.log`. To also see it live on the terminal, pass a level:

```bash
pytest python_tests/test_sfpu_plot.py -k Log --logging-level=INFO
```
