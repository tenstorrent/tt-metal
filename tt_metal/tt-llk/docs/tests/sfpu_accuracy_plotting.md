# SFPU Accuracy Plotting Harness

`test_sfpu_plot.py` is a table-driven harness for checking the accuracy of a single SFPU op on hardware. For each configured op, it sweeps an input range, runs it on the device, compares the result against a golden torch reference, and writes a multi-panel accuracy plot plus a text stats summary.

Use it when you want to see how an SFPU op behaves across its input domain: where it loses precision, whether it stays monotonic, and how big the worst-case error is.

There are two files, one per device family. They share all plotting and statistics code, so the figures look the same:

- `test_sfpu_plot.py` — Wormhole and Blackhole, on silicon.
- `quasar/test_sfpu_plot_quasar.py` — Quasar, on the simulator. See [Quasar](#quasar) below for what differs.

## Quick start

Run from the `tests/` directory.

```bash
# Wormhole / Blackhole (CHIP_ARCH is auto-detected from the device)
pytest python_tests/test_sfpu_plot.py -k Log -s      # one op
pytest python_tests/test_sfpu_plot.py -s             # every configured op

# Quasar (simulator only — both CHIP_ARCH and --run-simulator are required)
CHIP_ARCH=quasar pytest --run-simulator python_tests/quasar/test_sfpu_plot_quasar.py -k Exp -s
CHIP_ARCH=quasar pytest --run-simulator python_tests/quasar/test_sfpu_plot_quasar.py -s
```

- `-k <Op>` selects cases by name, for example `Log`, `Sqrt`, or `Reciprocal`. It matches every case whose id contains the text, so `-k Exp` also runs `Exp-fp32-approx`.
- `-s` lets the stats summary print to your terminal.
- `CHIP_ARCH` (`blackhole`, `wormhole`, or `quasar`) selects the target device.

Each case writes its plot to `python_tests/_plot_output/<arch>/sfpu_<id>.png`, where `<arch>` is `wh`, `bh`, or `qsr`, and asserts that the hardware result matches golden. The folder is next to the test files regardless of where you launch pytest from, and it is git-ignored. Keeping one folder per arch means a Wormhole run and a Blackhole run of the same case do not overwrite each other.

## Adding a test

Append one `Case(...)` to the `CASES` list. That is the whole workflow. In `test_sfpu_plot.py` the list is near the bottom of the file; in the Quasar file it comes right after the `Case` dataclass.

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

`StimuliSpec.ulp_sweep(low, high)` tests *every* representable value in the range instead of sampling it. Supported for BF16, FP16, and FP32.

```python
Case(op=MathOperation.Reciprocal, spec=StimuliSpec.ulp_sweep(low=0.01, high=10.0))
Case(op=MathOperation.Reciprocal, spec=StimuliSpec.ulp_sweep(low=1.0, high=2.0), fmt=FP32)
```

`input_dimensions` is set automatically — leave it unset. A range small enough for one run (all of BF16/FP16, or a narrow FP32 range) is swept in a single pass. A larger range — typically FP32, whose grid is far denser — is automatically split into batches and joined, so the sweep is always exhaustive. Wide FP32 ranges can therefore take a while (many batches).

How much fits in one run depends on the device family:

- Wormhole / Blackhole: a run is bounded by L1, 64 tiles (65,536 values).
- Quasar: a run is bounded by Dest, 8 tiles (8,192 values) for BF16/FP16 and 4 tiles (4,096 values) for FP32. Quasar sweeps therefore batch sooner and take more runs for the same range.

A sweep may take at most `_MAX_SWEEP_BATCHES` device runs (512, set in each harness file). A range that needs more is rejected up front with an error saying how far to narrow it. Only wide FP32 ranges can hit this; raise the constant if you deliberately want a longer sweep.

## Optional Case fields

Sensible defaults cover the common cases, so override only when needed.

- `expect_pass` — set to `False` to keep the run green while exploring a known-inaccurate op.
- `name` — custom test ID, which is also the plot filename and what `-k` matches. The default is `<Op>-<fmt>`; the Quasar file adds `-approx` when `approx_mode` is on, for example `Exp-fp32-approx`.
- `approx_mode` — the default is the accurate path. Set to `ApproximationMode.Yes` to test the fast, less accurate SFPU path.
- `clamp_negative` — enable the kernel's negative-input clamp. Wormhole/Blackhole only.
- `dest_acc` and `unpack_to_dest` — override the format-derived accumulator defaults. On Quasar only `dest_acc` exists; the data route, and with it `unpack_to_dest`, is decided by the format resolver.
- `dest_sync` and `implied_math_format` — Quasar-only kernel knobs. Defaults are `DestSync.Half` and `ImpliedMathFormat.Yes`, the same values the functional suite pins in perf mode.
- `input_dimensions` — set how many points sample the domain (see below).
- `batch_tiles` — tiles per batch for a large `ulp_sweep`. Auto-chosen if you don't set it; only matters when a range is too big for one run. Must be between 1 and the single-run size (64 on Wormhole/Blackhole, the Dest capacity on Quasar).
- `extra_undefined_ranges` — override the red undefined-domain shading on the plot.

### How many points to sample (`input_dimensions`)

`input_dimensions` is the input tensor shape; its element count is how many points sample the domain. The default `[32, 32]` is one tile = 1024 points.

1024 points cover BF16/FP16 well, but sample FP32's much finer grid sparsely. To sample FP32 more densely, add points using whole tiles — each dimension must be a multiple of 32, so use `[32, 32*K]` for `1024*K` points.

```python
# Denser FP32 sweep: 4 tiles = 4096 points.
Case(op=MathOperation.Sqrt, spec=StimuliSpec.ramp(low=0.0, high=100.0), fmt=FP32,
     input_dimensions=[32, 32 * 4])
```

On Quasar the whole run has to fit in Dest, so `input_dimensions` is capped at 8 tiles for BF16/FP16 and 4 tiles for FP32 (at `DestSync.Half`; double that at `DestSync.Full`). A larger value fails with an error before anything runs. The Quasar file already uses 4 tiles for its FP32 cases.

## Quasar

The Quasar harness produces the same plots but the device side is different. What you should know when reading or adding Quasar cases:

- **Kernel.** Every case runs through `sources/quasar/eltwise_unary_sfpu_quasar_test.cpp`, the kernel the functional suite (`quasar/test_eltwise_unary_sfpu_quasar.py`) drives.
- **Data route.** Quasar cannot take an arbitrary input/output format pair. `resolve_quasar_sfpu_variant` picks the Dest format, whether the input is unpacked straight into Dest or goes through the FPU, and the packer conversion. A case Quasar cannot execute fails with a clear error instead of running the wrong route. The route is printed in the log and in the plot header.
- **Case ids.** Ops with an approximate kernel are configured twice, exact and `-approx`, in each format. For a 16-bit Dest, `Exp-bf16` and `Exp-bf16-approx` are the same kernel path and their plots are identical.
- **Batching.** A run holds at most the Dest capacity, so exhaustive sweeps batch at 8 tiles (16-bit) or 4 tiles (FP32) rather than 64.

## Reading the output

The plot title names the op, the format and the device family (`[Wormhole]`, `[Blackhole]`, `[Quasar]`). Under it, a parameter line records the case id, input and output format, `dest_acc`, approx mode, the sampling distribution and the point count, so a saved image is self-describing. Quasar plots also spell out the data route (input → Dest → SFPU → pack → output) and the `dest_sync` / `implied_math_format` settings.

For details on how to read each plot, see [SFPU Accuracy Plots & Metrics](sfpu_accuracy_plots_metrics.md).
