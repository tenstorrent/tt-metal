# Calibration of the matmul default config selection

The selector's policies (`ttnn/cpp/ttnn/operations/matmul/device/config/factory_blocking_source.hpp`) hold two kinds
of values:

- **Chosen** (`struct Limits`): design limits set by judgment, each with its reason in the source.
- **Tuned** (`struct Tuned`): fitted to device timings. Every Tuned value is computed by a fit from a sweep in this
  folder, and `test_calibration.py` fails if the value in the source differs from what its fit computes from the
  committed data, or if a Tuned field has no entry in `tuned.py`.

Fits use systematic sweeps that target one decision each. The OOB benchmark suite (`../cases.csv`, `../run.py`) is
the acceptance test for the whole selection, not fitting data.

## Layout

| Path | Contents |
|---|---|
| `tuned.py` | The registry: every Tuned field, the sweep and fit that compute it, or a note while it is pending |
| `<sweep>.py` | One sweep: its grid and variants, `fit(path)`, and `--report` |
| `data/<arch>/<sweep>.csv` | The sweep's raw device timings on that architecture |
| `test_calibration.py` | Device-free check of the registry and of every fitted value |

## What a sweep measures

Device kernel time (the median of 10 timed calls after 2 warmup calls, from the device profiler) of every variant at
every grid point, as explicit program configs with the compute config pinned (bf16 HiFi2, packer L1 accumulation on
unless the sweep varies it), and the v2 default selection at the same point (a check that the sweep's replay of the
rule matches the selector). A fit scores each candidate value by its regret at every grid point: the time of the variant
the rule picks over the fastest variant timed there. It picks the lowest geomean regret, then the lowest worst-case
regret (minimax regret can't tell candidates apart when every rule of the form leaves the same worst point), and
reports the range of values within 0.3% of the best geomean.
Fits never reference another implementation: the selection is calibrated on its own measurements. Whether it
regresses against anything else (today the legacy selection, later a previous release) is the acceptance test's
question.

## Regenerating

```bash
source python_env/bin/activate
python tests/ttnn/unit_tests/benchmarks/matmul_oob/calibration/<sweep>.py            # sweep (resumable)
python tests/ttnn/unit_tests/benchmarks/matmul_oob/calibration/<sweep>.py --report   # fit and scores
pytest tests/ttnn/unit_tests/benchmarks/matmul_oob/calibration/test_calibration.py
```

Run the sweep after `ninja -C build_Release install`. On a new architecture, add `data/<arch>/` and, where its fit
differs, the value in that policy's `Params::for_arch`.

## Values

| Value | Decides | Sweep | Fitted value (range) |
|---|---|---|---|
| `HeuristicBlocking::Tuned::max_in0_block_w` | K depth cap | pending (2D K depth sweep) | 8, set on the OOB suite |
| 2D K floor (`deepen_to_legacy_k_depth`) | Deeper K for interleaved 2D in some cases | pending (2D K depth sweep) | borrowed from the legacy selection; to be re-derived or removed |
| `HeuristicBlocking::Tuned::large_block_tiles`, `large_block_in0_block_w` | Deeper K for large 2D blocks | pending (2D K depth sweep) | 64 and 16, set on 2D sweeps of suite cases |
| `HeuristicBlocking::Tuned::max_self_read_tiles_per_k_step` | K depth of the layouts that read an operand themselves | pending (self-read sweep) | 8, set on the OOB suite |
| `HeuristicFamily::Tuned::one_d_core_advantage` | 1D over 2D, mcast over Reuse | pending (family sweep) | 1.5, set on the OOB suite (1.25 to 2 alike) |

The pending values predate this folder: they were set on runs of the OOB suite whose data is not tracked.

## Parked studies

Sweeps whose rule is not in the selector, kept with their data for later.

| Sweep | Rule studied | Wormhole result | Why parked |
|---|---|---|---|
| `in1_out_block_split.py` | Split 1D in1-mcast output blocks while each keeps `out_block_h * Nt * Kt` >= W tile products | W = 192 (128 to 256 within 0.3%): regret geomean 1.059 against 1.091 unsplit, over 756 grid points | A small gain that fixes none of the regressions it was aimed at; fused-bias cases want splits the rule doesn't make |
