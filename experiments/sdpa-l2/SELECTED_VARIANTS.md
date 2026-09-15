# Selected SDPA options — research checkpoint

Locked September 14, 2026 following the four-point accuracy sweep. These are
the **four current options** for the project. Other experiments in this tree
are historical evidence or ablations, not additional selected options.

| Stable option | Harness mode | QK / PV fidelity | Destination | Recurrent state | Exponential / subtraction |
| --- | --- | --- | --- | --- | --- |
| `main_bf16` | `main` | HiFi2 / HiFi2 | BF16 | Main BF16 | Main approximation and subtraction |
| `fast_bf16` | `fast` | HiFi2 / HiFi2 | BF16 | Compensated BF16 numerator **and** denominator | Improved approximation; retained paired-denominator schedule |
| `balanced_fp32` | `qk4_pv2` | HiFi4 / HiFi2 | FP32 | FP32 | Improved biased approximation, effective-weight denominator, cheaper FPU subtraction |
| `accurate_fp32` | `accurate` | HiFi4 / HiFi4 | FP32 | FP32 | Unbiased refined approximation, full-FP32 subtraction |

All four are streaming implementations and consume original BF16 Q/K/V,
without Q preprocessing. Full-FP32 subtraction does **not** mean the final
output is FP32; all reported outputs are BF16.

## Evidence behind the selection

Matched Q256/K512/D128 single-core resident-input throughput is approximately
**1.994 / 1.591 / 0.935 / 0.760 TFLOP/s/core**, in the table's order. This is
not full-chip throughput and excludes recurring input transfers, not internal
score/state traffic. See [Pareto data](pareto-v1/measurements.json) and
[measurement contract](pareto-v1/README.md).

The distinct-input Q128/K512 sweep covers 4K/32K/256K keys, two new seeds and
five regular distributions, plus four separate stress cases. Balanced and
accurate are below 0.5% aggregate L2 on 30/30 regular cases each, with ranges
0.175–0.435% and 0.131–0.279%. FAST remains a few-percent option and main BF16
has substantial long-context drift. Neither FP32 option is universally below
0.5% on common-mode stress inputs. See the [full accuracy report](frontier-accuracy-v1/REPORT.md).

## Canonical source and runnable tests

`selected_variants.json` pins each option's frozen compute headers and SFPU
helpers by SHA256, and records its benchmark record and harness mode.
Validate those pins and the paired accuracy results without a device:

```bash
python3 experiments/sdpa-l2/validate_selected_variants.py
```

The common distinct-input runner is
`frontier-accuracy-v1/run.py`. It runs exactly `main`, `fast`, `qk4_pv2`, and
`accurate` with no fallback. Its FAST source matches production bit-for-bit
on the recorded 32K smoke case. QK4/PV2 remains an isolated experimental
implementation, not a newly exposed production API option.

In the configured Blackhole container, using fresh result labels:

```bash
python_env/bin/python experiments/sdpa-l2/frontier-accuracy-v1/run.py \
  --smoke --label selected-options-smoke
python_env/bin/python experiments/sdpa-l2/frontier-accuracy-v1/run.py \
  --label selected-options-sweep
```

No-DM Q256 benchmarks use the existing runners:

- Main: `single-core-resident-v1/run.py --mode main`.
- FAST: `bf16-denom-pair-v3/run.py --mode fast`.
- Balanced and accurate: `hybrid-mixed-v1/run.py --mode qk4_pv2` or `--mode accurate`.

Append `--q-repeats 16 --k-chunks 512 --warmup 20 --iters 10 --label FRESH_LABEL`.
Do not use the older hybrid experiment's Q256 distinct-input FAST path: its
failed live-input test is documented and is not part of the accepted evidence.

## What is locked, and what is not

This is a reproducible research checkpoint, **not a production-ready PR** or
a promise of new API/default behavior. Existing experimental dispatch guards
remain as measured. In particular, the Q128 harness forces FAST at 4K even
though the current production guard only enables its compensation from 32K.
Q256 no-DM wrappers do not imply production dispatch support for Q256.

The QK2/PV2 cheaper path, BF16-compute/FP32-state hybrid, mixed full-subtraction
variant, and older scheduling revisions remain available for investigation.
They are not in the current selected set. Changing the four selected numeric
configurations should require explicit re-evaluation and an update to the
manifest and evidence; this checkpoint does not delete historical experiments.

## Verification at checkpoint

- Host build passed on the allocated Blackhole machine with
  `CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install`.
- All selected device kernels JIT-built and executed in the latest sweep.
- All 136 executions reproduced bit-for-bit in a complete second sweep.
- New harness Black/clang-format checks and `git diff --check` passed.

Source, reports, compact JSON/JSONL results, provenance and the final plot are
versioned. Large tensor dumps, profiler captures, build/runtime logs, browser
preview files and caches remain local and ignored. Historical reports may
refer to those local-only artifacts; the selected results above do not depend
on them to recover the numeric configurations or reported measurements.
