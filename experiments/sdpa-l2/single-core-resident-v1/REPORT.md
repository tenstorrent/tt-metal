# Single-core resident-input SDPA, Q256/K512

September 14, 2026. **Main BF16 streaming reaches 72.1% of nominal HiFi2
peak. Compensated FAST reaches 53.1%, taking 35.9% longer in this compute loop.
ACCURATE reaches 55.0% of HiFi4 peak.** The earlier 1.9% full-operation FAST
overhead must not be treated as its intrinsic compute cost.

## Measured throughput

One Blackhole P100A on yyzo-bh-08, reservation 219338. Logical compute core
(0,0), physical (1,2), 1350 MHz profile clock; device compute grid 11x10.
Noncausal, D=128, Q chunk=256 and K chunk=512 for every version. BF16 inputs
and output, no input preprocessing. Each attention calculation processes
512 K chunks (logical KV length 262,144); the resident inputs are reused.

| Version | Useful TFLOP/s/core | Amortized us per Q/K block | Utilization vs HiFi2 peak | Utilization vs its own matmul fidelity peak |
| --- | ---: | ---: | ---: | ---: |
| Main BF16 streaming, HiFi2 | **1.9942** | 33.652 | **72.13%** | 72.13% |
| FAST: compensated BF16 streaming, HiFi2 | **1.4677** | 45.723 | **53.09%** | 53.09% |
| ACCURATE: full HiFi4, FP32 streaming | **0.7598** | 88.321 | **27.48%** | **54.96%** |

These are uninstrumented blocking trace-replay rates, cross-checked against
device cycles below. Useful FLOPs count only QK and PV:
`4 * 256 * 512 * 128 = 67,108,864` per Q/K block. Compensation, softmax,
normalization and auxiliary denominator work are not counted as useful FLOPs.

Nominal per-core peaks at 1.35 GHz are 2.7648 TFLOP/s for two fidelity phases
and 1.3824 TFLOP/s for four. The matrix instruction computes 8x16x16 MACs
per phase; one issue/cycle gives 4096/phase-count useful FLOPs/cycle.
The local tt-isa-documentation MVMUL/MatrixUnit descriptions and Blackhole
LLK matmul phase loops provide the instruction model. This is a nominal
issue-rate ceiling, not a separately measured matmul calibration.

The user's ~330 TFLOP/s chip figure is consistent with **120 cores** at this
clock: 120*2.7648 =331.776 TFLOP/s. This card exposes **110 compute cores**, so
the corresponding HiFi2 nominal total is 304.128 TFLOP/s. Multiplying these
single-core rates by core count gives the following *ideal scaling estimates*,
not measured chip throughput:

| Version | 110-core equivalent TFLOP/s | 120-core equivalent TFLOP/s |
| --- | ---: | ---: |
| Main | 219.36 | 239.30 |
| FAST | 161.45 | 176.13 |
| ACCURATE | 83.58 | 91.18 |

No conclusion about full-chip clocks, power limits, DRAM, NoC scaling or
Galaxy performance follows from multiplying the single-core rate.

## What “no-DM” means here

The harness calls the actual `sdpa_standard_v2` compute loop, not a standalone
matmul or an arithmetic-cost model. Q/K/V are loaded into L1 once, before
publishing Q. The K reader transposes the tile grid exactly as the ordinary
SDPA reader does; within-tile transpose remains in the compute kernel.

During the repeated attention calculations, the reader only recycles CB
availability, with **no recurring DRAM/NoC input transfer or L1 data copy**.
The writer drains results without transferring them, except for the last
256x128 output block used for correctness. Internal score/state L1 traffic,
unpack/pack, reductions, online updates and all compute synchronization remain.
This is not a register-only or zero-internal-data-movement benchmark.

BF16 uses two Q slots and double-buffered K/V, unchanged between main and FAST.
ACCURATE uses two Q slots and one K/V slot to fit its wider score/state buffers,
matching the prior Q256 FP32 buffering choice. This difference does not remove
any compute step or introduce repeated input transfers. CB allocations are
1,169,408 /1,333,248 /1,335,296 bytes for main/FAST/ACCURATE respectively.

Main uses compute_common.hpp, compute_streaming.hpp and the SFPU header from
base commit `2ba6fc2339d53300ae87c5202f335ef56492cfb3`. FAST and ACCURATE use
isolated copies of the retained improved headers. The only edit to each
copied common header redirects its SFPU include to its matching snapshot.
No production sources or dispatch guards were changed in this experiment.

**Both improved production guards exclude Q256.** The wrapper explicitly
instantiates those algorithms at Q256, sets their required defines and CB
formats, and cannot silently fall back. FAST enables both high/low state
compensations. ACCURATE enables mode4, HiFi4 QK/PV, FP32 DST/state, full-FP32
L1 subtraction, BF16 max CB, the new load-macro exp refiner and the exact
zero/one denominator phases 0+2. This is a benchmark-only extension, not
broad qualification of Q256 production dispatch.

## Timing checks

Fresh process per case, 20 warmup and 10 measured blocking trace replays.
Compilation and host transfers are outside the timed replay. The one-time
device preload and final output drain remain in trace wall time, but are
amortized across thousands of Q/K blocks. Device profiling starts the
`SDPA_RESIDENT` zone after initial Q/scaler readiness.

| Version | 8 Q repetitions: ms / TFLOP/s | 16 Q repetitions: ms / TFLOP/s | Reverse-order 16 repetitions: ms |
| --- | ---: | ---: | ---: |
| Main | 137.857 /1.99394 | 275.678 /1.99420 | 275.683 |
| FAST | 187.297 /1.46761 | 374.564 /1.46772 | 374.565 |
| ACCURATE | 361.774 /0.75981 | 723.530 /0.75982 | 723.519 |

Doubling repetitions changes throughput by less than 0.02%. Reverse order
confirms the comparison. All these rates use the same normal seed1236.
No power or clock controls were changed.

Separate 8-repetition instrumented runs, using MATH-thread resident-zone cycles:

| Version | Resident cycles | Resident ms at 1350 MHz | TFLOP/s/core |
| --- | ---: | ---: | ---: |
| Main | 185,989,630 | 137.770 | 1.99519 |
| FAST | 252,935,912 | 187.360 | 1.46711 |
| ACCURATE | 488,026,374 | 361.501 | 0.76038 |

All agree with uninstrumented rates within 0.1%. Counter reference durations
are below 2^32 and checked against kernel intervals; no wrap was accepted.
Only one compute core contributes. `analyze.py` reads the saved device CSVs.

## Where the extra cost appears

| Activity counter | Main | FAST | ACCURATE |
| --- | ---: | ---: | ---: |
| FPU active | 77.41% | 57.75% | 59.05% |
| SFPU active | 23.63% | 40.86% | 25.65% |
| Both active | 21.02% | 18.31% | 19.91% |
| Neither active | 19.98% | 19.70% | 35.22% |

These are busy-cycle fractions, **not useful-FLOP utilization**. FAST's
SFPU-only share increases from about 2.61% to 22.55%, consistent with exposed
compensation work rather than a nearly free update. Its total time increases
35.87%, and throughput decreases 26.40%, relative to main at identical chunks.

This establishes the intrinsic penalty in this resident-input workload.
It does not establish that the entire difference from the historical 1.9%
full-operation penalty is caused only by data movement: that earlier
comparison used Q128, not Q256, and a different board/run. Repeated K/V also
stabilizes the running maximum after the first chunk. A controlled full-DM
Q256 comparison would be needed to isolate DM's contribution quantitatively.
Nevertheless, the small earlier overhead is not a safe estimate for a
compute-bound FAST implementation. The compensation loop deserves more
optimization before describing FAST as near-main cost in general.

ACCURATE has a separate fidelity cost: HiFi4's useful-matmul peak is half
HiFi2's. Its ~55% of HiFi4 peak should not be confused with ~55% of the
~330 TFLOP/s HiFi2 chip figure. Its remaining neither-unit-active fraction
also leaves substantial internal scheduling/transfer work to investigate.

## Correctness and artifacts

Every successful uninstrumented case checks all 256x128 final output elements
for finiteness and exact final-trace/nontrace equality. Reference is dense
FP64 attention over the original BF16 resident inputs. Repeating an identical
K/V chunk preserves the exact normalized reference; no reference needs a
materialized 256K tensor. The 8/16/reverse runs have matching complete output
hashes for each version.

For this repeated-input diagnostic, normal L2 is 20.4106% /2.44567% /0.177617%
for main/FAST/ACCURATE. Held-out seed1237 at 64 K chunks gives
4.06150% /2.62952% /0.178880%. Constant V=1 at 512 K chunks gives
18.7177% /0.690534% /approximately zero. Constant-V PCC is not meaningful
because the reference differs from a constant only by FP64 roundoff.
These are harness checks, not substitutes for general-input qualification or
the earlier full-prefill L2 values. ACCURATE was explicitly gated below 0.5%.

The first smoke attempt raced the K-layout correction and failed its L2 check;
`smoke-main-v1.log` is retained as failed, not used for performance. Corrected
K tile-grid mapping is in all successful runs. There were no device hangs or
card resets in this experiment. All three kernels compiled and ran, the host
Release build/install succeeded, and C++/Python harnesses were formatted.
`git diff --check` and shell syntax checks pass. Per-case provenance records
include compile defines, CB specs, arguments and source hashes.

Reproduce in the configured remote checkout:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
python_env/bin/python experiments/sdpa-l2/single-core-resident-v1/run.py \
  --mode main --q-repeats 16 --k-chunks 512 --warmup 20 --iters 10 --label fresh-main
# Repeat in fresh processes with --mode fast and --mode accurate and fresh labels.
bash experiments/sdpa-l2/single-core-resident-v1/profile.sh fresh-profile
```

Remote checkout: `/localdev/cglagovich/tt-metal-blackhole-20260908`.
Raw logs, JSON results, snapshots and profiler CSVs are also copied into this
local experiment directory. No commit, PR, production guard change or
optimization of the measured implementations was made during this task.
