# Frontier compute optimization — 2026-09-18

Completed first compute-optimization sprint across all six frozen recipes.
All reported wins below have been measured and numerically checked on
Blackhole; nothing has been integrated into production yet.

## Results

Single-core, noncausal resident-input attention, fixed Q256/K512/D128. Useful FLOPs
count only QK and PV. These are **compute-only TFLOP/s/core**, not chip or
model throughput; input preprocessing is outside the timing.

| Variant | Frozen numerical recipe | Baseline TF/core | Optimized TF/core | Time reduction |
| --- | --- | ---: | ---: | ---: |
| D | FP32 destination/state, QK4/PV4, full-FP32 subtraction | 0.75952 | 0.81916 | 7.28% |
| C | FP32 destination/state, QK4/PV2, existing cheaper exp/subtraction | 0.93490 | 1.12546 | 16.93% |
| B | BF16 destination, HiFi2, compensated numerator and denominator | 1.59075 | 1.62996 | 2.405% |
| A | Original BF16 streaming, HiFi2, uncompensated | 1.99325 | 1.99724 | 0.199% |
| E | BF16 destination, LoFi, compensated state, BFP8 KV | 1.87590 | 1.91575 | 2.080% |
| G | BF16 destination, LoFi, compensated state, BFP4 KV | 1.87393 | 1.91551 | 2.171% |

Timing evidence: [C](fp32/C-final-abba-v1.json),
[D](fp32/D-final-abba-v1.json),
[B](bf16/B-block-reuse-fence-steady-01.json),
[A](bf16/A-final-steady-02.json),
[E](lowp/e-final-resident-v1.json), [G](lowp/g-final-resident-v1.json).
C/D use eight Q repeats; A/B/E/G use sixteen. Each uses 512 resident K
chunks. Within-variant paired comparisons use identical work. Compare
normalized TF/core across rows, not raw batch milliseconds.

The resident harness repeats the same KV payload. In particular, after the
first encounter it strongly exercises unchanged-max/identity-correction
branches. Real-input benefit depends on max-update frequency as well as
data movement. A small unchanged-dataflow two-core normal case measured
D 5.851→5.763 ms and C 4.777→4.313 ms; this was a correctness-oriented
check, not an interleaved full-chip benchmark or model speedup claim.

## What changed — and what did not

- **C/D:** batch recurrent-state transfers and independent numerator copies;
  defer unused subtraction setup on the existing exact-identity correction
  branch; reduce loop-control overhead in the same max-equality scan.
  C additionally schedules its unchanged cubic polynomial through load
  macros and hoists macro setup. Coefficients, MAD order and rounding stay
  unchanged. This is not a switch to D's polynomial.
- **B/E/G:** group numerator high/low planes for paired copy/pack, reuse the
  identical correction vector on adjacent broadcast columns, and remove
  one redundant synchronization already covered by correction-CB publication.
  The publication fence remains. Compensation arithmetic is unchanged.
- **A:** batch native approximate-exp replay across the existing eight DST
  tiles; same coefficients, arithmetic and stores. The gain is small.

No matmul fidelity, DST format, CB format/capacity, input-buffer depth,
Q/K chunk size, reduction arithmetic, reciprocal, or input preparation was
changed. A/B/C/D still take BF16 inputs without preprocessing. E/G still
round Q to seven bits in BF16; E uses five-bit RNE followed by native BFP8
KV packing, while G uses native BFP4 KV RNE with saturation. G is **not** an
ordinary BFP4 typecast. The canonical numerical authority remains
`flux2-frontier-v1/device_attention.py:recipe()` and `prepare()`.

The retained prototypes are qualified for this fixed geometry, not a
drop-in implementation for all production shapes. In particular, the
redundant-fence argument is scoped to the selected noncausal full-compensation
schedule; other geometries and masked/causal paths need their own review.

Independent cross-review found no new race in that inspected schedule, but
identified an integration guard requirement: today's restrictions are in
the experimental entrypoints, **not fully enforced in the generic candidate
header**. Its `pairs == 2` assertion permits other even head widths. Before
reuse in production, explicitly gate Blackhole/BF16-destination/noncausal
Q256/K512/D128 and supported mask/ring conditions, retaining the original
fence elsewhere until separately proved and qualified. Do not copy the
unconditional prototype fence removal into general dispatch.

Production reader/writer kernels are untouched. Resident timing preloads
input and removes recurring external data movement, but retains internal
L1/pack/unpack traffic and synchronization. Separate distinct-input checks
use the unchanged real dataflow. FP32 retains one KV slot; BF16 retains two.

## Numerical acceptance

Retained candidates target exact output-bit preservation, not merely equal
PCC or similar L2. Repeated resident KV establishes throughput only. Separate
qualification exercises genuinely distinct KV, changing maxima, multiple
distinct Q jobs per core, stress distributions and trace replay.

- C/D each pass eight distinct-KV cases (normal at 256K; seven stress cases
  at 32K) and four distinct-Q cases at Q2048/K8192 on two cores.
- B passes six odd-three-K-chunk cases at Q2048/K1536 and eight held-out
  distribution cases at Q2048/K8192, with four distinct Q jobs per core.
- E/G each pass seven Q2048/K8192 cases with eight distinct Q jobs/core,
  including common modes, outliers and constant V. All preparation and
  input/source immutability checks pass.
- A passes six initial Q2048/K8192 distinct-Q cases and eight held-out
  second-seed cases at Q1024/K1024, including two-K-chunk short-loop coverage.

These are finite qualification results, not a proof of universal bitwise
equivalence. Existing recipe weaknesses remain: the optimization does not
repair common-mode/outlier errors. For example, the C/D normal-256K L2
values remain 0.39054%/0.17989%, respectively; stress failures remain as
documented in the per-track reports.

Coordinator independently checked saved output hashes, equality flags,
preparation gates and useful-throughput arithmetic:
[C/D audit](audit-fp32-final.json),
[B/E/G audit](audit-compensated-final.json),
[A/B audit](audit-bf16-final.json),
[final held-out/profile-output audit](audit-final-closure.json).
These audits inspect recorded evidence; they are not extra hardware runs.

## Utilization and remaining opportunities

Separate device-cycle profiles distinguish useful matmul work from activity
counters. At 1350 MHz the relevant per-core roofs are 1.3824 TF for D,
1.8432 TF for equal-work QK4/PV2 C, 2.7648 TF for A/B and 5.5296 TF for
LoFi E/G. Comparing every recipe to the HiFi2 roof would be misleading.

| Variant | Useful own-roof utilization, before → after | Neither FPU nor SFPU active, before → after |
| --- | ---: | ---: |
| D | 54.99% → 59.22% | 35.23% → 30.18% |
| C | 50.73% → 61.07% | 37.45% → 30.67% |
| B | 57.56% → 59.02% | 19.80% → 18.27% |
| A | 72.14% → 72.37% | 20.00% → 19.85% |
| E | 33.92% → 34.60% | 36.84% → 36.23% |
| G | 33.89% → 34.65% | 37.04% → 36.29% |

C's larger gain includes less absolute SFPU work and fewer setup gaps;
D mainly benefits from state handling/setup. Neither reaches 80%.

A already overlaps most exp work with the FPU. Its batching reduces SFPU
activity, but largely removes work that was hidden; hence the small elapsed
time improvement. B's SFPU-active *percentage* rises slightly because total
time falls, even though absolute SFPU cycles decrease. Counter percentages
alone are not a work-reduction metric.

For E, the final profile has exactly unchanged FPU-active cycles. SFPU-active
cycles decrease by exactly 1,569,792: 384 eliminated replay instructions ×
511 recurrent K chunks × eight Q repeats. This independently corroborates
the correction-load optimization. G shows the identical absolute counter
reduction. Although LoFi matmuls are fast, compensation and dependent
state/pack/unpack scheduling remain expensive: E/G FPU activity is about
40.7%, while useful utilization is only about 34.6%. Some FPU work is
non-useful overhead under the original-attention-FLOP convention.

Remaining idle fractions are not automatically removable slack: they include
dependency waits and work outside the two arithmetic engines. Further gains
require identifying and restructuring critical-path state/pack/unpack
dependencies, not simply deleting waits. These experiments establish local
improvements, not global optimality.

## Rejected ideas and safety

Pack-width caching, several L1-mode/configuration hoists, and A's max-stat
copy batching were neutral or slower and are not retained. A cross-group
SFPU macro-setup cache failed G's distinct-input comparison and was rejected
outright; no numerical relaxation was accepted. Detailed raw failures remain
in the individual tracks.

Three independent agent tracks (C/D, A/B, E/G) shared logical device 0 under
the same exclusive lock used by safe pytest. Other chips were not run
concurrently because the recovery mechanism resets the reserved machine.
One device-initialization Ethernet-core failure required a coordinated reset
of the explicitly reserved PCI devices; exact matmul smoke passed afterward.
All final timing comparisons use fresh post-reset controls. Setup/argument
failures were separately reviewed before clearing the dirty guard.

Hardware: IRD 223862, `bh-lb-08`, Blackhole P150-class, 12×10 exposed grid,
firmware 19.13.1.0, KMD 2.9.0. The machine was reserved for eight hours;
no claim of eight hours of optimization is implied. Existing host libraries
were reused; all changed kernels were JIT-compiled and executed on hardware.

## Handoff

All new work is isolated under this directory; prior dirty research/model
files are preserved. No production dispatch or public API has been changed.

- [FP32 report, frozen flags and narrow unapplied patch](fp32/REPORT.md)
- [BF16 track](bf16/REPORT.md)
- [Low-precision report](lowp/RESULTS.md)
- [Protocol and ownership](PLAN.md)
- [Live chronology](STATUS.md)

Integration should select only the measured winner paths, not carry rejected
experiments or all screening switches into production. Broader shape/mask/
architecture qualification and model-level timing remain separate follow-up
work before a production merge.
