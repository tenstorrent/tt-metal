# KDA offset prototypes: status, accuracy, and performance

This briefing consolidates the offset experiments in this repository as of
2026-09-10. It separates measured facts from conclusions: all implementations
below are accurate for the cases they tested, but their coverage and performance
measurements are not identical, and none has a production Galaxy SP8xTP4 result.

## Geometry and terminology

Let:

```text
T = global KDA prefill length                    (5120 rows here)
P = number of sequence-parallel ranks
C = T / P = rows physically stored on each SP rank
S = actual_start = global start position
b = floor(S / C) mod P = boundary SP rank
o = S mod C             = local split offset
h = C - o               = boundary-rank head length
```

`C` is therefore not a channel count. It is the local sequence partition size:

| Topology or emulated geometry | P | T | C |
| --- | ---: | ---: | ---: |
| Galaxy SP8xTP4 | 8 | 5120 | 640 rows per SP rank |
| LoudBox SP2xTP4 | 2 | 5120 | 2560 rows per SP rank |
| LoudBox run emulating Galaxy's local shape | 2 | 1280 | 640 rows per SP rank |

When `o != 0`, rank `b` contains two causally non-adjacent pieces. Its first
`h=C-o` physical rows are the beginning of the interval, while its last `o`
rows are the end. For the production example `S=960`, SP8 and `C=640`:

```text
b = floor(960 / 640) = 1
o = 960 mod 640      = 320
h = 640 - 320        = 320

causal order:
SP1[0:320] -> SP2 -> ... -> SP7 -> SP0 -> SP1[320:640]
```

The named benchmark cases describe `o`, not necessarily the full `S`:

| Case | Definition | C=640 | C=2560 | Meaning |
| --- | --- | ---: | ---: | --- |
| Baseline | `S=0`, `o=0`, `b=0` | 0 | 0 | Existing physical and causal order |
| Rotation only | `o=0`, normally `S=C` | 0 | 0 | Causal rank order rotates, but no rank is split |
| Smallest split | `o=32` | 32 | 32 | One tile lies on the tail side |
| Midpoint / worst split | `o=C/2` | 320 | 1280 | Equal head and tail; maximizes ring traffic |
| Largest split | `o=C-32` | 608 | 2528 | One tile lies on the head side |

Legal local offsets are multiples of 32: `o in {0, 32, ..., C-32}`. Thus
`C=640` has 19 split offsets plus `o=0`; `C=2560` has 79 split offsets plus
`o=0`. Different values of `b` move the same local split geometry to another
physical rank.

## Prototype progression

| Stage | Branch | Mechanism | Current interpretation |
| --- | --- | --- | --- |
| Full reshard | `mvasilijevic/kda-prefill-offset` | Gather and repartition the full input into chronological SP order; run stock KDA; gather and restore output placement | Correctness oracle; too much wide activation traffic |
| Sequential tail | same branch | Leave rows in place; route convolution halos and affine summaries; scan head and tail separately | Proves full reshard is not mathematically required; slow and not SP8 L1-feasible |
| Shared base | `mvasilijevic/kda-offset-base` | Derive one canonical topology from `actual_start` and rotate both causal carry orders | Required correctness foundation for later prototypes |
| A: ring exchange | `mvasilijevic/kda-offset-ring` | Move `min(o,C-o)` activation rows one hop before and after otherwise-stock KDA | Broadly predictable, but bandwidth-heavy at `o=C/2` |
| B: split scan | `mvasilijevic/kda-offset-scan` | Move no activation rows; split causal work into chronological fragments | Fast at factorable splits, originally slow elsewhere |
| B + kernel wrap | `mvasilijevic/kda-offset-wrap` | Convolve a wrapped partition in one call and reload recurrence carry inside one scan | Removes duplicated convolution and scan plumbing |
| B + runtime, slot-free wrap | `mvasilijevic/kda-wrap-runtime` | One host recurrence path; runtime wrap counts; per-device tensor selection; affine algebra in L1 | Most advanced candidate; current HEAD `1bc289c7b2c` |

All named offset branches are local-only under their current names. The early
prototype tip also exists as `origin/momcilo/kda-prefill-offset`.

## Accuracy evidence

The verdict is **accurate on covered cases**, not yet “proven for every legal
offset” for every prototype.

| Prototype | Recorded evidence | Assessment |
| --- | --- | --- |
| Full reshard | 172 topology tests; combined hardware coverage at `S=0`, one device boundary, and `S=960`; output PCC 0.999947 on SP8 and 0.999944 on SP2; recurrent state 0.999893; convolution state 0.999997 | Accurate for representative cases; simplest oracle, but narrow offset coverage and zero initial state in the original prototype tests |
| Sequential tail | Same reference comparisons; 2 focused hardware tests plus combined coverage; output and both returned states checked | Accurate for covered cases; SP8 production-shape scan later failed L1 allocation, which is feasibility rather than a numerical failure |
| Shared base | 489 host topology tests over all 160 tile-aligned starts and all eight boundary ranks; device-boundary offsets and both TP axes; output and both carries | Strong evidence for topology and rotation-only correctness |
| A: ring | 12 device tests over smallest/midpoint/largest splits, every boundary rank represented by the test mesh, both TP axes, output and both carries, plus determinism; disabling the exchange fails PCC 0.9969 against 0.999 | Strong representative split coverage with a useful negative control |
| B: split scan | Byte-identical 12-test device suite to A; output and both carries plus determinism | Like-for-like accuracy with A for the measured comparison |
| B: runtime wrap | 12 offset tests, 84 op tests, 543 component/layer tests at the bug-fix commit; final tip reports 820 KDA tests; PCC >=0.99990 and relative RMSE <=0.021, plus per-tensor peak-error gates | Strongest current checks, including a metric that caught a localized error PCC missed; still only the three named split geometries, not every multiple of 32 |

The current peak-error gates matter. An intermediate build applied the wrap to
every chip rather than only the boundary chip. Eleven wrong rows out of 1280
still produced PCC 0.9996, but peak output error rose from about 0.016 to
0.125--0.273. The current tests therefore check output, recurrent state, and
convolution state with separate relative-peak bounds in addition to PCC and
relative RMSE.

## Performance comparison

### Early proofs: same SP2xTP4 run, `T=5120`, `S=960`

These measurements are directly comparable with each other:

| Path | Median warm trace | Over baseline | Relative |
| --- | ---: | ---: | ---: |
| No-offset baseline | 9.533 ms | -- | 1.000x |
| Full reshard | 12.331 ms | +2.798 ms / +29.3% | 1.293x |
| Sequential tail | 13.829 ms | +4.297 ms / +45.1% | 1.451x |

Full reshard won this early comparison, but neither result was attractive. The
numbers do not predict Galaxy: SP2 sees different collective volume and the
sequential-tail SP8 attempt was not L1-feasible.

### Production-quality prototypes A and B on LoudBox SP2xTP4

`T=5120`, `C=2560`, interleaved warm-trace timing. Each branch was normalized
to its own `S=0` baseline (A 10.674 ms, B 10.543 ms):

| Local offset case | A: ring exchange | B: optimized split scan | Winner |
| --- | ---: | ---: | --- |
| Rotation only, `o=0` | +1.7% | +1.6% | Tie |
| Smallest, `o=32` | +14.7% | +32.7% | A by 18.0 points |
| Midpoint, `o=1280` | +29.7% | +14.2% | B by 15.5 points |
| Largest, `o=2528` | +17.5% | +35.7% | A by 18.2 points |

A has a tent-shaped cost: traffic is proportional to `min(o,C-o)` and peaks at
the midpoint. B originally had a valley-shaped cost: it was cheap when both
fragments shared a useful group size, but lost parallelism and sliced seven
prepared tensors at unfavorable offsets. The production Galaxy offset
`S=960`, `o=320=C/2` lies at A's worst point and B's best point.

The attempted one-hop unicast transport did not improve A. At `o=C/2` it was
+187.9%, versus +29.7% for the all-gather-based shift. `point_to_point` achieved
about 12 times worse bandwidth per byte on SP2, so the unicast rung was retired.

### Evolution of B

The convolution-wrap change removed duplicated slicing and launches:

| C=2560 case | B before convolution wrap | B after convolution wrap |
| --- | ---: | ---: |
| Smallest | +32.7% | +27.2% |
| Midpoint | +14.2% | +8.7% |
| Largest | +35.7% | +30.9% |

The current slot-free runtime-wrap path then removed the offset-dependent host
path. Both columns below were reproduced on current HEAD on 2026-09-10 with
preconditioning and cyclically rotated measurement order:

| Case | C=640 local geometry | C=2560 local geometry |
| --- | ---: | ---: |
| Rotation only | 3.500 ms, +0.05% | 11.494 ms, +0.07% |
| Smallest split | 3.788 ms, +7.93% | 13.307 ms, +15.60% |
| Midpoint split | 3.791 ms, +7.92% | 13.290 ms, +15.43% |
| Largest split | 3.777 ms, +7.94% | 13.282 ms, +15.37% |

The sustained baselines were 3.493 ms at `C=640` and 11.504 ms at `C=2560`.
The three split shapes are flat within 0.02 and 0.23 percentage points,
respectively. They were executed on LoudBox SP2xTP4; this is not a Galaxy
communication result.

At `C=2560`, the current path improves B's former bad extremes but regresses
the midpoint relative to the specialized +8.7% path. A split pins the current
implementation to one group, while the `C=2560` baseline uses four. At
`C=640`, baseline and split both naturally use one group, so that penalty does
not exist.

A paired current-HEAD real-time device profile compared `S=0` with the largest
split (`S=608`). The split added 26 programs. Summing each program's maximum
duration across chips increased by 270.8 us, close to the 262.0 us wall delta;
programs can overlap, so the category deltas below are attribution evidence,
not additive wall-time accounting:

| Device-program category | Added programs | Delta in summed per-program maxima |
| --- | ---: | ---: |
| Matmul | 3 | +66.1 us |
| Data movement | 12 | +63.7 us |
| Elementwise | 5 | +56.3 us |
| All-gather | 1 | +54.4 us |
| Recurrent chunk scan | 1 | +16.0 us |
| Copy + data movement | 4 | +14.1 us |
| Mixed data movement + elementwise | 0 | +12.5 us |
| Plain copy | 0 | -11.3 us |
| All other unchanged categories | 0 | -1.1 us net |

Callsite profiling refines this category view. At `C=640`, the three matmuls
are the two affine-composition calls and the tail-seed call; data movement is
mostly expanded convolution-carry packing, summary conversions, and the final
state gather/slice. At `C=2560`, the dominant cost is different: pinning a split
from four recurrence groups to one costs 12.73% by itself and explains about
83% of the full 15.4% slowdown. See `kda_offset_performance_analysis.md` for
the exact callsites, counterfactual, drift-controlled timings, and limitations.

## Current status and remaining decisions

The best production candidate is the runtime-wrap evolution of B, especially
for Kimi-K3's known `S=960`, `o=320` case. It preserves MLA placement, has one
host recurrence path, and removes B's former offset-factorization cliff. It is
not ready to call production-complete:

1. **Galaxy SP8xTP4 is unmeasured.** All latency conclusions above come from an
   eight-chip LoudBox. The `C=640` run matches local compute geometry, not the
   production fabric topology or collective scale.
2. **Exhaustive offset validation is missing.** Tests cover rotation plus the
   smallest, midpoint, and largest splits. Bead `tt-metal_tracker-ea6.7`
   requires every 32-aligned offset and an operation-count assertion.
3. **Trace/cache identity needs resolution.** `wrap_chunk` is hashed, while the
   boundary rank is selected through a prebuilt tensor. Two starts can share a
   wrap value but use different boundary ranks. A caller-updated selector input
   is the documented candidate fix.
4. **Final-state transport is expensive.** The implementation uses an SP
   `all_gather` and slice because a masked `all_reduce` deadlocked under trace.
   It moves 12.6 MB on Galaxy to communicate one final state.
5. **G>1 under a split remains an optimization opportunity.** This dominates
   the residual `C=2560` cost but should not affect Galaxy's native `C=640`,
   one-group geometry.
6. **The branch is stale relative to upstream.** Local Git currently reports
   191 commits on `origin/main` not in this branch and 97 commits on this branch
   not in `origin/main`; it needs a current-main rebase and conflict/behavior
   audit before integration.

Bead `tt-metal_tracker-ea6` is open. Its tracking graph is partly stale:
`ea6.6` (single host path) is implemented but still open, `ea6.5` describes the
superseded extra-slot design, while `ea6.7` and part of `ea6.10` represent real
remaining validation/cache work.

## Bottom line

- All prototypes are numerically credible for their tested cases.
- Full reshard is the clearest oracle but moves too much data.
- Sequential tail proves placement can stay fixed, but that first version is
  slow and not production-feasible.
- A/ring is robust across split shapes but pays most at the production midpoint.
- Original B/split scan wins the production midpoint but formerly had severe
  offset-dependent cliffs.
- Current B/runtime wrap removes those cliffs for the measured shapes and is
  the leading candidate, with about +8.1% overhead at Galaxy-like `C=640` local
  geometry. The result must still be reproduced on actual Galaxy SP8xTP4 and
  across every legal offset before choosing it definitively.
