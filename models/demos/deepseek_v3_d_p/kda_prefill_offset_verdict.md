# KDA prefill offsets: measured verdict

Settles the open question in [`kda_prefill_offset_design.md`](kda_prefill_offset_design.md) --
whether a small activation exchange or a segment-aware scan gives the better
latency -- with device evidence rather than argument. Implementation detail is
in [`kda_prefill_offset_dev_spec.md`](kda_prefill_offset_dev_spec.md).

**Recommendation: prototype A, the one-hop ring exchange.**

## The two prototypes

| | A -- one-hop ring exchange | B -- uniform fragment split |
| --- | --- | --- |
| branch | `mvasilijevic/kda-offset-ring` | `mvasilijevic/kda-offset-scan` |
| idea | ship `min(o, C-o)` rows one hop so every chip holds a whole logical chunk, then run stock KDA | move nothing; split every chip at the boundary row and scan `2P` chronological fragments |
| activation bytes moved | `min(o, C-o)` rows, twice | zero |
| extra launches | 2 collectives + 2 selects | convolution, summarize, reduce, prefix and scan each run twice |
| realises | design Alternative B, with the all-gather replaced by a neighbour shift | design's preferred segment-aware execution |

Both root on PR #52799 head `71c0826018f`, both build on the shared base branch
`mvasilijevic/kda-offset-base`, and both are local only.

## Correctness

Both prototypes pass a byte-identical suite, so the comparison is like for like.

| evidence | result |
| --- | --- |
| host topology: 160 tile-aligned starts x 8 boundary chips, against MLA's own oracle | 489 passed |
| offset-zero regression: convolution + recurrence, both TP axes | 14 passed, numerically unchanged |
| offset-zero regression: distributed layer | 6 passed |
| device offsets A: 3 split sizes x every boundary chip x both TP axes, output and both carries, plus determinism | 12 passed |
| device offsets B: identical suite | 12 passed |

The device suite compares against the natural-order reference after undoing
MLA's permutation, and checks **both replacement carries** separately, because
an ordering bug can leave the output plausible while corrupting the state that
feeds the next chunk. A negative control confirms the suite is not vacuous:
disabling A's exchange fails `start=32` at PCC 0.9969 against the 0.999 gate.

## Measured cost

Blackhole LoudBox, SP2xTP4, production K3 at `T=5120`, `C=2560`. Each prototype
is normalised to its own `S=0` baseline (A 10.674 ms, B 10.543 ms).

| case | actual_start | A (ring) | B (split) | A - B |
| --- | ---: | ---: | ---: | ---: |
| device boundary (o=0) | 2560 | +1.7% | +1.6% | +0.1 pp (tie) |
| smallest split (o=32) | 32 | +14.7% | +32.3% | -17.6 pp (A) |
| worst case (o=C/2) | 1280 | +29.7% | +25.9% | +3.9 pp (tie) |
| largest split (o=C-32) | 2528 | +17.5% | +35.4% | -17.9 pp (A) |

Program cache stays bounded: A 102 entries, B 148.

### The shapes are opposites, and that is the whole result

**A is a tent peaking at `o = C/2`.** Its cost is the bandwidth it must move,
`min(o, C-o)`, which is maximal at the midpoint and near zero at both extremes.

**B is a valley bottoming at `o = C/2`.** Its cost tracks how badly the fragment
length factors, not how much data is displaced. At `o = C/2` both fragments are
40 chunks and group cleanly by 20; at `o = 32` and `o = C-32` one fragment is 79
chunks, which is prime, so it collapses to a single group and loses group-level
parallelism.

A therefore wins by roughly 18 points at both extremes and the midpoint is a
tie, leaving no offset where B is meaningfully ahead.

Rotation alone is free on both (`+1.7%` / `+1.6%`, inside noise), as theory
predicts: a device-boundary offset only reorders a carry list.

## Transport: the collective beats a true one-hop unicast

A's exchange has two transports behind one seam, proven to deliver identical
rows (backend-parity test, 18 passed).

| case | rows unicast moves | `slice_gather` | `unicast` |
| --- | ---: | ---: | ---: |
| device boundary | 0 | +1.8% | +1.1% |
| smallest split (o=32) | 32 | +14.7% | +17.5% |
| worst case (o=C/2) | 1280 | **+29.7%** | **+187.9%** |
| largest split (o=C-32) | 32 | +17.6% | +17.1% |

Cases moving 32 rows are a wash; the one moving 1280 rows explodes. On SP2 the
gather receives `P*o = 2560` rows against unicast's 1280 -- only twice the bytes
-- yet runs about six times faster, so `point_to_point` achieves roughly twelve
times worse bandwidth per byte than `all_gather`. **Keep `slice_gather`.**

This retires the fused C++ neighbour-shift escalation. Its condition was a real
bandwidth win consumed by launch overhead; instead dispatch count is fine and
per-byte throughput is the problem. The one condition that would revive it is
Galaxy SP8, where the gather receives eight times the payload rather than twice.

## Why A is the recommendation

- It is never meaningfully worse, and much better at two of three split cases.
- Its cost is *bandwidth*, which the untried unicast rung
  (`tt-metal_tracker-6ls.3`) attacks directly: the current rung moves `P*o` rows
  through one collective, while a true one-hop unicast moves `o` -- up to 16x
  less. B has no comparable lever; its cost is launch count and lost
  parallelism.
- It leaves the scan untouched, so KDA's most intricate machinery keeps one
  code path.
- Smaller program cache footprint.

## What was not measured, and what would change the answer

- **Galaxy.** Everything here is LoudBox SP2xTP4. Galaxy SP8xTP4 has a 640-row
  partition against LoudBox's 2560, which matters: B's prime-fragment pathology
  is partly a LoudBox artifact, since at `C=640` and `o=32` the head is 19
  chunks, which fits the ceiling of 20 and stays at one group with only 24
  owners. **B may look better on Galaxy than it does here.** Confirming the
  recommendation needs a Galaxy run.
- **Measurement error.** Per-offset spread is about 15-18% across five
  interleaved samples, so each median carries roughly +/-5%. The ~18 point gaps
  clear that comfortably; the 3.9 point midpoint difference does not, which is
  why it is called a tie.
- **Sequential timing is invalid here** and was discarded: two runs of the
  identical baseline differed by 13.5%, and the free rotation case read +11.8%
  when measured last against +3.2% when measured second. All numbers above come
  from interleaved timing, where every offset's trace is captured up front and
  samples round-robin.
- **B's baseline-sized summary exchange is unbuilt** (`tt-metal_tracker-6ls.5`),
  deliberately. B gathers `2P` fragment summaries where `P` would do; at 1.57 MB
  per summary pair that is 3.1 MB of avoidable traffic on LoudBox SP2, about
  91 us, or 3% of B's own overhead -- it would move B's best case from +25.9%
  to roughly +25.0% and cannot change the verdict. On **Galaxy SP8 the same
  change saves 12.6 MB, about 363 us or 9.2% of baseline**, so it becomes worth
  building the moment Galaxy numbers exist.
- **No regression budget is proposed.** The design defers that until Galaxy
  measurements exist, and they do not.

## Findings worth keeping

- **One canonical order, or it breaks.** B initially gave the chip after the
  boundary chip the boundary chip's *tail* carry; chronologically that chip
  follows the boundary chip's *head*. Cost: one carry, four wrong rows, PCC
  0.998 uniformly across every split geometry. The recurrence was already right
  because it composes over the shared order; only the convolution re-derived
  predecessors by hand. Both stages now walk the same order.
- **Correctness testing cannot substitute for production dimensions.** B built
  and passed everywhere at test scale, then failed to build at production scale:
  a prime fragment demanded 1896 summary owners against a 110-core grid, because
  group sizing honoured the performance ceiling but never the hardware budget.
- **Measure the thing the design promised.** A shipped `o` rows rather than
  `min(o, C-o)`, so its cost grew with `o` and it looked worse than B at
  `o = C-32` (+44.8%). Implementing the direction choice moved that case to
  +17.5% and inverted the verdict.
