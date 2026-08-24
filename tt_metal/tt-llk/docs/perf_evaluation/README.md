# LLK perf gate: methodology, measurements, and current state

This is the working manual for the LLK performance gate study. It records what
we set out to answer, how the measurement actually works, every number we
collected, how the threshold was derived, why the scope narrowed to `L1_TO_L1`,
and what is still open.

Tracking issue: [#53763](https://github.com/tenstorrent/tt-metal/issues/53763).
Per-run reports are under `results/`.

---

## 1. The goal

We want a PR/merge gate that fails a change which makes LLK slower.

A gate needs a threshold, and the only comparison tool in the repo,
`.claude/scripts/perf_regression_compare.py`, defaults to 5% — a number nobody
measured.

A threshold is squeezed from two sides:

- It must sit **above the measurement noise**, or the gate fails on jitter.
- It must sit **below the size of a real regression**, or it catches nothing.

This study measures the first. The second needs the commit history and is not
covered here.

A second question came with it: **what does a gate run cost?** A gate nobody can
afford to run is not a gate.

---

## 2. How a measurement is produced

Worth understanding before reading any number, because two of our findings are
consequences of this machinery rather than of the silicon.

### 2.1 Run types are separate binaries

`PerfRunType` has five values (`helpers/llk_params.py`): `L1_TO_L1`,
`UNPACK_ISOLATE`, `MATH_ISOLATE`, `PACK_ISOLATE`, `L1_CONGESTION`.

The kernel selects a different code path per run type at **compile time**:

```cpp
START_PERF_MEASURE("TILE_LOOP")
if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE) { ... }
else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)  { ... }
```

`PerfConfig` appends `PERF_RUN_TYPE(run_type)` to the **template** list
(`helpers/perf/core.py`), so each run type gets its own variant hash and its own
ELF. Selecting fewer run types is therefore the main cost lever: cost scales with
(variants × selected run types). `--perf-run-types` filters that list before
compilation.

Across 28 perf modules there are 70 (module × run-type) pairs. `L1_TO_L1` alone
accounts for 18 of them.

### 2.2 Zones become durations

`START_PERF_MEASURE` opens a profiler zone. The device writes `ZONE_START` and
`ZONE_END` entries into L1 with timestamps in **device clock cycles**, tagged by
thread (unpack, math, pack). Cycles, not time — so clock frequency does not
enter any number in this document.

`helpers/profiler.py` turns those into durations:

| run type | duration |
|---|---|
| `L1_TO_L1` | pack `ZONE_END` − unpack `ZONE_START` (spans all three threads) |
| `UNPACK_ISOLATE` | `ZONE_END` − `ZONE_START` within the unpack thread |
| `MATH_ISOLATE` | same, math thread |
| `PACK_ISOLATE` | same, pack thread |
| `L1_CONGESTION` | unpack and pack durations, as two columns |

Markers: `INIT`, `KERNEL`, `TILE_LOOP`, `UNINIT`. `TILE_LOOP` is the steady-state
loop and is the number anyone optimising LLK cares about. `INIT` and `UNINIT` are
one-time setup and teardown, a few hundred cycles each.

### 2.3 Each CSV value is a single device execution

`PerfConfig.run(perf_report, run_count=1)` defaults to **one** execution per
variant. Only two of 28 modules override it (`perf_fast_tilize` uses 2,
`perf_unpack_a_bcast_eltwise` uses 10). So `mean(<RUN_TYPE>)` is, almost
everywhere, a single measurement, and `std(...)` is dropped as all-NaN.

This matters: the value a gate compares carries the full single-shot jitter.
Raising `run_count` averages on device inside one variant, at no extra compile
and no extra pytest run — the cheapest noise lever available, and untried.

### 2.4 Two-phase execution, and the worker-to-core map

CI runs a producer/consumer split (`tests/run_llk_perf_<arch>.sh`):

```
pytest --compile-producer -n 10 ...   # compile every variant, then skip
pytest --compile-consumer -n 15 ...   # load the ELFs and measure
```

The split exists because the phases have different bottlenecks — compile is
host-CPU bound, measure is device bound — but there is a harder constraint. The
xdist worker index maps onto a **physical Tensix core**
(`helpers/test_config.py`):

```python
row, col = divmod(int(worker_id[2:]), 8)
TestConfig.TENSIX_LOCATION = f"{row},{col}"
```

`gw0` is core (0,0), `gw8` is (1,0). So `-n 15` is a promise that only cores
(0,0) to (1,6) are used. `-n auto` would address an 8×8 grid, and Wormhole ships
with harvested rows, so it is not merely slower — it is wrong. Compile can be
parallelised freely; measure cannot.

### 2.5 Raw totals versus per-tile

`<test>.csv` holds **raw loop totals** for `TILE_LOOP`. The per-tile figure
(divided by `loop_factor × tile_cnt`) exists only in `<test>.post.csv`
(`postprocess_tile_loop`). **Every number in this document comes from the raw
CSV.** A relative threshold is unaffected by that choice; an absolute cycle
threshold is not, so a gate built on `.post.csv` would need its cycle floor
re-derived.

---

## 3. What we ran

Both architectures, `main` plus the `--perf-run-types` harness commit, **speed of
light off** throughout. CI's perf job passes `--speed-of-light`; this gate is
being designed without it.

Deliberate differences from the CI runner:

- No `--speed-of-light`.
- No `--splits`/`--group` — one machine runs the whole selection, so sharding
  cannot mix per-machine noise into the measurement.
- No `-x` — it aborts mid-sweep and the combined CSV is silently partial, which
  reads downstream as measurements that went missing.

Machines: Blackhole `bh-qbge-15`, Wormhole `wh-lb-35`. One card each.

**Cost runs** wipe the build tree before each configuration, because a gate on a
fresh runner pays the compile.

**Noise runs** keep the build tree across iterations: iteration 1 compiles,
iterations 2..N reuse identical ELFs and measure only. Rebuilding the same
sources yields byte-identical ELFs, so this isolates device-side variance and is
far cheaper. `WIPE_BUILD=1` now exists for the opposite question.

Tools, all in `.claude/scripts/`:

| script | purpose |
|---|---|
| `perf_gate_budget.sh` | Cost of each run-type configuration |
| `run_perf_noise_baseline.sh` | N runs of one commit, snapshotting `perf_data` |
| `perf_noise_analysis.py` | Per-point movement statistics |
| `perf_outlier_investigation.py` | Characterises the points a rule flags |
| `perf_noise_plots.py` | Plots, so the 21 MB point CSVs need not be committed |

---

## 4. Cost

Wall clock, one card, cold build tree.

| config | run types | BH compile | BH measure | **BH total** | **WH total** |
|---|---|--:|--:|--:|--:|
| full | all declared | 15:33 | 4:08 | **19:41** | **30:14** |
| isolates | UNPACK, MATH, PACK | 9:55 | 2:45 | **12:40** | **18:45** |
| l1 | `L1_TO_L1` | 4:21 | 1:56 | **6:17** | **9:19** |

Raw seconds in `results/*/timings/`.

**Findings**

1. **A non-SoL gate is affordable.** The entire suite, unsharded, on one card, is
   under 20 minutes on Blackhole and about 30 on Wormhole. The architectures run
   on separate CI runners in parallel, so gate latency is the Wormhole figure:
   about **9 minutes** for `L1_TO_L1`, **19 minutes** for isolates.
2. **Compile is 65–79% of every configuration.** It dominates. But `tt-llk` is
   header-only, so any PR touching a kernel header invalidates essentially every
   ELF — a build cache helps only for PRs that do not touch LLK.
3. **Cost tracks (module × run-type) pairs almost linearly.** Isolates cover 61%
   of the 70 pairs and cost 64% of the time; `L1_TO_L1` covers 26% and costs 32%.
   The gap is fixed per-variant overhead that run-type selection cannot remove —
   that is the argument for reducing test variants.
4. **Wormhole is about 1.55× slower than Blackhole** on identical work.

Not measured: cost per individual isolate thread (a gate selects the three as a
set), and any speed-of-light comparison. Note that `--speed-of-light` folds
runtime parameters into the template list, which multiplies distinct ELFs, so it
should be *more* expensive to compile — worth confirming, since the opposite is
often assumed.

---

## 5. Noise

The same commit was run five times on one card. Nothing changed between runs, so
every difference between them is measurement noise.

> **In one sentence: we ran the same code five times, took the biggest wobble any
> measurement showed between its best and worst run, and set the threshold just
> above it.**

### 5.0 What is being counted

A **point** is one measured number — one test, one marker, one run type, one
sweep configuration. It is the same granularity a row-by-row gate compares.
Running five times gives that one number five values.

Each point has two quantities, which are the *same wobble expressed in two
units*:

| | definition | in words |
|---|---|---|
| **cycles** | `max − min` | how many cycles the number moved between its best and worst run |
| **move** | `(max − min) / median` | that same movement as a percentage of the number's typical value |

Worked example. One `INIT` measurement came out **348, 350, 350, 351, 373** cycles
across the five runs:

- `cycles` = 373 − 348 = **25 cycles**
- `move` = 25 / 350 = **7.1%**

Both are ranges rather than signed differences, so a point is counted whether the
noise made a run faster or slower. Nothing about false *improvements* is missing
from these numbers — see §6.3.

### 5.0.1 How to read the tables below

Every table in this section shares the same columns:

| column | what it is |
|---|---|
| `marker` | which part of the kernel is timed. `INIT` is one-time setup, `TILE_LOOP` the steady-state loop, `UNINIT` teardown, `KERNEL` the whole kernel |
| `measured` | how many points of that kind exist in the sweep |
| `>0.5%`, `>1%`, `>2%`, `>5%` | how many of those points moved by more than that percentage |
| `worst move` | the largest movement seen, in percent |
| `worst cycles` | the largest movement seen, in cycles |

The last two columns are **independent maxima and normally come from different
points**. The largest percentage is typically a small `INIT` value; the largest
cycle count is typically a large `TILE_LOOP` value. Do not read them as one point.

### 5.1 Blackhole, `L1_TO_L1` — 108,377 points

| marker | measured | >0.5% | >1% | >2% | >5% | worst move | worst cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| TILE_LOOP | 35,576 | 26 | 7 | **0** | 0 | 1.88% | 5,110 |
| KERNEL | 35,576 | 28 | 7 | **0** | 0 | 1.87% | 5,108 |
| INIT | 35,576 | 2,184 | 1,390 | 464 | 14 | 7.91% | **25** |
| UNINIT | 1,649 | 91 | 35 | 10 | 0 | 3.57% | **9** |
| all | 108,377 | 2,329 | 1,439 | 474 | 14 | 7.91% | 5,110 |

### 5.2 Blackhole, isolates — 311,352 points

| run type | marker | measured | >0.5% | >1% | >2% | worst move | worst cycles |
|---|---|--:|--:|--:|--:|--:|--:|
| MATH_ISOLATE | TILE_LOOP | 33,177 | 14 | 3 | **0** | 1.47% | 12 |
| MATH_ISOLATE | KERNEL | 33,177 | 7 | 0 | **0** | 0.93% | 17 |
| MATH_ISOLATE | INIT | 33,177 | 1,656 | 979 | 555 | 5.15% | 11 |
| MATH_ISOLATE | UNINIT | 1,649 | 114 | 114 | 114 | 14.29% | 6 |
| PACK_ISOLATE | TILE_LOOP | 35,573 | 2 | 1 | 1 | 19.75% | 353 |
| PACK_ISOLATE | KERNEL | 35,573 | 12 | 3 | 1 | 14.12% | 349 |
| PACK_ISOLATE | INIT | 35,573 | 1,873 | 1,216 | 287 | 4.73% | 21 |
| PACK_ISOLATE | UNINIT | 29 | 0 | 0 | **0** | 0.00% | 0 |
| UNPACK_ISOLATE | TILE_LOOP | 33,925 | 5 | 0 | **0** | 0.88% | 98 |
| UNPACK_ISOLATE | KERNEL | 33,925 | 7 | 2 | **0** | 1.21% | 98 |
| UNPACK_ISOLATE | INIT | 33,925 | 509 | 169 | 19 | 2.61% | 8 |
| UNPACK_ISOLATE | UNINIT | 1,649 | 108 | 108 | 108 | 3.85% | 1 |

Note `UNINIT` for MATH and UNPACK: the counts at >0.5%, >1% and >2% are
identical, because those points move by 6 and 1 cycles respectively. A one-cycle
move exceeding 2% means the value is under 50 cycles.

### 5.3 Wormhole, `L1_TO_L1` — 100,971 points, five runs

| marker | measured | >0.5% | >1% | >2% | >5% | worst move | worst cycles |
|---|--:|--:|--:|--:|--:|--:|--:|
| INIT | 33,657 | 8,776 | 3,882 | 742 | 2 | 5.65% | 27 |
| KERNEL | 33,657 | 176 | 61 | 26 | 0 | 4.60% | 9,165 |
| TILE_LOOP | 33,657 | 186 | 64 | 27 | 0 | 4.62% | 9,165 |
| all | 100,971 | 9,138 | 4,007 | 795 | 2 | 5.65% | 9,165 |

### 5.4 Two regimes, and why one number cannot express both

`TILE_LOOP` and `KERNEL` are close to deterministic on Blackhole: of 71,152
measurements, **none** moved more than 2%, and most did not move at all —
identical code returns identical cycle counts.

All the ordinary noise sits in `INIT` and `UNINIT`. The noise has two components,
and the key point is that **each one is constant in a different unit**. That is
the entire reason the rule needs two clauses.

**A fixed component of about 25 cycles.** `INIT` and `UNINIT` move by at most 25
cycles on Blackhole and 27 on Wormhole, no matter how large the point is. Because
it is constant in *cycles*, its *percentage* depends entirely on the size of the
number it lands on:

| point size | what 25 cycles is |
|---|--:|
| 350 cycles (a typical `INIT`) | **7.1%** |
| 12,000 cycles | 0.21% |
| 250,000 cycles (a typical `TILE_LOOP`) | **0.01%** |

**A proportional component under 2%.** Whatever else moves, moves by a share of
the value. Because it is constant in *percent*, its *cycle count* depends on the
size of the number:

| point size | what 2% is |
|---|--:|
| 350 cycles | 7 cycles |
| 250,000 cycles | **5,000 cycles** |

So there is no single percentage that describes the fixed part, and no single
cycle count that describes the proportional part. That is why knowing "the fixed
part is 25 cycles" does not hand you a percentage — the answer is different for
every point in the sweep.

The consequence for a threshold:

- **A percentage alone** must be at least 7% to survive 25 cycles landing on a
  small `INIT`. Far too loose to catch a real regression.
- **A cycle count alone** must be at least 5,000 to survive 2% landing on a large
  `TILE_LOOP`. Equally useless.
- **Requiring both** — more than 2% *and* more than 30 cycles — bounds each
  component in the unit it is actually constant in. The percentage clause governs
  large points, where 30 cycles is nothing; the cycle clause governs small points,
  where 2% is nothing.

See `results/*/plots/*_rule.png`, which plots movement against point size and
shows the flat 25-cycle floor and the proportional band directly.

The alternative to the cycle clause is to gate only `TILE_LOOP` and `KERNEL` and
ignore `INIT`/`UNINIT` entirely. That removes the small-number problem at the
source and allows a single percentage — see §10.

---

## 6. The threshold, and how it was derived

### 6.1 Procedure

1. For a candidate rule *more than X% **and** more than C cycles*, count the
   points where `move > X` and `cycles > C`. Those are points where some pairing
   of two runs makes the gate fire with no code change.
2. Take the smallest `(X, C)` for which that count is zero.

`C = 30` comes from the fixed component: `INIT`/`UNINIT` move by at most 25–27
cycles no matter how large the point is. `X = 2%` is what remains once that floor
is applied.

### 6.2 Relation to what a gate computes

A gate compares one baseline `b` against one current `c`: `(c − b) / b`. Over all
ordered pairs from N runs the largest is `(max − min) / min`, slightly **larger**
than `move = (max − min) / median`, because `min ≤ median`. At these magnitudes
the difference is second order — 2.00% versus about 2.02% — but the direction
matters: `move` marginally **under**-states the worst pairing.

### 6.3 Direction: false regressions and false improvements

`move` is a range, so the counts below describe a **two-sided** rule,
`|delta| > 2% AND |delta| > 30 cycles`. A regression-only gate fires on at most
the same set and in practice about half as often, since only one of the two
orderings is a slowdown. Nothing about improvements is missing from the
measurement.

Whether the gate should *report* improvements is a design choice. An unexplained
speedup is a common signature of a test that stopped doing work — an early
return, an optimised-away loop. Suggested: report them at the same threshold, do
not fail on them.

### 6.4 Results

> **Flag when a number is more than 2% slower AND more than 30 cycles slower.**

| arch | configuration | measurements | fires on unchanged code |
|---|---|--:|--:|
| Blackhole | L1_TO_L1 | 108,377 | **0** |
| Blackhole | MATH_ISOLATE | 101,180 | **0** |
| Blackhole | UNPACK_ISOLATE | 103,424 | **0** |
| Blackhole | PACK_ISOLATE | 106,748 | 2 |
| Wormhole | L1_TO_L1 | 100,971 | 53 |
| Wormhole | MATH_ISOLATE | 94,692 | **0** |
| Wormhole | UNPACK_ISOLATE | 96,846 | **0** |
| Wormhole | PACK_ISOLATE | 99,414 | 1,457 |

Both clauses are necessary. A percentage alone fires on `INIT`; a cycle count
alone fires on `TILE_LOOP`, which moves thousands of cycles and is still under
2%.

**Repeating runs does not help.** Averaging two runs per side leaves the p99
unchanged (0.41% → 0.51%) and improves only the extreme tail (7.91% → 4.89%).
Most points are bit-identical between runs, so averaging turns exact zeros into
small non-zeros. Use one run per side and do not pay for a second.

The Blackhole `PACK_ISOLATE` pair is one measurement seen through two markers:
`perf_pack_dest_bank`, Float16_b, `L1Accumulation.Yes`. Runs 2–5 were
bit-identical and run 1 was faster by 353 and 349 cycles. Since run 1 is the only
cold-build run, that could be a first-run effect; one point proves nothing.

---

## 7. Narrowing the scope to `L1_TO_L1`

`MATH_ISOLATE` and `UNPACK_ISOLATE` are gate-ready on both architectures today —
zero false failures across 396,142 measurements.

The failures are all in the packer path, and Wormhole `PACK_ISOLATE` is the worst
of them: 1,457 flags of 99,414, deviations up to 24%, no threshold can absorb it.
That is a hardware or kernel investigation, not a threshold question.

So the scope narrowed to the one configuration a gate would most plausibly ship
with: **`L1_TO_L1`, the end-to-end path**, which reaches 18 of 28 perf modules and
costs about 9 minutes on Wormhole. Blackhole is already clean. The remaining
question is the 53 Wormhole flags.

---

## 8. The Wormhole `L1_TO_L1` study, ten runs

Repeated with ten iterations instead of five to characterise the tail.

**Expectation to keep in mind:** more runs can only widen `max − min`, so ten
runs flag *more* points than five, not fewer. What they buy is the shape of each
deviation and whether the flagged set is reproducible.

### 8.1 Headline numbers

| | 5 runs | 10 runs |
|---|--:|--:|
| points | 100,971 | 100,971 |
| flagged by the rule | 53 | **83** |
| worst move | 4.62% | **8.83%** |
| tests involved | `perf_math_matmul` 45, other 8 | `perf_math_matmul` 60, `perf_matmul` 23 |

All 83 are matmul. Nothing else in the sweep flags at all.

The 83 rows are about **42 independent measurements**: `KERNEL` and `TILE_LOOP`
are paired views of the same point.

### 8.2 No cycle floor helps any more

| cycle floor | points above it | smallest % that flags nothing |
|--:|--:|--:|
| 0 | 77,578 | 8.83% |
| 30 | 19,427 | 8.83% |
| 100 | 6,216 | 8.83% |
| 500 | 968 | **8.83%** |

Flat. The offenders are 120,000–220,000-cycle matmul points, so no absolute floor
can reach them.

### 8.3 Excluding matmul

| scope | points | single % that flags nothing |
|---|--:|--:|
| `TILE_LOOP` + `KERNEL`, excluding matmul | 7,890 | **1.59%** |
| all markers, excluding matmul | 11,835 | 5.65% |
| everything | 100,971 | 8.83% |

The 5.65% is one `INIT` point moving **27 cycles**. Dropping `INIT`/`UNINIT`
removes the small-number problem without needing a cycle clause at all.

### 8.4 What the matmul outliers look like

They are **bistable**, not jittery. Representative values across ten runs:

```
120260 123065 120261 120261 123065 120261 120261 120260 120260 120261
  → eight runs at 120,261 and two at 123,065. Exactly two values.

166621 166619 166620 166621 166619 159472 166619 166619 166619 166619
  → nine runs together, one run 7,149 cycles FASTER.
```

| test | point size | jump | as % |
|---|---|---|---|
| `perf_matmul` | 10k–28k cycles | 245–1,112 | 2.1–8.8% |
| `perf_math_matmul` | 118k–218k cycles | 2,412–13,843 | 2.0–6.4% |

Roughly a third show strong mixing — splits of 6/4, 7/3, 2/8 across ten runs — so
the alternate state is not rare. The rest are 9/1 or 1/9.

**The alternate values repeat across different points.** Several 165k–169k points
all drop to about 158,7xx; several 184k points all rise to about 190,4xx.
Different configurations, different runs, near-identical alternate value. That is
a discrete state, not analogue variance.

### 8.5 It is not the configuration

Every sweep parameter was scored by how much the flag rate varies across its
values. Highest rates found:

| parameter | strongest value | rate | vs weakest |
|---|---|--:|--:|
| `dst_index` | 15.0 | 0.71% | 9× |
| `math_fidelity` | LoFi | 0.20% | 5× |
| `dest_sync` | Half | 0.20% | 4× |
| `tile_cnt` (`perf_matmul`) | 3 | 0.37% | 2.6× |
| `k_dimm` (`perf_matmul`) | 2.0 | 0.31% | 2× |

Even the most enriched value leaves **99.3% of its points clean**, and flagged
points appear at nine different `dst_index` values with `0.0` the most common. The
two tests do not even agree on direction — `Float32` output is the safest in
`perf_math_matmul` and among the worst in `perf_matmul`.

An earlier `dst_index = 15` hypothesis is **rejected**.

### 8.6 It is not a fixed set of configurations

| | count |
|---|--:|
| flagged with 5 runs | 53 |
| flagged with 10 runs | 83 |
| in both | **30** |
| only in the 5-run set | 23 |
| new in the 10-run set | **53** |

Only 30 of 53 recurred, and doubling the runs found 53 configurations never seen
before. The flagged set is a **sample**, not a population: any matmul
configuration can do this, at a low per-run probability. An exception list would
grow every time anyone re-measured, so it is not a viable gate mechanism.

### 8.7 It is not a per-run state

Over ten runs, the run holding the extreme value is spread 7.2% to 16.9% across
the ten, against 10% under independence. A state established once per run and
held would put nearly 100% on a single run. It does not.

---

### 8.8 It is not anything the harness controls

The harness can record which variant ran on which worker, in order
(`--record-test-order`), and replay that exact sequence (`--test-order-file`
with `--rewind-runner`). Replaying one worker's sequence at `-n 1` pins execution
to core (0,0) and fixes the order, leaving nothing under software control to
vary.

**Two replays of the same sequence.** 1,980 `TILE_LOOP` points. 1,377 differed,
but almost all by a few cycles — median 1, quartiles −8 to +6. One point differed
by more than 2%.

**Ten replays of the same sequence:**

| movement across the ten replays | points of 1,980 |
|---|--:|
| >0.1% | 264 |
| >0.5% | 46 |
| >1% | 12 |
| >2% | **2** |
| >5% | 0 |

The rule fires on **2 of 1,980 — 0.101%**. The parallel baseline rate is 42 of
29,712 matmul `TILE_LOOP` points, or 0.141%, which predicts **2.8** here.
Observed 2. Statistically indistinguishable.

**Serial execution does not suppress the effect.**

The shape is unchanged too:

```
14,804 ×9 runs, one run at 14,505     −299 cycles, 2.02%
```

Nine identical values and one discrete step — the same signature as the parallel
runs.

A separate observation from this experiment: at `-n 1` there is a floor of
few-cycle jitter on most matmul points, 264 of 1,980 moving more than 0.1%. It is
far below any threshold under discussion (±8 cycles on a 120,000-cycle point is
0.007%) and invisible in the baseline tables, but it is not zero.

### 8.9 What is left

Everything the test harness governs has now been excluded:

| hypothesis | verdict | how |
|---|---|---|
| Sweep configuration | Excluded | No parameter separates flagged from clean (§8.5) |
| A fixed set of bad configurations | Excluded | The flagged set is not reproducible (§8.6) |
| A per-run state | Excluded | The odd run is spread evenly across runs (§8.7) |
| Core placement | Excluded | `-n 1` uses only core (0,0); rate unchanged |
| Execution order, leftover state from the previous test | Excluded | Identical order file, different results |
| Cross-core contention from 15 concurrent tests | Excluded | No concurrency at `-n 1`; rate unchanged |
| Cold build / first-run effect | Excluded | Both replays warm |

Now line up which measurements are affected, across every configuration measured:

| what is measured | Wormhole flags |
|---|--:|
| `MATH_ISOLATE` — math thread only | **0** of 94,692 |
| `UNPACK_ISOLATE` — unpack thread only | **0** of 96,846 |
| `PACK_ISOLATE` — pack thread only | 1,457 of 99,414 |
| `L1_TO_L1` — unpack start to **pack end** | 83 of 100,971 |

**Every failing measurement depends on when the packer finishes.** The two that
never touch pack timing are clean across 191,538 Wormhole measurements and
204,604 on Blackhole. Add the format association within `PACK_ISOLATE` —
`Float16_b` output at a 13% flag rate against 0% for `Float32` and `Bfp8_b` — and
the conclusion is:

> **The packer's completion time is bistable.** It lands on one of two discrete
> values at low probability per execution, independent of core, execution order,
> concurrency, build state and sweep configuration, with a rate that depends on
> the output format.

That is a packer or hardware question, not a measurement artefact, and no
threshold can absorb it.

### 8.10 How matmul depends on the packer, and why that shows up here

This section is mechanism. The facts in §8.9 are measurements; what follows
separates what the architecture guarantees from what is still a hypothesis.

#### The pipeline, and where the measurement boundaries sit

A Tensix compute kernel runs on three threads that hand work along a chain:

```
  unpack (T0)  ->  math (T1)  ->  pack (T2)
      L1        SrcA/SrcB      DEST        L1
```

The unpacker moves tiles from L1 into the source registers, math computes into
the DEST register, and the packer moves DEST back out to L1.

The four measurements differ only in which part of that chain they bracket:

| measurement | from | to |
|---|---|---|
| `UNPACK_ISOLATE` | unpack `ZONE_START` | unpack `ZONE_END` |
| `MATH_ISOLATE` | math `ZONE_START` | math `ZONE_END` |
| `PACK_ISOLATE` | pack `ZONE_START` | pack `ZONE_END` |
| `L1_TO_L1` | **unpack** `ZONE_START` | **pack** `ZONE_END` |

`L1_TO_L1` is the only whole-pipeline measurement, and its closing boundary is
the packer. So anything that delays the packer's last write lands inside
`L1_TO_L1` and inside `PACK_ISOLATE`, and cannot appear in the other two. That is
not a hypothesis — it follows from where the zones are placed, and it explains
the table in §8.9 exactly.

#### Why matmul leans on the packer harder than anything else

Matmul accumulates across the `kt_dim` axis: math writes partial products into
DEST repeatedly before a tile is finished. DEST is a shared resource, so math and
pack must hand it back and forth, and the handshake is a semaphore:

- `llk_pack_common.h` — the packer waits on `MATH_PACK` so it does not run ahead
  of the math result, packs, then signals to release math.
- `llk_math_common.h` — the semaphore is seeded with a **max count of 1 for
  SyncFull and 2 for SyncHalf**.

That count is the whole difference between the two dest-sync modes. Under
`SyncFull` the packer and math take strict turns on the whole DEST. Under
`SyncHalf` DEST is split in two and the packer flips the active half, so **math
and pack genuinely overlap** — one works on a half while the other drains the
other half.

Matmul therefore spends most of the packer's life waiting on DEST availability,
and under `SyncHalf` that wait is a race rather than a queue.

#### What the data says about which configurations are affected

Association only — these parameters co-vary — but they line up in one direction.
Within `perf_math_matmul`, Wormhole `L1_TO_L1`, ten runs:

| parameter | value | flag rate |
|---|---|--:|
| `dest_sync` | **Half** | 0.20% |
| `dest_sync` | Full | 0.05% |

And in Wormhole `PACK_ISOLATE`, where the effect is ~100x more frequent:

| parameter | values | flag rate |
|---|---|--:|
| `formats.output` | **Float16_b** | 13% |
| `formats.output` | Float16 | 3% |
| `formats.output` | Float32, Bfp8_b | 0% |
| `dest_acc` | Yes / No | 9% / 2% |
| `formats.register_*` | Tf32 / Bfp8_b | 10% / 2% |

Every one of those touches the DEST-to-L1 path: the sync mode that lets math and
pack overlap, the accumulation width that changes DEST layout, and the output
format the packer must convert to on the way out.

#### The leading hypothesis

> **The math-to-pack handoff over DEST resolves two different ways.** When the
> packer arrives at the `MATH_PACK` wait it either finds the half already
> released and proceeds, or misses that window and waits for the next one. The
> penalty is one arbitration quantum, which is why the measurement lands on one
> of two discrete values rather than smearing.

This is consistent with everything observed, and nothing observed contradicts it:

- Only pack-dependent measurements move (§8.9).
- The outcome is discrete and two-valued, not analogue (§8.4).
- It is symmetric — the odd run is as often faster as slower — because the race
  can fall either way, and neither outcome is "correct".
- It is independent per execution, so it survives serial replay on a fixed core
  with a fixed order (§8.8).
- No sweep parameter determines it, because a race is decided by timing and not
  by configuration (§8.5) — while parameters that widen the race window, such as
  `SyncHalf`, do raise the rate.
- The alternate values repeat across different configurations, because the
  quantum is a property of the handshake rather than of the workload (§8.4).

It remains a hypothesis. It has not been confirmed against hardware counters, and
an alternative — that the packer's format-conversion path itself takes one of two
durations — would fit the format association at least as well.

#### What it costs

For a measurement: the reported cycle count for an affected matmul configuration
is one of two values, 2-6% apart on `L1_TO_L1` and up to 24% on `PACK_ISOLATE`,
chosen at low probability per execution.

For a gate: nothing can be done with a threshold. The effect is not noise that
averages out - repeating runs does not help (§6.4) - and it is not confined to a
nameable set of configurations (§8.6). Any matmul point can produce it on any
run. A gate must either exclude the affected measurements or wait for the cause
to be fixed.

For the numbers already published: they are unaffected. Every measurement in this
document is what the hardware actually reported.

#### What would settle it

Three experiments, in increasing cost:

1. **Is the penalty quantised?** Divide each observed jump by `loop_factor` and
   by `loop_factor x tile_cnt`. A small integer would mean one extra stall per
   iteration, and would give the quantum a size.
2. **Does it flip between back-to-back executions?** `PerfConfig.run()` takes
   `run_count`, which re-runs the same ELF on the same core inside one pytest
   item. With `run_count > 1` the report gains `std(...)` columns. A non-zero
   `std` places the cause below anything the test harness does.
3. **Which counter is bimodal?** The `TDMA_PACK` counter bank measures packer
   busy, dest-read availability and math availability. If dest-read availability
   splits in the same ratio as the timing, the handshake hypothesis is confirmed;
   if packer-busy cycles split instead, it is the conversion path.

Run these against `PACK_ISOLATE` on `perf_matmul` with `Float16_b` output, where
the rate is 13% rather than 0.14%.

---

## 9. What we know, and what we do not

### Established

1. **Perf measurement is largely deterministic.** Most points return identical
   cycle counts across runs.
2. **Ordinary noise is a fixed ~25 cycles**, which only matters for the few
   hundred-cycle `INIT`/`UNINIT` markers.
3. **`MATH_ISOLATE` and `UNPACK_ISOLATE` are gate-ready** on both architectures.
4. **Non-matmul `L1_TO_L1` is stable**: 7,890 `TILE_LOOP`/`KERNEL` points, worst
   movement 1.59%.
5. **Matmul is bistable**: measurements land on one of two discrete values, 2–6%
   apart, at low per-run probability.
6. **The cause is not the configuration** (§8.5), **not a fixed set of tests**
   (§8.6), **not a per-run state** (§8.7), and **not core placement, execution
   order or concurrency** (§8.8). Serial replay of a fixed order on one core
   reproduces it at 0.101% against a parallel baseline of 0.141%.
7. **Every affected measurement depends on packer completion time** (§8.9).
   `MATH_ISOLATE` and `UNPACK_ISOLATE` are clean across 396,142 measurements on
   the two architectures; `PACK_ISOLATE` and `L1_TO_L1` carry every failure.
8. **The packer path is the worst case** — Wormhole `PACK_ISOLATE` reaches 24%,
   with `Float16_b` output at a 13% flag rate against 0% for `Float32` and
   `Bfp8_b`.

### Not established

1. **The mechanism inside the packer.** We know the packer's completion time is
   bistable and that nothing the harness controls explains it (§8.8, §8.9). What
   we do not know is why — whether it is arbitration in the pack pipeline, an
   interaction with output-format conversion, or something below that. This needs
   someone who owns the packer path, and the `Float16_b` association is the lead.
2. **Whether this reproduces across machines and days.** Every measurement here
   is one card in one session, so all of it is a **lower bound** on the noise a
   real gate would see.
3. **How large real regressions are.** Everything here constrains the threshold
   from below only.

---

## 10. Where the threshold stands

Two defensible options, depending on whether a cycle clause is acceptable.

**With a cycle clause**

> Flag when a number is more than **2%** slower AND more than **50** cycles
> slower.

Zero false failures on Blackhole `L1_TO_L1`, `MATH_ISOLATE` and
`UNPACK_ISOLATE`, and on all non-matmul Wormhole `L1_TO_L1`. 50 rather than 30
because Wormhole's worst `INIT` move is 27 cycles, leaving only three cycles of
margin at 30; 1.59% still holds at a 50-cycle floor.

**Percentage only**

> Flag when a number is more than **2%** slower. Applied to `TILE_LOOP` and
> `KERNEL` only.

Worst observed movement is 1.59%, so 2% has margin. Excluding `INIT`/`UNINIT` is
what removes the need for a cycle clause — they are one-time setup cost, not the
steady-state number anyone optimises.

**Both options exclude matmul**, which is 88% of the `L1_TO_L1` sweep. That is
the honest cost of shipping today: a clean threshold over 7,890 of 100,971
points. Matmul cannot be gated until §9's open question is answered, and it
cannot be carved out by an exception list because the list is not reproducible.

---

## 11. Reproducing everything

```bash
# cost
.claude/scripts/perf_gate_budget.sh --arch <arch> --configs full,isolates,l1 \
  --build-root $HOME/llk-build --out $HOME/perf_gate_budget

# noise, N runs of one configuration
SKIP_MAIN_CHECK=1 ALLOW_DIRTY=1 PERF_RUN_TYPES=L1_TO_L1 \
  OUT_DIR=$HOME/noise_l1 \
  .claude/scripts/run_perf_noise_baseline.sh <arch> 10

# characterise what a rule flags
python3 .claude/scripts/perf_outlier_investigation.py \
  $HOME/noise_l1/noise_report.points.csv "<title>" out.md L1_TO_L1
```

`ALLOW_DIRTY=1` is needed because the dirty-tree guard trips on the downloaded
`tests/sfpi` toolchain. `SKIP_MAIN_CHECK=1` is needed while `--perf-run-types` is
unmerged.

Run long jobs detached — `setsid nohup … &` — and verify the `TT` column reads
`?` before disconnecting. To stop one, signal the process group, not the PID.

---

## 12. Known defects in the tooling

- `perf_gate_budget.sh` counts yield in `tests/python_tests/perf_data`, but the
  combiner writes to `tt-llk/perf_data`, so `rows`/`points`/`modules` read 0.
  Timings are unaffected.
- The dirty-tree guard in `run_perf_noise_baseline.sh` trips on `tests/sfpi`.
- `perf_outlier_investigation.py` counts distinct values exactly, so a point whose
  cluster spans 124,027 and 124,028 is scored as two states. That inflates the
  "scattered" share and understates how bimodal the data is. It should cluster
  with a tolerance.
- Per-worker CSVs, which would pair every value with the core that produced it,
  are deleted by the combiner (`helpers/perf/core.py`, `Path(file).unlink()`).
  Capturing them requires copying during the run.
