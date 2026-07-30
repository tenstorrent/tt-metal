# AttnRes roofline — Phase 8

What the composed op *can* cost, so Phase 9 has a number to be measured against
rather than a feeling. Analytical, from constants that are cited or measured — no
profiler has run. `DISTRIBUTION.md` covers the mapping; this covers the cost of it.

Two things here are measurements, not derivations, and they are marked **[measured]**:
which algorithm `ttnn.all_reduce` actually picks for the statistics tensor, and what
one op launch costs in the regime the harness runs in. Both changed the conclusion.

> **Phase 9 has since measured it.** The original text is kept intact — a pre-committed
> roofline is only worth anything if it can be caught being wrong — with dated amendment
> blocks in §4, §6, §7 and §8. Scorecard: §3's DRAM floor holds at **1.43×** (the composed
> op runs at 70% of it), §5's algorithm rule holds exactly, §4's fabric term is **off
> 2.7×**, and §6's verdict **inverts**. Read a section and its amendment together, in that
> order.

---

## 1. Constants

Blackhole. Everything in-repo is cited `file:line`; the two rows that are not in-repo
say so.

| constant | value | source |
|---|---|---|
| AI clock | 1.35 GHz | `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:56`; the dispatch golden reports `Clock` 1343–1350 MHz measured |
| DRAM bandwidth | 512 GB/s | `ttnn/core/operation.cpp:36,42`; `tests/tt_metal/tt_metal/perf_microbenchmark/8_dram_adjacent_core_read/test_dram_read.cpp:283-289` |
| DRAM banks | 8 | `tt_metal/soc_descriptors/blackhole_140_arch.yaml:11-21`; `dram_grid_size()` → `8-1` **[measured]** |
| DRAM per bank | 4 278 190 080 B (3.98 GiB) | `blackhole_140_arch.yaml:112-113` |
| worker L1 | 1 572 864 B (1536 KiB) | `blackhole_140_arch.yaml:109-110` — of which ~1464 KiB is usable, `tech_reports/TT-Distributed/HDSocketsModel.md:77` |
| matrix engine, LoFi | 5.4 TFLOP/s per core | `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:56,67` |
| compute-with-storage grid | 11 × 10 = 110 cores | `compute_with_storage_grid_size()` → `11-10` on this box **[measured]**; the SoC grid is 17 × 12, `blackhole_140_arch.yaml:1-3` |
| program dispatch, traced | 1.281 µs | `.../dispatch/pgm_dispatch_blackhole_golden.json`, `all_processors_all_cores_trace` |
| program dispatch, untraced, 1 CB + 1 sem | 1.663 µs | same golden, `all_processors_all_cores_1cb_1sem` |
| program dispatch, untraced, 32 unique RTAs | 4.78 µs | same golden, `all_processors_all_cores_32_rta` |
| fabric link, LoudBox | 400 Gbps per direction per link | `models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_ccl_perf.py:41-42` — **external hardware docs, not in-repo** |
| fabric link, Galaxy | 200 Gbps per direction per link | same — **external hardware docs, not in-repo** |
| links per direction, production | 2 | `test_sparse_mla_ccl_perf.py:38` |

### The unit check

The fabric constant is in **giga*bits*** per second and everything else is in
**giga*bytes***. Getting this backwards is an 8× error in the direction that makes a
collective look free, so it is written out once:

```
400 Gbps/direction/link  ×  2 links  ×  1 sustained direction (Topology.Linear)
  = 800 Gbps  =  800 / 8  =  100 GB/s
```

`/8` and the single sustained direction on a line both follow
`test_sparse_mla_ccl_perf.py:76-83`. `Topology.Ring` sustains 2 directions, so a ring
axis doubles this.

Then, because 1 GB/s is numerically 1 byte/ns:

```
time_ns  =  bytes / (GB/s)
```

which is the identity at `test_sparse_mla_ccl_perf.py:85-87` and the reason no factor
of 10⁹ appears anywhere below.

---

## 2. What one read actually moves

Counting DRAM touches — every ttnn eltwise or reduction reads its input from DRAM and
writes its output back, so a "touch" is one element read or one element written. With
`V = (S+1) · T_local · d_local` elements in the candidate set:

| step | op | touches |
|---|---|---|
| build `v` | `concat([block_residual, prefix_sum], dim=1)` | read `V`, write `V` → **2V** |
| `Σ_d v²` | `mul(v, v)` | read `2V` (one buffer, two CB streams), write `V` → **3V** |
| | `sum(dim=3)` | read `V` → **1V** |
| `Σ_d v·q` | `mul(v, q)` | read `V`, write `V` → **2V** |
| | `sum(dim=3)` | read `V` → **1V** |
| mixture | `mul(v, weights)` | read `V`, write `V` → **2V** |
| | `sum(dim=1)` | read `V` → **1V** |
| | | **12V** |

Everything else in the read — the reciprocal RMS, the hand-rolled softmax, the
statistics all-reduce — runs on `[1, C, T_local, 1]` tensors whose element count is
`d_local / 32` times smaller even after tile padding. They are ~0 bytes and, as §5
shows, all of the launch cost.

Three passes over `v` are structurally avoidable and one is not: the concat exists only
because `prefix_sum` and `block_residual` are separate buffers, and each of the three
`mul`s materializes a full `V` intermediate that its paired `sum` immediately consumes.
§7 prices removing them.

**Arithmetic intensity.** Those 12 touches carry ~6 flops per element of `v`
(three multiplies, three accumulate steps), so 6 flops per 24 bytes = **0.25
flop/byte**. Saturating 512 GB/s at that intensity needs 128 GFLOP/s. One Blackhole
core's matrix engine does 5.4 TFLOP/s at LoFi — 42× that, and there are 110 cores.
Compute cannot bind this op by three orders of magnitude, so nothing below needs a
compute roof: AttnRes is data movement.

---

## 3. The DRAM floor

`d = 7168`, `T = 5120`, `S = 8` (the widest read), bf16, `12V` touches at 512 GB/s:

| placement | `V` elements/device | traffic | floor |
|---|---|---|---|
| single device | 330 301 440 | 7 927 MB | **15 483 µs** |
| `(2, 4)` LoudBox | 41 287 680 | 991 MB | **1 935 µs** |
| `(8, 4)` Galaxy | 10 321 920 | 248 MB | **484 µs** |
| `(2, 4)`, `T = 64` (what the suite runs) | 516 096 | 12.4 MB | **24 µs** |

Over a whole forward the read schedule is not flat: 186 executed reads with `S` ramping
0 → 8, `Σ(S+1) = 1002`, mean `S+1 = 5.39`. On `(2, 4)` at `T = 5120` that totals
**110.3 GB** of DRAM traffic and a **215.5 ms** floor for the AttnRes reads alone.

That number is the reason this memo exists. 215 ms of pure data movement per prefill
forward, for an op with no matmul in it, is not a rounding error against the rest of the
model — and §7 says most of it is removable.

---

## 4. The fabric term

Per read, the statistics all-reduce moves `[1, 2(S+1), T_local, 1]` fp32. The last dim
is 1 and tile-pads to 32, so the padded payload is 32× the useful one — the deliberate
trade recorded in `DISTRIBUTION.md` §4.

Critical-path bytes follow the two possible algorithms (see §5): reduce-scatter + all-gather
moves `2·B·(R−1)/R` on the busiest link, a composite all-gather + local sum moves
`B·(R−1)`. At `S = 8`, `T = 5120`, production `num_links = 2`:

| mesh | padded `B` | RS+AG critical | RS+AG | composite critical | composite |
|---|---|---|---|---|---|
| `(2, 4)` @ 400 Gbps | 5 760 KiB | 8 640 KiB | **88.5 µs** | 17 280 KiB | 176.9 µs |
| `(8, 4)` @ 200 Gbps | 1 440 KiB | 2 160 KiB | **44.2 µs** | 4 320 KiB | 88.5 µs |

At `num_links = 1` — which is what `TtAttnRes.__init__` currently defaults to — every
number doubles. The analog's prefill uses 2 (`test_sparse_mla_ccl_perf.py:38`); the
default should follow it, and §8 carries that.

Against §3's DRAM floor the collective is **4.6%** on `(2, 4)` and **9.1%** on
`(8, 4)`. Whole-forward on `(2, 4)`: 985 MB of critical-path bytes, **9.85 ms**, against
215.5 ms of DRAM. The padded statistics are **0.595%** of DRAM traffic — `DISTRIBUTION.md`
§4 said 0.65%; 0.595% is the exact figure and the conclusion is the same one.

M1's claim that the sequence axis is free survives contact with arithmetic: it moves
zero bytes at every shape, and the only collective in the op is under 10% of a floor
that is itself DRAM.

> **Amended 2026-07-30 (Phase 9, P4 — measured, traced, `(2, 4)`).** The two µs columns
> above are wrong by 2.7× and the reason matters more than the factor.
>
> | shape | padded | useful | `links = 1` | `links = 2` | this table predicted |
> |---|---|---|---|---|---|
> | `[1, 18, 2560, 1]` | 5 760 KiB | 184 KiB | 348.1 µs | **235.9 µs** | 88.5 µs |
> | `[1, 18, 2560, 32]` | 5 760 KiB | 5 760 KiB | 348.2 µs | 235.5 µs | — |
> | `[1, 1, 2560, 18]` | 320 KiB | 180 KiB | **46.8 µs** | 50.2 µs | — |
> | `[1, 2, 2560, 1]` | 640 KiB | 20 KiB | 63.0 µs | 62.4 µs | — |
>
> The critical-path byte count is right; the bandwidth is not. The collective sustains
> ~18 KiB/µs above a ~29 µs floor — **18–25% of the 400 Gbps in §1** — and it is
> core-limited, not link-limited: the profiler shows AllGather on **2 worker cores** and
> ReduceScatter on 17. `num_links = 2` buys 1.48× at the 5 760 KiB payload and nothing at
> or below 640 KiB, so §8's "the default should follow production's 2" holds only for
> today's padded layout.
>
> **Rows 1 and 2 are the finding.** 184 KiB of real statistics in a 5 760 KiB tile-padded
> envelope costs *exactly* what 5 760 KiB of real data costs. The collective charges for
> padding at full price, so the paragraph above — "the deliberate trade recorded in
> `DISTRIBUTION.md` §4" — is a 7.4× overpayment on device time, not the rounding error the
> 0.595%-of-DRAM figure makes it look like. Folding the candidate axis into the last dim
> takes the collective from 348 µs to 47 µs, ~300 µs per read, against two `ttnn.permute`
> calls at ~40–120 µs traced. §6's deferral was right untraced and is wrong traced.
>
> The four rows fit `29 + 35.6·(S+1) µs` at `links = 1` to within 2% — one candidate slot
> is `2560 × 32 × 4 B = 320 KiB` padded, and there are `2(S+1)` of them. Over the same
> 186-read schedule (`Σ(S+1) = 1002`) that is **41.1 ms per forward, not §4's modelled
> 9.85 ms** — 10.8% of the 380 ms the op actually takes (§6's amendment), against the 4.6%
> this section claimed of a 215.5 ms floor it does not reach. Independent check: the slope
> difference between the traced `(2,4)` and `(8,1)` fits is 58 µs per `(S+1)`, of which
> this model accounts for 35.6 and the fp32 typecast pair and slices for the rest.
>
> The folded layout is `[1, 1, T/R, 2(S+1)]`, whose padded envelope is 320 KiB **for every
> `S` up to 15** — so the collective becomes ~47 µs flat, **8.7 ms per forward**, and stops
> scaling with the candidate count at all.
>
> **Amended again 2026-07-30 (P6 — the fold implemented, not modelled).** 8.7 ms is the
> collective's own cost and not the fold's worth: getting into and out of the layout costs
> two `ttnn.permute` calls at ~153 µs per read at `S = 8`, so the net is **147.6 µs per read
> (4.5%)**, fitting `18.6·(S+1) − 18` µs, or **15.3 ms of the 380 ms forward**. Half of what
> the 348 → 47 µs row above implies, and the error is this section's habit in miniature:
> pricing a collective in isolation and forgetting that a layout has to be reached. The
> permutes track the *padded* tensor exactly as the collective does, so the two terms shrink
> together and the fold's advantage never widens at small `S`.

---

## 5. Which collective the op actually gets **[measured]**

`ttnn.all_reduce` is not one algorithm. `all_reduce.cpp:42-45` resolves the topology and
forwards to `all_reduce_async`, which at `all_reduce_async.cpp:359` branches:

- **reduce-scatter + all-gather** when `finding_scatter_dim` finds a dim divisible by
  the participant count and neither composite predicate fires;
- **composite all-gather + `local_sum_float32` + two reshapes** otherwise.

`finding_scatter_dim` (`all_reduce_async.cpp:33-62`) converts the padded shape to *tile
units* — dividing the last two dims by 32 — then scans **from the last dim backwards**
for one divisible by `R`. If none matches it returns `logical_rank`, which fails the
`dim != composite_dim` guard and forces the composite path.

For the statistics tensor at `R = 4`, tile units are `[1, 2(S+1), ⌈T_local/32⌉, 1]`:

| shape | tile units | dims ÷ 4 | path |
|---|---|---|---|
| `[1, 2, 32, 1]` (`T=64`, `S=0`) | `[1, 2, 1, 1]` | none | composite |
| `[1, 4, 32, 1]` (`T=64`, `S=1`) | `[1, 4, 1, 1]` | dim 1 | RS+AG |
| `[1, 6, 32, 1]` (`T=64`, `S=2`) | `[1, 6, 1, 1]` | none | composite |
| `[1, 18, 32, 1]` (`T=64`, `S=8`) | `[1, 18, 1, 1]` | none | composite |
| `[1, 2…18, 2560, 1]` (`T=5120`, any `S`) | `[1, 2…18, 80, 1]` | dim 2 | RS+AG |

Measured on the `(2, 4)` box, 50 back-to-back reductions after a discarded warm-up:

| shape | padded `B` | µs/reduction |
|---|---|---|
| `[1, 2, 32, 1]` | 8 KiB | 769 |
| `[1, 18, 32, 1]` | 72 KiB | 770 |
| `[1, 2, 2560, 1]` | 640 KiB | 395 |
| `[1, 18, 2560, 1]` | 5 760 KiB | 481 |

The 8 KiB reduction is **1.9× slower than the 720× larger one**. That is not a bandwidth
curve; it is two different algorithms, and it lands exactly where the table above
predicts — composite on the `T = 32` rows, RS+AG on the `T = 2560` rows. Two extra
programs (the reshape pair) cost more than 5.7 MB of fabric traffic.

Two consequences:

1. **The distributed suite exercises a different collective than production does.** At
   `T = 64` the mesh tests spend most of their parametrizations on the composite path;
   at `T = 5120` production never touches it. The suite is therefore *better* coverage
   than intended — both algorithms are proven correct — and *useless* for timing.
2. `2(S+1)` is divisible by 4 only for odd `S`, so at small `T_local` the algorithm
   flips with the parity of the sealed count. Any Phase-9 measurement that sweeps `S`
   at small `T` will see a sawtooth that has nothing to do with AttnRes.

---

## 6. The launch term **[measured]**

`forward` at `S ≥ 1` is 16 ttnn calls on a single device — concat, two `mul`+`sum` pairs,
four for the reciprocal RMS and score, five for the hand-rolled softmax, two for the
mixture — plus 6 more for the statistics collective (concat, typecast, all-reduce, two
slices, typecast back). Call it **22 calls**; §5 shows the collective is itself 2–4
device programs, so ~25 programs.

Measured in the same loop as §5, a plain `ttnn.mul` on the same tensors:

| tensor | µs per `ttnn.mul` |
|---|---|
| `[1, 2, 32, 1]` fp32 (8 KiB) | 174 |
| `[1, 18, 32, 1]` fp32 (72 KiB) | 137 |
| `[1, 2, 2560, 1]` fp32 (640 KiB) | 137 |
| `[1, 18, 2560, 1]` fp32 (5 760 KiB) | 130 |

**Flat across a 720× size range.** ~130 µs per launch is host time — Python, the ttnn op
infrastructure, and program fan-out to 8 devices — not device time. It is also the
control that makes §5's numbers mean anything: without it, 481 µs for a reduction could
have been fabric.

The break-even is sharp. §3 gives a 1 935 µs DRAM floor per read at production shape on
`(2, 4)`, spread over 22 launches:

```
1935 µs / 22 launches  =  88 µs per launch before launches dominate DRAM
```

- **Untraced, Python-driven** — the regime the harness runs in today — 130 µs > 88 µs.
  The composed op is **launch-bound even at production shape**: 22 × 130 µs = 2.86 ms
  per read against a 1.94 ms floor, and 0.53 s per 186-read forward.
- **Traced** — the production target — 1.281 µs per program, 69× under break-even.
  ~25 × 1.281 µs = 32 µs against 1 935 µs. Firmly **DRAM-bound**, dispatch at 1.7%.

So the answer to "what binds this op" is *it depends on tracing*, and the two regimes
are 100× apart. Any Phase-9 number reported without saying which regime produced it is
uninterpretable.

> **Amended 2026-07-30 (Phase 9, P1–P3 — measured on hardware).** The last paragraph is
> the durable part of this section. The bullet above it is wrong twice over.
>
> **The break-even is not a break-even.** Dividing a DRAM floor by a launch count assumes
> host and device costs *add*. Dispatch is pipelined, so they do not: the cost is
> `max(host, device)`. Measured at production shape on `(2, 4)`, `S = 8` — host enqueue
> 3 348 µs, device-only (traced) 3 282 µs. Within 2%. The op is not launch-bound there; it
> is *balanced*, and this section's 1.47× penalty is really 1.02×.
>
> **A launch is 152 µs on eight devices, not 130.** 105 µs on one device — so the
> 8-device fan-out this section could not decompose is ~46 µs of it. The 22-call figure
> and the "flat across a 720× size range" finding both hold.
>
> **The conclusion survives, relocated.** Untraced totals on eight devices pin flat at a
> **2.2–3.6 ms host floor regardless of `S`**, so at `S = 1` the op waits 3.6× its device
> time on Python. `S` ramps 0→8 with `mean(S+1) = 5.39`, so most of the 186 reads live in
> exactly that regime. Over the real schedule: **622 ms untraced against 380 ms traced**,
> and tracing is worth **1.64× per forward** while being worth 1.00× at the shape this
> section chose to reason about. Peak shape is where fixed costs matter least — the one
> shape guaranteed to hide the launch term.
>
> **And "DRAM-bound traced" is confirmed with a number:** 2 766 µs measured on `(8, 1)`
> against §3's 1 935 µs floor is **70% of DRAM peak** (59% with TP). Dispatch traced is
> 2.6–10 µs, ≤0.3%.
>
> Both prices below need re-reading in that light. D10's stands — it compares launch
> counts and the ratio is unchanged, though the count is **28.3 programs per read**, not
> ~25 or D10's ~12. The statistics-payload deferral **does not**: §4's amendment shows the
> fabric saving is ~300 µs traced rather than 83 µs, against permutes that cost ~40–120 µs
> of device time rather than 260 µs of host time. The same decision inverts between the two
> regimes, which is the sharpest available argument for the paragraph above.

Two design decisions now have prices instead of arguments:

- **D10 — candidates on dim 1 rather than a list of `S+1` tensors.** The rejected form
  was ~90 launches per read instead of ~16. Untraced that is 11.7 ms versus 2.1 ms per
  read; traced, 115 µs versus 21 µs against a 1 935 µs floor. D10 was the right call and
  it is worth 5.6× in the regime we actually run in.
- **The deferred 18× statistics-payload fix** (`DISTRIBUTION.md` §4: permute the
  candidate axis into the last dim). It buys ~5.4 MB of fabric traffic worth 83 µs, and
  costs two `ttnn.permute` launches worth 260 µs untraced. Deferring it was correct, and
  §5 shows why more strongly than the byte count did: extra programs are the expensive
  thing on this path, not extra bytes.

---

## 7. The fusion ceiling — what Phase 10 is actually worth

A fused kernel reads `v` once and writes the mixed output once. `block_residual` and
`prefix_sum` are already in DRAM and the output is `V/(S+1)`, so the floor is
`V·(1 + 1/(S+1))` bytes against the composed form's `12V`:

| placement | `v` resident | % of 173.0 MB L1 | fused floor | composed | ratio |
|---|---|---|---|---|---|
| `(2, 4)`, `T = 5120` | 82.6 MB | 47.7% | 91.8 MB | 990.9 MB | **10.8×** |
| `(8, 4)`, `T = 5120` | 20.6 MB | 11.9% | 22.9 MB | 247.7 MB | **10.8×** |
| `(8, 4)`, `T = 20480` | 82.6 MB | 47.7% | 91.8 MB | 990.9 MB | **10.8×** |

The candidate set fits in aggregate L1 at every shape that matters — comfortably on
Galaxy, at half capacity on LoudBox — which is the precondition for the fusion being
real rather than a DRAM-to-DRAM rewrite. 215.5 ms of whole-forward DRAM floor becomes
20 ms.

10.8× is large enough that Phase 10 should not be decided on taste. It is also not
free: the fused kernel owns the two `d`-reductions, the cross-candidate softmax, and the
mixture, and the composed form has to stay as its oracle.

The `inter_block` + `merge` split is a *different* optimization and does not compound
with this one. It amortizes the sealed half's reciprocal-RMS pass across 12 layers,
which a fused kernel that reads `v` once has already collapsed.

> **Amended 2026-07-30 (Phase 9).** 10.8× is floor-to-floor. What Phase 10 would actually
> buy is measured against what runs today, and there are two corrections in opposite
> directions:
>
> - The composed form reaches only **70% of its own floor** (2 766 µs traced on `(8, 1)`
>   against 1 935 µs), so if the fused kernel hits its floor the realizable win is
>   **~7.6×**, not 10.8×.
> - It must be compared against **380 ms traced**, not the 622 ms the harness sees
>   untraced. Quoting the untraced number would inflate Phase 10 to ~12× and credit the
>   kernel for work `ttnn.begin_trace_capture` does for free.
>
> Two cheaper levers come first and neither is in this section: the 1.43× of headroom
> inside the composed form (76% of device time is 7 big-tensor ops; `mul(v,v)` + `sum`
> alone is 1 041 µs per read) and the statistics fold in §4's amendment. The split form's
> **1.43×** on a mesh (P5, re-measured after the fold landed) is real and, as this section
> says, does not compound with fusion.

---

## 8. What this memo does not know

> **Amended 2026-07-30 (Phase 9).** This list was written before any device time existed.
> Five of its seven items have since been measured; see §4's and §6's amendments and
> `bringup_log.md` §Phase 9 perf loop. Superseded items are struck through below. What is
> still open: `(8, 4)` and `[LINE, RING]`, decode, the PP boundary, real K3 weights, the
> `N`-batched matmul, and whether the collective's per-call global-semaphore creation is
> what makes its enqueue 481 µs against a 152 µs baseline.

- ~~**No profiler has run.**~~ **Superseded.** Tracy ran on `(2, 4)` at `S = 8`:
  650 programs over 23 reads, 28.3 per read, 76% of device time in 7 big-tensor ops, 13%
  in the collective, and the whole statistics path at 23% of device time for 0.6% of the
  bytes. Traced device time is 70% of §3's DRAM floor on `(8, 1)`. Read the floors below
  as "no faster than", and note the composed form now gets within 1.43× of one.
- Every DRAM and fabric figure here is still a floor from a cited peak bandwidth, not
  device time. The 512 GB/s row is a spec number; the DRAM microbenchmark it comes from
  targets 90% of it (`test_dram_read.cpp:283` and the goal at
  `6_dram_offchip/test_dram_offchip.cpp:331-332`). §4's fabric column turned out to be
  the loosest of them — 2.7× off, because a collective reaching 20% of link peak is
  core-limited long before it is link-limited.
- **The two measured numbers are host wall clock**, 50 iterations enqueued with one
  synchronize, on an otherwise idle box. They are throughput per call in a stream of
  calls, not one-shot latency, and they include Python. Phase 9 keeps the method and adds
  the traced counterpart, which is the only configuration that separates the two terms.
- ~~**~130 µs per launch is not decomposed.**~~ **Partly measured.** It is 105 µs on one
  device and 152 µs on eight, so 8-device fan-out is ~46 µs. Still not split between
  Python and the ttnn op infrastructure. The `ttnn.all_reduce` candidate is now a sharper
  target than it was: its enqueue costs 481 µs against a 152 µs baseline, and per-call
  global-semaphore creation is the obvious suspect for the ~3× — the analog hoists that
  out with `create_global_semaphores` (`tt_ccl.py`). Untested.
- ~~**`num_links` defaults to 1** while production uses 2.~~ **Answered — 1 is correct.**
  P4: 2 links buys 1.48× at the 5 760 KiB payload and nothing at or below 640 KiB. The fold
  landed (P6), so there is no payload left for a second link to help: on the real op it
  buys 4.7 µs per read on top of the fold. Taking the layout instead of the link is also
  the better Galaxy trade — the fabric is contended by dispatch/combine there.
- ~~**Nothing is measured at production `T` on a mesh.**~~ **Superseded.** Everything in
  Phase 9 is `T = 5120` on `(1, 1)`, `(8, 1)` and `(2, 4)`, which §5's rule puts on the
  RS+AG path — the one production takes. The correctness suite still runs `T = 64` and
  still exercises the other algorithm.
- **`(8, 4)` and `[LINE, RING]` are modelled, not measured.** A ring axis sustains 2
  directions and halves §4's Galaxy column; that has never been run. `(4, 2)` was skipped
  deliberately — it is between two measured points on both axes.
- **The `N`-batched matmul is unmeasured**, so the split form's remaining headroom above
  its measured 1.43× is unknown.
- No decode (`T = 1`), no PP boundary, no real K3 weights.

~~Phase 9 owns turning §3, §4 and §6 into device time. The first thing it should measure
is the launch term, because §6 says that is what decides which of the other two matters.~~

Phase 9 did that, and measuring the launch term first was the right order for the wrong
reason: it did not decide which of the other two matters, it revealed that the question
was posed at the wrong shape. §3's floor survived at 1.43×, §4's fabric term did not
survive at all, and §6's verdict inverted. The numbered iterations, their tables and the
four refutations are in `bringup_log.md` §Phase 9 perf loop.
