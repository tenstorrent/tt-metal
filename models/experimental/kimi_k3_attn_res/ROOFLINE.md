# AttnRes roofline — Phase 8

What the composed op *can* cost, so Phase 9 has a number to be measured against
rather than a feeling. Analytical, from constants that are cited or measured — no
profiler has run. `DISTRIBUTION.md` covers the mapping; this covers the cost of it.

Two things here are measurements, not derivations, and they are marked **[measured]**:
which algorithm `ttnn.all_reduce` actually picks for the statistics tensor, and what
one op launch costs in the regime the harness runs in. Both changed the conclusion.

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

---

## 8. What this memo does not know

- **No profiler has run.** Every DRAM and fabric figure is a floor from a cited peak
  bandwidth, not device time. The 512 GB/s row is a spec number; the DRAM
  microbenchmark it comes from targets 90% of it
  (`test_dram_read.cpp:283` and the goal at `6_dram_offchip/test_dram_offchip.cpp:331-332`).
  Read every µs here as "no faster than".
- **The two measured numbers are host wall clock**, 50 iterations enqueued with one
  synchronize, on an otherwise idle box. They are throughput per call in a stream of
  calls, not one-shot latency, and they include Python.
- **~130 µs per launch is not decomposed.** How much is Python, how much is ttnn op
  infrastructure, and how much is 8-device fan-out is unknown; the traced golden says
  the device-side floor is 1.281 µs, so essentially all of it is host. `ttnn.all_reduce`
  creating its global semaphores per call is one candidate — the analog hoists that out
  with `create_global_semaphores` (`tt_ccl.py`) — and is untested here.
- **`num_links` defaults to 1 in the op** while production uses 2. §4's table is at 2;
  the op as written pays double. Not yet changed, because changing it without a mesh
  perf harness would be a guess.
- **Nothing is measured at production `T` on a mesh.** §3 and §4 at `T = 5120` are
  arithmetic. The suite runs `T = 64`, which §5 shows is a different collective
  algorithm.
- **`(8, 4)` and `[LINE, RING]` are modelled, not measured.** A ring axis sustains 2
  directions and halves §4's Galaxy column; that has never been run.
- No decode (`T = 1`), no PP boundary, no real K3 weights.

Phase 9 owns turning §3, §4 and §6 into device time. The first thing it should measure
is the launch term, because §6 says that is what decides which of the other two matters.
