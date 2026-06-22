# X280 Pull-Based Profiler — Design Notes & Decision Framework

Design reasoning for using the X280 (L2CPU running Linux) as a **pull-based
observability engine** that drains Tensix per-core profiler markers over the NoC.
Companion to [`X280_BANDWIDTH_SUMMARY.md`](./X280_BANDWIDTH_SUMMARY.md) (measured
BW) and [`PERF_REPORT.md`](./PERF_REPORT.md) (pull-path progression). This doc is
the "why / how to decide" layer; those are the "what we measured" layer.

---

## 1. The hard constraint: X280 NoC reads BLOCK

- The X280's only NoC mechanism is **TLB-window memory-mapped loads/stores** (no
  DMA engine; software NIU cmd-buf issue disproven on the L2CPU's own NIU).
- A NoC **read = a CPU load instruction**. Its whole job is to put a value in a
  register, so the **in-order** core stalls until the data returns. Uncached
  System-Port loads are strongly ordered → **~1 read outstanding per hart**, each a
  full **~309–364 ns NoC round-trip**.
- **Reads overlap WITHIN a 64 B flit** (issue 8× u64 in one expression → one
  transaction) but **NOT across flits** on one hart. Across flits you add harts.
- **Writes are posted** (fire-and-forget) — they do NOT stall the hart. Export
  (X280→host) is writes; pull (L1→X280) is reads. That asymmetry drives everything.
- Per-hart read ceiling ~175–207 MB/s; **aggregate ceiling ~430 MB/s at 3 harts**
  (shared NoC0 port). 4 harts collapse (Linux needs ≥1). **Budget = 3 worker harts.**

## 2. Device profiler buffer facts (from `profiler_common.h`)

- `PROFILER_L1_BUFFER_SIZE = 2048` → **2 KB per RISC**.
- Marker = 8 B (`PROFILER_L1_MARKER_UINT32_SIZE = 2`). 256 markers/buffer = **250
  usable** ("optional") + 4 guaranteed + 2 program-id.
- `DeviceZoneScopedN` = 2 markers (begin+end) = **16 B** → **~125 zones / RISC**.
- A core's two DM RISCs (BRISC+NCRISC) = 2×2 KB = **~4 KB/core** (matches our 4 KB
  pull benchmark).
- Kernel-duration envelope the profiler is designed for: **100 ns – 10 ms**
  (`FALLBACK_MIN_NS`/`FALLBACK_MAX_NS`). **Sub-µs kernels are below any pull sample
  rate** → event/push capture only.

## 3. Why the naive fill→flush→handshake model is bad

Fill the 2 KB buffer, then bulk-pull it, then signal the core "go":
- **Flush = X280 hart blocked on serial flit reads.** 2 KB = 32 flits × ~309 ns =
  **~10 µs** (1 core, 1 hart); 4 KB ≈ ~20 µs; **all 110 cores @ 4 KB, 3 harts ≈ 1 ms**.
  The "go-ahead" is a single posted write (~free); the cost is the reads.
- In a single-buffer handshake the **Tensix core also stalls** that whole ~10 µs →
  ~100× perturbation of a 100 ns loop. The act of profiling dominates the workload.
- A buffer fills in ~12–25 µs but the all-core flush is ~1 ms → buffers **overflow
  ~40–80× faster than you can drain them**.

## 4. The fix: lock-free flit-ring (SPSC), X280 as multi-hart consumer

Treat each core's profiler buffer as a **circular buffer, cell = 1 flit (64 B)**:
- Tensix **writes while `w` is behind `r`** (don't overwrite unread cells); never
  bulk-stalls — it streams into the ring.
- **Put `r` (read ptr) in Tensix L1**, written by X280 over NoC; Tensix checks free
  space with a **local load** (no NoC, no stall). **Put `w` in Tensix L1**, written
  locally by Tensix; X280 reads it over NoC.
- **X280: 2 consumer harts** pipeline flit reads (one after another → 2 outstanding,
  hides latency); **3rd hart** polls `w` and publishes `r`. → **all NoC cost on the
  X280; the profiled core pays ~zero NoC penalty.**

Single-core rates (100 ns/zone = 4 zones/flit = 1 flit/400 ns):
- Produce: 64 B / 400 ns = **160 MB/s**.  Consume (1 hart): 64 B / ~317 ns = **~202
  MB/s** → **one hart already out-paces the producer.** 2nd hart = jitter headroom,
  3rd = pointers. Lossless, neither side stalls. Ring depth (32 flits ≈ 3.2 µs)
  absorbs scheduling jitter.

**Unverified caveat:** our 2× hart scaling was measured reading ACROSS many tiles.
Two harts on the SAME tile may serialize at that tile's L1/NIU response path —
MEASURE single-tile 2-hart scaling before relying on >1 consumer hart. (1 hart is
enough for the 100 ns case anyway.)

**Scope:** ~3 harts per core → this is a **deep single-core (lossless) tracer**, not
a grid sampler.

## 5. Grid-wide: three modes, none dominates

| Mode | Coverage | Rate | Loss / cost |
|---|---|---|---|
| **Thin sweep** (1 flit/core) | all 110 | ~16 µs/sweep, 3 harts (~62 kHz); ~40 µs, 1 hart | **samples** — drops ~97.5% of markers at 100 ns cadence |
| **Flit-ring drain** (§4) | ~2–3 cores | continuous | **lossless**, ~free, on those cores only |
| **Backpressure / stall-for-lossless** | all 110 | runs at drain rate | **lossless** but **~40× workload slowdown** at 100 ns/zone |
| 4 KB full flush | all 110 | ~1 ms | lossless only if aggregate produce < ~430 MB/s |

## 6. The unifying idea: ~430 MB/s drain IS the grid's marker budget

Aggregate marker production vs the X280's ~430 MB/s read ceiling decides the regime:

```
Produce (1 zone/100 ns × 110 cores) = ~17.6 GB/s
Drain (3 harts)                      = ~0.43 GB/s
Overhead (if backpressured lossless) = produce / drain ≈ 40×
Lossless-capacity (if not stalling)  = drain / per-core = ~2-3 cores
```

- **At or under budget** (~430 MB/s aggregate = **~1 zone per ~4 µs/core**,
  ~62 k zones/s/core): lossless, ~zero overhead, full grid. ← the sweet spot.
- **Over budget:** either **drop** (thin sweep / sampling, no slowdown, distorts
  nothing but loses markers) or **stall** (backpressure, lossless, slows the
  workload 1:1 with the excess). Stall cost is pure throttling — the spin is a
  cheap *local* `r` load, no NoC traffic.

## 7. Decision framework

- **Perf profiling (timing must be real):** stay **under budget** — coarsen zones or
  instrument fewer cores. A 40×-slowed workload profiles a *different* workload
  (caches, races, NoC contention all change). Never backpressure a perf run.
- **Functional / causal tracing (ordering matters, timing doesn't):** backpressure
  is fine — eat the slowdown, keep every marker.
- **Deep-dive one hot core:** flit-ring (§4), lossless, ~free.
- **Statistical grid picture:** thin sweep (~62 kHz), accept sampling.
- **Sub-µs kernels, or dense lossless grid-wide:** NOT pullable — use the on-device
  **push-to-DRAM** profiler (each core streams its own markers, never waits on a
  remote puller).
- **Hybrid (recommended default):** thin sweep across the grid + flit-ring deep-trace
  on the 1–2 cores under investigation.

## 8. Export side (the other half — see BANDWIDTH_SUMMARY)

Pulled data still has to leave the chip. Device DRAM is off-limits under X280 Linux;
SLIRP network is ~14.5 MB/s. The validated fast path is the **D2H socket** (X280
NoC-writes → PCIe tile → host pinned FIFO): **raw 268 MB/s** (2 harts), **sustained
lossless stream ~158 MB/s** (host-drain-limited). So the export budget (~158 MB/s
lossless) is *narrower* than the pull budget (~430 MB/s) — and both share the 4
harts. Ship deltas/markers, not full buffers.

## 9. Open experiments (for a fresh session)

1. **Same-tile 2-hart read scaling** — does the flit-ring get 2× on one tile, or does
   the target serialize? (Gates whether >1 consumer hart helps per core.)
2. **Prototype the flit-ring** — tiny Tensix producer kernel (flit-ring writer,
   local `r` check) + X280 2-consumer/1-pointer reader; measure sustained lossless
   marker rate on one core, and the producer's added overhead.
3. **`CMD_BUF_AVAIL` under a real workload** — confirm worker cores claim all 4 NoC
   cmd buffers in practice (rules out remote-NIU-push as a concurrent mechanism).
4. **Read-side NoC1 split** — `noc_sel` to push the ~430 MB/s pull ceiling toward
   ~860 (write-side split gave no gain; read side untested).

---

## 10. Prototype results — SPSC flit-ring, built & measured (2026-06-19, bh-3/P100A)

We built the §4 flit-ring end to end and measured it on silicon. Code:
`tt_metal/programming_examples/x280_spsc/` (producer kernels + host launcher) and
`tools/tracy/x280/x280_ring_consumer{,1}.c` + `x280_ring_grid.c` (X280 consumers).
All numbers: AICLK 1.349987 GHz, 1 *record* per 64B flit (stress config — real
profiling packs ~8 markers/flit, so multiply marker rates by ~8).

### Step 1 — rate-limited producer (`kernels/producer.cpp`)
BRISC of Tensix logical (0,0)=NOC0 (1,2) publishes an 8B record (seq + wall-clock,
seq written last = torn-read-safe commit) every ~400 ns, paced off the wall-clock
debug reg `0xFFB121F0`. Verified: 539–540 ticks/record (target 540) — rock-solid.

### Step 2 — SPSC flit-ring with backpressure (`kernels/ring_producer.cpp`)
32-cell × 64B ring in L1; `w` (producer→X280) and `r` (X280→producer) both in the
core's L1. Producer **blocks while `w-r ≥ N`** (never overwrites unread). Cell[k] =
`word[0]=word[15]=k` (commit + integrity), `word[1]=ts`; `fence` before bumping `w`.
- Host-as-consumer self-check: 0 errors, backpressure exercised, producer parked at
  exactly `r+N` — backpressure bound is precise.
- **X280 3-hart consumer, lossless e2e: 8.18M flits in 5s, 0 errors** (1 pointer hart
  reads `w`/publishes `r=min(flusher progress)`; 2 flusher harts read flits + verify).
- Key fix: consumer must start at the producer's current **`r`** (oldest unread), not
  `w` — blocking guarantees `[r,w)` is always valid, so there is no stale-slot risk.

### Single-tile drain ceiling (answers §9.1, §9.2)
Reading **one tile** harder does NOT scale — it is NIU/L1-response bound:

| consumer (one Tensix tile) | ns/flit | Mflit/s | MB/s |
|---|---|---|---|
| 1 hart, per-flit `w`-read + data + `r`-write | 1189 | 0.84 | 54 |
| 1 hart, pointer ops amortized (batch=32) | 748 | 1.34 | 86 |
| 3-hart (orig) | 611 | 1.64 | 105 |
| 3-hart + **cache-line separation** + batched publish | **495–510** | **1.96–2.02** | 126–129 |
| *ref:* pure idle read, no protocol (`pollall`, cross-core) | 364 | 2.74 | 175 |

- The big multi-hart win was **cache-line separation** of `g_w`/`f_next[]` (false
  sharing), not batching. Two flushers on one tile give only ~1.5×, not 2× — **same-
  tile reads are contention-bound; ceiling ≈ 2.0 Mflit/s ≈ 128 MB/s.**
- At 2.0 Mflit/s the puller sits just *below* one core's 400 ns/marker max (2.5M), so
  a saturated 100 ns-zone core backpressures it ~20% — but ×8 markers/flit that is
  ~16M markers/s, far above realistic zone rates.

### Step 3 — grid-wide, 1 hart per physical column-band (`x280_ring_grid.c`)
The scaling axis: don't pile harts on one tile — give each hart a **disjoint physical
region** (independent NIUs + disjoint NoC routes). Ring producer on all 110 cores
(11×10); 3 X280 consumer *processes* (multi-process = zero shared state), each pinned
to a hart over a disjoint NoC-X column-band, each using a disjoint TLB-window block.
- **2.76 Mflit/s = 176 MB/s aggregate across all 110 cores, 0 errors (lossless full
  grid).** Regions 40/30/40 cores → 0.93/0.90/0.93 M each. Beats single-tile 2-hart
  (2.0M) *and* covers the whole grid. ≈25k flit/s/core (≈200k markers/s/core) drained.
- **Caveat:** still ~2.4× below the pure-poll grid ceiling (`pollall` 3-hart = 6.7
  Mflit/s / 430 MB/s). Gap = SPSC protocol overhead (per-core-visit `w`-read+`r`-write
  amortized over too few flits, drain bookkeeping, producer L1-port contention from
  110 cores writing while harts read). Only bites when producers are *unpaced*; at
  realistic zone rates the grid is far under budget.

### Net
Lossless, in-order, backpressured pull over the real NoC is **proven** at both deep-
single-core and full-grid scale, with the spatial-partition (hart-per-region) model
as the right grid architecture. Open optimization levers: amortize per-core pointer
ops, cross-tile pipelining (issue reads to several tiles before consuming), read-side
NoC1 split — to close the gap to the ~430 MB/s pure-read ceiling.
