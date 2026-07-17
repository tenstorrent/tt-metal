# X280 Reader-Drain Microbenchmark — Results

Standalone bare-metal microbenchmark of the X280 round-robin **reader** in
isolation (no relay / D2H). `N` reader harts each own a disjoint slice of the
core grid and, per core, POLL the 16-word control region then DRAIN `K` markers
(16 B self-describing, 4 words each) from that core's L1 buffer **into the
reader's own streaming 256 KiB LIM SPSC ring** (advancing producer offset,
wrapping — models the real SPSC-write cost, not a hot overwritten scratch). No
live producer: each core's L1 is pre-filled with a static buffer, so the same
fill serves every `K`.

- FW: `tools/x280_bm/src/rdrbench.c`
- Host: `tt_metal/programming_examples/profiler/test_x280_rdrbench/`
- Box: **bh-11** (single-chip P100A, 11×10 = 110 worker cores), branch `x280-on-yusuf`
- Run gotcha: each boot needs a fresh `tt-smi -r <dev>` (re-asserting reset on a
  `wfi`'d L2CPU does not cleanly restart it). Then run with `--no-reset`.

## Headline curve — 2 readers, NoC0, ILP=1, streaming SPSC

Sweeping `K` from 4 to 5000 markers/core in one boot (110 cores, 200 rounds).
`K=0` is the poll-only lower bound (round-robin overhead, no marker drain).

```
┌──────────────────┬──────────┬───────────┬───────────┬──────────────┐
│ K (markers/core) │ µs/sweep │ Mmarker/s │ ns/marker │ GB/s (drain) │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 0 (poll-only)    │ 3.45     │ —         │ —         │ —            │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 4                │ 6.60     │ 66.7      │ 15.0      │ 1.06         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 8                │ 9.68     │ 90.9      │ 11.0      │ 1.45         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 16               │ 15.84    │ 111.1     │ 9.0       │ 1.78         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 32               │ 28.16    │ 125.0     │ 8.0       │ 2.00         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 64               │ 53.50    │ 131.6     │ 7.6       │ 2.11         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 128              │ 102.78   │ 137.0     │ 7.3       │ 2.18         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 256              │ 202.75   │ 138.9     │ 7.2       │ 2.21         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 512              │ 405.50   │ 138.9     │ 7.2       │ 2.23         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 1024             │ 811.01   │ 138.9     │ 7.2       │ 2.24         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 2048             │ 1599.49  │ 140.8     │ 7.1       │ 2.24         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 4096             │ 3198.98  │ 140.8     │ 7.1       │ 2.24         │
├──────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ 5000             │ 3905.00  │ 140.8     │ 7.1       │ 2.24         │
└──────────────────┴──────────┴───────────┴───────────┴──────────────┘
```

`µs/sweep` and `Mmarker/s` are computed from the two host-measured columns
(`µs/sweep = K × 110 × ns/marker`; `Mmarker/s = 1000 / ns/marker`) — the host
prints only `GB/s` and `ns/marker`, so these two columns carry no independent
measurement, just the definitional identity.

## Reading the curve

- **Asymptote ≈ 2.24 GB/s / 140.8 Mmarker/s / 7.1 ns per 16 B marker**, reached
  by **K ≈ 512**. Everything from 512 → 5000 is flat (2.23 → 2.24, within
  rounding); extending 10× past the old K=500 endpoint bought nothing. The old
  hot-scratch K=500 number (2.18 GB/s) was already ~97 % of the ceiling.
- **Knee at K ≈ 32–64** (2.00 → 2.11 GB/s). The last ~6 % of bandwidth costs an
  8× deeper buffer (K 64 → 512).
- **Poll-only floor 3.45 µs/sweep** ≈ 31 ns per core-poll (a 64 B ctrl NoC read),
  paid every sweep regardless of fill. Small buffers are dominated by this fixed
  per-core cost amortized over few markers, which is why ns/marker halves
  (15.0 → 7.1) from K=4 to K≥2048.

## What this pins down

- The **streaming SPSC write is free**: pushing each marker into a real 256 KiB
  LIM ring (fresh, wrapping addresses) lands within ~2–3 % of the earlier
  hot-scratch numbers at every K. LIM write bandwidth comfortably exceeds the
  read rate, so the write never gates. The reader is **NoC-read-bound**.
- Prior sweeps (see `../../.claude` memory / commit history) proved the other
  levers are dead ends: **4 readers** collapse ~8× (shared-L2CPU thrash),
  **NoC0/NoC1 split** is a no-op, and **ILP 1→16** is a no-op (bandwidth-bound,
  no read latency to hide). So 2 readers on NoC0 at ILP=1 is the operating point.
- **Real-workload implication:** live workers emit only ~4–6 markers/op, so the
  reader runs in the steep small-K regime (~11–15 ns/marker, ~1.1–1.4 GB/s) —
  mostly paying the poll to drain a handful of markers. The only lever left is
  **fewer bytes read**: shrink the marker (drop the pad word: −25 %; raw 8 B:
  ~halves the read) and/or visit each core less often but drain more per visit
  (walk up the K-curve).
