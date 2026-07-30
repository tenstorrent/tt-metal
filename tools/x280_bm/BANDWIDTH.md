<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# How the X280 drain reaches 1.5–1.8 GB/s

One page. Every lever that carries the bandwidth, every lever that turned out not to, and the ceilings that
bound the design. Numbers are measured, not modelled; the derivations live in [`FINDINGS.md`](FINDINGS.md)
sections referenced per item.

## Where the bandwidth comes from

**1. Keep many NoC reads outstanding.** The single decisive fact. One hart on the uncached System Port,
reads-in-flight 1→2→4→8: **246 → 447 → 793 → 914 MB/s** (§11). This is what proved the long-quoted
"530 MB/s mesh wall" was false: it was `poll4`'s dependent read→consume→next loop keeping *one* read
outstanding, not a link limit.

> **Superseded mechanism, surviving conclusion.** The shipped reader has **no explicit ILP loop**. It uses a
> single `vsetvli e32, m8` + `vle32.v/vse32.v` copy (`profzone.c` `copy_words`), and at VLEN=512 one such
> instruction has 128 words (512 B) in flight — more outstanding than the hand-unrolled ILP-8 experiments
> achieved. This is why `rdrbench` later measured explicit **ILP 1→16 as a no-op**: once the reader does wide
> contiguous vector copies of whole rings, there is no latency left to hide and the path is bandwidth-bound.
> ILP was the diagnostic; vector width is the implementation.

**2. Two reader harts. Not three, not four.** Sequential grid: 2h = 1782, 3h = 1818, but **4h collapses to
~275 MB/s** — and the *same* 16 outstanding reads from 2 harts gives 1782, so it is contending **issuers**,
not outstanding count, that thrashes the port. Real scatter drain: 2h = 1533, and ≥3 harts crater
(3h = 455, 4h = 277). `rdrbench` confirmed 2 is a hard ceiling with 3 being **7.7× worse** (§11, §12).

**3. Stage through LIM so each hart owns one NoC direction.** Reader = NoC-read + cheap *local* LIM store;
relay = cheap *local* LIM load + NoC write. Dropping LIM and writing grid→host directly is **worse: 753 MB/s
(2 harts), 294 (4 harts)** — an in-order hart cannot overlap NoC reads with NoC writes, and the posted write
consumes the outstanding-read budget that was hiding latency. LIM staging is not a buffer, it is what buys
specialization, and that beats avoiding the double LIM crossing (§16, row l).

**4. Wide vector copies.** `vsetvli e32,m8`: reader **249 → 7.9 cyc/word**, relay **52 → ~2 cyc/word**
(§21). Production readers now sit at **~4.2–4.5 cyc/word**, which is the figure that reflects the shipped
design.

**5. Batched relay — descriptor ring, not per-flit.** A contiguous flit ring plus a per-ring descriptor lets
the relay do one descriptor read and one wide contiguous copy per 64-flit ring, replacing 64× (destination
read + scattered posted write). **1097 → 1197 MB/s** (§16, row g).

**6. Split the NoC directions.** Read on NOC0, write on NOC1 — two-sided, and worth the 1097 peak on its own
(§16, row d).

**7. Two relays / two D2H sockets, to delete the funnel.** A single relay runs **92.6% busy** at ~1.2–1.3
GB/s and *is* the bottleneck. Dual relay drops relays to ~59% busy, readers rise to ~77% at 4.5 cyc/word →
**~1.8 GB/s aggregate read**, at which point the wall is the fundamental NoC read (§22).

**8. Fewer bytes per marker.** 2-word (8 B) markers with identity moved *out* of the marker into sticky
packets (STICKY_SRC/TIMER 1 word, PROG 2), reconstructed on the host by forward-filling. Combined with
**BULKCORE** — the reader ships a whole core's five rings verbatim and the host does the packet walk —
this removes per-marker on-device reshape entirely. That removal collapsed producer stall duration
**~2.5 ms → ~12 µs (~200×)** (§22, and `x280_rt_profiler_backpressure`).

## Correctness work that was a prerequisite for sustained rate

- **Whole-snapshot contiguous relay drain.** Drain one reader's entire published snapshot before switching
  readers. Fixed marker **misattribution** — the tell was an exact marker count with 27% sequence gaps,
  originally misdiagnosed as cross-hart cache incoherence (§21 correction).
- **Per-ring host drain + non-blocking round-robin.** Each reader owns a host ring, its own HSENT/HACKED and
  posted window; N host threads drain in parallel. Relay `hostfull` **270k → ~13k** spins (20× less).
- **Adaptive bulk-vs-per-RISC** switch per core (threshold: 4× rings pending).
- Two silicon-only bugs: tear-free 64-bit wall-clock read, and reserving ring room **before** timestamping
  (a full-ring stall was back-dating the marker that followed it).

## Host side — without this the device never reaches its rate

- **BroadcastRing decoupling of the Tracy sink.** Every X280 stage is lossless-*blocking*, so a slow sink
  propagated to silicon. **826 producer stalls → 0**, relay0 HOST-WAIT **15,852 ms → 0** on UFLD-v2 (§26).
- **Bounded pages per read (1024).** An unbounded read spends ~10 ms per pass with the D2H FIFO unpolled.
  **557 stalls → 0** at the knee (§27).

## Levers that did nothing (the negative space is half the value)

| Tried | Result |
|---|---|
| Cached Memory Port instead of System Port | **worse** — 472 vs 793 MB/s |
| Static VC pinning per hart | no change (552 either way) |
| Scatter reads across different cores | a wash — 1423 vs 1495; the System Port already overlaps to one endpoint |
| Explicit ILP on the relay copy | no gain (1206 vs 1197) |
| Deeper flit buffer, 8→64 rings | no gain (1207) — not stalls, steady-state bound |
| Second relay hart (two pipelines) | only +6% (1267) — not relay-capacity bound |
| Third reader hart | no help (1192) |
| DMA engine as the relay | **412 MB/s**, ~7× worse than a hart's posted `vse64` — the DMAC is core-offload, not throughput |
| Direct grid→host, no LIM | **worse** — 753 / 294 MB/s |
| NoC0/NoC1 read split, explicit ILP 1→16 (`rdrbench`) | both **no-ops** — already bandwidth-bound |

## The ceilings that bound the design

| Limit | Value |
|---|---|
| Raw 2-hart NoC read | ~1490–1533 MB/s (≈2.24 GB/s `rdrbench` asymptote) |
| D2H export, X280→host write | ~3.0 GB/s @ 1 hart — **never** the bottleneck |
| LIM SRAM, crossed twice by store-and-forward | ~2.5 GB/s → ~1.2–1.27 GB/s |
| Production, 2 readers + 2 relays | **~1.8 GB/s** aggregate read |
| End-to-end knee, production path (2026-07-29) | **~1.5 GB/s** (§27) |

## In one line

Wide vector reads keeping many NoC requests in flight, held to **two** reader harts, each hart specialized to
one NoC direction through LIM staging, with batched relay copies across **two** sockets and an 8-byte
raw-bulk wire format — is what turns a supposed 530 MB/s wall into a sustained 1.5–1.8 GB/s.

**The remaining wall is genuinely the NoC read**, confirmed twice by independent routes: `test_x280_gridilp`
measured 1.5 GB/s for the real scatter drain, and the end-to-end production knee lands on the same number.
Every host-side and hart-topology lever is exhausted, so **the only lever left is bytes per marker.**
