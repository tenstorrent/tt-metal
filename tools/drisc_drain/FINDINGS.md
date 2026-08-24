> **STATUS BANNER (2026-08-07). READ §N+21 FIRST.**
> This file is an audit log, not a summary. It contains claims that were later **retracted** —
> deliberately kept, because the reasoning that killed them is the most useful content here.
> **§N+21 "CONSOLIDATED STATE" is the current truth and wins any conflict with sections above it.**
>
> Dead — do not resurrect: egress bandwidth · ingest/producer delay · cumulative runs · config churn ·
> NoC choice · host poll pressure · the periodic device read (§N+18) · static-TLB immunity ·
> degradation-follows-a-hang · knee-as-safety-limit.
>
> Three DISTINCT failures, never pool them: **WEDGE** (card `Unknown|63`) · **TEARDOWN** (healthy card,
> core-wait hangs) · **DEGRADED** (13× MMIO latency, needs a box freeze).
>
> **§N+24 SOLVES the slow-dispatch TEARDOWN** and overrides §N+21 B: it was the harness passing
> `--gx 0` (= 12x10 under slow dispatch) while the drainer polls 11 columns. Harness fixed; slow cells are
> now 10/10 clean and slow dispatch is unblocked. **All previously recorded slow-dispatch cells are void.**
> `14-2..14-11` are TENSIX WORKERS (one column), not the DRAM/DRISC column.
>
> **§N+25: the DRISC drainer now polls the FULL 120-core grid by default** (verified lossless),
> which makes `--gx 0` safe by construction on that arm. `TT_METAL_PERF_DEBUG_FULL_GRID=1` is gone,
> replaced by the opt-out `TT_METAL_PERF_DEBUG_RESERVE_COLUMN=1`. Tensix still always reserves.
>
> **§N+38 (newest): the knee is set by the WORST sweep's CREDIT-WAIT (71-93% of it), not by read or proc
> (17.8 us of 125 us) — optimize the worst sweep, not the mean.** The per-read page cap must clear at least
> one whole FRAME (`kPagesPerSlot` = 165): cap 176 gives knee 150 → 100, cap 165 and below stalls badly,
> cap 0 is broken. Derive the default from `kPagesPerSlot`, not the magic 1024.
>
> **§N+37: the REAL knee is 1 drainer ~250, 2 drainers ~150 (~1.7x, not 5x).** "Knee 20" is
> RETRACTED — it was measured on an MMIO-degraded card, where 2.3 us host WRITES stretch the serial
> slow-dispatch launch ~13x and DESYNCHRONIZE the producers, collapsing peak concurrent rate. Forcing a
> Gen1 link (slow reads, fast writes) does the opposite and pushes the knee past 150, which is how the
> mechanism was identified. **Quote a knee only with the dispatch mode AND the ACK-WRITE probe value.**
>
> **§N+36: on a HEALTHY card the HOST IS NOT THE CONSTRAINT** — writer 70-83% idle, decoder ~60%,
> while producers stall 22,000 times; the limit is the device's O(num_cores) sweep. Several host claims from
> earlier the same day are RETRACTED as degraded-card artifacts (the "2.6 us ack", the reads-not-bytes
> framing diagnosis, and "pacing gives 0 stalls at 20-125"). **Check the ACK-WRITE probe printed on every
> run — healthy ~175 ns, degraded ~2,300 ns — before trusting ANY timing.** What survives: `resize`
> eliminated (9.6 ms -> 0.0), and three instrument bugs fixed.
>
> **§N+34: the multi-drainer bring-up hang is FIXED** — it was one `LaunchProgram` per NIU flip,
> so the second flip's `dram_barrier` ran across the first drainer's already-stream-mode core. All flips
> now go in one launch. 80/80 clean (was 42/80 + a host freeze). **`kNSockets = 2` is now the default and
> the knee is 20, not 60** — N+30's "knee 60" is RETRACTED, it was measured on a sweep whose own failed
> runs were depressing it. The single-drainer wedge is untouched and still open.
>
> **§N+29/§N+30: a DRISC drainer is only safe on a DRAM core in row `y == 0`** (exactly two
> exist: `0-0`, `9-0`; y!=0 hangs at 12.8%, p~0.006). **Two drainers lower the knee 100 -> 60 (1.67x)
> but FROZE THE HOST and left the card DEGRADED** — `kNSockets` stays at 1. Before trusting any timing
> measured after a freeze, check the ACK-WRITE probe: healthy ~170-190 ns, degraded ~2300 ns.
>
> **§N+26/§N+27 kill BOTH standing wedge hypotheses.** The IOMMU page-fault lead is
> decorrelated in both directions — do not resurrect it. The drainer/`dram_membar` subchannel collision
> is REFUTED and inverted: bank 0 (which collides) is the safe default at 0/61, bank 1 (which does not)
> hangs at 15.6%. **Keep `TT_METAL_PERF_DEBUG_DRISC_BANK` at 0**, and validate every bank before running
> more than one drainer. Also in §N+26: WEDGE / MMIO-timeout / NocHangError are ONE state, so cascades
> must be **scored per event, not per run** — per-run scoring turned a p~0.001 effect into "noise".

<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# DRISC profiler drainer — measurements

Can a Blackhole DRAM-core RISC (DRISC) host the device-side profiler drainer: pull markers out of
worker-core L1 over the NoC, and push them to the host?

Everything below was measured on **bh-07** (`yyzo-bh-07`), firmware bundle 19.12.0, 120-core logical
worker grid (12x10 — the arch max is 140, this die has 2 harvested Tensix columns), single DRISC on
bank 0's free subchannel unless stated otherwise.

## Running it

```
cmake -B build -DTT_METAL_BUILD_TESTS=ON
cmake --build build --target unit_tests_api -j$(nproc)
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_api --gtest_filter="*DRISC*"
```

`TT_METAL_SLOW_DISPATCH_MODE=1` is mandatory — `BlackholeSingleCardFixture` skips without it.

Two preconditions, either of which silently turns everything into a skip:

- **Firmware bundle >= 19.12.0.** Below that, `should_enable_blackhole_dram_programmable_cores()`
  turns DRAM programmable cores off and the fixture skips with "DRAM programmable cores not enabled".
  The floor guards a syseng-core collision on older firmware (#45751), and device init loads DRISC
  firmware on **all 16** metal-managed DRAM cores, so overriding it risks the collision at
  `CreateDevice`, not at kernel launch.
- **Stream mode.** A DRISC cannot initiate *any* NoC transaction in the default NOC2AXI mode. Every
  kernel here sets it; `NIU_CFG_0` persists across programs, so whoever sets it owns restoring it.

Sources: `tests/tt_metal/tt_metal/api/test_dram_kernels.cpp` (`DramKernelDRISCScatterFixture`) and
`tests/tt_metal/tt_metal/test_kernels/misc/drisc_{rdrbench,adaptive_drain,niu_mode}.cpp` +
`misc/socket/drisc_d2h_egress.cpp`.

## The cost model

Every number here is explained by three terms:

```
per-visit = latency (~231 ns, hidden by depth)
          + issue cost 29.46 ns   (~40 cycles, independent of payload)
          + wire time  bytes / 86.3 GB/s
```

**86.3 GB/s is `NOC_PAYLOAD_WIDTH` (512 bits = 64 B) x ~1.35 GHz.** It was not assumed — it fell out
of the K sweep as the incremental cost between the K=512 and K=1280 rows. A physical constant
reproducing itself from timing data is what justifies converting device cycles with aiclk.

Instrumentation note: `get_timestamp()` costs **26 cycles** on the DRISC, comparable to the ~40-cycle
quantity being measured. Phases are therefore separated by **ablation** (compile out one phase and
diff), never by bracketing hot code with timer reads.

## Ingest — reading markers out of worker L1

Marker = 2 words = 8 B. `K` = markers per read, `B` = reads issued before the barrier.

Per-visit cost depends on **B only, essentially not on K**: 43.23 ns at B=16 for K=1, K=4, and K=16
alike, rising only to 43.62 at K=64. A 64x payload increase costs 0.9% more time.

**The drain is issue-bound, not bandwidth-bound. Over-reading is free.** The lever is more bytes per
transaction, not fewer bytes per marker.

| K (bytes) | B=1 | B=8 | B=16 | B=32 | B=64 | B=120 (issue-all) |
|---|---|---|---|---|---|---|
| 1 (8 B) | 0.031 | 0.140 | — | 0.226 | — | 0.272 GB/s |
| 32 (256 B) | 1.004 | 4.489 | 5.922 | 7.241 | 8.149 | **8.689** |
| 64 (512 B) | — | 8.862 | 11.737 | — | 16.247 | 17.348 |
| 128 (1 KB) | — | 17.500 | 23.262 | 28.645 | — | — |
| 256 (2 KB) | 7.351 | 34.136 | 45.704 | — | — | — |
| 512 (4 KB) | 13.549 | 50.714 | — | — | — | — |
| 1280 (10 KB) | 27.412 | **67.367** | — | — | — | — |

`ns/visit` converges to **29.46 ns** at issue-all regardless of K — the pure issue floor.

Reference points from the real profiler geometry (`profiler_common.h`, `dev_msgs.h`):
`PROFILER_L1_VECTOR_SIZE` = 512 words = **2 KB = 256 markers per (core,RISC) lane**; a whole core is
64-word control vector + 5 rings = **~10.25 KB**, comfortably under the 16 KB `NOC_MAX_BURST_SIZE`.
So **K=256 is one full lane and K=1280 is a whole core in a single transaction.** K>256 is not
reachable per-lane.

### Whole-core sweeps and the depth cap

120 cores x 10 KB = 1.2288 MB per sweep.

| B | slots | ns/core | full sweep | GB/s |
|---|---|---|---|---|
| 1 | 1 | 373.57 | 44.83 µs | 27.41 |
| 4 | 4 | 185.28 | 22.23 µs | 55.27 |
| **8** | **8** | **152.46** | **18.29 µs** | **67.17** |
| 16 | 8 shared | 137.08 | 16.45 µs | 74.70 |
| 32 | 8 shared | 128.33 | 15.40 µs | 79.79 |
| 64 | 8 shared | 123.97 | 14.88 µs | 82.60 |
| 120 | 8 shared | 121.79 | 14.61 µs | 84.08 |

**B=8 is a buffer limit, not a hardware limit** — 8 x 10 KB = 80 KB against 86 KB of DRISC L1
UNRESERVED. The NIU tracks far more (`NOC_MAX_TRANSACTION_ID_COUNT` = 255 across 16 trids). Rows
marked *shared* let outstanding reads land on top of each other to expose the transport ceiling; that
corrupts data and is **not a usable drainer configuration**. The usable figure is **18.29 µs**.

At B=120 the port is 97.4% saturated (84.08 of 86.3 GB/s), with only 3.1 ns/core of non-overlapped
overhead. That is the end of the road for one NIU.

**A full whole-core sweep in 8 µs is impossible on one NIU at any depth:** 1.2288 MB / 8 µs =
153.6 GB/s = 1.78 ports. The floor for one port is 14.24 µs.

### Multi-DRISC scaling

Grid partitioned across DRISCs on separate banks, K=32 / B=8:

| DRISCs | GB/s | vs 1x |
|---|---|---|
| 1 | 4.489 | — |
| 2 | 8.671 | 1.93x |
| 4 | 17.281 | 3.85x |

Linear, with per-DRISC cycles flat. Separate DRAM banks each have their own NIU pair and do not
contend.

## The poll, and why it dominates

Round-robin read of every core's 64-word (256 B) control vector, with and without the adaptive
tail-delta arithmetic (`full += tails[r] - heads[c*NRISC+r]` over 5 lanes):

| B | read only | read + tail-deltas |
|---|---|---|
| 1 | 344.1 cyc/core → 30.59 µs | 364.1 cyc → 32.37 µs |
| 8 | 77.0 cyc → 6.84 µs | 141.4 cyc → 12.57 µs |
| 32 | 47.7 cyc → 4.24 µs | 111.0 cyc → 9.87 µs |
| **120** | **39.8 cyc → 3.54 µs** | **102.8 cyc → 9.14 µs** |

The examine work costs a stable **~63 cycles/core** (five volatile L1 loads at ~12.6 cyc each plus
the adds). **61% of poll cost is CPU, not NoC.** Shrinking the read would buy nothing — the read is at
its payload-independent 39.8-cycle floor. The lever is the arithmetic: the five tails are contiguous
words 5..9, so wider loads would replace five 32-bit volatile loads.

### Full adaptive sweep

Poll all 120, run the threshold decision (`ADAPT_THRESH` = 4 x 512 words), then one whole-core 10 KB
read per core that trips it. The host primes control vectors so a chosen number trip:

| cores hot | cyc/sweep | µs/sweep | bulk reads | KB drained | drained |
|---|---|---|---|---|---|
| 0/120 | 16,177 | **11.98** | 0 | 0 | — |
| 1/120 | 16,755 | 12.41 | 1 | 10 | 0.8 GB/s |
| 12/120 | 19,347 | 14.33 | 12 | 120 | 8.6 GB/s |
| 60/120 | 31,871 | 23.61 | 60 | 600 | 26.0 GB/s |
| 120/120 | 47,517 | 35.20 | 120 | 1200 | 34.9 GB/s |

**With zero cores drained the sweep already costs 11.98 µs.** Draining is comparatively free — the
marginal cost of a whole-core bulk read is 261 cycles (193 ns). Polling is what consumes the budget.

Consequence: **polling only pays below ~48% occupancy.** Always-bulk with no poll costs 261 cyc/core
= 23.2 µs for the full grid; poll-then-bulk-everything costs 35.20 µs. Crossover at
`hot x 261 = 16,177`, i.e. 58 cores.

This measurement is *optimistic*: heads were primed to zero, so the kernel never loads a head mirror.
A real drainer needs five more loads per core, pushing the poll floor toward ~17 µs on one DRISC.

## Egress — D2H socket

Real socket push (`socket_reserve_pages` / write through the PCIe tile / `socket_push_pages` /
`socket_notify_receiver`), DRISC as sender. Device-measured and host-observed rates agree to within
0.2% everywhere. Every configuration below is the median of 3 repeats.

**Note on repeats:** a first pass with single samples had the same configuration landing 74% apart
between sections. The cause is the host staging buffer: `sink` grows with pages-per-read, and the
first run at each new size eats first-touch page faults inside the timed loop. Revisiting a size the
allocator has already served is clean (spread 1.00x). Single samples here are worthless.

### Host-side tuning

| configuration | GB/s | device time waiting |
|---|---|---|
| 32 KB pages, 1 page per read (baseline) | 16.85 | 79.8% |
| 64 KB pages | 18.94 | 42.7% |
| 80 KB pages | 19.32 | 39.2% |
| 80 KB, 2 pages per read | 20.72 | 28.7% |
| 80 KB, 4 pages per read | 22.84 | 33.3% |
| **80 KB, 8 pages per read** | **25.36** | 26.3% |
| 80 KB, 4 pages/read, ack every 4 | 23.17 | 37.5% |
| **80 KB, no memcpy (`discard_pending_pages`)** | **57.60** | **3.5%** |

Page size and pages-per-read both help (16.85 -> 25.36 GB/s, +50%). Host ack batching does not.
80 KB is near the maximum: DRISC L1 UNRESERVED measured 88,448 B and the socket config buffer is only
64 B, so the page is bounded by L1, not by the socket.

**The memcpy inside `D2HSocket::read()` is the whole remaining gap: 25.4 -> 57.6 GB/s, 2.3x.**

### Per-page breakdown

A page costs thousands of cycles against a 26-cycle timer, so unlike the read benchmarks in-loop
timestamps are affordable here (4 probes/page, ~2%).

| phase | tuned (80 KB x 8 pg/read) | ceiling (no memcpy) |
|---|---|---|
| `socket_reserve_pages` (wait on host) | 804-891 ns (26%) | 50 ns (3.5%) |
| `noc_write_page_chunked` (issue to PCIe tile) | 1921-1925 ns (60%) | 1053 ns (74%) |
| `socket_push_pages` + `notify_receiver` | 397 ns (12%) | 274 ns (19%) |
| loop + barrier | 64 ns (2%) | 49 ns (3.4%) |
| **total** | **~3200 ns** | **1424 ns** |

**These measure where the core blocks, not intrinsic operation cost.** The writes are posted, so
`t_write` is issue cost plus whatever back-pressure the command buffer absorbs. The same device code
takes 1053 ns at the ceiling and 1921 ns under memcpy; in the 32 KB baseline it is only 111 ns,
because there the host is slow enough (1511 ns of wait) that the NoC drains before the next issue.

That decomposition shows the memcpy hurting in two roughly equal places: +808 ns of `wait` (host
slower to free FIFO space) and +899 ns of `write` (the host's PCIe reads contending with the device's
PCIe writes). **The memcpy does not merely delay the consumer, it degrades the producer.**

### Device-side notify batching: no effect

`socket_notify_receiver` is a 4 B PCIe write; `socket_push_pages` is local state. Publishing
`bytes_sent` every N pages instead of every page:

| notify every | total ns/page | write | notify | GB/s |
|---|---|---|---|---|
| 1 | 1424 | 1053 | 274 | 57.47 |
| 2 | 1423 | 1167 | 156 | 57.55 |
| 4 | 1424 | 1229 | 95 | 57.54 |
| 8 | 1423 | 1260 | 64 | 57.57 |
| 16 | 1422 | 1274 | **48** | 57.60 |

The notify phase collapses 5.7x exactly as intended -- **and the write phase absorbs precisely the
same amount.** Total is pinned at ~1423 ns; throughput does not move.

The notify was never costing wall-clock: being a posted write, it overlapped with writes still
draining. Removing it only exposes more of the underlying drain in the write phase. **On a
posted-write pipeline, blocking time migrates between phases when you remove work -- only the total
is trustworthy.** Keep notify batching anyway (it frees ~340 ns/page of DRISC time, which a combined
read+push drainer can spend on ingest), but do not expect throughput from it.

### What egress is actually bound by

1423 ns per 80 KB page is **57.6 GB/s**, rock-steady across every configuration (spread 1.00x, 3.5%
wait). That is not the 86.3 GB/s NoC port the reads hit -- it is ~90% of a **PCIe Gen5 x16** link's
64 GB/s. Egress is PCIe-bound, and no device-side software change will move it.

## Egress alternative: DMA to a GDDR buffer (leg A)

The DRISC can write its own bank's GDDR at 64 GB/s, which beats the PCIe path, needs no host in the
loop during the run, and turns a 3.98 GB DRAM view into the buffer. The catch is that **a DRISC
cannot land NoC traffic directly in GDDR**: in stream mode (required to initiate reads at all) NoC
traffic terminates at L1, and DRAM is reachable only through the L1 + DMA path. So every byte crosses
L1 twice — written by the NIU, read by the DMA engine. That double crossing is the structural risk for
this path: a buffer crossed twice can only sustain half its own bandwidth.

**It does not reproduce here. Combining the two legs costs ~10%, not half.**

### Standalone DRISC <-> GDDR DMA

From the pre-existing `DramKernelDRISCWriteToDRAM` / `ReadFromDRAM` tests:

```
Write BW: 447.89 GB/s (7 banks x 1 endpoint)   ->  64.0 GB/s per DRISC
Write BW: 458.73 GB/s (7 banks x 2 endpoints)  ->  +2.4% for twice the DRISCs
```

**One DRISC saturates its channel's DMA path.** A second DRISC on the same bank adds nothing.

### Flow

```
per batch:
  dma_async_write_barrier(cur)        wait for the prior DMA out of buffer[cur]
  N x noc.async_read -> buffer[cur]   one 10,240 B whole-core read each, all outstanding
  noc.async_read_barrier()            wait for the batch to land in L1
  dma_async_write(cur, buffer[cur] -> GDDR ring)
  cur ^= 1                            ping-pong the buffer and the DMA TX stream
```

### Where the time goes

Per batch, in ns. `kDoRead`/`kDoDma` compile out either leg, so the same batching and buffer layout is
measured three ways and the difference is attributable to the interaction, not to a changed access
pattern. All configurations repeated 3x at spread 1.00x.

| config | GB/s | dma-wait | read-issue | read-wait | dma-issue | loop | total |
|---|---|---|---|---|---|---|---|
| read only 2buf x 4 | 48.26 | 21.5 | 160.7 | 607.6 | 21.5 | 37.5 | 848.8 |
| read only 1buf x 8 | 62.08 | 22.2 | 280.7 | 964.0 | 21.5 | 31.4 | 1319.7 |
| dma only 2buf x 4 | 63.97 | 492.3 | 25.2 | 22.2 | 63.7 | 36.9 | 640.3 |
| **read+DMA 2buf x 4** | **43.18** | 33.4 | 160.0 | 648.7 | 65.2 | 41.3 | 948.6 |
| read+DMA 1buf x 8 | 33.41 | **1123.0** | 278.5 | 956.3 | 60.8 | 33.2 | 2451.8 |
| read+DMA 1buf x 4 | 29.41 | 540.4 | 157.0 | 600.7 | 61.5 | 33.1 | 1392.8 |
| read+DMA 2buf x 3 | 38.58 | 35.5 | 135.6 | 514.3 | 68.9 | 41.9 | 796.2 |
| read+DMA 2buf x 2 | 30.81 | 34.8 | 94.1 | 423.0 | 71.1 | 41.7 | 664.7 |

`dma only` reproduces the standalone test to 0.05% (63.97 vs 64.0) on a completely separate code path,
which is the cross-check that the DMA leg is being driven correctly.

At the winning configuration the DMA is essentially free: **dma-wait is 33 ns** because the DMA moves
40,960 B in 640 ns while the reads take 849 ns, so it has always finished before the buffer is needed
again. Adding DMA to the read-only case costs 99.8 ns/batch, landing as +41 ns of read-wait (the NIU
stream slows while the DMA engine pulls from L1), +44 ns of dma-issue and +12 ns of dma-wait.

### Why 4 cores per batch and not 8

Deeper reads are genuinely better — **1buf x 8 reads at 62.08 GB/s versus 48.26 at depth 4, +29%** —
but combined it loses, 33.41 vs 43.18. With one buffer the 80 KB DMA is fully exposed and `dma-wait`
goes to 1123 ns, almost exactly the 81,920 B / 64 GB/s = 1280 ns transfer. With two buffers that same
transfer hides behind the next batch's reads and costs 33 ns.

The measured trade: 1buf x 8 buys +356 ns/batch of read throughput and pays +1090 ns/batch of exposed
DMA. `1buf x 4` isolates it — same read depth as the winner, no overlap, 32% slower.

Going shallower loses too (2buf x 3, 2buf x 2), since `dma-wait` is already near zero at 2buf x 4.

**2 buffers x 4 cores is the optimum of the eight configurations, and the two constraints meet exactly
at the L1 budget:** read depth is as large as the remaining L1 allows once a second buffer is reserved
to hide the DMA. Getting both deep reads and hidden DMA needs more than 86 KB — 2 buffers x 8 cores
would want 160 KB.

## The reserved DRAM profiler region

The old DRAM-based profiler's slice is reusable as the DRISC drainer's DRAM buffer, and it already has
the right geometry. `bh_hal.cpp` puts it at a **fixed bank-relative offset**:

```
DRAM_BARRIER_BASE  = 0
DRAM_PROFILER_BASE = DRAM_BARRIER_BASE + DRAM_BARRIER_SIZE     // measured: 0x40
```

DRAM addresses are bank-relative, so that constant means "this offset inside every bank" -- exactly
what a DRISC needs, since `gddr_dma` only reaches its own bank. The host readback already matches:
`issueSlowDispatchReadFromProfilerBuffer` loops channels reading the same address from each.

Measured on bh-07: **4,800,000 B per bank (4.80 MB)** = 600k markers / 300k zones, 33.6 MB across the
7 banks. That is far smaller than a DRAM view (3.98 GB) -- at leg A's 43 GB/s a bank's slice fills in
**0.11 ms**, and it holds only 3.9 full-grid sweeps. Size scales linearly with
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` (default 1000), and `get_dram_profiler_size()` returns 0
without `TRACY_ENABLE`.

**One bank per DRISC, not one DRISC per bank.** Nothing requires using all banks; unused slices simply
sit idle (the host loop would need to skip them). But the ceiling is one drainer per bank -- 7 here,
8 on a full die -- because only the free subchannel is safe to put in stream mode, and a second DRISC
on the same bank adds 2.4% to DMA anyway.

### Host pull from the region is not viable

`cluster.read_dram_vec`, the path the existing profiler uses under slow dispatch:

| transfer | 1 channel | all 7 channels |
|---|---|---|
| 64 KB … 4.8 MB | **0.047 GB/s** | **0.047 GB/s** |

Flat across every size and identical for one channel or seven, so it is a genuine sustained rate, not
a small-transfer artifact. Draining the full 33.6 MB takes **712 ms**; a 9 MB trace takes 191 ms. That
is **~1230x slower than the device pushing** (57.6 GB/s), which is what host-initiated non-posted PCIe
reads through a TLB window cost.

Caveat: this is the slow-dispatch path, the only one available under `BlackholeSingleCardFixture`.
Production's `issueFastDispatchReadFromProfilerBuffer` uses `enqueue_read_shard_from_core`, which is
device-initiated and a different mechanism entirely. Unmeasured.

## Full circle: A -> DRAM ping/pong -> B -> host

A sweeps the whole grid into one DRAM buffer while B drains the other to host, with a 2-credit SPSC
handshake (A publishes `fill_idx` into B's L1, B publishes `drain_idx` into A's L1, A blocks while both
buffers are outstanding). A is on bank 0's free subchannel; B is on bank 1 and **NoC-reads bank 0's
DRAM through bank 0's NOC1 worker endpoint** -- it cannot use DMA, which is bank-local.

200 sweeps of 1,228,800 B = 246 MB. Warm pass:

| host pages/read | end-to-end | A | A blocked on B | B | B: dram-read / push |
|---|---|---|---|---|---|
| 1 | 19.21 | 19.52 | 51.0% | 19.21 | 52.7% / 44.6% |
| 2 | 20.17 | 20.43 | 48.7% | 20.13 | 55.3% / 42.1% |
| 4 | 20.76 | 20.86 | 47.2% | 20.69 | 57.1% / 40.6% |
| **8** | **20.85** | 20.89 | **47.1%** | 20.72 | 57.2% / 40.5% |

B is blocked on A only 0.3%, so **B is the bottleneck and A idles ~47% of the pipeline**. Host-side
batching helps only +8.5% here (vs +50% standalone) because B's DRAM-read phase already hides most of
the reserve-wait. The cross-bank DRAM read works and is fast -- 36.4 GB/s effective.

With ping/pong only, there is **no burst absorption**: A throttles to B's rate immediately. Elasticity
would need many more buffers than two, and the region holds 3.9 sweeps total.

## Direct: ingest -> host on one DRISC

The same grid sweep, but reads land straight in a socket page and get pushed -- no DRAM hop, no second
core. Page size and read batch are the same object.

| config | memcpy consumer | zero-copy (discard) |
|---|---|---|
| 2 buf x 4 cores (40 KB page) | 21.88 | 23.18 |
| **1 buf x 8 cores (80 KB page)** | **24.51** | **27.71** |

Phases at the winner, memcpy vs discard: wbar 12.7 / 14.1%, read 36.9 / 41.8%,
**reserve 5.2 / 1.7%**, push 42.8 / 40.3%.

**Removing the host memcpy is worth only +13% here, against +132% in the egress-only test.** The
reserve column explains it: at 5.2% the host is barely stalling the DRISC, because **the NoC reads have
already absorbed the host's slowness**. Egress-only had nothing else to do, so every host inefficiency
landed directly on reserve-wait.

So the direct path is **device-bound at ~28 GB/s, not host-bound** -- read, push and write-barrier are
all serial on one core, which is 98% busy. Past that needs a second DRISC on another slice of the grid
(the linear-scaling case), not a split pipeline.

Note the inversion against leg A: here `1 buf x 8` beats `2 buf x 4`, because the buffer is freed by a
slow PCIe write rather than a 640 ns DMA, so the larger page outweighs the overlap.

## Scaling the direct drainer: ingest fans out, egress does not

N DRISCs, each on its own bank, each owning a slice of the grid and pushing to **its own D2H socket** --
a fan-out of readers in which every reader is also its own egress path, so
there is no relay and no shared ring. The grid splits by whole 8-core pages (15 of them), as evenly as
15/N allows. Host drains all sockets round-robin on one thread.

| DRISCs | slice (pages) | device GB/s | host GB/s | worst reserve |
|---|---|---|---|---|
| 1 | 15 | 24.02 | 24.43 | 7.7% |
| **2** | 8+7 | **25.00** | **25.01** | 46.7% |
| 3 | 5+5+5 | 24.29 | 25.01 | 64.2% |
| 4 | 4+4+4+3 | 24.11 | 24.13 | 70.3% |
| 1, zero-copy consumer | 15 | 27.74 | 27.75 | 1.7% |

**Flat at 24-25 GB/s from 1 to 4 DRISCs.** The reserve column is the whole story: 7.7% -> 46.7% ->
64.2% -> 70.3%. Each added DRISC spends proportionally more of its life blocked waiting for FIFO space.
The work redistributes; the total does not move.

This is the opposite of ingest, which scaled 1.93x / 3.85x at 2 and 4 DRISCs, and the difference is
structural: **ingest is N independent NoC paths, egress is one PCIe link and one host consumer.** The
Reader fan-out only helps when reads are the constraint; here the constraint is downstream of
every reader, so nothing upstream can move it.

Practical consequence: add DRISCs to poll the grid faster or cover more cores, never to get more data
to the host. One drainer already holds essentially the entire egress budget.

### Hazard: several senders with an unthrottled consumer wedge the card

Sweeping the zero-copy consumer (`discard_pending_pages`) past one DRISC is **reproducibly fatal**.
Two DRISCs pushing with nothing throttling them kills the machine at the same point every time:

```
MMIO per-op timeout: 4B load took 220218 us (budget=2 ms)
```

It survived hardening the poll loop, so it is not host polling pressure. The mechanism fits the
numbers: one DRISC alone pushes 27.7 GB/s and the link ceiling is ~57.6 GB/s (~90% of PCIe Gen5 x16),
so two unthrottled senders demand roughly the whole link, the host's small non-posted MMIO reads starve
to 220 ms, and UMD's 2 ms budget trips. With the memcpy consumer the senders are throttled to ~47%
reserve and the link never saturates.

The zero-copy variant is therefore swept at N=1 only. Recovery is `tt-smi -r`.

## Two-tier adaptive drainer: monitor, then drain at the right granularity

Per sweep: poll every core's 64-word control vector, then per core either take the **whole core in one
bulk read** (control vector plus all five rings, 10,496 B, one NoC packet) when *any* lane is at or
above 70% of its ring, or **read only the valid run of each non-empty lane**.

The partial tier exists for egress, not ingest. A read costs ~40 cycles regardless of payload, so
per-lane reads are not much cheaper to issue -- but they fetch only live markers, and egress is the
scarce resource. Bulk-reading a near-empty core spends 10 KB of host bandwidth to deliver a few
hundred bytes. Variable-size items are packed into fixed-size socket pages; an item that does not fit
flushes the page, so pushed bytes can exceed valid bytes.

Swept by uniform per-lane occupancy so the 358-word threshold crossing is visible. The kernel's page
count and valid-byte count were checked against an independent host-side model of the same packing.

| per-lane fill | tier | us/sweep | useful GB/s | pad | poll | decide | fetch | push |
|---|---|---|---|---|---|---|---|---|
| 8 w (1.6%) | PARTIAL | 56.07 | 0.34 | 0.6% | 10.8% | 18.7% | 0.1% | 0.8% |
| 64 w (12.5%) | PARTIAL | 59.98 | 2.56 | 0.0% | 10.1% | 17.5% | 1.1% | 6.1% |
| 256 w (50%) | PARTIAL | 73.62 | 8.35 | 0.0% | 8.2% | 14.2% | 3.7% | 19.9% |
| 358 w (69.9%) | BULK | 66.31 | 18.99 | 0.0% | 9.1% | 15.8% | 8.6% | 46.5% |
| 480 w (93.8%) | BULK | 66.34 | 18.99 | 0.0% | 9.1% | 15.8% | 8.6% | 46.5% |

**The partial tier wastes nothing** -- every byte pushed is a live marker. **The bulk tier's useful
throughput is flat** from 70% to 94% occupancy, because it moves the whole ring regardless of how much
is in it: at 358 words it transfers 10,496 B to deliver 7,160 B of markers.

**The sparse floor is ~56 us/sweep even at 1.6% occupancy**, where almost nothing moves. Poll is only
11% of that; the rest is the per-core loop issuing up to five reads each -- 600 tiny reads per sweep,
each paying the same ~40-cycle issue as a large one. That is the per-lane issue cost, showing up as
latency rather than bandwidth.

### Page size must be a multiple of the bulk item

A 40,960 B page cannot hold a whole number of 10,496 B whole-core items -- three fit and 9,472 B is
padding on every page. Making the page **41,984 B (4 x 10,496)** costs nothing and packs exactly:

| page | us/sweep | useful GB/s | pushed | pad |
|---|---|---|---|---|
| 40,960 | 78.93 | 15.96 | 20.76 | 23.1% |
| **41,984** | **66.31** | **18.99** | 18.99 | **0.0%** |

**+19% useful throughput** for a one-constant change. The partial tier is unaffected.

### Hysteresis removes the redundant poll and buys 0.5%

On the bulk tier the poll is redundant: a bulk read re-fetches the same control vector it just polled,
so every hot core is visited twice. Hysteresis skips the poll for any core that went bulk last sweep
and re-evaluates its tier from the control vector its bulk read already carried.

| tier | hysteresis | us/sweep | useful | polls/sweep | poll | decide |
|---|---|---|---|---|---|---|
| BULK 358 w | off | 66.31 | 18.99 | 120 | 9.1% | 15.8% |
| BULK 358 w | **on** | 65.99 | 19.09 | **1** | **1.5%** | **5.0%** |
| PARTIAL 256 w | off | 73.62 | 8.35 | 120 | 8.2% | 14.2% |
| PARTIAL 256 w | on | 77.04 | 7.97 | 120 | 8.8% | 14.2% |

It works exactly as designed -- polls drop 120x and poll+decide collapses from 24.9% to 6.5% of the
sweep -- **and throughput moves 0.5%.** Push rises 46.5% -> 48.4% and fetch 8.6% -> 11.1%: the removed
work was overlapped, not on the critical path. On the partial tier it is a 5% *loss*, since no core is
ever bulk so all 120 are still polled and the state check is pure overhead.

**Not worth the state machine.** The 8-9% poll share was real phase occupancy but not real cost.

## End to end: a DRISC services REAL producers

Every measurement above drained **synthetic** tails the host had written into the control vector. This
is the first run against the real producer path -- Tensix RISCs emitting zones through the ordinary
`DeviceZoneScopedN` macro, with a DRISC as the only consumer.

Test: `DramKernelDRISCScatterFixture.DRISCServicesRealProfiledWorkers`, kernels
`profiler_zone_producer.cpp` (BRISC + NCRISC), `profiler_zone_producer_compute.cpp` (TRISC0-2) and
`drisc_service_workers.cpp` (DRISC). **All five SPSC lanes of every core are live**, which is what
exercises per-lane indexing and the five-head write-back rather than just lane 0.

**The test cannot pass by accident.** `kernel_profiler.hpp`'s producer BLOCKS on a full ring by design
("a profiled run REQUIRES the consumer to be draining"). A wrong tail offset, a missing head
write-back, or a desynced mirror does not produce a bad number -- it hangs.

Two rounds, back to back on one device session:

| | 4x4, 2000 zones/lane | full grid, 500 zones/lane |
|---|---|---|
| producing lanes | 80 | **600** |
| words drained | 644,748 (2.52 MB) | 1,210,724 (4.73 MB) |
| tails advanced this round | 644,748 -- **exact** | 1,210,724 -- **exact** |
| lanes still behind | 0 | 0 |
| lanes silent | 0 | 0 |
| ring overflows | 0 | 0 |
| **max run observed** | **511 / 512** | **511 / 512** |
| device time | 11.2 ms | 75.3 ms |

**Max run 511 of 512 is the result.** The producers sat at the ring ceiling for the whole run: they
really were blocking in `ring_ensure_room`, and the head write-back really is what released them. The
loop closes -- and it closes for **600 lanes against a single DRISC**.

`lanes silent = 0` means no ring was left untouched. `tails advanced == words drained` on both rounds
is exact conservation: nothing lost, nothing double-counted.

The second round is deliberately not a repeat. It starts with heads already at ~644K words, so it is
the regression test for monotonic-tail seeding -- a drainer assuming a zero-based stream computes a
garbage run here.

Word counts check out: 16 x 5 x 2000 x 4 = 640,000 expected vs 644,748 seen, and
120 x 5 x 500 x 4 = 1,200,000 vs 1,210,724. The excess is FW wrapper zones and stickies.

**The drainer currently paces the workload.** The full grid completed in **6 productive sweeps**; the
other 4,096 were the quiet window used to detect completion. Producers spent most of their life blocked
waiting for the next sweep, because this drainer serializes a 10 KB read plus a barrier per core and
does no batching -- it was written for correctness, not rate. Closing that gap is what the tiering and
pacing results above already measured.

### Three things this run settled

1. **Tails are monotonic for the whole FW session.** `init_profiler` seeds `wIndex` from L1 once and
   explicitly does *not* re-read `TAIL_INDEX` per launch. A drainer must therefore never assume the
   stream starts at zero -- seed the mirror from the worker's **heads**, which ride free in the 256 B
   control vector the poll already fetches. (This is also the honest answer to "why head loads?": once,
   to seed, never per sweep.)

2. **Poll-then-drain -- but NOT for the reason first claimed here.** This drainer polls the control
   vector, then bulk-reads the **rings only**, not the control vector a second time.

   An earlier version of this section claimed a fused read of the whole 10,496 B span was unsafe
   because a fresh tail could pair with stale data. **That is backwards.** `profiler_msg_t` places the
   control vector at *lower* addresses than the rings, so one burst samples the tail **before** the
   data -- the tail is conservative relative to the data it authorises, which is the safe direction.
   The unsafe order (data first, tail second) is not what this layout produces. Wrap is covered
   separately: the producer blocks rather than overwrite `[head, tail)`, since head has not moved.

   So the fused single-read shape is **probably safe**, and worth pursuing -- it deletes the poll
   outright, roughly halving transactions, and cost here is issue-dominated (~29.5 ns/read regardless
   of payload). What it rests on is whether a NoC burst samples its source in address order. That is
   an assumption, not a verified fact: confirm it in the NoC spec before relying on it.

3. **A third run mode had to exist.** Producers emit whenever `get_profiler_enabled()` is set, but both
   pre-existing modes ship a competing consumer: default starts `RealtimeProfilerManager`, and
   `TT_METAL_PERF_DEBUG_PROFILER=1` boots the perf-debug drainer. Two consumers on one SPSC ring corrupt each other.
   `TT_METAL_DRISC_PROFILER=1` now disables both the RT manager (`mesh_device.cpp`) and the DRAM
   profiler's per-program control-buffer reset (`profiler.cpp`), which would otherwise rewind the ring
   tail mid-drain. Note `TT_METAL_NO_RT_PROFILER` is read **nowhere** in the tree and never disabled
   anything.

Run it with:

```
TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_DRISC_PROFILER=1 \
  ./build_Release/test/tt_metal/unit_tests_api --gtest_filter='*DRISCServicesRealProfiledWorkers*'
```

Still open: only BRISC produced, so 4 of 5 lanes per core were empty; no host egress (the drained
bytes are counted and checksummed, not shipped); and no decode of the marker stream.

## End to end to the HOST: framed egress over the D2H socket

The servicing test proved the flow-control loop closes but discarded the payload -- it could only
count. This carries every word off the device and re-attributes it host-side.

Test `DramKernelDRISCScatterFixture.DRISCDrainsRealWorkersToHost`, kernel
`socket/drisc_drain_to_host.cpp`, wire format in `test_kernels/misc/drisc_drain_frame.h`.

| | 4x4 / 2000 zones | full grid / 500 zones |
|---|---|---|
| host received | 646,108 words | 1,210,592 words (4.62 MB) |
| device sent | 646,108 | 1,210,592 |
| **producers advanced** | **646,108** | **1,210,592** |
| lanes reconciled | 80 | **600** |
| frames / pages | 1,686 / 341 | 2,441 / 652 |
| malformed frames | 0 | 0 |
| ring overflows | 0 | 0 |

**Three independent counters agree exactly**: the host decode, the drainer's own tally, and the tails
the *producers* advanced. Two of those could agree and both be wrong; the third closes it.

The per-lane reconciliation is the assertion with teeth. Totals alone cannot catch a mis-labelled
frame -- swap two frames' identities and every total still balances. Reconciling each
`(core_xy, lane)` against that lane's own tail advance cannot be satisfied that way, and all 600 lanes
match.

### Format

A page is a flat run of frames; a frame is a 2-word header (`kind | lane | nwords`, then
`core_xy = (y<<16)|x`) followed by exactly the live words. `KIND_PAD` terminates a page.

Three decisions worth keeping:

1. **Identity is free.** `SPSC_CORE_XY` is already in the 256 B control vector the poll fetches, so the
   drainer never constructs or injects identity -- it copies a word the core wrote about itself, so no
   sticky-source machinery is needed at all.

2. **Read per-lane straight into the page.** The alternative -- one whole-core 10 KB read into staging,
   then a local copy of the live words into the page -- costs a device-side memcpy and ships dead ring
   space. Each read instead lands exactly where it belongs. This deliberately trades *more read issues*
   (up to 5/core rather than 1) for *fewer bytes*, the opposite of the pure-ingest tuning above,
   because here the bytes have to cross PCIe.

3. **Ring wrap is resolved on the device**, by splitting the run into two reads that land contiguously.
   The host never needs the ring geometry, so the decoder stays a flat walk.

4. **One shared header for the wire format**, included by kernel and host both. Duplicating those
   constants is exactly how readers rot -- each copy self-consistent, all of them wrong.

### Cost of the framing

652 pages x 8 KB = 5.34 MB shipped for 4.62 MB of payload, so **~87% efficiency**. Headers are
negligible (2,441 frames x 8 B = 19 KB); essentially all of the 0.72 MB overhead is page padding,
because a page is flushed whenever the next frame would not fit and a frame can be up to 514 words.
Two ways to close it if it matters: let frames straddle pages (costs a reassembly buffer on the host),
or raise the page size. Not tuned here -- correctness first.

Throughput is again **not** meaningful from this run: the device figure is dominated by the quiet
sweeps used to detect completion, and the host wall (0.076 s) includes them.

## Zone decode: the stream is real

The framed egress carried words; this parses them back into zones host-side, against `spsc_packet.h`
(the same definitions the device's `ppfmt` uses -- not a host copy).

Decoder walk per lane: `STICKY_TIMER` (1 word, sets the wall-clock HIGH half for everything after it),
`ZONE_START/END/TOTAL` (2 words: `type | 16-bit srcloc`, then the LOW half), `STICKY_PROG` (1 word),
`DATA/EVENT` (2 + size). START pushes, END pops.

Full grid: **604,940 markers -> 302,410 zones across 600 lanes**, and the srcloc histogram reconciles
exactly against what each kernel ran:

| srcloc | zones | expected |
|---|---|---|
| `0xaee8` (compute kernel) | 180,000 | 360 TRISC lanes x 500 = **180,000** |
| `0x6401` (data movement) | 120,000 | 240 DM lanes x 500 = **120,000** |
| `0x7fff` (`PROFILER_STALL_ZONE`) | 1,810 | producers blocking on full rings |

4x4 likewise exact: 96,000 = 48 x 2000 and 64,000 = 32 x 2000. Zero unknown types, unmatched ends,
srcloc mismatches or backwards timestamps.

### Two bugs it caught that word-counting could not

Both were invisible to the egress test, which passed while the payload was quietly wrong -- a warning
about what "the totals reconcile" does and does not prove.

**1. The head was published before the reads completed.** Advancing `head = tail` while payload reads
are in flight frees the producer to overwrite the very slots being read: a use-after-free on the ring.
It does not fail cleanly. The stream stays plausible, then a word that is really the middle of a later
marker is parsed as a header and the walk desynchronises. **596 of 600 lanes died.** Fixed by
completing the read before publishing.

**2. NoC alignment.** Reading each run straight into the page put src at `ring_base + si*4` and dst at
`page + doff*4` -- arbitrary 4-byte offsets, violating `L1_ALIGNMENT` (16 B on Blackhole) and not
congruent with each other. **The NoC mis-delivers rather than rejects.** The symptom was a *single
substituted word* at a frame boundary -- total word count still reconciled perfectly, so only the
marker walk could see it. 440 of 600 lanes desynced.

A DRISC moves data with NoC DMA and is bound by the alignment rules, so word-granular copy idioms that
are safe on a CPU issuing plain loads and stores are not safe here. **Check the alignment rules before
porting any copy idiom to a DRISC.**

Fixed with one aligned whole-core read into staging plus a local copy of the live words into the page
-- the shape rejected earlier on efficiency grounds. Cost: 77 -> 88 ms on the full grid. The
aligned-direct variant is still available as an optimisation: round the read down to a 16 B boundary,
pad the page to match, and carry the skew in the frame header.

### Markers before the first sticky are normal

121,084 of 604,940 full-grid markers arrived before their lane's first `STICKY_TIMER`, and that is
**not** an error: the producer emits a sticky only when the clock's HIGH half CHANGES, so a capture
starting mid-stream inherits a high half it never saw. Those timestamps are unanchored and must be
excluded from ordering checks; a real consumer either drops them or back-fills from the first sticky.
Asserting they were zero was a decoder bug, not a device one.

## Verdict: the DRAM round-trip does not pay

| approach | GB/s | DRISCs | DRAM traffic |
|---|---|---|---|
| **ingest -> host, direct** | **24.51** (27.71 zero-copy) | **1** | none |
| A -> DRAM -> B -> host | 20.85 | 2 | 2x the trace |
| egress only, no ingest (reference) | 25.36 (57.60 zero-copy) | 1 | none |

Direct is **18% faster with half the cores**. Splitting across two DRISCs means neither can absorb the
other's stall, and B's DRAM read is work the direct version never performs. The DRAM buffer sits
upstream of the host wall, so it cannot raise throughput -- its only product is elasticity, and at
4.8 MB per bank that is 3.9 sweeps.

## What it means: the zone budget

A zone is 2 markers = 4 words = 16 B, and a lane's ring is 512 words, so **128 zones per lane**
before it wraps. Dividing a sweep period by 128 gives the shortest zone that can be sustained
back-to-back on one lane:

| limited by | rate | min zone duration |
|---|---|---|
| ingest alone, whole-core sweep @ 18.29 us | 67.4 GB/s | **143 ns** |
| leg A: ingest + DMA to GDDR, one DRISC | 43.18 GB/s | 222 ns |
| **whole drainer: ingest -> host, one DRISC** | **24.51 GB/s** | **392 ns** |
| whole drainer, 2-4 DRISCs each with its own socket | 24-25 GB/s | ~384 ns (flat) |
| whole drainer, same but zero-copy host | 27.71 GB/s | 346 ns |
| A -> DRAM -> B -> host, two DRISCs | 20.85 GB/s | 460 ns |
| egress only, no ingest (reference) | 25.36 GB/s | 387 ns |
| egress only, zero-copy (reference) | 57.60 GB/s | 167 ns |

The single-DRISC direct drainer at **392 ns/zone** is the end-to-end number; everything above it in
this table is a component measured in isolation.

Host tuning closed a third of the gap. **Zero-copy consumption would bring egress to 167 ns against
ingest's 143 ns -- the two sides finally comparable**, and the drainer viable at ~150-170 ns zones.
Until then egress binds by ~2.7x.

## Design implications

1. **Read whole cores, never per-lane.** Reads cost ~40 cycles regardless of payload, so five
   per-lane reads cost 5x one whole-core read that fetches the same data plus slack. Speculative
   over-reading is free, and the control vector rides along in the same transaction.
2. **Issue every read before waiting.** Depth is what hides the ~231 ns round-trip; issue-all is worth
   ~2x over B=8 on small reads.
3. **Poll is the budget, not the drain.** Optimize the tail arithmetic, and add a second adaptive tier
   that skips the poll entirely above ~48% occupancy.
4. **Do not build a reader/relay split.** Separate DRISC cores share no SRAM, so a relay would
   NoC-read the reader's L1 and pay every byte twice.
5. **Scale with DRISCs.** Scaling is linear to at least 4, and it is the only fix for poll latency,
   which does not benefit from bandwidth at all.
6. **On egress, tune the host, not the device.** 80 KB pages and 8 pages per host read are worth +50%
   together. Host ack batching and device notify batching are both worth nothing. The device write
   path is already at ~90% of the PCIe link.
7. **Add DRISCs for ingest, never for egress.** Ingest scales 1.93x/3.85x at 2/4 cores; egress is flat
   from 1 to 4 because one PCIe link and one host consumer sit downstream of every sender.
8. **Do not route the trace through DRAM to get it to the host.** Measured 18% slower than pushing
   direct, using twice the cores. The host wall is downstream of the DRAM buffer, so the round-trip
   cannot raise throughput. Use DRAM only if you specifically need elasticity, and size it knowing the
   reserved region is 4.8 MB per bank.
9. **Expect zero-copy to matter less in a combined kernel.** Worth 2.3x when the DRISC only pushes,
   +13% once ingest shares the core, because the reads already absorb the host's stalls.
10. **If the trace goes to DRAM, ping-pong two buffers.** One DRISC doing ingest + DMA runs at
   43.18 GB/s, ~10% below reads alone, and the DMA hides completely behind the reads as long as a
   second batch buffer exists. A single deeper buffer trades that away and loses 23%.
11. **Consume the FIFO in place**, but see 9 -- the payoff shrinks once ingest shares the core.
12. **Size the socket page as a whole multiple of the largest item.** A 40,960 B page against a
    10,496 B whole-core item pads 23% of every page; 41,984 B packs exactly and is worth +19%.
13. **Do not bother with poll hysteresis.** It removes the redundant poll completely and buys 0.5%,
    because the poll was overlapped with egress rather than on the critical path. The `memcpy` in `D2HSocket::read()` is the single largest remaining
   cost anywhere in the pipeline -- 2.3x -- and it penalises the producer as well as the consumer,
   because the host's PCIe reads contend with the device's PCIe writes.

## Gotchas

**A stale `LD_LIBRARY_PATH` silently shadowed the build tree.** (Diagnosed and FIXED -- an earlier
version of this note blamed `cmake --install`, which was wrong. **No install is required at all.**)

The test binary's RUNPATH is correct and self-sufficient: its first entry is
`/localdev/$USER/tt-metal/build/tt_metal`, and `build` is a symlink to the active build dir, so it
already points at exactly what `ninja` just wrote.

What defeated it was `~/.config/tt-mo/env/bh-07`, a `tt` per-target env file containing
`LD_LIBRARY_PATH=/localdev/$USER/tt-metal/lib` -- an install prefix last written days earlier. `tt run`
exports that before every command, and **`LD_LIBRARY_PATH` takes precedence over `RUNPATH`**, so every
run loaded the stale library while `ninja` reported success and the binary relinked.

This produced a false verification: a run that appeared to confirm a new env-var gate was executing a
library that had never heard of the variable. **Kernel** edits are exempt -- they are JIT-compiled from
source at runtime -- which is exactly why it hid: device-side changes took effect immediately while
host-library changes silently did not.

Fixed by repointing the env file at the build tree
(`LD_LIBRARY_PATH=/localdev/$USER/tt-metal/build/tt_metal`). Going through the `build` symlink rather
than an install prefix means it tracks whatever ninja produces and can never go stale. `bh-06` carried
the same entry and was fixed too; `bh-11` never had it.

Sanity check whenever host-side behaviour does not match the source:

```
ldd build_Release/test/tt_metal/unit_tests_api | grep libtt_metal
```

If that is not under `build/`, the run is lying to you. (Separately: `cmake --install` does fail here
with `RPATH_CHANGE could not write new RPATH`, and a failed install scatters `cmake/ include/ lib/
share/` into the repo root as untracked files -- never `git add -A` after one. But it is not on the
critical path, because nothing needs installing.)



- **Use the free subchannel.** `pick_unused_dram_logical_core(bank)` returns the non-endpoint
  subchannel — the only one safe to flip into stream mode. The other usable one is the bank's NOC1
  worker endpoint, which Tensix uses to reach that bank's DRAM. Its logical coord is in the
  preferred-first space `CreateKernel(DramConfig)` indexes, which is *not* UMD's raw subchannel order.
- **D2HSocket with a DRISC sender** goes through `ExternalConfigBuffer{address, sender_uses_physical_noc_addr=true}`
  — the flag really means "sender is not a Tensix worker" — with a **NOC0** sender coord (`CoordSystem`
  has no `PHYSICAL`; it is `NOC0`).
- **Stream mode must be set before the socket is constructed.** `ExternalConfigBuffer::address` is
  `uint32_t` and cannot carry DRISC L1's `0x2000000000` NoC tag; in stream mode inbound traffic
  terminates at L1 so plain local addresses work. Consequently every host access to DRISC L1 in the
  egress test uses the **plain** address, not `drisc_l1_noc_addr_`.
- **`socket_api.h` works on a DRISC** given the one-line
  `CBInterface cb_interface[NUM_CIRCULAR_BUFFERS]` shim (no CB infrastructure on DRAM cores) — the
  same shim the shipping `tensor_prefetcher.cpp` uses.
- **Host access to DRISC L1 changes with NIU mode.** In NOC2AXI the host must use the tagged
  `drisc_l1_noc_addr_`; a plain address in that range is forwarded to GDDR and returns stale data. In
  stream mode inbound traffic terminates at L1 and plain addresses are correct. Kernels that restore
  NOC2AXI on exit therefore need tagged readback; the egress test needs plain, because it leaves the
  NIU in stream mode. Same core, opposite convention.
- **Never run multiple DRISC senders with a zero-cost host consumer.** Two unthrottled senders
  saturate the PCIe link and starve host MMIO past UMD's 2 ms budget, reproducibly taking the card
  down. A real (throttling) consumer is what keeps the link out of saturation.
- **Do not pipe a long device test through `head`.** Closing the pipe early SIGPIPEs the test binary
  mid-run and can leave the card wedged (`Read 0xffffffff over PCIe ID 0`). Redirect to a file and
  grep the file. Recovery is `tt-smi -r`.
- **No device Tracy zones on DRISC.** `kernel_profiler.hpp` does not gate on `COMPILE_FOR_DRISC`, so
  `DeviceZoneScopedN` compiles to nothing. Use the watcher ring buffer + `get_timestamp_32b()` idiom.

## Open questions

**Does a NoC burst sample its source in address order?** The fused single-read shape (control vector +
rings in one 10,496 B transaction) depends on it, and it would delete the poll phase outright. See the
poll-then-drain note above.



- **Zero-copy host consumption** is the last big lever: 25.4 -> 57.6 GB/s, taking the minimum zone
  duration from 387 ns to 167 ns against ingest's 143 ns. `discard_pending_pages()` proves the
  ceiling but throws the data away; a real consumer needs to serialize out of the pinned FIFO
  directly. `ProcessScope::CrossProcess` already exports the FIFO for exactly this.
- Poll cost with a real head mirror (this measurement primed heads to zero, so it is optimistic by
  roughly five L1 loads per core).
- The second NIU is untouched. `Noc(uint8_t noc_id)` lets one kernel drive both, which is the only way
  a single DRISC could exceed one port's 86.3 GB/s on ingest.
- A combined read+push drainer has not been built. Ingest and egress were measured separately and
  both are near their respective walls; whether they interfere when interleaved on one core is
  unknown, and notify batching's freed ~340 ns/page would matter there.
- Attempted clock calibration from `cycles / host_wall_time` did not converge — JIT compile time
  dominates the host wall clock even at a 512 MB target. The 86.3 GB/s wire-rate match is the better
  anchor; a warm-JIT second pass would settle it directly.

## A caution on the instrumentation

Two different regimes, and they need opposite treatment:

- **Reads (~40 cycles/op)** cannot be timed with a 26-cycle timer. Separate phases by *ablation* —
  compile one out and diff the totals.
- **Egress pages (thousands of cycles)** are fine to bracket in-loop, but the writes are *posted*, so
  a phase timer records where the core blocks, not what an operation costs. Removing work migrates
  the blocking time into a neighbouring phase rather than shrinking the total. Notify batching cut
  its phase 5.7x and moved throughput 0.2%. **Only the total is trustworthy.**

## §N — The conduit drainer, and the knee at 125 (bh-05, 2026-08-04)

Single-DRISC drain of the full 110-core grid, measured end to end with per-phase counters on both sides.
**Knee moved 875 → 125** (7×) against the earlier baseline. Everything below is measured on
`test_perf_debug_zones --gx 0 --gy 0 --iters 500 --delay N`, conduit drainer, no Tracy capture.

### The design: the DRISC is a conduit, not a processor

One **fused read** per core covers the whole 10,496 B `profiler_msg_t` — control vector *and* all five
rings — in a single NoC burst (`NOC_MAX_BURST_SIZE` is 16,384 B on Blackhole: 256 words × 64 B). It then
writes that same staging slot straight to the host. It **authors zero words inside a frame**.

The fused read is legal because staleness runs the safe way: the control vector is at offset 0 and the
rings at 256+, so ascending in-burst delivery samples the tail NO LATER than the data behind it.
Everything in `[head, tail)` is already in the snapshot. The dangerous ordering — fresh tail, stale data —
is ruled out by address order. Torn reads would appear as `run > cap` and are counted, never trusted.

Identity, geometry and progress all reach the host inside the worker's own control vector. The host keeps
its own head mirror (`head(frame N) == tail(frame N-1)`, exact because the FIFO is ordered and lossless)
and uses the shipped head field as a **consistency check** — `head_drift` was 0 across ~15,000 frames.

### ★★★ The CPU copy was 45% of the drainer — never repack on a DRISC

Packing each lane's live run exactly, with CPU loads/stores out of the snapshot, cost **~11 cycles per
word**: 20.4 µs to move one core's 2,490 words. A busy sweep was **2,271 µs** against an idle sweep's 36 µs.
A `volatile tt_l1_ptr` word loop cannot be unrolled, widened or pipelined.

Removing it (ship the raw span, let the host slice) took the busy sweep **2,271 → 244 µs** and producer
stalls **21,175 → 0**. It costs shipping dead ring space, which is nearly free: the socket credit wait was
0.0% and the PCIe write 0.1% at the time. **Exact packing traded a resource costing 0.1% for one costing 45%.**

### The deadline model — occupancy is forced, not chosen

    occupancy ≈ worst sweep / ring fill time,   fill = 128 zones × zone duration

    delay 150:  94.9 / 129 us = 74%   observed 372/512 = 73%
    delay 125:  95.4 / 107 us = 89%   observed 440/512 = 86%
    delay 100:  95.0 /  86 us = 110%  saturates → stalls on all 110 cores

Three points, within ~3%. At delay 100 the sweep is LONGER than the fill time, so it physically cannot
return before the ring fills — the cliff is arithmetic, not a tuning threshold.

**Corollary: pacing does not move the knee.** A gap lengthens the revisit period, which raises occupancy
and cuts bytes, but the knee IS the point where period == fill. Pacing trades safety margin for byte
efficiency. Its value is away from the knee: at delay 900 occupancy is forced to 11% and over-send is ~9×,
where a large gap is nearly free.

### Per-sweep budget at the knee (delay 150, no host decode)

    IDLE 31.9 us = read 18.0 + proc 12.0 + barrier 0.5 + misc 1.4      (15,704 of 15,777 sweeps)
    BUSY 83.6 us = read 18.0 + proc 19.6 + transport 38.9 + notify 4.6  reserve 0.0
                   read 1.15 MB / 18.0 us = 64 GB/s   (FINDINGS ingest ceiling: 18.29 us / 67.4 GB/s)
                   transport 1.14 MB / 38.9 us = 29 GB/s (egress reference 25.36 GB/s)

`write` splits as **noc-chunk 1.82 µs/push (85%) | notify 0.29 (13%) | push_pages 0.04 (2%)**. The chunk
figure is 40.6 GB/s, which is impossible for real movement — these are POSTED writes, so it measures
*issue*; completion is the `wr-barrier` phase. Both hardware phases are at their documented ceilings.

**Sub-ring shipping is a dead end.** It would fragment ~16 contiguous pushes into up to 550 small writes
per sweep (66–200 µs of issue against 28.8 µs today) to save 1.9× on bytes, and would reintroduce the
per-lane metadata that the conduit deleted. Transport at high occupancy is irreducible with this frame.

### ★★ Stall counting must not go through the pipeline — put a counter in L1

`SPSC_STALL_COUNT_0` (8 slots, indexed by processor id) is incremented by the producer in its own stall
path and read by the host straight out of L1 at teardown. Validated against decode: **12,487 vs 12,459**
(0.2%, counter higher — the correct direction, since decode can lose events and the counter cannot).

This matters because every stall number measured through the decode path travelled DRISC frame → PCIe FIFO
→ buffer pool → decoder → BroadcastRing, and every one of those was observed dropping data (1.29 M records
at the ring alone). It also removes the host from the measurement entirely.

### The host was the bottleneck, and it was never the memcpy

Writer-thread phases at delay 300, full decode: **sock-read 3.8 ms (15.5 GB/s) | decode 13.5 | publish 8.7**.
The copy is **15%** of host work. Decode + publish are 85%, and they sat between one socket ack and the
next, so the ack rate — and hence the DRISC's credit wait — was gated by per-marker work.

Proof by construction: same device code, delay 300, decode ON → **17,366 stalls**; minimal decode → **0**.

Moving decode+publish to their own thread (bounded buffer pool, discard-and-count on exhaustion rather
than back-pressuring the device) fixed delay 900 outright (0 stalls, complete records, previously bimodal
0/5,115) and took delay 300 from 404 → 192 µs busy sweeps. It did not fully fix delay 300: the tail
(worst sweep 368 µs vs 257 µs fill) still stalls, prime suspect the reader's 50 µs poll backoff.

### Knee progression, and the negative results

| configuration | knee | zone | aggregate |
|---|---|---|---|
| prior baseline (§27) | 875 | 5.88 µs | 187 M mk/s, 1.50 GB/s |
| DRISC, per-lane sliced reads | ~750 | 5.03 µs | 219 M mk/s, 1.75 GB/s |
| conduit + stall-only decode | 250–300 | 1.68–2.01 µs | 655 M mk/s, 5.2 GB/s |
| **conduit + L1 counters, host read+ack only** | **125** | **0.84 µs** | **1.31 G mk/s, 10.5 GB/s** |

Repeatable: 125 clean ×2 (occupancy 440/512 both runs, identical), 100 stalls ×2 (1,676 / 1,509, all 110
cores). With the host in the path this configuration was bimodal (0 or ~5,000 stalls); with it out, the
busy sweep is flat at 83–86 µs across every delay and occupancy repeats exactly.

**Measured and rejected:**
- **Bigger host reads.** `MAX_PAGES` 1024 → 16384 → 0: worst credit-wait 93 µs → 1.8 ms → **25.8 ms**,
  stalls 984 → 884 → 10,010. A page cap bounds pages, not per-pass time. (Reconfirms ERA-2, §27.)
- **Page size = frame size.** Collapsed page ops 165× (2.5 M → 8.9 K) and cut bytes 41%, and bought
  NOTHING: total busy time 34.0 → 34.3 ms. It concentrated the same wait into fewer, longer stalls
  (167 → 339 µs/busy sweep) and turned 0 stalls into 952. Socket page bookkeeping is not the egress wall.
- **Unpaced bulk with the old per-lane framing.** 41× the NoC bytes of a control-vector poll; 18 GB of
  reads to find 0.3 cores/sweep with data.

### Next, in order

1. **More DRISCs** — the only lever left on the knee. `read` and `proc` are per-core costs that halve with
   a second drainer: ~95 → ~48 µs, knee ≈ 65. Bounded by egress: we are at 12 GB/s of a ~25 GB/s reference,
   so 2 works and 3 hits the wall.
2. **Pacing** — for host load and NoC traffic away from the knee, where over-send is ~9×.
3. **Reader backoff** — the 50 µs sleep is the suspect for the delay-300 latency tail.

## §N+1 — Read/write NoC split, and the two-DRISC probe (bh-05, 2026-08-04)

Both entries here are measured at **delay 150, `TT_METAL_PERF_DEBUG_NO_DECODE`, 110 producer cores**, so
they are comparable to §N's 83.6 µs busy sweep. Word conservation was exact in every run and stalls were 0
throughout, which is the loss-proof check: the L1 counters are incremented by the producer itself.

### Two DRISCs: 1.43×, and it does not move the knee (measured, then reverted)

Two drainers, disjoint halves of the worker grid, one D2H socket and one reader+decoder thread pair each,
nothing shared on the device side:

| metric | 1 DRISC | 2 DRISC |
|---|---|---|
| idle sweep | 31.9 µs | **16.2 µs** (0.51×) |
| busy sweep | 83.6 µs | 58.3 µs (1.43×) |
| **worst sweep** | ~95 µs | **~95 µs (unchanged)** |
| words | 11,002,970 | 5,501,485 + 5,501,485 |

The idle sweep halves exactly, because an idle sweep is pure per-core polling and each drainer sees half the
cores. The busy sweep does not, and the **worst** sweep — the one that actually sets the knee, since the
deadline is `worst sweep < ring fill time` — did not move at all. The reason is that only the per-core
phases (`read`, `proc`) are divided; transport is not. Splitting the same total bytes across two senders
gives each half the bytes but does not make the egress path faster, so the transport term stays put and caps
the gain. §N's prediction of "knee ≈ 65 with two DRISCs" was wrong for exactly this reason: it modelled
`read` and `proc` and forgot that transport is not per-core.

Reverted to `kNSockets = 1`; the host side stays fully parameterized by it, so raising the constant
re-enables the path.

### The NoC split: reads and writes on different NoCs

Blackhole DRISCs have both NIUs live — firmware runs `noc_local_state_init()` for every NOC and
`drisc_set_stream_mode_all()` puts both into stream mode — so the drainer can read on one and ship on the
other. `kReadNoc = NOC_INDEX == 0 ? 1 : 0`; egress stays on `NOC_INDEX`.

That alone changes nothing. The gain needs the loop rearranged so the read barrier is *last*: free the
generation about to be refilled, issue its reads, process the **previous** batch (whose writes now fly
concurrently with those reads), and only then wait for the reads. Previously the read barrier sat between
the read and the ship and forced them apart no matter which NoC each used.

| metric | 1 NoC | 2 NoC (2 runs) |
|---|---|---|
| busy sweep | 83.6 µs | **72.3 / 77.3 µs** (1.08–1.16×) |
| worst sweep | 94.9 µs | 89.8 / 94.3 µs |
| idle sweep | 31.9 µs | 28.8 / 29.1 µs |
| max occupancy | 372 | 336 / 360 |

**What it revealed is worth more than the speedup.** Write *issue* collapsed 5×: `noc-chunk` went
1.82 → 0.12 µs/push. Issuing a write had been stalling on NoC command-buffer availability while the reads
held the same NoC — invisible before, because it was charged to the write phase and looked like transport.
That time did not disappear, it moved to completion: `wr-barrier` rose 1.7% → 4.3%.

**Read these numbers with two caveats.** `unaccounted` rose to 23.1%, which is expected and not a bug: the
phases are now genuinely concurrent, so they no longer partition wall time and the counters cannot sum to
100%. And run-to-run spread is ~7% across the two samples, so **1.1× is not resolved** — it needs 3–4
repeats per configuration. The knee has not been re-measured under the split.

### Ping-pong depth is the next knob, and 2 is probably too shallow

The two staging generations *are* ping-pong — one fills while the other drains. The reason it only bought
~1.1% – 16% is likely depth. Staging is `kNStage = 7` slots of one fused span each (10,560 B), and
`kGenSlots = kNStage / 2 = 3`, so a batch is 3 worker cores and there are ~37 batches per sweep:

```
read   3 x 10,496 B @ 64 GB/s  ~ 0.5 us
proc   3 cores @ ~0.18 us      ~ 0.55 us
ship   3 x 10,560 B @ 25 GB/s  ~ 1.3 us
```

A generation is refilled **two batches** — about 2.3 µs — after its ship was issued, which is the same order
as PCIe completion latency. So the reuse barrier still blocks, which is precisely what the `wr-barrier`
rise says. Deeper buffering gives each ship more time to land:

| layout (gens × slots) | batch | reuse distance | cost |
|---|---|---|---|
| 2 × 3 (current) | 3 cores | ~2.3 µs | shallow; barrier bites |
| **3 × 2** | 2 cores | ~4.6 µs | likely sweet spot |
| 7 × 1 | 1 core | ~14 µs | 110 notifies/sweep ≈ 24 µs — swamps the gain |

`notify` is 0.22 µs per push, so depth is paid in pushes: fewer cores per batch means more pushes per sweep.
3 × 2 is where the two curves cross on paper. Untested.

### Next, in order

1. **Generation-depth sweep** {2, 3, 7} at delay 150, 3–4 repeats each, reading busy *and* worst sweep.
2. **Knee re-measure** at 125/100 under the split.
3. **Reader backoff** — the 50 µs sleep is still the suspect for the delay-300 latency tail.

## §N+2 — The static TLB lever, and what the degraded state actually costs (bh-05, 2026-08-06)

### The DRISC was on the slow host write path for no reason

`D2HSocket::read()` ends in `notify_sender()`, one 4 B device write of `bytes_acked`. The DRISC drainer paid
a **UMD dynamic-TLB reconfigure** on every one of them; the Tensix drainer did not. The socket gated its
static window on `!sender_uses_physical_noc_addr_`, and `perf_debug_profiler` sets `sender_uses_physical_noc_addr = !tensix_drain` — so
the DRISC inherited "no static TLB window exists", which is false for a DRAM core.

Measured, healthy card, warm runs:

| | ack write | path |
|---|---|---|
| DRISC before | 382 ns | DYNAMIC (reconfigure per access) |
| **DRISC after** | **172 ns** | **STATIC window** |
| Tensix (unchanged) | 171–176 ns | STATIC window |

Two things had to change. The socket now **asks** UMD (`is_tlb_mapped`, address overload) instead of
inferring from the sender kind. And the caller has to create the window at all: metal maps one 4 GB DRAM
window per channel at init, but only on the channel's **preferred worker endpoint** (`ddr_to_noc0` takes the
last of 3 NoC ports), while the drainer deliberately sits on the *unused* port — so a DRISC core is normally
unmapped. `perf_debug_profiler` now configures a 2 MB window at address 0 (spans the DRISC's 128 KB L1)
before constructing the socket.

`TT_METAL_PERF_DEBUG_NO_STATIC_TLB=1` forces the old path so the two can be A/B'd on one binary.

### The degraded state is per-ACCESS cost, not per-reconfigure — and it hits reads too

A card degraded mid-session, which finally supplied the two numbers previously recorded as unmeasured:

| probe | healthy | degraded | ratio |
|---|---|---|---|
| ack write, STATIC window | 172 ns | **2303 ns** | 13.4× |
| ack write, DYNAMIC (reconfigure) | 382 ns | 2555 ns | 6.7× |
| 4 B device read (`cluster.read_core`) | ~770 ns | **~2950 ns** | 3.8× |
| flow-control poll (host memory, control) | 13–14 ns | 13–14 ns | 1.0× |
| sock-read | 18.6 GB/s | 10.0 GB/s | 1.9× |

Three conclusions:

1. **A static window is no escape.** It degrades 13×, more than the dynamic path does in relative terms, and
   the two nearly converge (2303 vs 2555) — the ~210 ns of reconfigure is noise once an access costs 2.3 µs.
   So the penalty is attached to the **device access itself**, not to the driver's TLB work. That kills
   "dynamic TLB reconfigure is the mechanism" as a theory of the degraded state.
2. **Reads are hit too** (3.8×), so it is not write-specific — but the ratios differ (13.4× / 6.7× / 3.8×),
   so it is not a flat per-access adder either. Bigger relative hit on the cheaper access.
3. The host-memory poll is unchanged at 13 ns, as always. It remains the control, never the signal.

### Measure the state before you trust any host-side number

Every host-side cost in this file — sock-read GB/s, per-read overhead, the ack write, `wait_until_cores_done`
polling — moves by 2–13× with card state, and nothing in the run output announces it. Read the ACK-WRITE
probe first (~170 ns static / ~380 ns dynamic healthy, ~2.3–2.6 µs degraded) and only then compare runs.

**A run does not have to be below the knee to leave the card degraded.** These runs were all at delay 500
with 0 stalls.

**CORRECTED CAUSE — the box FROZE, it was not an interrupted run.** The first account here blamed a killed
run (the sweep's background job died and its ssh dropped mid-drain). The box's own records say otherwise:
`last -x` shows that login session ending in **`crash`**, and the next boot has **no preceding `shutdown`
record**, unlike every deliberate reboot. So the sequence was: box hard-froze under the sweep → IRD watchdog
rebooted it → **the card came up degraded**. The dropped ssh and the "killed" job were symptoms of the box
going down, not the cause of anything. Causality was backwards.

That signature — two processes stop together, the kernel logs nothing, and the physical host never
reboots — is a host CPU stalling on MMIO to a wedged card. It occurs on the DRISC perf-debug path.

**How to tell the two apart, because the observable is identical** (your ssh dies, work stops):

    last -x | head            # a session ending in "crash" and a boot with NO shutdown record = FROZE
    cat /proc/uptime          # small value = the box went down, your run did not merely get killed

Check this BEFORE theorizing about what your run did to the card. An hour was spent here writing an
operational rule ("never cancel a sweep") for an event that was never a cancellation.

### `tt-smi -r` does not recover a degraded card (measured, 2026-08-06)

Tested directly on bh-05 while degraded, ack-write probe as the instrument, nothing else changed:

| | ack write (STATIC) | 4 B device read | sock-read |
|---|---|---|---|
| pre-reset | 2303 ns | 2997 ns | 10.1 GB/s |
| **post `tt-smi -r`**, 3 runs | **2301–2306 ns** | 2940–2994 ns | 7.9–9.9 GB/s |
| after a host `sudo reboot` | **386 ns** | ~770 ns | 17.2 GB/s |

Identical to three significant figures. **Only the host reboot recovers.** The reset itself was clean —
exit 0, "Re-initializing boards after reset", box still up — so the old warning that `tt-smi -r` takes the box
down applies to a **PCIe-HUNG** card (both ports refused, watchdog reboot), not to a degraded-but-responsive
one. Keep the two states distinct: reset is safe here and simply useless.

**Both drainers degrade identically**, which is how we know the state is card-wide and not a property of the
new static window: the Tensix drainer writes through the worker window metal configures at device init, and it
reads 2303–2308 ns — the same as the DRISC's freshly configured one. Device *reads* degrade too (3.8×) and
never touch that window at all.

**The degraded state costs capture completeness, not just time.** 50-iteration runs land at 550,152–550,908
markers where a healthy card returns exactly 551,100 every single time — the host cannot ack fast enough, so
the drainer drops frames rather than block the producers. Treat a short marker total as a reason to check card
state, not automatically as a capture bug.

### Warm reboot is NOT a reliable cure (3 data points, 2026-08-06)

| boot | preceded by | card after |
|---|---|---|
| 11:18 clean reboot | deliberate `shutdown` | **healthy** (172 ns) |
| 12:08 watchdog reboot | **freeze/crash**, no shutdown record | degraded |
| 13:41 clean reboot | deliberate `shutdown` | degraded |
| 13:49 clean reboot | deliberate `shutdown` | **degraded** (2313 ns) |

One warm reboot recovered a degraded card; two later ones did not. So the "a plain `sudo reboot` restores the
fast state" correction recorded earlier the same day was over-generalized from a single success — it *can*
work and cannot be relied on. Post-freeze degradation looks stickier than post-hang degradation, though with
one sample of each that is a shape, not a mechanism. A cold power cycle remains the only cure measured to work
repeatedly (`tt-smi -r` is now measured NOT to work at all, and IRD reservation restarts did not either).

## §N+3 — Matched test: why the Tensix drainer cannot hang the card (bh-05, 2026-08-06)

"The Tensix never hung the card" was only ever *never observed*, and the one comparative sweep was
order-confounded (DRISC got first crack at each new lower delay and hung at 50; the Tensix never got there).
Worse, **delay is the wrong axis**: the Tensix drainer only runs under slow dispatch, which de-synchronizes
producers, so the same delay is not the same load. Matched on the right axis instead:

**CORRECTED 2026-08-06 (the first version of this table was WRONG).** I used `words x 4 B` as the payload,
but `words` counts MARKER words while the drainer ships whole fixed-size span frames. Real shipped bytes are
`pages x 64 B` — 3.3x larger — and the two drainers batch differently, so they do not even ship the same total
bytes for one workload. Applying one numerator to both was invalid. Re-measured, `pages x 64` over
busy-sweep time:

| delay | DRISC bytes / busy → rate | Tensix bytes / busy → rate | Tensix max occ |
|---|---|---|---|
| 500 | 26.0 MB / 2.05 ms → **12.7 GB/s** | 29.0 MB / 2.74 ms → **10.6 GB/s** | 216 |
| 300 | 18.0 MB / 1.40 ms → **12.9** | — | — |
| 200 | 13.8 MB / 0.90 ms → **15.4** (clean, 0 stalls) | 17.5 MB / 1.65 ms → **10.6** | 188 |
| 150 | **PCIe HANG** | — | — |
| 100 | — | 11.1 MB / 1.26 ms → **8.8** | 288 |
| 25 | — | ~ | (1596 producer stalls) |
| 0 | — | 4.8 MB / 0.93 ms → **5.2** | **511/512** (1663 stalls) |

**Two things the corrected numbers overturn:**

1. **Shipped bytes DECREASE as pressure rises** (26.0 → 13.8 MB for the DRISC; 29.0 → 4.8 MB for the Tensix).
   Frames are fixed-size spans, so denser producer rings mean FEWER frames for the same marker count. Egress
   bandwidth therefore cannot be what distinguishes delay 200 (clean) from delay 150 (hang) — it goes the wrong
   way. The original framing of this section, "the Tensix cannot reach the DRISC's egress", was built on a
   metric that does not track the trigger.
2. **The Tensix drainer is READ/PROCESS-bound, not egress-bound.** At delay 0 it is fully saturated — ring
   occupancy 511/512, 1663 producer stalls — while pushing only 5.2 GB/s. It chokes on the per-sweep read and
   process of 110 cores long before PCIe is stressed. The DRISC pushes 12.7-15.4 GB/s throughout.

What survives: the DRISC sustains a higher egress rate than the Tensix at every measured point, and the Tensix
saturates without stressing PCIe. What does NOT survive: any claim that ~4.6 vs ~5.1 GB/s is the boundary, or
that egress rate is the hang trigger.

### What this does NOT establish

- **Egress rate is a working model for the trigger, not a proven mechanism.** It is the axis the DRISC data
  correlates with. The delay-150 egress number does not exist, because the run that hangs the card never
  completes — so the hang boundary is only bracketed as **> 5.08 GB/s**.
- **The ceiling is configuration-specific**: 110 producers, one drainer, slow dispatch. Slow dispatch is not a
  choice for the Tensix (a resident non-CQ program cannot coexist with fast dispatch), so it is structural
  today — but any future setup that pushes a Tensix past ~5 GB/s voids the argument.

### The knee is NOT a safety limit — retract that rule

Today's hang was at **delay 150, ABOVE the recorded knee of 125**, and the run before it (delay 200) reported
**0 producer stalls across all 110 cores**. No counter warned. This is the third card-level event on the DRISC
path in one day (a freeze at delay 500, a hang at 150) and **none of them required a below-knee run**. Treat
"stay above the knee" as insufficient rather than protective, and do not leave DRISC sweeps unattended.

**Run the Tensix arm FIRST** in any future comparison — on a known-good card, before a DRISC pass that may take
the box with it. That ordering is what made this measurement possible at all.

## §N+4 — The egress amplifier: PCIe bandwidth is NOT the hang trigger (bh-05, 2026-08-06)

The Tensix drainer saturates on read/process (511/512 occupancy at only ~5 GB/s), so it can never stress PCIe
by raising producer pressure. To decouple the two, `TT_METAL_PERF_DEBUG_SHIP_REPEAT=N` makes the drainer
**re-send each staged frame N times** — the extra sends skip the read and process phases entirely. The host then
gets N duplicate copies, so it is a stress tool, not a capture: pair it with `TT_METAL_PERF_DEBUG_NO_DECODE=1`
and read the page/byte counters, never the markers.

| repeat | delay | shipped | egress while busy | max occ | notes |
|---|---|---|---|---|---|
| 1 | 500 | 35.9 MB | 12.98 GB/s | 80 | baseline |
| 2 | 500 | 46.2 MB | 16.63 GB/s | 188 | |
| 4 | 500 | 47.5 MB | **17.25 GB/s** | 368 | already past the DRISC's safe 15.4 |
| 8 | 500 | 54.7 MB | 17.17 GB/s | 511 | 7 credit timeouts, 1386 stalls |
| 16 | 500 | 84.6 MB | 16.40 GB/s | 511 | 23 credit timeouts, 1591 stalls |
| 4 | 0 | 18.7 MB | 18.30 GB/s | 511 | 1707 stalls |
| **8** | **0** | 34.6 MB | **19.32 GB/s** | 511 | 6 credit timeouts, 1665 stalls |

Plus **20 consecutive runs at repeat=8 / delay 0**: zero hangs, ack write 166–178 ns throughout, card healthy
after. The pipeline was genuinely saturated — host sock-read pinned at ~21 GB/s, rings at 511/512, producers
stalling, credits timing out.

**A Tensix sustaining 19.3 GB/s does not hang the card.** That is 25% above the highest rate the DRISC survived
(15.4 GB/s at delay 200) and well above the ~13 GB/s it was pushing when it hung at delay 150.

### So egress bandwidth is not the trigger — two independent lines now say so

1. Shipped bytes FALL as delay drops (§N+3 corrected), so the DRISC pushed LESS egress at the delay that hung
   it than at the delay that did not.
2. Forcing a Tensix above every DRISC rate ever survived, repeatedly, does nothing.

Both the "knee as a safety limit" rule and the "Tensix cannot reach the DRISC's egress" explanation are now
dead. Neither delay nor bandwidth is the causal axis.

### What is left, and the experiment that separates it

Whatever differs about the DRISC path itself, not the rate it runs at:
- the **DRAM-core NIU** issuing posted writes into the PCIe tile, versus a worker NIU;
- the **fast-dispatch environment** the DRISC runs in (dispatch cores contending on the same NoC), versus slow
  dispatch, which is the only mode the Tensix drainer can use;
- the **read side** rather than the write side (the DRISC's fused 10,496 B reads across 110 cores).

**The symmetric test: amplify the DRISC at a safe delay.** If ~19 GB/s from a DRAM core hangs the card where
the same rate from a Tensix does not, the trigger is the core/NIU path and not the load — the sharpest available
result. Budget a cold power cycle for it, since a positive result means a hung card.

### The symmetric arm: the amplified DRISC DOES hang — so it is the path, not the load

Same amplifier, same kernel, same knobs, on the DRISC at the most relaxed producer setting (delay 500):

| repeat | shipped | egress while busy | max occ | notes |
|---|---|---|---|---|
| 1 | 31.0 MB | 15.49 GB/s | 92 | clean (this config has run dozens of times today) |
| 2 | 38.2 MB | **18.37 GB/s** | 192 | clean |
| 4 | 39.4 MB | 18.36 GB/s | 372 | clean, 16 credit timeouts |
| 8 | 45.1 MB | 17.43 GB/s | 510 | clean single run, 17 credit timeouts |

Then sustained at repeat=8: **7 clean runs, HUNG on run 8.** `tt-smi` "Error in detecting devices",
`current_link_speed=Unknown`, `current_link_width=63`, AER clean — the documented endpoint-internal wedge. The
box stayed up (no freeze this time).

| | rate | runs | outcome |
|---|---|---|---|
| **Tensix**, repeat=8, delay 0 | **19.32 GB/s** | 20 | all clean, card healthy after |
| **DRISC**, repeat=8, delay **500** | **17.43 GB/s** | 12 | 7 clean, **HUNG on run 8** |

**The Tensix was run HARDER and survived; the DRISC hung.** So the trigger belongs to the DRISC path, not to
the load — and it needs neither a low delay (500 is the most relaxed setting) nor record bandwidth. Within the
DRISC path load still aggravates it (15.5 GB/s clean all day, 17.4 GB/s dies on the 8th run), but load cannot
be *the* cause when a Tensix sits at 19.3 GB/s untouched.

### The one missing cell — the experiment to run next

The DRISC ran under FAST dispatch and the Tensix under SLOW, so "the DRISC path" still bundles two things: the
**DRAM-core NIU** posting into the PCIe tile, and the **fast-dispatch environment** (dispatch cores contending
on the same NoC). The DRISC can also run under slow dispatch (`force_slow_dispatch`), which completes the 2x2:

| | fast dispatch | slow dispatch |
|---|---|---|
| DRISC | **HANGS** (17.4 GB/s, run 8) | **← the missing cell** |
| Tensix | impossible (resident program) | clean at 19.3 GB/s x20 |

Amplified DRISC in slow dispatch: still hangs ⇒ the DRAM-core NIU is the trigger. Survives ⇒ fast dispatch is.
Either way it halves the hypothesis space in one run. Budget a cold power cycle.

## §N+5 — The slow-dispatch DRISC wedge: an UNSATISFIABLE write barrier (bh-05, 2026-08-06)

Under slow dispatch the DRISC drainer wedges on sweep 1 and the host reports FAILED TO START with
`done=0x0 hb=1 phase=11 stop=0`. phase=11 is the sweep-body write barrier.

**ROOT CAUSE (measured, not inferred): the barrier's predicate can never become true.**
`ncrisc_noc_nonposted_writes_flushed` is
`NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_acked[noc]` — a HARDWARE ack counter
against a SOFTWARE mirror incremented at ISSUE time. Publishing both sides from inside the spin (kernel writes
them to the liveness pad at `done+12` / `done+16`; the host prints them on failure):

```
HW_ACK_RECEIVED=7309  vs  SW_acked=7311      frozen, identical across runs
```

**Exactly +2, every time.** Two nonposted writes were issued and their acks never arrived, so the equality is
unsatisfiable and the loop spins forever. This is NOT a stalled NoC and NOT a hung core — the core is running,
reading the register, and comparing. Both of the earlier explanations in this file were wrong:

- ~~"the core is stuck inside the NIU register read"~~ — it is a plain register read that always returns. That
  claim came from "neither the cycle deadline nor a 4M-iteration cap fires", which is equally explained by a
  predicate that is simply never true. **Also: the run that appeared to prove the iteration cap useless had its
  diagnostic TRUNCATED by my own `sed` — I never saw its phase.**
- ~~"pre-existing, reproduces on the pre-amplifier commit"~~ — that test ran against an ALREADY-WEDGED core.
  A wedged DRISC spins forever, so a new `LaunchProgram` cannot take the core over; the host then reads the OLD
  kernel's liveness words and reports FAILED TO START no matter what code was just built. Every "reproduces
  with X disabled" result in that window is void for the same reason.

**THE RELIABLE PATTERN, after a `tt-smi -r`: run 1 SUCCEEDS, run 2 onward wedge, always +2.** So a single
slow-dispatch DRISC run per reset is usable — which is enough to measure the missing 2x2 cell, at the cost of one
`tt-smi -r` (~20 s) per data point.

A start-of-kernel `noc_local_state_init()` on both NoCs (resyncing the mirrors from hardware) does NOT fix it —
verified with a forced JIT rebuild. That rules out pure inheritance: the two non-acking writes are issued WITHIN
the failing run. It is kept anyway: the mirrors persist across launches on a resident core that is never reset,
so resyncing at entry is correct regardless, and the Tensix build already needed it for the read NoC (a stale
read counter makes `noc_async_read_barrier()` return EARLY — silent corruption rather than a wedge).

### Next step, and it is one run

Publish `NIU_MST_NONPOSTED_WR_REQ_SENT` alongside the ack counters and snapshot all three around each distinct
write site in sweep 1 — the head write-backs, `socket_push_pages`, `socket_notify_receiver`. Whichever site
leaves issued-minus-acked at 2 is the leak. Prime suspect: under slow dispatch the producer workers are held in
RESET until the workload launches, and the drainer sweeps before that, so a write to a core in reset may never
ack; fast dispatch has dispatch firmware running on those cores. Two is a suspiciously specific count, though,
so measure rather than assume.

### It also poisons later runs, which is a trap

The wedged DRISC is resident and spins forever, so the NEXT run — even under fast dispatch — reports FAILED TO
START while the card itself is perfectly healthy (probes 170-183 ns, tt-smi fine). `tt-smi -r` clears it. Three
distinct states now: **WEDGED DRISC** (reset clears it), **DEGRADED** (cold power cycle), **HUNG** (cold power
cycle). Check which one you have before debugging anything, and reset before trusting any negative result.

## §N+6 — Egress saturates at ~17 GB/s, and the hang is NOT on that axis (bh-05, 2026-08-06)

### The slow-dispatch wedge does not reproduce any more — cause unattributed

On the current binary, from a fresh `tt-smi -r`, the slow-dispatch DRISC drainer runs **clean**: 4 runs at the
110 grid, 6 at 120, plus ~10 more across the controls below. **The missing 2x2 cell is therefore filled: DRISC
under slow dispatch works.** What fixed it is unknown, and three candidate explanations were each tested and
killed:

- **"120 cores gives every core firmware"** — no. A `TT_METAL_PERF_DEBUG_FULL_GRID=1` knob was added to drop the
  reserved column, and 120 runs clean — but so does 110. The grid is not the variable.
- **"`noc_local_state_init()` at kernel entry is the fix"** — no. A `TT_METAL_PERF_DEBUG_NO_NOC_INIT=1` knob now
  makes the resync switchable so the wedge can be brought back on one binary. With the resync **off**, 4 runs
  still pass. (The knob is worth keeping: a fix you cannot un-apply is a fix you cannot prove.)
- **"a killed run leaves the resident core spinning and poisons the next"** — not sufficient. A `SIGKILL` 0.35 s
  into a drain does not wedge the next run, and neither does the `WRITER_DIE_AFTER` hook (consumer vanishes
  mid-stream), tested with the resync both on and off.

**A methodology trap worth repeating, because it caught me twice more today.** A wedged DRISC is resident, so the
host reads the OLD kernel's liveness words and reports FAILED TO START regardless of what was just built. A
"wedge" observed at the 110 grid was inherited from a previous killed run, and the `HW_ACK=436036 vs
SW_acked=436038` printed with it was **frozen at one value across four consecutive runs** — the giveaway that
nothing was writing those words. Any counter identical across runs is stale, not reproducible. Reset first.

The one genuinely reproducible footgun: **workload grid > drainer poll list hangs the workload forever.**
`--gx 0` (120 producers) against a 110-core poll list leaves 10 cores undrained; their rings fill and they block.
Match them, or use `FULL_GRID=1`.

### Egress is capped at ~17 GB/s and the amplifier cannot exceed it

Fast dispatch, delay 500, ascending `SHIP_REPEAT`, 3 runs each. Egress = `pages x 64 B / busy-sweep time`:

| repeat | shipped | egress | max occ | stalls |
|---|---|---|---|---|
| 1 | 309 MB | 16.20-16.24 GB/s | 92 | 0 |
| 2 | 282-318 MB | 14.76-16.64 | 232 | 0 |
| 4 | 271-325 MB | 14.15-16.93 | 464 | 0 |
| 6 | 291-325 MB | 14.39-16.78 | 510 | 6.6k-20k |
| 8 | 378-379 MB | 16.78-17.03 | 510 | 20.9k |
| 12 | 557-559 MB | 16.78-16.96 | 510 | 21.2k |
| 16 | 738-740 MB | 16.87-16.93 | 511 | 21.2k |

**21 consecutive runs, no hang.** Amplification past repeat=4 buys longer busy sweeps and more bytes, never more
bandwidth — a 16x amplifier lands within 4% of a 1x one. **~17 GB/s is a hard wall**, and repeat=1 is already
within 5% of it.

### The hang is on the INGEST axis, not the egress axis

Holding the amplifier fixed and raising producer rate (fast dispatch, 3 runs per cell):

| delay | repeat | egress | outcome |
|---|---|---|---|
| 500 | 4/8/16 | 16.8-17.0 | clean x9 |
| 200 | 4 | 16.64-16.95 | clean x3 |
| 200 | 8 | 16.87-**17.47** | clean x3 |
| 200 | 16 | 16.92-17.05 | clean x3 |
| 100 | 4 | 16.62-16.79 | clean x3 |
| **100** | **8** | — | **run 1 timed out mid-run, run 2 died in `is_pcie_hung`** |
| **0** | **4** | — | **hung on run 1** |

**There is no egress bandwidth at which the DRISC hangs.** 30 clean runs sit at 16.2-17.5 GB/s, and the highest
rate ever recorded here (17.47 GB/s, delay 200 repeat 8) is *clean*. The cells that hang are at the **same**
egress; what changes is producer delay. Occupancy (511/512) and stall counts (~21k) are also already matched
between the clean delay-200 and the hanging delay-100 cells, so neither of those is the discriminator either.

**The axis is what the DRISC READS, not what it writes** — its fused 10,496 B reads across 110 cores, at a rate
set by producer delay. That inverts the working assumption behind every amplifier experiment: the amplifier
loads the wrong side of the drainer, which is why 16x of it is harmless.

Card left HUNG (`current_link_speed=Unknown`, `width=63`, AER all-zero, `Read 0xffffffff over PCIe`). Needs a
**cold power cycle**; do NOT `tt-smi -r` a hung card.

## §N+7 — Tensix vs DRISC: every difference in the path, host and device

Both drainers run the SAME kernel source, ship the SAME frame format, over the SAME socket protocol, to the
SAME destination (the PCIe tile at `PCIE_NOC_X/Y`), issued by the SAME function
(`write_to_host_chunked` -> `noc_wwrite_with_state`, same `NOC_UNICAST_WRITE_VC`, same `write_cmd_buf`, same
`NOC_MAX_BURST_SIZE` chunking). Both use `NOC_INDEX` for egress and `kReadNoc = NOC_INDEX ^ 1` for ingest, and
both are created on NoC 0. So the divergence is NOT in the write code. It is in **who issues it and from where**.

### Device side

| | DRISC | Tensix BRISC |
|---|---|---|
| **NIU mode** | **flipped to STREAM mode** (`NIU_CFG_0.AXI_SUBORDINATE_ENABLE` cleared) before the run, restored to NOC2AXI in a tail gated on the host writing `stop=2` | untouched -- a worker NIU is already a NoC master |
| `cb_interface` | shim defined by the kernel (no CB infra on DRAM cores) | provided by firmware |
| L1 addressing | needs the `0x2000000000` tag in NOC2AXI mode; plain local addresses in stream mode | plain worker L1 |
| L1 size | 128 KB total (`MEM_DRISC_L1_SIZE`) | 1.5 MB, staging clamped to the DRISC's 7 slots to stay comparable |
| kernel config | `DramConfig{.noc = NOC_0}` | `DataMovementConfig{RISCV_0, RISCV_0_default}` + `DRAIN_ON_TENSIX` |

### Host side

| | DRISC | Tensix BRISC |
|---|---|---|
| socket `sender_uses_physical_noc_addr` | **true** -- physical NoC coord + full L1 address | false -- logical coord, worker-L1 semantics |
| socket core coord passed | **NOC0 PHYSICAL** (`drisc_phys`) | **LOGICAL** (`drisc_logical`) |
| static TLB | configured explicitly here (2 MB @ 0, Strict) because the drainer sits on the DRAM channel's *unused* port, which `ll_api::configure_static_tlbs` does not map | free -- metal maps all workers at device init |
| NIU mode call | `set_drisc_niu_mode(..., 1)` before the socket is built | none |
| profiler-region zeroing | DRAM `PROFILER` region | TENSIX `PROFILER` region |
| core placement | unused port of a DRAM bank (die edge) | reserved worker column (gx, d) |
| dispatch mode | either | slow dispatch ONLY (resident program on a worker) |

### The asymmetries that could plausibly reach card level, ranked

1. **The NIU mode flip is the only difference that writes a persistent chip register.** `NIU_CFG_0` survives
   program exit and process exit; only a chip reset restores the NOC2AXI default. The restore tail is gated on
   the host writing `stop=2` with a 200M-spin timeout, so **any run that dies before that handshake leaves a
   DRAM endpoint parked in stream mode**. In stream mode that endpoint no longer forwards inbound DRAM-range
   addresses to GDDR -- it terminates them at DRISC L1. Nothing else in either path mutates chip state that
   outlives the process. This is the single best structural candidate for "why does only the DRISC path leave
   the card in a bad state", and it costs one run to test: flip to stream mode, restore, and never drain at all.
2. **The egress route differs.** A DRAM-bank endpoint on the die edge and a worker in column gx reach the PCIe
   tile over different hop counts and different routers. Same VC, different links.
3. **Shared DRAM-bank infrastructure.** The drainer's NIU is an unused *port* of a bank whose other ports carry
   real GDDR traffic. `pick_unused_dram_logical_core` guarantees no one else uses that port, not that the port
   is isolated from the bank.
4. Addressing/coordinate differences (`sender_uses_physical_noc_addr`, physical vs logical) are ADDRESSING ONLY and are
   exercised identically on every clean run -- they cannot be rate-dependent, so they do not explain a hang that
   only appears at low producer delay.

### What this predicts, given §N+6

§N+6 put the trigger on the INGEST axis (producer delay), not egress. Items 2 and 3 above are egress-side and so
are poor fits. Item 1 is neither -- it is a state the DRISC path enters regardless of rate, which would explain
*persistence* (why the card stays bad) without explaining *onset*. The honest reading is that onset and
persistence may be two different mechanisms, and only onset is rate-dependent.

## §N+8 — The hang is CUMULATIVE, not a property of (delay, repeat). §N+6's "ingest axis" is confounded.

Deliberate repro attempt on a freshly cold-booted card, to read the new worker-core control probe on a degraded
card. It did not go as §N+6 predicted, and the correction matters more than the original result.

**The cells that "hung" do not hang from a fresh card.** 12 consecutive attempts at exactly the two configs
§N+6 recorded as hanging -- delay 100 repeat 8 (hung run 1 there) and delay 0 repeat 8 (hung run 1 there) --
all completed rc=0 on a cold-booted card. The card stayed healthy throughout (ack 177 ns, worker 713 ns).

**Replaying the full ascending ladder hung it at run #31, in a cell that had just run clean.** delay 200
repeat 6 -- and delay 200 at repeats 4, 8 and 16 had all passed x3 earlier in that same ladder and in §N+6.

**Both hangs land at ~the same cumulative run count.** §N+6's hang came after 21 (delay 500) + 9 (delay 200) +
3 (delay 100 rep 4) = run ~34. This one at run #31. Two independent ladders, hang at 31 and ~34 runs.

### The confound, stated plainly

Every ladder run so far was **ascending in load**, so lower delay ALWAYS came later in the session. Cumulative
runs and producer delay were perfectly correlated, and §N+6 attributed the hang to the axis it happened to be
sweeping. The variable that actually predicts the hang is **cumulative runs / sustained time under load**, and
delay 200 repeat 6 -- mid-ladder, unremarkable, previously clean -- is enough to trigger it once that budget is
spent.

This is exactly the ordering trap already written down in this file for the Tensix-vs-DRISC comparison ("a sweep
that runs A then B at each delay gives A the first crack at every new delay"). It was not applied to the delay
axis itself.

**Retract from §N+6:** "the axis is what the DRISC READS, not what it writes", and the delay/repeat table's
implication that specific cells hang. What survives from §N+6 is only the egress measurement itself -- egress
saturates at ~17 GB/s and amplification cannot exceed it (21 clean runs), so egress bandwidth is still not the
trigger. The positive claim about ingest is withdrawn.

**What a correct experiment looks like:** randomize or alternate cell order within the ladder, and record
cumulative-runs-since-power-cycle as a first-class column. A single cell repeated N times from a fresh card is
the cleanest form -- 12 such runs at the two "hanging" cells are already the first data point, and they are clean.

Hung-card state confirmed again: `current_link_speed=Unknown`, `width=63`, AER fatal sum 0.

## §N+9 — Fixed config has never hung; every hang came from a VARYING sequence. Rare and stochastic. (bh-05, 2026-08-06)

§N+8 said the hang tracked cumulative runs. That was also wrong, and this isolates it. One cell repeated from a
fresh `tt-smi -r`, probes captured every run:

| phase | config | runs | result |
|---|---|---|---|
| 1 | delay 500, repeat 1 | 50 | clean |
| 2 | delay 200, repeat 6 (the cell that hung mid-ladder) | 50 | clean |
| 3 | delay 500, **repeat 16** (max amplification, occ 511) | 50 | clean |
| 4 | **churn**: repeat 1 -> 2 -> 4 -> 6 at delay 500 | **4** | **HUNG at run 4** |

**150 consecutive fixed-config runs, no hang.** That kills three hypotheses at once:
- **Cumulative run count** (§N+8) -- 50 > the 31 and ~34 where the ladders died, three times over.
- **High amplification / sustained load** -- 50 runs at repeat 16, pinned at 511/512 occupancy, are harmless.
- **A dangerous cell** -- delay 200 repeat 6 is the exact cell that hung mid-ladder; alone it runs 50 times.

The one thing the ladders had that a repeated cell does not is **changing the drainer's configuration between
runs**, and that hung the card on its FOURTH run.

**No warning signal exists.** Probes are flat across all 150 clean runs -- ack 167-188 ns, drainer-core read
761-798, worker read 701-738 -- with one paired blip at run 40 of phase 3 (drainer 889 / worker 832, both
elevated together, recovered by run 50). Nothing trends toward the hang, so latency cannot be used as an early
warning.

### TRIAL B: churn survived 49 runs. Churn is NOT reliably reproducible.

A second churn trial from a fresh card, identical script, ran the full 49 runs CLEAN. So the phase-4 hang at
run 4 was not "churn hangs it fast" -- it was a rare event that happened to land early. Standing tally:

| regime | runs | hangs |
|---|---|---|
| fixed config (phases 1-3) | 150 | **0** |
| varying config (churn A+B, ladders §N+8) | ~118 | **3** (at run 4, 31, ~34) |

**What is defensible: fixed config has never hung, and every hang ever observed came from a varying-config
sequence.** What is NOT defensible: that churn causes hangs on any particular timescale, or that the difference
is established -- 3/118 vs 0/150 is suggestive (Fisher p ~ 0.08), not significant. The heading of this section
originally read "It is CONFIG CHURN"; that was overstated on one sample and is corrected here.

The variance is large: run 4, run 31, run ~34 for the three hangs.
JIT recompilation is NOT the mechanism: every repeat variant in the churn cycle was already built and cached by
the §N+8 ladder on this same binary, so no run in phase 4 compiled anything new.

What "changing configuration" physically does that a fixed config does not is still unidentified -- each run is
its own process with its own device open either way. The honest statement is that the CORRELATION is strong
(150 clean fixed vs a hang at run 4 of churn) and the MECHANISM is unknown.

### The experimental rule this establishes

**Never conclude anything about a cell from a sweep that visits other cells.** Every hang result in this file
before §N+9 came from a ladder, and ladders confound the swept axis with the act of sweeping. Fixed-config
repetition from a fresh card is the only sound form for this measurement, and it is cheap (50 runs ~ 3 min).

## §N+10 — The Tensix-vs-DRISC comparison was confounded with DISPATCH MODE all along

The matched control, finally run. Same churn cycle (repeat 1,2,4,6,8,12,16 @ delay 500), same 2 trials x 49
runs, each from a fresh `tt-smi -r`:

| arm | dispatch | grid | runs | hangs |
|---|---|---|---|---|
| DRISC | **fast** | 120 | ~118 (churn A+B, ladders §N+8) | **3** (run 4, 31, ~34) |
| DRISC | **slow** | 110 | 98 | **0** |
| Tensix BRISC | slow | 110 | 98 | **0** |

**Every hang ever recorded on this box is a FAST-DISPATCH DRISC run.** The same DRISC, churned identically
under slow dispatch, is clean for 98 runs -- as is the Tensix.

### Why the whole Tensix control was never a control for core type

A Tensix drainer is a resident program on a worker, which **fast dispatch will not allow**, so the Tensix arm
can ONLY run slow dispatch. Every "Tensix survives where the DRISC hangs" result therefore compared
`DRISC + fast dispatch` against `Tensix + slow dispatch` and varied two things at once. The one-line summary
this file has carried for two sessions -- "it is the DRISC path, not the load" -- silently included the dispatch
environment in "the path". With the DRISC now working under slow dispatch (§N+6), the within-core-type
comparison is possible for the first time, and it splits on dispatch mode, not on core type.

Statistically this is still weak: 3/118 vs 0/196 is Fisher p ~ 0.055. Direction is consistent, significance is
borderline, and the honest claim is **"dispatch mode is now the leading candidate and core type is not
supported"** -- not that the DRAM core is exonerated.

### The experiment that settles it, and it is within ONE core type

Fast-dispatch DRISC churn vs slow-dispatch DRISC churn, many trials, alternating order. No Tensix needed, so
nothing is confounded with the resident-program restriction. If fast dispatch keeps hanging and slow dispatch
keeps not, the trigger is the dispatch environment -- dispatch cores contending on the same NoC, which was on
the original candidate list in §N+4 and got dropped when the Tensix comparison looked decisive.

**The 2x2 can never be completed on the Tensix side** (fast dispatch + resident worker program is illegal), so
core type can only ever be tested by holding dispatch mode fixed at SLOW -- where, so far, neither core hangs.

## §N+11 — THE 2x2 IS COMPLETE: only DRISC *AND* fast dispatch hangs (bh-05, 2026-08-06)

The Tensix arm CAN run under fast dispatch. `dispatch_core_manager` pops a worker off the BACK of the dispatch
pool for the real-time profiler and removes it from `logical_dispatch_cores`, so FD never allocates it; with the
RT profiler off, that core is idle and a resident non-CQ program can own it. Wired up via
`get_reserved_realtime_profiler_core()` -- the drainer lands on logical (11,9) and services all 110 cores,
5,501,088 markers, 0 stalls. The long-standing "impossible (resident program)" entry in this file was wrong.

Same churn cycle (repeat 1,2,4,6,8,12,16 @ delay 500), 2 trials x 49 runs per cell, each from a fresh reset:

| | **fast dispatch** | **slow dispatch** |
|---|---|---|
| **DRISC** | **3 hangs / ~118 runs** | 0 / 98 |
| **Tensix BRISC** | **0 / 98** | 0 / 98 |

**Neither factor alone does it.** The DRISC under slow dispatch is clean for 98 runs. Fast dispatch with a
Tensix drainer is clean for 98 runs. Only the CONJUNCTION hangs. 3/118 in that one cell against 0/294 across the
other three (Fisher p ~ 0.01) -- the strongest signal this investigation has produced, and the first one resting
on a complete factorial rather than a ladder.

**Supersedes §N+10.** That section concluded "dispatch mode is the leading candidate, core type unsupported",
which was the best reading available when the Tensix arm was believed slow-dispatch-only. With the fourth cell
filled it is wrong: dispatch mode alone does not explain it either. It is DRAM-core egress *in the presence of
fast-dispatch traffic*.

### Why an interaction is mechanically plausible

Fast dispatch keeps dispatch kernels resident and moving on the same NoC the drainer ships on. A DRISC egresses
from a die-edge DRAM endpoint whose route to the PCIe tile differs from a worker's, so it is the one combination
where drainer egress and dispatch traffic share links they otherwise would not. That is a testable next step:
vary dispatch INTENSITY (number of CQs, dispatch core placement) with the DRISC arm fixed.

### Standing caveats

- 3 hangs total. Rare-event statistics, and the three took 4, 31 and ~34 runs to appear -- huge variance.
- Grid differs between the arms that hang (120 under fast dispatch when FULL_GRID is unset, 110 elsewhere).
  Pin the grid before treating the interaction as established.
- No early-warning signal: probes stay flat (ack 163-188 ns, worker 701-742) across every clean run in all four
  cells.

### §N+11 addendum — the grid caveat was WRONG, and the interaction got stronger

The §N+11 caveat "the hanging cell runs 120 while the others run 110" is **withdrawn**. Under fast dispatch
`compute_with_storage_grid_size()` already returns 11x10, so a `--gx 0` fast-dispatch run polls **110**, not 120
-- verified directly: `DRISC 0 resident on logical (0,1) [noc0 (0,0)], cores [0,110) of 110`. The 120 figure
came from the separate `FULL_GRID=1` SLOW-dispatch experiments and was carried over by assumption. **All four
cells of the 2x2 were already grid-matched at 110.** Grid never rode along with the interaction.

A further DRISC+fast churn trial at a confirmed 110 cores (sanity line printed) **hung at run 16**. Updated:

| | fast dispatch | slow dispatch |
|---|---|---|
| **DRISC** | **4 hangs / ~134 runs** (at runs 4, 16, 31, ~34) | 0 / 98 |
| **Tensix BRISC** | 0 / 98 | 0 / 98 |

4/134 against 0/294 across the other three cells: Fisher p ~ 0.004. The interaction is now the best-supported
claim in this file, and it is grid-matched, dispatch-matched and core-type-matched by construction.

The four hang points -- runs 4, 16, 31, ~34 -- remain wildly variable, so this is a rare stochastic event with a
strongly non-uniform cell preference, not a deterministic trigger. Mean ~21 runs to hang in that cell.

## §N+12 — Bisection: egress-only is clean, NoC swap changes nothing (bh-05, 2026-08-06)

Two ablations behind `TT_METAL_PERF_DEBUG_ABLATE`, plus a NoC swap behind `TT_METAL_PERF_DEBUG_NOC`.

**Egress-only (`ABLATE=1`) does NOT hang: 98 runs, 0 hangs.** The drainer re-ships fixed mock bytes with no
worker reads and no per-core processing; the PCIe push, socket credit loop, write barrier and notify are
untouched. 378 MB and ~5,100 pushes per run. At the full config's rate (4/134) ~4 hangs were expected.
CAVEAT: continuous shipping is CREDIT-BOUND at ~3.9 GB/s sustained, where a real run is bursty (268 busy sweeps
of 15,477, ~1.7% duty) and bursts to 16 GB/s. The `ABLATE_SPIN` knob had NO effect at any value precisely
because of this -- when you are already blocked on credits, a spin displaces wait time instead of reducing
throughput. So what is shown is "egress at a lower burst rate is safe", not "egress is safe".

**Read-only (`ABLATE=2`) is NOT YET MEASURABLE.** With egress compiled out the host writer waits for data that
never arrives and teardown blocks past the harness timeout. The first attempt returned rc=124 at run 1 and
looked like an instant hang -- the card was FINE (link 32 GT/s, next run clean, probes normal). Needs the host
told not to expect data before this arm means anything. **rc=124 is a harness timeout, not a hang: always probe
the card before believing it.**

**The NoC swap changes nothing.** `TT_METAL_PERF_DEBUG_NOC=1` moves egress to NoC 1 and reads to NoC 0:

| arm | egress | reads | egress rate | result |
|---|---|---|---|---|
| NoC 0 | NoC 0 | NoC 1 | 16.2 GB/s | hung at run 16 |
| NoC 1 | NoC 1 | NoC 0 | 16.0 GB/s | **hung at run 16** |

Load matched within 1%. **The hang is symmetric across both NIUs and both routes**, so "NoC 0's route from the
DRAM endpoint to the PCIe tile" is dead as an explanation.

Both arms hung at run 16, cycle 3, repeat 2 -- identical coordinates. Other churn hangs landed at run 4 and the
ladders at 31/~34, so this is probably coincidence, but "repeat=2 after two clean cycles" is worth a targeted
check before assuming randomness.

### A wrong deduction, and how one run killed it

Switching to NoC 1, I inferred from `NOC_0_X_PHYS_COORD(noc, size_x, x) = noc == 0 ? x : size_x - 1 - x` that
the PCIe tile needed a mirrored encoding, and built a host-side override for it. **Wrong.** With the override,
decode yields **0 markers from 2.37M pages**; without it, 5,501,058 markers decode correctly. That macro mirrors
WORKER coordinates; the PCIe encoding is in TRANSLATED space -- the kernel is built with `PCIE_NOC_X=19,
PCIE_NOC_Y=24`, both outside the 17x12 NOC0 grid, which was visible in the build line the whole time. The
socket's NOC0-derived `pcie_xy_enc` is correct on BOTH NoCs. Mirroring is retained only behind
`TT_METAL_PERF_DEBUG_NOC_MIRROR=1` so the dead end stays documented.

**Diagnostic worth keeping: pages flowed while zero markers decoded.** Socket credits advance on a different
path than the payload writes, so a wrong payload destination looks like healthy throughput -- 2.37M pages
"delivered" into a ring full of nothing. A page count cannot tell you the data arrived; only decode can.

## §N+13 — The degradation is CARD-WIDE, not the DRAM endpoint (bh-05, 2026-08-06)

The worker control probe added in §N+11 finally caught a degraded card (this one came back degraded from a warm
reboot; the previous three came back healthy -- warm reboot remains an unreliable cure).

| probe | healthy | degraded | ratio |
|---|---|---|---|
| ack write (DRAM core) | 172 ns | 2301-2322 | 13.4x |
| device read (DRAM core) | 775 ns | 2935-3004 | 3.8x |
| **worker read (CONTROL)** | 710 ns | **2861-2885** | **4.0x** |
| host sock-read | 21-22 GB/s | 11.0-12.0 | ~half |
| PCIe link | 32 GT/s x16 | 32 GT/s x16 | unchanged |

**The worker is as slow as the DRAM core -- 2861 vs 2935 ns, within 3%.** Card-wide MMIO latency; the drainer
core is incidental. Every degraded figure previously measured at the drainer therefore generalizes, and
critically **the Tensix arm was never blind to degradation** -- a worry raised in §N+11 that would have
invalidated the entire Tensix-vs-DRISC comparison. Workers show it just as clearly.

The ack WRITE degrades much harder (13.4x) than either READ (~4x). Writes and reads are not equally affected.

### Read-only ablation: the arm runs, the harness still does not

`TT_METAL_PERF_DEBUG_WRITER_TIMEOUT_S=5` makes the host writer stop waiting for data that will never arrive
(safe here specifically because with egress compiled out the drainer never reserves pages, so nothing starves).
The drainer itself is provably healthy in this mode:

```
done=0x0 (still resident) | heartbeat 682,627,025 -> 690,790,353 (advancing) | phase 2 (POLL)
```

Sweeping and reading normally. But the run still burns the full 150 s harness timeout somewhere after the
writer gives up, so the read-only arm is STILL not measurable for hang statistics. Teardown needs tracing next.

## §N+14 — CAUGHT LIVE: the "PCIe hang" is an ENDPOINT wedge, not a link failure (bh-05, 2026-08-07)

First time the hang was captured with the process still alive and attachable. Prior runs always killed the
harness at 150 s, which destroyed the evidence. This run used a 600 s per-run timeout.

**How it was caught.** 84-run monitored churn (12 cycles x repeat {1,2,4,6,8,12,16}, DRISC + fast dispatch,
`--gx 0 --gy 0 --iters 500 --delay 500`, NO_DECODE=1) ran **completely clean** — 0 hangs. The hang then fired
on the very next invocation, an ordinary health probe with no SHIP_REPEAT (i.e. cumulative run 85). Run 84's
log shows a clean device close. So the hang is not correlated with the ladder; it is a rare stochastic event.

### The decisive observation: the LINK IS UP

| what | value at hang |
|---|---|
| root port `0000:00:01.1` `current_link_speed` / `width` | **32.0 GT/s / 16** — full speed, full width |
| endpoint `0000:01:00.0` config space, first 64 B | **all `ff`** |
| endpoint `current_link_speed` / `max_link_speed` | `Unknown` / `64.0 GT/s` (all-ones decode) |
| endpoint `max_link_width` | `63` (all-ones decode) |
| endpoint + root port AER (fatal/nonfatal/correctable) | **all zero** |
| `dmesg` | **nothing after boot** — no link-down, no AER, no timeout |

The endpoint's *static capability* fields (`max_link_speed`, `max_link_width`) read correctly before the run
and read as all-ones after. Those fields cannot change. So this is not a downtrained or retraining link —
**config-space reads themselves are not being completed.** Meanwhile the root port immediately upstream is
still linked at full 32 GT/s x16 and logs no error.

Conclusion: the PCIe *link layer* is healthy. The *endpoint* has stopped responding to TLPs — config and
MMIO alike. Reads master-abort and the CPU gets all-ones back. Because nothing ever drops the link, nothing
in the PCIe stack resets the device, and the kernel has no event to report. **That is exactly why `tt-smi -r`
and warm reboots do not clear this state and only a cold power cycle does** — the wedge is in device logic
that sits behind a link that never went down.

Caveat, stated because it looks like a smoking gun and is not: the root port's Secondary Status register
(offset 0x1e) reads `0x2000` = Received Master Abort. RMA is sticky and is routinely set during ordinary
boot-time bus enumeration (scanning absent devices on bus 01 master-aborts). Without a pre-hang baseline it
is **not** evidence. Do not cite it.

### Where the host is stuck (gdb, all threads)

Two threads matter; the other 16 are taskflow workers parked in `commit_wait`, Tracy, and PMIx.

```
Thread 19 (LWP 6456)  state=R  -- SPINNING, not blocked
  MetalContext::get_cluster()
  read_cq_host_ptr<true>(SystemMemoryManager const&, int, unsigned char, ...)
  SystemMemoryManager::completion_queue_wait_front(unsigned char, std::atomic<bool>&) const
  FDMeshCommandQueue::read_completion_queue_event(MeshReadEventDescriptor&)
  FDMeshCommandQueue::read_completion_queue()

Thread 1 (LWP 6416)   state=S  wchan=futex_wait_queue_me
  main()
  FDMeshCommandQueue::finish(...)
  FDMeshCommandQueue::finish_nolock(...)
  FDMeshCommandQueue::wait_for_outstanding_reads(std::unique_lock<std::mutex>&)
  pthread_cond_wait()
```

The completion-queue reader is **spinning** (state R) in `completion_queue_wait_front`, polling for a
dispatch completion that will never arrive because the device no longer answers. Main is parked on the
condvar waiting for that reader. There is no software deadlock and no lock-ordering bug — this is a
liveness failure imposed from below by dead hardware. The infinite spin with no timeout is why the
symptom presents as a wall-clock hang rather than an error.

### What this rules in and out

- **Ruled out:** PCIe link training / retrain / downtrain as the mechanism. The link is up.
- **Ruled out:** a host-side deadlock in the dispatch stack. Both threads are exactly where they should
  be if the device stopped answering.
- **Ruled out:** the kernel silently recovering something. It never saw an event at all.
- **Ruled in:** the DRISC's egress traffic wedges the PCIe *tile / endpoint application layer* on the
  device. This is consistent with §N+11 (only DRISC+fast dispatch hangs) and with the fact that DRISC
  and Tensix drainers write to the same PCIe destination — what differs is who is driving it.

### Consequence for the metric

`is_pcie_hung` and "link Unknown/63" were being read as *link* failure in every earlier note. They are
an artifact of all-ones config reads from an endpoint that stopped completing. Any future note must say
**endpoint not completing TLPs**, and must check the ROOT PORT's link state to distinguish the two.

### Rate

Hangs so far: cumulative runs 4, 16, 16, 31, and 85. 84 consecutive clean runs immediately preceding the
85th. Treat the per-run probability as low single-digit percent with wide error bars; do not quote "1 in 21".

## §N+15 — WHY the CQ gets stuck in `finish`: the host spins on a pointer only the dead device can move

Read the code behind the §N+14 backtrace. The hang is fully explained, and it is not a bug in the dispatch
logic — it is an unbounded wait with no timeout enabled by default.

### The wait

`FDMeshCommandQueue::finish` -> `wait_for_outstanding_reads` blocks on a condvar until the completion-queue
reader thread drains the outstanding read. That reader sits in
`SystemMemoryManager::completion_queue_wait_front` (`system_memory_manager.cpp:749`), whose predicate is:

```cpp
wait_condition = [&]{ return cq_interface.completion_fifo_rd_ptr == write_ptr and
                             cq_interface.completion_fifo_rd_toggle == write_toggle; };
```

i.e. **"spin while the completion queue is empty."** It exits when the device's completion *write* pointer
advances past the host's read pointer.

### Where that write pointer lives — the crux

`write_ptr` comes from `get_cq_completion_wr_ptr` -> `read_cq_host_ptr<true>` (`command_queue_common.cpp:35`).
The template parameter is documented at line 32:

> *Device-written pointers live at an offset within the hugepage that includes the device channel offset,
> while host-written pointers do not*

So `<true>` = a **device-written** pointer, and it is fetched with `read_sysmem(...)` — **a read of HOST
memory**, not device MMIO. The device pushes that pointer into the hugepage over PCIe as it completes work.

That is the whole hang in one sentence: **when the endpoint stops completing TLPs it can no longer write the
completion pointer into host DRAM, so the host polls a stale host-memory location forever.**

Two consequences that matched the observations exactly:

- **The spinning thread never touches the dead device.** It is a memcpy from host DRAM in a tight loop. That
  is why it showed as state R (running hot, not blocked in MMIO), why it generates no bus traffic, and why
  the kernel logged nothing at all. The hang is *silent by construction* — we were looking for host-side
  evidence of a device access that never happens.
- The `advance_device_execution` call in the loop body is a TTSim no-op on real hardware.

### Why no timeout fired

`completion_queue_wait_front` does wrap the wait in `loop_and_wait_with_timeout`, but:

```cpp
if (timeout_duration.count() > 0.0f) { ...timeout path... }
else { do { ...; func_body(); } while (wait_condition()); }   // unbounded, no yield
```

and the default is **zero**: `rtoptions.hpp:340` — `timeout_duration_for_operations = duration<float>(0.0f)`.
So the timeout is **disabled unless you ask for it**, and the `else` branch is a bare infinite spin. Nothing
about our configuration turned it off; that is stock behaviour.

### The knob we should have been using all along

```
TT_METAL_OPERATION_TIMEOUT_SECONDS=45          # arms the timeout (default 0.0 = never)
TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=<script>   # run at hang time
```

On timeout, `on_timeout` sets `exit_condition`, calls `MetalContext::on_dispatch_timeout_detected()` and
throws `TT_THROW("TIMEOUT: device timeout, potential hang detected, the device is unrecoverable")`.
`on_dispatch_timeout_detected` (`metal_context.cpp:781`) logs, optionally serializes Inspector RPC data, and
**`std::system()`s an arbitrary command** — the hook exists precisely to run `tt-triage`. This is the
built-in version of the 600 s babysitting harness from §N+14, and it fires in seconds instead of minutes.

Note the timeout is progress-gated: it only fires if the wait condition holds AND `get_cq_dispatch_progress`
(an L1 read from the dispatcher core, i.e. a real device read) is unchanged. On a dead endpoint that read
returns all-ones constantly, so it correctly reads as "no progress". There is one spurious reset the first
time progress is sampled, because `last_progress_value` is initialized to 0 and the first all-ones read
differs from it — costs one `dispatch_progress_update_ms` interval, nothing more.

### What this does and does not tell us

It explains the *symptom* completely and kills any theory of a host-side dispatch deadlock. It says nothing
about *why the endpoint dies* — that is still the open question, and the reason for the num_hw_cqs experiment
in §N+16.

## §N+16 — Arming the operation timeout SUPPRESSES the hang; and degradation needs a FREEZE, not a hang (bh-05, 2026-08-07)

### The result: a host-side spin knob changes whether the DEVICE wedges

`TT_METAL_OPERATION_TIMEOUT_SECONDS` does more than add a timeout. In `loop_and_wait_with_timeout`, a
non-zero duration selects a **different loop body**: the timeout path calls `std::this_thread::yield()`
each iteration, while the default (0.0) path is a bare `do { func_body(); } while (wait_condition());`
with no yield at all. So arming it converts the completion-queue poller from a flat-out spin on host
DRAM into a yielding one.

Two blocks, identical in every other respect (DRISC + fast dispatch, `--gx 0 --gy 0 --iters 500
--delay 500`, alternating `num_hw_cqs`, same binary, same card):

| block | timeout | hangs / runs |
|---|---|---|
| A | armed (45 s) | **0 / 179** |
| B | not set (stock tight-spin) | **5 / 125** |

Fisher p ~ 0.02. Block B's first hang landed on **run 1**. Pooling B with the earlier no-timeout data
(2x2 era + NoC-1 + the 84-run monitored churn) gives ~6/235 = 2.6% against 0/179.

The build is not the variable: diffed against `80b5263c240`, the last commit that demonstrably hung, the
only deltas are a removed `if constexpr (kAblate == 2)` early-return (dead code when ABLATE is unset) and
a comment block. **Functionally identical on the default path.**

Mechanistically this is consistent with §N+15: the poller hammers the hugepage that the device writes
completions into. Relieving that pressure appears to make the endpoint wedge less often. Not proven as
the mechanism — but it is a host-side knob that changes a device-side failure, which is worth knowing.

### The num_hw_cqs comparison is VOID — attribution artifact, not just low power

All 5 of block B's hangs landed in the `q=1` arm, 0 in `q=2`. That is an artifact of the harness:

```
q2_22   init-ok  drain=1335 sweeps  exits CLEANLY (cluster.cpp:811) 03:05:25.455
q1_23   profiler init trips MMIO watchdog 03:05:27.053, then hangs
```

Same shape for `q2_28` -> `q1_29`. The wedge's onset **straddles the run boundary**, and because the arms
strictly alternate q1->q2->q1, a wedge seeded by a q2 run always surfaces in the following q1 run. Only
`q1_1` (first run of the block, 1470 sweeps then wedged at teardown) is unambiguously a q1 event.
Any future arm comparison must randomize or block the order and probe health between runs.

### A real early-warning signal exists (§N+11's "no early warning" is superseded)

The failing runs log `MMIO per-op timeout: 4B load took 220214 us (budget=2 ms)` — **220 ms**, not 220 s;
check the units, the whole run spans ~1 s. A 4-byte MMIO load ~110x over budget, at the leading edge of
the wedge. The profiler then disables itself and the run continues unprofiled before hanging.

### Degradation requires a box FREEZE, not a card hang

**~510 runs today, 9 hangs, ZERO degradation.** ACK-WRITE never exceeded 471 ns (dynamic) / 219 ns (static).
Conditions covered: delay 500, and the full knee region 150/100/50/0, on both TLB paths.

The reason is already in this file: the historical degradation followed the **box hard-freezing** under a
sweep and the IRD watchdog rebooting it — not a PCIe hang. Verified on bh-05 with the documented check:
`last -x` shows **every** reboot today has a preceding `shutdown` record and there are no `crash` entries.
We were never in the state that produces degradation. Card hangs recover clean; freezes are what leave it
degraded.

### RETRACTED before it spread: "the static TLB window is immune to degradation"

Proposed on the basis that healthy static is 175 ns and healthy dynamic is 386 ns, matching the historical
"386 ns healthy" exactly — so the old degraded/healthy pair was indeed measured on the dynamic path. But the
immunity claim is **already refuted above in this file**: a static window degrades 13x, both drainers degrade
identically, and device *reads* degrade 3.8x without touching that window. The penalty attaches to the device
access itself. The static/dynamic split explains the healthy baselines and nothing about degradation.

### Correction to the recovery matrix: `tt-smi -r` DOES recover a HUNG card

Measured 3x today on cards reading `Unknown|63` with all-ones config space: `tt-smi -r` completed cleanly,
did not take the box down, and restored full health (post-recovery ack 392 / 404 / 171-176 ns). The standing
warning that reset kills a hung box did not reproduce. Warm `sudo reboot` also recovers (4/4). Neither
produces degradation.

### Dynamic-path block, for the record

160 runs at delay 150/100 with `NO_STATIC_TLB=1`: **2 hangs** (run 30 @ 150, run 90 @ 100), 0 degradation,
ack 367-471 ns throughout. An early "hung after 2 runs on dynamic" reading was coincidence — the full block
puts the dynamic-path rate at ~1.2%, indistinguishable from static.

## §N+17 — POLL PRESSURE IS REFUTED, and §N+16's "the timeout suppresses the hang" is in serious doubt

Three-arm test at the knee (delay 150), 100 runs each, **randomized order** (not alternating -- §N+16
showed a fixed rotation mis-attributes a boundary-straddling wedge):

| arm | timeout | yield in CQ poll | device progress read | hangs / 100 |
|---|---|---|---|---|
| `stock` | off | no | no | **3** |
| `yield` | off | **yes** (`TT_METAL_CQ_POLL_YIELD=1`) | no | **4** |
| `noprog` | on | yes | interval 3.6e6 ms = never | **3** |

**The poll-pressure theory is dead.** Adding `std::this_thread::yield()` to the stock completion-queue poll
-- the single mechanical difference the theory rested on -- changes nothing: 4/100 vs 3/100. The host is not
wedging the endpoint by hammering the hugepage.

**And the timeout does not suppress the hang.** `noprog` has the timeout armed and hangs at exactly the stock
rate. So §N+16's headline (0/179 armed vs 5/125 unarmed, Fisher p ~ 0.02) does not reproduce at delay 150.

Two candidate explanations remain, and they are being separated by a fourth arm:

1. **Block A's 0/179 was a lucky streak.** At the 3.3% rate measured here, P(0 in 179) ~ 0.0025 -- a 1-in-400
   coincidence, unlikely but not impossible, and exactly the kind of run this file has been fooled by before.
2. **The periodic device progress read is the active ingredient, not the yield.** `noprog` deliberately
   removed that read; block A had it at the default 100 ms. A fourth arm (timeout armed, DEFAULT interval,
   delay 150, 100 runs) is the missing cell. Clean => the read matters; ~3 hangs => explanation 1.

Note the two blocks also differed in delay (A/B at 500, this test at 150), so an interaction is not excluded
either. Do not quote §N+16 as established until the fourth arm lands.

**Method note that finally worked:** randomize arm order and record each run's PRECEDING arm. The hang detail
shows `prev` scattered across all three arms (`prev=stock`, `prev=yield`, `prev=noprog`), which is what the
alternating design could never produce -- under alternation the wedge always appeared to belong to whichever
arm followed. Randomization converts that bias into noise.

## §N+18 — FOUND IT: a periodic DEVICE READ prevents the endpoint wedge (bh-05, 2026-08-07)

The fourth arm lands and the matrix is complete. All at delay 150, 100 runs each, randomized order:

| arm | timeout | yield in CQ poll | periodic device progress read | hangs / 100 |
|---|---|---|---|---|
| `stock` | off | no | no | 3 |
| `yield` | off | **yes** | no | 4 |
| `noprog` | on | yes | **never** (interval 3.6e6 ms) | 3 |
| `arm4` | on | yes | **default, every 100 ms** | **0** |

**The active ingredient is the periodic device read, not the yield and not the timeout.** The three arms
without it sit at 3-4%; the one with it is zero. `yield` vs `stock` isolates the yield (no effect), and
`noprog` vs `arm4` isolates the read with everything else held identical (3 -> 0).

Pooling with the earlier delay-500 blocks, which used the same default interval:

| | hangs / runs | rate |
|---|---|---|
| periodic read present (block A @500 + arm4 @150) | **0 / 279** | 0% |
| periodic read absent (stock+yield+noprog @150, block B @500) | **15 / 425** | 3.5% |

Fisher p ~ 4e-4. At 3.5%, P(0 in 279) ~ 5e-5. This holds across two different delays.

### What the read actually is

`get_cq_dispatch_progress` (`command_queue_common.cpp:131`) reads L1 from the dispatcher core -- a small
MMIO read to the device, issued every `dispatch_progress_update_ms` (default 100) while a completion wait
is outstanding. It exists purely for timeout detection. Nothing about it was designed to touch the failure
path.

### Mechanism: unproven, but there is a strong hint

A small periodic host->device READ keeping the endpoint alive points at an idle/quiescent-state problem
rather than a traffic/pressure problem -- the opposite of every theory tried so far (egress bandwidth,
ingest rate, NoC choice, dispatch contention, host poll pressure). Suggestive supporting detail: the hung
runs log `Setting power state failed on device 0: Invalid argument`, and the driver logs
`Failed to set initial power state: -22` on failed opens. A device dropping into a power state it cannot
cleanly leave, kept out of it by periodic MMIO traffic, would fit every observation:
- the wedge appears at run boundaries / teardown, when device traffic goes quiet (S N+16)
- the link stays UP while the endpoint stops completing TLPs (S N+14)
- AER is clean and the kernel logs nothing -- no error occurred, the endpoint just stopped answering
- a 4 B MMIO load taking 220 ms is the leading edge (S N+16), i.e. an access that nearly did not come back

**This is a hypothesis, not a measurement.** Next step is to test it directly: vary
`dispatch_progress_update_ms` across e.g. 10 / 100 / 1000 / 10000 ms and look for a threshold, and read the
device's power-state telemetry across a wedge.

### Consequences

- **A cheap workaround exists today**: `TT_METAL_OPERATION_TIMEOUT_SECONDS=<n>` (any non-zero value) with
  the default progress interval. 0 hangs in 279 runs. It costs one 4 B device read per 100 ms.
- **S N+16 is REPAIRED, not retracted.** Its observation (0/179 armed) was real; its attribution to the
  yield was wrong, and S N+17 correctly killed that. The suppression is real and traces to the read.
- **S N+17's "the timeout does not suppress" is superseded**: it does, but only with the progress read
  enabled, which `noprog` had deliberately removed.
- Any hang-rate figure in this file must now state whether a periodic device read was in flight.

## §N+19 — RETRACT §N+18. The timeout escapes a TEARDOWN wait; it does not prevent the wedge.

§N+18 claimed the periodic device progress read prevents the endpoint wedge (0/279 armed vs 15/425 unarmed).
**That comparison was never apples-to-apples.** Timing the runs exposed it: armed runs take ~45 s, not ~5 s,
and the extra time is one stall whose duration equals `TT_METAL_OPERATION_TIMEOUT_SECONDS` exactly.

The stall is not in the completion queue:

```
TT_THROW: Device 0: Timeout (45000 ms) waiting for physical cores to finish: 14-3   (assert.hpp:104)
Device 0: Exception waiting for dispatch cores to finish during teardown.
          Continuing with cleanup.  (llrt.cpp:406)
```

`TT_METAL_OPERATION_TIMEOUT_SECONDS` bounds `wait_until_cores_done` at **teardown** as well. The exception is
**caught** and cleanup proceeds, so the process exits **0**.

| | teardown wait on core 14-3 | outcome | how I scored it |
|---|---|---|---|
| armed | bounded at 45 s, exception caught | exits 0 after ~45 s | **"clean"** |
| unarmed | unbounded | spins forever, killed at 300 s | **"hang"** |

Fired in **73/74** msweep runs and **27/100** arm4 runs. So the armed arms' zero hang count is substantially
an artifact of the timeout rescuing a teardown wait, not evidence that the endpoint stopped wedging.

**Corroboration missed at the time:** poll-pressure hang `k=138` recorded `card=32.0 GT/s PCIe|16` -- a
HEALTHY card. At least one "hang" was never an endpoint wedge. **Two distinct failure modes were being
pooled under one label**:
1. genuine PCIe endpoint wedge (`card=Unknown|63`, all-ones config space)
2. teardown `wait_until_cores_done` never completing (card healthy)

Any future run MUST classify by card state at failure, not by exit code alone.

**What survives from §N+18:** nothing about mechanism. The 0/279 number is real but measures "did not
hang OR was rescued by the timeout", which is not the quantity of interest.

**What still needs explaining**, and is now the sharper question: why does teardown wait on core 14-3 at all?
The DRISC drainer is resident by design, so a teardown wait for it to finish should either always fire or
never -- yet unarmed runs complete cleanly 97% of the time. Something makes that wait unsatisfiable
intermittently. That is a better-posed problem than the endpoint wedge and is probably upstream of it.

**Method lesson, third instance:** measure run DURATION, not just exit code. A 9x slowdown sat in the data
across ~280 runs and went unnoticed because every one of them exited 0. The earlier two instances were the
ordering trap (twice). Cheap invariants -- runtime, card state, log length -- catch this class of error
before it becomes a finding.

## §N+20 — §N+11's 2x2 IS SUSPECT: it was scored before WEDGE and TEARDOWN could be told apart

§N+11 concluded "only DRISC *and* fast dispatch hangs" (4/134 vs 0/294, Fisher p ~ 0.004) and called it the
strongest signal in this file. **It was scored on exit code / `PcieHangError`** -- i.e. it pools the two
failure modes that §N+19 proved are distinct:

- **WEDGE** -- PCIe endpoint stops completing TLPs, card reads `Unknown|63`
- **TEARDOWN** -- `wait_until_cores_done` never completes, card perfectly HEALTHY

Pooling these is exactly what produced and then destroyed §N+18. A cell recorded as "0/98 clean" only means
no non-zero exits were seen; it does not establish that no wedge occurred, and it says nothing at all about
teardown hangs, which under an unarmed build are SILENT (no log message -- see §N+19).

The file already documents a separate slow-dispatch failure at line ~1363: *"Under slow dispatch the DRISC
drainer wedges on sweep 1 and the host reports FAILED TO START"* -- a third state again, and one that would
not have been scored as a hang in the 2x2 at all.

**If the wedge occurs under slow dispatch too, the interaction claim is dead** and "DRAM-core egress in the
presence of fast-dispatch traffic" loses its only support. Everything §N+11 built on -- including the
`num_hw_cqs` experiment, which was chosen precisely to vary dispatch intensity -- inherits the doubt.

**Required: re-run the full 2x2 with the classifying harness**, scoring WEDGE / TEARDOWN / MASKED separately
and per-arm randomized. Needs two knobs wired into `drisc_hang_harness.sh`:
`TT_METAL_SLOW_DISPATCH_MODE=1` and `TT_METAL_PERF_DEBUG_DRAIN_TENSIX=1`. Until then, treat the
DRISC-x-fast-dispatch interaction as UNCONFIRMED, not established.

**Pattern worth naming, since it has now cost four findings in one day:** every wrong conclusion here came
from a coarse observable standing in for the thing of interest -- exit code for failure mode, wall-clock for
health, arm label for causation. The fix each time was a finer discriminator (card state, run duration,
randomized order), never more runs. When a result surprises, sharpen the observable before spending silicon.

## §N+21 — CONSOLIDATED STATE (2026-08-07). Read this section first.

Everything above is kept for audit -- five claims were retracted in one day and the reasoning that
killed them is worth more than a tidy file. **Where this section conflicts with anything above, this
section wins.**

### A. Armed vs unarmed, 400 runs at delay 125, randomized, fully classified

| arm | n | WEDGE | TEARDOWN | MASKED | CLEAN |
|---|---|---|---|---|---|
| UNARMED | 200 | **4** | 3 | 0 | 193 |
| ARMED (timeout 45 s, progress interval default 100 ms) | 200 | **0** | 2 | 1 | 197 |

- **Wedges 4 vs 0: Fisher p ~ 0.12. SUGGESTIVE, NOT SIGNIFICANT.** Do not quote as established.
- Pooled across all armed blocks at default interval: **0 wedges / 479 runs** (delay 500 n=179,
  delay 150 n=100, delay 125 n=200), against 4/200 unarmed in the one cleanly-classified block.
- **Teardowns are MATCHED: 3 vs 3** (counting the armed arm's 1 MASKED). Arming does not change how
  often teardown happens, only sometimes the outcome -- and **not reliably**: armed run k=338 still
  hung 301 s. The timeout is not a fix.
- The armed/unarmed split therefore separates cleanly: **teardown rate is a system property; wedge
  rate may not be.**

### B. The 2x2, re-run with classification (13 runs/cell, interleaved + randomized)

| | fast dispatch | slow dispatch |
|---|---|---|
| **DRISC** | 0 WEDGE, 1 TEARDOWN / 13 | 0 WEDGE, **13 TEARDOWN / 13** |
| **Tensix** | 0 WEDGE, 0 TEARDOWN / 13 | 0 WEDGE, **13 TEARDOWN / 13** |

**Slow dispatch fails 26/26 on BOTH drainers**, deterministically, card healthy every time:

```
TT_THROW: Device 0: Timeout (45000 ms) waiting for physical cores to finish:
          14-6, 14-7, 14-8, 14-11, 14-10, 14-9, 14-3, 14-2, 14-5, 14-4.
terminate called ... timeout: the monitored command dumped core
```

Ten cores (the whole DRAM/DRISC column) never finish, and **this call site's throw is UNCAUGHT** --
unlike the fast-dispatch teardown on the single core 14-3, which hits a try/catch and exits 0.

Two consequences:

1. **§N+11's "0/98 clean" for both slow cells cannot stand.** But note 26/26 vs 0/98 is too extreme
   for a pure scoring artifact -- a 100% failure rate would have blown the old harness's timeout every
   run. Something has ALSO changed since. Treat this as documenting current behaviour, not as a clean
   refutation.
2. **The 2x2 cannot answer the wedge question for slow dispatch, under any arming.** Slow runs abort
   at teardown before reaching the state where a wedge could appear. "0 wedges in slow" is
   structurally uninformative. Testing it requires FIXING the 10-core teardown first (stop the
   resident drainer before `wait_until_cores_done`, or give that call site the catch the fast path
   has). That is a code change, not a harness change, and it is the blocker.

### C. THE WEDGE, specifically -- consolidated

Distinct from TEARDOWN (healthy card, core-wait never completes) and from DEGRADED (13x MMIO latency).
Everything below is about `card = Unknown|63` only.

**Signature.** Endpoint config space reads **all `ff`**, including `max_link_speed` / `max_link_width`
-- static capability fields that cannot change and read correctly minutes earlier. Meanwhile the
**root port `0000:00:01.1` stays linked at 32.0 GT/s x16**. AER zero on both. So the *link* is healthy
and the *endpoint* has stopped completing TLPs. Always read the ROOT PORT to tell these apart; the
endpoint's own sysfs cannot.

**Host symptom.** The completion-queue reader spins (state R) in `completion_queue_wait_front`, whose
predicate compares the host read pointer against a **device-written** pointer fetched by
`read_cq_host_ptr<true>` -> `read_sysmem` -- i.e. **host DRAM**, not device MMIO. A dead endpoint can
never update it, so the host polls a stale host-memory word forever, touching nothing and logging
nothing. Unarmed that loop has no timeout, no yield, no device access. Main parks on a condvar in
`wait_for_outstanding_reads`.

**NEW AND MOST PROMISING -- IOMMU page faults.** `IOMMU: enabled`, and dmesg carries:

```
tenstorrent 0000:01:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x000d address=0xb50  flags=0x0000]
tenstorrent 0000:01:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x000d address=0x1000 flags=0x0000]
tenstorrent 0000:01:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x000d address=0x2000 flags=0x0000]
tenstorrent 0000:01:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x000d address=0x3000 flags=0x0000]
```

**Exactly four addresses, exactly 10 times each** = ten identical bursts. Constant domain `0x000d`,
`flags=0x0000`. Three are consecutive 4 KB pages from `0x1000`; `0xb50` is an unaligned lead-in.
Near-zero IOVAs repeating identically = a **stale or zeroed DMA address register**, not corruption.

This explains what previously looked contradictory: link up + AER clean because it is a *translation*
fault, not a PCIe error; endpoint stops completing because its DMA is blocked upstream. The earlier
"kernel is completely silent" claim in §N+14 is **WRONG** -- it was checked immediately after a reboot,
before any faults had accumulated.

**NOT yet established:** fault timestamps have not been correlated against individual wedge times.
"Accompanies" is not "causes". Next step is exactly that correlation (dmesg seconds-since-boot vs the
harness CSV), plus reading dmesg immediately after a captured wedge.

**Rate.** ~2-3% per run unarmed, across delays 125 / 150 / 500 -- no delay dependence found. The knee
is irrelevant to it.

**Recovery.** `tt-smi -r` clears it (measured 3x, box stays up, full health restored) -- seconds, not a
3-minute reboot. A warm host reboot also works (4/4). **Neither produces degradation.**

### D. Claims that are DEAD -- do not resurrect

- ~~egress bandwidth is the trigger~~ (saturates ~17 GB/s; amplification cannot exceed it)
- ~~the ingest axis / producer delay is the trigger~~ (confounded with cumulative runs)
- ~~cumulative runs predict it~~ (150 fixed-config runs clean)
- ~~config churn causes it~~
- ~~NoC choice matters~~ (NoC 1 hangs identically)
- ~~host poll pressure~~ (yield injected directly: 4/100 vs 3/100 over 300 randomized runs)
- ~~the periodic device read prevents the wedge~~ (§N+18; the comparison was masked teardowns)
- ~~the static TLB window is immune to degradation~~ (a static window degrades 13x)
- ~~degradation follows a card hang~~ (it follows a BOX FREEZE + watchdog reboot; `last -x` shows
  `crash` with no `shutdown` record)
- ~~the knee is a safety limit~~

### E. The method rule that would have prevented all of it

Every wrong conclusion came from a **coarse observable standing in for the thing of interest** -- exit
code for failure mode, wall-clock for card health, arm label for causation, a single block for a rate.
The fix was never more runs; it was always a finer discriminator. **When a result surprises, sharpen
the observable before spending silicon.**

## §N+22 — Tensix does not wedge: 0/200 against DRISC's 4/200, egress-matched

200 runs, delay 125, unarmed, fast dispatch, Tensix-BRISC drainer, fully classified:

| drainer | n | WEDGE | TEARDOWN | CLEAN |
|---|---|---|---|---|
| **DRISC** (§N+21 A, same conditions) | 200 | **4** | 3 | 193 |
| **Tensix BRISC** | 200 | **0** | 1 | 199 |

**The arms are egress-matched**, verified at delay 125 before the block ran -- within 0.2% on every
metric, so a Tensix null cannot be dismissed as "it just does fewer DMAs":

| | DRISC @125 | Tensix @125 |
|---|---|---|
| frames / pushes | 7,841 / 2,638 | 7,857 / 2,643 |
| words | 11,002,970 | 11,002,970 |
| pages | 1,293,765 | 1,296,405 |
| busy sweeps | 72 @ 71.6 us | 72 @ 71.5 us |

Staging is identical too (`7 slots x 10560 B`), so L1 capacity is not a factor. The only differences are
minor and both say the Tensix drainer is slightly less eager: max occupancy 430/512 vs 348/512, and
worst credit-wait 13.8 us vs 0.1 us -- consistent with sitting on a worker core contending with
dispatch rather than on a dedicated DRAM core. Neither changes bytes shipped.

**Statistics, stated honestly:**
- Today's matched comparison alone: 4/200 vs 0/200, **Fisher p ~ 0.12 -- suggestive, NOT significant.**
- Pooling §N+11's 196 Tensix runs: 4/200 vs 0/396, **p ~ 0.02**. Those old runs were exit-code scored,
  which we voided for TEARDOWN -- but a WEDGE leaves the card dead so the NEXT process dies loudly at
  device open with `PcieHangError`. Exit-code scoring is therefore *valid for wedges* even though it is
  blind to silent teardowns. The pooling still spans ~24 h of code changes, so treat p ~ 0.02 as
  indicative rather than clean.

**TEARDOWN hits both** (Tensix 1/200, DRISC 3/200) -- it is not drainer-specific. Only the WEDGE is.

**This is the strongest surviving structural claim about the wedge, and it fits the IOMMU lead**
(§N+21 C): the DRISC drainer egresses from a DRAM core whose DMA path differs from a worker's, and the
faults are four repeating near-zero IOVAs. A per-path address register that is stale or zeroed on the
DRAM-core path only would produce exactly this. **Next: correlate fault timestamps against wedge times,
then compare the DMA setup of the two paths.**

## §N+23 — Tensix does not wedge UNDER STRESS either: 0/400 vs DRISC 4/200 (p ~ 0.02)

Extends §N+22 to the below-knee regime. Delay 50 is genuinely stressed and matched to DRISC's:
**21,252 producer stalls, max occupancy 511/512** (vs 0 stalls / 430 occ at delay 125).

| condition | n | WEDGE | TEARDOWN | CLEAN |
|---|---|---|---|---|
| Tensix @125 (unstressed) | 200 | 0 | 1 | 199 |
| Tensix @50 (stressed) | 200 | **0** | 2 | 198 |
| **Tensix total** | **400** | **0** | 3 | 397 |
| DRISC @125 | 200 | **4** | 3 | 193 |

**Fisher 4/200 vs 0/400: p ~ 0.02.** Significant on today's data alone, without pooling §N+11's
exit-code-scored runs. Note the knee itself was re-measured today and sits between delay 100 and 50
(0 stalls at 100, ~21,260 at 50) -- NOT the historically recorded 125, so delay 125 was an unstressed
condition and delay 50 is the real back-pressure test.

**TEARDOWN hits Tensix at both delays** (1 and 2 per 200) -- same order as DRISC's 3/200. Teardown is
not drainer-specific and does not track stress. Only the WEDGE is DRISC-only.

**Conclusion: the wedge is a property of the DRISC egress path, not of load, not of the drainer's
staging, and not of egress volume.** All confounds are closed:
- egress matched at delay 125 within 0.2% (frames 7,841 vs 7,857; pages 1,293,765 vs 1,296,405;
  words identical at 11,002,970; busy sweeps 72 both)
- staging identical (`7 slots x 10560 B` = 73,920 B) on both core types
- stress covered on both sides of the knee
- grid identical (110 cores, 550 lanes, 5.5M markers)

This is now the strongest claim in the file and it converges with the IOMMU lead (§N+21 C): a DRAM
core's DMA path to host memory differs from a worker's, and the faults are four repeating near-zero
IOVAs -- the signature of an address register that only the DRAM-core path leaves stale or zeroed.

**Next, in order:** (1) correlate `AMD-Vi` fault timestamps against wedge times from the harness CSVs;
(2) get the actual IOVA of the CQ region and of the D2H socket buffer and see which sits near zero;
(3) diff the DMA setup between the DRAM-core and worker egress paths.

## §N+24 — SOLVED: the slow-dispatch TEARDOWN is a HARNESS GRID BUG, not a device fault (bh-05, 2026-08-07)

**§N+21 B's "slow dispatch fails 26/26 on BOTH drainers" is explained and fixed. It was the harness
running `--gx 0 --gy 0`.** The slow cells are now 10/10 CLEAN. Slow dispatch is no longer blocked, and
the "fix the 10-core teardown in code before slow dispatch can answer the wedge question" blocker in
§N+21 B is **void** — it was a one-flag harness change, not a code change.

### The mechanism

`compute_with_storage_grid_size()` is **dispatch-dependent**, and the two sides disagreed:

| | compute grid | drainer polls |
|---|---|---|
| fast dispatch | 11x10 = 110 (dispatch reserves the rest) | 11 columns |
| slow dispatch | **12x10 = 120** (`core_descriptor.cpp:247` logs `Using full logical grid (12, 10)`) | **11 columns** |

Nothing is reserved for dispatch under slow dispatch, so the compute grid gains a 12th column, while
the drainer deliberately holds that last column back (`perf_debug_profiler.cpp:613`, `reserve_column`).
`--gx 0` means "use the full grid", so the harness put producers on a column **no drainer polls**.
Those producers are lossless: they fill their SPSC rings, block forever in `ring_ensure_room`, never
finish, and the host dies in `wait_until_cores_done` — an **uncaught** 45 s throw that dumps core.

**The code already warned about exactly this**, at `perf_debug_profiler.cpp:603-607`: *"Run the workload
with `--gx 11` to match — the poll list built below stops at column 11, so a producer placed there would
both go undrained and scribble on the drainer's L1."* The harness never got the memo.

### The two arms fail differently — and that difference is the proof

| arm | hanging cores | why |
|---|---|---|
| **DRISC** | **10** — `14-2..14-11`, exactly column 12 | drainer is on a DRAM core and is fine; only the undrained column blocks |
| **Tensix** | **all 120** | the drainer *lives* in the reserved column, so a producer lands on its core and scribbles its L1 — nothing drains at all |

### ⚠ `14-2..14-11` ARE TENSIX WORKERS, NOT THE DRAM/DRISC COLUMN

Earlier notes called this "the whole DRAM/DRISC column". **Wrong, and it sent the investigation at the
drainer instead of at the producers.** `blackhole_140_arch.yaml` lists `1-2 … 16-11` under
`functional_workers`; `14-2..14-11` is one full worker column (x=14, rows 2–11). x=14 is the **12th**
worker column in noc0 x order (1,2,3,4,5,6,7,10,11,12,13,**14**,15,16) — i.e. logical x=11, precisely
the column the drainer reserves. **Always resolve a core list against the soc descriptor before naming
its core type.**

### Verified from three directions (bh-05, delay 125, `--iters 500`, card reset between)

| producers | drainer polls | result |
|---|---|---|
| 12x10 (`--gx 0`) | 11 cols | **HANG** — 10 cores (DRISC) / 120 cores (Tensix) |
| **11x10 (`--gx 11 --gy 10`)** | 11 cols | **CLEAN, rc=0** — both drainers |
| 12x10 (`--gx 0`) + `TT_METAL_PERF_DEBUG_FULL_GRID=1` | **12 cols** | **CLEAN, rc=0** (DRISC; 600 lanes) |

Closing the gap from *either* side fixes it, which is what makes this a grid mismatch and not a fault.

### Harness fixed

`drisc_hang_harness.sh` now takes `GX`/`GY` (default **11x10**) and never passes `--gx 0`. Post-fix,
`DISPATCH=slow` at delay 125, unarmed: **DRISC 5/5 CLEAN, Tensix 5/5 CLEAN** (was 13/13 TEARDOWN each),
2.1 s median both. 11x10 is also exactly the fast-dispatch grid, so **all four 2x2 cells now offer an
identical 110-core / 550-lane load** — the cells are comparable for the first time.

### Consequences

1. **Every slow-dispatch cell ever recorded is void**, including §N+21 B's 26/26 and §N+11's "0/98
   clean". They measured a misconfigured grid.
2. **The wedge question is now testable under slow dispatch.** Nothing has to be fixed in code first.
3. **TEARDOWN counts under FAST dispatch are unaffected** — fast dispatch's full grid is already 11x10,
   so there was never a mismatch there. §N+21 A / §N+22 / §N+23 (all fast) still stand.
4. Fast-dispatch teardowns on a *single* core (`14-3`) are a **different** thing from this — same
   column, but one core and caught by a try/catch. Do not merge the two without re-checking.

### Proposed guard (NOT implemented — decide deliberately)

`PROFILER_TERMINATE` (`profiler_common.h:166`) is checked **only** inside `ring_ensure_room_slow`
(`kernel_profiler.hpp:277`) and makes a producer *drop* the marker instead of spinning — it is a
producer escape hatch, **not** a drainer stop flag, so it is safe to set on a core that is not drained.
Setting it on the reserved columns at boot would turn the DRISC arm's silent 45 s hang into "column 12
is simply unprofiled". **It does not save the Tensix arm** — there the stray producer physically
overwrites the drainer's L1, which no flag prevents. Full protection needs the workload to clamp to the
drainer's producer grid, which means publishing that grid.

## §N+25 — DRISC now polls the FULL 120-core grid by default (bh-05, 2026-08-07)

Direct follow-on from §N+24. Reserving the last worker column was **only ever a comparability device** --
it made a DRISC (120 cores) and a Tensix (110, one of them being the drainer) sweep the same poll-list
length, so a difference between the arms could not be blamed on the grid. It was never a functional
requirement for a DRISC: that drainer sits on a **DRAM core** (`pick_unused_dram_logical_core`), so no
worker core is ever spent on it.

`TT_METAL_PERF_DEBUG_FULL_GRID=1` (opt-in) is replaced by `TT_METAL_PERF_DEBUG_RESERVE_COLUMN=1`
(opt-out). Default DRISC behaviour under slow dispatch is now the full 12x10.

**This also removes the §N+24 footgun at the source for the DRISC arm.** With the poll list covering all
120 cores, `--gx 0` ("use whatever grid the device offers") is safe by construction: the producer grid
and the poll list are the same 120 cores. The Tensix arm keeps the reservation -- its drainer physically
lives in that column -- so there `--gx 11` is still mandatory.

### Verified on silicon (bh-05, slow dispatch unless noted, card reset before the block)

| # | condition | poll list | result |
|---|---|---|---|
| T1 | DRISC default, `--gx 0` | 120 | CLEAN, 12x10, 600 lanes |
| T2 | DRISC default, `--gx 11 --gy 10` | 120 | CLEAN -- polling a column with **no program on it** is safe |
| T3 | DRISC + `RESERVE_COLUMN=1`, `--gx 11` | **110** | CLEAN -- the knob works, `cores [0,110) of 110` |
| T4 | DRISC default | **120** | `cores [0,120) of 120` |
| T5 | Tensix, `--gx 11` | 110 | CLEAN, drainer on logical (11,0) -- unchanged |
| T6 | DRISC default, `--gx 0`, **REAL DECODE** | 120 | **LOSSLESS** (below) |
| T7 | **FAST** dispatch, DRISC | 110 | unchanged -- `reserve_column` is gated on slow dispatch |

T2 was the one genuine risk in this change and it passes: the drainer polls the zeroed control vector of
an idle core and finds nothing, exactly as it does for a quiet producer.

**T6, the one that matters -- full grid, real decode, nothing lost:**

```
DRISC 0 resident ... cores [0,120) of 120, 7 staging slots x 10560 B
lanes=600  total markers=1200000
DRISC: 1526 sweeps (1492 idle), 2037 frames, 792 pushes, 2403240 words, 336105 pages,
       max occ 384/512, overflows 0
socket 0 drained: 336105 pages, 1201200 markers (329 reads); producer stall zones: 0
L1 STALL COUNTERS -- 0 producer stalls across 0 of 120 cores
BroadcastRing: ... consumer took 1201200 records, dropped 0
```

1,201,200 drained vs 1,200,000 offered: the +1,200 is exactly **2 sticky-timer markers per lane**
(600 lanes x 2), not loss. Zero overflows, zero stalls, zero drops, on all 120 cores.

### The harness now forces the reservation on slow cells

`drisc_hang_harness.sh` sets `TT_METAL_PERF_DEBUG_RESERVE_COLUMN=1` whenever `DISPATCH=slow`. Without it
a slow DRISC cell would poll 120 against Tensix's 110 and the 2x2 would be comparing **poll-list length**
-- i.e. idle sweep cost -- while appearing to compare core types. Measurement only; real captures should
take the full-grid default. Post-change: DRISC slow 3/3 CLEAN.

### ⚠ §N+6 ALREADY FOUND THE §N+24 FOOTGUN, AND IT WAS LOST

§N+6 states it outright: *"The one genuinely reproducible footgun: workload grid > drainer poll list
hangs the workload forever. `--gx 0` (120 producers) against a 110-core poll list leaves 10 cores
undrained; their rings fill and they block. Match them, or use `FULL_GRID=1`."* Fifteen sections later
§N+21 B met the same hang and classified it as a device-side TEARDOWN, which then blocked slow dispatch
entirely and voided a 2x2. **A known footgun buried in a long audit log is a footgun you will pay for
twice.** §N+6 also independently confirms full-grid runs are clean ("4 runs at the 110 grid, 6 at 120"),
which is corroborating evidence for this default change.

## §N+26 — The IOMMU lead is DEAD, and three failure names are one state (bh-05, 2026-08-07/08)

### RETRACT the IOMMU page-fault lead (§N+21 C's "BEST LEAD")

Decorrelated in **both** directions on the same day:

- **Faults without wedges.** The last `AMD-Vi IO_PAGE_FAULT` burst was **16:58:41**; the block running
  at that time was entirely clean.
- **Wedges without faults.** Two wedges at ~**18:50** and ~**19:10**. No fault logged at either, and
  none after 16:58:41 at all (53 fault lines total, all earlier).

The four near-zero IOVAs are real and still unexplained, but they are **not** the wedge. Do not spend
more time correlating them. §N+22/§N+23's "it fits the IOMMU lead" conclusions lose that support --
the DRISC-only asymmetry stands, its proposed explanation does not.

### The wedge reproduces with a fixed signature

120 runs, slow dispatch, DRISC, delay 125: **3 events**, each identical -- two aborts on a **healthy**
card, then `Unknown|63`. Rate ~2.5%, matching §N+21 A's 4/200 under fast dispatch.

The first symptom is not a dead endpoint, it is a **slow** one:

```
MMIO per-op timeout: 4B load took 220212 us (budget=2 ms), 4 of 4 bytes remaining.
  LaunchProgram -> Cluster::dram_barrier -> LocalChip::dram_membar
    -> insert_host_to_device_barrier -> set_membar_flag -> read_from_device
```

A 4 B MMIO load taking **220 ms** against a 2 ms budget. Note this is the DISCOVERY site, not
necessarily the fault site: `dram_barrier` is the first substantial MMIO read in a fresh process, so it
is simply where a card that died earlier gets noticed.

### WEDGE / MMIO_HANG / NocHangError are ONE state -- and this breaks per-run scoring

Three different-looking failures are the same hung card:

| surface | what it is |
|---|---|
| card `Unknown\|63` | endpoint config space all-ones |
| `MMIO per-op timeout` | a load that never completed |
| `NocHangError: NOC0 is hung on PCIe device ID 0` | UMD noticing at device open |

A hung card fails **every following run** until a reset, so one event produces an 8-9 run cascade. A
harness that resets only on `Unknown` lets cascades span many runs -- and if arms are interleaved, the
cascade contaminates **both** arms.

**SCORE CASCADES PER EVENT, NOT PER RUN.** An event is the first non-clean run after a clean one;
runs inside an ongoing cascade are uninformative and must leave the denominator. In §N+27 the same
data read 6-vs-3 ("probably noise") per run and 10-vs-0 (p ~ 0.001) per event. Randomizing arm order
does **not** rescue per-run scoring here -- it spreads the contamination evenly and hides the effect.

### One more scoring trap: a drainer that never starts

A run whose drainer fails to start **disarms the producers**, finishes fast, and scores CLEAN -- with
no egress at all, so no wedge was ever possible. An early 120-run wedge-rate block was scored entirely
this way and was void; the tell was `stalls=NA` on every row. Always record whether the drainer came
up (`resident on logical` in the log) and exclude runs where it did not.

## §N+27 — REFUTED: the drainer/`dram_membar` subchannel collision is not the wedge (bh-26, 2026-08-08)

**The hypothesis.** `pick_unused_dram_logical_core()` reserves a bank's WORKER and ETH endpoints and
returns the first subchannel left. UMD's `dram_membar()` barriers **subchannel 0** of every channel
(`dram_membar(channels, subchannel = 0)`; `Cluster::dram_barrier` passes no subchannel). On banks 0 and
4-7 the worker/eth endpoints are `[2,1]`, so the only free port **is** subchannel 0 -- the drainer lands
exactly on the core the host barriers, and the profiler puts that core in **stream mode**, which
redefines what an inbound DRAM-range address does. Banks 1-3 have endpoints `[0,1]`, so the drainer
gets subchannel 2, which the barrier never polls. Prediction: bank 0 wedges, bank 1 does not.

**The test.** `TT_METAL_PERF_DEBUG_DRISC_BANK` shifts the drainer's bank. Verified placement first:
bank 0 -> noc0 (0,0) = ch0 sub 0 (**collides**); bank 1 -> noc0 (0,3) = ch1 sub 2 (**clear**).
Everything else identical -- same drain, same 120-core poll list, same egress, same NIU flip. 200 runs,
randomized, delay 125, `--iters 500`.

| bank | vs barrier core | events | informative n | rate |
|---|---|---|---|---|
| **0** (default) | **collides** | **0** | 61 | **0.0%** |
| **1** | clear | **10** (all MMIO timeouts) | 64 | **15.6%** |

**Fisher p ~ 0.001**, events spread evenly across the block (k = 10, 24, 35, 84, 106, 122, 138, 154,
171, 197), so not an ordering or warm-up artifact.

**The prediction is not merely unsupported, it is inverted.** The colliding port is the SAFE one and
moving off it is ~15% fatal per run. The collision is not the mechanism.

**What it does establish:** the port `pick_unused_dram_logical_core()` hands back on bank 1 -- noc0
(0,3), channel 1 subchannel 2 -- is **not actually free to repurpose**, despite being neither a worker
nor an eth endpoint. "Unused" is again narrower than it reads (cf. §N+24, where "unused" column meant
unused-by-dispatch).

**KEEP THE DEFAULT AT BANK 0.** It is the known-good port. Bank 0 is NOT wedge-free -- it is the
baseline the wedge was always measured on (3/120 here, 4/200 in §N+21 A, ~1.7% pooled) -- but 0/61 at
that rate is unremarkable (p ~ 0.2), not immunity.

**CONSEQUENCE FOR MULTI-DRISC:** drainer `d` takes bank `d`, so scaling past one drainer places
drainers on exactly the banks measured dangerous here. **Validate every bank before running more than
one drainer**, or the socket-per-DRISC work sits on a substrate that hangs ~15% of the time.

**Still open after two dead hypotheses** (IOMMU §N+26, subchannel collision here): DRISC-only,
~2% per run, `tt-smi -r` recovers, and the NIU stream-mode flip remains the only state that outlives
the process.

## §N+28 — Where the 120-core knee actually goes: the SCAN, not read and not egress (bh-26, 2026-08-08)

Baseline, 120-core grid, slow dispatch, DRISC, `--iters 500`, warm runs only (run 0 discarded at every
delay -- the JIT run's idle-sweep prelude is ~15,400 against a warm ~1,480, and at delay 100 it showed
1,508 stalls where all three warm runs showed 0):

| delay | stalls (3 warm) | occ |
|---|---|---|
| 50 | 21,885 / 21,721 / 21,670 | 511 |
| 75 | 14,552 / 14,481 / 14,612 | 510 |
| **100** | **0 / 0 / 0** | 396 |
| 125 / 150 / 200 | 0 | 324-256 |

**Knee ~100** on this box at 120 cores.

### The phase split is stable across every delay -- and it says the obvious targets are the wrong ones

| phase | share of busy |
|---|---|
| **proc** | **46%** |
| unaccounted | 22% |
| read | 22.7% |
| wr-barrier | 7.2% |
| write | 1.5% |
| reserve(credit-wait) | 0.3% |

Egress **write is 1.5%** and read is 22.7%. So NoC0/NoC1 pipelining and bulk/bucket sizing -- the
standing optimization ideas -- attack a fifth of the cost and are capped at ~1.3x even if perfect.
Credit-wait is ~0, so the host is not back-pressuring.

### Splitting `proc` killed the obvious suspect

Added a sub-split (`c_ph_head`, results word 40/41) separating the per-core head write-back from the
local scan. On an issue-bound drainer, 120 small NoC writes per sweep is the natural suspect:

```
DRISC proc split: head-write-back 1.2% of busy (2.7% of proc) | scan 45.0% of busy
```

**The head write-back is 1.2%. The SCAN is 45%.**

### Ruled out: `volatile` serialization in the scan

The staged control vector is a settled snapshot -- the bulk read landed it and the read barrier already
waited -- so `volatile` was forcing 10 strictly-ordered loads per core for nothing. Dropped it:

| | scan % of busy | busy sweep |
|---|---|---|
| volatile | 45.0 | 70.0 us |
| plain loads | **44.9** | **69.5 us** |

No effect. The scan is not load-serialization bound.

### What the scan really is: the sweep is O(num_cores), full stop

~260 ns per core for ~10 L1 words of staged control vector, times 120 cores = ~31 us of scan per busy
sweep, on top of a 271 ns/core idle poll (32.6 us per idle sweep). Busy sweep 70 us, idle 32.6 us.
**That per-sweep latency is the knee**: a producer's 512-word ring must survive a whole sweep, and the
sweep cost does not depend on how many cores actually have data.

**So the lever is fewer cores per drainer, not faster reads.** `kNSockets` (perf_debug_profiler.hpp:90)
is `static constexpr uint32_t kNSockets = 1`, but every other part of the multi-drainer path is already
written for N: contiguous disjoint core slices, a D2H socket per drainer, separate L1, separate head
mirrors, no shared device state. N drainers divide both the scan and the poll.

**Blocked by §N+27**: drainer `d` takes bank `d`, and bank 1 measured 15.6% fatal, so `kNSockets = 2`
would place drainer 1 on a known-bad bank. A bank safety sweep across all 8 banks is the prerequisite.

## §N+29 — A DRISC drainer is only safe on a DRAM core in NoC row y == 0 (bh-26, 2026-08-08)

Prerequisite for multi-DRISC: drainer `d` takes bank `d`, and §N+27 measured bank 1 as fatal, so
"which banks are usable?" had to be answered before raising `kNSockets`.

8 banks x 25 runs, bank order randomized across the whole block, **card reset after every non-clean
run** so runs are independent and no cascade spans banks. Delay 125, full 120-core drain.

| bank | drainer core | channel row | y | fails / 25 |
|---|---|---|---|---|
| **0** | `0-0` | ch0 `[0-0, 0-1, 0-11]` | **0** | **0** |
| **3** | `9-0` | ch4 `[9-0, 9-1, 9-11]` | **0** | **0** |
| **7** | `0-0` (same core as bank 0) | ch0 | **0** | 1 |
| 1 | `0-3` | ch1 | 3 | 3 |
| 2 | `0-8` | ch2 | 8 | 2 |
| 4 | `9-2` | ch5 | 2 | 3 |
| 5 | `9-9` | ch6 | 9 | 5 |
| 6 | `9-5` | ch7 | 5 | 3 |

**Grouped: y == 0 -> 1/75 (1.3%); y != 0 -> 16/125 (12.8%). Fisher p ~ 0.006.**

**It is NOT the subchannel.** Banks 4/5/6 land on `9-2`, `9-9`, `9-5`, each of which IS subchannel 0 of
its channel, and they hang anyway. The one property every safe placement shares is the **top DRAM row**.
Failures are overwhelmingly `MMIO per-op timeout` -- the host's small reads stop completing.

There are exactly **two** safe cores on this part, `0-0` and `9-0`, one per DRAM side. Encoded as
`kSafeBanks[] = {0, 3}`, with a `TT_FATAL` if `kNSockets` exceeds it and
`TT_METAL_PERF_DEBUG_DRISC_BANK` retained as a diagnostic override.

**`pick_unused_dram_logical_core()` does not know any of this** -- it reserves worker and eth endpoints
only. "Unused" means unused by workers/eth, NOT safe to repurpose. That is the same trap as §N+24,
where the "unused" column meant unused-by-dispatch. Third time this file has been bitten by a name
that promised more than it meant.

*Why* row 0 is special is unproven. The soc descriptor notes CMFW reads DRAM telemetry through a
particular noc0 endpoint ("to avoid SYS-1419"), which is a lead, not a finding.

## §N+30 — Two DRISCs lower the knee 1.67x AND FREEZE THE HOST. Not worth it. (bh-26, 2026-08-08)

With drainer 0 on `0-0` and drainer 1 on `9-0` (both §N+29-safe), cores split `[0,60)` / `[60,120)`,
pages split evenly, 0 stalls:

| metric | 1 drainer | 2 drainers | |
|---|---|---|---|
| idle sweep | 32.6 us | **18.4 us** | ~2x |
| busy sweep | 70.0 us | **38.0 us** | 1.84x |
| **worst sweep** | 81.2 us | **58.8 us** | 1.38x |
| **knee** | **100** | **~60** | **1.67x** |

The worst sweep is the metric that matters -- an earlier 2-drainer attempt left it unchanged at ~95 us
and the knee did not move. On the safe placement it improves and the knee follows.

### And then it froze the host

| | 1 drainer | 2 drainers |
|---|---|---|
| per-run failure | ~0-1.3% (bank 0: 0/25, bank 3: 0/25) | **2/25 (~8%) at delay 125** |
| failure mode | card wedge `Unknown\|63` | abort, then **HOST FREEZE** |
| recovery | `tt-smi -r`, seconds | watchdog reboot, **card left DEGRADED** |

Evidence chain: the stability block stopped writing at 12:16 -> host uptime 11 min at 12:28 ->
`last -x` shows `reboot` with **no preceding `shutdown`** (the freeze signature) -> post-reboot probes
**ack-write 188 ns -> 2339 ns, device-read 787 ns -> 2942 ns**, i.e. the DEGRADED state, which
[[degraded_state_is_mmio_latency]] records as requiring a freeze to occur at all. The watchdog reboot
did NOT clear it.

**Each bank was 0/25 ALONE**, so this is an INTERACTION between two resident DRISCs -- two DRAM cores
held in stream mode simultaneously -- not additive per-drainer risk. Unexplained.

**`kNSockets` reverted to 1.** The code stays parameterized, so re-enabling is one line, but it needs a
STABILITY block and not just a knee measurement. A single drainer's worst case costs a `tt-smi -r`;
this one costs the box, and leaves it degraded afterwards.

**Consequence for measurement hygiene:** a degraded box silently inflates every sweep and knee number
by ~12x on small MMIO. Check the ACK-WRITE probe (healthy ~170-190 ns static) before trusting ANY
timing measured after a freeze.

## §N+31 — The 220 ms MMIO stall IS the root-port completion timeout, and the endpoint logs UnsupReq (bh-26, 2026-08-08)

Two register reads that change the shape of the wedge/freeze problem.

### The stall is a timeout firing, not a slow read

```
root port 00:01.1   DevCtl2: Completion Timeout: 65ms to 210ms, TimeoutDis-
endpoint 01:00.0    DevCtl2: Completion Timeout: 50us to 50ms,  TimeoutDis+
```

The root port's completion timeout is **enabled at 65-210 ms**, and the measured MMIO hang was
**220,212 us** (§N+26). Those are the same number. So the failing access is not slow -- **the endpoint
never completes it**, and the root port abandons it at ~210 ms. Every "MMIO per-op timeout: 4B load
took 220 ms" is one completion timeout. That also means the 2 ms UMD budget can never be met by a
retry: the hardware floor for a non-completing read on this box is ~210 ms.

### The endpoint IS logging PCIe errors -- and AER never showed them

```
endpoint 01:00.0    DevSta: CorrErr+ ... UnsupReq+
```

**`UnsupReq+` = Unsupported Request detected**, plus correctable errors, live on the current boot.

**This is why "AER is all zero" was misleading.** `DevSta` (PCIe capability + 0x0A) is a DIFFERENT
register from the AER capability. Every previous classification in this file read AER and concluded
"no PCIe errors". **Read `DevSta` too** -- `sudo lspci -vvv -s 01:00.0 | grep DevSta`. Its bits are
RW1C, so they are sticky until cleared, which also makes them a usable per-run probe.

An Unsupported Request is exactly what a host access to an address the endpoint cannot service looks
like -- and stream mode changes what an inbound DRAM-range address MEANS on a DRISC's core. That is a
mechanism, not yet a proof.

### The freeze chain this supports

1. a DRISC in stream mode makes some inbound address unserviceable -> UR, or no completion
2. host MMIO read stalls until the root port gives up at ~210 ms
3. UMD's 2 ms budget blows -> the `MMIO per-op timeout` abort
4. **two drainers = two such cores**, so twice the exposure and twice the rate of 210 ms stalls
5. enough concurrent 210 ms stalls in driver paths -> box unresponsive -> external reset

Consistent with 1 drainer wedging a card recoverably while 2 take the host down: same mechanism, twice
the surface, each event parking a CPU for a fifth of a second.

### Why nothing was logged when the host froze

`nmi_watchdog=1`, `watchdog=1`, `hung_task_timeout_secs=120` -- **the detectors were armed and none
fired**, `/sys/fs/pstore` is empty, and the previous boot's kernel log simply stops 11 minutes before
the freeze with no lockup, RCU stall, AER, MCE, panic or `tenstorrent` message. `systemd-detect-virt`
returns **none**, so this is bare metal and the physical host really did reset -- NOT the other
freeze pattern, in which the reservation VM hard-freezes (two processes stopping together, kernel
silent) and the IRD watchdog reboots the VM while the physical host stays up. There is no watchdog device and no
`/dev/ipmi0`, so the reset came from outside the box (IRD infrastructure) and nothing local recorded it.

### Correction to the recovery matrix

**A host reboot does NOT clear DEGRADED.** bh-26 rebooted at 12:17 and still measured ack-write
2339 ns / device-read 2942 ns afterwards. Earlier revisions list "host reboot" as the DEGRADED
recovery; that is wrong. Assume a cold power cycle or a different box.

## §N+32 — UnsupReq REFUTED; the hang strikes at DRAINER BRING-UP, inside set_drisc_niu_mode (bh-26, 2026-08-08)

### The UnsupReq hypothesis is dead

60 runs, 2 drainers, healthy card, `DevSta` cleared before every run so its RW1C bits are a per-run probe:

| class | n |
|---|---|
| CLEAN | 55 |
| MMIO_HANG | 1 |
| NOC_HUNG (cascade after it) | 4 |
| **UnsupReq set, on ANY run** | **0** |

`DevSta` read `0000` on the hang and on every cascade run. With AER already clean, **the card stops
completing MMIO while logging no PCIe error of any kind.**

**Correction to §N+31:** the `UnsupReq+` seen there was a STICKY bit on a card that had just been
through a freeze and a Gen1 downtrain -- residue from a different event, read as a live signal. The
per-run clear is what settled it. Sticky status bits are evidence only if you clear them first.

### Two scoring facts that change how to run these blocks

1. **A hung card reads HEALTHY for several runs.** `current_link_speed` stayed `32.0 GT/s` through all
   four cascade runs and only collapsed to `Unknown` after the block ended. So `Unknown` is a LATE
   symptom, and a harness that resets on `Unknown` (as this one did) under-resets and lets cascades
   run. Key recovery off the PROCESS outcome, not the card state.
2. **The rate is ~1.8%, not ~8%** -- 1 event / 56 informative runs. The earlier 2/25 was small-sample
   noise, quoted for a while as though it were solid.

### WHERE it hangs: the second drainer's bring-up

```
13:29:33.782  DRISC 0 resident on logical (0,1) [noc0 (0,0)], cores [0,60)   <- drainer 0 fine
13:29:34.227  init failed (MMIO per-op timeout: 4B load took 220219 us)      <- 445 ms later
              terminate ... 4B load took 198890 us
```

`ndrainers=1` on the failing run: drainer 0 up and resident, drainer 1 never reported. Two consecutive
~200 ms non-completing reads (each one root-port completion timeout, §N+31). **The failure is in
bring-up, not in the drain loop** -- which is a far narrower window to search.

### WHY: set_drisc_niu_mode launches a PROGRAM, and LaunchProgram does a dram_barrier

```cpp
void PerfDebugProfiler::set_drisc_niu_mode(...) {
    Program p = CreateProgram();
    CreateKernel(p, ".../drisc_niu_mode.cpp", drisc_logical, DramConfig{...});
    detail::CompileProgram(...);  detail::WriteRuntimeArgsToDevice(...);
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);
}
```

Flipping the NIU mode is not a register write -- it is a **full program launch on the DRAM core**, and
`LaunchProgram` calls `dram_barrier` (tt_metal.cpp:972), which MMIO-polls a core in EVERY DRAM channel,
then `wait_until_cores_done` polls the launched core. So bringing up drainer 1 runs a barrier over all
DRAM channels **while drainer 0 is already resident with its DRAM core in stream mode** -- and stream
mode is exactly what changes the meaning of an inbound DRAM-range address on that core.

This resurrects the §N+27 mechanism at a different trigger. §N+27 tested the collision by varying the
drainer's BANK under the user workload's barrier and refuted it; it never tested the profiler's OWN
barrier, issued during bring-up, against an already-resident stream-mode drainer. That is the untested
case, and it is where the failure actually happens. It also explains the 1-drainer rate: the user
workload's `LaunchProgram` barriers over the single resident drainer's stream-mode core every launch.

**Candidate fix: set NIU_CFG_0 by a direct host MMIO write instead of launching a kernel.** That
removes `dram_barrier` and `wait_until_cores_done` from the bring-up path entirely. Needs confirming
that NIU_CFG_0 is host-writable; if it is, the whole hazardous step disappears.

## §N+33 — A Gen1 downtrain is software-recoverable: force re-equalization, no cold power cycle

After a warm reboot, bh-26's link came up at **2.5 GT/s (downgraded)** against a 32 GT/s LnkCap, on BOTH
endpoint and root port, with reads ~3.5x slow and writes fine.

**Diagnose with the LnkCtl2/LnkSta2 pair, not the latency probes:**

```
LnkCtl2: Target Link Speed: 32GT/s     <- nothing pinned it to Gen1, so a retrain MAY reach Gen5
LnkSta2: EqualizationComplete- EqualizationPhase1-   <- the actual fault
```

Equalization gates training above 8 GT/s. It had failed, so the link fell back to Gen1. **Setting the
Retrain Link bit ALONE does nothing** -- equalization must be explicitly requested:

```bash
sudo setpci -s 00:01.1 ECAP_SECPCI+04.l=0x00000001    # LnkCtl3 bit 0 = Perform Equalization
cur=$(sudo setpci -s 00:01.1 CAP_EXP+10.w)             # then Retrain Link, on the ROOT PORT
sudo setpci -s 00:01.1 CAP_EXP+10.w=$(printf "%04x" $((0x$cur | 0x20)))
```

Result: `LnkSta: Speed 32GT/s (ok)`, and every probe back to baseline (ack-write 2306 -> 173 ns,
device-read 2738 -> 790 ns, worker-read 2661 -> 722 ns).

**Then reset the ASIC.** Immediately after the retrain the probes read perfectly healthy while the drain
path was still wrecked -- 12,979 stalls at delay 125 and a 276 us worst sweep. `tt-smi -r` cleared it and
the baseline landed exactly on the pre-incident numbers (idle 18.2-18.8, busy 39.2-40.5, worst 56-60,
0 stalls). **Clean latency probes are necessary but not sufficient; re-baseline the drain path too.**

Also: **`sock-read` GB/s is not a PCIe indicator.** It read 17.28 GB/s on a Gen1 x16 link whose ceiling
is ~4 GB/s, because it measures the host reading the D2H socket out of HOST DRAM.

## §N+34 — FIXED: the bring-up hang was one-launch-per-NIU-flip. Two drainers now default, knee 100 -> 20 (bh-26, 2026-08-08)

### The instrumented repro named the site, 4 times out of 4

A bring-up step tracker (`g_bringup_step`, reported in the `init failed` message) turned a 445 ms window
containing several MMIO paths into one line:

```
init failed at step [niu-mode(3,1)->1:LaunchProgram(dram_barrier+wait_until_cores_done)]
```

`(3,1)` is drainer 1's core (noc0 `9-0`). Every cascade-initiating run had `ndrainers=1` -- drainer 0 up
and resident, drainer 1 dying. **The failure is the SECOND drainer's NIU flip**, not the drain loop.

Mechanism: `set_drisc_niu_mode` is not a register write, it is a full `LaunchProgram` on the DRAM core,
and every `LaunchProgram` carries a `dram_barrier` (MMIO-polls a core in EVERY DRAM channel) plus
`wait_until_cores_done`. Done one flip per drainer, the second flip's barrier runs while the first
drainer is **already resident with its DRAM core in stream mode** -- and stream mode is precisely what
changes the meaning of an inbound DRAM-range address on that core. The read never completes and the host
eats a root-port completion timeout (~210 ms, N+31).

Note this is the N+27 mechanism at a trigger N+27 never tested. N+27 varied the drainer's BANK under the
USER WORKLOAD's barrier and refuted it; the profiler's OWN barrier during bring-up was the untested case.

### The fix: one launch for all NIU flips

`set_drisc_niu_mode` now takes a vector of cores and issues a single `LaunchProgram` over a
`CoreRangeSet`, as a pre-pass before the drainer loop. One barrier, and it runs **before any core is in
stream mode**, so the ordering cannot arise.

| 80 runs, 2 drainers, delay 125 | before | after |
|---|---|---|
| CLEAN | 42 | **80** |
| MMIO_HANG | 4 | **0** |
| NOC_HUNG (cascade) | 31 | **0** |
| WEDGE | 3 | **0** |
| host freeze | yes | **no** |

Fisher on cascade-initiating events (3/45 informative vs 0/80): **p ~ 0.045**. Significant but thin on
its own -- the mechanistic evidence (4/4 named site, fix removes exactly that ordering) is the stronger
half.

### RETRACT "the 2-socket knee is 60" (N+30). It is 20.

The N+30 sweep contained **4 failed runs** -- the very hangs fixed here -- and they were not merely noise
next to the measurement, **they were depressing the measurement**. Re-run on the fixed path, 3 warm
repeats per point, no failed runs anywhere:

| delay | stalls (3 warm) | occ |
|---|---|---|
| 0 | 23,286 / 23,292 / 23,287 | 511 |
| 10 | 22,867 / 22,948 / 23,013 | 511 |
| **20** | **0 / 0 / 0** | 432-444 |
| 30 / 40 / 50 / 60 / 75 / 100 | 0 everywhere | 228-352 |

**Knee 20**, sharp, corroborated by occupancy (pinned 511 below it, ~435 above). Against the 1-drainer
knee of 100 that is **5x**, well beyond the ~2x that halving per-sweep cost predicts -- below the knee a
single drainer falls into a feedback loop (producers stall -> rings pin at 511 -> sweeps cost more) that
two drainers never enter. Volume verified, not assumed: 6,001,620 words per drainer, 12,003,240 total =
2 words x 6M markers, split evenly, 0 overflows.

**The lesson worth keeping:** the earlier sweep's failed runs were noted as contamination and the knee
was quoted anyway. Checking that the DRAINER started was not enough -- the CARD's health has to hold
across the whole sweep, or the knee is measured on a machine that is partly broken.

### kNSockets = 2 is now the default

One D2H socket per drainer. 2 is the ceiling for two independent reasons: exactly two DRAM cores measure
safe (row y == 0, N+29), and TLB windows allow 2x7=14 of 16 while 3x7=21 does not.

**Still open and NOT addressed by this fix:** the single-drainer wedge (~1.8%/run) has a different
trigger -- one drainer only ever does one flip, so this ordering never existed there.

## §N+35 — The single-drainer wedge does NOT reproduce on bh-26: it is BOX-DEPENDENT (2026-08-08)

250 runs, **one** drainer, delay 125, 120-core grid, slow dispatch, card reset after any non-clean run
so the runs are independent, bring-up tracker armed: **250/250 CLEAN.** Zero wedges, zero MMIO hangs,
zero PCIe status bits, host untouched.

Pooled with the bank-safety sweep's safe-bank runs (banks 0 and 3, also single drainer, also 0 failures):
**0 / 300 on bh-26.**

| box | single-drainer config | events | runs | rate |
|---|---|---|---|---|
| **bh-05** | bank 0, slow dispatch, delay 125 | **3** | 120 | **2.5%** |
| **bh-26** | bank 0, slow dispatch, delay 125 | **0** | 300 | **0%** (95% upper bound 0.99%) |

**Fisher one-sided p = 0.023.** The two boxes do not have the same wedge rate. Configuration matches --
same grid, same dispatch mode, same delay, same drainer core (`0-0`), same binary lineage.

### CORRECTION: "~1.8% per run for a single drainer" was wrong

That figure came from the 60-run UR-probe block, which ran **two** drainers -- and its one event was the
bring-up hang that §N+34 then fixed. It was quoted here as a single-drainer rate. The only real
single-drainer evidence is the table above.

### Consequence: bh-26 is the wrong instrument for this bug

At bh-05's 2.5%, a 300-run block on bh-26 should have produced ~7 events. It produced none. So:

- **Any "fix" for the single-drainer wedge validated on bh-26 is meaningless** -- the box cannot
  distinguish a fix from its own baseline. This is the same class of error as measuring a knee on a
  degraded card, and it would have been very easy to declare victory here.
- Chasing it needs **bh-05 back, or another box that demonstrably shows it**. Confirm the box reproduces
  BEFORE testing any hypothesis on it.
- It also reopens what "DRISC-only" meant: §N+22/§N+23's 4/200-vs-0/400 DRISC-vs-Tensix comparison was
  measured entirely on bh-05. A box-dependent effect could equally be a box-plus-core-type effect, and
  nothing here separates them.

The bring-up tracker is in the tree and armed, so whenever a box that DOES reproduce is available, the
next single-drainer wedge names its own stall site instead of leaving a 445 ms window to guess inside --
which is how three hypotheses died (IOMMU N+26, subchannel collision N+27, UnsupReq N+32).

## §N+36 — Decode+publish on a healthy card: the HOST IS NOT THE CONSTRAINT, and several earlier claims were degraded-card artifacts (bh-26, 2026-08-08)

### ⚠ READ THIS FIRST: a large block of today's host work was measured on a DEGRADED card

The box froze and rebooted at 14:01. **The card came back degraded and I did not re-check it** -- despite
§N+33 saying to. Everything measured between 14:01 and ~17:45 ran with ack-write at **2.3 us instead of
175 ns**, i.e. a ~13x MMIO penalty, which throttles producers as well as the host.

**The ACK-WRITE probe prints on EVERY run.** It read 2,318 ns for hours. Healthy is ~175 ns. Compare it
before trusting any timing; it costs nothing and would have caught this immediately.

### RETRACTED, all measured on the degraded card

- **"The ack costs 2.6 us/read, 15x the probe."** No anomaly at all: the probe ALSO read 2.3 us at the
  time. On a healthy card the ack is **0.1 ms/run, ~110 ns/read** -- the cheapest stage in the pipeline.
  The sfence/store-drain theory built on top of it was chasing a phantom (and was independently refuted:
  pre-draining changed the ack by 2%).
- **"Cost is proportional to READS, not bytes"** and the whole framing-efficiency diagnosis. That
  signature came from a 2.3 us ack making per-read overhead dominate. On a healthy card the stalls fall
  MONOTONICALLY with delay -- the non-monotonic inversion the theory explained does not exist.
- **"Pacing gives 0 stalls at delay 20-125 with decode on."** On a healthy card nothing in 20-125 is
  clean. The controller's tuned thresholds were fitted to a broken machine.

### The corrected host budget (healthy card, decode+publish, Tracy off, pacing off)

| | delay 20 | delay 125 |
|---|---|---|
| WRITER thread busy | **17%** | **30%** |
| copy | 7.2 ms | 11.7 ms |
| ack | 0.1 ms (168 ns/read) | 0.2 ms (131 ns/read) |
| resize | **0.0 ms** | 2.3 ms |
| DECODER thread | ~28 ms (60% of writer wall) | ~30 ms (62%) |
| producer stalls | 22,368 / 22,227 / 22,093 | 98 / 7,709 / 612 |

**The writer is 70-83% IDLE and the decoder ~60% loaded, while producers stall 22,000 times.** The host
has ample headroom, so it is not what stalls the device. Device phases agree: credit-wait is only
4-6%, i.e. the drainer is barely waiting on the host at all -- it simply cannot sweep 120 cores fast
enough. The limit is the O(num_cores) sweep, which is why two drainers helped and host tuning does not.

### What the host work DID buy: resize is gone

`resize` was **9.6 ms/run, 76% of the writer's workload**, against 3.0 ms for the copy it precedes --
pure `std::vector` value-initialization of 64 KB immediately before `memcpy` overwrote every byte.

Two fixes were needed and the first alone did nothing:
1. **Grow-only sizing** + carrying the valid length in `DecodeItem::words` (not `buf.size()`), and
   dropping the `clear()` that reset every pooled buffer to size 0.
2. **Pre-populating the pool** (320 buffers/socket at max read size). This was the one that mattered:
   the decoder runs **231-253 buffers behind**, so the pool was dry and nearly every read
   default-constructed a FRESH size-0 vector, which grow-only cannot help.

Result: **9.6 ms -> 0.0 ms** at delay 20 (2.3 ms at 125), writer 27% -> 17% busy. Real, and independent
of card health -- but it did **not** change producer stalls (22.2k at delay 20 either way), which is
itself the evidence that the host was never the constraint.

### Instrument fixes (three separate unit/attribution bugs, all of which changed conclusions)

- **steady_clock::now() costs ~650 ns here.** An EMPTY timed region measured **1,303 ns** -- the
  instrument was a third of the "ack" it was measuring. Per-read timers moved to `rdtsc` (~16 ns).
  Always time an empty region before trusting a fine-grained split.
- **Ticks reported as nanoseconds.** After the rdtsc move, one summary line still divided ticks by 1e6,
  reporting the same copy as "56.0 ms" while the split line correctly showed 12.4 ms.
- **Two threads, one budget.** `decode`/`publish` run on the DECODER thread but were charged against the
  WRITER's wall, producing "134% busy" with NEGATIVE idle. Now reported as separate budgets.

### The pacing controller is kept, UNTUNED

`TT_METAL_PERF_DEBUG_FILL_PCT` (default 70) drives the inter-sweep gap from measured span fill, with a
3-level response: hard stop to 0 above 7/8 ring occupancy, 25% back-off above 3/4, otherwise a
multiplicative fill-error step. `TT_METAL_PERF_DEBUG_GAP_MAX` (default 200,000 cycles) bounds it; a gap
pinned exactly at that value in the results means the ceiling is clipping the loop.

**Its thresholds were fitted on the degraded card and must be re-derived**, and on current evidence the
host is not the bottleneck it was meant to relieve. Kept because the mechanism is sound and the hook was
always intended ("the hook a pacing controller would drive"), not because the tuning is trustworthy.

## §N+37 — The REAL knee on a healthy card: 1 drainer ~250, 2 drainers ~150 (~1.7x). "Knee 20" was a desynchronized-launch artifact (bh-26, 2026-08-08)

### RETRACT "knee 100 -> 20, a 5x improvement" (N+34)

Both of those numbers were measured AFTER the 14:01 freeze, on the MMIO-degraded card. Re-measured on a
verified-healthy card (ack-write 171-186 ns, device-read ~790 ns), 120 cores, slow dispatch, 2 sockets,
`NO_DECODE`, pacing off, discard + 2-3 warm per point:

| delay | 1 drainer | 2 drainers |
|---|---|---|
| 100 | 4,623 / 5,042 | 8,378 / 7,888 |
| 150 | 3,127 / 3,215 | **0 / 15** |
| 200 | **0 / 312** | -- |
| 250 | **0 / 0** | **0 / 0** |
| 400 | 0 / 0 | 0 / 0 |

**1-drainer knee ~250, 2-drainer knee ~150. The two-drainer win is ~1.7x, not 5x.** It is real, and it
is the same direction as before -- only the magnitude was inflated.

### The mechanism was NOT "a degraded card throttles the producers"

That explanation was wrong and a deliberate experiment killed it. **Producers are Tensix cores running
on-device nop loops; PCIe latency cannot slow them.**

Forcing the link to Gen1 on purpose (`LnkCtl2` Target Link Speed = 1 + retrain -- a controlled,
reversible way to degrade PCIe) produced the OPPOSITE of a fake knee:

| delay | Gen1 link | healthy |
|---|---|---|
| 10 | 23,398 | -- |
| 20 | 23,368 | 22,440 |
| 60 | 23,243 | 9,997 |
| 150 | 22,977 | 0 / 15 |

Flat ~23,000 stalls at every delay, occupancy pinned 511. The knee moved PAST 150, because slow reads
(2,735 ns) cripple the DRAINER's polling.

### What actually produced the fake knee: LAUNCH SKEW

The two degraded states differ in WHICH DIRECTION is slow, and that is the whole story:

| state | ack (host write) | device-read |
|---|---|---|
| Gen1 downtrain | **184 ns (fast)** | 2,735 ns |
| MMIO-degraded (the 14:01 one) | **2,318 ns (SLOW)** | 2,940 ns |

Under slow dispatch the launch messages are host MMIO **writes**, issued serially to all 120 cores. At
2.3 us instead of 175 ns the launch takes ~13x longer, so producers start badly SKEWED instead of firing
together. Peak *concurrent* marker rate collapses while total markers are unchanged, and the drainer
copes trivially -- exactly the effect [[drisc_knee_not_comparable_across_dispatch]] already documents for
fast-vs-slow dispatch (knee 25 vs 125).

**Gen1 cannot reproduce it because Gen1 leaves WRITES fast**, so launch stays synchronized and only the
drainer suffers. Occupancy corroborates: fake-knee runs sat at occ ~350 (rings never filling, i.e. low
peak rate) while healthy runs pin at 511.

So the knee of 20 was real for a machine whose producers were desynchronized by a 13x slower launch
path. It was never a property of the drainer.

### The rule this yields

**A knee is only meaningful with the card's health stated alongside the dispatch mode.** The existing rule
was "never quote a knee without the dispatch mode"; add "and without the ACK-WRITE probe value", because
a slow WRITE path silently desynchronizes producers and flatters every number downstream.

## §N+38 — The knee is set by the WORST sweep's CREDIT-WAIT, and the per-read page cap must clear a whole FRAME (bh-26, 2026-08-08)

### Averages pointed at the wrong phase for hours

Mean-sweep phases said read ~46% and proc ~23%, so three micro-optimizations went there. All three were
end-to-end null:

| change | phase effect | stalls |
|---|---|---|
| scan unrolled into registers | scan 42% -> 23% | none |
| read split across cores (both NoCs) | read 47% -> 42% | none |
| read split within one core's span | -- | none |

Capturing the phases OF THE WORST SWEEP (not the mean) explained why instantly:

```
delay 100:  WORST 125.1 us = read 3.8 + proc 13.9 + credit-wait 89.4 + write 7.1 + wr-barrier 7.0  (97% accounted)
delay 150:  WORST 128.3 us = read 3.8 + proc 14.0 + credit-wait 93.0 + write 6.7 + wr-barrier 7.0  (97%)
```

**Credit-wait is 2.8% of the MEAN sweep and 71-93% of the WORST one.** The knee is decided by the worst
sweep beating ring-fill time, so read+proc -- 17.8 us of a 125 us worst sweep -- could never move it.
Note the worst sweep does LESS read work than a mean sweep (3.8 us vs ~22 us): it is not busy, it is
BLOCKED.

**Rule: optimize the worst sweep, not the mean. They have different bottlenecks.**

### THE FIX: the per-read page cap must clear at least one whole frame

`TT_METAL_PERF_DEBUG_MAX_PAGES` at delay 100, 120 cores, 2 drainers, `NO_DECODE`, healthy card:

| cap (pages) | stalls |
|---|---|
| 4096 | 14,652 / 14,943 |
| 1024 (compiled default) | 8,432 / 8,269 |
| 512 | 3,903 / 3,622 |
| 256 | 764 / 1,015 |
| 208 | **0 / 0** |
| 192 | 0 / 333 |
| **176** | **0 / 0** |
| **165 (= kPagesPerSlot, one frame)** | 1,550 / 2,114 |
| 160 | 1,904 / 3,683 |
| 144 | 7,266 / 8,230 |
| 128 | 15,441 / 15,214 |
| 0 (uncapped) | **run produces no output at all -- broken** |

**The cliff is exactly at one frame.** `kPagesPerSlot = 165` pages is one core's span. At or below it a read
cannot be guaranteed to retire a WHOLE frame, so the host keeps pulling partial frames, credit is returned
in dribs, and the drainer waits; just above it every read clears at least one complete frame. That is a
STRUCTURAL boundary, not a fitted constant -- which is why 165 still stalls and 176 is clean.

So the right default is `kPagesPerSlot + headroom`, DERIVED, not the magic 1024.

### Knee 150 -> 100 (1.5x)

With cap 176, 2 drainers, `NO_DECODE`:

| delay | cap 176 | default 1024 |
|---|---|---|
| 60 | 12,496 / 12,330 | 9,997 / 9,992 |
| 80 | 11,206 / 13,398 | -- |
| **100** | **0 / 0** (occ 476-484) | 8,378 / 7,888 |
| 150 | -- | 0 / 15 |

First genuine end-to-end win of the session -- everything else made a phase faster while the total stood
still. Occupancy at the knee drops off the 510 ceiling to 476-484, so there is real margin.

**Caveats:** below the knee the small cap is slightly WORSE (12,000 vs 10,000 stalls at delay 60) -- once
saturated, smaller reads cost throughput. And this is `NO_DECODE`; decode+publish adds host work that will
move the optimum, so 176 must not be adopted as a default until re-measured with decode on.

### Also fixed here

A phase-accounting underflow: `c_proc += elapsed - nested` wrapped when the nested ship_run time exceeded
the elapsed span, printing `proc 18727729111430.1%` and silently corrupting the whole breakdown. Now
saturating.

## §N+39 — The DRAM frame ring is RUNWAY, not headroom: the volume knee is 20k zones/RISC, and 12 MiB is enough for 5k (bh-26, 2026-08-11)

Two sweeps, both at 120 cores / slow dispatch / delay 40 / full decode+publish. All numbers are 3 warm
reps; run 1 of each JIT key discarded.

### Sweep A — ring size at fixed volume (5,000 zones/RISC, 6.0 M markers, Tracy attached on rep 1)

| ring | frames | producer stalls r1/r2/r3 | cores hit | ring filled | records dropped | ts regressions |
|---|---|---|---|---|---|---|
| 64 MiB (default) | 6355 | 0 / 0 / 0 | 0 | no | 0 | 0 |
| **12 MiB** | 1191 | **0 / 0 / 0** | 0 | no | 0 | 0 |
| 10 MiB | 992 | 0 / 0 / 45 | 0/0/17 | 1 of 3 | 0 | 0 |
| 9 MiB | 893 | 662 / 102 / 80 | 16-37 | 3 of 3 | 0 | 0 |
| 8 MiB | 794 | 698 / 448 / 656 | 30-35 | 3 of 3 | 0 | 0 |
| 7 MiB | 695 | 1635 / 1076 / 1288 | 55-75 | 3 of 3 | 0 | 0 |
| 6 MiB | 595 | 1712 / 1743 / 2038 | 78-83 | 3 of 3 | 0 | 0 |

**Undersizing the ring never costs correctness.** Every size down to 6 MiB dropped zero records with zero
timestamp regressions -- the back-pressure chain works exactly as designed: ring fills, filler waits for
room, producers stall, nothing is lost. Record counts come in slightly ABOVE nominal 6,001,200 because
the stalls themselves emit markers. What undersizing costs is workload perturbation, monotonically:
0 -> ~600 -> ~1300 -> ~1800 stalls as you go 12 -> 8 -> 7 -> 6 MiB (at 6 MiB that is 0.034% of markers
across 80 of 120 cores).

12 MiB is the smallest size that is stall-free 3/3, and it holds because the natural burst high-water is
~900-1,000 frames -- 1,191 gives ~20% margin. 10 MiB has essentially none and failed 1 of 3.

### Sweep B — volume at the default 64 MiB ring (no tracy-capture attached)

| iters | zones/RISC | markers | producer stalls r1/r2/r3 | cores | filler high-water | ring-room waits | records dropped |
|---|---|---|---|---|---|---|---|
| 500 | 5,000 | 6.0 M | 0 / 0 / 49 | 0-18 | 645-926 (10-15%) | 0 | **0** |
| 1000 | 10,000 | 12.0 M | 0 / 0 / 0 | 0 | 3011-3246 (47-51%) | 0 | **0** |
| 1500 | 15,000 | 18.0 M | 5 / 0 / 0 | 0-2 | 5696-5938 (90-93%) | 0 | 0.80-1.10 M |
| **2000** | **20,000** | **24.0 M** | **14144 / 13793 / 12273** | **120/120** | **6355 (100%)** | **250-284** | 6.8-6.9 M |
| 2500 | 25,000 | 30.1 M | 37982 / 35518 / 35626 | 120 | 6355 (100%) | 574-621 | 13.1-13.2 M |
| 3000 | 30,000 | 36.1 M | 59932 / 60159 / 61007 | 120 | 6355 (100%) | 912-956 | 19.1-19.2 M |
| 4000 | 40,000 | 48.2 M | 106891 / 109052 / 107321 | 120 | 6355 (100%) | 1581-1646 | 30.6-31.2 M |
| 5000 | 50,000 | 60.3 M | 154485 / 153763 / 153361 | 120 | 6355 (100%) | 2232-2257 | 41.4-42.1 M |

**The knee is 20,000 zones/RISC (24 M markers).** 15k is clean (0-5 stalls, ring 92%, never fills); 20k
is a cliff -- ring pins at 100%, ring-room waits appear, ~13,000 stalls across ALL 120 cores. Past it,
stalls grow ~11x from 20k to 50k for 2.5x the volume.

### THE STRUCTURAL POINT: high-water tracks total volume and never plateaus

14% -> 49% -> 92% -> 100% at 5k/10k/15k/20k zones/RISC. The movers drain permanently slower than the
fillers stage, so the ring absorbs a running DEFICIT, not bursts. It buys a fixed amount of TIME before
back-pressure reaches producers -- ~16-17k zones/RISC at 64 MiB -- consumed monotonically from the first
zone.

Consequences, and they invalidate two earlier readings:

- **"0 stalls at 5k zones" is not a steady-state property.** The workload ends before the ring fills.
- **A low high-water does not mean the ring is oversized.** ARCHITECTURE.md's recommendation to cut
  `ROLE_RING_MB` "a long way" rested on the 8.4%-18.9% occupancy of 5k-zone runs; that occupancy is a
  statement about volume, not about sizing. Corrected there.
- **Doubling the ring cannot make a long capture stall-free** -- 128 MiB moves the knee to ~32k
  zones/RISC and no further.

### Host-side loss breaks FIRST, one sweep point earlier

At 15k zones producers are still clean but the host record ring is already dropping 0.8-1.1 M records.
The first failure as volume scales is SILENT HOST LOSS, not producer perturbation -- watch stall
counters alone and you will call 15k healthy while a twentieth of the trace is already gone.

The host consumer took ~16.9-17.6 M records in EVERY run from 15k zones up, whether 18 M or 60 M were
published -- essentially the `TT_METAL_PERF_DEBUG_RING_RECS=16777216` cap. That flat ceiling, not the
DRAM ring, is the real volume limit, consistent with decode being 1820% oversubscribed at 50k.
Order integrity held everywhere: 0 regressions at publish and consume in all 42 runs.

### DRAM footprint: the allocator is LOCK-STEP

Cost is `page_size x num_banks`, not `npages x page_size`. `page_size == ring_bytes` here (one ring per
page), and bh-26's allocator has **7** DRAM banks -- 8 channels, 1 harvested -- confirmed on silicon by
forcing the ring-bank validation message (`TT_METAL_PERF_DEBUG_ROLE_RING_BANKS=99,100` ->
"the allocator has 7 DRAM banks"). So:

| ring | DRAM reserved | vs old profiler |
|---|---|---|
| 64 MiB (default) | **448 MiB** (128 addressed, 320 wasted) | 14x |
| 12 MiB | **84 MiB** | 2.6x |
| old profiler | 4.58 MiB/bank = **32.0 MiB** | 1x |

The old profiler's figure is `per_risc_bytes x MaxProcessorsPerCoreType x CEIL_NUM_CORES_PER_DRAM_CHANNEL`
= `48,000 x 5 x 20` = 4,800,000 B/bank, reserved at `DRAM_PROFILER_BASE` in every bank
(`bh_hal.cpp:41-53`); `48,000 = 2 marker words x (2 program-id + 4 guaranteed) x 4 B x 1000`
(`DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT`, `profiler_state_manager.cpp:19,40-60`). Note theirs is
PROPORTIONAL (scales with RISCs x cores/bank x program count) where ours is FIXED, so the ratio is a
property of this grid, not universal.

**Two traps found here.** The runtime log understates the footprint ~4x -- it prints
`npages x ring_bytes` (191 MiB) rather than what the allocator reserves (448 MiB); budget from that line
and you are off by 257 MiB. And which bank a ring occupies is irrelevant to footprint: lock-step reserves
the same offset in every bank, so keeping rings off banks 0/3 costs nothing and saves nothing. The only
lever is `page_size`.

### The two lanes are asymmetric

Filler 1 fills first in every undersized run while filler 0 sometimes never fills at all (8 MiB:
794/794 with 36 ring-room waits vs 648/794 with 0), and filler 1 consistently stages ~150 fewer frames at
every size. One mover drains slower than the other, so **the required ring size is set by the worse
lane** -- evening the two out would buy more headroom than any sizing change.

### Measurement gotchas from this session

- **`grep | tail -1` over a 4-DRISC log is a trap.** The reference high-water was first recorded as 646
  frames; that was filler 1's MOVER line. The true peak is filler 0 at 897 (9.03 MiB). The whole "6.4 MB
  should be enough" premise came from that mis-grep -- 6/7/8 MiB are all BELOW the natural burst, which
  is exactly why all three fill and stall. Always print one line per DRISC.
- **Stalls are bimodal even below the knee**: 500 iters rep 3 gave 49 stalls where reps 1-2 gave 0, and
  10 MiB rep 3 gave 45 where reps 1-2 gave 0. A single clean run proves nothing; 3 reps minimum.
- **`local` expands all its words before assigning any of them**, so
  `local M=$1 TAG=$2 L=/tmp/x_${M}_${TAG}.log` reads M/TAG while still unset and aborts under `set -u`.
  Cost one dead harness run.
- Sweep B ran without tracy-capture attached (24 runs). Justified by the 50k pair showing the sink barely
  moves producer stalls (152,196 without Tracy vs 153,924 with), but the knee at 2000 iters has NOT been
  re-verified with Tracy attached.

## §N+40 — 4 FILLERS + 2 DUAL-RING MOVERS: the knee moves 60 -> 15 (4x), and `max batch` does NOT halve (bh-26, 2026-08-11)

The role split now runs **six** DRISCs: 4 fillers at 30 worker cores each, and 2 movers each draining TWO
DRAM frame rings into its own socket. Every number below is 120 cores, slow dispatch, `--iters 500`
(5,000 zones/RISC, 6.0 M markers), full decode + publish, 4 warm reps per point. No JIT-cold run is in any
table -- the cache was warm throughout, verified by rep 0 matching reps 1-3 at every delay (so the usual
"discard run 1" did not bite here; it still applies to a cold key).

### The knee: delay 60 -> 15

Producer stalls from the L1 counters, one row per delay, all 4 reps shown because stalls are bimodal:

| delay | 2 fillers x 60 cores | 4 fillers x 30 cores |
|---|---|---|
| 5 | -- | 10,704 / 11,118 / 11,302 / 10,719 |
| 8 | -- | 2,210 / 2,149 / 2,148 / 2,245 |
| 10 | -- | 701 / 778 / 683 / 797 |
| 12 | -- | 163 / 136 / 25 / 66 |
| 13 | -- | 0 / 0 / **6** / 0 |
| **15** | -- | **0 / 0 / 0 / 0** |
| 20 | 18,553 / 18,305 / 18,458 / 18,512 | **0 / 0 / 0 / 0** |
| 25 | 15,609 / 15,168 / 15,254 / 14,979 | **0 / 0 / 0 / 0** |
| 40 | 9 / 2 / 0 / 0 | **0 / 0 / 0 / 0** |
| **60** | **0 / 0 / 0 / 0** | 0 / 0 / 0 / 0 |

Knee, defined as the first delay clean in 4/4 reps: **60 -> 15, a 4.0x improvement.** On the looser
"first delay that is ever clean" definition it is 40 -> 13, 3.1x. The delay-20 and delay-25 points are the
sharpest statement: 18,500 and 15,200 stalls become **exactly zero**, three and four sweep points below
where the 2-filler configuration first stops stalling.

Control, same binary, role split OFF (2 full-job drainers) at delay 40: **9,604 stalls across 120/120
cores.** So the ladder at delay 40 is 9,604 (no split) -> 0-9 (2 fillers) -> 0 (4 fillers).

### Why: the scan halved, exactly as the model predicted

The knee is the FILLER's SCAN over its slice (§N+28), and halving the slice halved every filler cost. Per
sweep at delay 20:

| | 2 fillers x 60 | 4 fillers x 30 | ratio |
|---|---|---|---|
| FILLER idle sweep | 16.7 us | 8.2-8.5 us | 2.0x |
| FILLER busy sweep | 27.8-28.0 us | 13.1-14.0 us | 2.0x |
| FILLER worst sweep | 39.0-40.3 us | 17.2-19.7 us | 2.1x |
| of which `proc` (the scan) | 15.0-15.3 us | **7.5 us** | 2.0x |
| of which `read` | 3.8 us | 2.1 us | 1.8x |
| FILLER worst credit-wait | 0.1 us | 0.1 us | -- |

Clean 2.0x on the thing being halved. Note the knee moved **4x** on a **2x** cost reduction -- the same
super-linear behaviour §N+34 saw going 1 -> 2 drainers, and for the same reason: below the knee a filler
falls into a feedback loop (producers stall -> rings pin -> spans arrive fuller but later -> sweeps cost
more) that a faster filler stays out of entirely.

### `max batch` stays 7 per peer -- the main a-priori risk did NOT materialise

The stated risk was that a mover's 7 staging slots split across two rings would drop `max batch` to 3-4 and
raise per-frame egress overhead exactly where the credit-wait knee lives. It does not happen, because the
peers are visited **sequentially, each with the whole staging area**, separated by the write barrier that
staging reuse already required. **`max batch 7` on all four peers, in all 4 sweep points and all 25
stability runs.** Splitting the slots would have bought nothing anyway: both peers push into ONE socket, so
their egress could never have overlapped.

Push counts confirm it. Per mover at delay 40: **608 / 623** pushes with one ring each, **656 / 659** with
two rings each -- +7% for twice the rings, not +100%.

### What it cost

| | 2 fillers | 4 fillers | delta |
|---|---|---|---|
| MOVER idle sweep | 0.3-0.4 us | 0.6-0.8 us | 2x (two head reads/sweep) |
| MOVER busy sweep | 5.2-5.5 us x ~515 | 13.5-13.6 us x ~330 | total busy 2.8 ms -> 4.5 ms |
| MOVER worst sweep | 33.8-37.5 us | 51.9-52.9 us | +40% |
| host bytes @ delay 40 | 71.9 MB | 79.3 MB | **+10.2%** |
| host bytes @ delay 60 | 79.5 MB | 81.1 MB | **+2.0%** |
| DRAM reserved | 448 MiB | **448 MiB** | **0** |

A mover is still ~97% idle after doubling: 191,357 idle sweeps x 0.8 us = 153 ms of a 161 ms wall. Its worst
sweep is 52 us of which 28-34 us is socket credit-wait -- i.e. the mover's worst case is still the HOST, not
the second ring.

**Egress bytes went UP and the knee still improved 4x.** That is direct confirmation that the knee is not
egress-bound: 4 fillers ship 2-10% more bytes for the same 6.0 M markers (quarter-grid sweeps complete
sooner, so spans come back marginally less full) and stalls still went 18,500 -> 0. Compare only clean runs
here -- the delay-20 frame counts look like +40% but that is the 2-filler baseline being stalled, and a
stalled producer yields fewer, fuller spans.

**The four rings are free.** The rings live in the HAL per-bank DRAM PROFILER region (`a0eef213134`), which
is reserved at the same offset in all 7 allocator banks whether 1 or 7 rings sit in it. Going 2 -> 4 rings
moved 128 MiB of the 448 MiB reservation from "region no ring uses" to "region carrying a ring". Three banks
are still unused, so rings 5-7 would also be free.

### Stability: 25/25 clean

25 consecutive runs at the reference config (delay 40, 120 cores, slow dispatch, 6 DRISCs, 4 rings):

- **0 producer stalls** in all 25
- **`frames staged == frames moved` per ring** in all 25 x 4 rings (e.g. 1884/1884, 1860/1860, 1899/1899, 1863/1863)
- **`hs_bad` 0**, `ring-room waits` 0, `credit timeouts` 0, no `FAILED TO START`, no failed handshake, no undrained ring at teardown
- host record ring **dropped 0**; **regressions 0 at publish and 0 at consume** in all 25
- records **exactly 6,001,200** in all 25 (the nominal count -- any stall would add markers)
- `max batch 7` on all 4 peers in all 25
- per-ring high-water **433-730 of 6,355 frames** (328-770 across the whole delay sweep), against **73-1,266** for the 2-ring configuration

Fast dispatch (110 cores, non-divisible by 4) also checked: slices come out 27/28/27/28 and the run is
clean. The role-split-OFF path is untouched -- still 2 drainers over `[0,60)`/`[60,120)`.

### Refutations and things deliberately given up

- **REFUTED: "a dual-ring mover halves its batch."** 7 per peer, measured (see above). The barrier that
  makes sequential peers safe was already there for staging reuse, so it is not even a new cost.
- **REFUTED: "the knee needs more egress."** More bytes, better knee.
- **GIVEN UP: rings on channels no drainer occupies.** 6 drainer channels + 4 rings = 10 against 7 allocator
  banks, so the old insurance is unreachable. Rings now sit on banks 1, 2, 4, 5 -- three of which also host a
  filler. Kept: no ring on a MOVER bank (0, 3), because host-facing duty is where §N+29's hazard was
  measured. 25 runs with the overlap in place show no penalty (0 ring-room waits, 0 hs_bad, staged == moved).
  A ring terminates at its channel's PREFERRED WORKER endpoint while a drainer sits on the unused
  subchannel, so a shared channel still means a different core and a different NIU.
- **NOT RE-MEASURED at 4 rings:** §N+39's ring-sizing table (12 MiB smallest stall-free at 5k zones/RISC) and
  the 20k zones/RISC volume knee. Per-ring high-water roughly halved, which *suggests* a smaller per-ring
  minimum, but that is inference. Re-run Sweep A before lowering `ROLE_RING_MB` here.
- **NOT re-verified with tracy-capture attached.** All 4-filler runs above are decode+publish to the
  BroadcastRing with no Tracy sink attached, the same regime §N+39's Sweep B used.

### Two hazards found and closed while extending the roster

Neither was hit on silicon -- both are failure modes the 2-DRISC roster could not express, found by reading
what the new one makes possible.

1. **Two DRISCs on one core.** `pick_unused_dram_logical_core()` takes a DRAM VIEW and reserves that view's
   worker/eth endpoints; it has no idea another view may resolve to the SAME physical port. §N+29's own
   table records views 0 and 7 both as NoC core `0-0`. At two DRISCs the roster was hardcoded `{0, 3}` and
   this could not arise; with six banks in play and a filler-bank env override it can, and the result is two
   resident kernels overlapping one L1's staging, socket config, results and handshake -- with no counter
   that would notice. `boot_device` now `TT_FATAL`s on duplicate cores. (This is also why filler 3 sits on
   view 1 and not view 7.)
2. **One magic for four fillers proves nothing.** The bring-up probe planted a single
   `kProbeFillerMagic` in every filler, so an echo only proved the mover read SOME filler's probe word -- a
   mover whose peer-1 coordinate named the wrong filler would have passed, and then one ring would be drained
   twice while another was never drained at all, back-pressuring a lossless producer into wedging the
   workload. Magics are now per-peer: the host plants `kProbeFillerMagic + <filler index>` and checks the
   echo against the index it MEANT, and the mover writes back `kProbeMoverMagic + <peer slot>`.

### Implementation notes worth keeping

- **Per-peer state must be strictly separate**: `mv_tail`, `mv_moved`, `mv_max_n`, `ring_hi`, the probe
  words and the live head/tail telemetry are all `[2]` arrays. One shared `mv_tail` across two rings would
  ack frames on ring A that were only read from ring B -- handing a filler room it does not have, which is
  the same silent-corruption shape as the `hs_bad` bug.
- **The ordered quiesce is now per PEER RING, not per mover.** Phase 2 waits for `tail >= head` on each of a
  mover's rings; a mover caught up on ring 0 can still owe hundreds of frames on ring 1, and that loss is
  invisible in every host counter because the records were simply never sent.
- **The ring's bank-relative address is per-filler now.** It used to be one compile arg guarded by a
  `TT_FATAL` demanding every ring bank have the same `get_bank_offset` -- fine at 2 rings on a part where
  every offset is 0, but at 4 rings that FATAL would kill capture on any part where they differ, for no
  reason: each filler already carries its own `(bank, addr)` and a mover gets its peers' pairs explicitly.
- **The slice denominator is no longer `kNSockets`.** Fillers divide the grid `kNFillers` ways while sockets
  stay at 2. Getting that wrong yields a build that works perfectly and simply does not improve.
- The results block is now FULL: `out[0..63]`, with peer 1's counters at `out[58..63]`. A third ring per
  mover would need it widened (and `kMiscBytes`/`hs_addr` re-laid out with it).
- Per-DRISC L1 telemetry behind `done` is now 13 words (52 B of the 64 B pad): 5 shared + 4 per peer.

### Measurement gotchas

- **One log line PER RING, never per DRISC.** A dual-ring mover with one healthy ring and one short one has
  to be visible at a glance, so the mover's role-split line is printed once per peer slot and names the
  filler. §N+39 already lost a baseline to a `grep | tail -1` over a 4-DRISC log; at 6 DRISCs and 8 ring
  lines that is worse, not better.
- **`grep -o 'L1 STALL COUNTERS -- [0-9]* producer stalls' | grep -o '[0-9]*' | head -1` returns 1** -- the
  "1" in "L1". Every stall figure in the first pass of this session read `stalls=1`. Parse with `sed -n
  's/...\([0-9]*\) producer stalls.../\1/p'` instead.

## §N+41 — DRISC SELF-PROFILING: the drainer frames its own zones as a worker span, and the sampler has to trigger on WORK, not on sweep number (bh-26, 2026-08-12)

The drainers were the only cores in the system with no row in Tracy. They are now producers of the same wire
format everything else uses: **a DRISC emits its own zones into its own SPSC ring and frames that ring into the
DRAM/socket path it already owns**, so the host decoder, the frame layout and the Tracy handler are all
untouched. Gated behind `TT_METAL_PERF_DEBUG_DRISC_ZONES=N` (default OFF).

Two things dominated the work, and neither was the framing:

1. **A sweep-number sampler captures nothing.** Both roles are >99% idle, and the phases only mean anything on
   a busy sweep. The trigger has to be *discovered work*.
2. **The instrumentation's presence costs more than its output.** Getting the emitter calls off the hot path
   mattered 6x more than anything about the frames.

### What it looks like

Six new Tracy contexts (one per drainer core, at its own NoC coords), each carrying a two-level tree:
`DRISC-SWEEP` at depth 0 with `DRISC-READ` / `DRISC-READ-WAIT` / `DRISC-PROC` / `DRISC-WR-BARRIER` as children
and `DRISC-CREDIT-WAIT` / `DRISC-WRITE` nested inside `DRISC-PROC` on a filler (directly under `DRISC-SWEEP` on
a mover, which has no proc phase).

Measured from `tracy_captures/drisc_selfzones.tracy` (`tracy_zone_csv`: **126 contexts** = 120 workers + 6
drainers, 3,001,949 zone rows), every captured sweep work-triggered. Per-zone means, one row per drainer:

| DRISC | role | ctx | SWEEP zones | DRISC zones | depths | SWEEP mean | READ | READ-WAIT | PROC | CREDIT-WAIT | WRITE | WR-BARRIER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | filler | 0 |  5 | 251 | 0:5, 1:192, 2:54 | 16.80 us | 128 ns | 67 ns | 849 ns | 54 ns | 138 ns |  81 ns |
| 1 | filler | 1 |  5 | 241 | 0:5, 1:184, 2:52 | 16.71 us | 128 ns | 68 ns | 856 ns | 54 ns | 140 ns | 103 ns |
| 2 | filler | 2 |  5 | 247 | 0:5, 1:192, 2:50 | 16.39 us | 128 ns | 67 ns | 825 ns | 54 ns | 138 ns |  64 ns |
| 3 | filler | 3 |  5 | 241 | 0:5, 1:184, 2:52 | 16.95 us | 128 ns | 69 ns | 864 ns | 54 ns | 140 ns | 120 ns |
| 4 | mover  | 4 | 15 | 123 | 0:15, 1:108      | 11.95 us | 1,774 ns | - | - | **2,698 ns** | 1,065 ns | 359 ns |
| 5 | mover  | 5 | 18 | 150 | 0:18, 1:132      | 11.47 us | 1,425 ns | - | - | **2,781 ns** |   831 ns | 525 ns |

The mover rows are the point of the exercise. §N+38 established that the knee is the WORST sweep's credit wait;
these are the first per-occurrence numbers for it, and **credit-wait is the largest single phase on a mover by
a factor of ~1.6** (2.70-2.78 us mean, against a 1,425-1,774 ns read and an 831-1,065 ns write). A filler's
zones say the opposite thing about itself: its 825-864 ns PROC per batch dwarfs everything, and its credit-wait
(DRAM ring room) is 54 ns -- three orders of magnitude apart on the same named phase, in one capture.

**Zone counts match the device's own capture accounting exactly on all six drainers** (5/5/5/5/15/18 SWEEP zones
against 5/5/5/5/15/18 sweeps reported captured), which is what ties the Tracy rows to the counters.

### Mechanism: the frame is a worker span, and nothing else changed

- The self frame is **the ordinary `PP_BULK_SPAN` layout**: 16-word prefix, a 64-word control vector with the
  drainer's own `SPSC_CORE_XY`, five 512-word rings. `myRiscID == PROCESSOR_INDEX == 0` on a DRISC, so **only
  ring 0 is ever live** and the host's per-RISC walk yields nothing from rings 1-4 (`tail == head == 0`).
- It lives in **staging slot `kNStage`**, one past every slot the drain pipeline can touch. There was no room to
  add one: a DRAM core's UNRESERVED L1 is 86,448 B and 7 slots x 10,560 B plus the scratch/misc/socket-config
  reserve leave 1,792 B. So the kernel is handed `nstage - 1` slots instead. **L1 does not grow**; the only
  behavioural cost is a mover's largest batch dropping 7 -> 6 (a filler's is bounded by `kGenSlots = 3` either
  way, so it is unaffected).
- **A filler self-frames into its own DRAM ring; a mover pushes into its socket.** Both are literally
  `emit_run(kSelfSlot, 1)`, i.e. the same call the payload takes, so single-producer-per-ring is preserved and
  no mover ever writes into a peer's ring.
- **Markers are stamped from timestamps the drain loop had already read** (`t_batch0`/`t_issue`, `stage_run`'s
  `t0`/`t1`/`t2`, the barrier's `t_b0`). This adds no clock reads, and it makes a zone's duration *the same
  quantity the `out[]` phase counter accumulates* -- so zones and counters agree by construction and a
  disagreement can only be framing or decode. It is also what allows retroactive arming (below).
- **The clock needed nothing.** `get_timestamp()` reads `RISCV_DEBUG_REG_WALL_CLOCK_L/H`, the same register the
  workers use, so DRISC zones land on the worker timeline with no anchor, no high-bit graft, no calibration.
  Nothing X280-shaped was needed or added.
- Zone ids are FIXED (`0x7FF0..0x7FF6`, the band `PROFILER_STALL_ZONE_ID = 0x7FFF` already uses) and named
  host-side next to PRODUCER-STALL, because these zones are not scoped by the `DeviceZoneScopedN` macros and so
  have no `#pragma message` source location for `generateZoneSourceLocationsHashes()` to harvest.

### ALIGNMENT: idle samples make the capture unreadable, and they feed the movers phantom work

Reported from the GUI, and correct: *"the activity on the DRISC is not aligned with the activity on the cores,
they are all over the place, coming in both before and after."* Two causes, and **neither is the clock** -- the
fillers' work zones landed inside the worker window all along, which is itself the proof that a DRAM core's
wall clock agrees with the Tensix ones.

| | worker zones | DRISC zone window | DRISC span |
|---|---|---|---|
| with idle sampling (as first built) | 1.926 ms | starts **-187 ms** to -150 ms before the workload, ends +2.9 ms after | 151-192 ms |
| work-triggered only (now the default) | 1.931 ms | starts **+0.008 to +0.067 ms** after the first worker zone | 1.69-4.81 ms |

1. **The drainer is resident from device open, so its idle sweeps span the whole process (~190 ms) while the
   workload is a 1.9 ms sliver of it.** Sampling them scatters DRISC zones across a window 100x the
   workload's. The first zone in the capture was an idle-sample sweep 187 ms before any worker zone existed.
2. **The instrument feeds itself.** A FILLER's self frame is a real frame in its DRAM ring, so its MOVER then
   ships it -- manufacturing mover credit-wait and write zones at instants when no worker produced anything.
   Proven, not inferred: **all 13 pre-workload mover credit-wait zones followed a peer filler's captured-sweep
   publish by 1.7-2.4 us** (median 2.0 us on mover 4, 2.1 us on mover 5), with the peer mapping matching
   `peer_of` exactly. It also BIASED the mover's phase profile -- a self-frame push has no credit wait, so
   mixing those in reported credit-wait as 1.6-1.8 us where the work-only figure is **2.70-2.78 us**.

So idle sampling is now **OFF by default** (`TT_METAL_PERF_DEBUG_DRISC_ZONE_IDLE`), every capture is
work-triggered, and the result lines up: all six drainers start within 8-67 us of the first worker zone, the
four fillers end 148-202 us BEFORE the last one (they stop staging once the producers are drained), and the two
movers extend **+2.5 and +2.9 ms past it** -- which is not misalignment but the DRAM ring's drain tail, the
thing you would want to see. An idle drainer's poll cost is still visible WITHIN a captured sweep: a mover's
empty-peer READ zone and a filler's per-batch zones over cores with nothing live ARE the idle cost, in context.

### The sampler: work-triggered, retroactively armed, rewindable  **[SUPERSEDED -- see "Full tracing" below]**

A "sample every Nth sweep" rule was written first and is **refuted by the sweep distribution**: a filler moves
frames in ~114 of ~25,000 sweeps (0.5%) and a mover in ~350 of ~230,000 (0.15%). Measured with that rule:

| rule | filler captures | of which DID WORK | mover captures | of which DID WORK |
|---|---|---|---|---|
| every Nth sweep + follows-busy, shared budget | 55 | **1** | 64 | **11** |
| work-armed, split budget, tight work spacing | 8 | 5 | 30 | 21 |

Three separate defects produced that first row, each of which looked healthy in the summary counts:

1. **Uniform samples ate the budget.** A mover runs ~550,000 sweeps, so even 1-in-6,400 is ~85 captures; they
   took 53 of a 64-frame budget and left 11 for the bursts. Idle samples were first given their own eighth of
   the budget; they are now OFF by default outright (see ALIGNMENT above), which is the real fix.
2. **An idle filler sweep could not be rewound.** It still walks 40 batches and emits ~320 markers = ~640 words,
   overflowing the 512-word ring, so it published a frame mid-sweep and became unrewindable. The ring-full path
   on a no-work sweep now **abandons** it (one assignment, nothing shipped).
3. **One shared rate limiter starved work captures.** A drainer is resident for the whole process but the
   workload occupies a short window of it, so a filler's ~114 busy sweeps are packed into a few hundred
   consecutive ones -- a 200-sweep limiter allowed exactly one. Work spacing is now `N/8`, idle spacing `N*32`.

The arm is **retroactive**, which only works because markers carry explicit timestamps: `self_arm` emits
`DRISC-SWEEP`'s START with the sweep's own `t_sweep0` long after that instant passed. A mover arms *before*
issuing its DRAM read (`n != 0` is known first), so a busy mover visit is captured whole. A filler arms at the
end of the first batch that found live cores, so **the batches of that sweep before the first live core are not
recovered** -- their timestamps are gone. Every later batch of the sweep is complete. Stated, not hidden.

### The cost is the CALL SITES, not the frames — and it was worth 6x

The emitters are too large for `-Os` to inline, so putting the `self_on` check *inside* them still costs a real
call at every site, and one of those sites is in the scan loop. Measured, feature ON, nothing being captured:

| | idle sweep | busy sweep | worst sweep | worst-sweep proc | stalls @ delay 15 |
|---|---|---|---|---|---|
| OFF | 8.38 us | 13.28 us | 18.57 us | 7.53 us | **0** (4/4 reps) |
| ON, checks inside the emitters | 10.2 us | 16.0 us | 22.6 us | 10.5 us | 292-362 |
| ON, checks at the call sites | 8.72 us | 13.94 us | 21.03 us | 10.26 us | 49-140 |

Two further changes were needed on top of the guards:

- **Nothing may go inside the scan.** The scan is register-pressure bound (see its comment: hoisting the head
  mirror into scalars is what made it fast), so a call in there spills `m0..m4`. The filler's work-arm moved out
  to the end of `process_batch`, keyed on `frames != frames_at_p0` -- `frames` already advances once per live
  core, so asking "did this batch find work" costs nothing in the loop.
- **One ring per captured sweep, hard.** A captured busy filler sweep wants ~480 markers (~960 words) and so
  needed a mid-sweep publish; the marker writes plus that publish put the worst sweep at 23.2 us. Past one ring
  the sweep is now **truncated** (counted, excluded from the counter cross-check, `DRISC-SWEEP` still closed so
  no lane's Tracy stack is left open).

  **RETRACTED: this bound was credited with taking 292-362 stalls down to 49-140. Not supported.** A
  re-check of every ON run found truncation NEVER FIRED -- 0 runs report it. A filler slice is 27-30
  cores = 9-10 batches = ~100 markers = ~200 words, comfortably inside the 504-word ring; the "~480
  markers, 40 batches" figure behind the claim was the UN-SPLIT 120-core drainer, not the 4-filler
  split. The measured win was moving the `self_on` guards to the call sites (idle sweep 10.2 -> 8.72 us).
  The 300 -> 100 stall change across those runs is UNEXPLAINED, and most likely the same bimodal
  run-to-run variance documented below. The bound stays as cheap insurance for wider slices.

### Perturbation, 4 warm reps each (run 1 of every config discarded)

| | filler idle | filler busy | filler worst | filler worst proc | mover idle | mover busy | mover worst | mover worst credit-wait |
|---|---|---|---|---|---|---|---|---|
| OFF | 8.38 [8.1-8.5] | 13.28 [12.7-13.6] | 18.57 [17.0-19.6] | 7.53 [7.5-7.9] | 0.75 | 13.79 | 51.0 [41.6-62.9] | 32.5 [22.5-40.7] |
| ON  | 8.72 [8.5-8.8] | 13.94 [13.4-14.2] | 21.03 [19.9-22.1] | 10.26 [8.3-10.5] | 0.75 | 12.84 | 40.9 [35.3-48.5] | 18.9 [16.8-21.1] |

- **+4% on a filler's idle and busy sweep, +13% on its worst sweep.** The worst-sweep proc figure (7.53 ->
  10.26 us) is the captured sweeps themselves: a captured sweep IS the worst sweep, so this is a measurement
  bias, not a background cost. The mover columns are inside run-to-run noise (its worst sweep is credit-wait
  dominated and spans 41.6-62.9 us with the feature off).
- **`out[]` phases keep their exact meaning with the feature on.** `self_publish` saves and restores
  `c_reserve`/`c_write`/`c_wr_*`/`pages`/`pushes`/`max_reserve`, and `c_self` joins the nested term
  `process_batch` subtracts, so the self frame's egress is billed to `c_self` alone. It shows up as
  *unaccounted* (89% -> 71% of the worst sweep is accounted), which is the honest place for it.
- **Cost in bytes: 0.23-1.00% of a drainer's egress** (4-35 frames x 10,560 B against 17-36 MB). The 4/5 waste
  from shipping five rings to carry one is therefore not worth removing. A 592-word frame (16 prefix + 64 ctrl +
  512 ring = 2,368 B, exactly 37 socket pages) **would** decode correctly -- the decoder derives the frame
  length from `payload_words` and only touches ring `r` when `run > 0` -- but the DRAM ring is indexed in whole
  slots, so a filler cannot use it without leaving the slot's tail on the wire as garbage. Not shipped.
- **At delay 60, off the knee, ON costs 0 stalls (4/4 reps), same as OFF.** The stalls at delay 15 are a
  knee-crossing, not a general cost: that config sits at 472-488 of 512 ring words, where one late sweep blocks
  dozens of lossless producers at once.

### FAST DISPATCH, delay 60: clean, and the mover's credit wait is BIMODAL (not a raised floor)

Self-profiling was built and tuned against the slow-dispatch reference, so it was re-run under fast dispatch.
The role split works there unchanged -- the drain kernels are launched with `force_slow_dispatch=true` onto DRAM
cores, which touch none of the fast-dispatch worker grid or dispatch column, so a resident drainer is
independent of dispatch mode.

**RUN RECIPE: fast dispatch needs `--gx 11 --gy 10`, not 12.** `compute_with_storage_grid_size()` returns 11x10
under FD because dispatch reserves the last column itself, so the poll list is **110 cores** (fillers get
27/28/27/28) against 120 under slow dispatch. A producer outside the poll list is undrained, and the producers
are lossless, so `--gx 12` there wedges the workload -- the same failure class as §N+33's `--gx 0`.

4 warm reps each, run 1 discarded at both configs, 110 cores, delay 60, `--iters 500`:

| | producer stalls | records (publish) | consume | dropped | ts regressions |
|---|---|---|---|---|---|
| OFF | **0** (4/4) | 5,501,100 exactly, all 4 reps | == publish | 0 | 0 / 0 |
| ON  | **0** (4/4) | 5,502,618 - 5,502,774 | **== publish, all 4 reps** | 0 | 0 / 0 |

- **0 producer stalls with the feature ON**, which the slow-dispatch reference at delay 15 could not manage
  (49-140 there). Consistent with the delay-60 slow-dispatch result: the stalls are a knee crossing, not a
  general cost of the instrument.
- **`publish == consume` exactly in every ON rep.** The ~1,200-record tail truncation recorded under slow
  dispatch (drainers still publishing self frames after the consumer is told to stop) **does not reproduce
  here**, so that caveat is slow-dispatch-specific rather than inherent.
- Every captured sweep DID WORK on all six drainers (3/3 per filler, 13/13 and 14/14 per mover), 0 frames to
  idle samples. Cost **0.15-0.34% of a drainer's egress** -- lower than slow dispatch, because FD moves ~20%
  more bytes per drainer (20.0-20.5 MB per filler, 40.2-41.1 MB per mover).

Perturbation, and the interesting part is which role moves:

| | filler idle | filler busy | filler worst | mover idle | mover busy | mover worst | mover worst credit-wait |
|---|---|---|---|---|---|---|---|
| OFF | 7.80 [7.6-8.2] | 16.46 [15.0-17.6] | 17.64 [16.0-19.7] | 0.75 | 20.35 [17.1-23.2] | 83.88 [50.5-106.0] | 53.00 [26.9-74.9] |
| ON  | 8.12 [7.9-8.5] | 17.38 [15.9-18.6] | 20.35 [19.2-21.7] | 0.75 | 16.34 [13.7-20.2] | 57.14 [37.1-118.6] | 31.51 [16.4-65.9] |

Filler cost is the same +4% idle / +5.6% busy / +15% worst-sweep shape as slow dispatch. The mover columns are
NOISE, not improvement: its worst sweep spans 50.5-106.0 us with the feature off and 37.1-118.6 with it on.

**What the zones say that the counters did not: the mover's credit wait is BIMODAL.** The per-sweep means
below were the first read of this, and the framing they suggested -- "fast dispatch makes the mover more
credit-bound" -- is WRONG, because a mean over two separate populations describes neither. A later trace
(`tracy_captures/drisc_selfzones_fd60.tracy`, same config) gives the per-occurrence distribution:

| | n | p50 | under 1 us | >= 5 us | max | mean |
|---|---|---|---|---|---|---|
| FILLER (DRAM ring room) | 27 | 52 ns | **100%** | 0% | 75 ns | 54 ns |
| MOVER 4 (host FIFO credit) | 25 | **66 ns** | 52% | **40%** | 9,324 ns | 3,568 ns |
| MOVER 5 (host FIFO credit) | 28 | **67 ns** | 54% | **29%** | 14,296 ns | 2,912 ns |

**The median push waits ~66 ns -- essentially free -- and roughly a third of them cost 9-14 us, with nothing
in between.** So the socket does not run a higher credit floor under fast dispatch; it momentarily runs OUT of
credit a fraction of the time, and each such event costs ~9-14 us. That is what makes a worst sweep 50-106 us
against a ~16 us mean busy sweep: a worst sweep is one that caught one or more tail events. Optimising a
"higher floor" would have been chasing something that does not exist -- the target is the frequency and depth
of the credit-exhaustion events.

The FILLER row is the control, and it is the §N+39 prediction landing: DRAM ring room is dead flat
(51-75 ns, 100% under 1 us), so 64 MiB of runway never binds. Same phase name, two different quantities,
~50x apart.

Also measured from that trace: filler PROC rose to **1,223-1,274 ns/batch** under FD against 825-864 ns under
SD, because producers burst harder so more cores are live per batch. FD therefore costs the filler in SCAN and
the mover in CREDIT-WAIT TAIL -- different phases, one cause.

The per-busy-sweep means that produced the original (wrong) framing, kept for the record:

| | mover credit-wait per busy sweep | mover read | mover write |
|---|---|---|---|
| slow dispatch, delay 15 | 4.7 us | 3.5 us | 1.8 us |
| **fast dispatch, delay 60** | **7.7 us (mover 4) / 12.7 us (mover 5)** | 4.0 / 3.3 us | 3.8 / 2.7 us |

and the OFF-path worst-sweep credit wait agrees independently (53.0 us mean under FD against 32.5 us under SD
at the tighter delay). Fast dispatch removes the host round trip between ops, so producers fill their rings in
harder bursts; the drainer's egress is unchanged, so the extra burstiness lands on the socket credit wait --
the one phase §N+38 identified as setting the knee.

**DO NOT QUOTE MOVER CREDIT-WAIT MEANS, INCLUDING THE ONES IN THE TABLE ABOVE.** With n ~= 30 captured pushes
and a third of them ~100x the median, the mean tracks how many tail events the sampler happened to catch. Two
slow-dispatch captures of the SAME config, compared directly:

| | mover 4 | mover 5 |
|---|---|---|
| capture A | p50 66 ns, mean 2,659 ns, 30% >= 5 us | p50 66 ns, mean **783** ns, **6%** >= 5 us |
| capture B | p50 66 ns, mean 2,698 ns, 27% >= 5 us | p50 66 ns, mean **2,781** ns, **28%** >= 5 us |

The p50 is 66 ns in all four. The MEAN on one core moved 3.6x between runs, and its tail fraction moved
6% -> 28%. So an earlier claim here that "the two movers differ by 1.6x from each other" is WITHDRAWN -- that
gap is inside run-to-run noise; mover 5 alone spans 783-2,781 ns across two runs of one config.

**Quote p50 (stable at 66-67 ns across every capture), the tail threshold and fraction with n stated, and the
max. Never the mean.** Raising the sample count would tighten the tail fraction but is NOT free -- see the
coverage/bias note below.

### Cross-check: the zones agree with the counters

The device accumulates the same five phase totals over **exactly the sweeps the zones cover** (`out[74..84]`,
restricted to sweeps instrumented from the top -- a retroactively-armed or truncated sweep has zones for only
part of itself and is excluded). DRISC 1 is the clean case, where all 4 captured sweeps are fully instrumented,
so the CSV totals and the counters cover the same set (from the idle-sampling configuration, which is what
produces a fully-instrumented-only drainer; still reachable with `TT_METAL_PERF_DEBUG_DRISC_ZONE_IDLE=1`):

| phase | from the Tracy CSV | device counter | delta |
|---|---|---|---|
| read (`sum(READ) + sum(READ-WAIT)`) | 8.014 us | 8.0 us | +0.2% |
| proc (`sum(PROC) - sum(CREDIT-WAIT) - sum(WRITE)`) | 16.180 us | 16.2 us | -0.1% |
| credit-wait | 0 | 0.0 us | exact |
| write | 0 | 0.0 us | exact |
| wr-barrier | 2.080 us | 2.1 us | -1.0% |

Agreement is within the log's print rounding (one decimal) plus the difference between the hardcoded
`kCycPerUs = 1.35e3` in the log line and the measured aiclk Tracy uses (~0.6%). On the other five drainers the
CSV covers 1-14 more sweeps than the counters and is larger by the right amount in every phase (e.g. DRISC 4:
CSV credit-wait 60.7 us vs counter 60.2 us over 22 of 30 sweeps -- the 8 extra are idle samples, which have no
credit wait).

### Why the sampler was replaced: the sweeps you could see were the ones you perturbed  **[SUPERSEDED]**

Coverage, measured (`TT_METAL_PERF_DEBUG_DRISC_ZONES=200`):

| | captured sweeps | busy sweeps | total sweeps | coverage of BUSY |
|---|---|---|---|---|
| fillers 0-3 | 5 each | 113-116 | 167k-379k | **~4.4%** |
| mover 4 | 15 | 349 | 946k | **~4.3%** |
| mover 5 | 18 | 370 | 247k | **~4.9%** |

~4-5% of sweeps that did work, ~0.002% of all sweeps. **Every gap between `DRISC-SWEEP` zones is an uncaptured
sweep, not idle time.** The sampler is unbiased WITHIN busy sweeps (rate-limited by sweep number, not by cost),
so per-zone statistics are representative -- but you cannot reason about a specific burst from that timeline,
and the worst sweep is invisible unless it happens to be one of the few captured.

**The cost is not uniform, and that is the reason not to just raise coverage:**

| | cost |
|---|---|
| sweeps NOT captured | +4% idle (8.38 -> 8.72 us), +5% busy (13.28 -> 13.94 us) -- irreducible, the guards and register pressure of the code being present |
| sweeps BEING captured | **+13-15%** (worst sweep 18.57 -> 21.03 us): ~100 markers ~= 0.7 us plus one 10,560 B publish and its barrier ~= 1-2 us |
| at the knee (SD delay 15) | 49-140 producer stalls vs 0 -- one late sweep blocks dozens of lossless producers at 472-488/512 occupancy |
| off the knee (delay 60, SD and FD) | **0 stalls, 4/4 reps each** |

So a captured sweep is inflated ~15% while an uncaptured one is inflated ~5%: SELECTIVE sampling means the
sweeps you can see are perturbed differently from the ones you cannot. That is a measurement bias, not merely
overhead.

**CORRECTION to the conclusion first drawn from this.** It said raising coverage was the wrong lever because
more samples would mean more perturbed sweeps. That reasoned from a cost estimate built on the wrong window.
The sweep totals above (167k-946k) are dominated by sweeps OUTSIDE the workload: the drainer is resident from
device open to teardown while the workload is ~1.93 ms, and once fill reads 0 the pacing controller creeps its
gap to the ceiling, so it polls emptily for the rest of the process. Inside the active window the sweeps are
almost all BUSY. Reconciled against a filler's own counters:

| | busy sweeps | x sweep time | active window | frames | words | fill |
|---|---|---|---|---|---|---|
| filler 0 | 115 | 13.6-14.2 us | ~1.6 ms (~80% of the 1.93 ms worker window) | 1,756 | 3,000,974 | 1,709/2,560 = **66.8%** |

and `1,756 / 115` = **15.3 frames per busy sweep** -- one sweep walks all 27-30 cores and ships a frame for
every core with live data, which is why there are ~115 busy sweeps and not ~1,200. (5,000 zones x 30 cores x 5
RISCs x 2 words = 3,000,000 predicted words: exact. 66.8% fill against the controller's 70% target: within 3
points, so the controller is landing.)

Full coverage of the active window is therefore ~150 filler sweeps and ~500 mover sweeps -- **~630 KB (3.6% of
a filler's egress) and ~250 KB (0.7% of a mover's)** -- not the tens of thousands the raw sweep counts imply.
Continuous coverage of the window is affordable, and it makes the bias UNIFORM rather than selective, which is
the ordinary profiler trade and is analysable. A biased 4% subset is not. The publish and short-frame
optimisations remain worth having, but as cost reductions, not as prerequisites.

### Full tracing replaced the sampler, and costs LESS

`TT_METAL_PERF_DEBUG_DRISC_ZONES=1` now traces **every** sweep in one contiguous window at sweep level
(detail 0); the child phases stay available behind `TT_METAL_PERF_DEBUG_DRISC_ZONE_DETAIL=1`. The
rate-limit / retroactive-arm / rewind / abandon / truncate machinery above is gone.

**It is cheaper than the sampler it replaced**, which is the counterintuitive part:

| at SD delay 15, 120 cores | producer stalls |
|---|---|
| sampled (`DRISC_ZONES=200`) | 49-171 across 29-47 cores |
| **full trace (`DRISC_ZONES=1`)** | **0 across 0 of 120** |
| OFF | 0 |

**The cost was never the markers, it was the per-sweep PUBLISH.** Detail 0 emits ~4 markers/sweep against
~100, and one frame now carries ~63 sweeps instead of one, so the 10,560 B publish amortises 63:1. The sampler
paid a publish per captured sweep, which is why capturing 4% of sweeps cost more than capturing all of them.
That also retires the measurement-bias problem rather than mitigating it: the window is now uniformly
instrumented, so its numbers are comparable to each other.

Measured, both dispatch modes (0 producer stalls, 0 ts regressions at publish and consume in both):

| | SWEEP | PACE | window covered | largest gap | cost of egress |
|---|---|---|---|---|---|
| SD d15 filler 0-3 | 12.1-13.6 us | 2.7-4.2 us | 96.1-97.5% | 0.33-1.14 us | 0.11-0.23% |
| SD d15 mover 4-5 | 5.1-6.1 us | **none** | 96.1-96.8% | 0.94-1.14 us | 0.11-0.23% |
| FD d60 filler 0-3 | 14.2-16.5 us | **20.9-22.2 us** | **98.8%** | <= 0.33 us | 0.10-0.17% |
| FD d60 mover 4-5 | 7.6-7.7 us | **none** | 97.4% | -- | 0.10-0.17% |

**`DRISC-PACE` is 1:1 with `DRISC-SWEEP` on every filler and ABSENT on both movers** (117/138/142/139 of each
on the four fillers; 817 and 982 SWEEPs with zero PACE on the movers). The absence IS the measurement:
`if constexpr (kFillPct != 0 && kRole != kRoleMover)` excludes movers from the fill-ratio controller, so their
gap is always 0. With 96-99% of each window inside a zone there is no whitespace left to misread as idle.

**Under FAST DISPATCH a filler spends more time PACED than SWEEPING** -- 20.9-22.2 us of pace against a
14.2-16.5 us sweep, ~58% of its wall time, where slow dispatch is the reverse (~25%). Same knob, opposite
balance: FD delivers markers in harder bursts, spans come back fuller, so the fill-ratio controller widens the
gap. Consistent with filler PROC rising to 1,223-1,274 ns/batch under FD. The controller working as designed,
now visible as a colour block instead of inferred from a `pace-gap ... cyc` log line.

Zone colours are per-ROLE as well as per-zone (filler SWEEP blue / mover SWEEP teal, PACE recessive grey),
because the two roles share zone NAMES but not meanings -- one colour across both invites reading one row's
scale onto the other, and a filler's CREDIT-WAIT is 54 ns of DRAM ring room where a mover's is us-scale host
FIFO credit.

### The pacing controller's whole range, and what actually sets bytes-per-marker

Three traces of the SAME 3,000,000-word payload (5,000 zones x 30 cores x 5 RISCs x 2 words), 0 producer
stalls and 0 ts regressions in all three:

| config | SWEEP mean | PACE mean | sweep share of wall time | egress per filler | busy sweeps |
|---|---|---|---|---|---|
| **FD delay 15** | 12.1-13.3 us | **67-208 ns** | **98-99%** | **15.3-15.9 MB** | 55-59 |
| SD delay 15 | 12.1-13.6 us | 2.7-4.2 us | ~75% | 17.4-17.6 MB | 113 |
| **FD delay 60** | 14.2-16.5 us | **20.9-22.2 us** | **~42%** | **20.1-20.7 MB** | 74 |

The controller sweeps its entire range across these three points -- from effectively off (back-to-back sweeps
at FD/15, where spans already arrive at or above the 70% fill target so it correctly gets out of the way) to
holding the filler off more than half its wall time at FD/60. It is now readable as a blue/grey ratio in the
GUI instead of a `pace-gap N cyc` integer in a log line.

**Note the direction of the egress column: the config that paces MOST ships the MOST bytes.** That is not the
controller failing, it is §N+36's mechanism made visible. Bytes-per-marker is set by the PRODUCER RATE, because
a span ships whole (10,560 B) regardless of how much of it is live; slower producers hand over emptier spans.
The controller widening its gap NARROWS that penalty but cannot close it -- delay 60 still costs 31% more bytes
than delay 15 for identical payload. So pacing is a mitigation for slow producers, not a fix, and the fix would
have to be a short frame (see the 592-word analysis) rather than more waiting.

Also from the FD/15 trace: **mover `SWEEP` is bimodal at the sweep level** -- p50 731-844 ns, p90 16.7-18.2 us,
max 43.5-48.4 us. A 20x median-to-p90 and ~55x median-to-max spread: most peer visits are near-empty polls, a
minority are long, and the long ones are where the credit wait lives. Same shape as the FD/60 credit-wait
distribution (p50 95 ns, p90 ~9-10 us) but visible without descending to detail 1.

Two things not to misread in that capture: the FD/15 worker window is **0.988 ms** against 1.93 ms for SD/15
at the same 500 iterations -- fast dispatch does the same work in half the wall time, so the drainer rows are
denser, not busier. And the mover's largest coverage gap is **7.39 us** under FD against 0.94-1.14 us under SD;
that hole is inside the traced window and is the mover's inter-sweep bookkeeping after a sweep that found
nothing, not missing instrument.

### The tail flush could never fire, and every counter read clean

The post-loop flush was gated on `self_on`, which is cleared at the END of every sweep -- so the condition was
never true when it was tested. A filler wrote 945 words of its own trace and shipped **504**, silently losing
its last 55 sweeps. Nothing reported it: the frame counters, `staged == moved`, drops and regressions were all
clean, because the lost words never became frames. The only visible symptom was one drainer's Tracy row being
1.0 ms where its siblings were 2.4 ms -- and that was first misread as "fillers finish early once producers
are drained", which is also a real effect.

Fixed by flushing on `self_tail != self_head` (state, not intent), and `out[87]` now carries words-shipped so
the host warns on any shortfall. **Same failure class as §N+29's impossible-head read and the 4-DRISC
teardown garbage: a silent loss whose only tell was a number that looked plausible.** A counter that cannot
distinguish "nothing to send" from "failed to send" is not a check.

### No silent truncation

Every way the instrument can under-report is counted and printed: `self_frames`, sweeps captured, **of which
DID WORK**, frames spent on idle samples, sweeps rewound, sweeps abandoned, sweeps skipped after the budget,
markers dropped to truncation and the sweeps they came from, and `c_self`. There is an explicit **warning when
every captured sweep was idle** -- the failure mode the coordinator flagged, because on a mover that means the
credit wait that sets the knee is absent while every count still looks healthy. It fired for real on DRISC 1.

### Two host bugs the extra cores exposed

- **`L1 STALL COUNTERS` read the DRAM cores.** `core_virt` now also holds the drainers, and reading the TENSIX
  profiler address on a DRAM core returns whatever is at that offset in DRISC L1: reported **"80,475,310,058
  producer stalls across 73 of 126 cores"**, which reads as a catastrophically perturbed run and is pure
  garbage. Bounded to `n_worker_cores`.
- The proc-split line reported "126 NoC issues/sweep" on a 120-core grid, same cause.

### Caveats

- **A DRISC's zones are labelled `BRISC`** in Tracy's per-hart lane column: it emits on ring 0 and the handler
  maps risc index 0 through `kRisc[5]`. The zone NAMES are all `DRISC-*`, so nothing is ambiguous in practice.
- **The publish/consume record counts can differ by ~1,200 with the feature on** (e.g. 6,005,320 published vs
  6,004,120 consumed) where they are exactly equal with it off. The drainers keep publishing self frames during
  teardown, after the consumer thread has been told to stop, so the tail of the *self* stream is lost. Producer
  records are unaffected -- ts regressions stay 0 at publish and at consume, and BroadcastRing drops stay 0.
- The self frame's ctrl vector is rewritten on every publish, so consecutive publishes must be separated by a
  write barrier; `self_publish` puts a bounded one at the END (after the write, before any further marker),
  because after a publish the live window is empty and the next marker overwrites a word the in-flight frame is
  still shipping. Reversing that order is silent frame corruption.
- **Self-profiling and the egress ablation are mutually exclusive** (`static_assert`): the ablation re-ships
  pre-staged slots forever and never runs the sweep body.

## §N+42 — The profiler's NoC footprint, measured by the hardware instead of derived (bh-26, 2026-08-13)

`TT_METAL_PERF_DEBUG_NOC_FOOTPRINT=1`: each of the six drainers samples its OWN `NIU_MST_*` counters once per
sweep -- 8 local `NOC_STATUS_READ_REG` loads on both NoCs, 32-bit subtract-then-widen into 64-bit accumulators
(they wrap at ~873,000 sweeps, so an end-to-end difference would be wrong on a long run). Local register loads,
so **collecting the data costs no NoC traffic**, which is why the instrument is free: 0 producer stalls, exactly
6,001,200 records, 0 dropped, 0 regressions, identical to OFF.

Reference config: bh-26, 120 cores, slow dispatch, `--iters 500`, delay 15.

### Measured, per drainer

| DRISC | role | window bytes | window txns | lifetime bytes | lifetime txns | sweeps |
|---|---|---|---|---|---|---|
| 0 | filler | 51.90 MB | 6,424 | **6,573 MB** | 657,934 | 21,831 |
| 1 | filler | 51.79 MB | 6,404 | 6,692 MB | 669,824 | 22,228 |
| 2 | filler | 52.17 MB | 6,467 | 6,180 MB | 618,617 | 20,519 |
| 3 | filler | 51.87 MB | 6,415 | 6,066 MB | 607,225 | 20,141 |
| 4 | mover | 70.89 MB | 6,858 | **88.69 MB** | 298,560 | 146,166 |
| 5 | mover | 70.09 MB | 6,945 | 88.85 MB | 314,341 | 154,083 |
| **total** | | **348.7 MB** | **39,513** | **25.1 GB** | 3.17 M | |

**The NoC split is confirmed exactly as designed**: a filler reads on NoC 1 (34.23 MB, zero writes there) and
writes on NoC 0 (17.67 MB, zero reads). No cross-traffic in either direction.

### The model was ~11% high, NOT 4.8x -- and the 4.8x was an artefact of the comparison

The §N+42 predecessor analysis derived 386 MB in-window / 8.04x payload. Measured aggregate is **348.7 MB**, so
the byte model was **11% high** -- good agreement. The "4.8x discrepancy" first reported was **a per-filler
measurement compared against an aggregate derivation**: derived 163.94 MB / 15,600 txn is the sum over four
fillers, and it was placed next to filler 0's individual 34.23 MB / 3,420 txn. Aggregate against aggregate:

| in-window span read | derived | measured | ratio |
|---|---|---|---|
| transactions | 15,600 | 13,680 (4 x 3,420) | 1.14x |
| bytes | 163.94 MB | 136.9 MB (4 x 34.23) | 1.20x |

and the residual 1.14x is **entirely the assumed sweep count**: the derivation assumed 130 sweeps per filler,
the hardware measured **114** (`3,420 txn / 30 cores = 114.0` exactly). Same single variable explains the
lifetime gap: 200,000 sweeps assumed against **21,831** measured is 9.16x, and 62.5 GB/filler derived against
6.57 GB measured is 9.53x. **The byte-per-sweep model was right all along; only the sweep count was invented.**
The predecessor analysis had itself written "the exact sweep count is config-dependent and must be read from
`res[4]`, never assumed" -- and the headline was then built on an assumed one anyway.

**Two traps, one on each side of the handoff, and both are the SAME trap:** a statistic quoted over the wrong
population. Aggregate vs per-filler on the reporting side; assumed vs measured sweeps on the deriving side.
This file has now recorded that error four times (means over a bimodal distribution, coverage over all sweeps
rather than in-window ones, and both halves of this one). **Always state the population in the same breath as
the number.**

### The finding: polling dominates, and it is entirely the FILLERS' over-read

Lifetime 25.1 GB against 348.7 MB in-window = **~74x more NoC traffic outside the workload than inside it**
(the earlier ~3,400x claim was the assumed-sweep-count error; direction survives, magnitude does not). But the
split between roles is the useful part:

| | lifetime bytes | sweeps | bytes per sweep |
|---|---|---|---|
| filler | ~6.4 GB | ~21,000 | **~307 kB** |
| mover | ~88 MB | ~150,000 | **~0.6 kB** |

**A mover runs 7x MORE sweeps than a filler and moves 74x FEWER bytes**, because a mover's idle sweep reads a
4 B head word per peer while a filler's idle sweep reads 30 full 10,496 B spans unconditionally. So the
profiler's resident NoC cost is *almost entirely* the filler's unconditional full-span read on sweeps that find
nothing. Nothing else in the system is within two orders of magnitude of it.

That makes the two-phase read (256 B control vector first, then only the live ring bytes) the only lever that
matters for mesh traffic: it targets the ~6.4 GB/filler directly. The 592-word short frame does not touch it.

### Instrument cost

247 cycles/sweep on both roles (5,392,503 cyc over 21,831 sweeps on filler 0; 36,103,246 over 146,166 on mover
4 -- identical per sweep, as expected for 8 fixed register loads). Against a ~13 us sweep that is **~1.4%**, and
it measurably does not perturb: 0 stalls at SD delay 15, where §N+41 showed a mere +4% on idle-sweep cost
crosses the knee.

### What this does NOT measure -- and why no percentage is printed

These are **master** counters on the drainer: traffic the drainer *initiated*, i.e. offered load. They say
nothing about
- **the victim side** -- what actually arrived at each worker's L1 (`NIU_SLV_*` on the worker, unread),
- **the workload's own traffic** -- so there is no denominator and the report **refuses to print a
  workload-relative percentage**, naming what would supply it instead,
- **contention** -- two streams sharing an unsaturated link do not interfere, and a master counter cannot tell
  you whether anything was delayed.

The report also refuses to print at all when `credit_timeouts`, `dropped_frames` or `egress_dead` are non-zero,
because `pages`/`pushes` are success counters that simply stop: a footprint computed from them after an egress
failure would read plausibly LOW. Same class as the §N+41 tail flush that could never fire.


## §N+43 — The NoC-footprint PLOTS land on the device timebase, and every code-size intuition was wrong (bh-26, 2026-08-13)

The per-sweep NoC sample now reaches Tracy as **plots on the device timebase**, so "how much NoC is the
profiler moving, right now, next to the zones that explain it" is a picture rather than a teardown total.
Capture: `tracy_captures/drisc_nocfp_plots_sd15.tracy`, SD delay 15, 120 cores, `--iters 500`, zones +
footprint on, 18.78 MB, 126 contexts, 25 plots.

> **CORRECTION (§N+46).** "Land on the device timebase" is right about the *conversion* and was measured on a
> part where it was also right about the *result* -- but only because **bh-26's Tensix-worker and DRAM-core wall
> clocks advance in lockstep** (measured duty ratio 1.0000; on bh-05 the worker side runs at 0.09 of the DRAM
> side and falls minutes behind). The anchor these plots were converted against
> (`chip_anchors_[chip]`) is measured on a **Tensix worker**, while the samples are produced by a **DRAM core**.
> On bh-05 (p100a) those two counters are zeroed by different events, so every number in the table below -- and
> every DRISC zone -- sat **10-42 minutes** right of the workload. The containment percentages here are real for
> bh-26 and were **not** a general property of the code. §N+46 measures the offset, explains it, and fixes it
> with a per-core anchor; re-read that section before quoting any alignment figure from this one.

### The alignment proof, which is the deliverable

A plot stamped at DECODE time would sit milliseconds right of its zones — the mover's ring tail alone trails
the last worker zone by 2.5–2.9 ms (§N+41). Measured with `tools/drisc_drain/tracy_plot_check`:

| | value |
|---|---|
| worker window (from `tracy_zone_csv`) | `[931,159,767 .. 933,083,161]` ns, span **1.923 ms** |
| DRISC zone window | `[931,167,176 .. 936,499,299]` ns |
| filler 9-9 plot: samples / span / inside window | **117** / 1.953 ms / **97.4%** |
| filler 9-9 first sample vs window open | **+16.7 us** |
| mover 9-0 last sample | **936,499,299 ns — the last DRISC zone, to the nanosecond** |
| filler sampling interval | dt p50 **17.3–17.6 us** = the SWEEP+PACE cadence |
| mover sampling interval | dt p50 **1.3–2.0 us** = its faster sweep rate |
| samples per filler | 117 / 140 / 138 / 136 vs §N+41's 117 / 138 / 142 / 139 SWEEP zones |
| **CONTROL: Tracy's own CPU-side `Frame` plot** | span **2,994 ms**, **0.0%** inside the worker window |

The control row is what makes this a proof rather than a plausible table: a host-timebase plot in the same
capture misses the worker window entirely, and the device-timebase ones sit inside it. Sample counts match
SWEEP-zone counts one-for-one, so the series really is one sample per sweep.

**Movers show only 33–35% containment and that is correct.** They keep draining the DRAM ring for ~2.5–2.9 ms
after the last worker zone, so most of their samples legitimately fall past the worker window. That is
§N+41's drain tail, not misalignment — and the fact that their LAST sample coincides exactly with the last
DRISC zone is the check that distinguishes the two.

### The mechanism: a client-only change, because the plot time was already on the wire

`Profiler::PlotData` stamps `GetTime()`. But the time is carried in `QueuePlotDataBase{name, time}` and the
server converts it with `TscTime(RefTime(...))`, so only the client stamp needed to change — **no server,
wire-format, GUI or `tracy-capture` change**. `PlotDataAt` takes the instant explicitly and the caller inverts
the device-zone mapping:

```
zone displays at  ConvertGpuTime(tgpu) = round(ts/freq) - anchor_ns + TscTime(host_anchor)
plot displays at  TscTime(tsc) = (tsc - baseTime) * timerMul
=>                tsc = (round(ts/freq) - anchor_ns) / timerMul + host_anchor
```

The server-private `baseTime` **cancels**, which is the only reason this is computable client-side. Verified
against `Worker::ProcessGpuNewContext` (`timeDiff = TscTime(cpuTime) - gpuTime`), `Worker::TscTime` and both
branches of `ConvertGpuTime` (they agree because `calibrationMod` stays 1.0). Backdating is safe because
`Dequeue()`'s delta encoder is a plain running subtraction with **no ordering assertion** — the same property
`GpuZoneEnd`'s `cpuTime` already relies on. The anchor comes from the handler's own `chip_anchors_`, so
nothing had to be plumbed out of `start()`.

### CODE SIZE: every intuition was wrong, and `noinline` was actively harmful

The DRISC code region is 11,264 B and the sample had to fit alongside self-profiling. Measured per change:

| build (zones + footprint on) | text | |
|---|---|---|
| series, first cut | 11,508 | 244 over |
| + shared `self_mark_w0` prologue | 11,348 | 84 over — real, −160 B |
| + constexpr index table | 11,332 | 68 over — marginal, −16 B |
| + `noinline` on the prologue | 11,332 | **zero effect** |
| + `noinline` wrapper over 47 `get_timestamp()` sites | 11,332 | **zero effect** |
| **− delete all three `noinline` attributes** | **11,228** | **36 B spare** — **−104 B** |

Three corrections, all evidence-backed:

- **`noinline` is NOT ignored here.** `nm` shows real out-of-line symbols (`ts_now`, `nf_sample_regs`,
  `nf_sweep_end`). An earlier note in the kernel claiming "GCC ignores the attribute" was a wrong diagnosis of
  a right observation, and is corrected in place.
- **`noinline` made the binary BIGGER.** Deleting three attributes bought 104 B. GCC's own `-Os` inlining beat
  the hint. **This is what made the series fit** — it fits because hints were removed, not added.
- **Outlining does nothing for small bodies.** Collapsing 47 inlined `get_timestamp()` call sites into one
  wrapper changed the ELF by **exactly 0 bytes**: a call sequence costs what the ~3 inlined instructions did.

The only mechanism that ever paid was restructuring **data** — flat arrays over `[2][4]`, runtime index
tables, and dropping a per-sweep 64 B stack snapshot (that trio bought 424 B). **Do not budget code space
against dead `if constexpr` branches either:** removing the ablation knobs freed 16 B, i.e. nothing, because a
compile arg that is 0 by default was never emitted.

**Splitting the drain kernel into separate filler/mover files would not help.** `kRole` is already a
compile-time arg so each role gets its own ELF: movers come out at 6,788 B against fillers at 11,228 B from
one source, and a filler ELF contains **zero** `socket`/`pcie`/`write_to_host`/`reserve_pages` symbols. The
dead role is already gone; separation would only remove `.bss`, and code is what is scarce.

### RETRACTION: the ~252 GB lifetime figure was ~10x high

§N+42's per-filler lifetime read was extrapolated as ~63 GB (~252 GB across four, ~3,400x the frame bytes)
from an assumed ~200,000 sweeps/filler, taken from §N+41's "167k–379k total sweeps". **That config runs
16,226 sweeps.** Measured: **4.87 GiB per filler**, ~19 GiB across four. The qualitative conclusion is
unchanged and still the headline — a filler's lifetime read is ~279x its in-window frame bytes, so the
profiler's NoC cost is dominated by POLLING, not payload — but the magnitude was wrong by an order of
magnitude and the arithmetic that produced it should not be reused.

**And any lifetime byte total must be quoted WITH ITS SWEEP COUNT.** Two runs of the same code and config gave
6.57 GB/filler at 21,831 sweeps and 4.87 GiB at 16,226 — the drainer is resident, so the total is a function
of how long the process lived, not a property of the workload. A lifetime figure without its sweep count is
not comparable to anything.

### Measurement gotchas

- **`tracy-capture`'s "Zones:" headline is CPU zones only.** This capture reports 8,179 while carrying
  3,003,119 device zone rows. Same trap as `tracy_capture_zones_headline_gotcha`; judge a device capture by
  `tracy_zone_csv` or `tracy_ctx_inspect`, never the headline.
- **Three ways to crash a Tracy server walk, all hit here**: resolving a GPU zone's source location, recursing
  `Child()` (its empty sentinel is not `>= 0`), and walking `GetGpuData()` in-process at all — the last
  segfaults even with empty-lane and null-entry guards. `tracy_plot_check` therefore does plots only and takes
  the zone windows from `tracy_zone_csv`, which already does that walk correctly. Two small tools beat one that
  crashes.
- **The default 4 Mi host record ring silently drops at this volume**: 1,801,941 records dropped until
  `TT_METAL_PERF_DEBUG_RING_RECS=16777216`. Producer stalls stayed 0 throughout, so the stall counters do NOT
  tell you the trace is incomplete — check the BroadcastRing line.

## §N+44 — The profiler on a REAL model: ResNet-50 trace+2cq (bh-26, 2026-08-13)

> **CORRECTION (§N+46): bh-26 is a `p100a`, not a p150.** `tt-smi` on 2026-08-14 reports `board_type p100a`,
> `board_id 0000043231911006`, with firmware byte-identical to bh-05's. Every "bh-26 is a p150" / "p150 vs p100a"
> comparison in this section and in §N+45 is therefore built on a wrong SKU attribution -- either the box was
> re-provisioned or the original reading was wrong. Absolute times here are still not cross-box comparable (two
> different physical cards), but **not for the reason stated**.

First real-model run of the DRISC perf-debug profiler with the NoC-footprint instrument on. Everything below
is from `models/demos/vision/classification/resnet50/blackhole/tests/test_perf_e2e_resnet50.py::test_perf_trace_2cqs[16-0.0018-30-device_params0]`
-- batch 16, 30 iterations, trace capture + replay, 2 command queues, 110 cores, ~844 ms of device time.
Env: `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PERF_DEBUG_PROFILER=1 TT_METAL_PERF_DEBUG_ROLE_SPLIT=1
TT_METAL_PERF_DEBUG_DRISC_ZONES=1 TT_METAL_PERF_DEBUG_NOC_FOOTPRINT=1`, `RING_RECS=67108864`.

### It passes, and the instrument is nearly free

| config | reps | inference time (avg) | producer stalls | records / dropped |
|---|---|---|---|---|
| no profiler | 3 | 0.0015957 / 0.0015872 / 0.0015820 s | -- | -- |
| profiler only | 3 | 0.0016193 / 0.0016144 / 0.0016171 s | **0** of 110 | 1,581,952 / **0** |
| + self-zones + NoC footprint | 3 | 0.0016195 / 0.0016268 / 0.0016199 s | **0** of 110 | ~1,736,000 / **0** |

**e2e overhead ~+2%** (mean 0.0015883 -> 0.0016170 s), against **+4.6%** for the X280 generation. The self-zone
and NoC-footprint instruments add **+0.3%** on top of the profiler, which is inside the baseline's own
run-to-run spread (0.0015820-0.0015957 = 0.9%), i.e. not resolvable.

**The perf gate still passes with profiling on**: the test's target is 0.0018 s and every config lands at
0.0016x. So a real model can be profiled without breaching its own threshold -- the practically important
result. 0 stalls and 0 dropped records in every profiled rep: ResNet's marker density (~1.6-2.5 M records over
844 ms) is far below the host sink's ~16.78 M ceiling, so the §N+39 worry that a real model would saturate the
ring did not materialise. **Nothing here was capped or dropped.**

### The NoC footprint is a different animal from the synthetic workload

Per drainer, in-window, at stock defaults:

| | synthetic 120-core / 2 ms | ResNet-50 / 844 ms |
|---|---|---|
| filler bytes | ~52 MB | **2,231-2,556 MB** |
| filler txns | ~6,400 | 227,341-259,191 |
| mover bytes | ~70 MB | 378-382 MB |
| mover txns | ~6,900 | **1,509,408-1,672,852** |
| filler read rate p50 | 17.90 GB/s | **1.87 GB/s** (max 31.22) |

Two things follow, and they are the argument for the two-phase read:

1. **A filler's read p50 collapses to 1.87 GB/s of a 31.22 max.** Real ops leave the L1 marker rings nearly
   empty nearly all the time, so a typical sweep hauls a mostly-empty 10,496 B span. That is §N+36's
   under-full-span problem in its natural habitat, and it is waste, not throughput.
2. **The two roles have opposite traffic shapes**: fillers ~2.4 GB in ~240k transactions (~10 kB each, whole
   spans); movers ~380 MB in ~1.6 M transactions (~240 B each). Same instrument, and no single number
   describes both -- which is why the plots are per-role and why the aggregate row had to be split.

### THE SAMPLER IS A SAMPLER, and on a real model that is stark

At stock defaults (500 us hold, idle sampling off):

| | traced sweeps | of ALL sweeps | did work | work windows | frames | % of egress |
|---|---|---|---|---|---|---|
| filler | 710-751 | **0.4-0.5%** | 259-269 | **94-95** | 100-102 of 256 | 1.5-1.8% |
| mover | 87,358-95,959 | 4.5% | 2,618-2,619 | 165-173 | 1,795-1,935 of 4,096 | **13.2-15.2%** |

**The dark gaps in a filler lane are the emission window being CLOSED, not the drainer idling.** A filler is
always sweeping or pacing; emission is armed by discovered work and held open for
`TT_METAL_PERF_DEBUG_DRISC_ZONE_HOLD_US` (default 500 us) after the last work-bearing sweep. 844 ms of ResNet
produces 94-95 disconnected windows. So **a filler row is a sample of the run, not a record of it** -- a dark
region means "nothing interesting for at least 500 us", never "nothing happened".

Raising the hold to 50 ms collapses 96 windows to **1** and gives continuous filler lanes (coverage 0.4% ->
4.0-4.6%, frames 100 -> 200-220 of 256), but **movers then hit their cap** (4,097 of a 4,096 budget, 29.5-31.7%
of egress) and truncate instead. Both roles cannot currently be continuous over 844 ms at these sample rates.
Kept at 500 us by choice: nothing is capped there and both roles cover the whole run.

Note the mover's **13-15% of egress at defaults against a filler's 1.5%**. That is sample RATE, not budget: a
mover samples every ~1.27 us because that is how often it sweeps. It is the largest instrument cost in the
system, and decimating the mover's interval -- 1.27 us resolution over 844 ms is far finer than any question
needs -- is the obvious lever if it ever matters.

### Two defects the model exposed, and three fixes that did not survive

**Role-aware self-frame budget (FIXED).** The shared 256-frame cap stopped BOTH movers ~20% into the run and
they never came back: the cap is permanent, not periodic, so nothing re-arms them. Diagnosed by the user
observing that a *sampler* would resume and these did not. Movers now get 16x. Measured need for full coverage:
filler ~100 frames, mover 1,814-1,970. `DRISC_ZONE_FRAMES=0` does NOT mean unlimited -- the code maps 0 -> 256.

**Pacing hysteresis too sticky (FIXED, by constants).** The controller is asymmetric: up x1.25, down x0.875,
and a `kPaceCritical` collapse throws the gap to 0 outright. On a sparse model with 94-96 work windows,
collapses fire constantly and the gap needed ~35-40 sweeps to climb back, so it effectively never arrived --
the PACE distribution from the trace came out **bimodal, 24-32% of samples under 1 us and ~50% at the 148 us
ceiling**, while `out[42]` reported only the FINAL gap and hid it completely (the fifth instance in this file of
a statistic quoted over the wrong population). Now x1.5 with a floor of 512: ~13 sweeps, and a filler's total
sweeps fall **475k-564k -> 166k-178k** with the knee unchanged (delay 15 = 0/0/0). Tuned by IMMEDIATES on
purpose: `gap >> 1` is the same instruction as `gap >> 2`, so it costs **zero DRISC code** -- which is what made
it viable when a branch was not.

**REJECTED 1: seeding the gap at kGapMaxCycles.** Premise "start where it settles" was false -- the controller
settles at 8k-34k cycles on the synthetic workload and only pins at the ceiling on a sparse model. Delaying the
first sweep by up to 148 us moved the knee: delay 15 went 0/0/0/0 -> **35/100/107/75**, delay 10 roughly
doubled, `ring-room waits` still 0, i.e. the **L1 marker** rings filled while the drainer waited.

**REJECTED 2: seeding it for MOVERS too** (a regression introduced while fixing the above, caught by the user).
The seed keyed on `kFillPct` alone while the controller that lowers it is gated
`kFillPct != 0 && kRole != kRoleMover`, so movers pinned at 148 us with no feedback path: ~114x slower to drain,
and a 20 ms trace replay took ~100 ms to finish. **The symptom looked like efficiency** -- mover sweeps fell
150x and were reported as a win. Rule: **a role with no feedback path must never be seeded away from 0**, and a
seed must be gated on the same condition as the controller that maintains it.

**REJECTED 3: a two-regime fast ramp** (x5 below a quarter of target), in three variants -- unbounded, bounded
to the low eighth of the range, and gated on having staged a frame. All three cost **3-8 stalls at delay 15-20**
where the gentle ramp gives 0, because before producers start, fill reads 0 and looks identical to sparseness.
The final variant **overflowed the DRISC code region**, the drainer FAILED TO LOAD, and the run still passed
while reporting a plausible time with no device zones at all.

### METHOD ERROR that invalidated several intermediate readings

**The DRISC kernel is JIT-compiled AT RUNTIME, so a code-region overflow appears in the RUN log, never in the
`ninja` output.** Several "built, no overflow" checks during this work greped the build and proved nothing; the
instrumented configuration was silently not loading while an uninstrumented one was being tuned and measured.
Only the loud-failure path (`FAILED TO LOAD ... THIS CAPTURE WILL BE EMPTY`) surfaced it. **Check the run log.**

### A configuration that stalls, not yet bisected

`TT_METAL_DEVICE_PROFILER=1` + `TT_METAL_DRISC_PROFILER=1` **without** `TT_METAL_PERF_DEBUG_PROFILER=1` sat at
99.7% CPU in userspace for 10+ minutes on this test, where comparable reps finish in ~3.7 s, with every worker
thread parked in `futex_wait` -- the signature of a host poll for a consumer that never came up. One
observation, and `DRISC_PROFILER=1` is the documented way to suppress the legacy RT manager and DRAM profiler
(§647), so someone will reach for it. Worth a two-arm env bisect.

### The MOVER had no controller at all, and its traffic was almost entirely empty polling

Discovered from the footprint table above: a mover moved ~380 MB in ~1.6 M transactions -- about **240 B per
transaction** -- while the frames it ships are 10,560 B each. That gap is 4 B head polls on an empty ring. A
mover swept every ~1.27 us whether or not its DRAM ring held anything, because `gap` stayed at 0: movers are
excluded from the FILL-ratio controller (correctly -- they have no worker grid, so span fill is permanently 0
and that controller would creep them to the 148 us ceiling and pace the consumer instead of the producer).

Added a mover-side controller with a **different criterion and a different ceiling**:

- **Criterion is FRAMES AVAILABLE**, not span fill: collapse to 0 the instant a sweep moved anything, back off
  only when the ring was empty. It therefore cannot delay a real drain, only suppress polling an empty ring.
  Reuses the existing `frames != frames_at_sweep_start` busy signal, so there is no second definition of
  did-this-sweep-do-anything to drift from.
- **Ceiling 10 us (13,500 cycles)**, ~15x tighter than a filler's 148 us. Pacing a CONSUMER is only safe while
  it is provably idle and while one missed observation cannot materially extend the drain tail.
- Room to add device code at all: **each role compiles its own ELF**, and the mover binary is ~6,788 B against
  the 11,264 B code region (~4.4 KB spare) where a filler has ~36 B. This is the only place new DRISC code fits.

Measured on ResNet-50 trace+2cq, 3 reps:

| | before | after |
|---|---|---|
| mover NoC transactions | 1,509,408-1,672,852 | **182,959-186,255** (~8x fewer) |
| mover bytes | 378-382 MB | 271-290 MB (~25% less) |
| mover total sweeps | 1.93-2.13 M | ~200,000 (~10x fewer) |
| mover self-zone cost | 13.2-15.2% of egress | **3.7-4.1%** |
| mover plot sample interval | 1.27 us | **11.47 us** |
| settled gap | 0 | 13,500 cyc (the ceiling) |
| producer stalls | 0 | **0** (3/3 reps) |
| staged == moved | exact | exact on all four rings, 3/3 |
| inference time | 0.0016146-0.0016195 s | 0.0016121-0.0016185 s (indistinguishable) |

The ~8x transaction cut confirms the diagnosis rather than merely being consistent with it: if the traffic had
been payload, pacing an EMPTY ring could not have removed it. Note the self-zone cost fell to under 4% as a
CONSEQUENCE of fewer sweeps, with no change to the sampling rule -- the instrument got cheaper because there was
less pointless activity to describe.

**Drain tail measured, not inferred**, because `staged == moved` proves the rings emptied by teardown and not
WHEN: the last DRISC zone lands **+1.294 ms** after the last worker zone on an 839.6 ms workload (0.15%). Knee
unaffected: delay 15 = 0/0/0, delay 20 = 0/3/0.

### Where a DRISC lane's DARK REGIONS come from -- emission, not loss

Worth stating because it reads like dropped data and is not. A dark region is an **emission decision on the
device**, before any marker exists: the `self_on` gate means that sweep writes no markers, so there is no
packet, no frame and no bytes to lose downstream. That is why the instrument costs ~1.5% of a filler's egress --
it declines to generate rather than generating and discarding.

Three causes, distinguishable by the SHAPE of the darkness:

| what you see | cause | stage |
|---|---|---|
| gaps scattered mid-run | work-armed window closed (no work-bearing sweep for `ZONE_HOLD_US`, default 500 us) | device, emission |
| dark from a point to the END, never returns | self-frame budget spent -- permanent, nothing re-arms it | device, emission |
| dark at the FRONT, contiguous after | host record ring overwrote the OLDEST unread records | host, transport -- the only real drop |

The payload path is lossless by construction from producer to host ring: the L1 marker ring BLOCKS a producer,
the DRAM frame ring makes the filler wait for room, and the socket FIFO makes the mover wait for credit. Only
the `BroadcastRing` discards, and because it overwrites the oldest unread it eats the BEGINNING of a run and
leaves the tail contiguous -- which is why the 50k-zone synthetic capture was 31-46% complete yet not a sieve.

**A dark region is therefore not evidence the drainer was idle.** It swept and paced through there and found
nothing worth describing for at least 500 us. On this model a filler traced 16,367 sweeps of 212,690 with only
~2,400 doing work, so even with the window forced open the lane is ~97% idle polling.

### Captures

`tracy_captures/resnet50_default500us.tracy` is canonical (stock defaults, nothing capped).
`resnet50_hold50ms.tracy` has continuous filler lanes with truncated movers. `resnet50_trace2cq_nocfp.tracy`
(movers cut at 20%) and `resnet50_roleaware.tracy` (contains the mover cold-start regression) are superseded
and misleading -- delete them.

## §N+45 — e2e overhead across FOUR real models, and the LLM decode regime (bh-05 / **p100a**, 2026-08-14)

§N+44 measured one model (ResNet-50) on one part (bh-26, p150) and got **+2%**. This widens that to four
models on a **different part -- bh-05 is a p100a**, not a p150 -- and adds the regime that actually stresses the
host sink: **LLM decode, which emits ~46x ResNet's marker volume** (73.6 M records vs 1.58 M).

Two comparability warnings before any number is read:

- **Everything here is p100a**; §N+44 is p150. Overhead *ratios* are within-box and sound; absolute times are not
  cross-box comparable.
- The 5-model set this task inherited (Llama / Mistral / BERT-large / ResNet-50 / DistilBERT) was originally a
  **wh-09 n150 WORMHOLE** study of *device kernel time* under `tracy -p -r`. **Nothing here is comparable to it** --
  that was not e2e, not Blackhole, and not this profiler. Only the model names carry over.

### Method

Three configs per model, **4 reps each with rep 1 DISCARDED** -- rep 1 pays DRISC-firmware JIT and, for the LLMs,
the one-time HF -> tt weight-cache conversion. Those runs are 1.3-8x the warm time (ResNet 39 s vs 10 s; Mistral
150 s vs 18 s) and would wreck any mean.

| config | env |
|---|---|
| `base` | nothing |
| `drisc` | `TT_METAL_DEVICE_PROFILER=1 TT_METAL_PERF_DEBUG_PROFILER=1 TT_METAL_PERF_DEBUG_ROLE_SPLIT=1 TT_METAL_PERF_DEBUG_RING_RECS=67108864` |
| `driscz` | `drisc` + `TT_METAL_PERF_DEBUG_DRISC_ZONES=1 TT_METAL_PERF_DEBUG_NOC_FOOTPRINT=1` |

No `tracy-capture` attached for the timing numbers -- it starves the decoder and would change the workload being
timed. Interpreter: **`python_env/bin/python` (3.10.18, torch 2.11.0+cpu)** from `create_venv.sh`; `/opt/venv`
exists in the container but has no `ttnn`. Harness + raw logs live OUTSIDE the repo at
`/localdev/mmemarian/e2e_ovh/`.

### The headline: overhead falls as per-iteration work grows

| model | test | metric | no profiler | + DRISC profiler | overhead | + instrument | overhead |
|---|---|---|---|---|---|---|---|
| ResNet-50 b16 trace+2cq | `test_perf_e2e_resnet50.py::test_perf_trace_2cqs[16-0.0018-30-device_params0]` | inference s | 0.0015495 | 0.0016443 | **+6.1%** | 0.0016338 | **+5.4%** |
| DistilBERT b8 seq384 | `test_perf_distilbert.py::test_performance_distilbert_for_qa` | inference s | 0.0165536 | 0.0175138 | **+5.8%** | 0.0175479 | **+6.0%** |
| Mistral-7B b1 decode | `simple_text_demo.py::test_demo_text -k "performance and batch-1"` | ms/token | 26.33 | 27.30 | **+3.7%** | 27.21 | **+3.4%** |
| Llama-3.1-8B b1 decode | `simple_text_demo.py::test_demo_text -k "performance and batch-1"` | ms/token | 48.03 | 49.21 | **+2.5%** | 49.18 | **+2.4%** |

Baseline run-to-run spread is **0.09-0.19%** in every model, so a +2.4% effect is ~13x the noise floor -- these
overheads are real, not scatter. The ordering is the point: **the cost is close to fixed per unit of device work,
so it dilutes as the op gets bigger.** ResNet at 1.55 ms/inference pays 6.1%; Llama at 48 ms/token pays 2.5%
*while emitting 46x more markers*. Marker volume is NOT what sets e2e overhead.

**The instrument (self-zones + NoC footprint) is free**: -0.7%, +0.2%, -0.3%, -0.1% against the profiler-only arm,
i.e. inside noise in all four, and *negative* in three. Consistent with §N+44's +0.3%. It can stay on by default.

### Capture integrity -- every profiled rep

| model | records taken | dropped | ts regressions publish / consume | producer stalls |
|---|---|---|---|---|
| ResNet-50 | 1,581,952 | **0** | **0** of 1,581,952 / **0** of 1,581,952 | **0** of 110 cores |
| ResNet-50 +instr | 1,820,769 | **0** | **0** of 1,714,278 / **0** of 1,714,278 | **0** of 110 |
| DistilBERT | 495,324 | **0** | **0** of 495,324 / **0** of 495,324 | **0** of 110 |
| DistilBERT +instr | 635,176 | **0** | **0** of 571,912 / **0** of 571,912 | **0** of 110 |
| Mistral-7B | 70,755,924 | **0** | **0** of 70,755,924 / **0** of 70,755,924 | **0** of 110 |
| Mistral-7B +instr | 72,788,482 | **0** | **0** of 71,859,760 / **0** of 71,859,760 | **0** of 110 |
| Llama-3.1-8B | 73,551,420 | **0** | **0** of 73,551,420 / **0** of 73,551,420 | **0** of 110 |
| Llama-3.1-8B +instr | 75,557,854 | **0** | **0** of 74,647,408 / **0** of 74,647,408 | **0** of 110 |

Checked on every profiled run, not sampled: **zero `FAILED TO LOAD`, zero `overflows region`** across all 26
profiled runs, and every one carries a full drainer report (94 log lines; 126 with the instrument). A profiled
run with no report lines would be a run with no drainer and a meaningless time -- that trap did not fire here.

`consumer took` exceeding `RING_RECS` (70.8 M of a 67.1 M cap) is not a defect: the cap is ring *capacity*, the
count is cumulative throughput. **The sink absorbed 73.6 M records with 0 dropped** -- so the §N+39 worry that a
real model would saturate the ring does not materialise even on LLM decode, at 64 Mi records of ring. Where
`took` exceeds `order-checked` (driscz only) the difference is the DRISC self-zone series, which has no per-lane
order invariant -- both regression counts are still 0.

Host-writer cost at the LLM extreme (Llama, one rep): 647,387 reads, 37,288 MB, **sock-read 19.41 GB/s**,
**decode 20.8 ns/marker**, publish 237.6 ms, ack 97 ns/read.

### THE LLM REGIME IS THE EDGE, and it is the ack path that frays first

Two events, both **only in Llama**, both in the **4th consecutive profiled rep**, never in ResNet/DistilBERT/Mistral:

1. `drisc` rep4: **`CREDIT WAIT TIMED OUT 1x` twice, dropping 7 + 3 = 10 frames**, with **3 producer stalls across
   3 of 110 cores**. The cause is printed: `writer: first data 2977 ms after thread start [large => the FIFO sat
   unserviced and producers will have stalled once]`. The host consumer stopped acking long enough to trip the
   credit wait; the drainer dropped frames **rather than block the workload**, which is the designed trade.
   Order invariants stayed 0 -- the survivors are not scrambled.
2. `driscz` rep4: **stalled mid-decode at iteration 42 of 200** and made no progress for minutes at ~113% CPU;
   reaped by the harness `timeout`. Reps 1-3 were clean and identical (49.18 ms/token), so the driscz Llama mean
   above is over **n=2** warm reps, not 3 -- stated because it is a weaker mean than every other cell here.

So at ~74 M records the profiler is at the edge of the **host ack path**, not of bandwidth or of the ring: 0
dropped records, 0 regressions, ring high-water untouched, but the writer thread can go unserviced for seconds.
**The failure mode is scheduling/servicing latency on the host, and it appears as lost frames, not as corruption.**
That both events landed on the 4th rep of a series is suggestive of accumulation across device opens (cf. the
periodic-teardown result) and is NOT explained here -- it needs a dedicated repeat-count sweep.


### THE HANG: Llama + driscz wedges the CARD, and the wedge outlives the process

The `driscz` Llama rep4 above is worth its own subsection, because the failure is **not** confined to the run
that caused it.

**What was observed, in order:**

1. Reps 1-3 of `llama driscz` were clean and near-identical (49.18 / 49.18 ms/token, 75.5 M records, 0 dropped,
   0 regressions, 0 producer stalls).
2. Rep 4 logged decode iterations up to **42 of 200** (last line 14:19:33), then a pair of drainer
   `socket N drained: 86,023,575 pages, 9,342,747 markers (94,611 reads); producer stall zones: 0` lines at
   14:21:33 -- and then **nothing at all for 36 minutes**, at a steady ~113% CPU, until the harness reaped it.
   The workload stopped advancing; the process did not crash and did not print an error.
3. **After that run, the CARD was wedged.** A `base` (no profiler, no DRISC, nothing) ResNet-50 e2e run --
   the exact invocation that had completed in 9.0 s a hundred times that afternoon -- hung within one second of
   entering the model, last line `==== Running conv2` in the warmup, and had to be killed.

So the state left behind is a device-level wedge, not a Python-level one: **the next victim is an innocent
unprofiled run, and it fails at a completely unrelated place (first conv).** Anyone bisecting from that symptom
will chase ResNet and find nothing. Recovery is a chip reset (`tt-smi -r`); nothing softer was tried and no host
reboot is warranted or was performed.

This is consistent with the resident-drainer/NIU-restore contract: the DRISC drainers are launched to outlive
user programs and restore NIU state on a host handshake. If the run dies between "drainers resident" and "NIU
restored", the DRAM cores are left in the drainer's NoC configuration and the next program's traffic wedges.
**That makes the safety property clear: any path that can abandon a resident drainer must be treated as a
card-corrupting path, not merely a lost capture.**

**Repro procedure** (bh-05, p100a, `mo/drisc_drain_fast` @ a8e56fbdc4f; n=1 so far, so treat as a candidate, not
a recipe with a known hit rate):

```
ssh -p 42756 yyzo-bh-05                        # container as mmemarian; `tt run bh-05` is the HOST, not this
cd /localdev/mmemarian/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_HOME=/localdev/mmemarian/hf_cache TT_CACHE_PATH=/localdev/mmemarian/tt_cache
export HF_MODEL=unsloth/Llama-3.1-8B-Instruct  # ungated mirror; base_model_name -> Llama-3.1-8B

# run this FOUR TIMES IN A ROW in the same shell, fresh process each time.
# Reps 1-3 are expected to pass in ~34 s; the failure showed up on rep 4.
env TT_METAL_DEVICE_PROFILER=1 TT_METAL_PERF_DEBUG_PROFILER=1 \
    TT_METAL_PERF_DEBUG_ROLE_SPLIT=1 TT_METAL_PERF_DEBUG_RING_RECS=67108864 \
    TT_METAL_PERF_DEBUG_DRISC_ZONES=1 TT_METAL_PERF_DEBUG_NOC_FOOTPRINT=1 \
    python_env/bin/python -m pytest -s -q \
    models/tt_transformers/demo/simple_text_demo.py::test_demo_text -k 'performance and batch-1'

# WEDGE CHECK -- the point of the exercise. After the run that hangs, with NO profiler env at all:
python_env/bin/python -m pytest -s -q \
  'models/demos/vision/classification/resnet50/blackhole/tests/test_perf_e2e_resnet50.py::test_perf_trace_2cqs[16-0.0018-30-device_params0]'
# healthy: ~9 s, "inference time (avg): 0.00155".  wedged: hangs at "==== Running conv2".
```

Or via the harness (same thing, logs + rep accounting for free):
`/localdev/mmemarian/e2e_ovh/run_ovh.sh llama driscz 4`

**Signals to collect when it fires**, in priority order: whether the last log line is a `socket N drained`
report (drain accounting) or a decode iteration; the `writer: first data N ms after thread start` value; whether
`CREDIT WAIT TIMED OUT` appears at all (in the `drisc` rep4 it did -- 2 timeouts, 10 frames dropped, 3 producer
stalls -- and that run *survived*; the driscz rep4 that hung shows the safety valve is not sufficient in every
path); and `py-spy dump` on the live process to separate "host thread blocked on device" from "spinning".

**Two adjacent facts that will mislead a bisect, recorded so they do not:**

- `TT_METAL_PERF_DEBUG_DRISC_ZONES` + `NOC_FOOTPRINT` were on in the run that hung, and the profiler-only
  (`drisc`) arm survived the same rep-4 position with dropped frames instead. That is **suggestive, not
  established** -- one hang is one hang, and the drisc arm's rep4 anomaly says the underlying pressure exists
  without the instrument.
- **`test_perf_debug_zones` is NOT a usable card-health probe at this commit.** Run with no profiler env it
  aborts in `sync_build_steps` with a *source* error, `'WALL_CLOCK_LOW_INDEX' is not a member of
  'kernel_profiler'` (`programming_examples/profiler/test_perf_debug_zones/kernels/zones_dm.cpp:46`), because
  the unprofiled variant of that kernel does not compile. It builds and runs fine *with* the profiler env, so
  the example's non-profiled path is simply broken here. It looks exactly like a wedge and is not one. Use an
  unprofiled ResNet e2e rep as the health probe instead.

### The NoC footprint: LLM decode is a different animal again

Per-role bytes/txns in the workload window (4 fillers + 2 movers, driscz rep 2):

| model | filler bytes each | filler mean B/txn | mover bytes each | mover mean B/txn |
|---|---|---|---|---|
| ResNet-50 | 1.84-2.15 GB | 10,243-10,310 | 0.27-0.29 GB | 1,693-1,822 |
| DistilBERT | 1.80-2.55 GB | 10,261-10,329 | 0.26-0.27 GB | 1,411-1,427 |
| Mistral-7B | 26.9-39.8 GB | 8,680-9,207 | 34.7-35.1 GB | 6,274-6,378 |
| Llama-3.1-8B | 37.0-53.2 GB | 9,007-9,425 | 37.6-37.8 GB | 5,566-5,638 |

**A mover transaction is 3.3-4.5x fuller on LLM decode than on ResNet** (5.6-6.4 kB vs 1.4-1.8 kB). §N+44's
"under-full span" complaint -- real ops leave the L1 rings nearly empty, so a sweep hauls a mostly-empty
10,496 B span -- is therefore **a ResNet-shaped artifact, not a general property**. On LLM decode the frames
arrive full and the two-phase-read argument loses most of its force. Filler mean B/txn stays pinned at ~9-10 kB
in all four because a filler reads the WHOLE span regardless of fill, so that column measures span size, not
occupancy; the mover column is the one that measures occupancy.

Mover bytes go from ~4% of filler bytes (ResNet) to **~90-100%** (LLMs): with full frames, the egress side stops
being a rounding error and becomes half the traffic.

### Model availability on this part -- what is measurable and what is not

- **BERT-large: NOT MEASURABLE on Blackhole.** Both e2e perf tests hard-skip on BH:
  `models/demos/metal_BERT_large_11/tests/test_perf_bert11.py::test_perf_bare_metal_wh` is
  `@pytest.mark.skipif(is_blackhole(), reason="Not functional on BH")` plus `@run_for_wormhole_b0`, and
  `tests/ttnn/integration_tests/bert/test_performance.py` is `skipif(is_wormhole_b0() or is_blackhole())`.
  Its CI entry is commented out (issue #48191). There is no `blackhole/` BERT-large variant. **4 models, not 5** --
  reported rather than substituted with a device-perf test (which needs the profiler and so cannot measure
  profiler overhead).
- **LLM weights came from UNGATED MIRRORS.** bh-05 has no HF token, no `/mnt/MLPerf`, and no local checkpoints;
  `meta-llama/Llama-3.1-8B-Instruct` is `gated=manual` and cannot be fetched. Used
  **`unsloth/Llama-3.1-8B-Instruct`**, chosen deliberately: `get_base_model_name` maps its leaf name to
  **`Llama-3.1-8B`**, the exact key the tree's config branches use, whereas a `Meta-Llama-*` mirror maps to
  `Meta-Llama-3.1-8B` and would take a different config path. Mistral needed no mirror --
  `mistralai/Mistral-7B-Instruct-v0.3` is ungated. **These are community mirrors, not verified byte-identical to
  the originals**; the config path is identical, the weights are equal only by reputation.
- **Mistral-7B is not BH-validated upstream**: no `bh_p150` entry in `models/model_targets.yaml`, WH-only SKUs in
  `models_e2e_tests.yaml`, `tt_transformers/README.md` lists n150 only. It runs unguarded and its numbers above
  are self-consistent, but it is not a supported BH config.
- **Perf gates**: ResNet **passes its 0.0018 s target in every config** including profiled (0.00163-0.00166) --
  the §N+44 result holds on p100a. DistilBERT **fails its 0.0165 s target in ALL THREE configs, baseline
  included** (0.016554 s, a 0.3% miss): that is the p150-tuned threshold meeting a p100a, **not** a profiler
  effect. Llama/Mistral have no BH target, so no gate fires.

### Reproducing

```
# container on bh-05 is reached as mmemarian via: ssh -p 42756 yyzo-bh-05
/localdev/mmemarian/e2e_ovh/run_ovh.sh <resnet|distilbert|mistral|llama> <base|drisc|driscz> 4
/localdev/mmemarian/e2e_ovh/gen_findings.py     # regenerates every table above from the rep logs
```

## §N+46 — The DRAM-core clock has its own ORIGIN, and it is board-dependent (bh-05 / **p100a**, 2026-08-14)

Every DRISC drainer zone and every NoC-footprint plot sample on bh-05 sat **tens of minutes to the right of the
workload it describes**, while its *durations* were correct. §N+43 declared this timebase solved; it was solved
on a part where the bug is invisible. This section measures the offset, names the mechanism, and fixes it.

### The measurement that identifies it: spans right, origin wrong

Three captures, same code, same box, differing only in how long the card had been up since its last reset:

| condition | DRISC − worker zone offset |
|---|---|
| card up ~31 min, ~30 device opens | 1,111,245,448,933 ns = **18.52 min** |
| 41 min later, **no reset** | 2,544,449,032,352 ns = **42.41 min** |
| 12 s after `tt-smi -r` | 9,367,303,880 ns = **9.367 s** |

Zone **spans** agreed with worker spans to **0.15–0.17%** in all three. So the frequency was never wrong — only
the origin — and the offset tracks *time since chip reset*, not time since anything else.

### The mechanism — measured, and NOT the obvious one

**RETRACTED before it was ever written down: "the worker clock is re-zeroed at every device open".** That was the
working hypothesis for most of this investigation and it is **false**. It predicts that two consecutive processes
read back the same small value; they do not. Three back-to-back `--clkprobe` processes, starts 32 s apart, on an
un-reset card:

| | probe 1 | probe 2 | probe 3 | advance per 32 s of WALL |
|---|---|---|---|---|
| worker v(1,2) | 97,519,878,564 | 99,979,838,639 | 102,560,804,955 | **+1.82 s, +1.91 s** |
| DRAM bank 0 v(18,12) | 2,261,193,437,386 | 2,287,850,854,682 | 2,314,637,641,541 | **+19.75 s, +19.84 s** |

Read off that table:

- **Neither counter re-zeroes at device open.** Both are **strictly monotonic across separate processes**. There
  is one zero point for both — the **chip reset** — and nothing in a device open touches either.
- **Neither is a wall clock.** Over 32 s of wall time the worker advanced 1.8 s and the DRAM core 19.8 s.
- **In-process both tick at exactly 1350.0 MHz** (the 500 ms bracket inside each probe measures 1350.0 on every
  core, every time). So the frequency is right and only the accumulated total differs.

So the divergence is **clock GATING, not re-zeroing**: the two counters sit in different clock domains with
duty cycles that differ by ~11x. The Tensix counter advances only while Tensix is out of reset (~1.8 s per short
process, i.e. it is stopped nearly all the time), while the DRAM-tile counter keeps running whenever the card is
powered but at a reduced clock when idle (19.8/32 ≈ 62% of the 1.35 GHz active rate, i.e. ~0.84 GHz idle). Both
started at 0 at the last chip reset; by the time the card has been up a while the DRAM counter has banked
**~22x** the worker's total (2,315 s vs 103 s here).

That is why the offset **grows monotonically with card activity and is not predictable from anything the host
knows**. It also explains the three-point table above without any extra assumption: right after a reset both
totals are near 0 and the offset is seconds; leave the card up and the DRAM side runs away.

**Do not cite the "worker window minimum is 5.6-6.0 s in every capture" observation as evidence about this.**
A worker window is in Tracy DISPLAY ns, already rebased onto host TSC by the anchor — it says nothing about the
raw device counter, and using it that way is what made the re-zeroing story look confirmed.

The defect itself is unchanged by the corrected mechanism:

- `sync_device_clock()` was called with `w = ctx.core_virt[0]` — a **Tensix worker core** — and the single result
  registered per chip via `AddDevice(chip_id, ...)`. The DRAM-core contexts, pre-created in `PreCreateContexts`,
  **inherited that worker-derived anchor**.
- Every zone pushes a **raw** device timestamp once `ctx.synced` (`pkt.timestamp = r.ts - base`, `base = 0`), so
  the DRAM-core rows were displaced by exactly the reset→open gap. The NoC-footprint plots inherited the same
  anchor through `chip_anchors_` and were displaced identically.

**Reset discipline cannot fix this, and that was tried.** Device open takes ~9–12 s and a ResNet workload window
is ~753 ms, so the gap is **structurally larger than the window it would have to fit inside**. There is no
reset-then-run cadence that makes a worker anchor correct for a DRAM core. (With the gating mechanism in hand the
reason is sharper still: the DRAM counter banks ~20 s for every 32 s the card merely *exists*, so the offset is
already tens of seconds before the first workload runs.)

### It is PER-CARD — validate on bh-05, never on bh-26. And the anomaly is on the TENSIX side.

On **bh-26** the offset is **microseconds**, so the bug is invisible there and a broken build looks perfect. Every
§N+43 alignment number was taken on bh-26. **A fix that looks correct on bh-26 proves nothing.**

Measured on bh-26 with the same 3-probe, 32 s-apart procedure (read-only; compiled to `/tmp` and linked against
the existing lib, because **bh-26's `build_Release` is root-owned** — `docker exec` pollution, do not rebuild
through it):

| | worker advance | DRAM advance | worker/DRAM **duty ratio** | totals |
|---|---|---|---|---|
| **bh-26** | +19.60 s / 34 s | +19.60 s / 34 s | **1.0000** | 50.8 h ≡ 50.8 h |
| **bh-05** | +1.82 s / 32 s | +19.75 s / 32 s | **0.09** | 72 s vs 1,675 s (**23x**) |

bh-26's worker−DRAM offset across the three probes: **4.04, 5.45, 3.56 us** — that is just the latency between
the two `read_reg` calls, i.e. **zero to measurement precision**, and it reproduces the "+0.003 ms" datum exactly.

**This inverts the framing of the whole bug.** The DRAM counters are IDENTICAL on both cards (~19.6–19.8 s per
32 s, ~60% of wall — the chip clock-gates when idle, on both). So the DRAM side is *normal everywhere*. What is
anomalous is **bh-05's TENSIX wall clock, which is stopped ~91% of the time**. The worker-derived anchor is not
"the wrong clock" so much as *a clock that has been frozen for most of the card's life*, and it falls behind by
exactly the accumulated shortfall. On bh-26 the two domains run in lockstep, so borrowing the worker's anchor is
accidentally valid.

**Three explanations RULED OUT by measurement, so nobody re-runs them:**

| hypothesis | verdict |
|---|---|
| different board SKU (§N+44 records bh-26 as a **p150**) | **FALSE — bh-26 reports `p100a`**, same as bh-05. Either it was re-provisioned or §N+44 is wrong. |
| different card firmware (bh-05 has a `p100a.fwbundle` + `tt-flash`, i.e. it was flashed) | **FALSE — byte-identical**: `fw_bundle 19.12.0.0`, `cm_fw 0.34.0.0`, `dm_app_fw 0.28.0.0`, `gddr 2.16` on both. |
| bh-26 re-zeroes both counters at device open | **FALSE** — bh-26's counters are monotonic across processes and read **50.8 h**, so they have not been zeroed since its last chip reset either. |

So it is a **per-card difference between two same-SKU, same-firmware p100a boards**, and the residual question is
why one card gates its Tensix clock domain at idle and the other does not. **NOT ESTABLISHED.** The obvious next
experiment is a `tt-smi -r` on bh-05 followed by a re-probe: if the duty ratio returns to 1.0 the gating is a
recoverable *state*, and if it stays at 0.09 it is a property of that part. Worth noting bh-05 is the box with the
degraded-MMIO / VM-freeze / card-wedge history (§N+45, `degraded_state_is_mmio_latency`), so "this card is in an
unusual state" is a live lead rather than a wild guess.

**None of this changes the fix**, which measures the actual per-core offset at every device open and is therefore
correct on a card with a duty ratio of 1.0000 and on one at 0.09 alike.

### PREREQUISITE, proven before any profiler edit: a DRAM tile answers the wall-clock register

`RISCV_DEBUG_REG_WALL_CLOCK_L/H` (`0xFFB121F0` / `0xFFB121F8`) is documented as a **Tensix-tile** debug
register. Nothing had confirmed a **DRAM tile** answers at all, and the whole per-core-anchor approach dies if it
does not. Tested in isolation first — `test_perf_debug_zones --clkprobe 1`, which reads the register on a worker
and on **all 7 DRAM views** and does not boot a drainer, so a fault would indict the read and nothing else:

| | result |
|---|---|
| readable on a DRAM tile | **yes, all 7 views**, no fault, no hang, no reset needed |
| non-zero | yes (5.2e11 cycles) |
| advances | yes |
| rate over a 500 ms host interval | **1350.0 MHz on all 8 cores** = aiclk exactly |
| spread across the 7 DRAM views | ~28k cycles ≈ 21 us — the sequential read latency, not clock skew |

Two probes 40 s apart also show the divergence happening: the DRAM value advanced +53.7e9 cycles while the worker
advanced only +2.49e9, and the offset grew 377,758 ms → 415,699 ms. **This is also the command that killed the
re-zeroing hypothesis** — see the mechanism table above; run it 3x back-to-back and both counters increase
monotonically, which "re-zeroed at open" cannot produce.

### The fix: a per-core anchor, host-side only

`sync_device_clock()` now runs **once per DRISC on its own DRAM core**, and the result is registered with a new
`PerfDebugTracyHandler::AddCore(chip, noc0_x, noc0_y, ...)` into a `core_anchors_` map keyed by `ContextKey`.
Anchor resolution became `LookupAnchor`: **per-core entry first, chip entry as fallback**. Both consumers use
it — `GetOrCreateContext` (zones) and the `SPSC_DATA_ID_NOCFP` branch of `HandleWorkerEvent` (plots), which had
its own `chip_anchors_` lookup and would otherwise have kept every plot point at the old offset.

Five properties worth stating, because each one is a way to get this wrong:

- **Registered before `PreCreateContexts`.** A Tracy context bakes its anchor in at creation, so an anchor that
  arrives after the context is silently ignored.
- **Keyed on NOC0**, resolved through `ctx.virt_to_noc0`. `drisc_virtual` is the VIRTUAL space; registering the
  virtual pair would look up nothing and degrade in silence — the same trap `SetDriscRole` documents.
- **Unconditional, not gated on "the offset looks big".** On a part whose counters share an origin the per-core
  anchor is the same number measured on the core that actually produced the timestamps, which is strictly better
  than borrowing a worker's.
- **Each DRISC gets its own frequency fit**, not the worker's, so `device_at_anchor` and `frequency` come from
  one self-consistent least-squares fit. Measured: 1.349902–1.350039 GHz across the six, worker 1.350038.
- **Degrades loudly.** A failed DRAM-core fit falls back to the worker anchor with a `log_warning` naming the
  core, because that silently reinstates the entire bug.

Cost: 6 × 100 register reads at device open, ~2 ms of a 9–12 s open. Nothing on the drain path. **No firmware
change** — the device-side rebase was not needed. And the claim near `perf_debug_profiler.cpp:671` that the
drainer FW co-samples these registers in a `calibrate()` is **FALSE**: `calibrate` appears nowhere in
`drisc_profiler_drain.cpp`, in `tt_metal/hw`, or anywhere else device-side. Corrected in place — there was never
a device-side co-sample to lean on, which is exactly why a host per-core anchor was needed.

The run log now states the offset per DRISC, which is the board-dependence tell — microseconds means the part
shares an origin and this changed nothing; minutes means it just fixed the capture:

```
Device 0 DRISC 0 NOC0 (9,9) clock sync: frequency=1.349902 GHz,
  device_time_at_anchor=897228525747 cycles, offset vs worker anchor +649728.554 ms
```

### VERIFIED TWICE, 25.5 min apart on an UN-RESET card — which is the whole point

One aligned capture right after a reset proves nothing: it has to hold **as the two counters diverge**. Both
captures are ResNet-50 `test_perf_trace_2cqs[16-0.0018-30-device_params0]`, zones + footprint on, under
`tracy-capture`, no reset between them. The offset **grew 2.35x** across the pair and alignment did not move:

| | capture 1 (17:26) | capture 2 (17:51) |
|---|---|---|
| worker anchor | 20,158,353,764 cyc (14.93 s) | 90,649,806,954 cyc (67.15 s) |
| DRISC 0 anchor | 897,228,525,747 cyc (664.6 s) | 2,149,036,633,766 cyc (1592.2 s) |
| **anchor offset the fix removes** | **+649,728.554 ms (10.83 min)** | **+1,524,808.622 ms (25.41 min)** |
| worker window | 773.640 ms | 770.017 ms |
| DRISC window | 775.034 ms | 771.497 ms |
| **DRISC window OPEN vs worker open** | **−0.007 ms** | **−0.145 ms** |
| per-DRISC opens (6 cores) | −0.007 … +0.131 ms | −0.145 … +0.048 ms |
| DRISC window CLOSE past worker close | +1.386 ms (0.18%) | +1.335 ms (0.17%) |
| filler plot containment | **99.3–99.4%** | **97.9–99.4%** |
| mover plot containment | **99.0%** | **99.0–99.1%** |
| CONTROL: Tracy's own CPU `Frame` plot | 11.0% | 9.6% |

Reading the residuals, because none of them is misalignment:

- **The close offset (+1.3–1.4 ms) is the §N+44 drain tail**, measured there as +1.294 ms on an 839.6 ms
  workload. Movers keep emptying the DRAM ring after the last worker zone; they are *supposed* to end late.
- **A DRISC window that opens a few tens of us BEFORE the first worker zone is correct.** The drainers are
  resident and sweeping before the workload starts, so `starts-inside=NO` on one context at −7 us / −145 us is
  physics, not skew. It is also the reason containment is 97.9–99.4% rather than 100%.
- **The control row is weaker here than in §N+43** (9.6–11.0% vs 0.0%) and that is an artifact of window ratios,
  not of the fix: `Frame` spans the whole 7,300 ms process at ~101 ms per sample, so ~10% of its samples land
  inside a 770 ms window by chance. §N+43's window was 1.9 ms of a 2,994 ms plot, where chance gives 0.

### No regression (4 reps each, NO `tracy-capture` attached — it starves the decoder and would change the timing)

| | base | + profiler + instrument |
|---|---|---|
| inference s, reps 2–4 mean | 0.0015536 | 0.0016415 |
| overhead | — | **+5.66%** |
| §N+45's own figure for this box | 0.0015495 | 0.0016338 → +5.44% |
| agreement with §N+45 | +0.27% | +0.47% |
| producer stalls | — | **0** of 110, all 4 reps |
| records dropped | — | **0**, all 4 reps (1.817–1.823 M taken) |
| ts regressions publish / consume | — | **0 / 0**, all 4 reps |
| `overflows region` / `FAILED TO LOAD` | none | **none** |

The per-core anchor costs 6 × 100 register reads inside device open (~2 ms of a 9–12 s open) and touches nothing
on the drain path, which is why the timing is indistinguishable from §N+45's pre-fix numbers.

**Losslessness re-checked on the synthetic harness too**, `--gx 12 --gy 10 --iters 500`, `ROLE_SPLIT` on, zones
off: **5,501,100 markers drained, 5,501,100 consumed, 0 dropped, 0 stalls, 0 regressions.**

**On the 6,001,200 target — it is unreachable on bh-05 for a PART-GEOMETRY reason, not a harness one.** 600 lanes
needs a 12-column compute grid. **p100a's `compute_with_storage_grid_size()` is 11x10**, so `--gx 12` clamps to
11 (the harness prints `dispatching 11x10`) and the ceiling is 550 lanes → 550 x 500 x 10 x 2 + 2 x 550 =
**5,501,100**, which is exactly what drained. An earlier attempt reached the same "unreachable" conclusion from
the harness's `gx=2, gy=2` DEFAULTS (`400 x iters + 40`), i.e. the right answer for the wrong reason — the grid
was explicitly passed here and the part is still the binding constraint. **6,001,200 is a p150/bh-26 number.**

### CONFIRMED: the fix does NOT regress bh-26, the card where the bug is invisible

A per-core anchor replaces a worker anchor on **every** part, including the one that never needed it, so the
obvious risk is making a healthy capture worse. Bounded analytically first — the two anchors differ by 3.6–5.5 us
on bh-26, so rows can move at most ~5 us on a multi-hundred-ms window — then measured. Synthetic harness,
`--gx 12 --gy 10 --iters 500` (clamps to 11x10 on p100a), zones + footprint on, under `tracy-capture`:

| | bh-26 WITH the fix |
|---|---|
| worker window | 242.139 ms |
| DRISC window open vs worker open | **−0.073 ms** (per-DRISC −0.073 … +0.040 ms) |
| DRISC close past worker close | +0.399 … +1.338 ms — the drain tail |
| **plot containment, all 9 series** | **99.7%** (fillers *and* movers) |
| CONTROL: Tracy's own CPU `Frame` plot | **2.0%** |
| producer stalls / dropped / ts regressions | **0 / 0 / 0** |

**Do NOT read the mover containment as an improvement over §N+43's 33–35%.** That figure came from a 1.9 ms
worker window against a 2.5–2.9 ms drain tail, where most mover samples legitimately fall outside; here the
window is 242 ms so the ~1.3 ms tail is negligible. Different run length, not a different quality. The claim
supported by this table is only the one that matters: **bh-26 still aligns, so the fix is safe where it is a
no-op.**

One artifact worth knowing when reading the new log line: on bh-26 the reported per-DRISC "offset vs worker
anchor" is a monotonic ladder, **+0.207 / 0.388 / 0.560 / 0.732 / 0.895 / 1.049 ms**. That is not a clock
difference — it is the **sequential measurement lag**, since each `sync_device_clock` is 100 reads (~0.36 ms) and
the six run in order after the worker's. It **cancels in the mapping**, because each anchor is a self-consistent
(host_anchor, device_at_anchor) pair measured together. On bh-05 the same ladder is buried under a 10–25 minute
offset and is invisible. So: **sub-millisecond ladder = this part is fine; minutes = the bug is present.**

Housekeeping: bh-26's `build_Release` was **entirely root-owned (5,038 files)** from `docker exec` pollution,
which made `ninja` fail with `Error writing to build log: Permission denied`. Repaired with
`sudo chown -R mmemarian:mmemarian build_Release`. The read-only probe earlier in this section was deliberately
compiled to `/tmp` to avoid touching the tree before that was understood.

### REJECTED: "just synchronise the DRAM and Tensix wall clocks"

The obvious next idea, and it is the wrong layer. Recorded so it is not re-litigated.

**Within a capture the two clocks are ALREADY in lockstep.** Both read exactly 1350.0 MHz in every probe on
every core, and the independent per-core least-squares fits agree to ~1e-4 (worker 1.350038 GHz, the six DRISCs
1.349902-1.350039). That bounds intra-run divergence to **~77 us over a 773 ms window** — the same order as the
0.007-0.145 ms window-edge agreement actually measured. The counters drift apart only **while nothing is being
timed**. Only the ORIGIN was ever wrong, and an anchor fixes an origin exactly.

Four reasons not to equalise the counters:

1. **It would not stay equal on bh-05.** Its Tensix clock is stopped ~91% of the time, so any equality decays at
   ~0.9 s per idle second — 18 s of error per 32 s of idle. It would have to be re-established at every device
   open, which is *exactly what the per-core anchor already does*, without needing write access.
2. **That register is a SHARED timebase, and writing it corrupts every other consumer**: worker
   `kernel_profiler`, the RT profiler, `cq_realtime_profiler`, `mem_bench`, `sync_kernel`, and the drainer FW's
   own timing all read it. Every reference in the tree is a READ; nothing writes it. A rebase mid-flight makes
   timestamps jump BACKWARDS and trips the invariant this system asserts on every run ("per-lane ts regressions
   MUST be 0").
3. **Writability is UNDOCUMENTED.** `RISCV_DEBUG_REG_WALL_CLOCK` does not appear in tt-isa-documentation at all —
   checked `BlackholeA0/TensixTile/BabyRISCV` and `WormholeB0/TensixTile/BabyRISCV`; the tile control/debug page
   404s. The tile debug register window `0xFFB1_2000-0xFFB1_2FFF` is documented, the counter's semantics are not.
   So a write is an experiment on an undocumented register that other subsystems depend on.
4. **The anchor is verified at both extremes** — bh-05 with the clocks minutes apart, bh-26 with them in lockstep
   (duty ratio 1.0000). A sync could only ever be validated on the card that is already anomalous.

**Two things the instinct gets RIGHT, kept as separate work items:**

- **Device-side comparability is a genuine gap.** Raw timestamps from a DRAM core and a Tensix core are not
  comparable outside the host's Tracy mapping, so on-device cross-core causality — a drainer reasoning about
  worker ordering — is not currently possible. The fix for that is to co-sample and subtract a **measured**
  offset (the shape-2 device-side rebase), not to write the counters.
- **bh-05's ~91%-stopped Tensix clock may itself be a FAULT, not a design property.** bh-26 with byte-identical
  firmware does not do it. If so the right remedy is a card/CMFW bug report, not a profiler change — and the
  per-core anchor is immune either way.

**The one condition that WOULD force a real sync**: a part that clock-gates one domain **mid-capture**. That
would appear as DRISC-vs-worker zone SPAN disagreement beyond the drain tail. It did not occur in any of the five
captures here (bh-05 x2 ResNet, bh-26 x1 synthetic, plus the two probe series) — the residual is fully accounted
for by the tail. **Span disagreement is the trigger to revisit this decision.**

### RESOLVED: a chip reset does NOT fix the gating — it is a property of the PART

The discriminating experiment, run on bh-05 on 2026-08-14: probe, `tt-smi -r`, then re-probe the **duty ratio**
(not the totals — the totals necessarily reset, which is the trap this measurement avoids).

| bh-05 | worker advance | DRAM advance | **duty ratio** |
|---|---|---|---|
| pre-reset | +1.82 s / 32 s | +19.75 s / 32 s | **0.090** |
| **post-reset, B→C** | +1.81 s / 32 s | +19.73 s / 32 s | **0.092** |
| **post-reset, C→D** | +1.82 s / 32 s | +19.74 s / 32 s | **0.092** |
| bh-26, for reference | +19.60 s / 34 s | +19.60 s / 34 s | **1.0000** |

**Unchanged across a chip reset.** So bh-05's Tensix wall clock being ~94% stopped is **a property of that
part**, not a recoverable state and not a consequence of its wedge/degradation history. Two same-SKU,
same-firmware p100a boards, one at 0.092 and one at 1.0000, and a reset does not move it. **This belongs in a
card/CMFW bug report; it is not a software defect and no profiler change can address it.**

Three results fall out of the same run:

1. **Both counters DO share a zero point at chip reset — now measured, not inferred.** 3 s after `tt-smi -r`
   they read **2.394 s and 2.555 s**. Every earlier statement about a shared origin rested on bh-26's equal
   totals plus the 9.367 s datum; this is direct.
2. **The offset regrows linearly at ~0.56 s per second of wall time** (`DRAM 0.617 of wall − worker 0.057 of
   wall`). Measured: **+161 ms → +18.08 s → +36.01 s** over the three probes.
3. **That kills the reset-discipline workaround quantitatively**, not just structurally. Earlier this section
   argued the gap "can never be smaller than the window" from device-open duration. Now with a rate: a device
   open takes 9-12 s, so by the time the workload starts the offset is already **5.0-6.7 s against a 0.77 s
   window — 7-9x too large**, for the tightest reset-then-run cadence physically available. There is no cadence
   that works.

Useful closed form for anyone reading a capture from this card: **offset ≈ 0.56 x (seconds since last chip
reset)**. Checked against the earlier runs — the 17:18 probe implies a reset at ~17:08, predicting 605 s of
offset at the 17:26 capture against 649.7 s measured (7% high, because the card was not uniformly idle and
activity changes both duty cycles).

Card healthy after the reset: a `driscz` ResNet rep gives 0.0016321 s, 0 stalls, 0 dropped, 0 ts regressions.

### A HOST REBOOT does not fix it either — chip reset AND warm host reboot both survived

`tt-smi -r` left the duty ratio unchanged; a **host reboot** is the stronger event (it re-runs PCIe enumeration
and CMFW init, neither of which a chip reset touches), so it was worth testing separately. Same 3-probe protocol,
run ~20 s after the container came back (host uptime 0 min):

| bh-05 | worker advance | DRAM advance | **duty ratio** |
|---|---|---|---|
| baseline | +1.82 s / 32 s | +19.75 s / 32 s | 0.090 |
| after `tt-smi -r` | +1.81 / +1.82 s | +19.73 / +19.74 s | 0.092 / 0.092 |
| **after HOST REBOOT** | **+1.84 / +1.91 s** | **+19.77 / +19.84 s** | **0.093 / 0.096** |
| bh-26, reference | +19.60 s / 34 s | +19.60 s / 34 s | 1.0000 |

**Unchanged across four independent measurements spanning a chip reset and a warm host reboot.** The card WAS
re-zeroed by the reboot (probe B read worker 23.6 s / DRAM 48.4 s, both small), so the counters restarted and the
ratio still came back at 0.09. The offset regrew at ~0.54 s/s as before (+24.8 → +42.7 → +60.6 s).

**Only a COLD power cycle remains untried**, and per §N+29/`drisc_pcie_hang_two_states` a warm reboot is
explicitly NOT equivalent to one for this card. So the honest status is: **not fixable by anything in-band**, and
a cold power cycle is the one lever left. Either way this is a card/CMFW matter and the profiler is immune.

**CAVEAT on what "duty ratio 0.09" means.** It is measured under a FIXED, idle-dominated protocol (three short
probe processes with 30 s of idle between them), and it is not a universal constant of the card: during host boot
the worker counter clearly ran much closer to full rate (at probe B it had banked 23.6 s, far more than 0.057 of
the elapsed wall). The Tensix clock runs at 1350.0 MHz *while Tensix is active* and is ~stopped while it is not,
so the ratio reflects the activity mix. It is a valid DISCRIMINATOR only because bh-26 was measured under the
identical protocol and returned 1.0000 — quote it with its protocol, never as a bare property.

Card healthy after the reboot: `driscz` ResNet rep **0.0016125 s**, 0 stalls, 0 dropped, 0 ts regressions.

**Reboot recovery gotcha (cost a `rc=127`):** `/home` is NOT persistent across a host reboot on this box, so
`/home/mmemarian/.local/share/uv/python/` vanished and `python_env/bin/python` dangled. The venv's
`site-packages` survive (they live under `/localdev`), so the repair is a one-line repoint to the system
interpreter of the same minor version: `ln -sfn /usr/bin/python3.10 python_env/bin/python` (3.10.12, same `cp310`
ABI — `import ttnn` then works). The C++ `--clkprobe` path needs no venv at all, which is why the clock
measurement above was unaffected.

## §N+47 — ONE SHARED FREQUENCY, measured on a long baseline: the per-core fits were BIASED, not noisy (bh-05 / **p100a**, 2026-08-14)

§N+46 gave every DRISC its own least-squares frequency fit and called that a virtue ("one self-consistent
fit"). It is the opposite. This section discards all six per-core rates, shares ONE frequency across every
context on the chip, and measures that frequency on a ~140x longer baseline. Per-DRISC spread on the
window-open metric falls **138 -> 24.5 us** and the worst offset **130.7 -> 17.7 us (7.4x)**, on a window that
is *longer* (774 -> 1042 ms), which if anything understates it.

> **READ THIS FIRST — the numbers in this section rest on an INVALID METRIC.** Every figure below compares a
> DRISC zone-window OPEN against the worker zone-window OPEN. Those are **two independent physical events**: a
> resident drainer's self-zone window arms when it first sees work, a worker's window opens when the workload
> starts. Their proximity is partly clock accuracy and partly coincidence, and the two are not separable from
> this measurement. A zone-window comparison **cannot validate an anchor** — it can only fail to detect a gross
> error. The 24.5 us is therefore an upper bound of unknown tightness, not a residual. §N+48 replaces it with a
> common-trigger sync event, where all six DRISCs mark the *same physical instant* and the spread in rendered
> timestamps is pure anchor + render error. Do not quote 24.5 us as "the anchor is good to 25 us".
>
> **NOW MEASURED (§N+48): the real per-core residual is 0.18 us** — 140x tighter than the 24.5 us below, which
> was almost entirely the old metric's own noise. And the "7.2x" this section claims is, on a valid metric, not
> a factor at all but the removal of an UNBOUNDED term: per-core rates drift apart at +72.3 ppm (218 us after
> 2.75 s, still growing), a shared rate at +0.2 ppm. Read §N+48 before quoting any figure from here.

### READ THIS SECOND: what changed is the SLOPE, not the ORIGIN — and confusing the two is the trap

Every device zone reaches the host as a raw **cycle count**. Placing it on the Tracy timeline is one conversion,
per core:

```
rendered_ns = (device_cycles - device_at_anchor) / frequency + host_anchor
                              \____ ORIGIN ____/   \_ SLOPE _/   \_ ORIGIN _/
```

So each core's mapping has exactly two parts: an **ORIGIN** — the pair (`host_anchor`, `device_at_anchor`),
meaning "at this host instant, this core's counter read X" — and a **SLOPE**, `frequency`, in cycles per ns.
`sync_device_clock()` produces both from one least-squares fit over 100 (host_time, device_cycles) samples: the
gradient is the slope, a point on the line is the origin.

- **§N+46 made the ORIGIN per-core, and that is KEPT.** It is what fixes the DRAM-vs-Tensix origin bug — the
  DRAM counter and the Tensix counter are zeroed by the same event but banked wildly different totals, so a
  DRAM core needs its own origin measured on itself.
- **§N+47 makes the SLOPE shared. That is the ONLY thing that changed.** Each core still measures its own origin
  on itself; only its own fitted rate is discarded in favour of one chip-wide value.

**Hence the trap: "per-core anchor" (kept, §N+46) and "per-core frequency" (discarded, §N+47) pull in opposite
directions.** They are different halves of the same conversion, and reading either name as "the per-core stuff"
will make one of the two sections look wrong.

**Why the slope is the dangerous half.** An origin error is a **constant shift** — the row sits N us to the left
for the whole capture. A slope error is multiplied by `(device_cycles - device_at_anchor)`, so it **grows with
time since the anchor**. Six cores rendered with six slightly different slopes therefore **drift apart** while
the hardware is in lockstep. The drift rate is just the difference between the slopes.

And those six slopes are not six rates — they are **six noisy estimates of one physical rate.** The counters
really do tick together: true rates agree to **~5 ppm**, while the fits scatter over **99 ppm on bh-05** and
**29 ppm on bh-26**. Two further facts rule out "each core simply has its own slightly different rate":

- the fits are **strongly biased low** — **52 of 54 negative** across 9 runs on 2 parts, mean **−33.3 ppm**
  (range −79.6 to +71.7). Not zero-mean, so averaging per-core fits cannot rescue them. **But not universally
  negative either**: 2 of 54 came out positive, so do not state this as "always negative" (an earlier draft of
  this section said "18 of 18" from the first three runs and that generalized too far).
- they are **unstable across device opens.** The per-run *range* of the six slopes — which is what actually sets
  how far the rows fan apart — varies **15.7 to 121.4 ppm** run to run on the same box. A genuine per-core rate
  would reproduce; a fit re-drawn at every device open does not.

**And fan-out is proportional to that range, to ~1%** — which is what closes the argument. Measured across all
four per-core runs:

| run | slope range | rendered fan-out at last trigger |
|---|---|---|
| bh-05 A | 72.1 ppm | 218.5 us |
| bh-05 B | 27.9 ppm | 85.3 us |
| bh-26 A | 29.5 ppm | 88.3 us |
| bh-26 B | **121.4 ppm** (the one mixed-sign run) | **365.9 us** |

Ratios within each box: 72.1/27.9 = 2.58 against 218.5/85.3 = 2.56; 121.4/29.5 = 4.11 against 365.9/88.3 = 4.14.
**This also retracts an "unexplained" note made earlier in this session** — bh-26's per-core fan-out jumping 4x
between two runs minutes apart is not unexplained variance, it is that run's slope range being 4.1x wider
(mixed signs widen it most). The drift you see is the spread of the estimates, nothing else.

**One shared slope makes differential drift zero BY CONSTRUCTION** — not "small", but arithmetically impossible,
since every row is scaled by the identical number.

**The honest caveat, stated plainly: a shared slope does NOT improve absolute rate accuracy.** If that one value
is wrong, every row is wrong by the same factor. What it removes is the *relative* error — and that is the only
kind that manifests as misalignment, because reading a timeline is entirely a matter of comparing rows against
each other. A uniform stretch of everything is invisible.

### The change, in two parts

1. **`spacing_us` on `sync_device_clock`**, and the worker sync now passes 500 us. 100 samples spread over
   ~50 ms instead of ~360 us of back-to-back MMIO round trips — a ~140x longer regression baseline for the
   slope. Cost: 50 ms of a 9-12 s device open.
2. **`sync.frequency` is used for every `AddCore`**, and each core's own `ds.frequency` is **discarded**. The
   anchor PAIR (`host_anchor`, `device_at_anchor`) is still measured per core on that core, so the per-core
   OFFSET that §N+46 exists to fix is untouched. Only the RATE is shared.

Note which call site got the long baseline, because the asymmetry is the whole story: the **worker** sync at
`perf_debug_profiler.cpp:835` passes `spacing_us=500`; the six **per-DRISC** syncs at `:898` still run at the
default 0. So the shared value is the well-measured one and the discarded values are the badly-measured ones.

### The evidence that the per-core fits were BIASED, which is what makes discarding them right

The log now prints each core's discarded fit as a deviation in ppm from the shared rate. All six, in log order
(DRISC 0..5 = `(9,9) (9,5) (9,2) (0,3) (0,0) (9,0)`):

| DRISC | 0 | 1 | 2 | 3 | 4 | 5 | mean | range |
|---|---|---|---|---|---|---|---|---|
| own fit vs shared, ppm | −13.6 | −51.0 | −37.0 | −57.0 | −40.2 | −43.0 | **−40.3** | 43.4 |

**Every one is negative.** That is the finding. Zero-mean fit noise produces a mix of signs; six negatives out
of six is a systematic bias in the short-baseline estimator (p = 1/32 under a fair-coin null, and the magnitudes
cluster far from zero besides). Two consequences, both of which invert the obvious remedies:

- **Averaging the per-core fits would NOT have helped.** The error is common to all six, so their mean carries
  it in full. Only lengthening the baseline removes it. Anyone reaching for "average the noisy fits" — which is
  where I would have gone first — would have kept ~40 ppm of bias and concluded the idea failed.
- **Nor would fitting each core more carefully.** The bias is a property of a ~360 us baseline against ~us of
  host-timestamp jitter, not of which core is being read.

Set against §N+46's own measurements of the two clocks' TRUE behaviour: the DRAM and Tensix counters' real rates
agree to **~5 ppm** during active operation (three 500 ms brackets: −4.6, +2.8, −5.8 ppm), while §N+46's
per-core fits scattered across **1.349902–1.350039 GHz = ~101 ppm**. So the fit noise was ~20x the physical
difference it was supposed to be measuring. Sharing one rate trades a ~101 ppm noise term for a ~5 ppm physical
one: **~3.8 us over a 757 ms window instead of ~75 us.**

### Why sharing a rate is correct, not a convenient approximation

**Alignment is a RELATIVE property.** What a reader needs is not that each row's rate is individually accurate
but that the rows agree with each other. With one shared rate:

- **Differential drift between rows is zero BY CONSTRUCTION.** No two contexts can drift apart, whatever the
  shared value is, because they scale identically.
- **Any error in the shared value is COMMON-MODE.** It shifts worker and DRISC rows together, which is
  invisible on a timeline — you cannot see a uniform stretch of everything.

A rate error is a *rate* error: it grows with time since the anchor. That is why it showed up as rows drifting
apart rather than as a constant skew, and it is why the ~1.3 s between the anchor and the workload turned
~40 ppm into tens of us of displacement. Measured in the previous session: each DRISC row was displaced 46-70 us
with the sign tracking its own fit error, **correlation 0.985** — the tell that identified the whole term.

### The measurement (re-derived from the captures, not transcribed)

The run logs from that session were lost with the container's `/tmp`, but the captures live under `/localdev`
and survived. Re-derived here with `tracy_zone_csv` + a Python window pass (`/tmp/win.py`); DRISC contexts are
0..5, identified by name via `tracy_ctx_inspect`:

| | run1 (BEFORE) | run2 (BEFORE) | sharedfreq (AFTER) |
|---|---|---|---|
| capture | `anchor_run1.tracy` | `anchor_run2.tracy` | `anchor_sharedfreq.tracy` |
| worker window | 773.640 ms | 770.017 ms | **1042.498 ms** |
| DRISC 0 `(9,9)` open | +130.7 us | +41.4 us | −5.7 us |
| DRISC 1 `(9,5)` open | +81.0 us | −145.0 us | −5.4 us |
| DRISC 2 `(9,2)` open | −7.2 us | +6.5 us | −6.8 us |
| DRISC 3 `(0,3)` open | +111.0 us | +18.1 us | +4.6 us |
| DRISC 4 `(0,0)` open | +86.2 us | +48.2 us | +0.5 us |
| DRISC 5 `(9,0)` open | +96.3 us | +34.3 us | +17.7 us |
| **open SPREAD (max−min)** | **138.0 us** | **193.2 us** | **24.5 us** |
| **worst \|offset\|** | **130.7 us** | **145.0 us** | **17.7 us** |
| stdev of offsets | 43.6 us | 66.6 us | **8.6 us** |

**The AFTER window is 35% LONGER (1042 vs 774 ms) and the offsets still collapsed.** Since a rate error grows
with elapsed time, a longer window is a harder test, so the ratio understates the improvement.

**Two arithmetic corrections to the figures this section was handed.** The improvement was reported as "7.2x,
spread 130 -> 25 us, worst |offset| 70 -> 18 us". Re-derivation gives spread **138.0 -> 24.5 us (5.6x)** and
worst |offset| **130.7 -> 17.7 us (7.4x)** — so the 7.2x is the *worst-offset* ratio, not the spread ratio, and
the two must not be quoted interchangeably. The "70 us" is a third quantity again: it is the 46-70 us
*frequency-attributable displacement* from the correlation analysis, not a measured window offset. Also note
the BEFORE case is unstable between its own two runs (mean offset +83.0 us vs +0.6 us, spread 138 vs 193), so a
single BEFORE run is not a reliable baseline — which is a further reason the window metric is weak.

### No regression

| | value |
|---|---|
| producer stalls | **0** |
| records dropped | **0** |
| ts regressions publish / consume | **0 / 0** |
| `overflows region` / `FAILED TO LOAD` | **none** |
| inference time | unchanged vs §N+46 |

Nothing on the drain path changed and no firmware changed; the whole diff is host-side anchor bookkeeping plus
50 ms of extra sampling inside device open.

### Scope limits, stated

- **The ~5 ppm true-rate agreement was measured on bh-05 only.** It is the justification for sharing a rate at
  all, and it has not been checked on bh-26 or any other part. A part whose two domains ran at genuinely
  different RATES (not merely different duty cycles, which is what §N+46 found) would have drift *injected* by
  a shared frequency rather than removed. The tell is DRISC-vs-worker zone SPAN disagreement growing with
  window length — the same trigger §N+46 records for revisiting the no-sync decision.
- **The DRAM counter's effective rate is not 1.35 GHz across an idle interval.** §N+46 measured it advancing
  ~19.75 s per 32 s of idle wall time, i.e. ~0.62 of the active rate. Zones are emitted during active work so
  the shared 1.35 GHz is right where it is used, but the ANCHOR-to-EVENT gap may contain idle time, and a fixed
  rate mis-scales that gap. This is a live residual, not a closed one, and it is one of the things §N+48's
  sync event is built to expose: fire a trigger near the anchor and again much later, and a rate/idle-scaling
  error appears as a residual that GROWS with time since the anchor.

## §N+48 — COMMON-TRIGGER SYNC EVENT: the anchor residual is 0.18 us, not 25 us (bh-05 / **p100a**, 2026-08-17)

Every alignment number through §N+47 compared a DRISC zone-window OPEN against a worker zone-window OPEN.
**Those are two independent physical events** — a resident drainer's self-zone window arms when it first sees
work, a worker's opens when the workload starts — so their proximity mixes clock accuracy with coincidence and
the two are not separable. A zone-window comparison can only fail to detect a gross error; it cannot validate
an anchor.

This section replaces it. All six drainers now mark the **same physical instant**, so the spread in their
rendered Tracy timestamps contains anchor + render error and nothing else.

**Result: the per-core residual is 0.18 us.** The 24.5 us of §N+47 was almost entirely the old metric's own
noise. The anchor is ~140x better than the best previous estimate of it.

### The mechanism: a rendezvous barrier, because poll granularity would have swamped the answer

A filler sweeps every ~157 us. A drainer that noticed the trigger only once per sweep would report a spread
dominated by SWEEP PHASE — up to 157 us of it — and would say nothing about a ~25 us residual. So the trigger
is a **barrier**, not a poll:

| step | who | what |
|---|---|---|
| 1 | host | write `req` = generation to all drainers |
| 2 | drainer | at the top of its next sweep, notice `req`, write `ack`, enter a **tight spin** |
| 3 | host | wait for EVERY `ack` — this is what guarantees the slowest one has stopped sweeping |
| 4 | host | write `go` to all — the release |
| 5 | drainer | ~5 instructions later, read its own clock and emit `DRISC-SYNC` |

Three words in the 64 B pad behind `stop`, of which only word 0 was ever used. **Step 3 is not optional**:
without it the host would release a core still mid-sweep and put the sweep-phase error straight back in.

**Observe-to-timestamp gap: 5 instructions** — `fence` (`invalidate_l1_cache` is a single fence on Blackhole),
the load of `go`, the not-taken branch, and the two latching register loads inside `get_timestamp()`. Measured
end to end as the emitted zone's own duration: **26-29 ns**. That is the systematic floor of the instrument,
three orders of magnitude under what it measures.

Carrier: `DRISC_ZONE_SYNC` (`0x7FF8`), a fixed id in the band `PROFILER_STALL_ZONE_ID` already uses, emitted
through the existing self-zone ring and shipped by `self_publish()`. **No new wire format, no host decoder
change, no new tool** — it comes out of `tracy_zone_csv` by name like any other zone.

### Trigger skew: measured, decomposed, and much smaller than the host thinks

The six drainers sit on scattered DRAM cores (`9-9, 9-5, 9-2, 0-3, 0-0, 9-0`), which is not a multicast
rectangle, so the release is six sequential unicast writes. Rather than assume a per-write cost, the release
**order is ROTATED by generation**. Across 12 generations (two full rotations of six) every core occupies every
position exactly twice, which turns the skew from an unknown into a measurement:

| | mean, us | stdev, us |
|---|---|---|
| **by RELEASE POSITION** (pos 0..5) | −1.24, −0.75, −0.25, +0.25, +0.74, +1.25 | 0.05-0.12 |
| **by CORE** (F0..M5) | +0.03, +0.07, +0.04, +0.03, −0.06, −0.11 | 0.83-0.87 |

- **Position spread 2.50 us** — a clean monotonic ladder at ~0.50 us per write. This is the trigger skew.
- **Core spread 0.18 us** — this is anchor + render error. Each core's larger stdev (0.83-0.87) is just the
  position ladder it inherits as its position rotates.

So within a single trigger, **essentially all of the observed spread is write-order skew** and the anchor
contributes ~180 ns. Raw per-trigger spread was 2.50 us mean (1.85-2.70 over 12), which the position ladder
accounts for entirely.

**CORRECTION worth carrying: the host-measured write span OVERESTIMATES on-device skew by ~5.5x.** The host
loop takes **13.76 us** to issue six writes, and I initially treated that as the skew — it would have been
comparable to the thing being measured. The device says the writes LAND 2.50 us apart. Host-side issue cost
(dynamic-TLB reconfigure per distinct target core) is not arrival skew, and only the device can tell you which
you have.

**And the ~2 us/write host cost is NOT card degradation.** That was my first explanation and it was wrong: the
same run's own probes read **ACK-WRITE 170 ns** and **DEVICE-READ 783 ns**, i.e. squarely the HEALTHY figures
(degraded is 2301 / 2935 — `degraded_state_is_mmio_latency`). The ack probe is fast because it reuses ONE
static TLB window on ONE core; the release writes hit six different cores through the dynamic path and pay a
reconfigure each. Same box, same run, 12x apart, for a reason that is about window reuse rather than card
state.

### A/B on a VALID metric: the shared-frequency change is not 7.2x, it is a change in KIND

> **The A/B knob has since been REMOVED.** `TT_METAL_PERF_DEBUG_PER_CORE_FREQ` existed only to produce the
> comparison below. Once it showed per-core to be strictly worse, keeping it would have been keeping a
> documented way to make captures worse, so it is gone and there is now exactly one slope. **To reproduce this
> table**, pass `ds.frequency` instead of `sync.frequency` to the `AddCore` call in `start()` — a one-line
> change at the site, which carries a comment saying so.

`TT_METAL_PERF_DEBUG_PER_CORE_FREQ=1` restored §N+46's per-core rates. Same harness, same 12 generations
250 ms apart, same box, minutes apart:

| | SHARED frequency (N+47) | PER-CORE frequency (N+46) |
|---|---|---|
| spread, first generation | 1.85 us | 20.11 us |
| spread, last generation (t = 2.75 s) | **2.63 us** | **218.48 us** |
| **core-mean spread** | **0.18 us** | **118.86 us** |
| **drift of spread vs time** | **+0.17 us/s (+0.2 ppm)** | **+72.26 us/s (+72.3 ppm)** |

**The per-core error GROWS LINEARLY and without bound; the shared one is flat at the trigger-skew floor.** That
is the whole argument of §N+47 made visible: a frequency error is a RATE error, so what it produces is drift,
and a shared rate makes differential drift zero by construction.

**Two independent measurements of the same quantity agree to 0.3%**, which is the check that makes this more
than a plausible story:

- the run log's own ppm column — each core's discarded fit vs the shared rate — spans **72.1 ppm**
  (−7.5, −79.6, −58.8, −70.3, −59.5, −73.4)
- the differential drift measured from rendered timestamps is **72.3 ppm**

**§N+47's "all six negative" reproduces in a fresh run**, on a different day, with different magnitudes: six of
six negative again. That is now two independent runs, and it is what rules out zero-mean fit noise.

So the honest restatement: **the 7.2x was measured on a metric that could not see the real effect.** On a valid
metric the improvement is not a factor, it is the removal of an unbounded term. At §N+47's ~1.3 s
anchor-to-workload distance the per-core error is ~94 us; over a 10 s run it would be ~720 us; the shared-rate
error at the same distances is ~0.2 us and does not grow.

### Configuration, and the code-space situation

`TT_METAL_PERF_DEBUG_SYNC_EVENT=<n>` fires n triggers, `..._GAP_MS` spaces them (default 250). Compile arg 38,
**default 0 — not one instruction is emitted when off**, and a `static_assert` requires zones to be on because
the event rides the self-zone ring.

**It fits, but with the NoC-footprint series OFF.** §N+43 left the instrumented filler with 36 B of headroom in
an 11,264 B region with zones + footprint both on, and this does not fit in 36 B. Footprint off frees that
space and the build loads clean (no `overflows region`, no `FAILED TO LOAD` in any run here). That is a
legitimate configuration rather than a dodge — this measures anchor accuracy, not NoC traffic — but **the two
instruments cannot currently be on together with the sync event**, and that is a real constraint on anyone
wanting plots and fiducials in one capture.

Firing happens at the **top of `stop()`**, which is load-bearing for three separate reasons:

- the workload is finished, so a parked drainer cannot starve a live producer — **0 producer stalls** in every
  run here, which a mid-workload barrier would have risked
- the drainers are still looping, so they can answer a rendezvous at all
- the host-side zone-name harvest is lazy and runs on the first drain; firing at bring-up would have run it
  before the workload's kernels were JIT-compiled and named every worker zone `Zone_<hash>`

### Integrity (every run in this section)

| | |
|---|---|
| sync events marked | **12 of 12 on all six drainers**, 3 runs |
| sync barrier timeouts | **0** |
| producer stalls | **0** |
| records dropped | **0** of 2,339,818 |
| self-zone words stranded | **0** |
| `overflows region` / `FAILED TO LOAD` | **none** |
| barrier park time | 5.7-135.2 us — i.e. within one sweep, as designed |

### NOT COVERED: the DRISC-vs-WORKER cross-domain offset

**This measures agreement AMONG the six DRISCs, all of which are DRAM-domain cores.** It does not measure the
DRISC-vs-worker alignment that §N+43/§N+46/§N+47 actually claim. That needs a core in the WORKER clock domain
reading ITS OWN wall clock at the same trigger instant, and nothing in this section provides one.

Two routes were examined and both are blocked:

- **A Tensix drainer** (`TT_METAL_PERF_DEBUG_DRAIN_TENSIX=1`) is a worker-domain core running the same self-zone
  code, so it looked free. It ran clean (12/12 marked, 0 timeouts) and is **useless for this**: both Tensix
  drainers' self-zones land in **one Tracy context named `Physical (0,0)`** — a wrong default identity — while
  their real coordinates `(16,2)`/`(16,3)` produce no context at all. A **pre-existing self-zone identity bug on
  the Tensix drain path**, unrelated to this work and not fixed here (that path is not production; the DRISC
  role split is). Anyone using Tensix-drainer self-zones for anything should know their identity is wrong.
- **A mixed roster** (one Tensix drainer alongside five DRAM ones) would give the cross-domain number in a
  single trigger with no new device code, but is not expressible: `rsplit = role_split() && !tensix_drain`, so
  the role split and the Tensix path are mutually exclusive by construction.

**The remaining route, and it is the right one:** have `fire_sync_events()` launch a tiny program on ONE worker
core that acks, spins on the same `go`, and opens a `DeviceZoneScopedN`. Its zone renders on the worker anchor
via the ordinary producer path, and the difference against the DRISC-SYNC zones IS the cross-domain alignment.
The blocker is picking a provably-safe L1 scratch word on a worker at teardown time — the profiler control
region is live and corrupting it breaks the SPSC protocol silently, which is the failure mode this file has
been burned by twice.

### ROOT CAUSE of both Tensix attempts: a TENSIX CANNOT ADDRESS A DRAM CORE'S L1 (bh-05, 2026-08-17)

The second attempt moved the Tensix work inside the sync-event **barrier** — drainers parked, nothing in flight,
no workload — on the theory that the first failure was contention with a live drain. **It wedged the card
anyway**, which refutes that theory and exposes something more basic.

Two facts from the same run, and it is their CONJUNCTION that identifies the cause:

| observation | count |
|---|---|
| clock handshakes **UNANSWERED** | **24 of 24** (4 generations x 6 drainers) |
| movers reporting `IMPOSSIBLE head reads` | **2** (DRISC 4 and DRISC 5) |
| `MMIO per-op timeout` | 440,441 us, then `rc=134` and a board at `0xffffffff` |

**Unanswered AND corrupting is the signature of MIS-ADDRESSING, not contention.** Contention would slow a
handshake or drop some; it would not make every single one fail while simultaneously scribbling on a third
party. The chain:

1. The Tensix NoC-reads a DRAM core at `stop_addr` — a DRISC-**local** L1 address.
2. From a **Tensix** requester that address does not terminate at the DRAM core's L1; it forwards to GDDR. This
   is the same NOC2AXI-vs-stream-mode forwarding behaviour §N+24 already documents in the other direction, where
   a released NIU made a filler's `head` read return GDDR contents (`0xF5AE93CB`).
3. So the read returns effectively random data. Random data is almost always non-zero and greater than the
   `served[]` watermark, so the responder **believes it has seen a request** and issues its write-back.
4. That write mis-delivers into the DRAM core's memory and lands on the filler's handshake block — which is
   exactly what `IMPOSSIBLE head read` exists to detect.

One cause, both symptoms: it never answered (its writes never reached the real `ack` word) *and* it corrupted
the fillers (its writes reached something else).

**THE THING THAT MADE THIS MISLEADING, and the lesson worth keeping: a MOVER reads a FILLER's L1 with exactly
these addresses, every sweep, reliably.** That is DRAM->DRAM. It does **not** generalise to Tensix->DRAM, and
because the NoC **MIS-DELIVERS RATHER THAN FAULTS** there is no error to notice — the same silent-failure hazard
that cost two debug cycles in §N+40 ("the NoC MIS-DELIVERS rather than rejects: one substituted word per frame
boundary, totals still perfect, 440/600 lanes desynced"). **Do not infer a NoC path from a working path with a
different requester type.** Prove it with a read-only magic before any write.

**ANSWERED, and it was not the obstacle anyone expected: THE FILLER ELF FITS.** With the handshake compiled in
on **all six roles**, the run log shows **zero** `overflows region` and **zero** `FAILED TO LOAD`. Code space was
flagged as the most likely blocker — §N+43 left the instrumented filler 36 B under an 11,264 B region — and it
simply was not one. The mover-only fallback (`TT_METAL_PERF_DEBUG_CLK_SYNC_ROLES=1`) is therefore unnecessary,
though it stays available. Two wedges were spent discovering that the binding constraint was NoC addressing, not
bytes.

**Cost, recorded honestly: two card wedges** (`Read 0xffffffff over PCIe ID 0`), each needing a `tt-smi -r`,
for a measurement that still does not exist. Both were the same underlying mistake — assuming a Tensix could
reach a DRAM core's L1 — approached from two different timings.

### The inverted design: DRISC pokes TENSIX (the direction that is already proven)

Chosen because **DRAM->Tensix NoC writes are the drain path's normal traffic**: it is how a mover reaches a
filler's L1 and how every frame reaches the PCIe tile. Hunting the correct Tensix->DRAM translation is rejected
for now — it has wedged the card twice, and "cheap to try" stopped being true after the second one.

The remaining problem is **address discovery, not ownership.** A `static` array in the Tensix kernel's `.bss` is
worker L1 owned by nothing else (that part is solved), but the DRISC has to be *told* where it is. Plumbing
options, and the non-negotiable precondition:

- have the responder publish its scratch address somewhere the host already reads, and the host forward it to
  the drain kernel; or pass it as a compile arg once the host learns it from the ELF;
- **and PROVE THE ADDRESS ROUND-TRIPS READ-ONLY BEFORE ANY WRITE.** A read that returns an expected magic costs
  nothing. A mis-addressed write is what corrupted the fillers, twice.

**What the DRISC-only result does license**, stated carefully: the anchor METHOD is reproducible to 0.18 us
across six independently measured cores, and the worker anchor is produced by the identical routine against the
same host clock. That is grounds for expecting the worker anchor to be comparably good — **but it is inference,
not measurement**, and the cross-domain offset remains unmeasured.

**Do not use zone-SPAN ratios as the substitute.** Tried here and it is a blunt instrument: in this capture the
DRISC span/worker span ratio implies +7,300 to +7,900 ppm for the fillers, all of it real drain tail rather than
rate error, and the movers read +28,000,000 ppm because the 3 s of sync generations legitimately extend their
activity past the workload. §N+46's "spans agree to 0.15-0.17%" is ~1,600 ppm of resolution — 4 orders coarser
than the 0.2 ppm this sync event resolves. Span agreement can only ever catch a gross fault.

### CONFIRMED ON bh-26, and it produces a result bh-05 could not: the two defects are INDEPENDENT

bh-26 is the part where §N+46's anchor bug is **invisible** (duty ratio 1.0000 against bh-05's 0.09), so it is
the natural non-regression check. Same code, same 12 generations 250 ms apart, healthy card (ACK-WRITE 180 ns,
DEVICE-READ 780 ns), 12/12 marked on all six, 0 timeouts, 0 stalls.

| | bh-05 | bh-26 |
|---|---|---|
| **SHARED freq — core-mean spread** | **0.18 us** | **0.20 us** |
| SHARED freq — drift | +0.2 ppm | **+0.1 ppm** |
| SHARED freq — per-trigger spread | 1.85-2.70 us | 1.93-2.90 us |
| PER-CORE freq — core-mean spread | 118.86 us | **48.73 us** |
| PER-CORE freq — drift | +72.3 ppm | **+29.2 ppm** |
| PER-CORE freq — spread, first → last gen | 20.11 → 218.48 us | 7.32 → 88.30 us |
| log's own ppm fit scatter | 72.1 ppm | **29.5 ppm** |
| **drift vs fit scatter agreement** | **0.3%** | **1.0%** |

**Non-regression: 0.18 us vs 0.20 us.** The anchor is equally good on both parts and both sit at the
trigger-skew floor, with the same position ladder. Nothing about the sync event or the shared frequency is
specific to the anomalous board.

**THE NEW RESULT — the frequency defect and the clock-domain defect are INDEPENDENT.** bh-26 never needed
§N+46's per-core *anchor* (its two domains share an origin to within microseconds), yet per-core *frequency*
still costs it **48.73 us and 29.2 ppm of drift**. So §N+47's change is **not** a follow-on repair for
bh-05's sick clock domain — it is a defect of the short-baseline fit itself, and it degrades a perfectly
healthy part. The two fixes address different things and both are needed everywhere. Anyone tempted to gate the
shared frequency on "is this board anomalous" would be reintroducing 29 ppm on the good boards.

**And the bias reproduces CROSS-BOX: all six fits negative on bh-26 too** (−50.2, −26.8, −20.7, −38.3, −27.2,
−29.7 ppm).

> **CORRECTED with a fuller tally.** This section first said "18 of 18 negative, zero-mean fit noise excluded
> conclusively" from three runs. Across **all 9 runs on both parts — 54 fits — it is 52 negative, 2 positive,
> mean −33.3 ppm.** The bias is real and large (so averaging per-core fits still cannot rescue them, which is
> the load-bearing point), but it is **not universal**, and "always negative" overstated it. Positive fits do
> occur, and the one run containing them is also the run with the widest slope range and the largest fan-out —
> see the range-vs-fan-out table in §N+47.

The drift-vs-fit-scatter agreement holding on both boards (0.3% and 1.0%, at two quite different magnitudes,
72 vs 29 ppm) is the strongest single check in this section: two independent instruments — the host's own
least-squares residual and the rendered device timestamps — measuring one physical quantity and agreeing twice.

### Method notes

- **Compute windows in Python.** Reconfirmed: `awk` overflows at INT32_MAX on these nanosecond values and a
  previous agent got `2147483647` from it.
- **`&` binds looser than `&&` in the shell.** `cd X && ... nohup A & B` runs B in the ORIGINAL cwd, and the
  symptom is `timeout: failed to run command './build_Release/...': No such file or directory` with the binary
  plainly present and `ldd` clean. Cost one run; the fix is a script, which is what the runs here use.
- Analysis: `tracy_zone_csv` for the zones, `tracy_ctx_inspect` to map context index to core name (both tools
  iterate `GetGpuData()` in the same order, so the index joins them), then Python. Captures
  `tracy_captures/sync_ev12.tracy`, `sync_percore.tracy`, `sync_tensix.tracy`.

## §N+49 — THE CROSS-DOMAIN NUMBER, and why in-capture alignment holds while between-capture offsets reach minutes (bh-05 / **p100a**, 2026-08-17)

Measured on a **verified-healthy** card (ACK-WRITE 168-174 ns, DEVICE-READ 781 ns) after a **clean** host reboot,
with the device-side DRISC<->TENSIX handshake of §N+48. This is the number §N+43/§N+46/§N+47 all needed and none
of them had.

### Phase 1: the card, and a correction about what recovers it

| | ACK-WRITE | DEVICE-READ |
|---|---|---|
| after the WEDGE + `tt-smi -r` | 2306-2327 ns | 2935-2956 ns |
| after a **CLEAN** host reboot | **174 ns** | **781 ns** |

**A clean warm host reboot DOES recover the degraded state** — `degraded_state_is_mmio_latency` was right
(2548 -> 386 ns) and `drisc_pcie_hang_two_states`'s "needs a COLD power cycle" is too pessimistic for this case.

**But the intervening reboot did NOT recover it, and the reason matters: that one was a FREEZE, not a reboot.**
`last -x` showed the 12:00 boot with **no preceding `shutdown` record**, while the 12:31 boot I issued has one.
So the sequence was: my second wedge **froze the host**, the IRD watchdog rebooted it, and the card came back
degraded. A probe taken after a watchdog reboot reads degraded; after a clean `sudo reboot` it reads healthy.
**Check `last -x` for the shutdown record before concluding a reboot failed to fix anything.**

This also escalates the wedge's blast radius, matching `x280_vm_freeze_churn`: **the Tensix->DRAM mis-addressing
bug does not merely wedge the card, it can take the host down.**

### Phase 2: the instantaneous offset, with its bound

72 of 72 handshakes (6 drainers x 12 generations), 0 timed out, **0 refused by the read-only probe**, 0
`IMPOSSIBLE head read`, 0 MMIO timeouts.

| | value |
|---|---|
| **DRISC − TENSIX offset** | **+47,127.640 ms** (47.1 s) |
| **round-trip bound `(t3−t1)/2`** | **0.197-0.294 us** |
| spread of that offset across the six DRISCs | **~2 us, constant** |
| card was reset ~90 s before | 47.1 s of divergence in ~90 s = **~0.52 of wall** |

That last row is a strong consistency check: §N+46 predicted the offset regrows at **~0.56 s per second of wall**
from an independent method (two host probes 32 s apart), and this device-side handshake independently gives
~0.52. Two unrelated instruments, one quantity.

The ~2 us spread across the six DRISCs is what it should be: they share a clock domain, so their offsets to a
Tensix core must agree, and they do — at the same order as §N+48's 0.18 us anchor residual plus the read-back
granularity.

### Phase 2b: THE RESULT THAT RECONCILES EVERYTHING — divergence needs Tensix IDLENESS

The offset **did not move at all** across 12 generations spanning 2.75 s: constant at 47,127.640 ms, i.e. drift
**< 1 us over 2.75 s (< 0.4 ppm)**. Then a deliberate long-gap run, 30 s between handshakes:

| | offset |
|---|---|
| gen 1 | +104,121.717 ms |
| gen 2, **+30 s later** | +104,121.720 ms |
| **change** | **+0.003 ms = 3 us over 30 s = 0.1 ppm** |

**§N+46 predicts ~0.56 x 30 s = ~16,800 ms of divergence over that interval. Observed: 0.003 ms — 5.6 MILLION
times smaller.**

The mechanism, and it is the point of this section: **the Tensix wall clock advances only while Tensix is out of
reset** (§N+46). The responder kernel is resident and spinning throughout, so the Tensix duty cycle here is ~1.0
BY CONSTRUCTION — and with both domains active the two counters are locked to ~0.1 ppm. The minutes-scale offsets
are accumulated **entirely during Tensix idleness**, not continuously.

Three things fall out, and together they close the loop this whole thread has been circling:

1. **In-capture alignment holds because a capture is, by definition, a period of Tensix activity.** During a
   workload both domains tick at ~1.35 GHz and their differential rate is ~0.1-5 ppm. That is why §N+48's
   0.18 us residual is achievable at all, and it is now explained rather than merely observed.
2. **The minutes-scale offsets live in the GAPS between captures**, which is exactly what an ORIGIN (the per-core
   anchor, §N+46) is for. An anchor measured at device open absorbs however much divergence accumulated while
   the part was idle; it does not need to track anything during the run.
3. **It retires the "shared frequency might inject drift" residual risk** that §N+47 recorded. That risk was
   conditional on the two domains running at genuinely different RATES. Measured directly here, cross-domain,
   they do not: 0.1 ppm over 30 s of activity. The duty-cycle difference is real and the RATE difference is not.

**Stated limit:** this measures the domains while a kernel is resident on each. It says nothing about the rate
during idleness — by construction, since observing requires running. The idle behaviour remains inferred from
§N+46's counter totals.

### Phase 3 (characterising the wedge): NOT ATTEMPTED

Deliberately not started. The card is currently healthy and that is the perishable resource — §N+48's numbers
and this section's both required it, and reproducing the wedge is expected to cost it again (twice now, once
taking the host with it). Two Phase 3 questions were nonetheless answered incidentally and are recorded above:
the degradation follows the **freeze**, and a **clean** warm reboot recovers it.

**Side effect worth knowing before the next session: bh-26 is currently BROKEN for profiled runs.** Mutagen
synced the new drain-kernel source (compile args 40/41) to it while its `libtt_metal.so` is stale and passes
only 39, so the JIT kernel fails with `DRISC 0 FAILED TO LOAD` and the capture is empty while the run exits 0.
**Rebuild bh-26 before using it.** This is the general hazard of a shared clone with per-box builds: source syncs,
binaries do not.

## §N+50 — e2e overhead on ResNet-50 **BATCH 32**: +2.64% (bh-05 / p100a, 2026-08-17)

The first overhead figure for this configuration measured as a proper base/profiled pair on one card in one
session. Card verified healthy **before and after** (ACK-WRITE 174 -> 190 ns, DEVICE-READ 781 -> 915 ns; degraded
is 2300+/2900+), so no part of the comparison was taken on a degraded card.

Test `test_perf_e2e_resnet50_batch32.py::test_perf_trace_2cqs` id `[32-0.003-30-device_params0]`, 4 runs per
config, **rep 1 discarded at every config**, **NO `tracy-capture` attached** (it starves the decoder and would
change the workload being timed).

| config | reps 1-4, ms | warm mean (2-4) | stdev |
|---|---|---|---|
| base (no profiler flags) | 2.3961 / 2.3957 / 2.3907 / 2.3945 | **2.3936 ms** | 2.16 us |
| profiled (drain + zones + NoC footprint) | 2.4664 / 2.4490 / 2.4600 / 2.4614 | **2.4568 ms** | 5.56 us |
| | | **+2.64%** | **+63.1 us** |

**This is a BATCH-32 number and must not be compared with §N+44/§N+45/§N+46's batch-16 figures** (~0.00163 s base,
+4.6% to +5.66%). Different batch, different test file, different absolute times. The apparent improvement from
~5% to ~2.6% is NOT a regression win — it is a larger workload amortising a roughly fixed per-op producer cost,
which is the same reason §N+45's LLM-decode regime showed the opposite.

Profiled variance is 2.6x the base variance (5.56 vs 2.16 us), consistent with the producers' emit cost being
data-dependent rather than a constant tax.

**Rep 1 was only +0.10% (base) / +0.39% (profiled) above warm here**, i.e. the discard mattered less than usual —
because these runs followed an earlier ResNet run that had already refilled the HF and ttnn caches. On the FIRST
run after a host reboot the tax is much larger, since `/home` is not persistent on this box and both caches are
empty (measured: `~/.cache/huggingface` = 56K, two months stale). **Discard rep 1 regardless**; it is cheap
insurance and its size is not predictable.

## §N+51 — Receiver v2 (zero-copy decode into per-stream rings) is lossless and count-exact; the delay-15 stall floor is DEVICE-side (bh-18, 2026-08-19)

The host receiver was reimplemented (commits 433ab8ce367..b89c0d5491c): per-socket threads fuse poll +
in-place decode straight off the D2H FIFO (`peek`/`pop`, no staging copy) + decode-paced acks; records are
emitted directly into per-stream BroadcastRings (direct-emit reserve/store/commit, NT 16 B slot stores);
one consumer thread per registered callback; the drain kernel sweeps-to-empty on stop=1; a teardown
completeness verifier compares every worker lane's tail against the receiver's consumed mirror.

**Losslessness is now verified per run, to the marker.** At gx12 gy10 iters10k, delivered zones equal
offered + stall-zone pairs + FW wrappers exactly (e.g. 60,362,777 = 60,000,000 + 362,177 + 600), and the
decoded stall-zone count independently matches the L1 stall counters to the digit. COMPLETENESS reports
600/600 lanes, 0 words stranded on every clean run; `WRITER_DIE_AFTER` (host killed mid-stream) leaves no
wedge and the verifier reports the stranded capture loudly (600 lanes, 43.7M words).

**The stall series that locates the floor** (gx12 gy10 iters10k delay15, L1 counters):

| host configuration | producer stalls |
|---|---|
| old receiver (memcpy-rate acks, ~26.7M records discarded host-side) | 352,442 |
| receiver v2 (decode-paced acks, lossless) | 361,279–361,443 |
| NO_DECODE (pop+ack only through the receiver) | 291,265 |
| bare `discard_pending_pages()` loop (no receiver machinery at all) | **289,657** |

Two independent maximal-ack hosts agree at ~290k, so that is a device-side floor: in those runs mover
credit-waits are 0.1 us and filler ring-room waits ~15 us — nothing downstream is pushing back — yet all
120 cores stall. The mechanism is the filler scan cadence itself: at delay 15 a core's 512-word ring
refills on the same timescale its filler revisits it among 30 cores. Consistent with that mechanism,
stalls RISE to 380,605 at delay 50 (lower span fill = more device sweep bytes per marker). Zero stalls at
delay 15 full grid therefore needs device-side changes (scan rate, ring depth, more fillers), not host ones.

The lossless receiver adds ~70k stalls over the floor because acks pace at decode speed: decode sustains
5.6–5.8 GB/s aggregate (SUSTAINED busy, 307–319 Mzones/s; 7.46 GB/s at delay 50) against ~7.4 GB/s offered.
Decode optimization rounds: NT slot stores + live-window prefetch were real (5.31 -> 5.81 GB/s); hoisting
the per-record stats out of the aliased Stream fields (perf attributed ~20% to a per-marker load-add-store
forced by the NT stores' pointer casts) measured within run-to-run noise — the remaining decode time is
cold-line latency on the DMA-fresh wire.

## §N+52 — --delay calibration: 1 unit = 10.00 cycles = 7.41 ns; 2 GiB rings retain a whole capture; ring hugepages are a wash (bh-18, 2026-08-19)

**Delay conversion, measured.** 2x2 x 2000-iter runs (0 stalls, unthrottled) at delay 15/500/2000 give device
zone windows of 4.0/75.8/298.0 ms. The slope is 7.407 ns per delay unit = **exactly 10.00 cycles at 1.35 GHz**,
intercept **86.7 ns fixed cost per zone**. So `zone body ~= 87 ns + 7.41 ns x delay`:

| --delay | ns/zone | unthrottled Mmarkers/s/lane |
|---|---|---|
| 0 | ~87 | 23.1 |
| 15 | 198 | 10.1 |
| 50 | 457 | 4.4 |
| 500 | 3,790 | 0.53 |
| 2000 | 14,900 | 0.13 |

At delay 15 the unthrottled offer is ~10M markers/s/lane x 600 lanes ~= 48 GB/s of wire — 6x any drain path —
so the 290-365k stalls there are the lossless throttle working, not a removable inefficiency. The harness now
prints the conversion.

**Ring sizing.** The per-stream record-ring default is now 2 GiB (128 Mi x 16 B records): a full judged
capture (120.7M records) fits, so the Tracy consumer delivered **120,720,976 records with 0 drops** (was
53.6M dropped at 512 MiB). SUSTAINED busy 6.36 GB/s with plain pages.

**Hugepages (mmap + MADV_HUGEPAGE for slot arrays >= 64 MiB; THP is madvise-only on this box): no measurable
effect** — decode threads 171.5/195.2 ms vs plain 172.5/172.7 ms, inside the ±10% per-thread spread observed
across ten runs (165-207 ms). Coherent: the ring is written once, sequentially, with NT stores, while the
decode wall is cold reads from the socket's pinned FIFO, which ring pages cannot touch. Kept for the page-table
footprint (2 GiB of 4 KiB PTEs is ~4 MB per ring).

## §N+53 — The D2H ladder: PCIe writes ~37 GB/s, pinned-FIFO reads 27 GB/s, DECODE 5.6-6.4 GB/s -- the wall is interpretation, not the pipe (bh-18, 2026-08-19)

Three probes on the judged config (gx12 gy10 iters10k delay15), one layer each:

| layer | probe | measured |
|---|---|---|
| device -> host PCIe write | SHIP_REPEAT=8 + NO_DECODE (17.4 GB shipped) | **36.8 GB/s** per mover during the write chunk (74 KB / 2.01 us); 29.1 GB/s per push incl. push_pages + notify |
| host read of the pinned FIFO | READ_ONLY=1 (peek + line-stride sum + pop) | **27.0 GB/s** aggregate over 2 threads (568 MB / 41.9 ms each) |
| decode (interpret + emit records) | normal run | **5.6-6.4 GB/s** aggregate (~2.9-3.2 GB/s/thread) |

So "D2H throughput" as the receiver reports it is a decode-compute number: the link and the memory both do
tens of GB/s today. Per thread, the same bytes read raw at 13.6 GB/s and decode at ~2.9 -- the walk spends
~4.7x the raw-read time on interpretation and record emission, ~12-14 cycles per 8 B record (2 masked loads,
type dispatch, ts assembly, meta pack, NT store, order check).

Comparison point: the tensor prefetcher's 21-23 GB/s single-RISC DRAM reads come from deep batching --
enormous bytes moved per instruction issued. The wire here is 8 B markers decoded scalar, one at a time.
The equivalent lever is a vectorized fast path for the dominant case (a full span's run is almost entirely
contiguous 2-word zone markers): AVX2-check 8 packets' types at once, build 8 records with shuffles, store
128 B NT -- an expected 3-6x per-record, putting decode in the 15-30 GB/s range where the FIFO read (27)
and PCIe write (29-37) layers sit.

READ_ONLY's stall count (307,897) slots between the max-ack floor (~290k) and decode-paced (~361k),
consistent with the layer costs above.

## §N+54 — AVX2 zone-block decode: 2.1x, SUSTAINED busy 5.8 -> 12.2 GB/s D2H, 667 Mzones/s (bh-18, 2026-08-19)

The scalar walk's ~12-14 cycles per 8 B marker (§N+53) was the D2H wall. The decode SSoT now screens each
16-word block with one AVX2 shift+compare over the deinterleaved word0s; when all eight packets are plain
zone markers -- the dominant case in a full span -- the receiver's vectorized sink builds all eight records
in registers (timestamps th<<32|w1 via cvtepu32, meta by masked shift/or with the bit-field layout pinned by
a TT_FATAL bit_cast probe, order/stall/min-max stats vectorized) and issues eight 16 B NT stores. Any other
type in the block falls back to the scalar walk.

Judged run: decode 189.5 -> 89.6/90.5 ms per thread; SUSTAINED busy 12.19 GB/s D2H / 10.67 GB/s marker-wire
/ 666.9 Mzones/s. Capture stays count-exact (zones == 60M + 352,217 stall-pairs + 600 wrappers, matching the
L1 stall counters exactly), 0 order regressions, 0 consumer drops. Producer stalls 361k -> 352,217,
converging toward the ~290k device floor (§N+51). Per-thread decode is now 6.2 GB/s against the 13.6 GB/s
raw-read ceiling (§N+53), so roughly another 2x of headroom remains in the scalar residue (screen misses,
sub-16-word tails, per-block overheads) before reads become the wall.

## §N+55 — Decode is memory-LATENCY bound: ALU trims are wall-time-neutral, two-pass scanning is a regression (bh-18, 2026-08-19)

Three structural variants of the AVX2 zone-block path, measured on the judged run:

| variant | decode ms (2 threads) | busy GB/s |
|---|---|---|
| v2: per-block screen + cvtepu assembly + full order chain | 89.6 / 90.5 | 12.19 |
| v3: scan-then-emit stretches (screen pass + emit pass) | 92.9 / **120.9** | 9.11 |
| v4: per-block screen + endpoint order check + unpack assembly | 88.2-91.8 / 92.7-99.0 | 11.2-11.9 |

v3's second touch of every block (reload + re-shuffle, even from L1) cost more than its branchless emit
stretch saved -- REVERTED. v4 deletes ~14 uops/block versus v2 (endpoint-only order compare; records built
by 32-bit interleaves against splatted th/prog then 64-bit pairing, no widening) and measures WITHIN NOISE
of v2 -- kept for the smaller loop, but the lesson is that per-thread decode (~6 GB/s vs the 13.6 GB/s
raw-read ceiling) is bound by dependent access latency on cold lines, not by instruction count. §N+54's
"~2x headroom in the scalar residue" is refuted: the residue is latency.

Remaining rungs, in order of expected value: cross-frame latency overlap (prefetch frame N+1's live
windows while decoding frame N), AVX-512 (Zen 4 present; halves uop count but double-pumped and the path
is latency-bound, so expect modest), decode fan-out via frame sharding (unattractive on this 6-core part:
2 decode + consumer + workload + Tracy client already fill it), and producer-side wire changes (the big
lever, out of receiver scope). Context: at delay 15 the host already drains within 1.3x of the device's
offered rate; host-attributable stalls are ~60k of ~352k.

## §N+56 — Cross-frame window prefetch REGRESSES 30-40%: bulk cold-line prefetch starves demand loads (bh-18, 2026-08-19)

Software-pipelining the prefetch (while decoding frame f: blind-prefetch f+2's ctrl lines; read f+1's
head/tail fields and prefetch its full live windows) measured 121-129 ms decode on BOTH threads across two
judged runs, versus 88-99 ms with the per-lane just-in-time prefetch -- reverted. Mechanism: a full-fill
frame's windows are ~160 cold lines; issuing them as one burst saturates the core's line-fill buffers, so
the CURRENT frame's demand loads queue behind speculative prefetches. The per-lane form (<= 32 lines,
consumed immediately) fits the LFB budget, which is why it measured +9% when introduced (§N+54) while the
"better" pipelined form loses.

Combined with §N+55 (ALU trims wall-time-neutral), the single-thread decode ceiling on this part is set by
DMA-cold DRAM latency at LFB-limited concurrency: ~6 GB/s/thread decoding vs 13.6 GB/s raw-streaming the
same bytes. AVX-512 (present on this Zen 4) is predicted neutral by the same evidence -- it halves
instruction count, which §N+55 showed does not move wall time, and 512-bit ops are double-pumped here so
load-port pressure is unchanged. Not built; the dispatch + code duplication is complexity with no predicted
return. The remaining single-thread levers are wire-side (producer-packed records / higher span fill), not
host-side.

## §N+57 — AVX-512 emit measured: neutral-to-worse, as N+56 predicted; not kept (bh-18, 2026-08-19)

Built for real (Zen 4 has avx512f/bw/dq/vl): a target-attributed emit path taking 16 markers (32
screen-verified words) per call, with `vpermt2d` collapsing the whole shuffle network -- one pair of
permutes deinterleaves the block, one pairs (w1, meta), and one per 4 records expands the pairs against
the splatted (th, prog) constants into finished 64 B record lines, stored with 64 B NT streams when
position-aligned. CPUID-dispatched (base build is v3); correctness gate exact (10,020 zones, 0 order
regressions, 0 stranded).

Measured on the judged run, two repeats: decode 106.9/96.9 and 93.9/93.8 ms (10.32 / 11.76 GB/s busy)
against the AVX2-only 88-99 ms band (12.1 GB/s) -- inside noise, leaning worse (the extra call boundary
per block is not free and the saved uops were not the constraint). Confirms N+55/N+56: the single-thread
ceiling is DMA-cold line latency at fill-buffer concurrency, and no instruction-count reduction moves it.
The code is NOT kept (measured-neutral complexity); this entry preserves the construction should a
latency-relieved future (warm staging via a copier thread, or a denser wire) make ALU width matter again.

## §N+58 — Live-extent packed frames: NIU-gathered packing at the socket push; 3.0x fewer bytes at 29% fill, exact everywhere (bh-18, 2026-08-19)

The wire moved to the format profiler_common.h always documented: per frame, prefix + control vector +
each RISC's live window `[tail - run, tail)` packed flat, `w[1]` = the real payload length, frames
variable-length and page-rounded. The 45%-of-drainer CPU copy that killed exact packing originally is NOT
reintroduced: ship_once parses the staged control vector (~20 L1 loads/frame), patches the length word,
and the NIU gathers the windows straight from staging — ~11 NoC writes per frame instead of one burst per
batch. Packing happens ONLY at the socket push: fillers, the DRAM-ring protocol and the scan are
byte-identical, so the DRAM slots stay raw and the mover packs blind from the frame's own ctrl.

The part the proposal underestimated: NoC L1->PCIe writes require dst ≡ src (mod 16 B,
NOC_PCIE_WRITE_ALIGNMENT_BYTES; the NoC mis-delivers rather than rejects). Each live run therefore
carries 0-3 pad words bringing the wire offset to its ring phase — spsc_span_pack_pad() in
profiler_common.h, the single rule both sides compute; the wrap continuation is congruent for free
(512 ≡ 0 mod 4). Worst-case frame is now 2,656 words (10,624 B, 166 pages) — BIGGER than raw when all
five rings are full. Pad and page-tail words are never written; the host walks past them.

Host: decode_pass parses the peeked window header-by-header (partial tail left unpopped — the device
only pushes whole frames), acks by actual pages, and the per-lane walk went circular->linear: ring mask,
wrap-stitch scratch and the AVX2 ring-boundary bailout are all deleted. The decode returns the
ctrl-implied payload and the receiver checks it against w[1], so a pack-rule disagreement is a counted
bad frame, not silent corruption.

Measured (judged delay-15, 12x10 x10k):
  - Correctness exact on every gate: 2x2 20,040 records; role-split, full-role, self-zones; judged runs
    0 bad frames / 0 anomalies / 0 resyncs / 0 order regressions; COMPLETENESS 600/600, 0 stranded.
  - Wire 553 -> 502 MB/socket (-9%); busy 12.63 GB/s, 759 Mzones/s (baseline 12.10, 667); stalls
    351,347 vs 352,621. Best judged run of the series, but within run-to-run spread (632-759 over 3).
  - WHY only 9%: this benchmark saturates producers by design, so snapshots run 88-96% full — there is
    almost no dead data to strip. The 2.3x projection used a stale 44% fill figure; the baseline's real
    fill was 667 Mzones/s x 16 B / 12.1 GB/s = 88%.
  - Sparse regime (delay 150, 29% fill): 61,897 frames, 215.3 MB packed vs 653.6 MB raw-equivalent
    (frames x 10,560 — same-run arithmetic, no A/B needed) = 3.04x; decode busy 19.4 ms vs ~54 ms raw
    at the same GB/s. The win scales inversely with fill; near-empty spans (real-model traces) approach
    the 165-page -> 5-page floor.
  - Device cost: mover noc-chunk 6.98 us per 7-frame push at ~96% fill (~9.9 GB/s per mover, ~20 GB/s
    aggregate — above the device's ~15.9 GB/s offer, so egress stays off the critical path; stalls
    unchanged agree). Hoisting noc_write_init_state out of the per-chunk path was neutral (7.04 -> 6.98):
    the cost is the ~77 write issues per push, not command-buffer setup. The judged run's DRAM rings hit
    100% high-water with ~2,550 filler ring-room waits — that is host-decode < device-offer backlog
    (250 MB vs 256 MB ring capacity), present in the baseline arithmetic too, absorbed losslessly.
  - Deferred option if mover issue cost ever matters: a raw-fallback bit for >=~90%-full frames (one
    burst again). Rejected for now — it is a second wire format for a path that is not the bottleneck.

## §N+59 — Op-perf CSV consumer vs the classic device profiler: semantics match; the capture costs this workload ~8-10% (bh-18, 2026-08-19)

The ops-csv consumer (TT_METAL_PERF_DEBUG_OPS_CSV, first register_consumer user) was validated against
the classic device profiler's ops_perf_results on a 20x ttnn.matmul(512x512) loop. The classic producer
is dead on this branch (the SPSC producer replaced it at compile time), so the reference CSV came from
temporarily flipping kernel_profiler.hpp's ARCH_QUASAR seam to the parked push producer, then reverting.
Join key: our GLOBAL CALL COUNT (= device run_host_id) == classic's >> 10 (host op ids reserve low bits).

Two attribution bugs found and fixed by the comparison, both host-consumer-side:
  1. Only BRISC emits STICKY_PROG, so rec.prog on NCRISC/TRISC lanes is exact only to drainer-sweep
     granularity. Back-to-back cache-hit launches put an op boundary inside nearly every frame and the
     per-op union swallowed a neighbouring kernel: ops 5+ measured exactly ~2x (48.6k vs 23.7k cycles)
     while compile-spaced ops 1-4 were clean. Fix: BRISC kernel windows (exact, in-ring order with the
     sticky) define each op's span per core; other lanes' pairs are assigned by timestamp.
  2. "Latest BRISC start at or before the pair" still misassigned ~half the pairs (NCRISC/TRISC +150%,
     first-to-last-start +10,000%): the five RISCs get go signals independently, so a lane's wrapper can
     open slightly BEFORE its own op's BRISC window. Fix: nearest window start (launch skew is sub-us,
     op gaps tens of us).

After both fixes, per-op agreement across all kernel columns: mean +8-10%, first-to-last-start +-2%,
uniform across columns -- a run difference, not a measurement difference. It is systematic:
  - within-arm spread: perf-debug +-1% (3 runs), classic -0.2% +-0.7% (2 runs);
  - NOT sweep contention: FILL_PCT=90 (throttled sweeps) left the median at 16.7 us vs 16.5-16.9;
  - NOT ring blocking: 0 producer stalls;
  - lead suspect: SPSC per-marker emit cost (flow-control check per marker vs push's plain store) in
    kernels that emit zones in loops (compute CB waits). OPEN -- needs a marker-count ablation.
Also: no *-FW columns are possible on this branch -- the SPSC lifecycle wrapper deliberately emits no
FW markers -- so the CSV reports kernel columns only, plus kernel start/end cycles.

## §N+60 — Per-lane STICKY_PROG: exact op attribution structurally; supersedes N+59's window method (bh-18, 2026-08-19)

N+59's consumer-side attribution (BRISC windows + nearest-start assignment) was a timestamp-proximity
heuristic -- safe only while launch skew << op gap, i.e. guaranteed to misattribute on trace replays.
Root cause was the wire: only BRISC emitted STICKY_PROG, so the other lanes' rings carried no op
identity at all and rec.prog on them was sweep-granular. Now every RISC emits its own sticky at its
launch point (ncrisc.cc/trisc.cc mirror brisc.cc; each reads host_assigned_id from its own mailbox) and
decode scopes prog per LANE -- rec.prog is exact on every record by in-ring ordering, the same mechanism
that always made BRISC exact. The ops-CSV consumer's window/matching machinery is deleted; it keys on
rec.prog directly.

Cost, measured:
  - Judged delay-15: 351,090 stalls (baseline 350,988-352,621), 11.50 GB/s busy / 691 Mzones/s (spread
    of prior runs 632-820 at 10.5-13.7 GB/s) -- within run noise, no pipeline cost.
  - Launch-heavy matmul loop: medians 16,610-17,196 ns post vs 16,450-16,850 pre (n=3 each, overlapping
    ranges) -- bounded above by ~2%, indistinguishable from noise; the sticky is ~20-40 cycles per
    subordinate per launch, outside the kernel wrappers and the marker hot path.
  - Wire: +8 words (32 B) per core per launch.
  - Gates: pytest passed, 2x2 exact (20,040 records), COMPLETENESS 600/600.

vs classic after the structural fix: same uniform +11% on kernel columns, first-to-last-start +2.8%
mean -- confirming N+59's residual is the capture's zone-scope overhead (user: perf-debug zone scope is
currently ~2x the device profiler's per-zone cost), not attribution.

## §N+61 — ResNet-50 validation: perf-debug ops CSV vs classic, non-trace and trace (bh-18, 2026-08-19)

Blackhole ResNet-50 batch-16 (BFLOAT8_B/LoFi), classic arm via the temporarily re-enabled push
producer, perf-debug arm with TT_METAL_PERF_DEBUG_OPS_CSV. Kernel-duration agreement per op:
  - non-trace: 230/230 classic device rows matched 1:1; median +0.80%, mean +2.05%, p90 6.8%.
  - trace (1 replay): 233/233 matched; median +0.46%. The classic report's extra 115 rows are the
    capture-time host ops with EMPTY device columns (trace capture records without executing); ours
    reports exactly the device executions.
  - tails (max ~30-54%) are cross-RUN variance on tiny ops, not measurement disagreement: the arms are
    separate processes and short ops swing several us run-to-run.
  - The matmul microbench's +10% capture footprint does NOT appear on ResNet (<1% median): the
    zone-scope overhead is per zone, so the footprint scales with zone density, not op runtime.

Execution splitting exercised for real on test_perf_trace (15 trace replays): 3,714 rows, max
EXECUTION 14, 3,220 rows with EXECUTION>0. Per-op duration CoV across an op's 15 executions: median
1.58% -- replays of the same program measure consistently, which is a self-consistency check no
single-execution run can provide. Consumer delivered 1,581,952 records with 0 drops; 0 producer
stalls; COMPLETENESS 550/550 on every arm. First realistic-model validation of the packed-frame
capture path end to end.

## §N+62 — Per-op repeatability under capture: perf-debug jitters 3-4x more than classic, medians unbiased (bh-18, 2026-08-19)

N+61's cross-profiler tails were claimed to be run-to-run variance; measured (ResNet-50 non-trace,
230 ops, per-op |kernel-duration diff| distributions):
  pd vs pd (3 pairs):      median 0.71-0.82%, p90 5.0-5.8%, max 23-32%
  classic vs classic:      median 0.21%,      p90 2.1%,     max 7.0%
  classic vs pd (2 pairs): median 1.5-1.7%,   p90 6.1-6.8%, max 29-34%
The cross tails match pd's own within-arm spread -- no hidden per-op disagreement -- but the variance
is OURS: under capture the same op repeats ~3-4x less tightly than under classic, with no mean shift
and 0 stalls. Fits sweep-collision contention: which ops a filler's 10.5 KB span reads land on differs
with sweep phase per run, injecting per-op jitter without bias. The ~1% cross-median residual above
pd's within-arm floor is the long-op-amortized zone-scope cost. Honest footprint statement for real
models: medians unbiased to ~1.5%, per-op jitter p90 ~5.5% vs classic's ~2%.

## §N+63 — Raw fallback: per-frame cheapest-encoding selector; mover ceiling 45 GB/s at max load (bh-18, 2026-08-19)

The packed gather's issue cost was the mover's problem at high fill (N+58: 6.98 us/push, ~10 GB/s/mover
ceiling; small writes backpressure the command buffer, and the init-state hoist proved it is issue
count, not setup). Fix: one compare per frame in the drainer -- payload > ~2/3 of live capacity ships
the RAW span as the old single burst, flagged by SPSC_SPAN_RAW_FLAG in w0's reserved bits (the exact
DEFLATE stored-block shape). The host decode branches per frame; the raw branch is the pre-packing
circular walk resurrected verbatim.

Measured:
  - Judged delay-15 (raw-dominant): exact counts (60.35M records, 0 bad frames/anomalies through the
    raw branch), stalls 352,709 (baseline 352,621), wire 546 MB/socket, decode 88.4 ms -- the raw
    baseline restored where raw is optimal.
  - Mover at high fill: noc-chunk 1.24-1.55 us/push -- 5x better than packed (6.98) and well under the
    prior raw estimate (~4.5): big posted bursts issue in ~1.3 us and the transfer hides behind them;
    packed's cost was cmd-buffer backpressure from ~77 small writes per push.
  - Egress ceiling (SHIP_REPEAT=4 + NO_DECODE, wall): 45.1 GB/s aggregate -- ~3x the device's 15.9 GB/s
    max offer. The mover max-load margin concern is closed.
  - 2x2 (packed path) exact; ops-CSV pytest passed. Both decode branches now gate-covered every run.
Low fill keeps the packed 3x (N+58 delay-150); high fill keeps raw's burst economics; the selector is
one compare and one reserved bit.

Final-state confirmation (post raw-fallback, 57f8b10815b): the three classic comparisons re-run on the
shipped build -- matmul 20/20 median +10.79% (the N+59 zone-density footprint), ResNet non-trace
230/230 median +0.80%, ResNet trace 233/233 median +0.26%; tails match the N+62 jitter distributions.
Sparse workloads select packed, as designed; nothing moved.

## §N+64 — DRISC code-region overflow at max instrumentation: one u64 division cost a 956 B soft-div (bh-18, 2026-08-20)

The flagged N+58 risk was real: DRISC_ZONES=1 + DRISC_ZONE_DETAIL=1 + NOC_FOOTPRINT overflowed the
11,264 B code region by 888 B (graceful: drainer refuses to load, capture empty, run exits 0). ELF
autopsy (debug-line histogram) first CLEARED the packed-gather code -- it is compile-time absent from
the filler ELF that overflowed -- then found the real hog: the pacing controller's mean-fill division
divided a u64 by a u32, dragging the 956 B __udivdi3 soft-division routine into every drainer build.
One sweep's word delta is bounded by grid x live capacity (~300 K, static_asserted), so a u32 cast on
the numerator is exact; the soft-div is gone, the max config fits with ~180 B of headroom, and no
inlining changed anywhere. Lesson for this code region: a single 64-bit divide costs more text than
any feature; the FAILED-TO-LOAD message now says so.

## §N+65 — The "knee regression" audit: historical knees were RUNWAY measurements, the sustained wall is the MOVER's per-frame rate, and NO_DECODE is not the gentlest host (bh-05 + bh-26, 2026-08-24)

The question: production (4F+2M) with NO_DECODE showed stall onset at delay 100 / clean at 150
(bh-05, fast, iters 10k), against remembered era knees of 40-50 (harness) and 15 (§N+40). Audit
verdict: **no device-side service regression — the gap is four confounds, dominated by CAPTURE
VOLUME**, plus one real (and newly named) sustained-regime wall.

### 1. Capture volume is the dominant confound: every era knee was a BURST number

Same code (80bbf89e439), same box, same dispatch, same delay, NO_DECODE:

| | iters 500 (5k zones/RISC — every historical knee) | iters 10k (100k zones/RISC — every recent knee) |
|---|---|---|
| bh-05 fast d100 | **0** | **32,219** |
| bh-05 fast d40  | **0** | 299k (earlier log) |
| bh-26 slow d40  | **0** | 275-283k |

At 5k zones/RISC the 4x64 MiB DRAM rings are runway for the entire capture (§N+39 said this — the
volume knee at 20k zones/RISC — and it was not carried into any later knee comparison). At 100k
zones/RISC the measurement becomes SUSTAINED service rate. Era-volume burst absorption on today's
build is clean at fast delay 40 = 5.04 Gw/s offered — **1.9x above the era conduit's best documented
clean point** (2.66 Gw/s = §N+23's clean-at-100, old wire), and clean where the era stalled (§N+23:
21k at delay 50 = 4.81 Gw/s).

Unit conversion used throughout (measured, §N+52): 1 delay = 7.407 ns; zone period = delay x 7.407 +
86.7 ns (2-marker wire, 4 w/zone) or + ~31 ns (atomic wire, 3 w/zone). v5 clean-at-200-old-wire ==
v6 clean-at-150-atomic == ~1.4 Gw/s: the v5→v6 delay-axis shift is entirely the producer getting
faster (117→42 cyc/zone), zero device change. Box confound: **zero** (bh-26 on the same commit:
fast d100 = 32.5-33.5k, d150 = 0-12 — identical to bh-05). Dispatch confound: only mattered for
§N+40's "knee 15" (bh-26 + slow + iters 500); at iters 10k slow dispatch flatters nothing
(d15 = 319-320k, matching §N+51's bh-18 floor of ~290k).

### 2. The sustained wall: the MOVER's per-frame rate, pinned at ~1.7 frames/us

At the iters-10k knee (bh-05 fast d100, split-on): all 4 DRAM rings at 100.0% high-water with
3.7-4.0k ring-room waits per filler; filler worst sweep 84.6-95.0 us of which 68-81 us is c_reserve
(= DRAM ring room for a filler); mover worst HOST credit-wait 0.1 us (host absent); movers at max
batch 14 every busy sweep, 16.3 us/sweep = **1.16 us/frame**. System frame rate is pinned at the
mover ceiling at both measured delays (1.70 f/us @150, 1.73 @100). Frames are fixed 2,640-word slots
at 33-46% fill under fast-dispatch-synchronized producers + 4 fast quarter-grid fillers, so
sustained words = 1.7 f/us x fill x 2,640 = the measured 1.44 Gw/s clean.

Confirmations: ROLE_RING_MB 64→256 MiB: 32,219 → 5,289 (pure runway; rings still filled).
ROLE_SPLIT=0 (direct-ship, no DRAM hop): 32,219 → ~1.0-1.2k at d100, clean at 125 — the split costs
~1.2x sustained clean rate. Forcing RAW shipping (kPackMaxPayload=0): WORSE (38k; D2H 17.9 GB/s hits
the byte wall) — the mover is pinched between issue-bound (packed) and byte-bound (raw); the 2/3
threshold is right. Pacing proven out of the loop THREE ways: valve logic (gap=0 at max_run >= 384),
occupancy 507 >> 384 at the knee, and FILL_PCT=0 ablation (32.8/33.0k vs 32.2k — no change).
§N+40's "more fillers" lever OPTIMIZED THE BURST REGIME; in the sustained regime it back-fires
(faster fillers → emptier frames → more frames/word → mover binds sooner). The sustained lever is
frames-per-word: gate per-core staging on run size (fill 46% at d100 ⇒ ~2x capacity headroom).

**Rule: quote a knee with dispatch mode, box, ack-probe, wire version, AND capture volume vs runway.**

### 3. Env-lever floor for the slow-dispatch knee (no code changes)

ROLE_RING_MB=448 (the per-bank region max; 44,485 frames/ring): slow iters-10k NO_DECODE knee moves
150 → **clean at 25, 3/3 warm reps** (8.3 Gw/s offered). d22 = ~420, d20 = 32.5k. Below ~22 the
constraint changes identity: filler worst sweep 17.5-19.5 us with credit-wait 0.5 us and rings <=62%
— the §N+51 cadence floor (worker 512-word ring refills in ~30 us at d20 vs ~17-26 us revisit). No
env knob touches it: READ_SPLIT regresses (32.5k→85k), NSTAGE=14 and FILL_PCT=0/90 are no-ops.
Caveat: 448 MiB x 4 is runway — quote with iters; drain tail grows (wall 91→137 ms at 256 MiB).

### 4. Onset anatomy (Tracy captures, on Mac ~/Projects/tt-captures/)

Onset = ceiling-grazing, not capacity failure: at slow d22/RING=448 (full pipeline), exactly 3,376
PRODUCER-STALL zones (matches L1 counters to the digit), mean 0.2 us / max 1.3 us / total 0.67 ms
across 600 lanes (~0.006% perturbation), drizzled over the whole 19 ms run on 81 cores; filler
sweeps p50/p99/max = 13.3/14.1/14.7 us vs 32.7 us fill window. Instrumentation cost measured by
onset shift at constant regime: DRISC_ZONE_DETAIL=1 raises filler sweeps 13.3→16.3 us (+23%) and
onset delay 22→30 (~35%); the workload's 3 point markers/iter (PP_EVENT + 2x PP_DATA — the
BR/NC_Flag/_Data/_Iter rows) are worth ~45% of onset delay (pure zonescopes: onset 22→15).

### 5. OPEN — NO_DECODE shows ~9x MORE onset stalls than the full Tracy pipeline

Same box (bh-05), same device config (slow, d15, RING=448, pure zonescopes), same day:
NO_DECODE = 2,183 / 2,413 / 2,470 / 2,633 (4 warm reps, 66-71 cores) vs full decode+Tracy = 262
(39 cores). The L1 counters are host-independent, so this is a real device-visible host-mode effect.
Working hypothesis: instant acks let the movers burst DRAM-ring reads + L1 handshake traffic
flat-out, and at a 0.2-us-graze onset the added DRAM/NoC/L1-port contention against filler scans and
stage-writes manufactures grazes; decode-paced acks spread mover traffic. NOT yet confirmed —
resolve via filler sweep-time deltas across host arms (NO_DECODE vs decode-no-consumer vs
decode+Tracy). Until resolved: NO_DECODE is the CONSERVATIVE onset count, not the gentlest host.

## §N+66 — RESOLVED: the NO_DECODE onset inversion is MOVER BURST DENSITY; decode-paced acks are an accidental traffic-shaper (bh-05, 2026-08-24)

§N+65's OPEN item, closed by a three-arm host matrix at the sensitive point (slow, d15, RING=448,
pure zonescopes, iters 10k; producer stalls from L1 counters, host-independent):

| arm | acks | producer stalls |
|---|---|---|
| A: NO_DECODE (pop+ack only) | instant | 2,183 / 2,340 / 2,367 / 2,413 / 2,470 / 2,633 |
| B: decode, NO consumer | decode-paced | **164 / 180 / 212** |
| C: decode + Tracy consumer + tracy-capture | decode-paced | 262 |

**B == C, so Tracy is irrelevant — the whole 9-14x inversion is ack pacing.** The phase reports
locate the mechanism on the MOVERS: identical frames moved and ~identical busy-sweep counts in all
arms, but arm A compresses the work into HALF the time per sweep (busy 10.6-12.6 us, credit-wait
0.1 us) where arm B/C spread it (busy 26.7-30.5 us of which ~30 us IS the credit wait). Same bytes,
2.4x the instantaneous burst density on the DRAM banks (ring reads), the NoC (socket pushes), and
the filler-L1 handshake.

The FILLERS' own counters barely move (busy mean +0.3 us ~ 2%, worst sweep flat at 17.6-19.7 us in
both arms) -- which is the tell. The one leg no filler-side counter can see is the per-core HEAD
WRITE-BACK: a POSTED NoC write whose ISSUE the filler's clock records, but whose LANDING is what
releases a blocked producer. At the d15 onset the graze margin is fractions of a us (mean stall
0.2 us, §N+65 §4), so a mover-burst-inflated landing-latency tail converts directly into grazes
without any filler phase changing. Every other channel is excluded by measurement: host credit-wait
0.1 us (A) cannot touch producers, filler cadence flat, rings <=44% with 0 ring-room waits.

Consequences:
1. **The host is not the bottleneck in ANY arm** — worst mover credit-wait is 0.1 us (A) / ~30-54 us
   (B/C, absorbed entirely by the DRAM-ring runway). The onset bottleneck is the pipeline's own
   internal traffic: filler revisit cadence vs the 512-word worker ring, with the margin eroded by
   mover burst density.
2. **NO_DECODE is the conservative (higher) onset count, not the gentlest host.** Quote onset
   numbers with the host arm. Sub-knee/deep-saturation counts are unaffected (§N+65: NO_DECODE ==
   full pipeline from delay 40 down).
3. **Deliberate mover pacing is protective at onset.** The mover controller's "never pace a
   productive consumer" rule (gap=0 the instant frames exist) is exactly what maximizes burst
   density; decode-paced acks accidentally provided the shaping. A small fixed mover gap (or credit
   budget) at high ring runway would give arm-A capture modes arm-B onset behavior — code change,
   measured motivation here.
4. The +2% filler busy-sweep inflation under bursts is real but secondary; the landing-latency tail
   is the scaling term (2% cadence cannot explain 14x stalls; a us-scale tail on release latency at
   a 0.2-us margin can).

Direct confirmation would need a device-side probe that timestamps head-writeback LANDING (e.g. a
worker-side release-latency counter) — not in the current instrumentation, deliberately left as the
follow-up.
