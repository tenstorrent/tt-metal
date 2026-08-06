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

**The drain is issue-bound, not bandwidth-bound. Over-reading is free.** This is the inverse of the
X280, where the only lever was fewer bytes per marker. Here the lever is more bytes per transaction.

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
L1 twice — written by the NIU, read by the DMA engine. That is structurally what killed the X280,
whose LIM ran at 2.5 GB/s and was crossed twice for ~1.2 GB/s.

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
the same fan-out shape as the X280 reader pool, except every reader is also its own egress path, so
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
X280 reader fan-out helped because reads were the constraint; here the constraint is downstream of
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

2. **Poll-then-drain is what X280 does -- but NOT for the reason first claimed here.** This drainer
   polls the control vector, then reads the rings, matching X280: `profzone.c` reads the five tails in
   one 20 B vector-load NoC transaction (`read_tails`) and then bulk-reads the **rings only**
   (`rbufs = cbase + ring_off`, 5 x 2048 B, no control vector).

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
   `TT_METAL_PERF_DEBUG_PROFILER=1` boots the X280. Two consumers on one SPSC ring corrupt each other.
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
   drainer never constructs or injects identity -- it copies a word the core wrote about itself. This
   is why the X280's `PP_STICKY_SRC` machinery has no counterpart here.

2. **Read per-lane straight into the page.** The alternative -- one whole-core 10 KB read into staging,
   then a local copy of the live words into the page -- costs a device-side memcpy and ships dead ring
   space. Each read instead lands exactly where it belongs. This deliberately trades *more read issues*
   (up to 5/core rather than 1) for *fewer bytes*, the opposite of the pure-ingest tuning above,
   because here the bytes have to cross PCIe.

3. **Ring wrap is resolved on the device**, by splitting the run into two reads that land contiguously.
   The host never needs the ring geometry, so the decoder stays a flat walk.

4. **One shared header for the wire format**, included by kernel and host both. Duplicating those
   constants is exactly how the X280 readers rotted -- each self-consistent, all wrong.

### Cost of the framing

652 pages x 8 KB = 5.34 MB shipped for 4.62 MB of payload, so **~87% efficiency**. Headers are
negligible (2,441 frames x 8 B = 19 KB); essentially all of the 0.72 MB overhead is page padding,
because a page is flushed whenever the next frame would not fit and a frame can be up to 514 words.
Two ways to close it if it matters: let frames straddle pages (costs a reassembly buffer on the host),
or raise the page size. Not tuned here -- correctness first.

Throughput is again **not** meaningful from this run: the device figure is dominated by the quiet
sweeps used to detect completion, and the host wall (0.076 s) includes them.

## Zone decode: the stream is real

The framed egress carried words; this parses them back into zones host-side, against `prof_packet.h`
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

The X280 copies word-granular safely because its L2CPU uses CPU loads and stores; a DRISC moves data
with NoC DMA and is bound by the alignment rules. **Do not port X280 copy idioms to a DRISC unchanged.**

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
- **D2HSocket with a DRISC sender** goes through `ExternalConfigBuffer{address, sender_is_l2cpu=true}`
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

**`profstream.c` is stale in two places, not one.** Alongside the known `CTRL_TAIL(r) = 5u + r` it
hardcodes `rbufs = cbase + 128`. 128 B is 32 words -- the OLD control-vector size; it is now 64 words
(256 B). Both literals are self-consistent for the old layout and both wrong for the current one, so
that reader would land 128 B inside the control vector rather than on ring data. `profcons.c` and
`profll.c` carry the same tail literal. `profzone.c` is fine: the host passes `SPSC_RING_TAIL_0` and
`PROFILER_L1_CONTROL_BUFFER_SIZE` through the boot nonce, with 128 only as a documented fallback for
an out-of-date host.

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
**Knee moved 875 → 125** (7×) against the X280 baseline. Everything below is measured on
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
| X280 (§27) | 875 | 5.88 µs | 187 M mk/s, 1.50 GB/s |
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
static window on `!sender_is_l2cpu_`, and `perf_debug_profiler` sets `sender_is_l2cpu = !tensix_drain` — so
the DRISC inherited "no static TLB window exists", which is true of the X280 L2CPU and false of a DRAM core.

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

That is the same signature as the X280-era freezes: two processes stop together, the kernel logs nothing, and
the physical host never reboots — a host CPU stalling on MMIO to a wedged card. Worth knowing it recurs on the
DRISC perf-debug path, not just the X280 one.

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
