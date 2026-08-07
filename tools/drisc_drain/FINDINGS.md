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
> **§N+24 (newest) SOLVES the slow-dispatch TEARDOWN** and overrides §N+21 B: it was the harness passing
> `--gx 0` (= 12x10 under slow dispatch) while the drainer polls 11 columns. Harness fixed; slow cells are
> now 10/10 clean and slow dispatch is unblocked. **All previously recorded slow-dispatch cells are void.**
> `14-2..14-11` are TENSIX WORKERS (one column), not the DRAM/DRISC column.

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
| socket `sender_is_l2cpu` | **true** -- physical NoC coord + full L1 address | false -- logical coord, worker-L1 semantics |
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
4. Addressing/coordinate differences (`sender_is_l2cpu`, physical vs logical) are ADDRESSING ONLY and are
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
