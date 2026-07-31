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

## What it means: the zone budget

A zone is 2 markers = 4 words = 16 B, and a lane's ring is 512 words, so **128 zones per lane**
before it wraps. Dividing a sweep period by 128 gives the shortest zone that can be sustained
back-to-back on one lane:

| limited by | rate | min zone duration |
|---|---|---|
| ingest alone, whole-core sweep @ 18.29 us | 67.4 GB/s | **143 ns** |
| leg A: ingest + DMA to GDDR, one DRISC | 43.18 GB/s | **222 ns** |
| egress to host, untuned consumer | 16.85 GB/s | 580 ns |
| egress to host, tuned consumer | 25.36 GB/s | **387 ns** |
| egress to host, zero-copy (measured via discard) | 57.60 GB/s | **167 ns** |

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
7. **If the trace goes to DRAM, ping-pong two buffers.** One DRISC doing ingest + DMA runs at
   43.18 GB/s, ~10% below reads alone, and the DMA hides completely behind the reads as long as a
   second batch buffer exists. A single deeper buffer trades that away and loses 23%.
8. **Consume the FIFO in place.** The `memcpy` in `D2HSocket::read()` is the single largest remaining
   cost anywhere in the pipeline -- 2.3x -- and it penalises the producer as well as the consumer,
   because the host's PCIe reads contend with the device's PCIe writes.

## Gotchas

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
- **Do not pipe a long device test through `head`.** Closing the pipe early SIGPIPEs the test binary
  mid-run and can leave the card wedged (`Read 0xffffffff over PCIe ID 0`). Redirect to a file and
  grep the file. Recovery is `tt-smi -r`.
- **No device Tracy zones on DRISC.** `kernel_profiler.hpp` does not gate on `COMPILE_FOR_DRISC`, so
  `DeviceZoneScopedN` compiles to nothing. Use the watcher ring buffer + `get_timestamp_32b()` idiom.

## Open questions

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
