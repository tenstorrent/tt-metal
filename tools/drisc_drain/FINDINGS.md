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
`socket_notify_receiver`), DRISC as sender:

| page | device GB/s | ns/page | device time waiting on host | host-observed GB/s |
|---|---|---|---|---|
| 2 KB | 4.539 | 451.2 | 79.3% | 4.535 |
| 8 KB | 10.633 | 770.5 | 87.7% | 10.643 |
| 32 KB | **16.551** | 1979.8 | 79.8% | **16.581** |

Device-measured and host-observed agree within 0.2%.

**~80% of device time is spent spinning in `socket_reserve_pages`, so 16.55 GB/s is the host
consumption rate, not the DRISC's.** Backing the wait out gives the device write path:

- 8 KB: 94.8 ns for 8192 B → **86.4 GB/s**
- 32 KB: 400 ns for 32768 B → **81.9 GB/s**

At 8 KB and above the DRISC writes at the same 86.3 GB/s NoC port limit the reads hit. **The device
side of egress is already maxed; every remaining lever is host-side** — larger pages (64 KB still fits
DRISC L1), zero-copy instead of the `memcpy` into the sink buffer, or parallel readers. The page-size
trend says the host loop is per-page-overhead-bound.

## What it means: the zone budget

A zone is 2 markers = 4 words = 16 B, and a lane's ring is 512 words, so **128 zones per lane**
before it wraps. Dividing a sweep period by 128 gives the shortest zone that can be sustained
back-to-back on one lane:

| limited by | rate | min zone duration |
|---|---|---|
| ingest, whole-core sweep @ 18.29 µs | 67.4 GB/s | **143 ns** |
| egress, today's host consumer | 16.55 GB/s | **580 ns** |
| egress, device side alone | ~86 GB/s | 112 ns |

**Egress binds, by 4x — and it binds on host software, not hardware.** Fix the consumer and ingest
becomes the constraint again at 143 ns.

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
- **No device Tracy zones on DRISC.** `kernel_profiler.hpp` does not gate on `COMPILE_FOR_DRISC`, so
  `DeviceZoneScopedN` compiles to nothing. Use the watcher ring buffer + `get_timestamp_32b()` idiom.

## Open questions

- Can the host consumer approach the device's ~86 GB/s? That decides whether the design is
  ingest-bound at 143 ns/zone or stuck at 580.
- The second NIU is untouched. `Noc(uint8_t noc_id)` lets one kernel drive both, which is the only way
  a single DRISC could exceed one port's 86.3 GB/s.
- Poll cost with a real head mirror (this measurement primed heads to zero).
- Attempted clock calibration from `cycles / host_wall_time` did not converge — JIT compile time
  dominates the host wall clock even at a 512 MB target. The 86.3 GB/s wire-rate match is the better
  anchor; a warm-JIT second pass would settle it directly.
