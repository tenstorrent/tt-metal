# DRISC drain: cores, roles, and the four different "rings"

Measured on bh-26 (Blackhole, single card), branch `mo/drisc_drain_fast`, 2026-08-11.

## Naming, because "ring" means four different things here

Use these names. Confusing them has already cost real debugging time.

| name used in this file | lives in | holds | who writes | who reads |
|---|---|---|---|---|
| **L1 marker ring** | worker L1, one per RISC | 2-word markers | the worker RISC (producer) | the FILLER's bulk span read |
| **DRAM frame ring** | device DRAM, one per FILLER | whole frames | its FILLER | its MOVER |
| **socket FIFO** | host RAM, one per MOVER | 64 B pages | its MOVER, over PCIe | the host writer thread |
| **host record ring** | host RAM, one shared | 24 B `PerfDebugRec` | the decoder threads | the consumer thread -> Tracy |

So: `ring_ensure_room` / `PROFILER_STALL_ZONE` / `SPSC_STALL_COUNT_0` are about the **L1 marker ring**.
`ring-room waits` and `head`/`tail` are about the **DRAM frame ring**. `reserve_pages` / credit-wait /
`fifo_pages` are about the **socket FIFO**. `BroadcastRing` / `dropped` is the **host record ring**.

## The four DRISCs

A DRISC only exists on a DRAM core, and metal exposes exactly ONE per core
(`bh_hal_dram.cpp` registers a single DM processor class, "DRISC0"; `DramConfig` has no `.processor`).

| DRISC | role | bank | core | NoC row | job |
|---|---|---|---|---|---|
| 0 | FILLER | 5 | `9-9` | y!=0 | sweep 60 worker cores -> its own DRAM frame ring |
| 1 | FILLER | 6 | `9-5` | y!=0 | sweep the other 60 -> its own DRAM frame ring |
| 2 | MOVER | 0 | `0-0` | **y==0** | its DRAM frame ring -> socket FIFO -> host |
| 3 | MOVER | 3 | `9-0` | **y==0** | its DRAM frame ring -> socket FIFO -> host |

DRAM frame rings are placed on banks 1/2, deliberately off both FILLER and MOVER channels (insurance
against an untested NIU interaction; NOT measured as necessary -- `TT_METAL_PERF_DEBUG_ROLE_RING_BANKS`
A/Bs it). MOVERS hold the row-0 cores because only NoC row y==0 is measured safe for HOST-FACING duty
(FINDINGS N+29: y==0 -> 1/75 failures, y!=0 -> 16/125). Two 25-run blocks on bank 5 -- N+29's worst core
at 5/25 as a full-job drainer -- then cleared y!=0 for FILLER duty: 0/25 with the NIU merely held in
stream mode, 0/25 doing filler-only work. Every DRISC needs its NIU in stream mode; in the default
NOC2AXI mode it cannot initiate NoC at all.

## Buffer stack and sizes

| level | where | unit | size | count |
|---|---|---|---|---|
| **L1 marker ring** | worker L1 | 2-word (8 B) marker | 512 words = **2,048 B** -> **256 markers** | 1 per RISC, 5 per core |
| control vector | worker L1 | heads + stall counters | 64 words = **256 B** | 1 per core |
| **span** (unit of transfer) | worker L1 | 64 ctrl + 5 x 512 ring words | 2,624 words = **10,496 B** | 1 per core |
| **staging slot** (frame) | DRISC L1 | 16-word prefix + span | 2,640 words = **10,560 B** = **165 pages** | **7** per DRISC (73,920 B) |
| **DRAM frame ring** | device DRAM | one frame | **64 MiB = 6,355 frames** | 1 per FILLER (2) |
| **socket FIFO** | host RAM | 64 B page | 196,608 pages = **12 MiB** | 1 per MOVER (2) |
| host read chunk | host RAM | pooled buffer | <= 1,024 pages = **64 KB** per read | pool <= 4,096 |
| **host record ring** | host RAM | 24 B `PerfDebugRec` | 4 Mi default = 96 MiB (runs use 16 Mi = **384 MiB**) | 1 shared |

Geometry notes that are load-bearing, not incidental:
- 2,640 words is a whole number of 64 B pages, so a frame never needs padding, and the bulk span read
  lands at `slot + 64 B` so prefix and span are contiguous and **one** NoC write ships the frame.
- DRAM frame rings are sized a whole multiple of 165 pages, so **a frame never wraps** and the device
  never needs a split write.
- A span fits ONE NoC burst: 10,496 B < `NOC_MAX_BURST_SIZE` = 256 x 64 = 16 KB.
- Only **3** of the 7 staging slots are ever live (`kGenSlots = nstage/2`), so just 3 cores' reads are
  in flight and the 7th slot is unused. That is why the sweep is read-latency bound and cannot be
  widened without more DRISC L1.
- A full span carries at most 1,280 markers (2,560 ring words / 2) across 165 pages = **7.80
  markers/page** against a host batch bound of 8.00 -- a 2.5% margin, which is why that batch
  occasionally fills (handled by flush-and-continue, never by dropping).
- The **socket FIFO cannot be enlarged**: it is mapped through device TLB windows,
  `kNSockets * nwin <= 16`, and 12 MiB already uses 14. That is the whole reason staging moved to DRAM.

## Activity, per role

**FILLER**, per worker core: one bulk NoC read of the whole span into a staging slot; reads go out on
`kReadNoc` (the NoC the writes do not use: `NOC_INDEX == 0 ? 1 : 0`, so reads on NoC 1 and writes on
NoC 0 by default). Software-pipelined across two staging generations -- generation G reads while G^1
ships -- with the read barrier issued **LAST**, which is what stops read and ship serialising. Then
inspect the control vector, patch the prefix, write the frame into its DRAM frame ring, and publish
`head` only after a *flushed* barrier. At exit it waits (bounded) until `tail == frames_staged`, so it
never reports a stale mirror -- `inflight = frames_staged - *hs_tail` IS the ring-room predicate.

**MOVER**: no worker grid at all, hence `proc 0.0 us` in its phase line. NoC-read its FILLER's `head`
out of that FILLER's L1, pull up to 7 frames from the DRAM frame ring into its own staging, write `tail`
back to the FILLER's L1 to release ring space, then ship to the socket FIFO in `NOC_MAX_BURST_SIZE`
chunks. Observed `max batch 7`.

## Measured costs and occupancy

Per 55-60 core sweep, invariant across every run: `read` **3.6 us**, `proc` **12.9 us**. FILLER blocking
wait **1.0 us** (it was 50-97 us of socket-FIFO credit-wait before the split). MOVERs idle-poll at
**0.3-0.4 us** and only ~440 sweeps are busy, so the configuration is overwhelmingly idle.

DRAM frame ring high-water: **533-1,200 of 6,355 frames (8.4%-18.9%)**, with `ring-room waits 0`
everywhere -- *including* the saturated delay-20 run (18,426 producer stalls). That is diagnostic: below
the knee the FILLERs are never blocked by the DRAM frame ring, so what limits delay 20 is the SWEEP
(read+proc over 60 cores), not absorption and not egress.

Which makes 64 MiB per ring look generous: ~192 MiB is allocated for 128 MiB of actual ring (the
interleaved allocation wastes a page) and under a fifth is ever used.
`TT_METAL_PERF_DEBUG_ROLE_RING_MB` should come down a long way before absorption erodes -- worth
measuring, since holding ~192 MiB of DRAM for the profiler's lifetime is the likeliest thing to bite on
a real model.
