# DRISC drain: cores, roles, and the four different "rings"

Measured on bh-26 (Blackhole, single card), branch `mo/drisc_drain_fast`, 2026-08-11.

## Naming, because "ring" means four different things here

Use these names. Confusing them has already cost real debugging time.

| name used in this file | lives in | holds | who writes | who reads |
|---|---|---|---|---|
| **L1 marker ring** | worker L1, one per RISC | 2-word markers | the worker RISC (producer) | the FILLER's bulk span read |
| **DRAM frame ring** | device DRAM, one per FILLER | whole frames | its FILLER | its MOVER |
| **socket FIFO** | host RAM, one per MOVER | 64 B pages | its MOVER, over PCIe | the host writer thread |
| **host record ring** | host RAM, one shared | 24 B `StreamingProfilerRec` | the decoder threads | the consumer thread -> Tracy |

So: `ring_ensure_room` / `PROFILER_STALL_ZONE` / `SPSC_STALL_COUNT_0` are about the **L1 marker ring**.
`ring-room waits` and `head`/`tail` are about the **DRAM frame ring**. `reserve_pages` / credit-wait /
`fifo_pages` are about the **socket FIFO**. `BroadcastRing` / `dropped` is the **host record ring**.

## The six DRISCs

A DRISC only exists on a DRAM core, and metal exposes exactly ONE per core
(`bh_hal_dram.cpp` registers a single DM processor class, "DRISC0"; `DramConfig` has no `.processor`).

| DRISC | role | bank (DRAM view) | core | NoC row | job |
|---|---|---|---|---|---|
| 0 | FILLER | 5 | `9-9` | y!=0 | sweep worker cores `[0,30)` -> DRAM frame ring on bank 1 |
| 1 | FILLER | 6 | `9-5` | y!=0 | sweep `[30,60)` -> ring on bank 2 |
| 2 | FILLER | 4 | `9-2` | y!=0 | sweep `[60,90)` -> ring on bank 4 |
| 3 | FILLER | 1 | `0-3` | y!=0 | sweep `[90,120)` -> ring on bank 5 |
| 4 | MOVER | 0 | `0-0` | **y==0** | rings of fillers **0 and 2** -> socket FIFO 0 -> host |
| 5 | MOVER | 3 | `9-0` | **y==0** | rings of fillers **1 and 3** -> socket FIFO 1 -> host |

With `TT_METAL_STREAMING_PROFILER_DRISC_ZONES=N` **every one of the six also becomes a PRODUCER of its own zones**,
framed as a worker span and pushed down the path it already owns -- a filler into its own DRAM frame ring, a
mover into its socket FIFO. Each therefore gets its own Tracy row at its own NoC coords, and the lane space
grows by `n_drisc * 5` (lanes `[600,630)` at 120 cores). Default OFF; see "Self-profiling" below and
FINDINGS §N+41.

**Why 4 + 2 and not 4 + 4.** The knee is the FILLER's scan over its slice (FINDINGS §N+28), so fillers are
the thing to multiply: 4 of them at 30 cores each moved the knee from delay 40-60 to delay 15 (§N+40).
Movers cannot follow, for two independent reasons: only NoC row y==0 is measured safe for HOST-FACING duty
(§N+29: y==0 -> 1/75 failures, y!=0 -> 16/125), which is exactly two cores on this part, and the socket
FIFO's TLB budget is `kNSockets * nwin <= 16` with 12 MiB costing nwin=7. So each mover drains TWO rings
instead. It has the headroom: a mover is ~97% idle even after doubling (idle sweep 0.6-0.8 us, ~330 busy
sweeps of ~200,000).

Mover m takes fillers m, m+2 -- **strided, not adjacent**, so each socket carries one low-half slice and
one high-half slice of the grid rather than both halves of one end.

Two 25-run blocks on bank 5 -- §N+29's worst core at 5/25 as a full-job drainer -- cleared y!=0 for FILLER
duty: 0/25 with the NIU merely held in stream mode, 0/25 doing filler-only work. Every DRISC needs its NIU
in stream mode; in the default NOC2AXI mode it cannot initiate NoC at all.

**Ring placement is no longer disjoint from drainer channels, deliberately.** With 6 drainer channels and 4
rings against 7 allocator banks, the old "a ring shares a channel with no drainer" insurance is unreachable.
What is kept is the part with evidence: a ring is never on a MOVER bank (0, 3), because host-facing duty is
where the §N+29 hazard was measured. Rings land on banks 1, 2, 4, 5 -- three of which also host a filler.
Measured with that overlap in place (25 runs): 0 ring-room waits, 0 impossible-head reads, staged == moved
on all four rings, and the knee improved. A ring's traffic terminates at its channel's PREFERRED WORKER
endpoint while a drainer sits on that channel's unused subchannel, so even a shared channel means a
different core and a different NIU, and ~1.4 GB/s each way is a rounding error against a GDDR channel.

Also enforced now: **two DRISCs must never land on the same core.** `pick_unused_dram_logical_core()` takes
a DRAM VIEW and knows nothing about other views resolving to the same physical port -- §N+29 records views
0 and 7 both as NoC core `0-0` -- so `boot_device` `TT_FATAL`s on a duplicate. Two resident kernels sharing
one L1 would overlap staging, socket config, results and handshake with no counter noticing.

## Buffer stack and sizes

| level | where | unit | size | count |
|---|---|---|---|---|
| **L1 marker ring** | worker L1 | 2-word (8 B) marker | 512 words = **2,048 B** -> **256 markers** | 1 per RISC, 5 per core |
| control vector | worker L1 | heads + stall counters | 64 words = **256 B** | 1 per core |
| **span** (unit of transfer) | worker L1 | 64 ctrl + 5 x 512 ring words | 2,624 words = **10,496 B** | 1 per core |
| **staging slot** (frame) | DRISC L1 | 16-word prefix + span | 2,640 words = **10,560 B** = **165 pages** | **7** per DRISC (73,920 B) |
| **self-frame slot** | DRISC L1 | one staging slot, reused as a frame the DRISC authors | **10,560 B** (of which 2,368 B carry data) | 1 per DRISC, only with self-profiling on |
| **DRAM frame ring** | device DRAM | one frame | **64 MiB = 6,355 frames** | 1 per FILLER (**4**) |
| **socket FIFO** | host RAM | 64 B page | 196,608 pages = **12 MiB** | 1 per MOVER (2) |
| host read chunk | host RAM | pooled buffer | <= 1,024 pages = **64 KB** per read | pool <= 4,096 |
| **host record ring** | host RAM | 24 B `StreamingProfilerRec` | 4 Mi default = 96 MiB (runs use 16 Mi = **384 MiB**) | 1 shared |

Geometry notes that are load-bearing, not incidental:
- 2,640 words is a whole number of 64 B pages, so a frame never needs padding, and the bulk span read
  lands at `slot + 64 B` so prefix and span are contiguous and **one** NoC write ships the frame.
- DRAM frame rings are sized a whole multiple of 165 pages, so **a frame never wraps** and the device
  never needs a split write.
- A span fits ONE NoC burst: 10,496 B < `NOC_MAX_BURST_SIZE` = 256 x 64 = 16 KB.
- Only **3** of the 7 staging slots are ever live (`kGenSlots = nstage/2`), so just 3 cores' reads are
  in flight and the 7th slot is unused. That is why the sweep is read-latency bound and cannot be
  widened without more DRISC L1.
- **DRISC L1 has no room for an eighth slot**, which is why self-profiling takes the seventh rather than
  adding one: UNRESERVED is 86,448 B, and 7 slots plus the 8 KB socket-config reserve, 4 KB head scratch and
  the misc block leave **1,792 B**. With the feature on the kernel is handed `nstage - 1` slots and uses index
  `kNStage` for its self frame, so L1 is unchanged and a MOVER's largest batch drops 7 -> 6 (a filler's is
  bounded by `kGenSlots = 3` either way).
- A full span carries at most 1,280 markers (2,560 ring words / 2) across 165 pages = **7.80
  markers/page** against a host batch bound of 8.00 -- a 2.5% margin, which is why that batch
  occasionally fills (handled by flush-and-continue, never by dropping).
- The **socket FIFO cannot be enlarged**: it is mapped through device TLB windows,
  `kNSockets * nwin <= 16`, and 12 MiB already uses 14. That is the whole reason staging moved to DRAM.
- A dual-ring MOVER does **not** split its 7 staging slots between its two rings. It visits them
  sequentially, each with the whole staging area, separated by the write barrier that staging reuse needed
  anyway -- so `max batch` stays **7 per peer** (measured on all four peers, every run). Splitting the slots
  would only pay if the two pushes could overlap, and they cannot: both go into ONE socket.

## Activity, per role

**FILLER**, per worker core: one bulk NoC read of the whole span into a staging slot; reads go out on
`kReadNoc` (the NoC the writes do not use: `NOC_INDEX == 0 ? 1 : 0`, so reads on NoC 1 and writes on
NoC 0 by default). Software-pipelined across two staging generations -- generation G reads while G^1
ships -- with the read barrier issued **LAST**, which is what stops read and ship serialising. Then
inspect the control vector, patch the prefix, write the frame into its DRAM frame ring, and publish
`head` only after a *flushed* barrier. At exit it waits (bounded) until `tail == frames_staged`, so it
never reports a stale mirror -- `inflight = frames_staged - *hs_tail` IS the ring-room predicate.

**MOVER**: no worker grid at all, hence `proc 0.0 us` in its phase line. **For each of its two peers in
turn**: NoC-read that FILLER's `head` out of that FILLER's L1, pull up to 7 frames from that ring into its
own staging, write `tail` back to release ring space, ship to the socket FIFO in `NOC_MAX_BURST_SIZE`
chunks, then one write barrier -- which covers the PCIe push, the tail write, and the reuse of the scratch
word the next peer's tail write sources from. Per-peer state is strictly separate (`mv_tail`, `mv_moved`,
`mv_max_n`, `ring_hi`, the probe words and the live head/tail telemetry): one shared `mv_tail` across two
rings would ack frames on one ring that were only read from the other. Observed `max batch 7` per peer.

**EITHER ROLE, with self-profiling on**: emits its own 2-word markers into a 512-word ring inside its
self-frame slot, stamped with timestamps the drain loop has ALREADY read (`t_batch0`/`t_issue`, `stage_run`'s
`t0`/`t1`/`t2`, the barrier's `t_b0`) rather than fresh clock reads -- so a zone's duration is the same
quantity the matching `out[]` phase counter accumulates. At the end of a captured sweep it sets its own
`SPSC_CORE_XY` / head / tail in the frame's control vector and ships the slot through the same
`emit_run(kSelfSlot, 1)` the payload uses, then a bounded write barrier. Nothing about the wire format, the
ring protocol or the host decoder changes: the frame IS a worker span, and only ring 0 of its five is live
(`myRiscID == PROCESSOR_INDEX == 0` on a DRISC), so the host's per-RISC walk yields nothing from the rest.

## Self-profiling: what the drainers say about themselves

Zone tree per drainer row: `DRISC-SWEEP` (depth 0) with `DRISC-READ`, `DRISC-READ-WAIT`, `DRISC-PROC` and
`DRISC-WR-BARRIER` as children, and `DRISC-CREDIT-WAIT` / `DRISC-WRITE` inside `DRISC-PROC` on a filler
(directly under `DRISC-SWEEP` on a mover, which has no proc phase). Per-occurrence means, bh-26:

| role | SWEEP | READ | READ-WAIT | PROC | CREDIT-WAIT | WRITE | WR-BARRIER |
|---|---|---|---|---|---|---|---|
| FILLER | 16.4-17.0 us | 128 ns | 67-69 ns | **825-864 ns** | 54 ns | 138-140 ns | 64-120 ns |
| MOVER  | 11.5-11.9 us | 1,425-1,774 ns | - | - | **2,698-2,781 ns** | 831-1,065 ns | 359-525 ns |

The two roles are bottlenecked on different things and their own zones say so: a filler's per-batch PROC
dwarfs everything else it does, while a mover's largest phase is the socket **credit wait** -- the quantity
§N+38 identified as setting the knee, now visible per occurrence, and ~1.6x its next-largest phase. The same
named phase differs by three orders of magnitude between the roles (54 ns of DRAM ring room on a filler
against 2.7 us of host FIFO credit on a mover), which is why one number for "the drainer" never meant anything.

**Superseded by full tracing (FINDINGS §N+41): `DRISC_ZONES=1` traces every sweep in one contiguous
window at sweep level, which measured CHEAPER than sampling because the per-sweep publish, not the markers,
was the cost. The sampler is described below for why it existed.** It triggered on discovered WORK, not on sweep number, because both roles are >99% idle (a filler
moves frames in ~114 of ~25,000 sweeps, a mover in ~350 of ~230,000). A sweep-number rule captured 1 working
sweep out of 55 on a filler and 11 of 64 on a mover. A mover arms before issuing its DRAM read, so its busy
visit is captured whole; a filler arms at the end of the first batch with live cores, so that sweep's earlier
batches are not recovered. An instrumented sweep that turns out idle is rewound or abandoned for free, and a
captured sweep is bounded to one ring (past that it is truncated, counted, and excluded from the counter
cross-check). Cost: **0.28-0.52% of a drainer's egress**, +4% on a filler's sweep time, 0 producer stalls at
delay 60 and 49-140 at delay 15 (a knee crossing).

**Sampling IDLE sweeps is off by default and should stay off** unless the question is specifically what an idle
poll costs. A drainer is resident from device open, so its idle sweeps span the whole process (~190 ms measured)
while a workload is a ~1.9 ms sliver: sampling them puts DRISC zones 187 ms before the first worker zone and
makes the rows unreadable. They are also not inert -- a filler's self frame is a real frame in its DRAM ring, so
its mover ships it, manufacturing mover credit-wait/write zones outside the workload window (measured: all 13
such zones followed a peer filler's publish by 1.7-2.4 us) and diluting the mover's credit-wait figure from
2.70-2.78 us down to 1.6-1.8. Work-triggered only, all six drainers start within 8-67 us of the first worker
zone; the fillers end just before the last one and the movers trail it by 2.5-2.9 ms, which is the ring's drain
tail.

## Measured costs and occupancy

Per-sweep costs at 120 cores, delay 20, and they halve with the slice as the model predicts (2 fillers at 60
cores each -> 4 at 30):

| | 2 fillers x 60 cores | 4 fillers x 30 cores |
|---|---|---|
| FILLER idle sweep | 16.7 us | **8.2-8.5 us** |
| FILLER busy sweep | 27.8-28.0 us | **13.1-14.0 us** |
| FILLER worst sweep | 39.0-40.3 us | **17.2-19.7 us** |
| of which `proc` (the scan) | 15.0-15.3 us | **7.5 us** |
| MOVER idle sweep | 0.3-0.4 us | 0.6-0.8 us (two head reads) |
| MOVER busy sweep | 5.2-5.5 us x ~515 | 13.5-13.6 us x ~330 |
| MOVER worst sweep | 33.8-37.5 us | 51.9-52.9 us (credit-wait 28-34) |

FILLER blocking wait is **0.5 us** (it was 50-97 us of socket-FIFO credit-wait before the split). A MOVER is
still ~97% idle after doubling: 191,357 idle sweeps x 0.8 us = 153 ms of a 161 ms wall.

DRAM frame ring high-water at 5k zones/RISC: **328-770 of 6,355 frames per ring** across all four rings,
against 73-1,266 for the 2-ring configuration because each ring now carries a quarter of the grid. `ring-room
waits 0` everywhere, including at delay 5 where producers stall ~11,000 times. That is diagnostic and it is
the whole basis of the 4-filler change: the FILLERs are never blocked by the DRAM frame ring or by egress,
so what limits low delays is the SWEEP.

## DRAM cost, and how big the ring actually needs to be

Measured, not estimated -- see FINDINGS §N+39 for the two sweeps and §N+40 for the 4-ring cost.

**The rings live in the HAL's per-bank DRAM PROFILER region** (channel-relative `0x40`), not in a
`MeshBuffer`. That region is reserved at the same offset in EVERY bank, and
`get_profiler_dram_bank_size_for_hal_allocation()` sizes it to
`max(old_profiler_size, streaming_profiler_dram_region_bytes_per_risc())` -- so `TT_METAL_STREAMING_PROFILER_ROLE_RING_MB`
is one knob for both the region and the ring, and the ring adapts to whatever was actually reserved
(`frames = region_bytes / 10,560`).

The consequence that matters: **the cost is per BANK and does not depend on how many rings you place.**
bh-26 has **7** allocator DRAM banks (8 channels, 1 harvested; confirmed on silicon via the ring-bank
validation message), so a 64 MiB setting reserves **448 MiB** whether 1 ring or 7 rings sit in it. Going from
2 rings to 4 therefore cost **zero additional DRAM** -- it moved 128 MiB of the reservation from "region no
ring uses" to "region carrying a ring", nothing more. There are still 3 unused banks, so a 5th, 6th and 7th
ring would also be free. For scale, the OLD profiler's region is 4.58 MiB/bank = **32.0 MiB** total, so 64
MiB puts us at **14x** the profiler we are replacing.

Which bank a ring lives in is IRRELEVANT to footprint. The only lever is
`TT_METAL_STREAMING_PROFILER_ROLE_RING_MB` itself.

**The earlier claim here -- that the ring "should come down a long way" because under a fifth is ever
used -- was wrong, and wrong for an instructive reason.** The 8.4%-18.9% high-water it rested on was
measured at 5,000 zones/RISC only. The ring does not have a steady-state occupancy: its high-water
tracks TOTAL VOLUME and never plateaus (14% at 5k zones/RISC, 49% at 10k, 92% at 15k, 100% at 20k).
The movers drain permanently slower than the fillers stage, so **the ring is runway, not headroom** --
it buys a fixed number of zones before back-pressure reaches producers, and 64 MiB buys ~16-17k
zones/RISC. A low high-water means the workload ended first, not that the ring is oversized.

So the ring size is a function of the capture VOLUME you intend to support:

| ring | frames | DRAM (x7 banks) | stall-free at 5k zones/RISC |
|---|---|---|---|
| 64 MiB (default) | 6355 | 448 MiB | yes (runway ~16-17k zones) |
| **12 MiB** | 1191 | **84 MiB** | **yes, 3/3 -- smallest that is** |
| 10 MiB | 992 | 70 MiB | 2 of 3 |
| 9 MiB | 893 | 63 MiB | no (0.7-1.7k stalls) |
| 6 MiB | 595 | 42 MiB | no (~1.8k stalls) |

12 MiB is the right default for 5k-zone captures -- 84 MiB, 2.6x the old profiler instead of 14x -- with
the env var as the escape hatch for high-volume runs. Undersizing costs producer perturbation but never
correctness: every size down to 6 MiB dropped **zero** records with **zero** timestamp regressions,
because the ring fills, the filler waits for room, and the producers stall. Nothing is lost.

Two structural notes for anyone tuning this. The lanes are **asymmetric** -- filler 1 fills first in
every undersized run while filler 0 sometimes never fills at all (at 8 MiB: 794/794 with 36 ring-room
waits vs 648/794 with 0), so the required size is set by the slowest lane, and balancing them buys
more than any sizing change. And a bigger ring cannot rescue a long capture: the host record ring is the
real volume ceiling (drops begin at 15k zones/RISC, one sweep point BEFORE producers stall), so doubling
the ring just doubles the runway.

**Not re-measured at 4 rings:** the sizing table above (12 MiB smallest stall-free at 5k zones/RISC, and the
20k zones/RISC volume knee) was taken with 2 rings. Four rings each carry a quarter of the grid, so per-ring
high-water roughly halved at 5k zones -- which suggests the same total runway spread over twice as many
rings, i.e. a smaller per-ring minimum. That is an inference, not a measurement; re-run §N+39's Sweep A
before lowering `ROLE_RING_MB` on the 4-ring configuration.
