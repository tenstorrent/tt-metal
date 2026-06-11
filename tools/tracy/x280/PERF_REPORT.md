# X280 → Tensix L1 Monitoring: Throughput Progression

**Platform:** Blackhole `bh-qb-05`, Linux on the L2CPU (SiFive X280) tile, reading
Tensix-core L1 over NoC TLB windows (uncached System Port). All NoC traffic on
NoC0. Tensix kernel increments an L1 counter on every compute core; the X280
polls it non-invasively. KMD 2.7.0, NoC @ 1.35 GHz.

## The constraint

The X280 is **in-order**, and uncached System-Port loads are strongly-ordered:
a hart can have **~1 NoC read outstanding at a time**. The workload is therefore
**latency-bound, not bandwidth-bound** — a 64 B flit round-trip is ~300 ns, but a
single NoC link moves ~86 GB/s, so we use <1 % of the wire. Every gain below comes
from either (a) overlapping reads *within* a flit, or (b) adding harts to get
independent reads in flight. There is no async/DMA path: the X280 issues NoC
transactions *only* via TLB-window loads (NIU command buffers and platform DMA
were both tested and don't work).

## Progression

| # | Method | Granularity | Threads | Cores | Per-unit | Throughput | Coverage |
|---|--------|-------------|:------:|:----:|----------|-----------|----------|
| 1 | Single u32, serial | 4 B | 1 | 1 | 257 ns/read | **15.6 MB/s** | — |
| 2 | Full flit, 8× u64 overlapped | 64 B | 1 | 1 | 317 ns/flit | **202 MB/s** | — |
| 3 | All-core sweep, naive (16 serial u32/core) | 64 B | 1 | 110 | 4044 ns/core | — | 444 µs/snapshot |
| 4 | All-core sweep, flit-overlapped | 64 B | 1 | 110 | 364 ns/flit | **175 MB/s** | 40 µs (2.74 Mflit/s) |
| 5 | + 2 harts | 64 B | 2 | 110 | 364 ns/flit | **351 MB/s** | 20 µs (5.48 Mflit/s) |
| 6 | + 3 harts | 64 B | 3 | 110 | 441 ns/flit | **435 MB/s** | 16 µs (6.79 Mflit/s) |
| 7 | Full 4 KB L1 buffer/core | 4 KB | 1 | 110 | 309 ns/flit | **207 MB/s** | 2.18 ms |
| 8 | Full 4 KB L1 buffer/core | 4 KB | 3 | 110 | 447 ns/flit | **430 MB/s** | 1.05 ms (~954/s) |

*(MB/s = Mflit/s × 64 B. "Coverage" = wall time for one full pass over all cores.)*

## Inflection points

- **1 → 2 (16.6× per byte): overlap within a flit.** Reading a 64 B flit as 8
  *independent* u64 loads in one expression keeps them outstanding together (one
  flit transaction, ~317 ns) instead of 8 serial round-trips (~2 µs). Reading 4 B
  and reading 64 B cost almost the same — pure latency, so always read the whole
  flit.
- **3 → 4 (11×): apply overlap to the full sweep.** Naive per-core reads (16
  serial u32) cost 4044 ns/core; the flit-overlap pattern drops it to 364 ns/core,
  taking the 110-core snapshot from 444 µs to 40 µs.
- **4 → 6 (2.5×): multiple harts.** One in-order hart can't overlap *across*
  flits, but independent harts can. Two harts scale perfectly (2.0×, no
  cross-hart slowdown); three still gain (2.5×) but each backs off ~17 %
  (364 → 441 ns/flit) from contention at the shared NIU. **Four harts collapse** —
  the tile has only 4 harts and Linux needs ≥1, so dedicating all four to
  spin-loops starves the kernel. **Sweet spot: 3 worker harts, 1 for the OS.**
- **7 → 8: 4 KB scales worse than 64 B (2.08× vs 2.5×).** A 4 KB read is 64
  back-to-back flits — sustained, high NoC occupancy — so three streams saturate
  the shared NIU harder (447 ns/flit, +45 %) than the gap-y 64 B sweep (+21 %).

## The wall

At 3 harts, **both the 64 B sweep (435 MB/s) and the 4 KB sweep (430 MB/s) plateau
at ~430 MB/s** — independent of access pattern. That is the read-response
bandwidth ceiling of the **single NoC0 NIU / tile egress port**. Net gain over the
naive starting point: **444 µs → 16 µs per 64 B snapshot (~28×)**; full 4 KB L1 of
all 110 cores snapshotted every **~1.05 ms (~950 Hz)**.

## Export to host (X280 → host)

The poll lands data in X280 DRAM; getting it off-chip uses the X280's own network
(device DRAM is off-limits under Linux, so the legacy "push to a device DRAM
profiler buffer, host reads over PCIe" path isn't available here). That network is
tt-bh-linux's SLIRP user-mode NAT (`10.0.2.x`).

| Stage | Rate |
|-------|------|
| Raw TCP X280 → host (`netsrc`/`netsink`, 2 GB) | **13.9 MB/s (111 Mbit/s)** |
| 3-hart 4 KB poll **with** export bolted on (`pollexport`) | poll **326 MB/s**, export **14.5 MB/s**, **95.5 % dropped** |

Findings:
- The SLIRP link is a hard **~14.5 MB/s** wall — ~22× below the 430 MB/s poll, and
  the dominant end-to-end bottleneck.
- Adding export costs **~24 % poll throughput** (430 → 326 MB/s): pollers now
  *store* every byte into a DRAM staging ring (vs XOR-and-discard), and that write
  bandwidth + the exporter draining rings + the kernel TCP copy contend on the
  X280 memory subsystem.
- At 326 MB/s produced vs 14.5 exported, **95.5 % of full-4 KB snapshots drop** at
  the ring. Raw full-buffer streaming is infeasible over this link.
- **Implication:** export budget ≈ 14.5 MB/s ≈ **1.8M markers/s (8 B) aggregate,
  ~16k/s/core** — fine for normal zone profiling, but you must ship **deltas/
  markers only, never full buffers**. High-rate (NoC tracing) needs the
  DRAM/PCIe fallback.

## Ruled out

- **NIU command-buffer DMA** — the X280 cannot issue NoC transactions by writing
  the NIU command registers (CMD_ACCEPTED never increments); ISA docs confirm
  L2CPU NoC access is TLB-window-only.
- **Platform DMA engine** — none exists on the X280 (no dmaengine in device tree).
- **Cached Memory Port** — gives non-blocking-cache overlap, but returns stale
  data; useless for continuous monitoring.

## Next lever

**NoC1 split.** The ~430 MB/s wall is one NIU. The tile has two (NoC0 + NoC1);
TLB windows pick their NoC via a per-window `noc_sel` bit. Routing half the windows
over NoC1 should push toward ~860 MB/s if the cap is the NIU — or reveal the wall
is further downstream (the mesh near the L2CPU tile) if it doesn't move.

## Tools (`tools/tracy/x280/`)

- `pollall.c` — single-thread 64 B/core sweep over a coordfile.
- `pollalln.c` — sweep `<nbytes>`/core (e.g. 4096); used for the 4 KB rows.
- `pollmt.c` — pthread version, pins one thread/hart, auto-runs N=1..4 *(has an
  unexplained ~2.2× per-thread handicap; multi-process `pollall`/`pollalln`
  pinned with `taskset` is the trusted measurement)*.
- `x280_niu_read.c`, `x280_niu_probe.c` — NIU register probes (DMA dead-end).
