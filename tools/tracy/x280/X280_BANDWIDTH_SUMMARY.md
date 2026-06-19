# X280 Bandwidth — Executive Summary (both directions)

Consolidated findings for the **X280-as-observability-engine** model: the X280
L2CPU tile (running Linux) PULLs Tensix-core L1 over the NoC, and EXPORTs the
collected data off-chip to the host. This doc summarizes both directions and the
async/DMA question. For the blow-by-blow pull-path progression see
[`PERF_REPORT.md`](./PERF_REPORT.md).

## Platform & the fundamental constraint

- X280 L2CPU tile: **4 harts, 2 NIUs (NoC0/NoC1)**, in-order, RVV. NoC @ 1.35 GHz,
  **flit = 64 B**. Linux needs ≥1 hart → usable worker budget = **3 harts**.
- **TLB-window memory-mapped loads/stores are the ONLY NoC mechanism** (no DMA
  engine, no software NIU cmd-buf path — see "Async/DMA status" below).
- Therefore **latency-bound, not bandwidth-bound**: an in-order hart keeps ~1
  uncached NoC read outstanding (~300 ns / 64 B flit). Gains come from (a)
  overlapping reads *within* a flit (8× u64 in one expression), and (b) adding
  harts for independent reads in flight.
- Measurement boxes: pull-path numbers on **bh-qb-05 / bh-14 (P300)**; D2H export
  numbers on **bh-03 (P100A)**. HW facts (flit size, hart/NIU count, clock) carry
  over; the ~430 / 268 ceilings reproduced across boxes.

---

## Direction 1 — Tensix L1 → X280 (PULL / read)

Uncached System-Port reads over NoC0. MB/s = Mflit/s × 64 B.

| Scope | Bytes/core | Harts | Rate | Time | Note |
|---|---|:--:|---|---|---|
| 1 core, single u32 serial | 4 B | 1 | 15.6 MB/s | 257 ns/read | worst case |
| 1 core, full flit (8× u64 overlap) | 64 B | 1 | **202 MB/s** | 317 ns | read whole flit, always |
| 1 core, 4 KB, naive serial | 4 KB | 1 | ~34 MB/s | ~120 µs | early baseline |
| 1 core, 4 KB, flit-overlapped | 4 KB | 1 | **~207 MB/s** | ~19.8 µs | 309 ns/flit × 64 |
| All 110 cores | 64 B | 1 | 175 MB/s | 40 µs | 2.74 Mflit/s |
| All 110 cores | 64 B | 3 | **435 MB/s** | **16 µs** | 6.79 Mflit/s, 2.5× |
| All 110 cores | 4 KB | 1 | 207 MB/s | 2.1 ms | per-hart ceiling |
| All 110 cores | 4 KB | 3 | **430 MB/s** | **1.05 ms** | ~954 snapshots/s, 2.08× |

**The wall: ~430 MB/s at 3 harts**, identical for 64 B and 4 KB access patterns →
the read-response ceiling of the **single NoC0 NIU / tile egress port**. 4 KB
scales slightly worse than 64 B (2.08× vs 2.5×) because sustained 64-flit streams
saturate the port harder than the gap-y 64 B sweep. 4 harts COLLAPSE (OS
starvation). "Time" for all-core rows = **full-grid snapshot latency**.

### Sampling-frequency ceilings (110 cores, 3 harts, pull-only)

| Bytes/core | Snapshot | Max freq |
|---|---|---|
| 64 B (1 flit, marker/header) | 16 µs | **~62 kHz** |
| 4 KB (full L1 region) | 1.05 ms | **~1 kHz** |

Frequency scales ~inversely with bytes/core and with core count. 1 hart: 64 B →
~25 kHz, 4 KB → ~475 Hz.

---

## Direction 2 — X280 → Host (EXPORT / write)

Two transports evaluated. Device DRAM is off-limits under X280 Linux (Linux
occupies DRAM tiles), so the legacy "push to device-DRAM profiler buffer, host
PCIe-reads it" path is unavailable.

### (a) SLIRP user-mode network (works, slow) — bh-qb-05

| Stage | Rate |
|---|---|
| Raw TCP X280 → host | **13.9 MB/s (111 Mbit/s)** |
| 3-hart 4 KB poll + export bolted on | poll 326 MB/s, export 14.5 MB/s, **95.5 % dropped** |

Hard ~14.5 MB/s wall → can only ship deltas/markers, never full buffers.

### (b) D2H socket: X280 NoC-write → PCIe tile → host pinned FIFO (the fast path) — bh-03

| Mode | Rate | vs SLIRP |
|---|---|---|
| Raw write, 1 hart | 159 MB/s | 11× |
| **Raw write, 2 harts (ceiling)** | **268 MB/s** | **18×** |
| Raw write, 3–4 harts | collapse (63–67) | — |
| Raw write, 2 harts + NoC1 split | 243 MB/s | no gain |
| **Sustained lossless flow-controlled stream** | **~158 MB/s** | **11×** |

- **268 MB/s raw ceiling @ 2 harts** = the X280 NIU/PCIe-tile write path. Identical
  on P300A and P100A → not chip/fabric-bound. NoC1 split (props_lo `noc_sel`)
  does **not** help on the write side. 3+ harts collapse (OS starvation, same as
  reads).
- **~158 MB/s lossless streaming** is **host-drain-limited** (host single-thread
  memcpy out of pinned memory), NOT the X280; bigger pages (64 KB vs 4 KB) don't
  move it (155 vs 158). The X280 throttles to match, `bytes_acked` trailing
  `bytes_sent` by exactly one FIFO → **zero loss**.
- Writes are posted (don't stall the issuing hart); reads through the PCIe tile
  **hang** the in-order hart, so all export tooling is **write-only**.

### D2H addressing (settled)

- PCIe tile coord = the socket's **`pcie_xy_enc` = 0x613** (translated (19,24)),
  used as the X280 TLB-window `props_lo` verbatim. **winsel 0, NO bit-60** (those
  were bh-qb-05 red herrings). Tensix is addressed by physical coord (e.g. (1,2));
  only the PCIe TILE wants the translated coord.
- Single-chip box (P100A): metal **device 0**, plain auto-discovery.
  Multi-chip box (P300A): X280's chip is dropped from fabric while Linux runs —
  open it with a 1×1 MGD that **pins** the node to the X280's ASIC position.
- Flow control: host `D2HSocket::read(notify_sender=true)` writes cumulative
  `bytes_acked` back to sender L1 at **config+32 (word 8)**; X280 reads that (safe
  Tensix read) to throttle.

---

## Synthesis — combined pull→export pipeline

- **Pull (≤430 MB/s) is wider than export (268 raw / 158 lossless).** Export is the
  narrower pipe; lossless export is the binding constraint.
- **Both consume the same 4 harts.** Pull wants 3, export wants 2, OS needs 1 — they
  contend. You cannot run peak pull *and* peak export at once; the SLIRP-era
  integrated test already showed poll dropping 430 → 326 under exporter contention.
- **You can read faster than you can ship.** 1 kHz × 440 KB (full 4 KB grid) =
  440 MB/s pulled, which exceeds the export ceiling → full 4 KB snapshots at 1 kHz
  can be *read* but not *shipped whole*. Ship deltas/markers, compress, or drop.
- **Profiler budget:** ~158 MB/s lossless ≈ ~40M 4-byte markers/s aggregate. Pick
  bytes/core to hit the target sampling rate, then confirm the data volume fits the
  export budget.
- **D2H is the right primitive:** 11–18× faster than SLIRP; replaced the two dead
  ends (SLIRP network, device DRAM).

---

## Async / DMA status (can we offload the copy without halting the CPU?)

**Strongly established NO, but not exhaustively hardware-proven.**

Confirmed dead (empirical):
- **Software NIU command-buffer issue** — `x280_niu_read.c` programmed cmd buf 0
  like a Tensix `ncrisc_noc_fast_read` and fired: `CMD_ACCEPTED +0`,
  `RD_REQ_SENT +0`. The NIU front-end never takes a software-issued command — this
  was the one path that would have given async NoC→DRAM copy.

Inferred (docs + Linux enumeration):
- ISA `L2CPUTile/TLBWindows.md` + README: L2CPU NoC access is **exclusively via TLB
  windows**; no DMA engine in the tile.
- X280 Linux exposes **no DMA controller** (empty `/sys/class/dma`, nothing in
  `/proc/devices` or the device tree).

Consequence: every NoC transaction is a CPU instruction; a NoC **read stalls the
in-order hart** (the unavoidable CPU-halting step). Writes are posted (free).

NOT yet tested (the gap in "fully"):
1. An **undocumented DMA block not declared in the tt-bh-linux dtb** — we only
   checked what Linux enumerated, not the raw SoC address map.
2. A **hardware/autonomous NIU copy mode** distinct from the software cmd-buf path.
3. The SiFive X280 TRM for a **PL2 / system DMA** block.

Closing it fully = scan the L2CPU/X280 memory map for an undeclared DMA controller
+ check the X280 TRM, rather than trusting the dtb.

Also ruled out: **Cached Memory Port** gives non-blocking-cache read overlap but
returns stale data → useless for continuous monitoring.

---

## Tools (`tools/tracy/x280/`)

Pull:
- `pollall.c` — single-thread 64 B/core sweep over a coordfile.
- `pollalln.c` — sweep `<nbytes>`/core (used for 4 KB rows).
- `pollmt.c` — pthread version (has a ~2.2× per-thread handicap; trust multi-process
  `pollall`/`pollalln` pinned with `taskset`).

Export (D2H, write-only, addressing derived from socket config at runtime):
- `x280_d2h_send.c` — one-shot single-page send + verify.
- `x280_d2h_bw2.c` — multi-hart raw write BW; `<tx> <ty> <cfg> [nharts] [secs] [chunk] [noc_split]`.
- `x280_d2h_stream.c` — sustained lossless flow-controlled stream; `<tx> <ty> <cfg> [total_MB] [page] [max_secs]`.

Host side: `tt_metal/programming_examples/x280_d2h/` — modes `selftest | listen | hold | serve`.

Dead-ends / probes:
- `x280_niu_read.c`, `x280_niu_probe.c` — NIU register probes (DMA dead-end).
- `x280_pcie_scan.c` — safe read-scan (superseded; PCIe-tile reads hang on bh-14/bh-03).
