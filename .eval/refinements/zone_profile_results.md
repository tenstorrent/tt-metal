# rms_norm per-zone (DeviceZoneScope) profiling

Measured on 8x8 Wormhole b0 (1000 MHz, 1 cycle = 1 ns), bf16, no-gamma, TILE layout.
Each shape sized so every active core processes exactly ONE tile-row (zones fire once/core).

**Method.** `DeviceZoneScopedN(name)` emits a ZONE_START/ZONE_END timestamp pair per RISC into
the device profiler L1 buffer, dumped to `generated/profiler/.logs/profile_log_device.csv` on
`ttnn.ReadDeviceProfiler`. Reader/writer zones live on a single RISC (NCRISC / BRISC) = clean
wall time. Compute zones fire on all 3 TRISCs; reported on the MATH thread (TRISC_1) busy span
(pack/unpack threads enter the next zone early and idle-spin, inflating a cross-TRISC span).

**Critical caveat.** Reader, compute, writer run CONCURRENTLY, pipelined through circular
buffers — the zones OVERLAP in wall-clock and do NOT sum to the total. Total device-kernel time
is gated by the slowest pipeline stage, which is the WRITER in every case.

All values are per-core MEAN nanoseconds.

## Regime A (row-parallel, no cross-core gather)

| shape | Wt | RDR-input | CMP-p1-square | p1-reduce | CMP-finalize | CMP-pass2 | WR-write | total |
|-------|----|-----------|---------------|-----------|--------------|-----------|----------|-------|
| (2048,256) | 8  | 4835  | 6166  | 487  | 2436 | 2736 | 15265 | 18560 |
| (2048,512) | 16 | 9563  | 11663 | 1022 | 2428 | 9714 | 28076 | 32870 |

## Regime B (wide-W cross-core W-split + all-reduce)

| shape | cores(K) | RDR-input | p1-square | p1-reduce | RDR-ar-wait | RDR-ar-xport | CMP-combine* | finalize | pass2 | WR-write | total |
|-------|----------|-----------|-----------|-----------|-------------|--------------|--------------|----------|-------|----------|-------|
| (32,4096)  | 16 | 1653 | 2932 | 502  | 1824 | 3080 | 4588 | 2428 | 2298 | 15386 | 16636 |
| (32,8192)  | 32 | 2697 | 3973 | 498  | 1824 | 5575 | 8157 | 2436 | 2840 | 20615 | 22028 |
| (32,16384) | 32 | 4550 | 6590 | 1029 | 3113 | 6040 | 8602 | 2428 | 7661 | 28932 | 30923 |

\* `CMP-combine` is mostly the compute MATH thread STALLED on `cb_partials_gathered` — i.e. the
cross-core all-reduce barrier, not arithmetic. It tracks `RDR-ar-xport` (the gather+broadcast).

## Findings

1. **The op is DRAM-write-bandwidth bound in BOTH regimes.** `WR-write` wall ≈ total
   device-kernel time on every shape (e.g. (2048,512): writer 28.1µs of 32.9µs total;
   (32,16384): 28.9µs of 30.9µs). Everything else is hidden behind the writer's drain.
   This is direct evidence for the changelog's R6/R7 conclusion (memory-bound → row-blocking
   net-negative).

2. **Regime A compute** is dominated by the two O(Wt) phases — PASS-1 square and PASS-2
   normalize — both roughly double from Wt=8 to Wt=16. The reduce (~0.5µs) and finalize
   (~2.4µs, a fixed 1-tile rsqrt) are small constant costs.

3. **Regime B adds a serial cross-core all-reduce barrier absent in A**: `RDR-ar-xport`
   (3.1→5.6→6.0µs, grows with K / total grid cores) plus the matching `CMP-combine` stall
   (4.6→8.2→8.6µs). This is exactly the term R6 (K-tuning, picks K to minimize per-core
   reduce + transport) and R9 (root-relay O(1) transport) were tuning. `RDR-ar-wait`
   (~1.8–3.1µs) is the reader stalling on compute's PASS-1 before it can gather.

4. **finalize and p1-reduce are regime-invariant** (~2.4µs and ~0.5–1.0µs) — confirms the
   per-zone measurement is sound.
