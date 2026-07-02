# bh_dram_read — progress & results log

Tracking doc for the `bh_dram_read` op (DRAM read-bandwidth experiment). One
worker core per DRAM bank; each core reads its bank's portion of a DRAM-
interleaved tilized tensor and discards it. Read-only, void host API.

Branches: `vsuresh/bh_dram_read` (spec only), `vsuresh/bh_dram_read_impl` (code).
Device: Blackhole p150, AICLK ≈ 1350 MHz, 8 DRAM banks. Theoretical peak DRAM
BW (tt-metal microbenchmark spec constant): **512 GB/s**.

Measurement: `measure_bh_dram_read_bw.py` runs the op on a large tensor under
`TT_METAL_DEVICE_PROFILER=1`, parses the `DRAM_READ` kernel zone from
`generated/profiler/.logs/profile_log_device.csv`, and computes
aggregate bytes/cycle = total_bytes / slowest-core-cycles.

## Commits

| commit | description |
|--------|-------------|
| `ca587a583cf` | docs: bh_dram_read op design spec |
| `edc51d68467` | feat: add bh_dram_read op (one core per DRAM bank) — initial scaffold (v1, serialized read-barrier-per-page kernel) |
| `670053b020f` | perf: trid pipeline + max-packet over-read (v2, 188→344 GB/s); adds measurement script + this log |

(Updated as work lands. Newer commits appended below once made.)

## Results

Tensor: 8192×8192 bf16 = 128 MiB, 65,536 tiles, 8 reader cores (one per bank).

| version | kernel | per-core B/cyc | aggregate B/cyc | GB/s | utilization |
|---------|--------|----------------|-----------------|------|-------------|
| v1 | TensorAccessor page-by-page, block of 8 + barrier per block | 17.4 | 139.3 | 188 | ~37% |
| v2 | trid ring (8 in flight), 16 KB max NOC packets (MoE dm0 pattern) | 31.8 | 254.7 | 344 | ~67% |
| v3 | v2 + **optimal NOC0 bank-adjacent core placement** (depth 2) | 46.3 | 370.7 | **500** | **~98%** |

### Notes
- v1: serialized-ish; barrier per 8-page block, single NOC, page-granular reads.
- v2: copies the MoE `dm0.cpp` transaction-id pipeline — `set_state` once per
  core (fixed bank), then `noc_async_read_one_packet_with_state_with_trid`
  reads of `NOC_MAX_BURST_SIZE` (16 KB on BH) with `NUM_TRIDS=8` outstanding,
  barriering a trid only when the ring is full. Goal: approach 512 GB/s.
- v2 reads constant max-size packets and rounds the packet count up: the last
  packet over-reads past the bank region (discarded, so harmless). The CB is
  plain tile-paged L1 scratch sized for NUM_TRIDS packets — not packet-paged.
  187→344 GB/s = ~1.8× over v1; identical numbers before/after the over-read
  refactor (the 8192-tile case is an exact multiple of 16 KB).

## Buffering-depth (NUM_TRIDS) sweep

8192×8192 bf16, 8 cores. NUM_TRIDS made a kernel compile-time arg (factory reads
`BH_DRAM_READ_NUM_TRIDS` env, default now 2). `sweep_bh_dram_read_trids.sh`.

| depth | GB/s | util |
|-------|------|------|
| 1 (single buffer) | 186 | 36% |
| 2 (double buffer) | 344 | 67% |
| 3 | 344 | 67% |
| 4 | 344 | 67% |
| 6 | 344 | 67% |
| 8 | 344 | 67% |
| 12 | 344 | 67% |
| 15 | 344 | 67% |

**Conclusion:** double-buffering is sufficient — 1→2 doubles BW (latency hidden
by 2 outstanding reads), 2→15 is byte-identical (no gain, no over-issue
penalty). At v2's arbitrary core placement the 67% wall was **not** a buffering
limit. Default set to 2 (same BW as 8, 4× less L1).

## Optimal core placement (v3)

Switched core selection from "first 8 worker cores" to
`device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0)` —
the worker core bank-adjacent for NOC0 reads. **344 → 500 GB/s (67% → ~98%).**
Per-core jumps to 46.3 B/cyc, essentially the per-bank DRAM ceiling
(512/8 GB/s ÷ 1.35 GHz ≈ 47.4 B/cyc). So the v2 bottleneck was **core
placement / NOC distance**, not buffering or packet size. We are now within ~2%
of peak DRAM bandwidth on a single NOC.

## Next ideas (later work)
- Dual-NOC (8 more bank-adjacent cores on NOC1, split each bank's region) —
  likely marginal now that a single NOC already reaches ~98%.
- Sweep packet size.
