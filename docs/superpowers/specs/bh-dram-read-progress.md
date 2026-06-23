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
| `edc51d68467` | feat: add bh_dram_read op (one core per DRAM bank) — initial scaffold, serialized read-barrier-per-page kernel |

(Updated as work lands. Newer commits appended below once made.)

## Results

Tensor: 8192×8192 bf16 = 128 MiB, 65,536 tiles, 8 reader cores (one per bank).

| version | kernel | per-core B/cyc | aggregate B/cyc | GB/s | utilization |
|---------|--------|----------------|-----------------|------|-------------|
| v1 | TensorAccessor page-by-page, block of 8 + barrier per block | 17.4 | 139.3 | 188 | ~37% |
| v2 | trid ring (8 in flight), 16 KB max NOC packets (MoE dm0 pattern) | 31.8 | 254.7 | 344 | ~67% |

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

## Next ideas (later work)
- Sweep `NUM_TRIDS` (8 → 16) and packet size; find the knee.
- Use both NOCs / multiple reader cores per bank.
- Verify arbitrary tile counts (non-multiple-of-packet tails, < num_banks tiles).
