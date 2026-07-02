# `bh_dram_read`: a Blackhole DRAM read-bandwidth microbenchmark

A worked, reproducible example of saturating Blackhole DRAM read bandwidth,
companion to
[Saturating_DRAM_bandwidth_Blackhole.md](Saturating_DRAM_bandwidth_Blackhole.md).
It walks through the `bh_dram_read` example op — a minimal, read-only TTNN op
that places one worker core per DRAM bank, has each core stream its bank's
portion of a DRAM-interleaved tensor into L1, and discards it — and shows how it
reaches ~98% of peak.

## What the op does

- Host API: `ttnn.bh_dram_read(input_tensor)` (read-only, returns nothing).
- One reader core per DRAM bank (`device->num_dram_channels()` = 8 on p150).
- Each bank's reader is placed on the bank-adjacent worker core for NOC0 via
  `device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0)`.
- Each core streams its bank's contiguous region in 16 KB (`NOC_MAX_BURST_SIZE`)
  packets, keeping `NUM_TRIDS` reads in flight (default 2 — double-buffered).

Source: `ttnn/cpp/ttnn/operations/examples/bh_dram_read/`.

## The reader kernel (core of the benchmark)

Each core reads from a single fixed bank, so the NOC read state is set once and
constant-size packets are streamed with a transaction-id ring:

```cpp
// reader_bh_dram_read.cpp (abridged)
constexpr uint32_t NUM_TRIDS = get_compile_time_arg_val(0);   // buffering depth
constexpr uint32_t PACKET    = NOC_MAX_BURST_SIZE;            // 16 KB on Blackhole

const uint64_t bank_noc_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
const uint32_t num_packets   = (region_bytes + PACKET - 1) / PACKET;  // round up: over-read tail

noc_async_read_one_packet_set_state(bank_noc_base, PACKET);   // size/bank fixed -> set once
uint32_t trid_issue = 1, trid_wait = 1, in_flight = 0, offset = src_addr;
for (uint32_t i = 0; i < num_packets; ++i) {
    noc_async_read_set_trid(trid_issue);
    noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
        bank_noc_base, offset, l1_base + (trid_issue - 1) * PACKET, trid_issue);
    offset += PACKET;
    ADVANCE_TRID(trid_issue);
    if (++in_flight == NUM_TRIDS) {                           // barrier only when the ring is full
        noc_async_read_barrier_with_trid(trid_wait);
        ADVANCE_TRID(trid_wait);
        --in_flight;
    }
}
while (in_flight > 0) { noc_async_read_barrier_with_trid(trid_wait); ADVANCE_TRID(trid_wait); --in_flight; }
```

This pattern is adapted from the MoE `dm0.cpp` reader. Because each core owns one
bank, `set_state` is hoisted out of the loop entirely.

## Build

The device profiler (Tracy) is enabled by default in `build_metal.sh`:

```bash
./build_metal.sh
source python_env/bin/activate
```

## Run / correctness

```bash
pytest tests/ttnn/unit_tests/operations/debug/test_bh_dram_read.py -q
```

The test covers awkward tile counts (1 tile, fewer tiles than banks, prime
counts that exercise the over-read tail, large) to confirm the op reads any
number of tiles.

## Measure bandwidth

`measure_bh_dram_read_bw.py` runs the op on a large tensor under the device
profiler, parses the `DRAM_READ` kernel zone from
`generated/profiler/.logs/profile_log_device.csv`, and reports bytes/cycle and
utilization (`aggregate = total_bytes ÷ slowest-core-cycles`):

```bash
TT_METAL_DEVICE_PROFILER=1 python3 measure_bh_dram_read_bw.py
```

Expected (8192×8192 bf16, p150):

```
reader cores (banks)  : 8
per-core bytes/cycle  : 46.34 B/cyc
AGGREGATE bandwidth   : 500.5 GB/s
theoretical peak      : 512 GB/s
UTILIZATION           : 97.7 %
```

## Tuning sweep (buffering depth)

`NUM_TRIDS` is a kernel compile-time arg; the program factory reads the
`BH_DRAM_READ_NUM_TRIDS` env var (1–15) to override it for sweeps:

```bash
bash sweep_bh_dram_read_trids.sh
```

| depth | GB/s | utilization |
|---|---|---|
| 1 | 186 | 36% |
| 2 | 344* | 67%* |
| 3–15 | 344* | 67%* |

\* with arbitrary core placement; with bank-adjacent placement the depth-2 run
reaches 500 GB/s / ~98%. The sweep shows buffering saturates at depth 2 — the
default. (The 344 → 500 jump comes from placement, not buffering; see the
companion doc.)

## How the numbers were obtained

- Timing: on-device profiler `DeviceZoneScopedN("DRAM_READ")` around the read
  loop; cycles read from the profiler CSV, converted with AICLK (~1350 MHz).
- Bytes: `physical_volume_in_tiles × tile_bytes` (the true tensor size; the
  rounded-up over-read of the final packet is excluded from the denominator).
- Aggregate BW: the 8 banks are read concurrently, so wall time is the slowest
  core; `aggregate B/cyc = total_bytes ÷ slowest_core_cycles`.
