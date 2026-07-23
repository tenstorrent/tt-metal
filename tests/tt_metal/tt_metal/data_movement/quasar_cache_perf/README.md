# Quasar Cache-Write Performance Test (test id 912)

Measures single-DM-core write performance to Tensix L1 on Quasar via two paths,
swept over total data size (1B .. 2KB, powers of two):

- **Uncached** — stores to `base + MEM_L1_UNCACHED_BASE` (+4MB alias); straight
  to TL1, no flush.
- **Cached+Flush** — stores to the cacheable window (`0..4MB`), then
  `flush_l2_cache_range(base, N)` (flushes `ceil(N/64)` 64B lines).

The kernel times each `write(+flush)` region with `DeviceZoneScopedN` and stamps
`Test id`, `Number of transactions` (=1), `Transaction size in bytes` (=N), and
`Write path` (0=uncached, 1=cached+flush).

## Requirements

Quasar emulator: `ARCH_NAME=quasar` and `TT_METAL_SIMULATOR` pointing at an
`emu-quasar-*` build (do NOT set it to `1` — it is a path). The 1x3 emu has no
fast-dispatch cores, so slow dispatch is required.

## Running

Functional (read-back correctness):
```bash
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_data_movement \
  --gtest_filter="*QuasarCacheWrite*"
```

Performance + plot:
```bash
export TT_METAL_SLOW_DISPATCH_MODE=1
pytest tests/tt_metal/tt_metal/data_movement/python/test_data_movement.py \
  --plot --report --arch=quasar --gtest-filter="*QuasarCacheWrite*"
```

Output: `data/quasar/Quasar Cache Write Sizes.png` — duration (cycles) vs total
data size, one line per path.

## Known limitation on the 1x3 emulator (slow dispatch)

The `emu-quasar-1x3` target has no fast-dispatch cores, so tests must run in
slow-dispatch mode (`TT_METAL_SLOW_DISPATCH_MODE=1`). Under slow dispatch the
profiler does not increment `run host ID` — every one of the 24 sweep runs is
stamped `run_host_id = 0`. Because `stats_collector.aggregate_performance`
groups runs by `run_host_id`, all runs collapse into a single aggregated point,
so the auto-generated PNG shows one point rather than the two 12-point curves.

The per-run data is still fully captured in
`generated/profiler/.logs/profile_log_device.csv`: each `RISCV1` `ZONE_START`/
`ZONE_END` pair is immediately followed, in order, by that run's
`Transaction size in bytes` and `Write path` stamps. The correct per-run tuples
can be reconstructed by walking the CSV in order. A proper fix (an order-based
aggregation fallback when `run_host_id`s collide, or incrementing `run_host_id`
under slow dispatch) is deferred until this runs on hardware / a fast-dispatch
grid, where `run_host_id` increments normally and the shared pipeline plots the
two curves directly.
