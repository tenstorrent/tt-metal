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
