# Quasar Cache-Write Performance Test (test id 912)

Measures single-DM-core write performance to Tensix L1 on Quasar, swept over total
data size (8B .. 2KB, multiples of 8), for four write modes (stamped as `Write path`):

- **0 — Uncached (1B stores)** — `base + MEM_L1_UNCACHED_BASE` (+4MB alias), written
  one byte at a time; straight to TL1, no flush.
- **1 — Uncached (8B stores)** — same uncached port, 64-bit stores (the natural
  DM-core store width).
- **2 — Cached + flush (fence per line)** — 64-bit stores to the cacheable window,
  then `flush_l2_cache_range(base, N)`, the library range flush, which fences twice
  per 64B line. This is the exact method used by the earlier standalone measurement.
- **3 — Cached + flush (single fence)** — identical cacheable 64-bit stores, then the
  touched 64B lines are flushed with bare flush-register writes (no per-line fence)
  and a single completion fence after all iterations. This is the efficient flush.

What the comparisons isolate: **0 vs 1** = store width on the uncached port; **1 vs 2**
= uncached vs cache+flush as originally measured; **2 vs 3** = the flush fencing cost.

## When to use which (write-only to L1)

- **Fire-and-forget writes, any size → uncached port with 8-byte stores.** Fastest at
  every size in the sweep.
- **Cache + flush → use the single-fence flush, not `flush_l2_cache_range`.** The
  library per-line fencing is much slower for multi-line ranges (e.g. ~4400 vs ~3900
  cyc at 2KB). Even optimized, cache+flush does not beat uncached-8B for write-only,
  no-reuse traffic — it pays off only when the data is subsequently read back, or when
  many scattered sub-64B updates coalesce in cache before one flush (not measured here).

## Measurement method

To measure steady-state (not one-shot) cost, each mode repeats its write(+flush)
region `num_iterations` times (default 100) inside a single `DeviceZoneScopedN`, and
the host divides the zone duration by the iteration count to get the amortized
per-write cost. Fencing discipline: the uncached modes (0/1) fence every iteration
(matching the standalone `DirectSram` baseline); mode 2 uses the library per-line
flush; mode 3 issues a single fence after all iterations. Sizes are multiples of 8 so
the 8-byte modes do exactly `N/8` stores with no sub-8B tail. The kernel stamps
`Test id`, `Number of transactions` (= iteration count), `Transaction size in bytes`,
and `Write path` (mode). Results are plotted as amortized per-write duration (cycles)
and bandwidth (bytes/cycle) vs data size.

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
profiler does not increment `run host ID` — every one of the 40 sweep runs
(4 modes x 10 sizes) is stamped `run_host_id = 0`. Because
`stats_collector.aggregate_performance` groups runs by `run_host_id`, all runs
collapse into a single aggregated point, so the auto-generated PNG shows one point
rather than the four per-mode curves.

The per-run data is still fully captured in
`generated/profiler/.logs/profile_log_device.csv`: each `RISCV1` `ZONE_START`/
`ZONE_END` pair is immediately followed, in order, by that run's
`Transaction size in bytes` and `Write path` stamps.

Stopgap — reconstruct the correct plot from the CSV order:
```bash
python tests/tt_metal/tt_metal/data_movement/quasar_cache_perf/plot_cache_write_from_csv.py
```
This walks the CSV in order (independent of `run_host_id`), recovers each run's
(size, path, duration) tuple, and writes a two-panel PNG (full range + a zoom on
the small-size crossover region) to `data/quasar/Quasar Cache Write Sizes.png`.

A proper fix (an order-based aggregation fallback when `run_host_id`s collide, or
incrementing `run_host_id` under slow dispatch) is deferred until this runs on
hardware / a fast-dispatch grid, where `run_host_id` increments normally and the
shared `--plot` pipeline plots the two curves directly.
