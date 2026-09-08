# Matmul Decode Fused RMSNorm Performance Regression

## Summary

Commit `bf64bc61c53` ("WIP Working, but slow new LLKS") reworked the fused
RMSNorm statistics gather in the full-width-sharded `matmul_decode` path to
fix a correctness bug introduced by an LLK update. The fix works, but it made
the fused-RMSNorm matmul **2.8x to 3.6x slower**.

Effectively all of the lost time is a single word-at-a-time L1 zero-fill loop
that one core executes on the critical path. The buffer it clears grew 32x in
the same commit. The zero-fill is not required for correctness under the new
reduction scheme, so the regression is recoverable without giving up the bug
fix.

## Measurements

Two checkouts on this box bracket the change:

| Checkout | HEAD | Gather scheme |
| --- | --- | --- |
| `metal_2` | `be3e309f4b2` | old: parallel producer push + hardware reduce |
| `metal_1` | `bf64bc61c53` and later | new: hub-issued reads + elementwise adds |

`be3e309f4b2` is the immediate parent of `bf64bc61c53`, so `metal_2` is a
clean baseline with no other differences in the op.

Comparing `metal_2/generated/profiler/reports/2026_09_08_09_54_54` against
`metal_1/generated/profiler/reports/2026_09_08_10_07_01`. Same op, same
config: `M=1`, ROW_MAJOR height-sharded input A, WIDTH_SHARDED row-major
output, `rms_norm=true`, `rms_norm_gamma=0.75`, `rms_norm_epsilon=1e-5`.

| Config | Baseline | After fix | Regression |
| --- | --- | --- | --- |
| `K=4096`, `N=1024`, 16 cores | 7.07 us | 25.21 us | 3.6x |
| `K=1024`, `N=512`, 8 cores | 4.55 us | 12.87 us | 2.8x |

The `rms_norm=false` matmul in the same trace is unchanged (3.60 to 3.61 us
and 1.16 to 1.19 us), which localises the cost to the RMSNorm epilogue rather
than to the matmul or to the new LLKs in general.

## Root cause

### The offending loop

In `full_width_rms_norm_transport.hpp`, the hub clears the entire gathered
statistics buffer before the transfer, one 32-bit word per iteration:

```cpp
if (is_hub) {
    rms_gathered.reserve_back(packed_tiles);
    auto* packed_dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rms_gathered.get_write_ptr());
    for (uint32_t byte = 0; byte < M_tiles * packed_bytes_per_row; byte += sizeof(uint32_t)) {
        packed_dst[byte / sizeof(uint32_t)] = 0;
    }
}
```

The previous version cleared only the unused tail of each row:

```cpp
constexpr uint32_t valid_bytes_per_row = num_producers * local_tile_size;
for (uint32_t mt = 0; mt < M_tiles; ++mt) {
    const uint32_t tail_start = mt * packed_bytes_per_row + valid_bytes_per_row;
    const uint32_t tail_end = (mt + 1) * packed_bytes_per_row;
    for (uint32_t byte = tail_start; byte < tail_end; byte += sizeof(uint32_t)) {
        packed_dst[byte / sizeof(uint32_t)] = 0;
    }
}
```

### Why the buffer grew 32x

`full_width_sharded_program_factory.cpp` changed how the packed slot count is
derived:

```cpp
// before
const uint32_t rms_stats_per_reduce_tile = rms_reduce_tile_size / rms_tile_size;
const uint32_t rms_packed_tiles_per_row = div_up(num_producers, rms_stats_per_reduce_tile);

// after
rms_packed_tiles_per_row = num_producers;
```

The relevant geometry for the row-major path:

- `rms_fp32_stats = rms_norm && !use_custom_mm`, and the row-major Blackhole
  path selects `custom_mm`, so the statistics are **BF16**.
- `rms_tile_size` is the local stat page: `output_tile` is 1x32, so BF16 gives
  **64 bytes**.
- `rms_reduce_tile_size` is a full 32x32 tile: BF16 gives **2048 bytes**.
- `M_tiles = div_up(M, inputA_tile_height) = div_up(1, 1) = 1`.

So `rms_stats_per_reduce_tile` was `2048 / 64 = 32`. Up to 32 producers packed
their 64-byte pages into **one** reduction tile, and `div_up(16, 32)` and
`div_up(8, 32)` were both `1`. After the change each producer gets a **whole**
2048-byte tile.

Combined with the switch from tail-only to whole-buffer clearing:

| Producers | Buffer before | Buffer after | Words cleared before | Words cleared after | Extra words |
| --- | --- | --- | --- | --- | --- |
| 16 | 2048 B | 32768 B | 256 | 8192 | 7936 |
| 8 | 2048 B | 16384 B | 384 | 4096 | 3712 |

### The arithmetic matches the measurement

| Producers | Extra words | Measured delta | ns per word |
| --- | --- | --- | --- |
| 16 | 7936 | 18.14 us | 2.29 |
| 8 | 3712 | 8.31 us | 2.24 |

Both configurations land on the same ~2.25 ns per word, which is roughly three
cycles for a volatile store plus loop overhead on a single data-movement RISC.
The regression scales linearly with producer count, exactly as a per-producer
2048-byte clear would. This accounts for essentially all of the lost time; the
NOC and math changes below are minor by comparison.

## Secondary costs in the same commit

These are real but small next to the zero-fill.

**The gather reversed direction.** Producers used to `async_write` their page
straight into a disjoint slot on the hub, all in parallel, overlapped with the
arrival semaphore. Now every producer only signals arrival, and the hub, after
waiting on the barrier, issues `num_producers` serial `async_read`s. The
transfer moved off the parallel path and onto one core's critical path, and the
producer physical coordinates now have to be passed as runtime args
(`producer_coords_arg_base`) so the hub can address them.

**The hardware reduction was replaced by hand-rolled adds.** In
`compute_full_width_sharded.cpp`, a single
`reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR>` over one packed tile became
`num_producers / 2` `add_tiles` calls over `num_producers` full tiles, with a
`binary_tiles_init` reconfiguration per M row.

**Per-core L1 grew 16x.** The gathered CB is allocated over
`inputB_core_range_set`, so all 16 producer cores now reserve 32 KB each
instead of 2 KB, even though only the hub ever reads it. The comment explains
the uniform allocation is needed so producers can derive the hub destination
from their own write pointer.

## Latent correctness bug in the new reduction

The accumulate loop in `compute_full_width_sharded.cpp` pairs slots two at a
time:

```cpp
for (uint32_t p = 2; p < rms_packed_tiles_per_row; p += 2) {
    add_tiles(rms_gathered_cb_id, rms_gathered_cb_id, row_start + p, row_start + p + 1, 0);
}
```

When `rms_packed_tiles_per_row` is odd, the final iteration reads slot
`p + 1`, which is past the end of the row: either the next M row's slot 0,
which is real data and silently corrupts the sum, or past the CB allocation
entirely. It is masked today only because 8 and 16 producers are both even.

This matters for the in-progress grouped RMSNorm work, which sets
`rms_packed_tiles_per_row = rms_max_hub_contributors`. Contributor counts are
derived from group-to-shard intersection and will readily be odd.

## Why the regression is recoverable

The new scheme only ever reads element `[0, 0]` of the accumulated tile, and
`add_tiles` is lane-wise: `result[i] = sum over p of page_p[i]`. Garbage in
lanes 1 through 31, or in the three faces beyond the first 64 bytes, can never
contaminate lane `[0, 0]`. NaN or Inf bit patterns in uninitialised L1 stay in
their own lane.

The whole-buffer zero-fill is therefore not needed for correctness. Its only
real function today is to make the odd-count overrun above read a defined
tile, and that is better fixed directly.

### Options

1. **Minimal.** Delete the whole-buffer zero-fill, fix the odd-contributor
   overrun in the add loop, and restore the parallel producer-push gather.
   Keeps the per-producer-tile layout that fixed the LLK bug. Should recover
   nearly all 18 us.
2. **Full revert of the layout.** Return to the packed single-tile layout and
   the hardware `REDUCE_SCALAR`, and have each producer zero the tail of its
   own 64-byte page. That restores the all-zero invariant the old code relied
   on, in parallel across producers, at ~31 words per core rather than 8192
   words on one core.
3. **Confirm the LLK behaviour first.** Establish exactly what the LLK update
   changed about `REDUCE_SCALAR` packing into a 1x32 tile, so it is clear
   which invariant can be relied on before reworking the transport again.

## Notes on the timeline

`metal_1`'s own run at `2026_09_08_10_56_11` shows 7.22 us and 4.57 us, back
at baseline, so a fast variant existed in the working tree at that point. The
`reset: moving to HEAD` entry in the reflog at 11:18 discarded those
uncommitted kernel edits and restored the slow versions now in the tree. Those
edits are not recoverable from git.

## Verification status

Analysis is from source inspection and from profiler CSVs already present in
both checkouts. No device runs were performed for this document. Any fix needs
on-silicon re-measurement of the two configurations in the table above, plus
the fused RMSNorm PCC tests, since the numbers here come from existing traces
rather than from a controlled A/B run.
