# Method 2 — Forge IR vs Isolated Test: Profiler Comparison

**Hardware:** Wormhole N150 · 64 Tensix cores · 288 GB/s peak DRAM BW

---

## Profiler Results (between signposts)

**Block A — 1×3×1536×1536**

| Op | Forge Count | Forge ms | Ours Count | Ours ms |
|----|:-----------:|--------:|:-----------:|--------:|
| `PermuteDeviceOperation` | 2 | 1.253 | 2 | 1.010 |
| `ReshapeViewDeviceOperation` | 1 | 0.720 | — | — |
| `TilizeDeviceOperation` | 1 | 0.190 | 1 | 0.184 |
| `MatmulDeviceOperation` | 1 | 0.234 | 1 | 0.418 |
| `UntilizeDeviceOperation` | 1 | 0.169 | 1 | 0.178 |
| `ShardedToInterleavedDeviceOperation` | 1 | 0.100 | — | — |
| **TOTAL** | | **2.666 ms** | | **1.791 ms** |

**Block C — 1×3×1280×2304**

| Op | Forge Count | Forge ms | Ours Count | Ours ms |
|----|:-----------:|--------:|:-----------:|--------:|
| `PermuteDeviceOperation` | 2 | 1.309 | 2 | 0.984 |
| `ReshapeViewDeviceOperation` | 1 | 1.053 | — | — |
| `TilizeDeviceOperation` | 1 | 0.218 | 1 | 0.212 |
| `MatmulDeviceOperation` | 1 | 0.292 | 1 | 0.533 |
| `UntilizeDeviceOperation` | 1 | 0.206 | 1 | 0.205 |
| `ShardedToInterleavedDeviceOperation` | 1 | 0.123 | — | — |
| **TOTAL** | | **3.201 ms** | | **1.934 ms** |

---

## Root Cause: ReshapeViewDeviceOperation on HEIGHT_SHARDED TILE tensor

### Which reshape is the device op?

Tracing the `forward()` function in the Forge TTNN IR (`yuv_conv2d_block_C_minimal`), there are four
`ttnn.reshape` calls. Three of them are free views (ROW_MAJOR DRAM, same underlying buffer).
The fourth — the first reshape in the unpack path — lands on a HEIGHT_SHARDED L1 TILE tensor:

```
linear output → tensor<1x1x92160x96xbf16>  TILE  L1_HEIGHT_SHARDED  (ttnn_layout18)
                                                         ↓
reshape [1,1,92160,96] → [1,40,2304,96]   TILE  L1_HEIGHT_SHARDED  (ttnn_layout19)
                                                         ↓  ReshapeViewDeviceOperation  0.72–1.05 ms
to_memory_config                           ROW_MAJOR  DRAM
                                                         ↓  ShardedToInterleavedDeviceOperation  0.10–0.12 ms
permute  ...
```

### Why it is not a free view

A `ttnn.reshape` on ROW_MAJOR DRAM returns a free view with zero kernel time: the element
addresses are identical before and after the reshape.

A `ttnn.reshape` on a HEIGHT_SHARDED TILE tensor cannot be a free view because:

1. **Sharding distributes data across 64 cores.** Each core holds a physical shard
   (`45×3` tiles for Block C). The shard boundaries are determined by the logical tensor shape.

2. **Reshape changes the tile index mapping.** Logical shape `[1, 1, 92160, 96]` has
   `92160/32 = 2880` tile rows. Shape `[1, 40, 2304, 96]` introduces a new dimension boundary
   at every 2304 elements (72 tile rows), which changes how tile indices map to per-core offsets.

3. **Metadata update requires a kernel dispatch.** TTNN must dispatch a lightweight kernel to all
   64 cores to re-register the local buffer view with the new logical dimensions. This is not free
   — it incurs full dispatch latency and per-core execution overhead, measured at 0.72–1.05 ms.

### Op flow comparison

**Forge IR (unpack path):**
```
linear   [1,1,92160,96]  TILE  L1_H_SHARDED
    ↓
reshape  [1,40,2304,96]  TILE  L1_H_SHARDED   ← ReshapeViewDeviceOperation  1.053 ms
    ↓
to_memory_config         ROW_MAJOR  DRAM       ← ShardedToInterleavedDeviceOperation  0.123 ms
    ↓
permute  [1,96,40,2304]  ROW_MAJOR  DRAM       ← PermuteDeviceOperation  1.309 ms
    ↓
reshape  [1,3,1280,2304] ROW_MAJOR  DRAM       ← free view  0 ms
```

**Our isolated test (unpack path):**
```
linear   [1,1,92160,96]  TILE  DRAM_INTERLEAVED
    ↓
to_layout ROW_MAJOR      ROW_MAJOR  DRAM        ← UntilizeDeviceOperation  0.205 ms
    ↓
reshape  [1,40,2304,96]  ROW_MAJOR  DRAM        ← free view  0 ms
    ↓
permute  [1,96,40,2304]  ROW_MAJOR  DRAM        ← PermuteDeviceOperation  0.984 ms
    ↓
reshape  [1,3,1280,2304] ROW_MAJOR  DRAM        ← free view  0 ms
```

### Fix

The `ReshapeViewDeviceOperation` (1.053 ms) + `ShardedToInterleavedDeviceOperation` (0.123 ms)
can be replaced by swapping their order — do `to_memory_config` first, then reshape:

```
linear   [1,1,92160,96]  TILE  L1_H_SHARDED
    ↓
to_memory_config         ROW_MAJOR  DRAM        ← ShardedToInterleavedDeviceOperation  ~0.12 ms
    ↓
reshape  [1,40,2304,96]  ROW_MAJOR  DRAM        ← free view  0 ms   ✓
```

This eliminates the `ReshapeViewDeviceOperation` entirely. The expected saving is 1.05 ms for
Block C and 0.72 ms for Block A, bringing the Forge total to within ~0.10–0.15 ms of our
isolated test result.
