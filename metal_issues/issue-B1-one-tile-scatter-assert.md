# DRAFT upstream issue — all_gather_async trips scatter-write `chunk_count>=2` assert on a one-tile packet (Blackhole 1×4)

> Draft for filing at `tenstorrent/tt-metal`. Not yet submitted — awaiting go-ahead. Fill the `<exact ...>` slots from the recovered probe before filing.

**Component / Area:** ops / CCL (fabric)
**Issue Type:** Deterministic hang (watcher assert)
**Board:** Blackhole P300 (qb2), 1×4 mesh, Line topology

## Observed
`ttnn.experimental.all_gather_async` deterministically trips a watcher `assert_and_hang` when a gather produces a packet carrying **exactly one tile**:

```
Watcher stopped the device due to tripped assert (watcher_device_reader.cpp:811)
TT_THROW: Watcher detected tripped assert and stopped device. (assert.hpp:104)
fault: ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp
     → populate_unicast_scatter_write_fields (inlined)
     → tt_metal/fabric/hw/inc/api_common.h  ASSERT(chunk_count >= NOC_SCATTER_WRITE_MIN_CHUNKS && chunk_count <= NOC_SCATTER_WRITE_MAX_CHUNKS)
```

## Root cause
`minimal_default_writer.cpp` sends gathered tiles via the fabric **unicast *scatter* write** path, batching up to `num_tiles_to_write_per_packet` (compile-time arg, `static_assert <= 4`) tiles per packet:

```cpp
uint32_t tiles_to_put_in_current_packet = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
...
fabric_unicast_noc_scatter_write_with_state<...>(
    ..., pkt_scatter_hdr, ...,
    NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, tiles_to_put_in_current_packet), ...);
```

`populate_unicast_scatter_write_fields()` asserts the chunk count is in `[NOC_SCATTER_WRITE_MIN_CHUNKS(=2), NOC_SCATTER_WRITE_MAX_CHUNKS(=4)]` (originally messaged *"scatter chunk_count must be between 2 and 4"*, #32395; message dropped in #33819). The scatter path is **designed for ≥2 chunks**, but the writer feeds it `tiles_to_put_in_current_packet == 1` whenever:
- the per-packet tile count degenerates to 1 (e.g. **FP32** payload where 1 tile already fills the packet), or
- a **tail/remainder** packet has a single tile (total tiles per batch-head not a multiple of `num_tiles_to_write_per_packet`).

`chunk_count == 1 < 2` → assert.

## Where it bites in practice
The on-device **split-greedy / top-k top-p sampler** on large-vocab models (vocab-sharded LM-head → gather candidate logits across the 4 devices) produces a tiny FP32 candidate tensor → one tile per packet → deterministic hang. This is the `all_gather_async` path recommended as the workaround in #48404 / #48469 for the trace-unsafe prim `ttnn.all_gather` — so models that adopt that workaround hit *this* assert next. Reproduced independently on gpt-oss-20b, Qwen3-32B, and Qwen2.5-Coder-32B during full-model bringup.

## Steps (minimal standalone repro, no model)
See `ccl_one_tile_scatter_assert.py` (attached). Summary:
- 1×4 mesh, `FabricConfig.FABRIC_1D_RING`, `ttnn.all_gather(dim=1, num_links=2, cluster_axis=None, topology=Topology.Linear)`
- input per device `[1,1,64,32]` → output `[1,4,64,32]`, dtype **`float32`**, `TILE_LAYOUT`, `DRAM` interleaved, `ShardTensor2dMesh(dims=(None,1), mesh_shape=(1,4))`
- run under `TT_METAL_WATCHER=120` → BRISC assert at `api_common.h` (~L260/277) on all replicas, watcher stops the device. Failing core dev0 (x=4,y=0), waypoint `NSMD`.
- Changing dtype to `bfloat16` (2 tiles/packet → chunk_count==2) makes the identical op pass — isolates the one-chunk trigger. (With watcher off the ASSERT compiles out and the bug is silent.)

## Expected
`all_gather_async` should handle a one-tile packet without asserting.

## Suggested fix (validated)
In `minimal_default_writer.cpp` (~L307), guard the scatter-header set-state + route-setup with `if constexpr (num_tiles_to_write_per_packet > 1)`. The one-tile data path already sends via the ordinary unicast header (`pkt_unicast_hdr`), so the scatter header is **dead** when `num_tiles_to_write_per_packet == 1` — it is only pre-initialized, and the folded compile-time test leaves the assert unconditional. Guarding the pre-init removes the assert without touching the (correct) HW contract in `api_common.h`. Diff:

```diff
     if (detail::valid_targets(direction)) {
         static_assert(num_tiles_to_write_per_packet <= 4, "tiles per packet > 4 is unsupported");
-        uint64_t dummy_addrs[4] = {0, 0, 0, 0};
-        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
-        fabric_unicast_noc_scatter_write_set_state<...>(
-            pkt_scatter_hdr, ...,
-            NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, num_tiles_to_write_per_packet),
-            page_size * num_tiles_to_write_per_packet);
+        if constexpr (num_tiles_to_write_per_packet > 1) {
+            uint64_t dummy_addrs[4] = {0, 0, 0, 0};
+            uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
+            fabric_unicast_noc_scatter_write_set_state<...>(
+                pkt_scatter_hdr, ...,
+                NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, num_tiles_to_write_per_packet),
+                page_size * num_tiles_to_write_per_packet);
+            ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
+        }
```
Confirmed on all 4 Blackhole replicas for FP32 and bf16 after rebuilding `_ttnn.so`.

## Related
- #48222 (on-device sampler corrupts at batch-32), #48404 (PR, workaround → this path), #48469 (all_gather barrier not trace-safe), #40592 (AllGatherAsync intermittent hang T3K).
