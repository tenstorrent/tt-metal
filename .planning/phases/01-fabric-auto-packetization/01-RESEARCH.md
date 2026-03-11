# Phase 1: Fabric Auto-Packetization - Research

**Researched:** 2026-03-10
**Domain:** TT-Fabric hardware kernel API (C++ device-side headers)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- All raw-size write APIs (non-addrgen variants taking explicit `size` + `NocUnicastCommandHeader`) are updated to auto-packetize
- Maintain a local `current_noc_addr` variable initialized from the command header; increment by chunk size each iteration
- Call `noc_async_writes_flushed()` before each header rewrite (before modifying packet header for the next chunk)
- Applies to: unicast writes, multicast writes (1D linear + 2D mesh `MeshMcastRange`), scatter writes (unicast + multicast)
- Does NOT apply to: inline_write, atomic_inc
- For connection_manager variants with multiple connections: send chunk N to ALL connections before chunk N+1 (breadth-first)
- Flush once after all connections complete chunk N, before starting chunk N+1
- All existing raw-size write/scatter/fused APIs being updated get renamed to `_single_packet` suffix
- New auto-packetizing wrappers keep the original names and call `_single_packet` internally
- Only rename APIs that are being updated (not inline_write/atomic_inc)
- Applies to BOTH `linear/api.h` and `mesh/api.h`
- `_set_state` APIs continue to initialize header with `min(page_size, FABRIC_MAX_PACKET_SIZE)` — unchanged
- `_with_state` addrgen overloads remain unchanged
- Raw-size `_with_state` APIs: one call handles full page send internally
- `atomic_inc` fires only on the final chunk for fused ops
- `linear/api.h` addrgen-based APIs: already complete — no changes needed
- `mesh/api.h` gaps: `multicast_fused_scatter_write_atomic_inc` addrgen variants need packetization
- Raw scatter write (non-addrgen, two pre-computed NOC addresses): stays as-is
- Sparse multicast: linear-only, no mesh equivalent needed
- New dedicated unit tests covering `N * FABRIC_MAX_PACKET_SIZE + remainder` payloads

### Claude's Discretion

- Exact internal variable naming within chunking loops
- Whether to extract NOC address from command header via a helper or inline
- Organization of `_single_packet` API documentation

### Deferred Ideas (OUT OF SCOPE)

- Sparse multicast for mesh/api.h
- Inline_write and atomic_inc packetization
- Raw scatter write auto-packetization (non-addrgen, two pre-computed NOC addresses)
</user_constraints>

---

## Summary

This phase adds transparent large-payload support to the raw-size (non-addrgen) fabric write APIs in `linear/api.h` and `mesh/api.h`. The addrgen overloads in `linear/api.h` already implement chunking loops using the established while-loop-with-`noc_async_writes_flushed()` pattern. The raw-size APIs today are single-packet only. The work is to rename each raw-size write/scatter/fused function to `_single_packet`, then add a new wrapper under the original name that loops over `FABRIC_MAX_PACKET_SIZE`-sized chunks.

The `NocUnicastCommandHeader` struct holds a single `uint64_t noc_address` field. The raw-size wrapper extracts this address, increments a local variable per chunk, and constructs a fresh `NocUnicastCommandHeader{current_noc_addr}` each iteration. No special helper is needed; this is exactly how the existing addrgen loops work (see `fabric_unicast_noc_fused_unicast_with_atomic_inc` addrgen overload in `linear/api.h` lines 2763-2814).

The connection-manager variants require restructuring from "for each header, send all chunks" (depth-first) to "for each chunk, send to all headers" (breadth-first). `PacketHeaderPool::for_each_header` is a callback-based iterator; the breadth-first rewrite wraps the entire chunk loop around a manual header iteration, or calls `for_each_header` once per chunk.

**Primary recommendation:** Use the existing addrgen loop pattern verbatim for raw-size wrappers, extracting `current_noc_addr` from `noc_unicast_command_header.noc_address` as a `uint64_t` and incrementing it per chunk.

---

## Complete API Inventory

### Research method

All function names were enumerated by grepping for `FORCE_INLINE void fabric_` in both header files. Raw-size functions are those taking an explicit `uint32_t size` parameter (plus a typed command header). The addrgen overloads are distinguished by an `AddrGenType` template parameter. The `_with_state` and `_set_state` variants that take no `size` or have `UpdateMask` templates but no addrgen are also counted as raw-size.

---

### linear/api.h — Raw-Size Functions Needing `_single_packet` Rename

These are in namespace `tt::tt_fabric::linear::experimental`. Each entry means: rename the listed function to `_single_packet`, add a new wrapper under the original name.

**Unicast write (raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_unicast_noc_unicast_write` | `fabric_unicast_noc_unicast_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastCommandHeader, num_hops` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastCommandHeader, uint8_t* num_hops` |

Note: `_with_state` and `_set_state` for unicast_write are NOT renamed — they do not carry an explicit `size` meant to be chunked; the `_with_state` addrgen overload already handles chunking.

**Unicast scatter write (raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_unicast_noc_scatter_write` | `fabric_unicast_noc_scatter_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastScatterCommandHeader, num_hops` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastScatterCommandHeader, uint8_t* num_hops` |

**Unicast fused scatter write + atomic inc (raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_unicast_noc_fused_scatter_write_atomic_inc` | `fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastScatterAtomicIncFusedCommandHeader, num_hops` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastScatterAtomicIncFusedCommandHeader, uint8_t* num_hops` |

**Unicast fused unicast write + atomic inc (raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_unicast_noc_fused_unicast_with_atomic_inc` | `fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastAtomicIncFusedCommandHeader, num_hops` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastAtomicIncFusedCommandHeader, uint8_t* num_hops` |

**Multicast unicast write (1D linear, raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_multicast_noc_unicast_write` | `fabric_multicast_noc_unicast_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastCommandHeader, start_distance, range` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastCommandHeader, uint8_t* start_distance, uint8_t* range` |

**Multicast scatter write (1D linear, raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_multicast_noc_scatter_write` | `fabric_multicast_noc_scatter_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastScatterCommandHeader, start_distance, range` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastScatterCommandHeader, uint8_t* start_distance, uint8_t* range` |

**Multicast fused unicast with atomic inc (1D linear, raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_multicast_noc_fused_unicast_with_atomic_inc` | `fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastAtomicIncFusedCommandHeader, start_distance, range` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastAtomicIncFusedCommandHeader, uint8_t* start_distance, uint8_t* range` |

**Multicast fused scatter write + atomic inc (1D linear, raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_multicast_noc_fused_scatter_write_atomic_inc` | `fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastScatterAtomicIncFusedCommandHeader, start_distance, range` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastScatterAtomicIncFusedCommandHeader, uint8_t* start_distance, uint8_t* range` |

**Sparse multicast unicast write (1D linear, raw-size) — IN SCOPE per decisions:**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_sparse_multicast_noc_unicast_write` | `fabric_sparse_multicast_noc_unicast_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, src, size, NocUnicastCommandHeader, hops` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastCommandHeader, uint16_t* hops` |

Note: The CONTEXT.md says "sparse multicast: linear-only, no mesh equivalent needed" meaning mesh does not need a sparse multicast implementation added, but the existing linear sparse multicast raw-size function is a raw-size write function and should receive the rename + wrapper treatment.

**NOT renamed (excluded by decision):**
- `fabric_unicast_noc_unicast_inline_write` (and variants)
- `fabric_multicast_noc_unicast_inline_write` (and variants)
- `fabric_unicast_noc_unicast_atomic_inc` (and variants)
- `fabric_multicast_noc_unicast_atomic_inc` (and variants)
- All `_with_state` / `_set_state` variants (these are either already stateful with no `size` argument, or are the addrgen overloads which already chunk)

---

### mesh/api.h — Raw-Size Functions Needing `_single_packet` Rename

These are in namespace `tt::tt_fabric::mesh::experimental`. The mesh raw-size functions use `(dst_dev_id, dst_mesh_id)` or `(RoutingPlaneConnectionManager&, route_id)` instead of `num_hops`. The `MeshMcastRange` struct carries `{e, w, n, s}` hop counts for 2D multicast.

**Unicast write (raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_unicast_noc_unicast_write` | `fabric_unicast_noc_unicast_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, src, size, NocUnicastCommandHeader` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastCommandHeader` |

**Unicast scatter write (raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_unicast_noc_scatter_write` | `fabric_unicast_noc_scatter_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, src, size, NocUnicastScatterCommandHeader` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastScatterCommandHeader` |

**Unicast fused scatter write + atomic inc (raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_unicast_noc_fused_scatter_write_atomic_inc` | `fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet` | (a) `FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, src, size, NocUnicastScatterAtomicIncFusedCommandHeader` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastScatterAtomicIncFusedCommandHeader` |

**Unicast fused unicast write + atomic inc (raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_unicast_noc_fused_unicast_with_atomic_inc` | `fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet` | (a) `FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, src, size, NocUnicastAtomicIncFusedCommandHeader` (b) `RoutingPlaneConnectionManager&, route_id, src, size, NocUnicastAtomicIncFusedCommandHeader` |

**Multicast unicast write (2D mesh, raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_multicast_noc_unicast_write` | `fabric_multicast_noc_unicast_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, MeshMcastRange, src, size, NocUnicastCommandHeader` (b) `RoutingPlaneConnectionManager&, route_id, MeshMcastRange*, src, size, NocUnicastCommandHeader` |

**Multicast scatter write (2D mesh, raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_multicast_noc_scatter_write` | `fabric_multicast_noc_scatter_write_single_packet` | (a) `FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, MeshMcastRange, src, size, NocUnicastScatterCommandHeader` (b) `RoutingPlaneConnectionManager&, route_id, MeshMcastRange*, src, size, NocUnicastScatterCommandHeader` |

**Multicast fused unicast with atomic inc (2D mesh, raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_multicast_noc_fused_unicast_with_atomic_inc` | `fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet` | (a) `FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, MeshMcastRange, src, size, NocUnicastAtomicIncFusedCommandHeader` (b) `RoutingPlaneConnectionManager&, route_id, MeshMcastRange*, src, size, NocUnicastAtomicIncFusedCommandHeader` |

**Multicast fused scatter write + atomic inc (2D mesh, raw-size):**

| Old name | New `_single_packet` name | Overloads |
|----------|--------------------------|-----------|
| `fabric_multicast_noc_fused_scatter_write_atomic_inc` | `fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet` | (a) `FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, MeshMcastRange, src, size, NocUnicastScatterAtomicIncFusedCommandHeader` (b) `RoutingPlaneConnectionManager&, route_id, MeshMcastRange*, src, size, NocUnicastScatterAtomicIncFusedCommandHeader` |

**NOT renamed (excluded):**
- All inline_write variants
- All atomic_inc variants
- All `_with_state` / `_set_state` raw (non-addrgen) variants

---

### mesh/api.h — Addrgen Variants Missing (New Work)

Confirmed by inspection: `multicast_fused_scatter_write_atomic_inc` has NO addrgen overloads in mesh/api.h. The last addrgen function is `fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state` (line 4533).

**Missing addrgen overloads to add:**

1. `fabric_multicast_noc_fused_scatter_write_atomic_inc` — addrgen single-sender overload
   `(FabricSenderType*, pkt_hdr, dst_dev_id, dst_mesh_id, MeshMcastRange, src, AddrGenType, page_id0, page_id1, semaphore_noc_address, val, offset0=0, offset1=0, flush=true)`

2. `fabric_multicast_noc_fused_scatter_write_atomic_inc` — addrgen connection-manager overload
   `(RoutingPlaneConnectionManager&, route_id, MeshMcastRange*, src, AddrGenType, page_id0, page_id1, semaphore_noc_address, val, offset0=0, offset1=0, flush=true)`

3. `fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state` — addrgen single-sender overload
4. `fabric_multicast_noc_fused_scatter_write_atomic_inc_with_state` — addrgen connection-manager overload
5. `fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state` — addrgen single-sender overload
6. `fabric_multicast_noc_fused_scatter_write_atomic_inc_set_state` — addrgen connection-manager overload

These follow the same pattern as `fabric_multicast_noc_fused_unicast_with_atomic_inc` addrgen overloads (mesh/api.h lines 4149-4277). The scatter variant: for pages small enough (`page_size * 2 <= FABRIC_MAX_PACKET_SIZE`), use scatter op directly. For larger pages, fall back to two separate unicast writes per chunk pair, then fused scatter only on final chunk.

---

## Architecture Patterns

### Pattern 1: Raw-Size Unicast Wrapper (NOC address extraction + increment)

`NocUnicastCommandHeader` is `struct { uint64_t noc_address; }`. The wrapper extracts `noc_address` as a `uint64_t`, increments it by `FABRIC_MAX_PACKET_SIZE` each loop, and constructs a fresh header.

```cpp
// Source: linear/api.h lines 2763-2814 (fused addrgen overload, same technique)
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header,
    uint8_t num_hops) {
    uint32_t remaining = size;
    uint32_t current_src_addr = src_addr;
    uint64_t current_noc_addr = noc_unicast_command_header.noc_address;  // extract

    while (remaining > FABRIC_MAX_PACKET_SIZE) {
        noc_async_writes_flushed();
        fabric_unicast_noc_unicast_write_single_packet(
            client_interface,
            packet_header,
            current_src_addr,
            FABRIC_MAX_PACKET_SIZE,
            tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr},  // construct fresh
            num_hops);
        current_src_addr += FABRIC_MAX_PACKET_SIZE;
        current_noc_addr += FABRIC_MAX_PACKET_SIZE;
        remaining -= FABRIC_MAX_PACKET_SIZE;
    }
    // Final/only packet
    noc_async_writes_flushed();
    fabric_unicast_noc_unicast_write_single_packet(
        client_interface,
        packet_header,
        current_src_addr,
        remaining,
        tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr},
        num_hops);
}
```

Key: `noc_async_writes_flushed()` is called before EVERY packet including the first — the addrgen loops in `linear/api.h` (lines 2361-2393) call it before the first send too. This is conservative but safe.

### Pattern 2: Connection-Manager Breadth-First Chunking

Current depth-first (single-packet) pattern inside `for_each_header`:
```cpp
// Depth-first (current): each header sends its full payload before next header
PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
    auto& slot = connection_manager.get(i);
    fabric_set_unicast_route(...);
    fabric_unicast_noc_unicast_write_single_packet(&slot.sender, packet_header, src_addr, size, cmd_hdr, num_hops[i]);
});
```

Breadth-first wrapper (new):
```cpp
// Breadth-first (new): all connections get chunk N before any gets chunk N+1
// Source: linear/api.h lines 2413-2464 (addrgen conn mgr variant — already breadth-first)

// Step 1: set route on all headers once
PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
    fabric_set_unicast_route(connection_manager, packet_header, i);
});

uint32_t remaining = size;
uint32_t current_src = src_addr;
uint64_t current_noc_addr = noc_unicast_command_header.noc_address;

while (remaining > FABRIC_MAX_PACKET_SIZE) {
    noc_async_writes_flushed();  // once per chunk, covers all connections
    // Send chunk to ALL connections
    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = connection_manager.get(i);
        fabric_unicast_noc_unicast_write_single_packet(
            &slot.sender, packet_header, current_src, FABRIC_MAX_PACKET_SIZE,
            tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr}, num_hops[i]);
    });
    current_src += FABRIC_MAX_PACKET_SIZE;
    current_noc_addr += FABRIC_MAX_PACKET_SIZE;
    remaining -= FABRIC_MAX_PACKET_SIZE;
}
noc_async_writes_flushed();
PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
    auto& slot = connection_manager.get(i);
    fabric_unicast_noc_unicast_write_single_packet(
        &slot.sender, packet_header, current_src, remaining,
        tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr}, num_hops[i]);
});
```

Note: The addrgen connection-manager variants in `linear/api.h` (lines 2413-2464) and `mesh/api.h` (lines 2367-2415) are already breadth-first — they call `for_each_header` inside the chunk loop. The raw-size wrappers must match this structure.

### Pattern 3: Fused Unicast + Atomic Inc Large Page

Intermediate chunks as regular writes, final chunk as fused. Already implemented for addrgen overloads in `linear/api.h` lines 2763-2814 and `mesh/api.h` lines 4149-4206.

```cpp
// Set route once before loop
packet_header->to_chip_unicast(num_hops);  // or mcast equivalent

uint32_t remaining = size;
uint64_t current_noc_addr = cmd_header.noc_address;

while (remaining > FABRIC_MAX_PACKET_SIZE) {
    noc_async_writes_flushed();
    fabric_unicast_noc_unicast_write_single_packet(  // regular write, not fused
        client_interface, packet_header, current_src, FABRIC_MAX_PACKET_SIZE,
        tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr}, num_hops);
    current_src += FABRIC_MAX_PACKET_SIZE;
    current_noc_addr += FABRIC_MAX_PACKET_SIZE;
    remaining -= FABRIC_MAX_PACKET_SIZE;
}
noc_async_writes_flushed();
// Final: fused write + atomic_inc (atomic_inc only fires here)
fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet(
    client_interface, packet_header, current_src, remaining,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{current_noc_addr, semaphore_noc_address, val, flush},
    num_hops);
```

### Pattern 4: mesh/api.h addrgen namespace

`mesh/api.h` includes `mesh/addrgen_api.h` which simply re-exports `linear/addrgen_api.h`. The addrgen utilities in mesh code use `tt::tt_fabric::addrgen_detail::get_noc_address` (without the `linear::` prefix) — this is because `linear/addrgen_api.h` defines the functions in `namespace tt::tt_fabric` scope directly under `addrgen_detail`, so they are accessible from mesh code via the same path.

### Pattern 5: SetRoute=false Template Parameter (mesh only)

mesh/api.h raw-size single-sender variants have a `bool SetRoute = true` template parameter. The chunking wrappers set route once before the loop, then call the `_single_packet` variants with `SetRoute=false` to skip redundant route initialization on subsequent chunks. The addrgen overloads demonstrate this pattern (mesh/api.h lines 2294-2349).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NOC address extraction | Custom bitfield parse | `noc_unicast_command_header.noc_address` (direct field) | `NocUnicastCommandHeader` is a simple struct with a single `uint64_t noc_address` field |
| Flush synchronization | Custom barrier | `noc_async_writes_flushed()` | Already used throughout existing chunking loops; semantics are established |
| Chunk size computation | Custom `min(size, MAX)` logic | `while (remaining > FABRIC_MAX_PACKET_SIZE)` loop pattern | Used in all 8+ existing addrgen chunking loops |
| Route setup before loop | Per-chunk route set | `fabric_set_unicast_route` / `fabric_set_mcast_route` called once before chunk loop | Existing addrgen variants all do this; avoids redundant header writes |
| Addrgen page size | Custom size field | `tt::tt_fabric::addrgen_detail::get_page_size(addrgen)` | Already defined for all addrgen types |

---

## Common Pitfalls

### Pitfall 1: Missing flush before first packet

**What goes wrong:** The header may be in use (from a prior send) when the wrapper starts. Skipping `noc_async_writes_flushed()` before the first `_single_packet` call corrupts the in-flight header.

**Why it happens:** Callers may reuse headers across invocations. The addrgen loops call `noc_async_writes_flushed()` before every packet including the first. Raw-size wrappers must do the same.

**How to avoid:** Always call `noc_async_writes_flushed()` before every `_single_packet` call, not just between chunks. Matches the addrgen loop pattern exactly.

### Pitfall 2: Depth-first vs breadth-first in connection-manager variants

**What goes wrong:** If the inner `_single_packet` connection-manager variant is called in the chunk loop (which itself calls `for_each_header` internally), the result is depth-first: connection 0 gets chunk 0 and chunk 1, then connection 1 gets chunk 0 and chunk 1. This violates the decided breadth-first ordering.

**Why it happens:** The inner connection-manager `_single_packet` wraps `for_each_header` — naturally producing depth-first if called per chunk.

**How to avoid:** In the raw-size connection-manager chunking wrapper, do NOT delegate to the inner connection-manager `_single_packet`. Instead, place `for_each_header` inside the chunk loop, calling the single-sender `_single_packet` directly per header. Mirror the addrgen connection-manager overload structure (linear/api.h lines 2413-2464).

### Pitfall 3: Not preserving route type for intermediate chunks in fused ops

**What goes wrong:** `to_chip_unicast(num_hops)` sets the chip-routing header field. If the first intermediate chunk calls the raw-size unicast write (which sets `to_chip_unicast` internally), but the next call is again the fused variant (which also calls `to_chip_unicast`), there is redundant setup but no corruption. However, for the `_with_state` path, intermediate unicast writes may clobber the `noc_send_type` field set for fused mode.

**How to avoid:** Follow the pattern from linear/api.h line 2863: explicitly set `packet_header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE` before the intermediate loop, then reset to `NOC_FUSED_UNICAST_ATOMIC_INC` before the final chunk. For the non-state-ful raw-size wrappers, the `_single_packet` single-sender calls handle this automatically (they call `to_chip_unicast` which sets both chip routing and noc_send_type).

### Pitfall 4: Scatter write large-page path

**What goes wrong:** The raw-size scatter variants (`NocUnicastScatterCommandHeader`) carry two pre-computed NOC addresses. For payloads larger than `FABRIC_MAX_PACKET_SIZE`, incrementing the scatter addresses would require updating both `noc_address[0]` and `noc_address[1]` independently, which is not feasible without knowing the individual chunk sizes.

**How to avoid:** Per the locked decision, raw scatter write (non-addrgen, two pre-computed NOC addresses) stays as-is. The `_single_packet` rename still applies, but the wrapper for scatter write does NOT add auto-packetization — the wrapper simply calls `_single_packet` unconditionally. Callers needing large scatter must use addrgen overloads. Document this in the API comment.

### Pitfall 5: `FABRIC_MAX_PACKET_SIZE` is a function call, not a compile-time constant

**What goes wrong:** `FABRIC_MAX_PACKET_SIZE` is defined as `tt::tt_fabric::get_fabric_max_packet_size()` (a runtime call). Using it as a template non-type parameter or in `static_assert` would fail.

**Why it happens:** The macro looks like a constant but expands to a function call reading from L1 configuration.

**How to avoid:** Use only in runtime `while` loop conditions. Never as a compile-time constant. The existing chunking loops already do this correctly.

---

## Code Examples

### NocUnicastCommandHeader structure
```cpp
// Source: tt_metal/fabric/fabric_edm_packet_header.hpp lines 164-166
struct NocUnicastCommandHeader {
    uint64_t noc_address;
};
```

Address is a raw 64-bit value; increment it as a `uint64_t` directly.

### PacketHeaderPool::for_each_header signature
```cpp
// Source: tt_metal/fabric/hw/inc/packet_header_pool.h lines 85-103
template <typename Func>
FORCE_INLINE static void for_each_header(uint8_t route_id, Func&& func) {
    auto [packet_headers, num_headers] = header_table[route_id];
    for (uint8_t i = 0; i < num_headers; i++) {
        func(packet_headers, i);  // func(volatile PACKET_HEADER_TYPE*, uint8_t index)
        packet_headers++;
    }
}
```

The lambda receives `(volatile PACKET_HEADER_TYPE* packet_header, uint8_t i)`.

### Complete addrgen fused unicast chunking loop (reference implementation)
```cpp
// Source: linear/api.h lines 2780-2813 — addrgen fused unicast with atomic inc
uint32_t remaining = page_size;
uint32_t current_src_addr = src_addr;
uint64_t current_noc_addr = noc_address;  // uint64_t from addrgen

while (remaining > FABRIC_MAX_PACKET_SIZE) {
    noc_async_writes_flushed();
    fabric_unicast_noc_unicast_write<FabricSenderType>(  // regular write for intermediate
        client_interface, packet_header, current_src_addr, FABRIC_MAX_PACKET_SIZE,
        tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr}, num_hops);
    current_src_addr += FABRIC_MAX_PACKET_SIZE;
    current_noc_addr += FABRIC_MAX_PACKET_SIZE;
    remaining -= FABRIC_MAX_PACKET_SIZE;
}
noc_async_writes_flushed();
fabric_unicast_noc_fused_unicast_with_atomic_inc<FabricSenderType>(  // fused only on final
    client_interface, packet_header, current_src_addr, remaining,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{current_noc_addr, semaphore_noc_address, val, flush},
    num_hops);
```

### Complete addrgen connection-manager breadth-first loop (reference implementation)
```cpp
// Source: linear/api.h lines 2421-2463 — addrgen unicast conn mgr
// Set route once for all headers before sending packets
PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
    fabric_set_unicast_route(connection_manager, packet_header, i);
});

uint32_t remaining_size = page_size;
uint32_t current_offset = offset;

while (remaining_size > FABRIC_MAX_PACKET_SIZE) {
    auto noc_address = tt::tt_fabric::linear::addrgen_detail::get_noc_address(addrgen, page_id, current_offset);
    noc_async_writes_flushed();
    // Send to ALL connections at current chunk (breadth-first)
    fabric_unicast_noc_unicast_write(  // calls for_each_header internally — but route already set
        connection_manager, route_id, src_addr, FABRIC_MAX_PACKET_SIZE,
        tt::tt_fabric::NocUnicastCommandHeader{noc_address}, num_hops);
    src_addr += FABRIC_MAX_PACKET_SIZE;
    current_offset += FABRIC_MAX_PACKET_SIZE;
    remaining_size -= FABRIC_MAX_PACKET_SIZE;
}
```

Note: In the addrgen case the inner `fabric_unicast_noc_unicast_write` (conn mgr) is called inside the loop — this works because that function itself loops over all headers. For the raw-size `_single_packet` wrapper, the equivalent would call the conn mgr `_single_packet` variant inside the loop. This IS breadth-first, since calling the conn mgr `_single_packet` sends chunk N to all connections before returning.

---

## Caller Impact Analysis

The following files outside the API headers call raw-size fabric write functions and will need updating when originals are renamed to `_single_packet`. Since the new wrappers keep the original names, these callers require NO changes — they automatically get auto-packetizing behavior. Only callers that explicitly want single-packet semantics would need to opt in to `_single_packet`.

**Confirmed callers (non-addrgen raw-size APIs):**
- `tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_unicast_write_sender.cpp` — uses `fabric_unicast_noc_unicast_write`, `fabric_multicast_noc_unicast_write`, `fabric_unicast_noc_scatter_write`, `fabric_multicast_noc_scatter_write`, `fabric_unicast_noc_fused_unicast_with_atomic_inc`, `fabric_unicast_noc_fused_scatter_write_atomic_inc`
- `tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_common.hpp` — likely uses same
- `ttnn/cpp/ttnn/operations/experimental/ccl/*/kernels/*.cpp` — multiple CCL kernels
- `ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/*.cpp` — broadcast kernels
- `models/demos/deepseek_v3_b1/unified_kernels/broadcast.hpp`

All these callers will transparently get auto-packetization with no source changes required.

**Callers needing explicit `_single_packet` (if they have correctness dependencies on single-packet behavior):** Zero identified. No caller has been found that requires exactly one packet. This is expected — callers care about data delivery, not packet count.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Callers manually loop over chunks | API auto-packetizes transparently | This phase | Callers sending > MAX bytes no longer hang or corrupt |
| `FABRIC_MAX_PACKET_SIZE` was a constant | Now a function `get_fabric_max_packet_size()` reading L1 config | Pre-existing | Cannot use as template param or constexpr |
| Addrgen-only chunking | Raw-size APIs also chunk | This phase | Parity between raw-size and addrgen paths |

---

## Open Questions

1. **Scatter write wrapper semantics**
   - What we know: Raw scatter has two pre-computed NOC addresses and stays as-is (no auto-packetization added)
   - What's unclear: Should the wrapper still be created for scatter (one that just calls `_single_packet` unconditionally, adding no chunking), or is the raw scatter rename-only without a new wrapper?
   - Recommendation: Add a wrapper that calls `_single_packet` directly (no chunking loop) and documents in the API comment that auto-packetization is not supported for raw scatter — users needing large scatter must use addrgen overloads. This preserves the naming convention and makes the limitation explicit.

2. **`noc_async_writes_flushed()` before the first packet of a session**
   - What we know: The addrgen loops call it before every packet including the first. For raw-size callers, the header is newly set up before calling the wrapper.
   - What's unclear: Is the flush before the first packet truly necessary, or only between packets?
   - Recommendation: Keep the flush before every packet (including first) to match addrgen loop behavior. The cost is one extra barrier for single-packet sends (which short-circuit through the remainder path, calling the barrier once).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Google Test (C++ host tests) + TT-Metal kernel test harness |
| Config file | None (no pytest.ini / jest.config — uses CMake/TT-Metal test runner) |
| Existing test pattern | `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/` runner pattern |
| Quick validation | Compile-only check (device headers are compile-time-only on host) |
| Full test | Run `test_basic_fabric_apis` / addrgen_write test suite on hardware |

### Phase Requirements to Test Map

| Behavior | Test Type | Notes |
|----------|-----------|-------|
| Raw-size unicast write with size > MAX packetizes correctly | integration (kernel) | New kernel test, payload = 2*MAX + 512 bytes |
| All chunks arrive at destination, in order | integration | Verify via receiver buffer comparison |
| Multicast chunking sends same data to all multicast targets | integration | Test MeshMcastRange variants |
| Connection-manager variant is breadth-first | integration | Verify by timing/ordering via two-receiver test |
| `atomic_inc` fires only on final chunk | integration | Verify semaphore count = 1 after fused send |
| Single-packet `_single_packet` APIs still work for small payloads | integration | payload < MAX, verify no regression |
| Wrapper for size <= MAX: single packet (no extra flush) | unit (compile + functional) | Covers common case |
| mesh/api.h `multicast_fused_scatter_write_atomic_inc` addrgen variant | integration | New addrgen overload test |

### Wave 0 Gaps

- [ ] `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp` — raw-size write sender kernel for multi-chunk payloads
- [ ] `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp` — multicast variant
- [ ] `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_main.cpp` — host runner using `AddrgenApiVariant`-style enum for raw-size paths
- [ ] `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp` — shared payload size constants

Existing infrastructure in `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/` (unicast_runner.cpp, test_common.hpp, multicast_runner.cpp) provides the send/receive framework pattern. New auto-packetization tests should follow the same structure, replacing addrgen-based kernels with raw-size API calls using oversized payloads.

---

## Sources

### Primary (HIGH confidence)
- Direct source inspection: `tt_metal/fabric/hw/inc/linear/api.h` — full function inventory, existing chunking loop patterns (lines 2347-3516)
- Direct source inspection: `tt_metal/fabric/hw/inc/mesh/api.h` — full function inventory, confirmed missing addrgen overloads (lines 1-4557)
- Direct source inspection: `tt_metal/fabric/fabric_edm_packet_header.hpp` — `NocUnicastCommandHeader` struct (line 164-166)
- Direct source inspection: `tt_metal/fabric/hw/inc/packet_header_pool.h` — `for_each_header` signature (lines 85-103)
- Direct source inspection: `tt_metal/fabric/hw/inc/linear/addrgen_api.h` — `FABRIC_MAX_PACKET_SIZE` macro definition, `addrgen_detail::get_noc_address` (lines 53-83)
- Direct source inspection: `tt_metal/fabric/hw/inc/api_common.h` — update mask enums, `populate_*` helpers
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/` — existing test infrastructure

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions — locked implementation choices confirmed consistent with source code

---

## Metadata

**Confidence breakdown:**
- API inventory (functions to rename): HIGH — enumerated directly from source
- NocUnicastCommandHeader encoding: HIGH — struct definition confirmed in source
- Chunking loop pattern: HIGH — multiple reference implementations exist in source
- Missing mesh/api.h addrgen variants: HIGH — confirmed by exhaustive grep + EOF check
- Caller impact: HIGH — grep of full repo confirmed no callers need `_single_packet`
- Test infrastructure: MEDIUM — existing test patterns identified; new test structure is by analogy

**Research date:** 2026-03-10
**Valid until:** Until any of the enumerated source files change (stable C++ headers — 30+ days)
