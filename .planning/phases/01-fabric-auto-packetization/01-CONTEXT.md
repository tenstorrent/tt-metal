# Phase 1: Fabric API Auto-Packetization - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Enable all fabric write APIs in `tt_metal/fabric/hw/inc/linear/api.h` and `tt_metal/fabric/hw/inc/mesh/api.h` to transparently handle payloads larger than `FABRIC_MAX_PACKET_SIZE` by auto-packetizing under the hood. The chunking is invisible to callers — sending 3×MAX bytes looks the same as sending 1×MAX bytes from the API surface. Existing single-packet APIs are renamed `_single_packet` so power users can still access raw, no-chunking behavior.

Excluded from this phase: inline_write, atomic_inc (inherently size-bounded, no packetization needed), sparse multicast for mesh (linear-only concept for now).

</domain>

<decisions>
## Implementation Decisions

### Raw-size API packetization strategy
- All raw-size write APIs (non-addrgen variants taking explicit `size` + `NocUnicastCommandHeader`) are updated to auto-packetize
- Maintain a local `current_noc_addr` variable initialized from the command header; increment by chunk size each iteration
- Call `noc_async_writes_flushed()` before each header rewrite (i.e., before modifying the packet header for the next chunk — required because we're modifying an in-flight header)
- Applies to: unicast writes, multicast writes (1D linear + 2D mesh `MeshMcastRange`), scatter writes (unicast + multicast)
- Does NOT apply to: inline_write, atomic_inc

### Multi-connection breadth-first ordering
- For connection_manager variants with multiple connections: send chunk N to ALL connections before sending chunk N+1 to any connection (breadth-first / round-robin)
- Applies to both unicast and multicast multi-connection APIs
- Flush semantics: flush once after all connections complete chunk N, before starting chunk N+1 (minimize flush calls)

### Naming convention: _single_packet rename
- All existing raw-size write/scatter/fused APIs that are being updated get renamed to `_single_packet` suffix
  - e.g. `fabric_unicast_noc_unicast_write(sender, header, src, size, cmd_header, num_hops)` → `fabric_unicast_noc_unicast_write_single_packet(...)`
- New auto-packetizing wrappers keep the original names and call `_single_packet` internally
- Only rename APIs that are being updated (not inline_write/atomic_inc which stay as-is)
- Applies to BOTH `linear/api.h` and `mesh/api.h`

### set_state / with_state contract
- `_set_state` APIs continue to initialize the header with `min(page_size, FABRIC_MAX_PACKET_SIZE)` — this is correct because a user may call `_with_state` (renamed `_with_state_single_packet`) in a single-packet context
- `_with_state` addrgen overloads remain unchanged — they have their own chunking loop via `addrgen::get_noc_address` per chunk
- Raw-size `_with_state` APIs: one call handles a full page send internally (may send multiple packets). Advances NOC addr internally across chunks
- `atomic_inc` fires only on the final chunk for fused ops — document transparently in API comments, no API contract change

### Coverage completeness
- `linear/api.h` addrgen-based APIs: **already complete** — all major write ops (unicast, multicast, scatter, fused_unicast_with_atomic_inc) have chunking loops
- `mesh/api.h` gaps to fill:
  - `multicast_fused_scatter_write_atomic_inc` addrgen variants — need packetization: intermediate chunks as unicast writes, final chunk as fused op
  - Verify all `_with_state` and `_set_state` addrgen variants in mesh have proper capping/delegation
- Scatter write large-page fallback: delegation to separate unicast writes (which themselves chunk) is sufficient — no explicit loop in scatter_with_state
- Raw scatter write (non-addrgen, two pre-computed NOC addresses): stays as-is (bounded by MAX_PACKET_SIZE). Large scatter must use addrgen overloads.
- Sparse multicast: linear-only, no mesh equivalent needed

### fused_unicast_with_atomic_inc large-page pattern
- For pages larger than `FABRIC_MAX_PACKET_SIZE`:
  - All intermediate chunks: sent as regular unicast writes
  - Final chunk: sent as fused unicast + atomic_inc
- Same pattern for `multicast_fused_scatter_write_atomic_inc` large-page fallback

### Testing strategy
- New dedicated unit tests covering:
  - Payloads of exactly `N * FABRIC_MAX_PACKET_SIZE + remainder` bytes
  - Verify all data arrives correctly at destination
  - Cover unicast and multicast variants, single and multi-connection
  - Test both addrgen and raw-size paths
- Build command: `./build_metal.sh -e -c --build-tests` (NOT cmake directly)
- Device kernel compile-only validation: use `tt::tt_metal::detail::CompileProgram(device, program)` to compile kernels that `#include` the API headers without running them on hardware. A `CompileOnlyKernels` GTest test serves as the compile probe.
- **Device test serialization constraint**: only one device test can run at a time. For parallel waves (e.g. wave 1 with plans 02-05), the device compile check is a **wave-level gate** — run ONCE after all parallel plans complete, not per-plan. Use `tt-smi -r` to reset devices before running if needed.
- Full hardware tests: run the test binary on Tenstorrent hardware with fabric configured

### Claude's Discretion
- Exact internal variable naming within chunking loops
- Whether to extract NOC address from command header via a helper or inline
- Organization of `_single_packet` API documentation

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FABRIC_MAX_PACKET_SIZE` macro (from `addrgen_api.h`): the chunk boundary constant used throughout
- `noc_async_writes_flushed()`: existing flush call between header rewrites — already used in all addrgen chunking loops
- `PacketHeaderPool::for_each_header(route_id, lambda)`: used in connection_manager variants to iterate all headers in a route
- `addrgen_detail::get_noc_address(addrgen, page_id, offset)`: used in addrgen loops to compute per-chunk NOC address
- `fabric_set_unicast_route` / `fabric_set_mcast_route`: route setup helpers, already called once before chunking loops in addrgen variants

### Established Patterns
- Addrgen chunking loop pattern (already in place for many functions):
  ```cpp
  uint32_t remaining = page_size;
  uint32_t current_offset = 0;
  while (remaining > FABRIC_MAX_PACKET_SIZE) {
      noc_async_writes_flushed();  // before header rewrite
      auto noc_addr = addrgen_detail::get_noc_address(addrgen, page_id, current_offset);
      fabric_unicast_noc_unicast_write_single_packet(..., FABRIC_MAX_PACKET_SIZE, ...);
      current_offset += FABRIC_MAX_PACKET_SIZE;
      remaining -= FABRIC_MAX_PACKET_SIZE;
  }
  // final/remainder send
  ```
- Scatter large-page fallback: `if (page_size * 2 <= FABRIC_MAX_PACKET_SIZE)` → scatter, else → two separate unicast writes

### Integration Points
- All callers of the existing raw-size APIs will need to update to either use the new auto-packetizing wrappers (no change needed) or explicitly call `_single_packet` variants if they want to keep single-packet semantics
- `api_common.h`: shared helpers (e.g. `populate_unicast_write_fields`, `CheckFabricSenderType`) used by both linear and mesh

</code_context>

<specifics>
## Specific Ideas

- "Keep a local variable of the current/next NOC address. Increment it and update the packet header with it."
- "Be sure to write sent barrier between subsequent writes with the same packet header." (i.e. `noc_async_writes_flushed()` before rewriting an in-flight header)
- "If sending to multiple connections/endpoints, be sure to round-robin in breadth-first fashion (send first chunk to every connection before sending second chunk to any connection)"
- "All existing single-packet APIs should be renamed to `_single_packet` so power users who do explicit packetization and/or want extreme control can use the old APIs. The new stuff can wrap it."
- `_set_state` should still set `min(packet_size, max_packet_size)` in case a user calls `_with_state` (single_packet variant) in a non-chunking context

</specifics>

<deferred>
## Deferred Ideas

- Sparse multicast for mesh/api.h — linear-only concept for now, not in scope
- Inline_write and atomic_inc packetization — inherently size-bounded, no packetization needed
- Raw scatter write auto-packetization (non-addrgen, two pre-computed NOC addresses) — callers needing large scatter must use addrgen overloads

</deferred>

---

*Phase: 01-fabric-auto-packetization*
*Context gathered: 2026-03-10*
