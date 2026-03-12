# Phase 3: api-detail-namespace - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Move `_single_packet` function definitions out of public API headers (`mesh/api.h`, `linear/api.h`) into detail headers and namespace. Public auto-packetizing wrappers remain in public headers and call through to `detail::`. Power users can access single-packet APIs via the `detail::` namespace (automatically available since public headers include detail headers). All existing tests must pass identically after restructure.

</domain>

<decisions>
## Implementation Decisions

### Detail Header Path
- Create `mesh/detail/api.h` and `linear/detail/api.h` as parallel structure to public headers
- Follows STL `detail` convention: public header at `mesh/api.h`, internals at `mesh/detail/api.h`
- Same structure for linear: `linear/api.h` (public) + `linear/detail/api.h` (detail)

### Namespace Structure
- Nested `detail::` inside existing `experimental` namespace
- mesh: `tt::tt_fabric::mesh::experimental::detail::`
- linear: `tt::tt_fabric::linear::experimental::detail::`
- Functions keep `_single_packet` suffix in detail namespace (e.g., `detail::fabric_unicast_noc_unicast_write_single_packet`)

### What Moves to Detail
- All `_single_packet` function definitions move to detail headers
  - mesh/api.h: 16 definitions (8 families x 2 overloads: raw sender + connection manager)
  - linear/api.h: 18 definitions (9 families x 2 overloads, including sparse multicast)
- For other helpers (is_addrgen, MeshMcastRange, route helpers): Claude decides per-item — if only used internally, move; if user-facing, keep public

### Include Model
- Public `api.h` includes `detail/api.h` via `#include`
- Power users get `detail::` functions automatically — no separate include needed
- Full doc comments (parameter tables, usage notes) kept in detail headers

### Call Style
- Public auto-packetizing wrappers use fully qualified calls: `detail::fabric_unicast_noc_unicast_write_single_packet(...)`
- No `using` declarations — explicit is better

### AddrGen Headers
- `mesh/addrgen_api.h` and `linear/addrgen_api.h` also updated to call `detail::` qualified
- Same pattern as main API headers — consistent across all callers

### Documentation
- Full doc comments preserved in detail headers (same parameter tables as current public definitions)
- Public wrappers keep their existing docs — no doc changes needed there

### Claude's Discretion
- Per-item decision on which helpers (is_addrgen, route helpers, etc.) move to detail vs stay public
- Internal organization within detail headers (grouping, ordering)
- Whether to split detail headers further by family (unicast vs multicast) or keep as single file

</decisions>

<specifics>
## Specific Ideas

- Power users should be able to call `_single_packet` APIs for performance-critical paths where they know payload fits in one packet — the `detail::` namespace signals "you know what you're doing"
- The auto-packetizing wrappers are the "safe" public API; detail is the "I'll manage my own chunking" API

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mesh/api.h` (5599 lines): 16 `_single_packet` definitions interleaved with auto-packetizing wrappers
- `linear/api.h` (4205 lines): 18 `_single_packet` definitions + sparse multicast variants
- `mesh/addrgen_api.h` (165 lines): addrgen overloads that call `_single_packet` internally
- `linear/addrgen_api.h` (7 lines): re-exports linear/addrgen_api.h

### Established Patterns
- Each function family has 2 overloads: raw `FabricSenderType*` and `RoutingPlaneConnectionManager&`
- `#ifdef FABRIC_2D` guards throughout — detail headers must respect same pattern
- `FORCE_INLINE` on all definitions — must be preserved in detail headers
- `using namespace` in device kernels — moving to detail:: won't break kernel code since kernels call auto-packetizing wrappers, not `_single_packet` directly

### Integration Points
- No external code calls `_single_packet` directly (confirmed by codebase grep)
- Auto-packetizing wrappers in `api.h` are the only callers — internal refactor only
- `api_common.h` and `tt_fabric_api.h` are dependencies of the detail headers (same as current)
- Build system: new headers need to be visible to the device toolchain (RISCV) and host (x86)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-api-detail-namespace*
*Context gathered: 2026-03-12*
