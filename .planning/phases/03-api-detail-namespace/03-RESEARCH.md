# Phase 3: api-detail-namespace - Research

**Researched:** 2026-03-12
**Domain:** C++ header namespace refactoring — Tenstorrent fabric API headers (device/RISCV toolchain)
**Confidence:** HIGH

---

## Summary

Phase 3 moves all `_single_packet` API functions from the public `mesh::experimental` and
`linear::experimental` namespaces into `detail::` sub-namespaces, each housed in a new
`mesh/detail/api.h` and `linear/detail/api.h` header. The public `mesh/api.h` and
`linear/api.h` files are then updated to `#include` the detail header and call through
`detail::` — eliminating any duplicate definitions.

The goal is to reduce public API surface clutter: power users who need raw single-packet
control can include the detail headers directly, while most users only see the
auto-packetizing wrappers in the public API.

**Key discovery:** This phase is already fully implemented in the repository as of the
commits in the `snijjar/fabric-apis-uplift-automation-experiments` branch. The detail
headers exist, the public headers have been updated, and the tests pass.

**Primary recommendation:** Phase 3 has been shipped. Research reflects the completed
implementation for documentation and retroactive planning purposes.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| API-04 | Move `_single_packet` APIs to `detail` namespace/header so they are accessible to power users but not cluttering the public API surface | Fully implemented: `mesh/detail/api.h` and `linear/detail/api.h` exist with all `_single_packet` definitions in `detail::` namespace; public `mesh/api.h` and `linear/api.h` call through `detail::` with no duplicate definitions |
</phase_requirements>

---

## Standard Stack

### Core Files Modified

| File | Role |
|------|------|
| `tt_metal/fabric/hw/inc/mesh/api.h` | Public 2D mesh API — includes `detail/api.h`, exposes `using detail::is_addrgen`, defines auto-packetizing wrappers only |
| `tt_metal/fabric/hw/inc/mesh/detail/api.h` | New detail header — namespace `tt::tt_fabric::mesh::experimental::detail` — all `*_single_packet` function templates |
| `tt_metal/fabric/hw/inc/linear/api.h` | Public 1D linear API — includes `detail/api.h`, exposes `using detail::is_addrgen`, defines auto-packetizing wrappers only |
| `tt_metal/fabric/hw/inc/linear/detail/api.h` | New detail header — namespace `tt::tt_fabric::linear::experimental::detail` — all `*_single_packet` function templates |

### Supporting / Unchanged Files

| File | Role |
|------|------|
| `tt_metal/fabric/hw/inc/api_common.h` | Shared types (`CheckFabricSenderType<T>`, etc.) |
| `tt_metal/fabric/hw/inc/mesh/addrgen_api.h` | `FABRIC_MAX_PACKET_SIZE` macro, addrgen types for mesh |
| `tt_metal/fabric/hw/inc/linear/addrgen_api.h` | `FABRIC_MAX_PACKET_SIZE` macro, addrgen types for linear |

---

## Architecture Patterns

### Namespace Structure (Implemented)

```
tt::tt_fabric::mesh::experimental         — public: auto-packetizing wrappers only
tt::tt_fabric::mesh::experimental::detail — private: all *_single_packet implementations
tt::tt_fabric::linear::experimental       — public: auto-packetizing wrappers + route helpers
tt::tt_fabric::linear::experimental::detail — private: all *_single_packet implementations
```

### Include Order in Public Headers

Both `mesh/api.h` and `linear/api.h` follow the same pattern:

1. `#pragma once` + standard includes (cstdint, type_traits, fabric headers)
2. Open public namespace block — define prerequisites (`MeshMcastRange` for mesh, route
   helpers for linear)
3. Close public namespace
4. `#include "tt_metal/fabric/hw/inc/{mesh|linear}/detail/api.h"` at file scope
5. Reopen public namespace — `using detail::is_addrgen;` + all auto-packetizing wrappers

This ordering is critical: detail functions need `MeshMcastRange` / route helpers to be
fully defined before the detail header is included.

### PACKET_HEADER_TYPE Workaround

The RISCV compiler has a namespace resolution bug where `PACKET_HEADER_TYPE` (a macro
that expands to `tt::tt_fabric::LowLatencyPacketHeader`) is incorrectly resolved when
inside a deeply nested namespace block. The fix: capture at file scope before any
namespace block opens:

```cpp
// In both detail/api.h files (at file scope, before namespace block):
using detail_packet_header_t = PACKET_HEADER_TYPE;
```

All `_single_packet` functions use `detail_packet_header_t` in their signatures, not
`PACKET_HEADER_TYPE` directly.

### detail/api.h Design Constraints

- No standalone `#include` directives in the detail headers (they rely on the parent
  `api.h` having already included everything).
- No `using namespace` declarations in the detail headers.
- The detail headers are designed to be included only from within `mesh/api.h` or
  `linear/api.h`, not standalone.

### Public Header Calls Into detail::

The auto-packetizing wrappers in the public namespace call `detail::` functions with
`SetRoute=false` (for all chunks after the first) to avoid redundant route setup on
every packet in a multi-packet transfer:

```cpp
// Example from mesh/api.h:
while (remaining > FABRIC_MAX_PACKET_SIZE) {
    noc_async_writes_flushed();
    detail::fabric_unicast_noc_unicast_write_single_packet<FabricSenderType, false>(
        client_interface, packet_header, dst_dev_id, dst_mesh_id, current_src,
        FABRIC_MAX_PACKET_SIZE, tt::tt_fabric::NocUnicastCommandHeader{current_noc_addr});
    // ...
}
```

### Anti-Patterns to Avoid

- **Standalone include of detail/api.h**: detail headers depend on parent includes and
  `using` declarations already being in scope. Including them directly in kernels will
  cause compile errors.
- **Duplicating `_single_packet` definitions**: The public headers must NOT contain
  their own `_single_packet` implementations — they must call `detail::` exclusively.
- **Macro expansion in nested namespace**: Do not use `PACKET_HEADER_TYPE` directly
  inside a nested namespace block on the RISCV compiler. Use the `detail_packet_header_t`
  type alias captured at file scope.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Type trait for addrgen detection | Custom SFINAE | `detail::is_addrgen<T>` (already in both detail headers) | Already implemented, re-exported via `using detail::is_addrgen` in public namespace |
| Packet header type alias | Direct macro use in nested ns | `detail_packet_header_t` type alias | RISCV compiler namespace resolution bug requires this indirection |

---

## Common Pitfalls

### Pitfall 1: RISCV Namespace Resolution of PACKET_HEADER_TYPE
**What goes wrong:** Using `PACKET_HEADER_TYPE` macro inside `tt::tt_fabric::mesh::experimental::detail` causes the RISCV compiler to fail to resolve `tt::tt_fabric::` prefix correctly.
**Why it happens:** The RISCV toolchain has a known limitation with deeply nested namespace blocks and macros that expand to fully qualified type names.
**How to avoid:** Define `using detail_packet_header_t = PACKET_HEADER_TYPE;` at file scope (outside all namespace blocks) before the `namespace tt::tt_fabric::...::detail {` block opens. Use `detail_packet_header_t` throughout the detail header.
**Warning signs:** Compile errors referencing `LowLatencyPacketHeader` as an unknown type, or errors about `tt::tt_fabric::` being undefined inside the detail namespace.

### Pitfall 2: Include Order Between Public Header and detail/api.h
**What goes wrong:** If `detail/api.h` is `#include`d before `MeshMcastRange` is defined (for mesh) or before the route helper functions are defined (for linear), the detail implementations that use those types will fail to compile.
**Why it happens:** `mesh/detail/api.h` uses `MeshMcastRange` in multicast `_single_packet` functions; `linear/detail/api.h` uses `fabric_set_unicast_route`/`fabric_set_mcast_route` helpers.
**How to avoid:** Always include detail headers after the prerequisites are defined in the parent namespace block. Close the public namespace, then `#include` the detail header, then reopen the public namespace.

### Pitfall 3: is_addrgen Duplication
**What goes wrong:** `is_addrgen<T>` type trait is defined in both `mesh/detail/api.h` and `linear/detail/api.h`. If both are included in the same translation unit (which happens when both mesh and linear are included), there is no ODR issue because they are in separate namespaces.
**Why it happens:** By design — each detail namespace has its own `is_addrgen` scoped to that namespace.
**How to avoid:** No action needed. The `using detail::is_addrgen;` re-export in each public namespace correctly scopes each trait.

---

## Code Examples

### detail/api.h Pattern (mesh)

```cpp
// Source: tt_metal/fabric/hw/inc/mesh/detail/api.h

// File-scope type alias (BEFORE namespace block) — avoids RISCV macro resolution bug
using detail_packet_header_t = PACKET_HEADER_TYPE;

namespace tt::tt_fabric::mesh::experimental::detail {

template <typename FabricSenderType, bool SetRoute = true>
FORCE_INLINE void fabric_unicast_noc_unicast_write_single_packet(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile detail_packet_header_t* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    ::tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header) {
    if constexpr (SetRoute) {
        fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    }
    packet_header->to_noc_unicast_write(noc_unicast_command_header, size);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
    client_interface->send_payload_flush_non_blocking_from_address(
        (uint32_t)packet_header, sizeof(detail_packet_header_t));
}

}  // namespace tt::tt_fabric::mesh::experimental::detail
```

### Public Header Include Pattern

```cpp
// Source: tt_metal/fabric/hw/inc/mesh/api.h (simplified)

#pragma once
// ... includes ...

namespace tt::tt_fabric::mesh::experimental {
struct MeshMcastRange { uint8_t e, w, n, s; };
}  // namespace tt::tt_fabric::mesh::experimental

// Include detail after MeshMcastRange is defined:
#include "tt_metal/fabric/hw/inc/mesh/detail/api.h"

namespace tt::tt_fabric::mesh::experimental {
using detail::is_addrgen;   // re-export for SFINAE use

// Auto-packetizing wrapper calls detail:: with SetRoute=false after first chunk
template <typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_unicast_write(...) {
    fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    while (remaining > FABRIC_MAX_PACKET_SIZE) {
        noc_async_writes_flushed();
        detail::fabric_unicast_noc_unicast_write_single_packet<FabricSenderType, false>(...);
    }
    // final chunk
    detail::fabric_unicast_noc_unicast_write_single_packet<FabricSenderType, false>(...);
}
}  // namespace tt::tt_fabric::mesh::experimental
```

---

## API Functions in detail:: Namespace

### mesh::experimental::detail (mesh/detail/api.h)

All functions live in `tt::tt_fabric::mesh::experimental::detail`:

| Function | Variants |
|----------|---------|
| `fabric_unicast_noc_unicast_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_unicast_noc_scatter_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_multicast_noc_unicast_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_multicast_noc_scatter_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `is_addrgen<T>` | Type trait for addrgen detection (SFINAE) |

### linear::experimental::detail (linear/detail/api.h)

All functions live in `tt::tt_fabric::linear::experimental::detail`:

| Function | Variants |
|----------|---------|
| `fabric_unicast_noc_unicast_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_unicast_noc_scatter_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_unicast_noc_fused_scatter_write_atomic_inc_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_unicast_noc_fused_unicast_with_atomic_inc_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_multicast_noc_unicast_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_multicast_noc_scatter_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_multicast_noc_fused_unicast_with_atomic_inc_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_multicast_noc_fused_scatter_write_atomic_inc_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `fabric_sparse_multicast_noc_unicast_write_single_packet` | FabricSenderType template + RoutingPlaneConnectionManager overload |
| `is_addrgen<T>` | Type trait for addrgen detection (SFINAE) |

---

## Test Infrastructure

### Compile-Only Tests

| Test | Fixture | What It Tests |
|------|---------|--------------|
| `CompileOnlyAutoPacketization2D` | `Fabric2DFixture` | `unicast_tx_writer_raw.cpp`, `multicast_tx_writer_raw.cpp`, `compile_probe_unicast_families.cpp`, `compile_probe_multicast_families.cpp` — all with `FABRIC_2D=1` define |
| `CompileOnlyAutoPacketization1D` | `Fabric1DFixture` | `linear_unicast_tx_writer_raw.cpp`, `linear_compile_probe_all_families.cpp` — without `FABRIC_2D` |

These tests call `tt::tt_metal::detail::CompileProgram(device, program)` — they invoke
the device toolchain (RISCV) but do NOT run the kernels on hardware.

### Silicon Tests (18 PASSED, 1 SKIPPED)

The test suite in `test_auto_packetization.cpp` uses a helper function
`run_silicon_family_test` to dispatch to either `run_raw_unicast_write_test` (from
`unicast_runner.cpp`) or `run_raw_multicast_write_test` (from `multicast_runner.cpp`).

Test filtering: `--gtest_filter=*AutoPacketization*`

**Test runner functions (stable signatures):**
```cpp
// In namespace tt::tt_fabric::test:
void run_raw_unicast_write_test(BaseFabricFixture* fixture, const RawTestParams& p);
void run_raw_multicast_write_test(BaseFabricFixture* fixture, const RawTestParams& p);
```

**SparseMulticast** remains `GTEST_SKIP`'d due to firmware issue #36581 (Ethernet core
lockup). This is expected and not a phase 3 concern.

---

## Validation Architecture

> Note: `workflow.nyquist_validation` is not set to false in `.planning/config.json`
> (the key is absent), so this section is included.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | GoogleTest (gtest) — no separate config file, linked via CMake |
| Config file | None — integrated into tt-metal CMake build system |
| Quick run command | `./build/test/tt_metal/fabric/fabric_tests --gtest_filter=*CompileOnly*` |
| Full suite command | `./build/test/tt_metal/fabric/fabric_tests --gtest_filter=*AutoPacketization*` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-04 | `_single_packet` definitions are in `detail::` namespace, not public namespace | compile | `CompileOnlyAutoPacketization2D` and `CompileOnlyAutoPacketization1D` tests | Yes — `compile_probe_*.cpp` and `linear_compile_probe_all_families.cpp` |
| API-04 | Public API calls into `detail::` — auto-packetizing wrappers still work end-to-end | silicon | Full suite: `--gtest_filter=*AutoPacketization*` | Yes — `test_auto_packetization.cpp` |
| API-04 | `mesh/detail/api.h` exists with all 8 `_single_packet` families in `detail::` | structural | Checked by compile probes | Yes — file exists |
| API-04 | `linear/detail/api.h` exists with all 9 `_single_packet` families in `detail::` | structural | Checked by compile probes | Yes — file exists |

### Sampling Rate

- **Per task commit:** `./build/test/tt_metal/fabric/fabric_tests --gtest_filter=*CompileOnly*`
- **Per wave merge:** `./build/test/tt_metal/fabric/fabric_tests --gtest_filter=*AutoPacketization*`
- **Phase gate:** Full suite green (18 PASSED, 1 SKIPPED for SparseMulticast #36581) before marking phase complete

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements. The compile-only
tests exercise all `detail::` function instantiation paths; the silicon tests exercise
the full auto-packetizing wrappers that call through `detail::`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `_single_packet` functions defined directly in public `mesh::experimental` / `linear::experimental` namespace | `_single_packet` functions in `mesh::experimental::detail` / `linear::experimental::detail` | Phase 3 (2026-03-12) | Public API surface is smaller; power users access detail:: directly |
| No `detail/` subdirectory | `tt_metal/fabric/hw/inc/mesh/detail/` and `tt_metal/fabric/hw/inc/linear/detail/` | Phase 3 (2026-03-12) | Conventional C++ pattern for implementation-detail APIs |

---

## Open Questions

1. **Downstream consumers of `_single_packet` APIs**
   - What we know: Within this repo's auto_packetization test suite, no kernels call `_single_packet` functions directly — they all use the auto-packetizing wrappers.
   - What's unclear: Other areas of tt-metal (outside auto_packetization tests) may include `mesh/api.h` or `linear/api.h` and call `_single_packet` functions. Those callers will now need to either use the auto-packetizing wrapper OR explicitly call `detail::fabric_*_single_packet(...)`.
   - Recommendation: Search for `_single_packet` usage across the full codebase before treating phase 3 as globally complete. The phase as scoped to the auto_packetization test suite is complete.

---

## Sources

### Primary (HIGH confidence)

- Direct file read: `tt_metal/fabric/hw/inc/mesh/detail/api.h` — complete implementation (629 lines)
- Direct file read: `tt_metal/fabric/hw/inc/linear/detail/api.h` — complete implementation (652 lines)
- Direct file read: `tt_metal/fabric/hw/inc/mesh/api.h` (first 100 lines) — confirms include structure and `using detail::is_addrgen`
- Direct file read: `tt_metal/fabric/hw/inc/linear/api.h` (first 100 lines) — confirms include structure
- Direct file read: `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/tx_kernel_common.h` — confirms kernels use public API (not `detail::`)
- Direct file read: `tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp` (lines 1-180) — confirms test structure
- Git log: commits `076c0ef94f` through `361d9070b1` confirm phase 3 work (mesh detail extraction, linear detail extraction, RISCV fix)

### Secondary (MEDIUM confidence)

- `.planning/ROADMAP.md` — success criteria and plan decomposition for phase 3
- `.planning/phases/04-test-infrastructure-cleanup/04-CONTEXT.md` — documents "fabric API headers already cleaned in phase 3" confirming completion

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — files read directly from repository
- Architecture: HIGH — include order and namespace structure verified from actual file contents
- Pitfalls: HIGH — RISCV macro bug verified from git commit `361d9070b1` ("fix(03-03): resolve RISCV namespace resolution errors in detail headers") and from `using detail_packet_header_t = PACKET_HEADER_TYPE` pattern visible in both detail headers
- Test infrastructure: HIGH — test files read directly, kernel paths verified against actual filesystem

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (stable — header file structure unlikely to change)
