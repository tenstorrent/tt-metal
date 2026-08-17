---
description: 'PR review rules for TT-Fabric and distributed runtime'
applyTo: 'tt_metal/fabric/**,tt_metal/distributed/**,tt_metal/hw/firmware/src/tt-1xx/**erisc*,tt_metal/hw/inc/internal/ethernet/**'
excludeAgent: "cloud-agent"
---

# TT-Fabric & Distributed Review

## 🔴 CRITICAL

- **Routing plane count consistency**: the number of routing planes must be identical at every hop in the fabric. A mismatch between link-level plane count and the fabric-level routing plane count will cause silent misrouting.
- **Channel buffer/semaphore parity**: in EDM channel setup, `local_buffer_addresses` and `local_semaphore_addresses` vectors must have identical sizes. A mismatch causes out-of-bounds access or silent data corruption.
- **Overlay register vs L1 mode correctness**: when writing to connection semaphores, the NOC write mode (REG vs L1) must match the actual destination memory type. Writing with REG mode to an L1 address triggers a hardware bug. Use `auto` or conditional compilation for adapter code shared between worker (L1) and ERISC (overlay reg) paths.

## 🟡 IMPORTANT

- **Prefer portable fabric APIs**: strongly prefer the public portable APIs from `tt_metal/fabric/hw/inc/linear/api.h` (e.g., `fabric_unicast_noc_unicast_write`) over the legacy private APIs from `tt_fabric_api.h`. The legacy APIs are not portable to 2D fabric topologies and are slated for removal. This is not a hard requirement, but reaching for the legacy APIs should be intentional and reasoned — call out why in the PR when you do.
- **Naming must match scope**: function/variable names must accurately describe their behavior. A function named `write_to_all_chips` that targets a single chip is a bug waiting to happen. API names at the control plane layer should be generic (`reserve_routing_plane`) rather than implementation-specific (`reserve_dispatch_link`).
- **Avoid performance-critical work in hot loops**: heartbeat/telemetry increments in the router main loop add per-iteration cost. Amortize by posting every N iterations (e.g., every 64–128 iterations), not every cycle.
- **Hoist loop-invariant stores**: stores through volatile pointers that don't change between loop iterations (like `std::min` results or config writes) must be moved out of the loop. Repeated volatile stores on every iteration are redundant and expensive.
- **Resolve addresses on host when possible**: if a device-side address override (e.g., moving a connection address to an overlay register) creates implicit coupling, prefer resolving the address on the host and passing it via runtime args. Document any device-side overrides with a comment explaining why.
- **No duplicate logic across builder and kernel args**: compile-time arg indices and constants defined in builder code must not be reduplicated in header files. Single source of truth.
- **Explicit checks before reservations**: control plane APIs that reserve resources must validate against the available set first (`TT_FATAL` with clear message), not rely on `unordered_map::at` throwing an opaque exception.
- **Define `constexpr` mode flags at struct top**: when a struct template is specialized on a type (e.g., `ConnectionSemaphorePtrType`), define a single `constexpr bool` at the top instead of repeating `std::is_same_v<...>` checks throughout the code.
- **Topology/routing consistency**: ring topology does not imply torus. 1D ring is a distinct routing constraint from 2D mesh. Ensure mesh graph descriptors and routing table tests match the actual topology being tested. Prefer `ALL_TO_ALL` layout primitives over manually enumerating connections when the MGD supports it.

## 🟢 SUGGESTION

- Document restrictions that should be lifted later (e.g., "shortcut paths disabled — lift when X is enabled") so the intent isn't lost.
- When flow control is available, enable it for higher packet counts to avoid subtle data loss on congested links.
- Golden bandwidth files should have tight margins (not 10%) for interference-free paths like neighbor exchange.
- Comments explaining protocol expectations (e.g., "skip_src_ch_id_update implies mux mode") should be on the code, not in a separate doc that drifts.
- Avoid spec-like prose in expectation/documentation files that becomes a conflicting second source of truth. Prefer executable assertions or tests over prose claims about behavior.

## Review Checklist

- [ ] Prefers portable fabric APIs (`linear/api.h`); any use of legacy `tt_fabric_api.h` is intentional and justified
- [ ] Routing plane count consistent across all hops
- [ ] Buffer and semaphore address vectors same size in EDM channel setup
- [ ] NOC write mode (REG/L1) matches actual destination memory type
- [ ] Function/variable names accurately describe scope and behavior
- [ ] No per-iteration cost in router hot loop that could be amortized
- [ ] Loop-invariant volatile stores hoisted out of loops
- [ ] Resource reservation APIs validate availability before reserving
- [ ] Topology assumptions documented and tested (ring vs mesh vs torus)
- [ ] No duplicate constants between builder code and kernel headers
