---
description: 'PR review rules for command dispatch, program compilation, and kernel management'
applyTo: 'tt_metal/impl/dispatch/**,tt_metal/impl/program/**,tt_metal/impl/kernels/**,tt_metal/impl/sub_device/**,tt_metal/impl/event/**'
excludeAgent: "cloud-agent"
---

# Metal Dispatch Review

## 🔴 CRITICAL

- **No new singletons**: do not introduce global/singleton state. New managers and services must live on `MetalContext` (accessed via device/mesh), not as free-standing singletons.
- **DeviceCommand size consistency**: `DeviceCommandCalculator` and `DeviceCommand` must agree on sizes. A mismatch between the calculated size and the actual command buffer will corrupt the command queue or cause hangs.
- **64-bit device reads require tearing protection**: 64-bit values read from L1 (timestamps, counters) can tear across two 32-bit reads. Use a read-low/read-high/re-read-low loop to ensure consistency.
- **Kernel binary per-device correctness**: when a `MeshWorkload` is loaded on multiple submeshes, kernel binary buffers must be stored per-MeshDevice (not as a single shared buffer). The `program_binary_status_` check must be per-device, not global.

## 🟡 IMPORTANT

- **Op-to-op latency is critical**: avoid adding unnecessary work to the dispatch hot path. Resets, state cleanup, and validation that can happen outside the critical path should not be inserted between program dispatches. Prefer per-kernel opt-in over blanket per-dispatch overhead.
- **Shadow copies for L1 counters**: when incrementing counters stored in L1, keep a shadow copy in local memory. Read-local → increment → write-local → write-L1 is faster than read-L1 → increment → write-L1 (local reads ~2 cycles vs L1 ~8 cycles).
- **RAII for resource lifetimes**: service cores, memory allocations, and hardware reservations must use RAII wrappers (like `MeshBuffer`) rather than manual claim/release pairs. Forgetting to release is a common source of leaks.
- **No magic numbers for enum values**: do not use integer literals (0, 1) to represent enum states. Cast to the enum class and compare symbolically.
- **Coordinate translation in submesh dispatch**: parent-mesh coordinates and child-mesh coordinates have an offset. Always translate when crossing the boundary. Intersect device ranges with the submesh range to avoid dispatching to devices outside the submesh.
- **Default arguments forbidden for device-critical values**: dispatch configuration parameters (e.g., telemetry modes, CQ counts) must not have default arguments that mask incorrect usage. Require explicit values so misconfigurations fail at compile time.
- **Internal APIs stay internal**: methods that expose hardware details (raw structs, device-side memory layouts) belong on `Impl` classes behind accessor methods, not on public-facing API surfaces. Raw device structs exposed through API headers must be `__attribute__((packed))`.
- **Program state checks before dispatch**: verify `program.impl().is_finalized()`, `program.impl().is_compiled()`, and that logical cores are non-empty before dispatching. Dispatching an uncompiled program is undefined behavior.

## 🟢 SUGGESTION

- When adding telemetry or debug instrumentation, ensure it can be compiled out completely (no overhead in release builds). Use `constexpr` guards or dedicated compile-time flags.
- Prefer NOC writes directly from L1 source addresses over intermediate CB copies when the source is already NOC-accessible.
- Tests for dispatch infrastructure should include higher-level behavioral assertions (e.g., "command counters update after N programs") in addition to unit-level struct checks.
- Avoid duplicating the same NOC/timer utility — check `risc_common.h` for existing helpers like `get_timestamp()` before adding new ones.
- Disambiguate profiler zones in loops by including the iteration index, so first-iteration vs last-iteration timing is distinguishable.

## Review Checklist

- [ ] No new singletons — state lives on MetalContext or MeshDevice
- [ ] DeviceCommand sizes match between calculator and actual buffer
- [ ] 64-bit device reads use tearing-safe read loop
- [ ] Kernel binaries stored per-device when targeting multiple submeshes
- [ ] No unnecessary work added to the dispatch hot path
- [ ] RAII used for resource lifetimes (no manual claim/release without guard)
- [ ] No magic integer literals for enum comparisons
- [ ] Coordinate translation correct when crossing parent/child mesh boundary
- [ ] Program validated (finalized, compiled, non-empty cores) before dispatch
- [ ] Internal APIs behind Impl classes, raw structs marked `packed`
