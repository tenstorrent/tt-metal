---
description: 'PR review rules for command dispatch, program compilation, and kernel management'
applyTo: 'tt_metal/impl/dispatch/**,tt_metal/impl/program/**,tt_metal/impl/kernels/**,tt_metal/impl/sub_device/**,tt_metal/impl/event/**,tt_metal/impl/buffers/**'
excludeAgent: "cloud-agent"
---

# Metal Dispatch Review

## 🔴 CRITICAL

- **No new singletons**: do not introduce global/singleton state. New managers and services must live on `MetalContext` (accessed via device/mesh), not as free-standing singletons. `MetalContext::initialize` must always be called with a context ID (not the default) if it's feasible to route one to it. Otherwise, comment on why that's not possible.
- **DeviceCommand size consistency**: `DeviceCommandCalculator` and `DeviceCommand` must agree on sizes. A mismatch between the calculated size and the actual command buffer will corrupt the command queue or cause hangs. ASSERT after all `DeviceCommand` commands are created that the `DeviceCommand` buffer is completely full.
- **NoC barriers before reuse/consume**: Call `noc_async_writes_flushed()` before reusing or overwriting a buffer that was the source of an async write, and `noc_async_read_barrier()` before consuming the result of an async read — unless transaction IDs are tracked to guarantee completion. Missing barriers produce data races that present as intermittent corruption or hangs.
- **Ring-buffer credit ordering**: Always acquire credits/pages *before* writing to a ring buffer, and release them *only after* the async writes that consume the payload have been flushed. For upstream (CmdDatQ) releases, ensure all downstream writes consuming that payload are flushed first. Releasing before flush corrupts the command queue.

## 🟡 IMPORTANT

- **Op-to-op latency is critical**: avoid adding unnecessary work to the dispatch hot path. Resets, state cleanup, and validation that can happen outside the critical path should not be inserted between program dispatches. Prefer per-kernel opt-in over blanket per-dispatch overhead.
- **64-bit device reads require tearing protection**: 64-bit values read from L1 (timestamps, counters) can tear across two 32-bit reads. Without a synchronization mechanism, use a read-low/read-high/re-read-low loop, re-reading the high word as well, until two consecutive reads agree. The exception is when performance is critical and tearing is harmless (e.g. it won't cause lasting problems and only affects some debugging code).
- **Shadow copies for L1 counters**: when incrementing counters stored in L1, keep a shadow copy in local memory. Read-local → increment → write-local → write-L1 is faster than read-L1 → increment → write-L1 (local reads ~2 cycles vs L1 ~8 cycles).
- **Transaction-ID hygiene**: Transaction ID 0 is reserved — never barrier on it, and it must be 0 for a command buffer at both kernel entry and exit. Anything you don't intend to barrier on via a TXN ID should use ID 0. When using non-zero TXN IDs, use the overflow-checking NoC command variants unless the caller provably bounds the transaction count (and add a comment stating the bound).
- **RAII for resource lifetimes**: service cores, memory allocations, and hardware reservations must use RAII wrappers (like `MeshBuffer`) rather than manual claim/release pairs. Forgetting to release is a common source of leaks.
- **No magic numbers for enum values**: do not use integer literals (0, 1) to represent enum states. Cast to the enum class and compare symbolically.
- **Coordinate translation in submesh dispatch**: parent-mesh coordinates and child-mesh coordinates have an offset. Always translate when crossing the boundary. Intersect device ranges with the submesh range to avoid dispatching to devices outside the submesh.
- **Default arguments forbidden for device-critical values**: dispatch configuration parameters (e.g., telemetry modes, CQ counts) must not have default arguments that mask incorrect usage. Require explicit values so misconfigurations fail at compile time.
- **Internal APIs stay internal**: methods that expose hardware details (raw structs, device-side memory layouts) belong on `Impl` classes behind accessor methods, not on public-facing API surfaces. Raw device structs exposed through API headers must be `__attribute__((packed))`.
- **Program state checks before dispatch**: verify `program.impl().is_finalized()`, `program.impl().is_compiled()`, and that logical cores are non-empty before dispatching. Dispatching an uncompiled program is undefined behavior. Consider all possible configurations of dispatch program and state, along these axes:
  - Worker vs idle ethernet dispatch — `distributed_dispatcher` is set on idle ethernet dispatch.
  - Row vs column dispatch.
  - 1 vs 2 CQ.
  - `dispatch_s` enabled vs disabled (only disabled with idle ethernet dispatch and 2 CQ).
  - Realtime profiler enabled or disabled (depends on other configuration).
  - IOMMU enabled vs hugepages used.

## 🟢 SUGGESTION

- Telemetry should be enabled in release builds. Debug instrumentation, by contrast, should be able to compile out completely (no overhead in release builds) — use `constexpr` guards or dedicated compile-time flags for the debug-only paths.
- Prefer NOC writes directly from L1 source addresses over intermediate CB copies when the source is already NOC-accessible.
- Tests for dispatch infrastructure should include higher-level behavioral assertions (e.g., "command counters update after N programs") in addition to unit-level struct checks.
- Changes to dispatch or prefetch commands should include changes to `test_prefetcher` and/or `test_dispatcher` (or at least non-performance changes should).
- Triage scripts depend on some (but not all) variables in the dispatch kernels. If the relevant variables are changed, the triage scripts must also be changed.
- Avoid duplicating the same NOC/timer utility — check `risc_common.h` for existing helpers like `get_timestamp()` before adding new ones.
- Disambiguate profiler zones in loops by including the iteration index, so first-iteration vs last-iteration timing is distinguishable.

## Review Checklist

- [ ] No new singletons — state lives on MetalContext or MeshDevice; `MetalContext::initialize` called with a context ID where feasible
- [ ] DeviceCommand sizes match between calculator and actual buffer; buffer asserted full after all commands are added
- [ ] `noc_async_writes_flushed()` before reusing a write's source buffer; `noc_async_read_barrier()` before consuming a read (unless TXN IDs tracked)
- [ ] Ring-buffer credits acquired before write, released only after consuming writes are flushed
- [ ] Transaction ID 0 reserved; command buffer TXN ID is 0 at kernel entry and exit; overflow-checking variants used for non-zero IDs
- [ ] 64-bit device reads use tearing-safe read loop (re-reading high), unless tearing is provably harmless
- [ ] No unnecessary work added to the dispatch hot path
- [ ] RAII used for resource lifetimes (no manual claim/release without guard)
- [ ] No magic integer literals for enum comparisons
- [ ] Coordinate translation correct when crossing parent/child mesh boundary
- [ ] Program validated (finalized, compiled, non-empty cores) before dispatch
- [ ] Internal APIs behind Impl classes, raw structs marked `packed`
