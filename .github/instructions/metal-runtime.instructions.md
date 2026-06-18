---
description: 'PR review rules for tt_metal runtime'
applyTo: 'tt_metal/**'
excludeAgent: "cloud-agent"
---

# Metal Runtime Review

## 🔴 CRITICAL

- **Stable API surface discipline**: constants, types, or functions in public headers (`tt_metal/api/`) must not be removed or change semantics without a deprecation path. If an internal-only constant (e.g., `max_runtime_args`) has no external consumers, move it to `impl/` rather than keeping it public "just in case."
- **No singletons or global state without ContextID**: global instance lookups (e.g., `find_any_existing_instance()`) are fragile with multiple contexts. Pass an explicit `ContextID` or context reference to managers and builders.
- **64-bit device reads require tearing protection**: reading a 64-bit value from device memory splits into two 32-bit reads. Use the read-low/read-high/read-low-again loop pattern to get a consistent snapshot.
- **Kernel binary portability**: kernel binaries are per-device and per-architecture. Precompiled/cached binaries must include arch and device identifiers in their cache keys. Do not assume a binary compiled for one device works on another.

## 🟡 IMPORTANT

- **API naming must reflect scope**: function names like `PrecompileProgramSpec` are misleading when the function actually precompiles kernel binaries within the spec. Name must convey what the user gets (e.g., `PreCacheKernelBinaries`). Challenge names that overpromise scope.
- **No default arguments for critical config**: parameters that determine correctness (dispatch core config, telemetry intervals, kernel compile targets) must not have defaults in internal APIs. Force callers to provide valid values explicitly to prevent silent misconfiguration.
- **Internal vs public header placement**: symbols only used within the runtime must live in `impl/` or `internal/` headers, not in `api/tt-metalium/`. Flag any new additions to public headers that have no external consumers. Use the `details` suffix for implementation headers (consistent with existing `*_details.hpp` pattern).
- **Environment variable access pattern**: never read envvars on every call — cache the result at startup via `rtoptions` or a one-time init. Environment variable parsing is expensive and inconsistent across runs if not cached.
- **Envvar semantics must respect value**: when an envvar is documented as boolean, checking only for its presence (not its value) is a bug. `TT_METAL_SLOW_DISPATCH_MODE=0` must mean "disabled", not "enabled because the envvar exists."
- **Portability of low-level primitives**: intrinsics (`__yield`, `__wfe`) and headers (`arm_acle.h`) are not universally available across compilers and architectures. Verify GCC and Clang support before using arch-specific builtins; add preprocessor guards and fallback implementations.
- **Use `std::chrono::duration` for time parameters**: raw `uint32_t` timeouts obscure units. Prefer `std::chrono::milliseconds` or `std::chrono::microseconds` to make the unit self-documenting.
- **Shared telemetry structs**: when multiple cores or devices write telemetry data that the host reads, define a single shared struct rather than scattering loose fields. This ensures alignment, simplifies versioning, and avoids field ordering bugs.
- **Compute on host, not device, when possible**: if the host has the data to derive a metric (e.g., utilization = delta_busy / delta_total), do the calculation there. Avoid expensive division on device just to simplify host code.
- **Sub-device awareness in utilities**: test fixtures and utility functions that query device properties (`compute_with_storage_grid_size`) must not accidentally include dispatch cores. Use the sub-device-aware APIs.

## 🟢 SUGGESTION

- When a legacy test kernel is only used by tests, move it from `tt_metal/kernels/` to `tests/`. Production kernel directories should not hold test-only code.
- Consider `always_inline` for very small firmware helper functions (cache invalidation, register reads) that are called in tight loops on device.
- When asserting or early-returning in firmware init paths, add a comment explaining what invariant is being checked and what external state could violate it.
- Use `enchantum::values_generator` with care — ensure all enum values, including sentinel/host-only values like `NONE`, are handled or excluded explicitly.

## Review Checklist

- [ ] Public API additions have external consumers; internal-only additions are in `impl/`
- [ ] No global state lookups — context/ID passed explicitly
- [ ] 64-bit device reads use tearing-safe pattern
- [ ] No default arguments on correctness-critical internal parameters
- [ ] Envvar reads cached at init, not per-call; value semantics respected
- [ ] Time parameters use `std::chrono::duration`, not raw integers
- [ ] Arch-specific intrinsics guarded with preprocessor and tested on target compilers
- [ ] Function/API names accurately describe what they do (not overly broad)
- [ ] Test-only code lives under `tests/`, not in production directories
- [ ] Telemetry/shared data uses a struct, not loose fields
