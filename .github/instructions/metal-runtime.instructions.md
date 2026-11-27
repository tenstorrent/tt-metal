---
description: 'PR review rules for tt_metal runtime'
applyTo: 'tt_metal/**'
excludeAgent: "cloud-agent"
---

# Metal Runtime Review

## 🔴 CRITICAL

- **Stable API surface discipline**: constants, types, or functions in stable public headers (`tt_metal/api/tt-metalium/`, excluding `experimental/`) must not be removed or change semantics without a deprecation path. Experimental and `tt_metal/api/internal/` interfaces carry no stability guarantee. Follow `scripts/validate_api/README.md` for tier boundaries and same-PR migration of deprecated production usages. If an internal-only constant (e.g., `max_runtime_args`) has no external consumers, move it to `impl/` rather than keeping it public "just in case."
- **No singletons or global state without ContextID**: global instance lookups (e.g., `find_any_existing_instance()`) are fragile with multiple contexts. Pass an explicit `ContextID` or context reference to managers and builders.
- **Kernel binary portability**: kernel binaries are per-device and per-architecture. Precompiled/cached binaries must include arch and device identifiers in their cache keys. Do not assume a binary compiled for one device works on another.
- **Ringbuffer and credit-counter wrap**: verify that ring offsets remain correct when credit counters wrap. Deriving an offset with `counter % ring_units` preserves the ring position across wrap only if the counter modulus is divisible by `ring_units`. Wrap-safe credit accounting alone does not guarantee correct addressing.

## 🟡 IMPORTANT

- **API naming must reflect scope**: function names like `PrecompileProgramSpec` are misleading when the function actually precompiles kernel binaries within the spec. Name must convey what the user gets (e.g., `PreCacheKernelBinaries`). Challenge names that overpromise scope.
- **No default arguments for critical config**: parameters that determine correctness (dispatch core config, telemetry intervals, kernel compile targets) must not have defaults in internal APIs. Force callers to provide valid values explicitly to prevent silent misconfiguration.
- **Internal vs public header placement**: symbols only used within the runtime must live in `impl/` or `internal/` headers, not in `api/tt-metalium/`. Flag any new additions to public headers that have no external consumers. Use the `_impl` suffix for implementation headers (consistent with existing `*_impl.hpp` pattern, e.g., `metal_env_impl.hpp`, `device_impl.hpp`).
- **New environment variables go through `RunTimeOptions`**: Env vars handled by `RunTimeOptions` (`tt_metal/llrt/rtoptions.{hpp,cpp}`) are user-facing inputs that no function signature shows, so new ones get a light review. Existing variables are grandfathered; this applies to variables a PR newly reads.
  - **Request changes** for: a new `getenv` / `std::getenv` / `os.environ` read in runtime code outside `rtoptions` (add an `EnvVarID`, a `HandleEnvVar` case, and a getter instead); a read on every call or inside a function-local static; a boolean that is enabled by mere presence (`=0` must mean off); a name without the `TT_METAL_` prefix; a variable that stands in for something that should be an API or config parameter (per call, per device, per mesh); a second spelling of an option that already exists.
  - **No finding needed** for diagnostic, debug, bring-up, simulator/emulator, or deployment values (paths, device visibility, cluster identity) that keep today's default behavior, especially when they extend an existing `TT_METAL_<FAMILY>_*` family.
  - **Ask, don't block**, when a new variable changes production performance, numerics, feature enablement, or an existing default: ask the author to say in the PR description why an API/config option, derived state, or a compile-time define would not work, and, if the knob is temporary, link a tracking issue for its removal.
  - **Hygiene to check**: a short comment at the `HandleEnvVar` case giving purpose and default; booleans parsed as exactly `0`/`1` and malformed numbers rejected with an error that names the variable and the value (prefer a shared strict parser; see #54446); a user-facing doc entry; a test for the option's effect where practical.
- **Portability of low-level primitives**: intrinsics (`__yield`, `__wfe`) and headers (`arm_acle.h`) are not universally available across compilers and architectures. Verify GCC and Clang support before using arch-specific builtins; add preprocessor guards and fallback implementations.
- **Use `std::chrono::duration` for time parameters**: raw `uint32_t` timeouts obscure units. Prefer `std::chrono::milliseconds` or `std::chrono::microseconds` to make the unit self-documenting.
- **Shared telemetry structs**: when multiple cores or devices write telemetry data that the host reads, define a single shared struct rather than scattering loose fields. This ensures alignment, simplifies versioning, and avoids field ordering bugs.
- **Compute on host, not device, when possible**: if the host has the data to derive a metric (e.g., utilization = delta_busy / delta_total), do the calculation there. Avoid expensive division on device just to simplify host code.
- **Sub-device awareness in utilities**: test fixtures and utility functions that query device properties (`compute_with_storage_grid_size`) must not accidentally include dispatch cores. Use the sub-device-aware APIs.
- **Prefer `constexpr` for new device-side APIs where feasible**: new device API functions and traits that can be evaluated at compile time should be `constexpr` to guarantee zero runtime cost — this applies especially to pure functions over compile-time constants, trait/type queries, address/offset computations, and small accessors with no I/O or side effects. Not every API can be `constexpr` (e.g., those that touch runtime device state); for the ones that can, make them `constexpr` up front rather than deferring the cleanup.
- **Accept typed resource references, not raw addresses**: kernel helper functions should accept references to typed resources (CB, DFB, NoC handle) rather than raw `uint32_t` addresses. This catches misuse at compile time and improves readability.
- **Prefer top-level APIs in tests, but allow impl access when justified**: integration and feature tests should exercise the public API surface rather than reaching into `*_impl` internals. However, reaching into the implementation is legitimate when a unit test needs to verify an internal invariant directly (e.g., that a sync was placed at the right point) — the alternative of forcing a race makes tests unreliable. It is also legitimate for unit tests to exercise concepts we intentionally do not expose via the public API (e.g., launching kernels on DRAM cores), so we are not forced into only large integration tests. When impl access is used to work around a genuinely missing public API (not for the cases above), prefer filing an issue.
- **Watcher validation for mutually exclusive modes**: when a hardware resource has exclusive modes (e.g., iDMA mode on a command buffer), add watcher checks that fire if the resource is misused while in that mode. Silent corruption is worse than a watcher assert.
- **Blackhole NoC MID hygiene**: On Blackhole cards, `noc_async_read`/`noc_async_write` and their `_one_packet` and `_with_state` forms do not program `NOC_TARG_ADDR_MID`/`NOC_RET_ADDR_MID`, which are required to read/write over PCIe and are sticky per command buffer. Route PCIe traffic through `noc_async_{read,write}_pcie` for a single transfer, or `_set_pcie_state`, then `_with_state`, then `_clear_pcie_state` for a loop so the setup and teardown are paid once. MID may only be written once `noc_cmd_buf_ready` reports NOC_CMD_CTRL idle; the clear helpers wait on it themselves, so no barrier is needed before them.
- **Track limitations with issues**: when asserting on a known limitation (e.g., restricting buffer resizing scenarios), open a tracking issue (P2+) for the future relaxation. The assert message should reference the issue number.

## 🟢 SUGGESTION

- When a legacy test kernel is only used by tests, move it from `tt_metal/kernels/` to `tests/`. Production kernel directories should not hold test-only code.
- Consider `always_inline` for very small firmware helper functions (cache invalidation, register reads) that are called in tight loops on device.
- When asserting or early-returning in firmware init paths, add a comment explaining what invariant is being checked and what external state could violate it.
- Use `enchantum::values_generator` with care — ensure all enum values, including sentinel/host-only values like `NONE`, are handled or excluded explicitly.
- Profiler timestamp zones should include disambiguation context (loop iteration index, core ID) so measurements are distinguishable in trace output.
- Discourage using DFB declarations for raw scratch memory — prefer `CoreLocalMem` or structured allocations that convey intent.

## Review Checklist

- [ ] Public API additions have external consumers; internal-only additions are in `impl/`
- [ ] No global state lookups — context/ID passed explicitly
- [ ] No default arguments on correctness-critical internal parameters
- [ ] New env vars are read only in `rtoptions`, once at init, `TT_METAL_`-prefixed, value-based (not presence-based), documented, and not standing in for an API parameter; behavior-changing ones are justified in the PR description
- [ ] Time parameters use `std::chrono::duration`, not raw integers
- [ ] Arch-specific intrinsics guarded with preprocessor and tested on target compilers
- [ ] Function/API names accurately describe what they do (not overly broad)
- [ ] Test-only code lives under `tests/`, not in production directories
- [ ] Telemetry/shared data uses a struct, not loose fields
- [ ] New device APIs that can be evaluated at compile time are `constexpr`
- [ ] Kernel helpers accept typed resource refs, not raw addresses
- [ ] Tests prefer public APIs; impl access is justified (internal-invariant or non-public concept) or backed by a filed issue
- [ ] Exclusive-mode resources have watcher checks for misuse
- [ ] Ring offsets remain correct across both ring wrap and credit-counter wrap
