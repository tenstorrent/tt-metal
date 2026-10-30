---
description: 'PR review rules for TTNN core'
applyTo: 'ttnn/core/**,ttnn/ttnn/**'
excludeAgent: "cloud-agent"
---

# TTNN Core Review

## 🔴 CRITICAL

- **Use `TT_FATAL` not `throw`**: error handling in TTNN core must use `TT_FATAL` with clear messages, not raw C++ exceptions. This ensures consistent error reporting and debuggability.
- **Program cache hash correctness**: any change to op attributes, sub-device config, or runtime state that affects program behavior must be included in the program cache hash. Missing hash inputs cause cache collisions where the wrong cached program is reused silently.
- **Align sharded tensor data cores with compute cores**: operations that work with sharded tensors must ensure the cores used for compute (obtained from `compute_with_storage_grid_size()`) are aligned with the cores over which the tensor data is sharded. Simply selecting a compute grid is not sufficient — for sharded specs the data cores and compute cores must match, otherwise an op reads/writes shards on cores it never computes on. Verify the shard spec's core ranges are covered by the compute grid.
- **Buffer lifetime in descriptors**: objects holding buffer pointers or references (WorkloadDescriptor, ProgramDescriptor) must not outlive the buffers they reference. Verify lifetime semantics when storing buffers in cached/reusable structures.

## 🟡 IMPORTANT

- **Avoid heap allocations in hot paths**: runtime argument patching, program cache lookups, and op dispatch are performance-critical. Flag multiple heap allocations (vector resizes, map insertions) in `override_runtime_arguments` or descriptor patching paths. Prefer pre-allocated storage.
- **`[[nodiscard]]` for RAII guards**: scope guards (sub-device guards, composite trace guards) must be marked `[[nodiscard]]` to prevent accidental creation without binding to a variable, which would immediately destroy the guard.
- **Generic function names**: utility functions and defaults must have descriptive names. `get_default_type()` is unreadable at the call site (`auto type = get_default_type()`). Name functions to be self-documenting when read in context.
- **Prefer structs over pairs/tuples**: for return values or stored data with named fields, use a small struct with named members instead of `std::pair` (avoiding `.first`/`.second` which obscure meaning).
- **Initialize primitive struct members**: when adding new structs (especially in program descriptors or device operations), always initialize primitive members. Uninitialized members in cached/reused structures produce non-deterministic behavior.
- **Naming of variables storing collections**: a variable named `resolved` sounds boolean. If it stores a container of resolutions, name it `resolved_args` or `resolutions`. Names must convey type and intent.
- **Preserve `std::move` on temporaries**: when refactoring code that constructs tensors or specs, do not accidentally drop `std::move` on values that were previously moved. This silently introduces copies in hot paths.
- **Optional references via struct, not raw pointer**: returning a pointer to convey "optional reference output" is error-prone. Prefer returning a struct that always populates the member, or use `std::optional<std::reference_wrapper<T>>`.

## 🟢 SUGGESTION

- When a new TTNN API duplicates an existing one with slightly different defaults (e.g., `open_device` vs `CreateDevice`), unify them via optional parameters rather than maintaining two code paths.
- Environment variable checks are expensive — evaluate once at startup and cache the result, not on every call.
- For experimental APIs whose stability is uncertain, place them behind `_ttnn` (the private C++ bindings namespace), not in the public `ttnn` namespace.
- Use `[[maybe_unused]]` on function parameters that are legitimately unused on some code paths, rather than casting to `(void)`.

## Review Checklist

- [ ] Uses `TT_FATAL` for error handling, not raw `throw`
- [ ] Program cache hash includes all inputs that affect program behavior
- [ ] Sharded tensor data cores aligned with compute cores (`compute_with_storage_grid_size()` covers the shard spec's core ranges)
- [ ] Buffer lifetimes valid — no dangling references in cached descriptors
- [ ] No unnecessary heap allocations in runtime arg patching or dispatch
- [ ] RAII guards marked `[[nodiscard]]`
- [ ] Primitive struct members initialized
- [ ] `std::move` preserved on refactored temporaries
- [ ] Function/variable names self-documenting at the call site
