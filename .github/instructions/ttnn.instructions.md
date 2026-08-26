---
description: 'PR review rules for TTNN'
applyTo: 'ttnn/**'
excludeAgent: "cloud-agent"
---

# TTNN General Review

## 🔴 CRITICAL

- **Python truthiness on enum values is a silent bug**: nanobind-exposed C++ enums with a zero-valued first member (e.g., `WORKER == 0`, `ROW == 0`) are falsy in Python. Using `if value:` to detect "was this argument provided" silently ignores explicitly-passed zero-valued enum members. Use `if value is not None:` instead.
- **Shard spec consistency between input and output**: when an op derives coordinates (core grids, padding) from its input tensor's ShardSpec but writes to an output with a different ShardSpec, the mismatch silently produces wrong results. Validate that input and output shard specs are compatible or explicitly handle divergence.
- **Hardcoded alignment values**: never hardcode alignment constants (e.g., `32`, `16`). Use HAL accessors (`hal::get_dram_alignment()`, `hal::get_l1_alignment()`) and `round_up`. Hardcoded values break silently when alignment requirements change across architectures.
- **Frozen runtime arguments after descriptor migration**: when migrating from `override_runtime_arguments` to `ProgramDescriptor`, raw computed values (e.g., `buffer->address() + offset`) are not auto-patched on cache hits. Only positions registered as `Buffer*` bindings get re-patched. Any derived address must be registered as a `DynamicRuntimeArg` or recomputed on each dispatch.

## 🟡 IMPORTANT

- **No debug print statements in production code**: remove `printf`, `std::cout`, custom debug print macros (e.g., `concat_db_print`), and leftover debugging artifacts before merge. Use `log_debug(tt::LogOp, ...)` for structured debug output that respects log levels.
- **No magic numbers**: numeric literals without context (tile sizes, loop bounds, buffer counts) must be named constants or derived from configuration. Reviewers should ask: "What is this number and what happens when it changes?"
- **Unify duplicate Python/C++ API paths**: when Python and C++ versions of the same API (e.g., `open_device` vs `CreateDevice`) diverge in defaults, unify via optional parameters handled in C++. Maintaining two paths with subtly different behavior is a long-term maintenance trap.
- **`std::find` over manual loops**: use STL algorithms (`std::find`, `std::any_of`, `std::transform`) instead of hand-written search loops in host-side code. They are more readable and less error-prone.
- **Negative dimension handling**: tensor operations that accept a dimension parameter must handle negative dimensions correctly (wrapping to positive) and include test cases for negative dim inputs.
- **No random files in project root**: test scripts, utility files, and scratch code must go in appropriate directories (`tests/`, `scripts/`). Root-level files pollute the project and confuse tooling.
- **CI test duplication**: before adding a test to a pipeline YAML, check if it's already covered by an existing pipeline entry. Duplicate test runs waste CI resources.
- **Factory method naming**: factory methods should clearly describe what they construct. `from_buffer`, `from_spec`, `create_tensor` are clear. Avoid generic names that don't convey what resource is being created or from what source.
- **Minimize `with_*` mutation patterns on specs**: prefer constructing specs correctly at creation time rather than mutating them post-hoc with `.with_memory_config()`, `.with_shard_spec()`, etc. Mutation chains obscure what the final spec actually is.
- **Use `mesh_device()` accessor on tensors**: when you need the mesh device from a tensor, use the accessor method rather than reconstructing or re-fetching it through other paths.

## 🟢 SUGGESTION

- When exposing new Python bindings, test the boundary between Python falsy/truthy and C++ semantics — especially for enums, zero values, and empty containers.
- Copilot-generated comments in code (e.g., inline explanations of obvious logic) should be removed during review. They add noise, not value.
- When a namespace collision occurs in unity builds, prefer renaming the conflicting internal namespace to be more specific rather than disabling unity builds for that target.
- For operations that support multiple program factories (e.g., different tiling strategies based on shape), ensure dtype additions are propagated to all factory variants, not just the primary one.

## Review Checklist

- [ ] No Python truthiness checks on values that could be zero-valued enums — use `is not None`
- [ ] Alignment values from HAL, not hardcoded
- [ ] No debug prints or leftover debugging artifacts
- [ ] No magic numbers — all literals named or derived
- [ ] Shard specs validated for input/output compatibility
- [ ] Descriptor-migrated ops handle all runtime args correctly on cache hits
- [ ] Duplicate API paths consolidated or justified
- [ ] Negative dimension inputs handled and tested
- [ ] Test files in proper directories, not project root
- [ ] Factory methods clearly named
