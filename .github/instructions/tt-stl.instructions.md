---
description: 'PR review for tt_stl — zero-overhead abstractions, regularity, and template complexity'
applyTo: 'tt_stl/**'
excludeAgent: "cloud-agent"
---

# TT-STL Review

`tt_stl/` provides foundational utilities consumed by every layer of the stack. Changes here have outsized impact on compile times, binary size, and correctness across the entire codebase.

## 🔴 CRITICAL

- **Overhead introduction**: any change that adds virtual dispatch, heap allocation, or an indirect function call to a type in `tt_stl/` must provide a benchmark showing it does not regress over the raw-pointer or value-type equivalent. No exception.
- **ODR violation risk**: `tt_stl/` headers are included across the entire codebase. A template specialization or `inline` function with differing definitions across translation units is an ODR violation. Requires careful review.

## 🟡 IMPORTANT

- **Regularity requirements**: new types should satisfy `std::regular` (default constructible, copyable, movable, equality comparable) unless there is a strong semantic reason not to (e.g., a move-only handle type). Document which concept the type satisfies.
- **Recursive template instantiation depth**: flag any new template metaprogramming that exceeds ~3 levels of recursion. Prefer `if constexpr` or fold expressions.
- **`enable_if` / SFINAE**: must be replaced with C++20 concepts. No new `enable_if` in `tt_stl/`.
- **Missing `[[nodiscard]]`**: factory functions and operations returning a new value should be `[[nodiscard]]`.
- **Namespace**: all new symbols must live in the `ttsl::` namespace. The legacy `tt::stl::` namespace is deprecated — do not add new symbols there.
- **Compiler portability**: `tt_stl/` must compile cleanly on both Clang and GCC, for both x86_64 and aarch64. Flag platform-specific intrinsics or compiler builtins without appropriate `#if` guards and tested fallbacks.
- **Use `std::chrono::steady_clock`**: never `high_resolution_clock` (not required to be monotonic) for any timing or spin-wait utility.
- **Use `<tt_stl/assert.hpp>`**: not `<cassert>`. The project's own assert infrastructure provides better diagnostics.
- **Compile-time cost**: new templates or constexpr functions that increase instantiation depth in widely-included headers should be justified. Prefer moving implementations to `.cpp` files or `_inl.hpp` suffixed headers where possible.
- **No test utilities in `tt_stl/`**: this directory is for production foundational code only. Test helpers belong in `tests/tt_metal/test_utils/`.

## 🟢 SUGGESTION

- New utilities: add a `static_assert` verifying the expected concept (`static_assert(std::regular<T>)`).
- Header includes in `tt_stl/`: minimize transitive includes — these headers are pulled into almost every translation unit.
- Prefer `constexpr` over runtime checks for invariants that can be validated at compile time.
- New foundational types should have unit tests — these types are used everywhere and regressions are expensive to debug downstream.

## Review Checklist

- [ ] No virtual dispatch or heap allocation without benchmark
- [ ] No ODR risk from inline/template definitions
- [ ] New types document which concept they satisfy
- [ ] No `enable_if`; concepts used instead
- [ ] New symbols in `ttsl::` namespace (not `tt::stl::`)
- [ ] Transitive include count not increased unnecessarily
- [ ] `[[nodiscard]]` on value-returning factory functions
- [ ] Platform-specific code has compiler/arch guards and fallbacks
- [ ] Uses `steady_clock` (not `high_resolution_clock`) for timing
