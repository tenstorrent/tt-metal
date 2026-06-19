---
description: 'C++20 coding standards for host-side C++ in tt-metal (not device kernels)'
applyTo: 'tt_stl/**/*.{cpp,hpp,h,cc},tt_metal/*.{cpp,hpp,h},tt_metal/api/**/*.{cpp,hpp,h},tt_metal/common/**/*.{cpp,hpp,h},tt_metal/detail/**/*.{cpp,hpp,h},tt_metal/distributed/**/*.{cpp,hpp,h},tt_metal/fabric/**/*.{cpp,hpp,h},tt_metal/hostdevcommon/**/*.{hpp,h},tt_metal/impl/**/*.{cpp,hpp,h},tt_metal/jit_build/**/*.{cpp,hpp,h},tt_metal/llrt/**/*.{cpp,hpp,h},tt_metal/logging/**/*.{hpp,h},tt_metal/tools/**/*.{cpp,hpp,h},ttnn/cpp/**/*.{cpp,hpp,h},ttnn/core/**/*.{cpp,hpp,h},ttnn/api/**/*.{hpp,h},tt-train/sources/**/*.{cpp,hpp,h},tools/**/*.{cpp,hpp,h}'
excludeAgent: "cloud-agent"
---

# C++ Host Code Review Rules

> **Scope**: These rules apply to **host-side C++20 code only** — code that runs on
> the CPU and compiles with the standard host toolchain.
> They do **not** apply to device kernel sources (files under `kernels/` directories,
> `tt_metal/kernels/`, `tt_metal/hw/ckernels/`, or `tt_metal/programming_examples/**/kernels/`).
> Kernel code runs on Tenstorrent hardware, cannot use the STL, exceptions, or
> dynamic allocation, and is covered by separate kernel-specific instructions.

## Review Priorities (C++ specific)

### 🔴 CRITICAL
- **Undefined Behavior**: aliasing violations, lifetime errors, signed overflow, null deref, uninitialized reads, use-after-free, unsequenced modifications
- **Data Races**: shared mutable state modified without explicit synchronization

### 🟡 IMPORTANT
- **Include hygiene**: unnecessary `#include` directives; missing forward declaration; `impl/` header included from `api/` header (Abstraction boundary violations).
- **Compile-time impact**: header fanout increase or new template instantiation depth in a widely-included header

### 🟢 SUGGESTION
- **Template complexity**: prefer `if constexpr` / concepts over `enable_if` / SFINAE / recursive instantiation

## Language Standard

C++20. Use modern facilities over legacy patterns:

| Prefer | Over |
|--------|------|
| Concepts (`requires` clauses) | `enable_if` / SFINAE |
| `if constexpr` | Tag dispatch / recursive templates |
| `constexpr` functions | Preprocessor macros |
| `std::optional` | Sentinel values (`-1`, `nullptr`) |
| `std::span` | Raw pointer + size pairs |
| Designated initializers | Positional struct construction |

## Ownership & Lifetime

- RAII for all resources. `std::unique_ptr` / `std::shared_ptr` over raw owning pointers.
- **Rule-of-0 first**: if a class needs rule-of-5, it almost always means a missing RAII wrapper.
- **`const` data members and reference members**: flag in classes that need to be movable — they silently delete move assignment.

## Template & Metaprogramming Policy

- No `enable_if`, SFINAE, or recursive template instantiation unless the author demonstrates concepts cannot express the constraint.
- Flag recursive template instantiation depth > 3 levels.

## Include Hygiene

- Prefer forward declarations over `#include` in headers.
- Never include `impl/` headers from `api/` headers.
- Flag any new `#include` in a header that is transitively included by >50 TUs — suggest PIMPL or forward declaration instead.

## Naming Conventions

Follow the surrounding convention in the file. Do not suggest renaming existing code to a different style unless it's a naming bug (e.g., `is_valid` returning an int, not a bool).

## Error Handling: Prefer TT_FATAL Over Silent Failures

Do not rely on exceptions from STL containers (e.g., `std::out_of_range` from `map.at()`) as error reporting. These produce unreadable crashes with no context. Instead, check the precondition explicitly and fail with `TT_FATAL`, which includes a formatted message describing what went wrong.

```cpp
// Bad: throws std::out_of_range with no useful context
auto& entry = my_map.at(key);

// Good: explicit check, readable failure
TT_FATAL(my_map.contains(key), "Key {} not found in map (size={})", key, my_map.size());
auto& entry = my_map.at(key);
```

Apply the same principle to array bounds, vector indexing, and any lookup that can fail — if a crash is possible, make it informative.

## Parameter Passing

- Complex types (vectors, tensors, structs) must be passed by `const&` unless the function needs ownership.
- Prefer `ttsl::Span<const T>` over `const std::vector<T>&` for input parameters — it accepts both `std::vector` and `std::array` without copying.
- Prefer `std::string_view` over `const std::string&` or `const char*` for read-only string parameters.
- Avoid bare `bool` arguments in public APIs — they're unreadable at the call site. Prefer `enum class` with descriptive enumerators.

## Performance Patterns

- **No dynamic allocation in hot paths**: if the size is known at compile time, use `std::array`. If bounded, use a small-buffer-optimized container. Flag `std::vector` construction inside frequently-called op dispatch functions.
- **Don't `std::move` on return**: it prevents Return Value Optimization. Just `return obj;`.
- **Never `const T&&`**: blocks move semantics silently.

## Static Storage & Globals

- **No global objects with non-trivial destructors**: destruction order is undefined across translation units. Use `ttsl::Indestructible<T>` for function-local statics that require dynamic initialization.
- **No global classes with mutexes or locks** except for specific debug instrumentation.
- **No `using namespace` in headers** — especially never `using namespace std;`.

## Initialization & Control Flow

- **Early exit for preconditions**: validate inputs at the top and return/fatal immediately. Avoid wrapping the entire function body in an `if (valid)` block.
- **Exhaustive `switch` on enums**: list every enumerator explicitly. Do not use `default:` — it silently swallows new values added later. Follow the switch with `TT_THROW("Unreachable");` to satisfy compilers that warn about missing returns.

## Implementation Location

- Move function bodies out of headers into `.cpp` files unless they are templates or `constexpr`. Template-heavy implementations go in a `*_inl.hpp` included at the bottom of the header.

## Unsafe Constructs — Flag Unconditionally

- `volatile` on non-hardware-register variables
- `.at()` without a preceding bounds check (use `TT_FATAL` instead)
- `using namespace` in any header file
