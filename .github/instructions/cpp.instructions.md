---
description: 'C++20 coding standards and review rules for all C++ files in tt-metal'
applyTo: '**/*.cpp,**/*.h,**/*.hpp,**/*.cc'
excludeAgent: "cloud-agent"
---

# C++ Review Rules

## Review Priorities (C++ specific)

### 🔴 CRITICAL
- **Undefined Behavior**: aliasing violations, lifetime errors, signed overflow, null deref, uninitialized reads, use-after-free, unsequenced modifications
- **Data Races**: shared mutable state modified without explicit synchronization

### 🟡 IMPORTANT
- **Rule-of-5/0 violations**: class manages a resource but is missing copy/move/destructor — or has them when rule-of-0 would suffice
- **Unsafe casts**: `reinterpret_cast`, C-style casts on non-trivial types, `const_cast` removing logical constness
- **Include hygiene**: unnecessary `#include` in a widely-included header; missing forward declaration; `impl/` header included from `api/` header
- **Compile-time impact**: header fanout increase or new template instantiation depth in a widely-included header

### 🟢 SUGGESTION
- **Template complexity**: prefer `if constexpr` / concepts over `enable_if` / SFINAE / recursive instantiation
- **Macro vs. constexpr/template**: flag macros that can be replaced

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
- Virtual destructors required on any base class with virtual functions (or make it non-copyable/non-movable by design).
- **`const` data members and reference members**: flag in classes that need to be movable — they silently delete move assignment.

## Template & Metaprogramming Policy

- No `enable_if`, SFINAE, or recursive template instantiation unless the author demonstrates concepts cannot express the constraint.
- No new macros unless there is a concrete reason `constexpr` or templates cannot work.
- Flag recursive template instantiation depth > 3 levels.

## clang-tidy

Active profiles: `bugprone-*`, `performance-*`, `modernize-*`, `readability-*`, `cppcoreguidelines-*`.

When a finding maps to a clang-tidy rule, cite it:

```
🟡 cppcoreguidelines-pro-type-reinterpret-cast: ...
```

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
TT_FATAL(my_map.count(key), "Key {} not found in map (size={})", key, my_map.size());
auto& entry = my_map.at(key);
```

Apply the same principle to array bounds, vector indexing, and any lookup that can fail — if a crash is possible, make it informative.

## Parameter Passing

- Complex types (vectors, tensors, structs) must be passed by `const&` unless the function needs ownership.
- Prefer `tt::stl::Span<const T>` over `const std::vector<T>&` for input parameters — it accepts both `std::vector` and `std::array` without copying.
- Prefer `std::string_view` over `const std::string&` or `const char*` for read-only string parameters.
- Avoid bare `bool` arguments in public APIs — they're unreadable at the call site. Prefer `enum class` with descriptive enumerators.

## Performance Patterns

- **No dynamic allocation in hot paths**: if the size is known at compile time, use `std::array`. If bounded, use a small-buffer-optimized container. Flag `std::vector` construction inside frequently-called op dispatch functions.
- **Don't `std::move` on return**: it prevents Return Value Optimization. Just `return obj;`.
- **Never `const T&&`** or return `const` values from functions — both block move semantics silently.
- **Move constructors must be `noexcept`**: STL containers will fall back to copying if the move constructor can throw. Flag any move constructor without `noexcept`.
- **`emplace_back` with lvalues**: if you pass a named variable to `emplace_back`, it copies. Either `std::move` it or use `push_back` — don't use `emplace_back` assuming it's inherently better.

## Static Storage & Globals

- **No global objects with non-trivial destructors**: destruction order is undefined across translation units. Use `tt::stl::Indestructible<T>` for function-local statics that require dynamic initialization.
- **No global classes with mutexes or locks** except for specific debug instrumentation.
- **No `using namespace` in headers** — especially never `using namespace std;`.

## Initialization & Control Flow

- **Initialize all primitive members on declaration** (`size_t count = 0;`, not `size_t count;`). Uninitialized primitives in structs are latent UB.
- **Early exit for preconditions**: validate inputs at the top and return/fatal immediately. Avoid wrapping the entire function body in an `if (valid)` block.
- **Exhaustive `switch` on enums**: list every enumerator explicitly. Do not use `default:` — it silently swallows new values added later. Follow the switch with `TT_THROW("Unreachable");` to satisfy compilers that warn about missing returns.

## Implementation Location

- Move function bodies out of headers into `.cpp` files unless they are templates or `constexpr`. Template-heavy implementations go in a `*_inl.hpp` included at the bottom of the header.

## Unsafe Constructs — Flag Unconditionally

- `reinterpret_cast` on non-trivial types
- C-style casts on anything other than arithmetic primitives
- `const_cast` removing logical constness
- `volatile` on non-hardware-register variables
- `goto`
- `.at()` without a preceding bounds check (use `TT_FATAL` instead)
- `using namespace` in any header file
