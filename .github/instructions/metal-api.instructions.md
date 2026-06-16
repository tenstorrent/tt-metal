---
description: 'PR review for tt-metalium public API headers — ABI stability, API ergonomics, deprecation policy, and include hygiene'
applyTo: 'tt_metal/api/**'
excludeAgent: "cloud-agent"
---

# Metal Public API Review

The public API surface lives in `tt_metal/api/tt-metalium/`. Everything here is consumed by downstream users (ttnn, tt-train, external customers). Changes require extreme care.

## 🔴 CRITICAL

- **ABI breakage**: any of the following in a public header is a release blocker:
  - Struct or class with changed field layout (added/removed/reordered members)
  - Virtual function added, removed, or reordered in a base class
  - Default argument added or changed on a public function
  - Symbol removed or renamed without a deprecation cycle
  - `sizeof()` of any exported type changed
- **`impl/` leak into `api/`**: a public header must never `#include` anything from `tt_metal/impl/`. Flag any such include — it exposes unstable internals to customers.
- **`experimental/` stability boundary**: headers in `tt_metal/api/tt-metalium/experimental/` have relaxed ABI rules but must NOT be included by stable (non-experimental) headers. If a stable header pulls in an experimental one, that experimental API becomes a de facto stable commitment.

## 🟡 IMPORTANT

- **Deprecation process (mandatory for stable APIs)**: modifying or removing any function in `tt_metal/api/tt-metalium/` (outside `experimental/`) requires a two-step process:
  1. Add the replacement function, update all internal callers, and annotate the old function with `[[deprecated("message")]]`. The message must include an expiration notice and a refactor instruction (e.g., `"Deprecated — will be removed. Replace with newFunction()."`).
  2. Remove the old function in a **separate PR** only after the deprecation has been on `main` for at least 4 weeks.

  Steps 1–3 may be in a single PR, but removal must always be separate. Flag any PR that removes a stable API symbol without a prior deprecation commit on `main`.
- **Parameter order convention**: new functions should follow the established pattern — device/mesh first, then buffers/tensors, then config structs, then optional parameters last.
- **Out-parameters**: prefer return values over output-parameter pointers. If an out-param is unavoidable, it must be the last parameter and clearly documented.
- **`const`-correctness**: public API functions that don't mutate their arguments must take them by `const&` or `const*`. Mutable references in a public API are a red flag unless the mutation is the function's primary purpose.
- **Header weight**: public headers are transitively included by many TUs. Avoid including heavy implementation headers — prefer forward declarations and keep includes minimal. Flag new includes of STL containers or fmt in public headers.
- **Metal 2.0 API (`experimental/metal2_host_api/`)**: this is the next-generation programming model. Changes here should maintain consistency with the existing Metal 2.0 conventions (`ProgramSpec`, `KernelSpec`, `NodeCoord`, etc.) and not introduce patterns that conflict with the stable API.
- **Naming must reflect semantics**: API names are permanent. A function named `CompileProgramSpec` that actually precompiles kernel binaries is misleading. Flag names that describe a different abstraction than what the implementation does. This applies especially to Metal 2.0 factory concepts where naming conventions (`create_program_spec`, `create_run_args`, `create_invariant_run_args`) carry semantic weight.
- **Prefer strong types over primitives**: use `enum class`, strongly-typed identifiers, or `std::chrono::duration` over raw `uint32_t`/`int`/`bool` for parameters that have domain meaning. Bare numeric types in public APIs are ambiguous at the call site.

## Three-Tier API Boundary

The API has three tiers with distinct stability guarantees:

| Directory | Audience | Stability |
|-----------|----------|-----------|
| `tt_metal/api/tt-metalium/` | External customers, ttnn, tt-train | Stable — breaking changes require 4-week deprecation cycle |
| `tt_metal/api/tt-metalium/experimental/` | Power users willing to accept churn | Unstable — no stability guarantee, path to promotion |
| `tt_metal/api/internal/` | Tenstorrent teams only | None — may change silently, never promoted |

Flag any code that:
- Imports from `internal/` outside of `tt_metal/impl/`
- Imports from `experimental/` in a stable header
- Places a new API in the wrong tier for its maturity level

## New Functionality Must Start in Experimental

All new API functionality enters through `experimental/` first. It graduates to the stable API only after it is battle-tested and reviewed by the Runtime team. Flag any PR that adds a brand-new public function or class directly to the stable `tt_metal/api/tt-metalium/` surface without going through `experimental/`.

To add an experimental method affiliated with an existing stable class:
- Create a free function in `tt::tt_metal::experimental::<stable_class_name>` namespace
- Place the header in `experimental/`
- If private member access is needed, `friend` the free function in the stable class (this is the accepted exception to the general "no friends" rule)

## 🟢 SUGGESTION

- Doxygen `@brief`, `@param`, `@return` on every new public function.
- `[[nodiscard]]` on functions whose return value is always meaningful (allocations, status codes).
- Group overloads together in headers for readability.
- Prefer `enum class` over `bool` parameters in public APIs for call-site clarity.
- Avoid `std::optional` when both `nullopt` and an empty value (e.g., empty vector) are semantically identical — it adds confusion without information. Use optional only when absence carries distinct meaning from "empty".

## Review Checklist

- [ ] No struct/class layout changes in public headers without version bump
- [ ] No virtual function signature changes in public base classes
- [ ] No `#include` of `impl/` from `api/` headers
- [ ] Stable API removals have a prior `[[deprecated]]` commit on `main` (≥4 weeks old)
- [ ] Deprecation messages include expiration notice + refactor instruction
- [ ] `experimental/` headers not included from stable headers
- [ ] New public functionality enters via `experimental/` (not directly into stable)
- [ ] New functions follow parameter order convention (device → buffers → config → optional)
- [ ] Public headers remain lightweight (no heavy includes added)
