# Expectations: TT-STL

> **Codeowners:** `@ayerofieiev-tt @akerteszTT @riverwuTT`
> **Paths:** `tt_stl/`
> **Status:** AI-generated draft — codeowners please review and correct

`tt_stl/` is the Tenstorrent Standard Library — shared C++ utilities, type traits, containers, and abstractions used across tt-metal and TTNN. Changes here have wide blast radius because this library is included by nearly every other component.

---

## Hard Blockers

- [ ] **No regressions in compile time.**
  TT-STL is included everywhere. Heavy template metaprogramming additions or deeply nested includes can add seconds to every translation unit. New heavy headers must be justified.

- [ ] **No silent changes to existing type semantics.**
  Changing the behavior of a widely-used utility type (e.g. `tt::stl::Span`, `tt::stl::reflection`, strong typedef helpers) is a breaking change even if the signature doesn't change. Any semantic change must be called out explicitly.

- [ ] **No dependencies on implementation details of other tt-metal subsystems.**
  `tt_stl/` must remain a leaf dependency — it must not `#include` from `tt_metal/`, `ttnn/`, or other higher-level subsystems. Circular dependency additions will be rejected.

---

## Guidance

- **Prefer backward-compatible additions.** Adding a new overload or template specialization is safer than changing existing behavior. When in doubt, add, don't modify.

- **Concepts and type constraints are welcome but must not break existing callsites.** Adding `requires` clauses to existing templates that were previously unconstrained can break callers that happened to work. Validate against a broad build before adding constraints to existing interfaces.

- **Reflection utilities are sensitive.** The reflection/introspection machinery in `tt_stl/` is used at compile time across the codebase. Changes here require extra care about instantiation costs and specialization ordering.

- **Test coverage for new utilities.** Every new utility should have a corresponding test in `tt_stl/tests/`.

---

## Common Feedback

- _"This adds a transitive dependency on tt_metal — tt_stl must be a leaf."_
- _"The template instantiation cost here looks heavy — did you profile compile time?"_
- _"Existing callsites in TTNN break with this constraint addition."_

---

## Testing Requirements

- [ ] Every new public utility must have unit tests in `tt_stl/tests/`.
- [ ] Changes to type utilities or metaprogramming should be validated against a full repo build (compile-time impact check).

---

## Notes for External Contributors

TT-STL is intentionally minimal and focused. Before adding a new utility, check whether a standard C++ equivalent exists (C++17/20) or whether the utility belongs in a higher-level component. We prefer "boring and correct" over "clever and fragile."
