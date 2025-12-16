---
applyTo: "tt_metal/**"
---

When reviewing or generating code under `tt_metal/` (low-level runtime):

- Highlight API changes, and/or ABI breakage in tt_metal/api/ directory.
- Favor zero-overhead abstractions
- Avoid architecture-specific `#ifdef` forks; prefer HAL/abstraction points and compile-time traits.
- Keep headers self-contained (IWYU-friendly); prefer forward declarations in headers.
- Recommend splitting header and implementation.
- Watch for perf pitfalls: unnecessary copies, non-`const&` params for large types, accidental `std::string` construction in hot paths.
