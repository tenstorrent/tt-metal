# Expectations: TT-Metalium Public API

> **Codeowners:** `@tenstorrent/metalium-api-owners`
> **Paths:** `tt_metal/api/`
> **Status:** AI-generated draft — codeowners please review and correct

The `tt_metal/api/tt-metalium/` directory is the **stable public surface** of TT-Metalium. Downstream consumers (models, external integrations, TTNN) depend on it being stable. This is the highest-scrutiny area in the codebase for breaking changes.

---

## Hard Blockers

- [ ] **Non-experimental API changes must be called out explicitly in the PR description.**
  New or modified APIs in `tt_metal/api/tt-metalium/` (non-`experimental/` subdirectory) require an explicit callout in the PR body explaining what changed and why it can't go under `experimental/` first.

- [ ] **No silent removal or rename of public symbols.**
  Removing or renaming a header, class, function, or enum from the non-experimental API is a breaking change. It must be flagged in the PR description and requires explicit approval from API owners.

- [ ] **No new includes of internal headers in public headers.**
  Public headers under `tt_metal/api/` must not `#include` anything from `tt_metal/impl/`, `tt_metal/detail/`, or other non-API internal paths. This leaks implementation details and creates fragile dependencies for downstream consumers.

- [ ] **No default-argument changes to existing public functions.**
  Changing a default argument value is a silent ABI break for callers that relied on the old default.

---

## Guidance

- **New APIs should land under `experimental/` first.** This signals instability and lets the API mature before being promoted to the stable surface. Only promote out of `experimental/` after sufficient internal usage and explicit review.

- **`experimental/` has sub-area ownership.** Changes to `experimental/fabric`, `experimental/tensor`, `experimental/disaggregation`, etc. have their own codeowner groups. Check `CODEOWNERS` before assuming the general API owners cover your change.

- **Forward declarations over includes in public headers.** Prefer forward declarations over full `#include`s in public-facing headers to minimize compilation coupling.

- **`device.hpp` and `buffer.hpp` are especially sensitive.** These are included by almost everything. Changes here have wide blast radius — be conservative.

---

## Common Feedback

- _"This should be `experimental/` first."_ — The most common API review outcome. If in doubt, land under `experimental/`.
- _"This include chain pulls in internal headers."_ — Public headers need to be clean for external consumers.
- _"The PR description doesn't mention this API change."_ — Even small signature tweaks to public APIs must be called out.

---

## Testing Requirements

- [ ] New public APIs must have at least a unit test in `tests/tt_metal/unit_tests/`.
- [ ] API changes that affect dispatch or device setup should be validated against the full sanity test suite.

---

## Notes for External Contributors

`tt_metal/api/tt-metalium/` is the surface that external SDK consumers rely on. When in doubt, add your feature under `experimental/` — it gets the same functionality with a lighter review burden and a clear path to stabilization.
