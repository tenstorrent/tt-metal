# Review Expectations: TT-Metalium Public API

> **Codeowners:** `@tenstorrent/metalium-api-owners`
> **Paths:** `tt_metal/api/`

The `tt_metal/api/tt-metalium/` directory is the **stable public surface** of TT-Metalium.
Downstream consumers (models, TTNN, external integrations) depend on it being stable.
This is the highest-scrutiny area in the codebase for breaking changes.

---

## Hard Blockers

- [ ] **Non-experimental API changes must be called out in the PR description.**
  New or modified APIs in `tt_metal/api/tt-metalium/` (non-`experimental/` subdirectory)
  require an explicit callout in the PR body explaining what changed and why it can't go
  under `experimental/` first.

- [ ] **No silent removal or rename of public symbols.**
  Removing or renaming a header, class, function, or enum from the stable API is a breaking
  change. Must be flagged in the PR description and requires explicit API owner approval.

- [ ] **No new includes of internal headers in public headers.**
  Public headers under `tt_metal/api/` must not `#include` from `tt_metal/impl/`,
  `tt_metal/detail/`, or other non-API internal paths.

- [ ] **No default-argument changes to existing public functions.**
  Changing a default argument value is a silent ABI break for callers that relied on the old
  default.

---

## Guidance

- **New APIs should land under `experimental/` first.** Only promote out of `experimental/`
  after sufficient internal usage and explicit review by API owners.

- **`experimental/` has sub-area ownership.** Check `CODEOWNERS` — `experimental/fabric`,
  `experimental/tensor`, `experimental/disaggregation`, etc. each have their own owners.

- **Prefer forward declarations over includes in public headers.** Minimizes compilation
  coupling for downstream consumers.

- **`device.hpp` and `buffer.hpp` are especially sensitive.** Included by almost everything;
  changes here have wide blast radius.

---

## Common Review Feedback

- _"This should go under `experimental/` first."_ — Most common outcome. When in doubt, use
  `experimental/`.
- _"This include pulls in internal headers."_ — Public headers must be clean for external
  consumers.
- _"The PR description doesn't mention this API change."_ — Even small signature tweaks must
  be called out.

---

## Testing Requirements

- [ ] New public APIs must have at least a unit test in `tests/tt_metal/unit_tests/`.
- [ ] API changes affecting dispatch or device setup should be validated against the full
  sanity test suite.
