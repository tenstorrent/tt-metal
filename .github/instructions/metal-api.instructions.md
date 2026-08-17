---
description: 'PR review for tt-metalium API headers — stability tiers, experimental segregation, and deprecation policy'
applyTo: 'tt_metal/api/**'
excludeAgent: "cloud-agent"
---

# Metal Public API Stability Review

The public API surface lives in `tt_metal/api/tt-metalium/`. Everything here is consumed by downstream users (ttnn, tt-train, external customers). Changes require extreme care. This review also covers placement into `tt-metalium/experimental/` and `api/internal/`.

## 🔴 CRITICAL

### Experimental / FAFO Segregation

All new experimental ("FAFO") work must reside in the `tt::tt_metal::experimental` namespace, with headers in `tt_metal/api/tt-metalium/experimental/`.

- **Experimental methods on stable classes**: do not add experimental methods directly to an existing stable class. Implement them as free functions in the `tt::tt_metal::experimental::<stable_class_name>` namespace, with headers under `experimental/`.
- **Friend access**: `friend` functions that access private members of stable classes are permitted solely for this segregation purpose.
- **Clarity**: the file must include comments explicitly stating that it is experimental and subject to change. Individual functions do not need their own experimental warning.
- **Stability boundary**: `experimental/` headers carry no API-stability guarantee, but must NOT be included by stable (non-experimental) headers. If a stable header pulls in an experimental one, that experimental API becomes a de facto stable commitment.

### Modifying or Deleting Stable APIs

- **Design alignment**: significant changes to existing stable APIs require an associated design document and documented pre-alignment before the PR is submitted.
- **Deprecation process**: for minor changes or deletions of stable APIs (outside `experimental/`), enforce the two-step deprecation process:
  1. Add the replacement, update internal callers, and annotate the old function with `[[deprecated("<message>")]]`.
  2. Remove the old function in a **separate PR** only after the deprecation has been on `main` for at least 4 weeks.
- **Required deprecation message details**: the `[[deprecated]]` message must explicitly include:
  - An expiration notice (e.g., "This is deprecated and will be removed.").
  - Specific instructions for the end user on how to refactor (e.g., "Replace with...").
- Flag any PR that removes a stable API symbol without a prior deprecation commit on `main` that is at least 4 weeks old.

### Graduation to Stable

Promoting experimental functionality to the stable API requires consultation with the Runtime team and a formal design review. Flag any PR that:

- Adds brand-new public functionality directly to the stable `tt_metal/api/tt-metalium/` surface without going through `experimental/`
- Graduates experimental APIs to stable without Runtime team consultation and design review

### Internal API Placement

`tt_metal/api/internal/` (`tt::tt_metal::internal`) is strictly for Tenstorrent-internal functionality that is **not meant to be productized**. It is not part of the public contract — no stability guarantee, and it may never be promoted.

- **Correct path**: new internal APIs must go in `tt_metal/api/internal/`, not `tt_metal/api/tt-metalium/internal/`. Flag additions under `tt-metalium/internal/`.
- **Intent check**: if the feature is on a path to productization into the stable API, it belongs in `tt-metalium/experimental/`, not `api/internal/`. Flag PRs that place productizable / eventually-stable work in `internal/`.
- **Audience**: `internal/` is for Tenstorrent teams only. Flag PRs that treat `internal/` headers or symbols as a public or downstream-facing contract.
- **No promotion path**: unlike `experimental/`, `internal/` has no commitment to stabilization. Do not graduate `internal/` APIs into the stable surface.

## Review Checklist

- [ ] New experimental work lives in `tt::tt_metal::experimental` with headers under `experimental/`
- [ ] Experimental methods on stable classes are free functions in `experimental::<stable_class_name>`, not members of the stable class
- [ ] Experimental files include a comment stating they are experimental and subject to change (per-function warnings not required)
- [ ] `experimental/` headers are not included from stable headers
- [ ] Significant stable API changes have a design doc and documented pre-alignment
- [ ] Stable API removals/changes follow the two-step deprecation process
- [ ] Deprecation messages include expiration notice + refactor instruction
- [ ] Deprecated code has been on `main` ≥4 weeks before removal
- [ ] Graduation from experimental to stable has Runtime team consultation and formal design review
- [ ] New internal APIs go in `api/internal/`, not `api/tt-metalium/internal/`
- [ ] `api/internal/` contains only non-productizable, Tenstorrent-internal functionality (productizable work goes in `experimental/`)
