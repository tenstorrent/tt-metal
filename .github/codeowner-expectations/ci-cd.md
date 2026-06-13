# Expectations: CI/CD and Build Infrastructure

> **Codeowners:** `@tenstorrent/metalium-developers-infra`
> **Paths:** `.github/`, `cmake/`, `infra/`, `scripts/build_scripts/`
> **Status:** AI-generated draft — codeowners please review and correct

CI/CD and build infra changes affect every contributor and every PR. Mistakes here can break the entire pipeline, hide test failures, or dramatically increase build times. This area requires extra care.

---

## Hard Blockers

- [ ] **No new required checks without infra team sign-off.**
  Adding a new required status check affects merge gates for everyone. This requires explicit infra team review regardless of how small the change looks.

- [ ] **No deletion of existing workflow jobs without confirming coverage.**
  Removing a CI job must be accompanied by evidence that the coverage it provided is maintained elsewhere. Silent removal of test coverage is not acceptable.

- [ ] **`[skip ci]` belongs in the PR title, not commit messages.**
  If CI needs to be skipped, put `[skip ci]` in the PR title. Never add it to commit messages — it can suppress CI on rebased commits unexpectedly.

- [ ] **No hardcoded runner labels or machine counts without infra team discussion.**
  Runner pool sizing and labels affect the shared infrastructure. Changes to `runs-on:` values or runner group assignments must be discussed with infra.

---

## Guidance

- **Workflow changes should be tested on a draft PR or fork first.** It's very easy to break a workflow with a syntax error or wrong ref. Always test on a non-main branch before merging.

- **Avoid expanding what runs on every PR.** New mandatory CI checks should be additive to the per-PR cost only when clearly justified. Prefer nightly or on-demand triggers for expensive tests.

- **CMake changes have compile-time impact.** Changes to `cmake/` or `**/CMakeLists.txt` should be validated for incremental build correctness — a CMake change that causes unnecessary full rebuilds is expensive across the team.

- **Keep secrets out of workflow files.** Don't reference new secrets without first confirming they're provisioned in the GitHub org. Use `${{ secrets.NAME }}` and never hardcode tokens or keys.

- **Pipeline YAML is code.** Name your steps clearly, add comments where non-obvious behavior exists, and keep conditions (`if:`) readable.

---

## Common Feedback

- _"This adds a required check — did you run this by infra?"_
- _"The workflow runs on every commit to main. Is that intentional?"_
- _"This CMakeLists.txt change looks like it'll force a full rebuild of X."_

---

## Testing Requirements

- [ ] New workflows must be tested on a draft PR before merging.
- [ ] CMake changes should be validated with a clean build and an incremental build.
- [ ] Security-sensitive workflow changes (permissions, secret access, PR write access) require explicit security review.

---

## Notes for External Contributors

CI/CD changes can have fleet-wide impact. If you're unsure whether your change is purely additive, open a draft PR and ask in `#github-ci-infra` before merging. The infra team is happy to review early.
