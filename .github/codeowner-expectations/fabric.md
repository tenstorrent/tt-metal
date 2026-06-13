# Expectations: Fabric

> **Codeowners:** `@ubcheema @aliuTT @aagarwalTT @tt-aho @SeanNijjar @yugaoTT @Riddy21 @cfjchu @p1-0tr`
> **Paths:** `tt_metal/fabric/`, `tt_metal/api/tt-metalium/experimental/fabric`
> **Status:** AI-generated draft — codeowners please review and correct

The fabric layer manages multi-chip interconnect, routing, and message-passing infrastructure. Changes here affect multi-card and cluster correctness and can cause non-obvious failures at scale that only appear on real hardware.

---

## Hard Blockers

- [ ] **Routing table changes must not silently alter existing traffic patterns.**
  If your change modifies how packets are routed between chips, describe the before/after routing behavior explicitly in the PR. Incorrect routing is extremely hard to debug post-merge.

- [ ] **No changes to the fabric API surface without explicit callout.**
  `tt_metal/api/tt-metalium/experimental/fabric` is still experimental but already has downstream users. API changes (especially to message types, addressing, or connection models) must be described in the PR.

- [ ] **EDM (Ethernet Data Mover) changes require hardware validation.**
  Changes to ethernet firmware paths or EDM configuration must be validated on real Wormhole/Blackhole hardware, not just simulation or unit tests.

---

## Guidance

- **New features should be gated behind experimental APIs.** The `experimental/fabric` path exists for a reason — stable routing infrastructure should only absorb changes after they've proven stable.

- **Buffer ring semantics are fragile.** Circular buffer slot sizes, read/write pointer arithmetic, and wrap-around conditions are a common source of deadlocks and data corruption. Be precise in PR descriptions when touching these.

- **Multi-chip tests are slow but necessary.** Don't skip t3000/galaxy-scale tests for changes that affect inter-chip paths.

- **Fabric topology changes affect cabling descriptor files** under `tt_metal/fabric/cabling_descriptors/`. If your change implies a new physical topology, these files need to be updated.

---

## Common Feedback

- _"Does this work on t3k? On galaxy?"_ — Multi-chip validation is not optional for fabric changes.
- _"Ring buffer size mismatch can cause silent corruption."_ — There's a bug-checker rule for this (see `.github/bug_checker/rules/ccl-ring-buffer-mismatch.md`).
- _"The experimental API changed — downstream users need a heads-up."_

---

## Testing Requirements

- [ ] Changes to routing or EDM paths must be tested on real WH hardware (t3000 or galaxy as appropriate).
- [ ] New fabric operations must have unit tests in `tests/tt_metal/`.
- [ ] Scale-out CCL changes should run existing CCL tests: `tests/ttnn/unit_tests/operations/ccl/`.

---

## Notes for External Contributors

Fabric is one of the most hardware-coupled subsystems in tt-metal. If you're adding a new feature, coordinate with the codeowners early — there are non-obvious constraints around link bandwidth, latency, and chip-to-chip addressing that aren't captured in the code.
