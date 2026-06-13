# Expectations: Distributed / Multi-Device

> **Codeowners:** `@cfjchu @aliuTT @tt-asaigal @jbaumanTT`
> **Paths:** `tt_metal/distributed/`
> **Status:** AI-generated draft — codeowners please review and correct

`tt_metal/distributed/` handles multi-device mesh abstractions, device mesh management, and distributed workload coordination. Changes here affect all multi-chip workloads and interact closely with the fabric and dispatch layers.

---

## Hard Blockers

- [ ] **No silent changes to MeshDevice or MeshBuffer ownership semantics.**
  Changes to how devices or buffers are owned, shared, or released across a mesh can cause use-after-free or double-free in complex workloads. These must be explicitly described in the PR.

- [ ] **API changes to `MeshDevice`, `MeshBuffer`, or mesh-level dispatch must be called out.**
  These are used directly by TTNN and model code. Breaking changes require a migration plan described in the PR.

- [ ] **Multi-process tests must not be removed without replacement.**
  Tests under `tests/tt_metal/distributed/multiprocess/` cover race conditions and process isolation that don't appear in single-process tests. Removing them without a clear rationale is not acceptable.

---

## Guidance

- **Distributed changes almost always require multi-card testing.** Single-card CI does not catch bugs in mesh coordination logic. Use t3000 or galaxy test suites when available.

- **Lifecycle invariants across the mesh.** Operations like `close_device`, `synchronize`, and buffer allocation have ordering requirements that are easy to violate. If your change touches device lifecycle, walk through the call graph carefully.

- **Device mesh topology is abstracted, but physical constraints are real.** The logical mesh API hides chip addresses, but underlying routing constraints can still bite. Validate against real hardware when adding new mesh shapes or addressing patterns.

---

## Common Feedback

- _"Did this run on a multi-chip setup?"_ — Single-card validation is not sufficient for distributed/ changes.
- _"This changes MeshDevice semantics — what's the migration path for TTNN?"_
- _"Lifetime of the MeshBuffer isn't clear here."_

---

## Testing Requirements

- [ ] Changes to mesh device/buffer lifecycle must be tested under multi-process scenarios.
- [ ] New mesh operations must have unit tests covering at least 2-chip configurations.
- [ ] Changes to dispatch paths should be validated against galaxy-scale tests.

---

## Notes for External Contributors

`tt_metal/distributed/` is tightly coupled to the fabric and dispatch layers. Before making changes here, read the mesh device architecture notes and coordinate with the codeowners — there are distributed-state invariants that are not obvious from the code alone.
