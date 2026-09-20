# Multicast implementation tracker

## Scope and baselines

Execute `MCAST_FEEDBACK_1022.md`, then `MCAST_COMPACT_RUNTIME_ARGS_PLAN.md`,
then `MCAST_UNIFIED_API_PLAN.md`. Keep each independent change in its own
commit. The three supplied plans and `helper_design/` were already untracked;
preserve them without adding unrelated files to commits.

- Initial checkout: primary `/localdev/sjovic/tt-metal`, branch
  `sjovic/mcast-helpers-review`, HEAD `7e5149ba1e1`.
- Helper-library branch merge base: `32e9f88020ced0683a744170dd3e43c0e849d1d9`.
  Operation-specific pre-migration hashes must still be established for the report.
- No worktrees, `tt_metal` edits, rebases, pushes, or resets.
- Host changes require `./build_metal.sh`; tests use the primary Python environment
  and `scripts/run_safe_pytest.sh`, sequentially, with one parametrization first.
- The compact-runtime plan explicitly requests an implementation Claude review;
  invoke the user-specified command only at that required review gate.

## 1. Feedback 1022

- [ ] Verify/minimize semaphore allocation for every construction path.
- [ ] Document receiver landing-region ownership when handshakes are disabled.
- [ ] Make preparation private and automatic; allow topology queries during collection.
- [ ] Expose only the array/span descriptor attachment interface.
- [ ] Replace `auto` function definitions/template parameters with explicit types,
      preserving numeric semaphore IDs and scoped binding tokens.
- [ ] Build and run focused contracts and affected regressions.

## 2. Compact runtime arguments

- [ ] Capture per-operation pre-migration/current layouts, placements, and baselines.
- [ ] Shared constexpr conditional layout, versioned CT metadata, and role inference.
- [ ] Host serialization and positional/native-spec decoding, including optional faces.
- [ ] Exact sender-range encoding/decoding with pipe-owned storage and explicit fallback.
- [ ] Descriptor, ProgramSpec, direct placement emitter, generic APIs, and caller audit.
- [ ] Literal goldens, negative compile checks, coordinate and attachment contracts.
- [ ] Build; sequential matmul smoke, helper contracts, rotating and consumer regressions.
- [ ] Matched construction/cache-hit, kernel, storage, binary, and compile measurements.
- [ ] Required Claude implementation review; evaluate findings and rerun affected checks.
- [ ] Three-baseline CT/RT report for all specified operations, aggregate traffic,
      signed deltas, residual increases, variant counts, and measurement limitations.

## 3. Unified host API

- [ ] New configuration, sender variants, receiver ordering, and prepared `Mcast` wrapper.
- [ ] Private owned handshake subset with per-sender ACK derivation and chain limits.
- [ ] Delegate descriptor/spec/direct interfaces; preserve operation call sites.
- [ ] Register host/tests; grouping, ownership, invalid-input, and attachment contracts.
- [ ] Operation feasibility fixtures and mixed W/W-1 device execution.
- [ ] Build and sequential focused/legacy regressions.

## Evidence and decisions

- Initial inspection: `required_semaphores_()` already returns 1 for multicast
  without handshakes, 2 with handshakes, and 3 for chain forwarding. Verify all
  emission paths and add a regression for allocation pressure before deciding
  whether allocator changes are needed.
- Completion remains unproven until every requirement in the supplied plans has
  direct evidence. Checkboxes track work, not a reduction of the requested scope.
