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
- Apply the consolidated `helper_design/mcast_pipe/migration/guardrails.md`:
  shared wire definitions, opaque operation boundaries, exact geometry, owned
  semaphores, setup outside loops, and measured performance claims. The newer
  requested plans supersede its historical public-preparation wording. Its
  all-nightly/shrink/2% gates explicitly concern the referenced operation migration
  queue; no new operation migrations are requested here.
- Host changes require `./build_metal.sh`; tests use the primary Python environment
  and `scripts/run_safe_pytest.sh`, sequentially, with one parametrization first.
- The compact-runtime plan explicitly requests an implementation Claude review;
  invoke the user-specified command only at that required review gate.

## 1. Feedback 1022

- [ ] Verify/minimize semaphore allocation for every construction path.
- [x] Document receiver landing-region ownership when handshakes are disabled.
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
- Feedback 1: all three emitters use `required_semaphores_()`. Added a regression
  with 14 occupied slots, both data-ready modes, descriptor/spec/direct paths,
  and a final operation-owned exchange credit. No allocator change is needed in
  this baseline. Build and execution are pending the feedback validation batch.
- Feedback 2: documented early payload arrival, prohibited pre-receive writes
  (including zero-fill), and completion-wait semantics on both host configuration
  and the receiver pipe API. Documentation-only; checked against send/receive flow.
- Feedback 3 implementation: private cached preparation runs from both attach
  paths and Program binding. Logical topology is cached separately, invalidated
  after additions, and queryable even after preparation failure. Existing const
  attachment APIs and eager rectangular wrappers are preserved. Removed explicit
  preparation from operation callers and Python bindings; revised native/Python
  lifecycle and negative-input tests. Build/execution pending.
- Feedback 4 implementation: removed all three single-kernel overloads; converted
  Conv2D, Conv3D, group-attention and native tests to one-element arrays. Added
  compile-time checks that single-kernel attachment is no longer public.
- Feedback 5 implementation: explicit structural `McastSemaphoreBinding` carries
  ID and scope from numeric IDs, native tokens, or nullptr. Typed pipe/decoder
  templates and return types replace `auto`; chain parameters have descriptive
  names. Host lambda parameters are explicit too. Native device contracts assert
  numeric/null and both non-default token scope types. Build/execution pending.
- Validation: `./build_metal.sh --build-ttnn-tests --enable-ccache` passed
  (`/tmp/mcast-feedback-build.log`). First partial-grid rotating matmul exposed
  SFPI C++17 `-ftt-nttp` rejection of class-valued parameters on out-of-line primary
  template methods. A minimal compiler reproducer isolated this; public value
  aliases now normalize to native token types for private pipe implementations.
  The same matmul passed after this kernel-only correction (1 test, 1.80 s;
  `/tmp/mcast-feedback-matmul-smoke.log`).
- Focused suite initially failed collection because the existing chain stress
  test imported deleted `_cores` from the family test. Updated it to the shared
  `core_set` utility; rerun pending.
- The first combined run also exposed native-child lock contention after Python
  compiler checks: closing the device fixture retains process-wide UMD ownership.
  Stopped the waiting native child before it opened a device. Compiler contracts
  now execute in isolated Python children; the existing native launcher never
  opens a Python device. Run the launcher before device-owning test modules.
- Current sequential rerun has passed all 44 native host contracts, all seven
  intentional negative compiler checks, and both native SpecDeviceSmoke/Matrix
  tests. Remaining Python and transfer/stress cases are still running.
- Completed focused run: 332 pytest cases passed in 214.45 s, including the
  44 native host tests and native device matrices (`/tmp/mcast-feedback-contracts.log`).
- Consumer regressions: Conv3D weight-sharing (3 cases) and wrapped GroupNorm
  (4 cases) passed. Interleaved GroupNorm caught an API compatibility regression:
  its operation-owned signaling needs the positional decoder's numeric semaphore
  constants. The decoder now carries explicit binding types, preserving numeric
  public constants for positional callers and native tokens/nullptr for spec
  callers. Added a native-token type assertion; the failing GroupNorm case now
  passes (1.68 s). Full affected consumer and decoder rerun is in progress.
- The native rerun confirmed the seven negative checks and 44 host contracts
  again, then exposed SFPI rejecting a dependent static nullptr value. Replaced
  the trait's stored value with a constexpr value constructor; native Spec smoke
  now passes (1.97 s). The device matrix and remaining consumers are rerunning.
- Completion remains unproven until every requirement in the supplied plans has
  direct evidence. Checkboxes track work, not a reduction of the requested scope.
