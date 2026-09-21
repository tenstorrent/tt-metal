# Multicast implementation tracker

## Scope and baselines

Execute `MCAST_FEEDBACK_1022.md`, then `MCAST_COMPACT_RUNTIME_ARGS_PLAN.md`,
then `MCAST_UNIFIED_API_PLAN.md`. Keep each independent change in its own
commit. The three supplied plans and `helper_design/` were already untracked;
preserve them without adding unrelated files to commits.

- Initial checkout: primary `/localdev/sjovic/tt-metal`, branch
  `sjovic/mcast-helpers-review`, HEAD `7e5149ba1e1`.
- Helper-library branch merge base: `32e9f88020ced0683a744170dd3e43c0e849d1d9`.
  Operation-specific pre-migration hashes are recorded in `MCAST_ARGUMENT_REPORT.md`.
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

- [x] Verify/minimize semaphore allocation for every construction path.
- [x] Document receiver landing-region ownership when handshakes are disabled.
- [x] Make preparation private and automatic; allow topology queries during collection.
- [x] Expose only the array/span descriptor attachment interface.
- [x] Replace `auto` function definitions/template parameters with explicit types,
      preserving numeric semaphore IDs and scoped binding tokens.
- [x] Build and run focused contracts and affected regressions.

## 2. Compact runtime arguments

- [x] Capture per-operation pre-migration/current layouts, placements, and baselines.
- [x] Shared constexpr conditional layout, versioned CT metadata, and role inference.
- [x] Host serialization and positional/native-spec decoding, including optional faces.
- [x] Exact sender-range encoding/decoding with pipe-owned storage and explicit fallback.
- [x] Descriptor, ProgramSpec, direct placement emitter, generic APIs, and caller audit.
- [x] Literal goldens, negative compile checks, coordinate and attachment contracts.
- [x] Build; sequential matmul smoke, helper contracts, rotating and consumer regressions.
- [x] Matched construction/cache-hit, kernel, storage, binary, and compile measurements.
- [x] Required Claude implementation review; evaluate findings and rerun affected checks.
- [x] Three-baseline CT/RT report for all specified operations, aggregate traffic,
      signed deltas, residual increases, variant counts, and measurement limitations.

Implementation committed at `5745f4b6d71`: wire v2, shared constexpr
conditional layout, per-placement roles, exact coordinate-range reconstruction,
pipe-owned expansion, positional/native decoders, combined direct emitter, and
descriptor/spec integration. First host build passed; fixed-1D device smoke
passed with the required 4/8 sender and 2/2 receiver helper/total RT counts.
Compiler, native topology, device, Watcher, and consumer validation passed.
Baseline source registry and matched measurements are in `MCAST_ARGUMENT_REPORT.md`;
the additional 22-configuration/96-variant tables and CT snapshots are in
`MCAST_OPERATION_COUNTS.md` and `MCAST_OPERATION_COUNTS.json`.

- Initial compact validation: all 13 compiler contracts passed (56.97 s), including
  single-role optional faces, negative direct/accessor cases, and old-tag rejection.
- All 48 native host contracts, both native device suites, and 225 Python helper
  cases passed. A stale 12-word chain-CT assertion stopped the remaining batch;
  updated it to the 19-word v2 header with signal source at 18. The subsequent
  100 chain stress cases plus four matched matmul cases passed. Chain RT remains
  byte-for-byte 11 words. Logs: `/tmp/mcast-compact-contracts.log`,
  `/tmp/mcast-compact-chain-and-matmul.log`.
- Added 61/64-sender row/column-major device lifetime tests, wrapping through all
  phases on both NoCs. All eight passed (`/tmp/mcast-compact-coordinate-lifetime.log`).
- Consumer regression batch: 171 passed in 77.93 s, including ordinary/pre/post
  LayerNorm, all GroupNorm families, Conv2D/3D, attention, sparse matmul, TopK, and
  partial-grid rotating matmul (`/tmp/mcast-compact-consumers-fixed.log`). Local
  GroupNorm required preserving generic metadata on empty descriptor placements:
  their sources are compiled even though they have no runtime core entries.
- Required Claude review completed (`/tmp/mcast-compact-claude-review.log`). Fixed
  sender coordinate access on mixed placements, cached generic metadata, removed
  pointer-set allocation and obsolete v1 layout functions, and rejected old tags
  before reading a short positional header. Expanded contracts later passed below.
- Added a separate `--watcher` safe-runner mode, with dispatch timeout/reset but
  no triage or lightweight ebreak assertions. Syntax/conflicting-flag checks and
  the mixed-placement coordinate-accessor device smoke passed (4.43 s,
  `/tmp/mcast-compact-watcher-smoke.log`). Full Watcher/measurement matrices and
  final reporting were subsequently completed below.
- Final review contracts passed: 19 launcher cases including 48 native host
  contracts, native smoke/matrix/old-tag checks, and negative compiler cases
  (`/tmp/mcast-compact-review-contracts-fixed.log`). Watcher passed the separate
  12-case lifetime/matmul matrix (`/tmp/mcast-compact-watcher-matrix.log`).
- Matched matmul profiling, host timings, 84 uncached SFPI object compilations,
  ELF frame/code-size inspection, and 64-sender explicit/range artifacts are
  recorded in the report. The range pipe owns 512 bytes of coordinates; its
  frame is larger, despite lower RT traffic. Remaining host and RT increases
  are reported without claiming blanket performance improvement.
- Full argument capture passed 171 consumers plus local DiT and dedicated DRAM
  smoke cases. Counts include native lowering, common addresses, prefix padding,
  idle cores, fixed rectangle capacity, and unchanged operation fields. All
  temporary factory/resource/CMake hooks were removed; `git diff -- ttnn tt_metal`
  was empty before the clean rebuild. No `tt_metal` files were edited.
- Clean host build passed (`/tmp/mcast-compact-clean-build.log`). Final clean
  validation: 176 consumer/audit cases passed in 21.44 s; all six local DiT cases
  passed in 3.70 s; eight compressed-coordinate and four matmul audit cases passed
  in 6.56 s; all three dedicated DRAM padding cases passed. Logs:
  `/tmp/mcast-compact-clean-{validation,dit,measurements,dram}.log`.
- Stage 2 is complete at `ab8a63a05b0`, the baseline for the additive unified API.

## 3. Unified host API

- [x] New configuration, sender variants, receiver ordering, and prepared `Mcast` wrapper.
- [x] Private owned handshake subset with per-sender ACK derivation and chain limits.
- [x] Delegate descriptor/spec/direct interfaces; preserve operation call sites.
- [x] Register host/tests; grouping, ownership, invalid-input, and attachment contracts.
- [x] Operation feasibility fixtures and mixed W/W-1 device execution.
- [x] Build and sequential focused/legacy regressions.

Implementation committed at `bc74b5cf3e1` (baseline `ab8a63a05b0`): new prepared
`Mcast` wrapper, equal-size
row/column grouping, four explicit sender-selection choices, and an owned
handshake subset. The private bridge derives ACK counts per sender without
changing legacy scalar overrides or wire v2. No operation call sites or `tt_metal`
files changed in this stage. No new Python bindings or worktrees were added.

- The initial host build exposed an existing unity-batch name collision in
  unrelated DiT factories. The new source now uses the existing helper-backend
  unity exclusion list. Final rebuild passed (`/tmp/mcast-unified-final-build.log`).
- All 58 native host contracts passed, including ten unified tests for grouping,
  schedule order, full emitted destinations, defaults/partial/empty/disabled
  handshakes, copied construction data, attachment parity, semaphore adoption,
  and invalid/unsupported inputs (`/tmp/mcast-unified-final-host.log`).
- Feasibility fixtures check Matmul (including mixed storage/compute ACKs),
  Conv2D/3D, ordinary LayerNorm readiness/final statistics, GroupNorm row/column and
  wrapped groups, attention's external senders, and TopK. They assert independent
  membership, sender order, and ACK counts; the operation-pattern matrix checks
  both NoCs and signal policies. These are semantic fixtures, not op migrations.
- The first mixed-ACK device smoke passed before the matrix. The final kernel
  uses scope-preserving native semaphore tokens. Matrix execution passed on both
  NoCs, with eight sender turns and three copied/rebound invocations per NoC.
  All four destinations retain the payload while only the two workers ACK;
  senders alternate between one and two expected ACKs. Logs:
  `/tmp/mcast-unified-device-smoke-final.log`, `/tmp/mcast-unified-host-matrix.log`.
- Counter publication and distinct per-round landing slots keep passive
  receivers safe when they lag. The host subset does not automatically filter
  kernel ACK calls. Partial/empty handshake sets on actual chains, rotating
  chains, more than three rectangles, and unequal receiver groups remain outside
  the supported unified API. No distributed DiT fabric coverage is claimed.
- The unified plan explicitly defers Claude review; no additional consultation
  was invoked. The compact implementation review and dispositions remain in
  `MCAST_ARGUMENT_REPORT.md`.
- Final sequential regression passed all 331 pytest cases (251.87 s), including
  the 58 native host contracts, negative compiler checks, native legacy/unified
  device suites, Mcast1D/Mcast2D/family cases, and chain lifetime stress. Log:
  `/tmp/mcast-unified-regressions.log`.
- All 171 affected operation regression cases passed (20.66 s), covering
  Conv2D/3D, GroupNorm, attention/cache reuse, partial-grid Matmul, ordinary/pre/post
  LayerNorm, sparse Matmul, and TopK (`/tmp/mcast-unified-consumers.log`).
- A separate Watcher-only mixed-ACK device matrix passed on both NoCs (2.53 s);
  no triage was combined with Watcher (`/tmp/mcast-unified-watcher.log`). No test
  hung or required a device reset in this stage. Formatting and diff checks pass.
- All three requested stages are complete. The unified API retains existing
  transport limits and does not migrate operations. Remaining compact-argument
  count, stack-storage, and host-time increases are explicitly reported rather
  than claimed as eliminated.

## Review follow-up: receiver API

- Removed the ordered-receiver-vector overload at the user's request, superseding
  that part of the original unified plan. No operation calls this overload; only
  the new synthetic host fixtures used it.
- Audited migrated helper consumers: Matmul/sparse Matmul, Conv2D, LayerNorm,
  attention, and TopK use whole receiver sets or row/column groups. Conv3D uses
  consecutive row-major groups or row strips. GroupNorm uses spatial-axis
  segments or consecutive wrapped groups; the migrated local DiT path uses
  column-major chunks. None requires an arbitrary receiver ordering.
- The previous GroupNorm block-group rationale was incorrect: its sharded
  factory requires `per_core_N % num_datum_row_per_group == 0`, so normalization
  groups cannot span channel shards. Replaced the synthetic block-group fixture
  with row/column spatial reductions and retained wrapped-group coverage.
- Removed the vector constructor, its duplicate-coordinate check, and the
  now-unnecessary private initialization wrapper. Added compile-time checks for
  accepting `CoreRangeSet` and rejecting receiver vectors. Explicit sender
  schedules and the legacy `McastFamily::add_group()` API are unchanged.
- Host rebuild passed (`/tmp/mcast-receiver-api-removal-build.log`). The single
  mixed-ACK device smoke passed first (1.94 s;
  `/tmp/mcast-receiver-api-removal-smoke.log`).
- Sequential validation then passed 53 pytest cases (51.55 s), including all
  58 native host contracts, all legacy/unified native device cases, wrapped and
  interleaved GroupNorm, block-sharded GroupNorm in both orientations with offset
  grids, and the three Conv3D weight-sharing modes. Four offset-grid cases were
  skipped because an 8x4 grid at offset (4,4) does not fit this device. Log:
  `/tmp/mcast-receiver-api-removal-regressions.log`.
- Diff and formatting checks pass; no operation call sites or kernel behavior
  were changed.

## Review follow-up: semaphore encapsulation

- Removed public `data_ready`, `consumer_ready`, and `signal_source` constants
  from both multicast decoder frontends. The decoder binds resource types
  directly into pipe implementations, preserving native scope without exposing
  semaphore IDs or tokens. Removed `McastSemaphoreValue`; positional arguments
  now supply native token types with the legacy local scope.
- Preserved low-level pipe construction syntax and ordinary `sender()` /
  `receiver()` calls. API version is 27; serialized wire format remains v2.
- Both interleaved GroupNorm sender kernels now construct their separate gather
  semaphore only in multi-core gather branches. Removed the unused single-core
  fallback to the multicast readiness semaphore; the Welford distributed gather
  follows the same conditional construction rule.
- Added negative compile contracts for all three removed decoder fields. Existing
  native scope assertions and execution contracts continue to cover ProgramSpec.
- Single-core legacy GroupNorm smoke passed (1.64 s;
  `/tmp/mcast-semaphore-encapsulation-smoke.log`). Sequential regression then
  passed 342 pytest cases (279.02 s), including all 58 native host contracts,
  native ProgramSpec/unified execution, negative compiler contracts, wrapper and
  family cases, chain lifetime stress, and all eight GroupNorm family/cache cases
  (`/tmp/mcast-semaphore-encapsulation-regressions.log`).
- Partial-grid rotating Matmul and all six local DiT GroupNorm cases also passed
  (7 cases, 5.59 s; `/tmp/mcast-semaphore-encapsulation-consumers.log`).
- Watcher-only validation passed the native ProgramSpec matrix and all four
  interleaved GroupNorm cases (5 cases, 34.73 s;
  `/tmp/mcast-semaphore-encapsulation-watcher.log`). No hangs, device resets, or
  Watcher errors occurred. Diff and helper/test-kernel formatting checks pass.
- Changes are kernel-only; no host rebuild is required. Multi-device distributed
  execution has not been tested on this single-device machine.

## Review follow-up: GroupNorm semaphore setup

- Moved gather semaphore construction to the setup section of both GroupNorm
  sender kernels, before their processing loops, as requested. A setup-time
  optional constructs it only for multi-core reductions; the loops reuse it for
  waits/resets. No multicast decoder semaphore access or dummy resource is used.
- Single-core legacy GroupNorm smoke passed (1.65 s;
  `/tmp/groupnorm-semaphore-setup-smoke.log`). All eight GroupNorm family/cache
  cases and six local DiT cases then passed (14 cases, 10.15 s;
  `/tmp/groupnorm-semaphore-setup-regressions.log`).
- Watcher-only validation passed both interleaved GroupNorm implementations in
  local and multi-core configurations (4 cases, 3.72 s;
  `/tmp/groupnorm-semaphore-setup-watcher.log`). Diff checks pass; no host code
  changed and no rebuild was needed. Multi-device execution remains untested.

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
- Feedback validation complete: the final 49-case consumer/native-device batch
  passed in 61.56 s (`/tmp/mcast-feedback-consumers.log`), including both spec
  device cases, Conv3D, all GroupNorm-family cases, attention/cache reuse, fixed
  and rotating matmul, and width/block/height-sharded Conv2D. Together with the
  332-case focused suite and the affected decoder reruns above, this closes
  stage 1. Immediate pre-revamp source baseline: `1674fc3cf08`.
- Completion remains unproven until every requirement in the supplied plans has
  direct evidence. Checkboxes track work, not a reduction of the requested scope.
