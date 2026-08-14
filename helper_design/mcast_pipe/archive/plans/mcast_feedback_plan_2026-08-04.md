# Archived: Prioritized `mcast_pipe` Feedback and Validation Plan

Date: 2026-08-04

## Summary

First restore the documented v9 implementation, then perform a tree-wide ABI-boundary cleanup across every migrated
kernel and host binding. No caller may infer the size of an `McastArgs` CT/RT block. Only after that invariant is
established should we change the wire, improve Matmul ownership, or finish Sort's handshake migration.

Performance passes when the median is no more than 1.5% slower than the recorded pre-migration baseline on the same
Blackhole p100a at AICLK 800, using 3 warmups and 20 measured records.

## Ordered Gates

### 1. Restore and verify API-005/API-006

- Recover the exact six-file implementation preserved in `refs/stash` (`91e9d02a8cf`) without dropping the stash until
  committed and verified.
- Preserve the unrelated dirty submodule and untracked files.
- Gate:
  - Complete `test_mcast_pipe.py --dev`: 73/73 pass.
  - Exact height-sharded VAE and width-sharded SegFormer Conv correctness nodes pass with fresh JIT evidence.
  - `vae_sdxl_hs` and `segformer_ws` meet the 1.5% performance gate.

### 2. Establish an opaque `McastArgs` ABI boundary everywhere

- Audit all 13 migrated kernel rows and all 12 migrated host bindings from the ledger, not only width-sharded Conv.
- In every device kernel:
  - The first CT field after a helper block starts at `mcast_args.next_compile_time_args_offset()`.
  - The first RT field after it starts at `mcast_args.next_runtime_args_offset()`.
  - Consecutive multicast blocks are chained from the previous block's two `next_*_offset()` methods.
  - Operation-specific tails use named offsets or a small operation decoder rooted at that boundary.
  - Remove assumptions such as `CT_BASE + 5`, `RT_BASE + 4`, `rt_args_idx += 4`, fixed post-mcast absolute indices, or
    comments/constants that restate the current helper width.
- In every host binding:
  - Insert the complete range returned by `compile_time_args()` and `runtime_args(core)` at one contiguous ABI point.
  - Never extract helper words with `[0]`, `[1]`, etc.
  - Reorder operation-specific prefix/suffix fields when needed instead of preserving historical numeric positions.
  - Cache override indices and buffer bindings must be derived from named operation boundaries after insertion.
- Apply this specifically to:
  - MIG-001's width-sharded Conv CT block and config-tensor tail.
  - MIG-003's Matmul-1D/2D CT/RT tuple extraction and sender `+= 4`.
  - Conv weight senders/receivers, all three GroupNorm helper blocks, and both Sort faces, even where current code already
    appears chained.
- Add a durable source-audit check over the migrated inventory that rejects:
  - indexed access into variables holding helper CT/RT output;
  - manual increments equal to the current helper wire width;
  - downstream numeric offsets whose value depends on helper size.
- Gate:
  - Rebuild with `./build_metal.sh` because Matmul host bindings change.
  - Run one compile-focused parametrization from Matmul, Conv height/block/width, GroupNorm, and Sort.
  - Run the complete helper host/device suites.
  - Run the full mapped correctness inventory for all migrated kernels, with fresh JIT evidence.
- This gate must be green before any CT/RT wire-size change is started.

### 3. Resolve API-004 and the semaphore-ownership part of MIG-003

- Extend `Mcast1D` to retain a rectangular origin and compute sender placement, line indexing, RT coordinates, and
  semaphore coverage relative to it.
- Replace Matmul-2D's vector of per-line `Mcast2D` objects with one offset-aware `Mcast1D` over
  `all_cores_with_work`.
- Let the helper own the Matmul in1 semaphore IDs.
- Add a legacy-`Program` bridge that creates every helper-owned semaphore in ID order and asserts that
  `CreateSemaphore` returned the declared ID; descriptor paths append all `owned_semaphores()`.
- Gate:
  - Rebuild host code.
  - Host gtests cover offset `PerRow`/`PerColumn` on both NoCs, uniform/diagonal placement, line indices, coordinates,
    semaphore ranges, and degeneracy.
  - Run one 1D Matmul node and both 2D transpose orientations first.
  - Run all `MM-IN1-ALL` cases and all three Matmul performance cases.

### 4. Resolve API-001 with a self-describing v10 wire

- Add a dedicated sixth CT word:
  `[active, data_ready, consumer_ready, num_active, flags, rotating_span]`.
- Change `McastArgs<CT_BASE, RT_BASE, SPAN>` to `McastArgs<CT_BASE, RT_BASE>`.
- Derive fixed/rotating mode, receiver type, runtime size, and both next offsets from the constexpr CT span.
- Bump `MCAST_PIPE_API_VERSION` to v10 and update every emitter, kernel, test, changelog entry, and ledger row.
- Because Gate 2 removed all size assumptions, production callers should need only removal of the third template
  argument; downstream operation layouts remain chained.
- API-002 sender/receiver-face enforcement remains explicitly deferred from this rollout.
- Gate:
  - Rebuild host code.
  - Host gtests verify exact fixed/rotating wire contents and chaining.
  - Complete helper device suite passes, including rotating and divergent cases.
  - Run one compile-focused case from every migrated family, then their complete mapped inventories.

### 5. Resolve API-003 and MIG-002

- Make handshaked `send_signal()` wait/reset consumer readiness and handshaked `receive_signal()` acknowledge readiness
  before waiting.
- Preserve existing behavior for `handshake=false`.
- Give Sort separate channels:
  - handshaked row-start Counter channel;
  - no-handshake sub-stage Counter channel.
- Remove the raw reader-ready semaphore; retain the operation-owned writer-done counter.
- Gate:
  - Rebuild host code.
  - Helper tests cover handshaked and non-handshaked signal-only behavior.
  - Run the exact long-sort compile/JIT case, both `Ht=2` deadlock regressions, then all seven long-sort cases.
  - `sort_single_row_524288` meets the 1.5% performance gate.

### 6. Resolve MIG-004 GroupNorm performance coverage

- Classify supported configurations using actual host-generated middle/first/last rectangles.
- Cover zero-edge, one-edge, and two-edge geometry in host tests.
- Profile legacy and Welford independently for every supported production shape class.
- Reuse the recorded rectangular results; gather matched baseline results for new wrapped cases.
- If a class has no supported production configuration, document that and retain synthetic host coverage.
- Any regression above 1.5% must be isolated to pipe construction, degenerate calls, or real sends and fixed before
  closing the item.

### 7. Final release gate

- Run `./build_metal.sh`.
- Run `McastHostFixture.*` and complete `test_mcast_pipe.py --dev`.
- Sequentially run:
  - `MM-IN1-ALL`;
  - `CONV-HEIGHT`, `CONV-BLOCK`, and `CONV-WIDTH`;
  - legacy and Welford `GN-SHARDED-PARAMETERIZED`;
  - `SORT-SINGLE-ROW-CONTROL`.
- Require fresh JIT evidence for all 13 migrated kernels and build coverage for all 12 host bindings.
- Run the complete ten-case performance matrix: four Conv, three Matmul, two GroupNorm, and one Sort case. Every
  median must meet the 1.5% gate.
- Re-run the opaque-ABI source audit from Gate 2.
- Finish with no migrated ledger row marked `needs_recheck`; feedback statuses, changelog, report, and dashboard must
  match v10.

## Public Interface Changes

- `McastArgs` becomes a two-template-argument decoder; rotating span comes from CT metadata.
- All helper block sizes are opaque to callers; only `next_compile_time_args_offset()` and
  `next_runtime_args_offset()` define following boundaries.
- `Mcast1D` supports non-zero rectangular origins.
- Helper-owned semaphore declarations gain a legacy-`Program` application path.
- Signal-only methods honor the existing handshake policy.
- API-002 face enforcement remains deferred.

## Assumptions

- Baseline is `origin/llk_helper_library` at `4a1d6a97ca9`.
- Kernel-only changes do not require a rebuild; host changes use `./build_metal.sh`.
- All device tests run sequentially through `scripts/run_safe_pytest.sh`; the first test after an ABI change is a
  single compile-focused parametrization.
- Borderline performance results receive two additional runs, but final acceptance still requires the median of run
  medians to remain within 1.5%.

## Execution Log

This section is the durable record of the automated execution of this plan. Commands are run from
`/localdev/sjovic/tt-metal`; device tests are serialized through `scripts/run_safe_pytest.sh` after activating
`python_env`.

### 2026-08-04 — Gate 1

- Confirmed the working branch is `sjovic/mcast-migration` at `9618098b754` and its intended baseline is
  `llk_helper_library` / `origin/llk_helper_library` at `4a1d6a97ca9`.
- Preserved the pre-existing modified `tt_metal/third_party/tt_ops_code_gen` submodule and unrelated untracked files.
- Inspected stash commit `91e9d02a8cf`: its first parent is the current branch tip and its worktree delta is exactly
  the six files named by Gate 1.
- Restored those six files without applying or dropping the stash. Verified all six working-tree files are byte-for-byte
  identical to the corresponding tree entries in `91e9d02a8cf` (`git diff --quiet 91e9d02a8cf -- <six files>` returned
  zero).
- Helper validation: `scripts/run_safe_pytest.sh --dev --run-all
  tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py` passed 73/73.
- Fresh-JIT correctness validation used isolated, initially empty `TT_METAL_CACHE` directories and
  `--dev --no-precompile`:
  - SDXL VAE `(1, 4, 512, 128, 128)`, height sharded, `auto_slice=False`: passed at PCC
    `0.9999324991817061` against `0.985`; JIT telemetry reported 0/30 hits. The relevant freshly built kernel hashes
    were sender `8964374999118028120` and receiver `7576033465502356753`.
  - SegFormer `(1, 576, 576, 8, 8)`, groups 576, width sharded, stride 1: passed at PCC
    `0.9998908936797226` against `0.985`; JIT telemetry reported 0/23 hits. The freshly built
    `activation_reader_width_sharded` hash was `15270333159450249187`.
- Confirmed with `tt-smi -s` that performance validation ran on the required single-chip Blackhole p100a at AICLK
  800. Every run used the checked-in real-time-profiler harness with three warmups and 20 measured records.
- `vae_sdxl_hs`: median `28,167.952788692528 ns`, +0.758530% versus the recorded `27,955.899 ns` pre-migration
  baseline — **PASS**. Raw result: `generated/mcast_migration_rt/gate1_restored_20260804_vae_sdxl_hs.json`.
- `segformer_ws` was borderline, so two additional runs were taken as required. Run medians were
  `38,423.58882732641`, `38,397.760748803565`, and `38,334.08213823536 ns`. Their median is
  `38,397.760748803565 ns`, +1.012100% versus the recorded `38,013.030879033 ns` baseline. The 1.5% limit is
  `38,583.22634221849 ns`; the result is below it by `185.465593414923 ns` — **PASS**. Raw results:
  `generated/mcast_migration_rt/gate1_restored_20260804{,_r2,_r3}_segformer_ws.json`.
- Gate 1 is green under the updated 1.5% acceptance rule. The restored implementation is correctness- and
  performance-verified and committed as `a0fad936c89` (`Restore mcast source lifetime optimizations`). The recovery
  stash remains present as an additional safety copy.

### 2026-08-04 — Gate 2

- Audited all 13 migrated kernel rows and all 12 migrated host-binding rows from
  `helper_design/mcast_pipe/migration/ledger.json`.
- Replaced helper-width assumptions in the migrated device kernels:
  - Matmul sender runtime parsing now resumes at `McastArgs::next_runtime_args_offset()`; the receiver's compile-time
    operation tail and tensor accessor now derive from `next_compile_time_args_offset()`.
  - Width-sharded Conv's activation/config tail now derives from the activation `McastArgs` boundary.
  - Both legacy and Welford GroupNorm sender/receiver kernels now name the full three-block mcast chain and derive the
    operation tail from the last block. The receiver faces name the otherwise-unused first/last blocks so host and
    kernel layouts cannot drift silently.
- Reworked both legacy and descriptor Matmul 1D/2D host paths to insert each helper compile-time and runtime output as
  one complete range. Cached-program output/bias patch indices are stored in shared variables and derived from the
  emitted helper runtime range size.
- Confirmed the migrated Conv, GroupNorm, and Sort host bindings already append, insert, assign, or move complete
  helper ranges without indexing helper output.
- Added `tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe_source_audit.py`. It derives the migrated inventory from the
  ledger, rejects indexed access into host helper outputs, and rejects the known fixed-width/fixed-tail device
  patterns. The audit passed 8/8.
- Host rebuild with `./build_metal.sh` completed successfully.
- Seven compile-focused device parametrizations passed sequentially through `run_safe_pytest.sh --dev
  --no-precompile`, each with its own initially empty `TT_METAL_CACHE`: Matmul 2D (0/27 JIT hits), Conv height
  (PCC `0.9999993139704398`, 0/27), Conv block (PCC `0.9999992596260122`, 0/29), Conv width
  (PCC `0.9999992597711427`, 0/26), GroupNorm legacy (0/31), GroupNorm Welford (0/31), and Sort long-tensor
  (0/29). The first Matmul invocation used the stale pre-collection node ID and was rejected by pytest without
  running code; the corrected current node passed.
- Complete helper validation is green: `McastHostFixture.*` passed 19/19 and
  `test_mcast_pipe.py --dev --run-all` passed 73/73.
- The complete mapped correctness inventory passed sequentially from initially empty per-family JIT caches:
  - `MM-IN1-ALL`: 302 passed / 188 expected skips / 490 selected, exactly matching the recorded baseline (four
    chunks: 56/72, 46/50, 46/50, and 154/16).
  - `CONV-HEIGHT`, `CONV-BLOCK`, and `CONV-WIDTH`: each feature inventory passed 48/48 runnable cases with 16
    expected row-major/BFLOAT8 skips; each matching DRAM-config node passed, and the shared DRAM inventory passed
    14/14.
  - `GN-SHARDED-PARAMETERIZED`: legacy passed 108/108 runnable cases with 2 expected offset-grid skips; Welford
    passed the same 108/2 split; the fixed/default-routing inventory passed 19/19 runnable cases with 6 expected
    20-core-only skips.
  - `SORT-SINGLE-ROW-CONTROL`: all seven long-tensor cases and both `Ht=2` deadlock regressions passed.
- Re-ran the durable opaque-ABI audit after the full inventory: 8/8 passed. `git diff --check` is clean.
- Gate 2 is green. No helper wire-size change was started before this gate completed.
- Committed the Gate 2 implementation as `adf71f3dcde` (`Make mcast helper ABI boundaries opaque`).

### 2026-08-04 — Gate 3

- Extended `Mcast1D` to accept any dense rectangular `CoreRangeSet`, retain its logical origin, and perform sender
  placement, line indexing, sender-coordinate lookup, multicast bounding-box construction, and semaphore coverage
  relative to that origin. Sparse/non-rectangular sets are rejected by comparing the set size to its bounding box.
- Added a legacy-`Program` bridge for helper-owned semaphores. It sorts declarations by ID, supports worker
  semaphores, creates each declaration through `CreateSemaphore`, and fails immediately if the allocated ID differs
  from the helper-declared ID. Descriptor factories continue to append the complete `owned_semaphores()` range.
- Replaced Matmul-2D's vector of per-line `Mcast2D` objects with one offset-aware `Mcast1D` over
  `all_cores_with_work`: `transpose_mcast=true` uses `PerRow`, while `false` uses `PerColumn`. The helper owns the in1
  semaphore pair in both legacy and descriptor paths. Matmul-1D retains its whole-grid `Mcast2D` geometry but also
  delegates ownership of its in1 semaphore pair to the helper.
- Added host gtests for offset `PerRow` and `PerColumn` layouts on NoC 0/1, uniform and diagonal sender placement,
  sender/receiver coordinates, line-relative selection, NoC-ordered bounding boxes, helper semaphore ranges,
  degenerate one-core lines, and the legacy semaphore bridge's declared-ID ordering.
- The first rebuild exposed only a namespace typo in the new bridge (`tt::tt_metal::CoreType`); correcting it to
  `tt::CoreType` resolved the compile error. The subsequent `./build_metal.sh` completed successfully.
- `McastHostFixture.*` passed 25/25, including all six new offset/bridge cases. The Gate 2 opaque-ABI source audit
  remained green at 8/8 and `git diff --check` remained clean.
- Focused device validation used one initially empty isolated JIT cache and `--dev --no-precompile`. One Matmul-1D
  in1-multicast case and both Matmul-2D `transpose_mcast=true/false` orientations passed sequentially without watcher,
  assertion, or timeout failures.
- The complete `MM-IN1-ALL` inventory passed from a separate initially empty cache: 302 passed / 188 expected skips /
  490 selected, exactly matching the recorded baseline. The four chunks were 56/72, 46/50, 46/50, and 154/16.
- Confirmed the required device remained a single-chip Blackhole p100a at AICLK 800. The checked-in real-time-profiler
  harness collected three warmups and 20 measured records for every Matmul case:
  - `matmul_2d_sdxl_ff_gelu`: `164,173.30109646538 ns`, +0.0286% versus `164,126.2962962963 ns` baseline — **PASS**;
  - `matmul_1d_sdxl_resnet_960_320`: `77,412.96296296295 ns`, +0.6133% versus `76,941.11111111111 ns` baseline —
    **PASS**;
  - `matmul_2d_transpose_mcast`: `12,308.518518518518 ns`, -1.1658% versus `12,453.703703703703 ns` baseline —
    **PASS**.
- None of the Matmul results was borderline against the 1.5% limit, so no additional performance runs were required.
  Raw results are `generated/mcast_migration_rt/gate3_20260804_matmul_*.json`.
- Gate 3 is green.

### 2026-08-05 — Gate 4

- Implemented API-001 and bumped `MCAST_PIPE_API_VERSION` from 9 to 10. The uniform helper CT block
  is now `[active, data_ready, consumer_ready, num_active, flags, rotating_span]`; zero selects the
  fixed four-word RT layout and a nonzero span selects `4 + 2 * rotating_span` RT words.
- Removed the caller-supplied rotating template parameter. `McastArgs<CT_BASE, RT_BASE>` now derives
  fixed/rotating mode, receiver type, sender count, runtime width, and both next offsets from the
  constexpr sixth CT word. The width-sharded Conv activation reader and rotating helper kernel were
  updated to the two-argument decoder. API-002 sender/receiver-face enforcement remains explicitly
  deferred.
- Updated exact host-wire expectations for fixed, rotating, divergent, and degenerate `Mcast1D` and
  `Mcast2D` layouts. `./build_metal.sh` passed; `McastHostFixture.*` passed 25/25; the complete helper
  device suite passed 73/73 from an initially empty cache, including rotating and divergent cases.
- Seven compile-focused production cases passed sequentially through `run_safe_pytest.sh --dev
  --no-precompile`, each with an initially empty isolated cache: Matmul 2D, Conv height/block/width,
  GroupNorm legacy/Welford, and Sort long-tensor. The width-sharded case—the material rotating-wire
  risk—passed at PCC `0.9999992597711427` with 0/26 JIT hits. Matmul, both GroupNorm variants, and
  Sort likewise reported zero cache hits (0/27, 0/31, 0/31, and 0/29).
- Complete mapped inventories passed sequentially from initially empty per-family caches:
  - `MM-IN1-ALL`: 302 passed / 188 expected skips / 490 selected (56/72, 46/50, 46/50, 154/16).
  - `CONV-HEIGHT`, `CONV-BLOCK`, and `CONV-WIDTH`: each feature inventory passed 48 runnable cases
    with 16 expected skips; each matching DRAM-config case passed; shared DRAM passed 14/14.
  - `GN-SHARDED-PARAMETERIZED`: legacy 108/2, Welford 108/2, fixed/default routing 19/6.
  - `SORT-SINGLE-ROW-CONTROL`: long-tensor 7/7 and both `Ht=2` deadlock regressions 2/2.
- Added a durable audit rejecting a third `McastArgs` template argument in any migrated kernel. The
  complete opaque-boundary audit passed 9/9, all 25 migrated ledger rows are API v10 with a
  2026-08-05 verification date, both migration JSON files parse, and `git diff --check` is clean.
- Gate 4 is green. The implementation and v10 metadata are committed as `3726bdd71f3`
  (`Make mcast rotating wire self-describing`).

### 2026-08-05 — Gate 5

- Made signal-only traffic honor the configured handshake policy. A handshaked `send_signal()` now waits for and
  resets all consumer acknowledgements before publishing its control signal; `receive_signal()` acknowledges the
  current sender before waiting and advances its sender round for both Counter and Flag modes. No-handshake behavior
  is unchanged.
- Split Sort's coordinator-to-worker control path into a handshaked row-start Counter channel and a no-handshake
  sub-stage Counter channel. The helper now owns row-start readiness; the raw reader-ready semaphore was removed,
  while the independent operation-owned writer-done counter remains.
- Extended the helper control-only test matrix across both handshake policies, 1x2/1x8 rectangles, and 2/32 rounds.
  `./build_metal.sh` passed; a fresh-JIT handshaked 1x2 compile case passed with 0/20 cache hits; the complete helper
  suite passed 77/77 from an initially empty cache with 0/133 cache hits.
- The exact long Sort `(1, 524288)` compile/JIT case passed from an initially empty cache with 0/29 hits. Both `Ht=2`
  deadlock regressions passed 2/2, and the complete long-tensor Sort inventory passed 7/7.
- Added a durable source audit requiring Sort row-start readiness to remain pipe-owned. The complete audit passed
  10/10. API-003 and MIG-002 are marked implemented, both migration JSON files parse, and `git diff --check` is
  clean.
- The Sort performance case was repeated three times because the result was near the limit. Run medians were
  `145,201,100.41355687`, `144,983,524.86661464`, and `145,768,174.93690944 ns`. Their median is
  `145,201,100.41355687 ns`, +1.195124% versus the recorded `143,486,262.2222222 ns` baseline. The 1.5% limit is
  `145,638,556.15555552 ns`; the result is below it by `437,455.7419986427 ns` — **PASS**. Raw results are
  `generated/mcast_migration_rt/gate5_20260805_sort_single_row_524288{,_r2,_r3}.json`.
- Gate 5 is green. The implementation and metadata are committed as `160effde4fc`
  (`Make mcast signal handshakes explicit`).

### 2026-08-05 — Gate 6

- Audited the actual GroupNorm sharded-v2 group construction, its rectangular-grid requirement,
  row/column traversal, batch/group divisibility checks, and every mapped block- and height-sharded
  production configuration. The mapped production inventory generates only rectangular groups, so
  its supported performance class is zero-edge. No mapped production configuration reaches a one-
  or two-edge wrapped partition.
- Added direct host coverage of the production `split_and_form_rectangle_grids` implementation.
  `GroupNormMcastGeometry` passes zero-edge, one-edge, and two-edge coordinate sequences 3/3; the
  latter two preserve defensive coverage for currently unreachable wrapped classes.
- `./build_metal.sh` passed. The combined `GroupNormMcastGeometry.*:McastHostFixture.*` run passed
  28/28 (3 geometry plus 25 helper-host cases).
- Reused the matched Blackhole p100a zero-edge results recorded before this gate, as the plan permits:
  SDXL `(1, 1920, 32, 32)` legacy is +0.248% and Welford is -0.485% versus baseline. Both pass the
  1.5% gate, so no helper/kernel hot-path change or new wrapped baseline was warranted.
- MIG-004 is marked implemented; the changelog, report, migration log, ledger JSON/Markdown, and
  validation evidence now record the production/synthetic distinction. The ledger JSON parses and
  `git diff --check` is clean.
- Gate 6 is green. The host coverage and metadata are committed as `fc4f251be19`
  (`Close GroupNorm mcast geometry coverage`).

### 2026-08-05 — Gate 7 (green after controlled baseline rerun)

- `./build_metal.sh` passed. `McastHostFixture.*` and the three GroupNorm geometry tests passed 28/28;
  the complete helper device suite passed 77/77 from an initially empty cache with 0/133 cache hits;
  and the opaque-ABI/source audit passed 10/10.
- All mapped correctness inventories passed sequentially from initially empty per-family caches:
  Matmul 302/188, each Conv height/block/width feature inventory 48/16 plus all mapped DRAM cases,
  GroupNorm legacy 108/2, Welford 108/2, fixed/default routing 19/6, and Sort long/deadlock 7/7 plus
  2/2. Fresh artifacts covered all 13 migrated ledger kernel paths; the current host build covered all
  12 migrated host bindings.
- Nine of ten current-code performance cases pass the 1.5% limit: four Conv cases range from -0.273%
  to +1.096%, three Matmul cases range from -0.074% to +0.929%, GroupNorm Welford is -0.358%, and
  Sort is +0.449%. GroupNorm legacy measured `49,812.22222222222 ns`, +2.508% versus its
  `48,593.7037037037 ns` baseline, and therefore fails.
- Two additional unchanged-code legacy runs measured `49,817.50296151212` and `49,854.10571354485 ns`;
  all three run medians fail, so this is not a borderline result under the plan's repeat rule.
- Focused correctness recompiles continued to pass from empty caches while isolating the regression.
  Compile-time removal of inactive first/last sends, explicit outlining of the send path, cached
  receiver coordinates, normal instead of forced internal inlining, an active/in-place send path, and
  conditional no-handshake storage were each measured independently. None closed the gate; all
  experimental code was removed. Skipping construction of the two inactive edge pipes was the only
  measurable improvement (`49,554.07536560204 ns`, +1.976%), but retaining it would still leave the
  release gate red.
- Device-profiler evidence shows stable individual BRISC kernels but increasing cross-core start/end
  skew over the 20 back-to-back launches; the first measured record remains near the historical
  baseline while later records form the failing median. The existing historical baseline artifact
  is stable across all 20 records, so a same-environment historical checkout is required to distinguish
  a code regression from a current profiler/dispatch-state effect before making a broader kernel change.
- The user authorized an isolated checkout, so the apparent blocker was reproduced in
  `/localdev/sjovic/tt-metal.worktrees/mcast-perf-bisect` with its own recursively initialized submodules,
  Release build, and Python environment. Import verification resolved `ttnn` from that worktree. Because
  the realtime benchmark was added after the older snapshots, the same harness from `f568dfe62ef` was
  carried into those snapshots as a test-only untracked file.
- Same-environment runs used the same Blackhole p100a, firmware `19.6.0`, realtime-profiler frequency of
  approximately 1.35 GHz, three warmups, and 20 measured records. The actual pre-migration baseline
  `4a1d6a97ca9` measured `49,694.26516945126 ns`; the previously passing migrated snapshot
  `28356d43846` measured `49,850.05759004791 ns`; and current `2699996541a` measured
  `49,836.38882787317 ns`.
- Current is +0.285996% versus the freshly reproduced pre-migration baseline and -0.027420% versus the
  previously passing migrated snapshot. The current result is `603.2903191198566 ns` below the 1.5%
  limit of `50,439.679146993025 ns` — **PASS**. Raw artifacts are under the worktree's
  `generated/mcast_migration_rt/` directory with `-worktree` commit tags.
- The old `48,593.7037037037 ns` artifact does not reproduce for either the baseline or the previously
  passing migrated snapshot under the current environment. The controlled comparison therefore rules
  out a migration-commit regression; no intermediate bisect or speculative production-code change is
  warranted.
- Final consistency cleanup marked API-004 and MIG-003 implemented with their Gate 2/3 evidence,
  corrected the ledger aggregate and test-map metadata to v10, and refreshed the README, changelog,
  report, and dashboard. API-002 remains the one deliberately open/deferred API item. The machine
  checks report API v10, 13 migrated kernels, 12 migrated host bindings, and zero open
  `needs_recheck` flags; both JSON files parse, the source audit passes 10/10, and `git diff --check`
  is clean.
- Gate 7 is green. All seven ordered gates are complete, and no correctness or production-code change is
  left uncommitted.
