# Prioritized `mcast_pipe` Feedback and Validation Plan

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
- API-002 remains explicitly deferred: no face metadata or RT compaction in this rollout.
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

### 2026-08-04 — Gate 2 (in progress)

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
