# Stage Review

Verdict: clean-pass

## Required Work

- None.

The two findings from the first review are closed:

- P1, page-boundary coverage: the optimized single-chip reference and TP2
  target now build the same cache state from a non-aligned 31-token prefill and
  ordered decode writes through positions 31..65.  The target uses disjoint,
  nonidentity physical mappings for virtual pages zero and one.  The test
  compares outputs at positions 63, 64, and 65; reconstructs rank-local K and V
  heads from both physical pages; proves the unwritten suffix of page one is
  zero; and replays a fixed position-65 paged decode trace five times.  The
  retained post-fix gate records PCC 0.9999658 or better for the three boundary
  outputs, PCC 0.9998869 or better for every physical page/rank/cache
  comparison, PCC 1.0 for every trace replay, and bitwise replay determinism.
- P2, CCL ownership: `MultiChipDecoder` now accepts a `TT_CCL` owner through
  both construction APIs, defaults to `get_tt_ccl(mesh_device)`, and rejects an
  owner from another mesh.  The hardware-free 32-layer regression proves all
  layers resolve to one owner and one cluster-axis-none barrier vector, with
  exactly 36 global semaphore creations for the mesh (6 barrier, 12
  all-gather, and 18 reduce-scatter).  The target-mesh correctness test also
  checks the live decoder's owner identity.  Correctness, watcher stress, and
  final warmed performance were rerun after this wiring change.

## Other Concerns

- No stage-blocking contradiction was found between the implementation,
  context contract, final-gate summary, candidate tables, canonical Tracy CSV,
  filtered `tt-perf-report` tables, and watcher log.
- The canonical profiler rows independently confirm the selected runtime
  policy: BFP4 weights and LoFi math for local `[4096,3072]`, `[2048,4096]`,
  `[4096,7168]`, and `[7168,4096]` projections, BF16 activation/CCL payloads,
  and paired reduce-scatter/all-gather records on both devices.  The final
  reproduced default is 1.948x prefill speedup at 97.4% TP efficiency and
  1.449x traced-decode speedup at 72.5% efficiency.
- The replicated-residual decision is earned rather than assumed.  The
  shape-faithful fractured boundary includes reduce-scatter, local residual
  add, distributed RMS statistics, all-gather, and the next BFP4 QKV
  projection.  It passes at approximately 0.999976 PCC but is 19.3% slower.
  The fused AGMM repair ladder reached the physical topology restriction only
  after adapting rank, persistent-shard count, and BFP4 weight placement.

## Hard-Check Gaps

- No blocking hard-check gap remains.  This reviewer did not open TT devices
  or rerun hardware commands, as required by `$stage-review`; hardware results
  were checked against `final_gate_results.txt`, the current assertions and
  call graph, the canonical profiler CSV, filtered human-readable reports, and
  the current generated watcher log.
- Fresh shard advice is unavailable because the installed prebuilt tt-mlir
  runtime and this checkout have an exact TTNN symbol/ABI mismatch.  The
  retained traceback proves that environment blocker.  The stage followed the
  advisor skill's no-local-rebuild rule and replaced no measured evidence with
  stale advice.

## Anomaly Ledger

- Observed anomaly: a 63-token optimized-baseline prefill exceeds Blackhole L1
  static-CB capacity.
  Evidence: the isolated padded-64 gate-matmul error requires 1,659,648 bytes
  against a 1,572,864-byte limit; `final_gate_results.txt` and `work_log.md`
  retain the classification.
  Affected path: the single-chip reference fixture for that exact prefill
  geometry, not the paged decode implementation or public TP2 alignment check.
  Control or comparison: all earlier tensors were explicitly deallocated, so
  fragmentation was refuted.  A valid non-aligned 31-token prefill followed by
  ordered writes reaches the identical logical cache state through position
  65 and compares TP2 to the optimized baseline at the boundary.
  Likely subsystem: single-chip padded-64 prefill MLP circular-buffer sizing.
  Investigation performed: deallocation/lifetime cleanup, isolated rerun, and
  the position-31..65 boundary-walk replacement.
  Resolution: controlled.

- Observed anomaly: full watcher Ethernet instrumentation passes model work
  and ten trace checks, then firmware 19.8.0 times out while restoring an
  active Ethernet core during MetalContext teardown.
  Evidence: `watcher_summary.txt`, `work_log.md`, and the retained watcher log.
  Affected path: watcher active-Ethernet detach/restore, after decoder and CCL
  execution have completed.
  Control or comparison: the post-shared-CCL target test exits zero with only
  ETH instrumentation disabled; worker, NoC, CB, stack, dispatch, ring-buffer,
  and assert checks remain enabled.  Independent correctness, trace,
  determinism, and Tracy records cover the fabric path.
  Likely subsystem: Blackhole watcher/active-ERISC teardown.
  Investigation performed: target-only isolation and ETH-only feature
  disablement.
  Resolution: controlled.

- Observed anomaly: early BF16/BFP8 experimental projection policies overflow
  L1, and early TP prefill used a decode-height sharded collective output.
  Evidence: `triage/AUTODEBUG.md`, retained triage output, and the work log.
  Affected path: rejected projection policies and the original prefill
  all-reduce memory configuration.
  Control or comparison: production BFP4/LoFi fits; prefill collectives now
  return DRAM-interleaved tensors while decode keeps width-sharded L1.  Current
  correctness and profiler evidence exercise both paths.
  Likely subsystem: matmul CB sizing and prefill/decode memory-contract split.
  Investigation performed: fresh-context AutoDebug, short-traceback reruns,
  shape-level diagnosis, and affected-gate reruns.
  Resolution: fixed.

- Observed anomaly: pytest failure formatting made an interrupted TTNN command
  look like a CQ/device hang, while deeper low-level triage reads hit an
  installed UMD `noc_read(memoryview)` mismatch.
  Evidence: `triage/AUTODEBUG.md`, `triage/tt-triage.txt`, and
  `triage/triage-summary.txt`.
  Affected path: failure reporting and optional diagnostic reads, not final
  decoder execution.
  Control or comparison: `--tb=short` exposes the real model/configuration
  errors; final bounded tests close cleanly and available device-health checks
  are normal.
  Likely subsystem: TT tensor repr during pytest reporting and diagnostic ABI.
  Investigation performed: stack reconstruction, bounded recovery, and
  controlled reruns.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: the complete original multichip-decoder goal;
  `.agents/skills/stage-review/SKILL.md`, `.agents/skills/multichip/SKILL.md`,
  `.agents/skills/tt-device-usage/SKILL.md`, and Multi-Device section 3.3 of
  `tech_reports/LLMs/llms.md`.
- Artifact paths: `doc/context_contract.json` and every relevant file under
  `doc/multichip_decoder/`, including `README.md`, `work_log.md`,
  `final_gate_results.txt`, candidate/topology CSVs, topology plan, shard-advice
  failure, watcher summary/log, triage reports, canonical Tracy CSV, and all
  filtered `tt-perf-report` tables.
- Code paths: `tt/multichip_decoder.py`,
  `tests/test_multichip_decoder.py`, `tt/optimized_decoder.py`,
  `tt/functional_decoder.py`, and `models/common/modules/tt_ccl.py`.
- Commands run: read-only `git status`/`git diff`/`find`/`stat`/`sed`/`grep`,
  JSON and CSV inspection scripts, `python -m py_compile`, `git diff --check`,
  and the hardware-free static selection
  `pytest -q .../test_multichip_decoder.py -k 'runtime_path_is_real_multichip or context_capacity_contract or stack_shares_one_ccl_owner'`
  (3 passed).  No TT hardware, reset, watcher, profiler, server, or long test
  was run by this reviewer.

## Residual Risk

- A 131,072-token full-model allocation/run remains downstream full-model work.
  This decoder stage does not lower the advertised contract: the checked
  per-device plan fits the allocator with 8.0 GiB BF16 or 4.25 GiB BFP8 KV
  cache, 1.828 GiB decoder projection weights, and documented conservative
  allowances.  Page size/count, local four-head ownership, partial-page
  semantics, and page-boundary execution are internally consistent.
- The implementation intentionally supports only the exact two-device P300
  mesh.  The physical line topology and ETH-watcher teardown limitation remain
  documented hardware constraints.
- At review time the stage is a live worktree based on checkpoint
  `00fecbe6a10`; the stage-owned checkpoint commit and SHA log are correctly
  pending this clean review.  The commit must include ignored CSV/profiler
  artifacts explicitly and must exclude the unrelated dirty `.agents` skill
  files, then record the local SHA without pushing.
