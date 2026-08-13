# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- Final Watcher runs intentionally disable Ethernet watching because the firmware Watcher configuration buffer cannot hold the active Ethernet kernel set. Worker and dispatch kernels remain watched, and the retained logs show clean checks and detach on all four devices; this is a scoped observability limitation, not evidence of a model failure.
- The retained Python hardware logs end with nanobind leaked-object/function diagnostics during interpreter teardown. Device close, Watcher detach, numerical results, and fallback audits complete first, so these diagnostics do not contradict the stage result, but they remain a repo/runtime hygiene issue outside this decoder optimization.
- The final default is deliberately only marginally different from the measured inherited baseline. The stage's substantive result is that the tested topology, collective, packing, movement, precision, and program families did not beat the restored default; the report correctly uses the restored-default measurements and does not claim the earlier persistent-buffer candidate win.

## Hard-Check Gaps

- Decode profiler captures use four measured trace steps and show a small profiler overhead versus the separate 16-step final timing runs (3.47% full, 0.79% linear). The raw CSV, human operation table, profiler-run host median, and unprofiled final median are all retained and explicitly separated; no invalid duration or stale-default contradiction was found.
- Linear prefill S128 Tracy postprocessing remains tool-limited, so the retained human profiler table is the S5 non-aligned path. S128 correctness, timing, stress, Watcher, replica equality, and fallback evidence are independently present, and this profiler limitation does not hide a correctness or selected-default decision.
- The worktree is live and contains unrelated dirty/untracked paths. This review assessed stage correctness and evidence only; stage-owned checkpoint isolation remains an orchestrator follow-up under the stage-review skill.

## Anomaly Ledger

- Observed anomaly: The persistent preallocated-CCL candidate's isolated sub-percent win did not reproduce in the final default and reduced the proven B32 context bracket.
  Evidence: `artifacts/candidates/{full,linear}_preallocated_ccl_b{1,32}.json`, superseded `artifacts/final/default_*.json`, restored `artifacts/final/default_restored_*.json`, and capacity artifacts.
  Affected path: TP4 row-parallel attention/output and MLP-down collectives.
  Control or comparison: Restored non-preallocated `default` reruns at B1/B32 and the original baseline artifacts.
  Likely subsystem: Benchmark noise plus persistent full-stack DRAM residency.
  Investigation performed: Candidate/default reproduction was compared, the full-stack buffer cost was modeled and capacity-probed, and the non-preallocated code default was restored and rerun.
  Resolution: fixed; the candidate is rejected, final B32 is 0.722050 ms full and 4.431181 ms linear, and the C82432-pass/C82496-fail bracket is preserved.

- Observed anomaly: The initial profiler text artifacts contained CSV-mode status chatter rather than human operation tables.
  Evidence: Prior `*_final/perf_table.txt` files versus restored-final `human_table.txt`, `perf.csv`, raw reports, and console/stderr provenance.
  Affected path: Final decode profiling and dtype/fidelity/advice audit.
  Control or comparison: `tt-perf-report` rerun without `--csv` on retained restored-default raw captures.
  Likely subsystem: Artifact postprocessing invocation.
  Investigation performed: Human and CSV outputs were regenerated separately and operation rows were checked directly.
  Resolution: fixed; readable rows show full BF16/HiFi2 attention projections, BFP4/LoFi MLP projections, linear BFP4/LoFi projections, and BF16/HiFi2 recurrent matmuls.

- Observed anomaly: Linear recurrent movement and intermediate placement were previously left open.
  Evidence: `artifacts/candidates/linear_packed_l1_b32.json`, its retained logs, recurrent geometry evidence cited in `work_log.md`, and the live `multichip_linear_packed_l1` implementation.
  Affected path: Linear-attention packed projection split and recurrent state/intermediates.
  Control or comparison: Restored default linear B32 at 4.431181 ms versus adapted packed-L1 candidate at 4.431050 ms.
  Likely subsystem: L1 capacity and conversion/layout costs.
  Investigation performed: The packed output was kept in L1 through split, downstream components were individually moved to DRAM after the 5,242,880-byte single-bank request proved illegal, and recurrent state size/free-L1 plus prior precision-locked grid/block/subblock sweep were audited.
  Resolution: controlled; the candidate is statistically neutral with more conversions, while the 12,582,912-byte/device BF16 recurrent value cannot reside in a 1,461,504-byte L1 bank (87,296-byte measured free block). The default remains selected.

- Observed anomaly: Linear S128 prefill stress exposed a 130-byte row-major slice read into a 128-byte CB page.
  Evidence: `triage/LINEAR_PREFILL_STRESS_{AUTOTRIAGE,AUTOFIX}.md`, focused and existing-slice Watcher logs, final prefill Watcher JSON/logs, and the live slice source/test diff.
  Affected path: Linear-attention prefill slices; shared row-major slice CB sizing.
  Control or comparison: Exact 128-byte BF16 useful row plus 2-byte prefix regression, existing 5-D misaligned slice, and full-attention S128 prefill control.
  Likely subsystem: Host-side slice circular-buffer page capacity.
  Investigation performed: Reader byte span was reconciled with host allocation; CB sizing now rounds `unpadded_row_size_bytes + misalignment`; rebuilt/installed runtime passed the focused control and original TP4 command.
  Resolution: fixed; final linear/full S128 warmup4+16 Watcher runs pass PCC 0.99999433/0.99999460, replica equality, and fallback hard-failure.

- Observed anomaly: Fused matmul-to-reduce-scatter left pending non-posted atomics.
  Evidence: `triage/AUTOTRIAGE.md`, focused Watcher logs/artifacts, and the guarded `noc.async_atomic_barrier()` live kernel diff.
  Affected path: Optional fused TP4 matmul/reduce-scatter candidate.
  Control or comparison: Matched separate async matmul plus reduce-scatter control across 8x4/8x6 grids and block widths 1/2/3/4/6.
  Likely subsystem: Fused `OpSignaler` atomic drain.
  Investigation performed: AutoTriage/AutoFix, guarded barrier, focused fused and ordinary Watcher controls, and candidate geometry sweep.
  Resolution: fixed for correctness and rejected for performance; fused remains 9.7-11.1% slower than separate async and is not the default.

- Observed anomaly: Adapted fused all-gather-to-matmul retries hit the receiver ledger assertion.
  Evidence: `triage/ALL_GATHER_MATMUL_AUTOTRIAGE.md` and the three focused Watcher retry logs.
  Affected path: Optional fused fractured-residual consumer candidate.
  Control or comparison: Separate collective/matmul and replicated/fractured stack candidates pass.
  Likely subsystem: Fused receiver transfer/K-block contract.
  Investigation performed: API, operand, semaphore, layout, and transfer-count adaptations plus AutoFix; speculative source change was reverted.
  Resolution: controlled exact runtime blocker; the candidate is rejected and absent from the final default.

## Scope Inspected

- Goal/skill paths: supplied optimized-multichip-decoder contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/optimize/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: `doc/optimized_multichip_decoder/{STAGE_REVIEW.md,README.md,work_log.md,artifacts,logs,triage}`; especially restored-default, 64-step decode stress, post-AutoFix S128 prefill stress, restored Tracy tables/CSV/raw captures, recurrent packed-L1 candidate, non-aligned/context/capacity evidence; `doc/context_contract.json`.
- Code paths: `tt/multichip_decoder.py`; multichip benchmark/smoke/stack/probe tests; `tests/ttnn/unit_tests/operations/data_movement/test_slice.py`; row-major slice program factory; fused matmul sender/writer kernel.
- Commands run: read-only `sed`, `find`, `rg`, `jq`, `git status`, `git diff`, artifact inventories, JSON inspection, profiler-table/CSV inspection, and log tail/error scans. No device was opened and no hardware/server/profiler experiment was run.

## Residual Risk

- Performance conclusions depend on short warmed layer-level runs and include sub-percent differences; the report appropriately rejects neutral/noisy candidates instead of advertising them as wins.
- Ethernet kernels were not Watcher-observed in final stress runs, though the selected default uses the already-established synchronous ring path and all worker/dispatch checks, PCC, state/cache progression, and teardown are clean.
- The shared slice CB fix and guarded fused-kernel atomic barrier have focused source-backed hardware controls but have not been represented here as a broad repository-wide CI run.
