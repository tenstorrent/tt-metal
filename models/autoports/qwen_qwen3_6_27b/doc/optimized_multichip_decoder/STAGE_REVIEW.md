# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The selected persistent-CCL default does not reproduce its claimed performance win and reduces supported B32 context.
  Evidence: `artifacts/candidates/full_preallocated_ccl_b32.json` reports 0.719232 ms versus the current-environment 0.722288 ms baseline, but the required final-default reproduction in `artifacts/final/default_full_b32.json` is 0.722339 ms, slightly slower than baseline. The same final artifact is the number correctly disclosed in the README. The selected path nevertheless reserves 52,428,800 bytes/device across a 64-layer stack; `doc/context_contract.json` lowers the measured B32 limit from C82432/C82496 to C82240/C82304. The final full-B32 result is also materially slower than the candidate relative to the entire 0.42% candidate improvement used to select it.
  Why this matters: The optimize contract requires the final default to reproduce the selected best candidate and forbids claiming an earlier sample when the final wiring/run is slower. Here the default trades away measurable context capacity without a reproduced full-B32 latency benefit.
  Required next step: Repeat matched, interleaved baseline/preallocated measurements with enough samples to establish whether the sub-percent change is real. Keep the persistent path only if the final default reproducibly wins the target workloads; otherwise restore the non-preallocated default and the prior capacity bracket. Then regenerate final-default, capacity, Watcher, and profiler evidence.

- P1: The purported human `tt-perf-report` tables are not human operation tables, and the required decode performance accounting is absent.
  Evidence: `artifacts/tracy/full_b32_final/perf_table.txt`, `linear_b32_final/perf_table.txt`, `full_prefill_s128_final/perf_table.txt`, and `linear_prefill_s5_final/perf_table.txt` contain CSV-mode command/status chatter (`Writing CSV output...`, warnings, and summary-generation messages), not rendered per-operation tables. This is exactly the artifact-construction error prohibited by the optimize skill. The README reports only modeled aggregate bandwidth (143/20 GB/s); it does not reconcile theoretical bytes/bandwidth roofline, signposted device-time/token, and warmed host end-to-end time from the same run, nor quantify dispatch/host gaps.
  Why this matters: Without readable operation rows and the three-number reconciliation, the review cannot verify dominant ops, dtype/fidelity actually used, residual/collective costs, actionable advice, or whether profiler durations agree with wall time. The goal explicitly requires tt-perf tables/CSV/provenance.
  Required next step: Re-run `tt-perf-report` without `--csv` to produce genuine advice-enabled human tables for the retained final profiles, retain separate CSV outputs/console logs, and add same-run theoretical, device-time, and end-to-end decode accounting for both layer kinds. Re-audit every recommendation and dominant row from those tables.

- P1: The operation-topology audit leaves a material linear-recurrent movement family open without candidate evidence.
  Evidence: The initial audit in `work_log.md` explicitly says the linear recurrent path has DRAM state traffic and `untilize`/slice/typecast movement, proposes persistent/preallocated state/intermediates plus memory placement, and marks that family “remain open.” The later work log supplies an inherited recurrent matmul geometry result, but no candidate table or exact blocker for persistent intermediates, removal of the DRAM conversion, or alternate recurrent memory placement. The final linear B32 profile reports only 3.8% modeled DRAM-roofline utilization and the final linear decode remains essentially unchanged (4.433326 to 4.429253 ms).
  Why this matters: The goal requires every applicable optimization to be tried with evidence, and the optimize skill specifically requires movement/layout work around dominant non-matmul ops rather than relying on inherited matmul geometry. An audit row cannot be closed by omission.
  Required next step: Measure the listed persistent/intermediate and memory-layout candidates on the real traced linear path, including whole-layer PCC and latency, or preserve an exact capacity/API/minimal-repro blocker. Use AutoFix if the implementation crosses multiple op/layout boundaries.

- P2: Optimized stress coverage required by the skill is not demonstrated.
  Evidence: Final artifacts contain eight-step timing runs and 16-step Watcher runs for B32 decode, plus focused non-aligned prefill cases. Neither README/work log nor artifacts identify a stress test for both representative layer kinds and all exercised modes; `rg` finds no stress classification or result. A short deterministic trace/Watcher run is correctness evidence, not automatically the optimized stress gate.
  Why this matters: The optimize final audit explicitly requires optimized stress to pass every representative layer kind and exercised mode; skipped stress is not passing evidence.
  Required next step: Run and label bounded optimized stress coverage for full-attention and linear-attention decode, and applicable prefill modes, with fallback/cache/state checks and clean Watcher evidence, or document the exact existing artifact and command if one already satisfies this gate.

## Other Concerns

- The final linear-prefill profiler is only logical S5 after two S128 postprocessing failures. S128 warmed latency/PCC is valid, but S5 is not a like-for-like profiler replacement for the requested warmed S128 before/after regime. Once genuine tables are regenerated, obtain a representative S128 linear profile or record this as an explicit hard-check gap with raw, sanity-checked fallback analysis.
- The guarded `noc.async_atomic_barrier()` changes a shared matmul dataflow kernel. The focused fused and ordinary controls are useful, but the report does not cite broader non-hardware unit/build coverage for the shared-kernel change.
- `optimized_decoder.official_weight_decode_pcc` in `doc/context_contract.json` remains different from the multichip official-weight PCC values reported by this stage. This may be legitimate section scoping, but the multichip section should record its own final PCC artifact/values to prevent readers from treating the older optimized-decoder numbers as the current TP4 result.

## Hard-Check Gaps

- No stage-local check/gate output was found that mechanically verifies artifact existence, candidate/default consistency, context-contract schema, or fallback/stress requirements.
- The worktree is intentionally live and dirty for review. A stage-owned local checkpoint commit cannot be assessed until required work is closed; unrelated untracked roots must not be included.

## Anomaly Ledger

- Observed anomaly: Final full-B32 preallocated default failed to reproduce its candidate win.
  Evidence: 0.719232 ms candidate versus 0.722339 ms final default and 0.722288 ms baseline.
  Affected path: Selected TP4 full-attention B32 decode default and context allocation.
  Control or comparison: Explicit `multichip_baseline` current-environment run.
  Likely subsystem: Sub-percent benchmark variability and/or final default/run reproducibility.
  Investigation performed: Cross-checked raw candidate, baseline, final JSON, README, code default selection, and context contract.
  Resolution: more-work-needed.

- Observed anomaly: S128 linear-prefill Tracy postprocessing dropped an operation twice.
  Evidence: `linear_prefill_s128_final/profile.log` and retry artifacts; only S5 produced CSV/table outputs.
  Affected path: Linear-prefill profiler evidence, not the separate warmed S128 correctness/timing run.
  Control or comparison: S128 timing/PCC JSON passes; S5 profile renders.
  Likely subsystem: Tracy/postprocessing capture completeness.
  Investigation performed: Checked retained profile directory contents and the work-log classification.
  Resolution: controlled for numerical execution, but profiler hard-check remains incomplete.

- Observed anomaly: Fused matmul-to-reduce-scatter BRISC pending atomics.
  Evidence: `triage/AUTOTRIAGE.md`, focused Watcher logs, and the guarded shared-kernel diff.
  Affected path: Fused TP4 matmul/reduce-scatter probe.
  Control or comparison: Separate matmul plus async reduce-scatter control.
  Likely subsystem: Fused `OpSignaler` non-posted atomic drain.
  Investigation performed: AutoTriage/AutoFix, 20-iteration fused and ordinary focused controls, geometry matrix.
  Resolution: fixed for the focused contract; broader shared-kernel regression evidence remains a concern.

- Observed anomaly: Fused all-gather-to-matmul receiver ledger assertion after adapted TP4 retries.
  Evidence: `triage/ALL_GATHER_MATMUL_AUTOTRIAGE.md` and `all_gather_matmul_w4_fused_fixed*` logs.
  Affected path: Rejected fused fractured-residual consumer family.
  Control or comparison: Separate collective/matmul paths pass.
  Likely subsystem: Fused receiver transfer/K-block contract.
  Investigation performed: API/layout/semaphore adaptations and AutoFix retries; speculative source fix reverted.
  Resolution: controlled exact op/runtime blocker for this candidate.

## Scope Inspected

- Goal/skill paths: stage contract supplied by the orchestrator; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/optimize/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: `doc/optimized_multichip_decoder/{README.md,work_log.md,artifacts,logs,triage}`; `doc/context_contract.json`; relevant inherited `doc/multichip_decoder/artifacts/tracy` provenance.
- Code paths: `tt/multichip_decoder.py`; modified/new multichip tests and probes; `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`.
- Commands run: read-only `sed`, `find`, `rg`, `jq`, `git status`, `git diff`, and artifact inventory/inspection commands. No device, server, profiler, or hardware test was run.

## Residual Risk

- Sub-percent latency deltas are not robust enough in the retained evidence to justify a persistent-memory/capability trade without repeated matched statistics.
- The missing real operation tables may conceal dtype/fidelity mismatches, slow dominant rows, or untried profiler advice.
- Capacity evidence is allocator-probe based rather than construction of the full 64-layer stack; it is useful hard-bracket evidence only if every modeled persistent allocation matches actual full-stack ownership and lifetime.
