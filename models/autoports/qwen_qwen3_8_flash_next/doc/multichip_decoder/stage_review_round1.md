# Stage Review

Verdict: more-work-needed

Fresh independent review, round 1 of 3, Stage 4 multichip decoder for Qwen/Qwen3.8-Flash-Next. The remaining required work is a limited performance-evidence refresh. I found no demonstrated wrong output in the selected implementation or its final correctness artifacts.

## Required Work

- P2: Report latency from separate runs with allocation tracking and diagnostics disabled.
  Evidence: `performance_summary.json` derives every baseline/mesh latency from `baseline_l{0,1,3}_contract.json` and `final_mesh_l{0,1,3}.json`. All six reports have allocation tracking and runtime auditing enabled; their provenance sets `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`. The timer encloses both the runtime-audit context and `execute_trace` in `tests/probe_multichip_decoder.py:583`, with the same pattern for B32. The tracing skill explicitly says, at line 207, “Collect performance numbers separately with tracking and diagnostics disabled, recording the settings for each run.” The owner confirmed that no separate clean final timing artifacts exist. The current implementation of the tracker only invokes garbage collection when it finds potentially unsafe buffers; I am not claiming that every sample incurred a GC pause or that the recorded arithmetic is wrong.
  Why this matters: The reported speedup and efficiency are currently measurements of the instrumented validation path. This leaves the required performance measurement regime unfulfilled, particularly for the small QSA B1 improvement. The existing checked runs remain valid correctness evidence.
  Required next step: Run only three paired one-chip/four-chip timing measurements on the unchanged selected decoder, with the same logical 33 prefill, capacity128, prepared inputs, B1 and diverse B32 workload. Use the existing `--clean-timing` option, disable allocation tracking/diagnostics, watcher and profiling, and preserve the exact environment/provenance. Use new output and reference names so the full-contract fixtures are not overwritten. Update the latency, speedup and efficiency fields/tables from these runs while keeping the tracked replay records as correctness evidence. No repeat of tails, long context, watcher, profiler or full-contract hardware gates is required for this evidence-only repair. A fresh diff-only round 2 can check the new timing records and derived summaries.

## Other Concerns

- The tested path is a useful Stage 4 stack baseline, with genuine EP4 expert ownership, local GDN heads, local/duplicated QSA KV heads, and identical stream-sharded layer input/output layouts. It is not established as globally optimal. The documented QSA indexer partitioning, packed-only indexed experts, fused collective/projection families, injection K parallelism, PLE lookup parallelism, and precision-locked geometry experiments remain performance work for the next stage. I do not turn those unmeasured candidates into additional Stage 4 correctness gates.
- The planned full-stack memory budget is conditional on a shared `MeshWorkspace`, shared immutable tables, bounded maximum-context IO, and the stated future embedding/head/final-mixer placement. The reservation probes correctly disclose that they do not instantiate 48 layers. Preserve these constraints when building the full model; B32 at capacity128 is not B32 at advertised maximum context.

## Hard-Check Gaps

- The harness computes `batch.whole_output_pcc`, but its final `passed` calculation at `tests/probe_multichip_decoder.py:809` still gates `min(batch.pcc) >= 0.995`. That is the historical, stricter per-request check, whereas the user explicitly made whole-output PCC the acceptance gate and per-row PCC diagnostic. All three final whole-batch values independently pass the actual user gate (minimum 0.9988927841), so this does not invalidate their acceptance evidence. Align the automated gate with the declared contract when next changing the harness; retain per-request values as diagnostics. The rejected native-AR experiment must not be described as having failed the user's aggregate gate, since its historical artifact lacks that aggregate.
- The Stage 4 manifest does not index the inherited allocation-tracker positive control. I found and inspected `../functional_decoder/trace_alloc_tracking/positive_control.json` and its adjacent log: a deliberately live allocation is rejected with the unsafe-buffer error, and replay succeeds after deallocation. `../optimized_decoder/acceptance_manifest.json` already links this control. The tracker/trace/allocator source has no diff from the optimized baseline, and the final runs independently assert that tracking is active and program-cache allocations are included. Add the inherited control to this stage's evidence index; I do not require a new control run solely to obtain another evidence format.
- Repeated trace checks cover each representative B1 and B32 signature separately, with only one live trace at a time. B1 refreshes three different prepared residual inputs and PLE token words while restoring a common prefix; B32 restores a fixed diverse batch. These establish the stated decoder replay contract, not future generator-wide coexistence, changing scheduler/page mappings under a live trace, or advancing token feedback. Those broader contracts belong to the later full-model/serving implementation.

## Anomaly Ledger

- Observed anomaly: Native TILE combine corrupted a selected expert slot despite finite raw expert outputs.
  Evidence: `mesh_l1_combine.json`, `mesh_l1_rowmajor_combine.json`, `AUTOFIX.md`, and final all-kind short/long contracts.
  Affected path: Multirow routed MoE combine.
  Control or comparison: Raw expert rows compared directly with selected combined slots; ROW_MAJOR input restores slot equality.
  Likely subsystem: Native TILE combine input handling; exact C++ root cause is not established.
  Investigation performed: Component, expert-output and slot-level isolation, followed by bounded ROW_MAJOR conversion and final-source reruns.
  Resolution: fixed by the selected device-side conversion; no claim that the upstream TILE implementation was repaired.

- Observed anomaly: The native-AR experiment exhausted L1_SMALL, and the persistent-resource replacement changed two B32 request PCCs to approximately 0.9907/0.9910.
  Evidence: `mesh_l0_native_ar.log`, `mesh_l0_persistent_ar.json`, `AUTOFIX.md`.
  Affected path: Nonselected collective candidate.
  Control or comparison: Ordered gather/FP32 sum remains selected and passes the whole-output gate.
  Likely subsystem: Cached semaphore growth in the initial implementation; different reduction arithmetic in the subsequent candidate.
  Investigation performed: Persistent semaphore experiment completed tails and repeated traces; numerical diagnostics and latency were recorded.
  Resolution: controlled by excluding the candidate. Its per-request diagnostics do not prove failure of the user's whole-B32 PCC gate.

- Observed anomaly: Earlier profiler captures lost markers; one intermediate QSA capture had an unmatched NCRISC endpoint.
  Evidence: `profiler_audit.md`, excluded captures in `evidence_manifest.json`, and final profile provenance/logs.
  Affected path: Historical performance attribution.
  Control or comparison: Fresh final captures use ordinary nesting validation and strict host/device joins; segmented baseline captures also complete.
  Likely subsystem: Demonstrated buffer exhaustion for the first failures; timestamp rollover remains a hypothesis for the unmatched endpoint.
  Investigation performed: Raw-ID/source audits, bounded profiler drains, and fresh final captures without validation bypass.
  Resolution: controlled. Failed captures are excluded; final CSV counts and timing sums match the report. Segmented totals are correctly labeled operator attribution, not critical-path latency.

- Observed anomaly: Advertised BFP8 experts plus PLE cannot fit the four-chip per-device budget.
  Evidence: `memory_capacity_plan.json`, `context_contract.json`, final short and 32769-token reservation runs.
  Affected path: Full-stack capacity planning.
  Control or comparison: BFP4 EP4 expert payload is 16,986,931,200 bytes/device; BFP8 experts plus PLE alone exceed the observed 34,178,731,008-byte budget.
  Likely subsystem: Physical DRAM capacity.
  Investigation performed: Explicit tensor/bank ledger and real-weight whole-output comparisons; final layer1/layer3 execution under approximately 31,203,443,520 bytes/device reservation.
  Resolution: controlled. The selected BFP4 policy passes this stage's comparisons and retains 262144 advertised positions; future full-stack runtime envelopes remain estimates.

- Observed anomaly: Published speedups come from tracked/audited timing runs.
  Evidence: Six paired timing source reports/provenances and the timing scope in the harness.
  Affected path: Final latency/speedup/efficiency claims.
  Control or comparison: Separate tracking-disabled measurements are absent.
  Likely subsystem: Evidence/benchmark configuration.
  Investigation performed: Traced timer/source inspection and independent arithmetic reconstruction; owner confirmed the missing clean measurements.
  Resolution: more-work-needed as specified above.

## Scope Inspected

- Goal/skill paths: `/home/ttuser/qwen38-flash/multigoal_logs/04-04-multichip-decoder.prompt.txt`; installed v0.1.24 `stage-review`, `multichip`, `optimize`, `tt-enable-tracing`, `tt-device-usage`, and `model-bringup` skills. No model-generation/qualitative gate was imported into this decoder-only stage.
- Workspace: live `main`, HEAD `cdb5e8f161972ca9d2958e81d809c29f686b817a`; selected implementation SHA256 `af4699739ad869466bbc825909daf868e8a2f01ed64e0af149602fe6128349c1`; optimized baseline commit `fb462731c5e32e50a3d2efadb8df3067d35dbaed`. Stage changes were staged during review, without implementation changes. Unrelated pre-existing fused-decoder artifacts were excluded.
- Artifact paths: Stage README/work log/mesh plan; context and memory ledgers; evidence manifest; paired short and long reports/provenance/logs; watcher reports and raw logs; final/baseline profiler provenance and tables; performance summaries/analysis; AutoFix and profiler audits; indexed-decode audit; inherited tracker positive control.
- Code paths: All of `tt/multichip_decoder.py`, `tests/probe_multichip_decoder.py`, and `tests/probe_multichip_long_context.py`; runtime fallback audit and tracker implementation; applicable inherited state/layout interfaces and baseline provenance.
- Commands run: Read-only `cat`, `sed`, `rg`, `nl`, `git status/diff/branch/rev-parse`; small Python JSON/CSV/hash/log analyses. Independently checked 109 existing manifest file hashes/paths, all three latency-ratio calculations, short-report state/replay/tail fields, six final profiler CSV row counts and time sums, and watcher log error patterns. No devices, servers, reset, target execution, or hardware test was used. The only written file is this report.
- Selected correctness evidence: Every final whole-output prefill/decode comparison passes; minimum 0.9996137023. All 96 tail cases pass; minimum 0.9995444417. All three B32 whole outputs pass; minimum 0.9988927841. All three cross-kind stacks pass; minimum 0.9963018298. Each kind records 16 B1 and 16 B32 replays with exact output/state parity against eager execution. The two 32769-token public-prefill reservation runs pass with minimum output PCC0.9998434904 and0.9991748793, respectively. State mapping, exact QSA positions/page tables and PLE history checks are implemented and recorded.
- Push/commit state: The explicit run contract is local checkpoint commits after passing review and never push. A pending acceptance commit is expected and is not a finding. This review does not infer push state from reflog history.

## Residual Risk

- This is inspection of recorded evidence, not an independent hardware reproduction. Representative-layer comparisons are against the optimized TTNN baseline, not new end-to-end HF accuracy or generated-text evaluations.
- The full 48-layer stack, its actual trace/cache footprint, maximum-context execution and serving concurrency are not implemented or claimed by this stage. The capacity arithmetic and reservation tests are appropriate preparation, with the explicit limitations above.
- After the limited clean timing refresh, the remaining listed concerns are hand-off/indexing/test-gate notes rather than demonstrated selected-path correctness failures. A diff-only follow-up should avoid repeating the completed full-stage review.
