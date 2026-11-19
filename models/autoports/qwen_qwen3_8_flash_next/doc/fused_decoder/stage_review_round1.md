# Stage Review

Verdict: clean-pass

Independent round 1 review of stage 2, fused-decoder, Qwen/Qwen3.8-Flash-Next. Reviewed the live `main` worktree against base `70dc6f5dd24298741e2f108a281160677e986155` and the explicit owner wrap-up contract. Runtime SHA256: `d638f10da2929e6898f63bac36151590c64eb5feb1ad184c3942053f59ae1b5b`.

## Required Work

- None. The delivered fused default and completed evidence satisfy this stage's current contract. Recording this review and making the authorized local stage checkpoint remain the normal closure steps; no push is authorized.

## Other Concerns

- PLE34 row5 and QSA34/66 rows30/44 remain literal-reference row sensitivities, not exact equivalence to literal HF. Independent CPU recomputation from the saved tensors reproduced prefill whole-output PCCs of 0.9982004687, 0.9997948503 and 0.9998067780, respectively; minimum row PCCs were 0.9944641349, 0.9936794411 and 0.9931326428. All outputs were finite. Each saved selected-candidate output, including decode steps, was exactly equal to the previous fused candidate. These satisfy the owner whole-output 0.995 and diagnostic-row 0.99 policy. `ple34_packed_backend_v1.json` and `qsa_tail_backend_cpu_v1.json` reproduce the affected-row sensitivity with explicitly changed HF lookup/accumulation controls; they do not establish that unchanged literal HF fails itself. The three current limitations are recorded with `original_model: "same"`, exact control scope, and evidence paths in `../eval_readiness/limitations.json`.
- This isolated decoder returns hidden streams. Hidden-coordinate argmax is correctly distinguished from vocabulary-token top-1, which cannot be established in this stage. Text quality and generation gates belong to later stages.
- The last writer improvement is small: final clean decode is 5.018603/6.341421/9.548145 ms for GDN/PLE/QSA. The same-run explicit candidate and constructor default agree within timing noise (QSA differs by about 0.0008 ms). These final reproduced values, rather than the slightly faster historical QSA writer sample, are used in `final_performance.json`. PLE's functional baseline is a separately serialized same-harness run at the same BFP8 expert policy because retaining two fully resident models is infeasible; the report discloses this.
- Official sparse matmul advice remains open: current rows use HiFi4, a small input block and small output subblocks, with low bandwidth utilization for down projection. Additional geometry/fidelity/sharding work and further residual-read fusion are explicitly listed under `Not tried (stop rule)`. The owner wall cap makes these handoff items, not stage blockers. No whole-decoder 70% roofline claim is supported or made.
- Context evidence retains its declared scope. Current 32K GDN/PLE runs advance reduced attention/state prefixes and execute complete public final-token decode; QSA's 32K smoke validates attention outputs and replay. This does not prove complete full-MoE output for every 262K prefix token. The inherited prepared PLE 4768/4767 profile is preserved, not generalized to every allocator history or simultaneous maximum batch/context.

## Hard-Check Gaps

- No blocking gap found. The historical strict per-row probes remain failed; acceptance under the new owner criterion is separately represented by `wrap_accuracy.json`, not by rewriting those records.
- The final queue's plan hash, all 21 successful provenance exit codes, recorded runtime hashes and log hashes match their on-disk artifacts. Every final correctness/context invocation enables `TT_METAL_TRACE_ALLOC_TRACKING=1` and retains program-cache allocations (`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`). Timing and profiler collection use separate untracked runs. The inherited tracker positive control is available at `../functional_decoder/trace_alloc_tracking/positive_control.json`; this review did not require a duplicate evidence format or another device run.
- The final synthetic log contains 14 actual passed cases, covering GDN and indexed QSA at logical lengths 1/2/31/33/34/65/66. The test fixture asserts `FusedDecoder`, fused residual writers, fused MoE, exact QKV where applicable, compact threshold 2, packed activation and the selected QSA components.

## Anomaly Ledger

- Observed anomaly: Expanded PLE/QSA fixtures contain rows below the former 0.995 row threshold.
  Evidence: `wrap_accuracy.json`, `ple34_control_comparison_v1.json`, `qsa_tail_control_comparison_v1.json`, saved `.pt` outputs and the two HF backend-control JSONs.
  Affected path: PLE34 row5; QSA34 row30 and QSA66 row44.
  Control or comparison: Functional baseline shows the same rows; old/new fused outputs are byte-identical; packed-lookup HF and attention-GR-down FP64 HF controls reproduce the respective sensitivity.
  Likely subsystem: Existing packed PLE lookup and BF16 projection/router sensitivity.
  Investigation performed: Independent saved-tensor PCC/equality recomputation; inspection of actual control metrics and limitations records.
  Resolution: controlled under the explicit owner acceptance policy. No unchanged-literal-HF failure is claimed.

- Observed anomaly: Earlier native norm/convolution and changed QKV reduction candidates lost correctness; grouped/packed allocations reduced PLE capacity.
  Evidence: `anomaly_ledger.md`, `pattern_assessment.json`, the corresponding AutoDebug/AutoFix records, exact component evidence, and `integrated_writer_remaining_plan_v2_progress.json`.
  Affected path: GR/router arithmetic, GDN convolution/QKV and fully resident PLE.
  Control or comparison: Original arithmetic and functional capacity controls; selected exact convolution/QKV; direct checkpoint grouping and early activation destination allocation.
  Likely subsystem: Rounding/reduction order and allocator fragmentation.
  Investigation performed: Inspected selected implementation, candidate-to-final source diff, current real-weight/batch/capacity/Watcher artifacts and repeated-state checks.
  Resolution: fixed in the selected path; rejected historical candidates are not presented as accepted evidence. PLE4768, 4767 and 4738/2-tail artifacts pass.

- Observed anomaly: A historical capture issued a host upload from `zeros_like`, and a profiler harness assumed the original projection representation.
  Evidence: `autofix_moe_cleanup/`, `autofix_trace_protocol/`, final replay and profiler logs.
  Affected path: Sparse prefill trace allocation and profile collection.
  Control or comparison: Explicit device empty/fill path; representation-aware profile harness; final tracked correctness and separate final reports.
  Likely subsystem: Allocation API and test harness assumptions.
  Investigation performed: Read current sparse dispatch and trace harnesses; checked final provenance and logs.
  Resolution: fixed. No unsafe-allocation/capture failure remains in the inspected final runs.

- Observed anomaly: Segmented prefill profiler totals differ from complete clean prefill latency; hardware logs report one visible chip on a P300 board and unknown motherboard mapping.
  Evidence: Six final official CSVs, `wrap_profile_summary.json`, final timing logs and `anomaly_ledger.md`.
  Affected path: Measurement interpretation and host device discovery.
  Control or comparison: Independently summed CSV rows; separate clean timings; configured single-chip mesh and successful open/close/Watcher runs.
  Likely subsystem: Profiler draining and host topology metadata.
  Investigation performed: Confirmed prefill gaps are explicitly unmeasured, decode is uninterrupted replay, and final logs contain no corresponding unresolved device-health failure.
  Resolution: controlled; neither a complete prefill latency nor multi-chip capability is inferred from these artifacts.

## Scope Inspected

- Goal/skill paths: `review_contract.md`; `/home/ttuser/qwen38-flash/multigoal_logs/02-02-fused-decoder.prompt.txt`; installed 0.1.24 `stage-review`, `graph-fusing`, `optimize`, `tt-device-usage`, `tt-enable-tracing` and `model-bringup` skills; repository `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifact paths: README, manifest, work log, pattern assessment, runtime boundary audit, anomaly ledger, final gate/performance/profile/accuracy summaries, context contract and limitations. Final queue/provenance/logs, final ranking rows, integrated writer real-weight/Watcher/batch/capacity records, current 32K context records, HF backend controls, prior PCC comparison and six official report directories were checked. Each official CSV independently reproduces its reported kernel sum: prefill 78.003737/92.397247/183.022748 ms; decode 4.966203/6.290691/9.404568 ms.
- Code paths: `tt/fused_decoder.py`, inherited functional/QSA dispatch where relevant, `tests/test_fused_decoder.py`, shared synthetic test, `run_fused_probe.py`, `run_fused_plan.py`, candidate/replay/linear/QSA/batch/residual/accuracy harnesses. Only `tt/fused_decoder.py` changes runtime code; no tracked changes outside this autoport were found. Public state ownership, paged updates and outer chunking remain inherited; new physical-count-2 routing and custom writers have explicit relevant tail coverage.
- Source binding: Compared `sources/integrated_writer_layer0_hf_v1/fused_decoder.py` to the delivered source. Differences are the two adopted defaults and standard-tile/physical-padding guards, with no arithmetic change. Earlier explicitly selected `best_writer_tail2` correctness/capacity/Watcher evidence therefore remains applicable, alongside final-default checks.
- Trace lifetime: Inspected exact-output acknowledgments, synchronous consumption before subsequent replay, state restoration, changed inputs, trace release before variant changes and the interleaved-prefill checks. No entire-forward/capture allocation exemption is used by the inspected acceptance harnesses.
- Commands run: Read-only `rg`, `sed`, `cat`, `tail`, `git diff`/status queries; small Python JSON/hash/CSV analyses; CPU-only Torch loading of saved PLE/QSA tensors and independent correlations/equality checks. No TTNN import, device access, reset, server, implementation edit or hardware test was performed. The only written artifact is this report.
- Push state: The run contract and manifest say no push; local stage checkpoint was pending review at inspection. This review does not authorize pushing.

## Residual Risk

- Inspection and existing device evidence support the current single-chip decoder scope; they do not establish untested hardware, full-model generation or serving behavior.
- Packed lookup and discrete routing remain numerically sensitive. Preserve the exact control scopes and literal metrics through later precision and full-model stages.
- The newest source was not rerun at 262144, as explicitly directed by the owner and the unchanged KV dtype/layout policy. Current 32K smokes, retained maximum-context evidence and final tail/state coverage support stage closure within that policy.
