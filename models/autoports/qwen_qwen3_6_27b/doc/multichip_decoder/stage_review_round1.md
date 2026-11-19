# Stage Review

Verdict: more-work-needed

Round 1 of 3. Independent, read-only review of stage 4, multichip-decoder, Qwen/Qwen3.6-27B. Reviewed the frozen worktree on `main`, HEAD `0915d93157bd320c7875533fba09bfbffeb2ebeb`. Delivered decoder SHA256 is `082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`; frozen baseline SHA256 is `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.

The accepted numerical, trace, stress, capacity, resource and performance evidence is internally consistent. One focused coverage gap remains at the final shape-dependent row-projection dispatch boundaries.

## Required Work

- P2: Validate neighboring logical lengths at the final row-projection dispatch boundaries.
  Evidence: `tt/multichip_decoder.py:354` selects native MMRS only for `512 < B*S <= 8192`; both the attention output and MLP down projection use this rule. `tests/collect_multichip_evidence.py:36` accepts correctness shapes `(2,1), (1,31), (3,513), (31,33), (1,1025)`, performance shapes `(1,512), (32,33), (32,513)`, and watcher shapes `(32,33), (3,513), (31,33), (17,33), (31,31)`. Inspection of the final-source JSON artifacts found no boundary pair at B2/S256 versus S257, or B32/S256 versus S257. Public S513 is internally split by `OptimizedDecoder.prefill_forward` into S512 and S1: at B32 it therefore exercises M16384 and M32, rather than the new M8192 crossover. The final accepted cases establish the separate and fused families and their native 32/64/96-row tails, but do not establish adjacent-length dispatch at these boundaries. `host_validation.json` checks recurrence ownership, page remapping and norm ranks; it has no row-dispatch boundary check.
  Why this matters: the model-bringup skill's “Validate changed paths” section and the stage-review skill explicitly require valid lengths immediately around changed dispatch/allocation boundaries and coverage of sibling consumers. The final crossover was added after the earlier safe-path evidence. Immediately above M8192, a valid B32/S257 prefill takes the separate path with an internal 512-row projection remainder in the same call; the current B32/S513 evidence has a different internal chunk/remainder arrangement. No failure is asserted here; this is missing required boundary evidence.
  Required next step: run the existing baseline-versus-TP4 comparison on final source for both layer kinds at B2/S256, B2/S257, B32/S256 and B32/S257, preserving the existing output/contribution/per-user/cache-state PCC bar. Run these focused TP4 cases with watcher and the existing trace-allocation checks so the native and separate branches share the established acceptance checks. Record the actual flattened row counts and selected projection path, and add these cases to the host aggregate. Existing performance, maximum-context, sustained-stress and 64-owner results can be retained when implementation source is unchanged. A focused follow-up review of these additions is sufficient.

## Other Concerns

- Several current-contract paragraphs retain historical wording. `kernel_contracts.md` says prefill fusion occurs when `B*S>32`, then later supplies the correct `512<M<=8192` rule; its earlier stack description also says GDN/full-attention/GDN while delivery uses checkpoint layers 0..4. Reconcile the current-contract paragraphs with the final implementation.
- `work_log.md:559` attributes changed-input trace output/state equality to the five-consecutive-layer test. `tests/run_multichip_stack.py:128` captures one original input, replays once and compares final output; it does not compare changed inputs or state tensors. Those stronger checks are provided by the separate single-layer and 64-owner resource tests. Correct the attribution. The combined evidence supports the stage contract, so no new stack experiment is requested for this wording issue.
- The remaining B32/S33 prefill reshape costs and the small replicated-layout B1 decode advantage are measured and disclosed. A persistent flat-prefill representation or phase-specific residual interface is a useful next optimization-stage candidate. Neither is an additional required experiment in this review.

## Hard-Check Gaps

- The aggregate checks named shapes and source hashes, but does not derive coverage from the final `512/8192` dispatch predicates. Add the focused boundary cases above to prevent a passing aggregate from overlooking this requirement.
- The aggregate trusts the runners' `pass` flags for initial numerical comparisons. I independently recomputed the minima from every accepted correctness/performance/watcher result: 0.997892283, 0.996754199 and 0.996481588 respectively, all above 0.995. This review found no numerical contradiction; direct scalar checks in the aggregate would be optional hardening.
- Archived evidence requires reconstruction before scripts that read original `.log` and large `.json` paths can run from a fresh checkout. The reconstruction contract is documented. All 649 archive entries passed independent stored/original length and SHA256 verification, honoring each entry's compression mode.

## Anomaly Ledger

- Observed anomaly: cold B32 nonfinite output under the initial semaphore grid.
  Evidence: `anomaly_ledger.md`, `AUTOFIX.md`, `debug_b32/`, cold-watcher controls, delivered watcher/resource artifacts.
  Affected path: CCL worker semaphore ownership.
  Control or comparison: full 11x10 semaphore allocation and fresh-cache watcher; final source retains that allocation.
  Likely subsystem: CCL global semaphore coverage.
  Investigation performed: inspected the model-local CCL subclass, repair history, accepted run provenance and logs.
  Resolution: fixed; no new accepted-run failure found.

- Observed anomaly: unpartitioned native MMRS failed watcher despite passing numerical comparisons.
  Evidence: `debug_narrow_watcher/AUTODEBUG.md`, `AUTOFIX.md`, safe-path controls and delivered watcher/resource reports.
  Affected path: row-parallel prefill projections.
  Control or comparison: separate projection/reduce-scatter and native pieces of at most 128 rows; actual B32 prefill with 64 resource owners.
  Likely subsystem: dedicated NoC counter ownership in the native tall-grid plan.
  Investigation performed: inspected final dispatch, shape restoration, resource ownership and watchdog evidence; verified unsafe historical profiles are excluded from accepted performance.
  Resolution: fixed for the measured selected target/path. Required Work above completes neighboring-length evidence for its final dispatch rule.

- Observed anomaly: GDN synthetic B31/S31 decode user PCC 0.994696745, and an artificial repeated-layer synthetic stack below 0.995.
  Evidence: `pcc31_control_summary.json`, `pcc31_default_synthetic.json`, matched separate/HiFi4 controls, real-input controls, `delivery_stack_*`.
  Affected path: LoFi attention precision on those artificial activation distributions.
  Control or comparison: separate MMRS leaves the failing scores unchanged; HiFi4 passes; recorded target-model activations and consecutive real checkpoint layers pass unchanged thresholds.
  Likely subsystem: accumulated attention projection rounding; no single faulty primitive is established.
  Investigation performed: reviewed controls, fixture provenance, runner thresholds and the optimization skill's real-activation precision policy.
  Resolution: controlled. The synthetic counterexamples remain explicit; full-model accuracy must be established in the later full-model stages.

- Observed anomaly: fused Linear AG candidate stalled on repeated queued launches.
  Evidence: `AUTOFIX.md`, `triage_agminimal/`, `triage_agminimal_async/`, fenced candidate comparisons.
  Affected path: rejected fused-AG candidate.
  Control or comparison: persistent buffers plus admission fence passed queued replay, then lost against the compatible best separate path.
  Likely subsystem: inter-launch admission/reset ordering.
  Investigation performed: inspected recorded adaptation/rejection evidence and confirmed final forwards use ordinary AG.
  Resolution: controlled; candidate is excluded from production.

- Observed anomaly: one profiler collection aborted on an orphan timestamp marker; the subsequent profile wrapper exited 127 after six completed jobs.
  Evidence: `debug_profiler_marker/`, `delivery_profile_retry_*`, `delivery_profiles_retry.console.log`, final accounting.
  Affected path: profiler collection and shell orchestration.
  Control or comparison: identical-source fresh collections with normal marker pairing; separate profiler-off latency and watcher runs.
  Likely subsystem: possible clock rollover during legacy sampling, plus the documented live-wrapper-edit error. The clock attribution remains an inference.
  Investigation performed: independently checked six raw-ops hashes, four ranks, runtime precision fields, completed artifacts and accepted logs. No dropped-marker/fatal diagnostic appeared in the accepted logs.
  Resolution: controlled collection failure; accepted numbers come from completed retry artifacts.

## Scope Inspected

- Goal/skill paths: `/home/ttuser/qwen36-27b/multigoal_logs/04-04-multichip-decoder.prompt.txt`; installed `stage-review`, `model-bringup`, `multichip`, `optimize`, `tt-enable-tracing` and `tt-device-usage` skills under `/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/`.
- Artifact paths: `doc/multichip_decoder/{README.md,gate_summary.json,work_log.md,mesh_plan.md,kernel_contracts.md,capacity_plan.json,optimization_checklist.md,performance.md,performance.csv,performance.json,profile_accounting.md,anomaly_ledger.md,AUTOFIX.md,evidence_archive.json,host_validation.json,runtime_audit.json,pcc31_control_summary.json,precommit_delivery_final.log}`; `doc/context_contract.json`; all accepted gate JSON paths; six accepted profile CSV/raw-ops/metrics/accounting sets; 40 accepted plain-text run logs; relevant historical control and diagnosis records.
- Code paths: complete `tt/multichip_decoder.py`; inherited optimized projection, norm, prefill, cache and attention paths; `tests/{run_multichip_decoder.py,run_multichip_context.py,run_multichip_stack.py,check_multichip_stack_resources.py,collect_multichip_evidence.py,validate_multichip_host.py,account_multichip_perf.py,multichip_assertions.py}` and profile renderer; gate case definitions and pytest entrypoint.
- Commands run: read-only `cat`, `sed`, `rg`, `nl`, `wc`, `git status`; small Python standard-library scripts for JSON/PCC/environment/source-hash checks, archive reconstruction in memory, raw CSV rank/dtype/fidelity verification, matched performance arithmetic and final-source shape coverage. No TTNN import, device operation, test, server or reset was performed. Only this report was written.
- Re-derived results: all 46 accepted gate artifact entries exist; string source hashes match delivered code; the 6 latency pairs use matched reference metadata and 100 queued replays with diagnostics disabled; all published speedups/efficiencies recompute from those times. Runtime matmul rows show BFP8 weights and the claimed LoFi/HiFi2 policy. Four full-context runs retain 262144 with 20 GiB/device additional reservation. Both 64-owner tests report one capture, zero recapture/cache growth, tracker positive control and exact changed-input output/state hashes.
- Push state: the supplied run plan delegates pushes to the supervisor; the stage owner never pushes. Local stage checkpoint commit is pending a passing review, as expected.

## Residual Risk

- This is decoder-layer evidence. It establishes neither 64 distinct real-layer numerical accuracy nor generated-text quality, and makes no full-model/vLLM claim.
- Capacity combines explicit all-layer accounting, additional device reservation and separate resource tests. Future full-model fragmentation and terminal logits allocations still require validation; the documented bounded-token logits contract is necessary.
- Native fused-MMRS routing depends on live L1 and the pinned target/runtime. The measured 1x4 mesh, serialized queue and resource envelope must remain the integration contract.
- The report requests only the missing neighboring-length evidence. No implementation correctness defect was established in the inspected accepted cases.
