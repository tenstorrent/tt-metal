# Anomaly ledger

## BF16 KV L1 admission

Observed anomaly: SDPA static CB overlaps live L1 by2048 bytes at prefill736.
Evidence: smoke_kv_bf16.log; AUTODEBUG.md; AUTOFIX.md.
Affected path: BF16 full-attention cache prefill only.
Control or comparison: baseline BFP8 cache; repaired BF16 Q64/K128.
Likely subsystem: dtype-dependent SDPA tile storage.
Investigation performed: independent factory/allocator derivation, reduced tracked129/1023/1025 checks, full BF16 candidate accuracy and performance.
Resolution: fixed. BF16 is rejected for target-pool capacity, not the repaired error.

## Long-prefix harness input conversion

Observed anomaly: torch could not infer dtype from tokenizer Encoding.
Evidence: baseline_bfp4_lofi.log; baseline_bfp4_lofi_final.log.
Affected path: new evidence harness after completed accuracy/performance checks.
Control or comparison: encode the already-rendered chat string with add_special_tokens=False, matching the readiness prompt convention.
Likely subsystem: tokenizer return-type boundary.
Investigation performed: corrected token-ID extraction and reran baseline/control candidates.
Resolution: fixed; no model change.

## Markdown fence in requested JSON reply

Observed anomaly: the correct deployment-label object is wrapped in a Markdown fence.
Evidence: */long_prefix.json, including refreshed baseline and BFP8 decoder controls.
Affected path: synthetic32,083-token multi-turn retention prompt.
Control or comparison: identical raw completion across baseline, selected and higher-precision controls.
Likely subsystem: prompt-following behavior, not introduced by selected precision.
Investigation performed: compared actual completions and label correctness across policies.
Resolution: controlled. This check is not strict tool-call/API conformance.

## Fixed qualitative cap

Observed anomaly: long explanations/story/code stop mid-answer at128 tokens.
Evidence: selected_default/qualitative_tt.json and qualitative_hf.json.
Affected path: shared fixed-cap qualitative suite.
Control or comparison: HF uses the same prompts and cap and is also truncated.
Likely subsystem: evidence generation limit.
Investigation performed: inspected every TT completion and corresponding HF control.
Resolution: controlled; no claim of complete executable code or complete long-form answers.

## Generic trace allocation warning

Observed anomaly: clean performance logs emit the generic post-capture allocation warning.
Evidence: selected_default.log; selected_state_b1/state.json.
Affected path: normal prepared generator execution with diagnostics disabled for timing.
Control or comparison: full-model repeated greedy/sampled/buffered replay with TT_METAL_TRACE_ALLOC_TRACKING=1 and TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0.
Likely subsystem: generic allocator warning for transient allocations after capture.
Investigation performed: tracked repeated replay passes; stable trace IDs and zero recaptures. Existing exact-tensor lifetime handling is unchanged. Batch32 tracked check also passed (selected_state_b32/state.json).
Resolution: controlled by allocation tracking, not suppressed to obtain a passing timing result.

## Teacher-forcing timing variation

Observed anomaly: selected default median37.795t/s/u versus search38.035t/s/u.
Evidence: selected_acceptance_snapshot/readiness_accuracy.json; head_bfp4/readiness_accuracy.json; both performance_b1.json files.
Affected path: host-driven traced teacher-forcing harness.
Control or comparison: first acceptance repeats37.722–38.089; token-out34.8265 versus34.8293.
Likely subsystem: host-boundary measurement variation.
Investigation performed: retained all repeated measurements, verified actual policy, and used the reproduced default value in final tables/plots. Both regimes still rank the selected policy first.
Resolution: controlled; final headline uses the reproduced default token-out result.

Batch queue interruption: selected_state_b1, selected_b32, and selected_state_b32 completed. The parent session then exited143 during selected_b16 weight load, before runtime policy or memory results. No model exception or device stall was logged; termination source is unknown. No surviving benchmark processes or /proc device owners were found. Preserve selected_b16_interrupted.log; rerun unmeasured batch16/8 individually after bounded recovery. This is infrastructure interruption, not a failed dtype gate.

Recovery completed: one bounded reset exited0; all4 chips listed; Ring mesh1x4 opened and closed successfully. No process killed or locks cleared; no second reset needed. Exact recovery summary is recovery.json.

Reviewer log audit: kv_bf16.log and decoder_bfp4_hifi2.log report AICLK1343MHz versus1350 requested. Preserve actual timings without normalization. DecoderHiFi2 is over12% slower; BF16KV is memory-ineligible (even a simplistic linear1350/1343 scaling would not exceed selected throughput). Generic board-discovery, semaphore and fabric packet advisories recur in baseline and selected runs. No failure is inferred from those messages. Allocation safety is established by separate tracking-enabled full B1/B32 controls.

Final-source default refresh completed exit0:93% top1,100% top5/top100; tracedTF median38.071377 (range38.043424–38.097308); autoregressive4096/128/B1 34.829999t/s/u, TTFT656.177ms. The runtime source hashes match current files. Final plots/report use these results; original acceptance timings remain historical only.
