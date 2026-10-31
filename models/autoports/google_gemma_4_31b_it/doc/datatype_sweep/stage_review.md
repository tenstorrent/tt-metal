# Stage Review

Verdict: clean-pass

Independent review of stage 08, datatype-sweep, completed on 2026-10-31 after the owner declared final artifacts ready. Reviewed the live worktree on `mvasiljevic/gemma4-31b-fresh-pipeline-v0811`, base HEAD `c673ec5a40de93afe05c3383a359eb6df0cb71d1`. The final source hashes match `selected_default/source_manifest.json` and all selected batch/state manifests. Model and tokenizer remain pinned to `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`.

## Required Work

- None for stage acceptance. The prescribed local checkpoint and recording its SHA follow this review; no push or vLLM integration is part of this verdict.

## Other Concerns

- The accepted result is **head_bfp4**: BFP4/LoFi decoder projections and head, BFP8 KV and prefill MLP/decoder-collective values, BF16 residual/decode-collective/logit/sampling values. The final default construction records **93% / 100% / 100%** top-1/top-5/top-100 against gates **90% / 98% / 100%**. These are 100 continuation positions from one AIME24 chat-template prompt, not dataset-level accuracy.
- Recomputed all 11 candidate rows from their readiness artifacts, checked the 60-layer runtime weight/kernel summaries against requested policies, and checked replay counters. All candidates meet the accuracy bar. BF16 KV fails target-pool memory capacity; other alternatives lose on measured speed. Material decoder and head groups have both BFP4 LoFi/HiFi2 and BFP8 LoFi/HiFi2 controls. The coherent BFP8 decode-collective/head combination is also measured.
- Final traced teacher-forcing median is **38.07137675 tokens/s/user**, with repeats **38.043424–38.097308** and median TTFT **93.862670 ms**. The separately measured, warmed **4096-input / 128-output / B1 / concurrency1** token-out result is **34.82999901 tokens/s/user**, TTFT **656.177368 ms**. Every raw token-out timing sample records 127 model replays, 128 sampler replays, two reads, and zero captures/token/page refreshes. Later serving comparisons should use this token-out result.
- B8/B16/B32 token-out results are **32.904974 / 29.597013 / 23.485308 tokens/s/user**, with identical workload lengths, streaming equality, and checked counter/median arithmetic. No whole-model device time or roofline value is inferred from these host timings. The telemetry packet carries the exact B1 autoregressive result and explicit missing-device-time reasons.
- The agentic checker has only the documented tied-weight representation advisory. The BF16 row-major lookup and transposed tiled BFP4 head require different layouts; both are inventoried. This is accounted storage, not an unexplained duplicate. The head shrinks from 374,341,632 to 198,180,864 bytes per ASIC. Selected B32 prepared allocations are 25,072,776,704 bytes, leaving **8,166,430,208 bytes** after the retained 768 MiB transient reserve.

## Hard-Check Gaps

- Schema checks alone would not establish policy propagation, valid ranking, or output quality. This review additionally followed `precision.py:33`, `model.py:25`, `multichip_decoder.py:20`, and their matmul/cache/collective consumers, inspected actual runtime summaries, recomputed timing/count arithmetic, and read all six selected/HF qualitative completions. The selected artifact is required by normal construction; the baseline remains an explicit override.
- Maximum-context evidence establishes full-stack execution/capacity at 262143/262144 tokens and decode input position 262143. It is explicitly not a new long-context HF agreement score. The selected head change does not alter cache/page/chunk semantics. Final-source B1/B32 tracked states cover 129- and 1025-token nonaligned requests and greedy/sample/greedy reuse with stable trace IDs, tracking enabled, and program-cache exclusion disabled.
- The earlier context and shared-suite artifacts are preserved byte-for-byte in `selected_acceptance_snapshot/`. Subsequent changes added an inactive BFP8 decode-collective branch and formatting/import normalization; selected BF16 collective operations and BFP8-cache geometry are unchanged. Final-source accuracy, token-out, memory, policy, and repeated-state evidence remove ambiguity for those paths. This does not require rerunning unaffected acceptance cases.
- The transient reserve remains the prior measured reduced-layer peak plus a disclosed conservative analytical bound, not a claimed full-stack peak. The selected change reduces head weights without increasing activation/output/cache shapes or dtypes. Fresh allocator inventories at all four batches reconcile with the updated context and agentic ledgers.
- No additional unsupported gate is imposed. vLLM adapter propagation/registration is correctly unclaimed because integration is explicitly excluded from this stage.

## Anomaly Ledger

- Observed anomaly: BF16 KV prefill failed L1 admission at physical length 736.
  Evidence: `smoke_kv_bf16.log:69`, `AUTODEBUG.md`, `AUTOFIX.md`, `smoke_kv_bf16_repaired/smoke.json`, `kv_bf16/readiness_accuracy.json`.
  Affected path: BF16 full-attention SDPA prefill.
  Control or comparison: BFP8 baseline; BF16 Q64/K128 repair.
  Likely subsystem: dtype-dependent SDPA circular-buffer footprint.
  Investigation performed: exact 2048-byte overlap derivation, all warm buckets, tracked 129/1023/1025 smoke, then full-model accuracy/performance.
  Resolution: fixed. BF16 KV is rejected for target memory, not its repaired admission error; its target KV payload alone exceeds available allocator capacity.

- Observed anomaly: long-prefix harness could not convert a tokenizer Encoding into token IDs.
  Evidence: `baseline_bfp4_lofi.log:142`, `baseline_bfp4_lofi_final.log`, `tests/datatype_long_prefix.py:25` relative to the autoport root.
  Affected path: evidence harness after accuracy/performance measurement.
  Control or comparison: explicit encoding of the already rendered chat string without extra special tokens.
  Likely subsystem: tokenizer return-type boundary.
  Investigation performed: corrected extraction and inspected completed baseline/candidate long-prefix artifacts.
  Resolution: fixed; no model runtime change.

- Observed anomaly: retained-label replies use Markdown fences despite the JSON-only request.
  Evidence: candidate `long_prefix.json` files and `selected_b32/long_prefix.json`.
  Affected path: 32083-token synthetic multi-turn free-running reply.
  Control or comparison: identical completion under baseline, selected, and higher-precision decoder policies; all 32 selected slots retain ORCHID-742.
  Likely subsystem: model prompt-following behavior.
  Investigation performed: compared actual strings, not only boolean pass fields.
  Resolution: controlled. This demonstrates retention, not strict tool-call conformance.

- Observed anomaly: longer shared-suite answers end mid-explanation/code; saved text contains the stop token.
  Evidence: `selected_default/qualitative_tt.json`, `qualitative_hf.json`, `qualitative_prompt_format.json`.
  Affected path: fixed 128-token qualitative suite.
  Control or comparison: same pinned chat prompts/cap in HF; stop-token retention is also present in HF text.
  Likely subsystem: evidence cap and decoding presentation.
  Investigation performed: read all six paired outputs. Haiku, translation, explanations, story, and code introduction remain coherent; no unexplained loops, gibberish, or cross-request leakage is visible.
  Resolution: controlled. Partial outputs are not represented as complete executable code or full answers.

- Observed anomaly: timing runs emit a generic warning about allocations after capture.
  Evidence: `selected_default.log:117`, `selected_state_b1/state.json`, `selected_state_b32/state.json`, `tt/generator.py:160` relative to the autoport root.
  Affected path: prepared generator prefill/replay and sampler state changes.
  Control or comparison: full-stack repeated streaming/buffered requests with `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`.
  Likely subsystem: generic warning for potentially unsafe allocation lifetimes.
  Investigation performed: inspected generator-wide preparation, stable buffers, actual replay ordering, source-matching B1/B32 checks, and stable trace IDs. No allocation exemption was added by this stage.
  Resolution: controlled by successful tracked replay evidence, not by suppressing the warning in correctness tests.

- Observed anomaly: early aggregation overwrote the first raw teacher-forcing repeat's throughput with its median.
  Evidence: `head_bfp4_hifi2.log:120`, corrected candidate readiness JSON, `work_log.md`, `tests/run_datatype_candidate.py:143` relative to the autoport root.
  Affected path: evidence serialization for early head controls.
  Control or comparison: preserved decode counts/elapsed times and original printed rates.
  Likely subsystem: Python list/dictionary aliasing in the runner.
  Investigation performed: owner restored affected rates from preserved counts/elapsed times and used deep copies; reviewer rechecked every raw repeat ratio and all matrix medians.
  Resolution: fixed. Medians/ranking are unchanged; final default measurements were rerun on current source.

- Observed anomaly: search and first acceptance teacher-forcing medians differed slightly.
  Evidence: `head_bfp4/readiness_accuracy.json`, `selected_acceptance_snapshot/readiness_accuracy.json`, final `selected_default/readiness_accuracy.json` and performance JSON.
  Affected path: host-driven traced teacher forcing.
  Control or comparison: repeated measurements and separately stable token-out results.
  Likely subsystem: timing variation around host boundaries.
  Investigation performed: retained historical results; final tables/plots/telemetry use the reproduced final-source result.
  Resolution: controlled. Both measurement regimes select the same policy.

- Observed anomaly: the parent queue exited 143 during B16 weight loading before measurements.
  Evidence: `selected_b16_interrupted.log`, `recovery.json`, completed `selected_b16/`, `selected_b8/`, and `selected_default/` artifacts.
  Affected path: host orchestration of remaining measurements.
  Control or comparison: successful individual reruns after one documented recovery, with no surviving owners before reset.
  Likely subsystem: external process termination; root cause remains unknown.
  Investigation performed: retained interruption evidence, did not infer a model/device failure, and reran all incomplete work. Reviewer inspected completed counters, outputs, source manifests, and allocator results.
  Resolution: controlled. No interrupted measurement is counted as passing evidence.

- Observed anomaly: two rejected controls report 1343 MHz instead of the requested 1350 MHz; common discovery/semaphore/packet advisories recur.
  Evidence: `kv_bf16.log:14`, `decoder_bfp4_hifi2.log:14`, baseline and selected logs.
  Affected path: runtime setup and comparison conditions.
  Control or comparison: decoder HiFi2 is over 12% slower; BF16 KV is memory-ineligible; recurring advisories also occur in completed baseline/selected runs.
  Likely subsystem: runtime setup/reporting; no device-health failure is established.
  Investigation performed: checked warning classes and whether the small clock difference could reverse selection. Timings remain unmodified.
  Resolution: controlled; no winning policy depends on dismissing a failing run.

## Scope Inspected

- Goal/skill paths: installed `tt-model-bringup/0.1.22/prompts/model_bringup_multigoal/08-datatype-sweep.txt`; `datatype-sweep`, `stage-review`, `agentic-serving`, `qualitative-check`, `model-bringup` startup, and `tt-enable-tracing`; review-core/router plus precision/trace review guidance. Also read `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, and the local trace-correctness guide.
- Artifact paths: datatype-sweep README/work log, all 11 candidate configs/results/runtime policies, JSON/CSV matrix, both Pareto images, selected config/propagation, baseline controls, selected default/acceptance snapshot, B1/B32 state checks, B8/B16/B32 performance/memory, long-prefix outputs, all six HF/TT shared outputs, debug/fix/anomaly/recovery reports, validation logs/summary, `doc/context_contract.json`, `doc/agentic_serving.json`, and the requested telemetry packet `ea72504e-4dea-4cca-978d-5dbbe52e79c7.json`.
- Code paths: complete `tt/{precision,model,generator,multichip_decoder,optimized_decoder,prefill,weights}.py`; datatype candidate/state/long-prefix/summarizer/finalizer/policy-test code; benchmark, inventory, context and qualitative helpers; pinned HF reference loader; relevant readiness timing and common sampler paths.
- Commands run: read-only `git status/diff/remote`, `rg`, file reads, image inspection, and small Python standard-library checks for source hashes, policy/count/timing arithmetic, CSV agreement, Pareto dominance, memory/context reconciliation, and acceptance-snapshot identity. No TTNN import, device command, server, vLLM run, reset, or hardware test was performed by this reviewer. Only this report was written.

## Residual Risk

- This is stage-specific acceptance, not a release-quality claim. The short readiness subset and six capped prompts do not establish broad task accuracy. The long-prefix probe establishes synthetic retention; maximum-context artifacts establish execution/capacity.
- Trace and memory evidence covers the selected standalone construction at the configured batches. Serving concurrency, scheduler integration, prefix reuse, and adapter registration require their later-stage checks.
- The unexplained host SIGTERM remains an operational uncertainty, but completed clean reruns supply every affected required measurement.
