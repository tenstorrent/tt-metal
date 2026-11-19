# Stage Review

Verdict: clean-pass

Independent review, round 1 of at most 3, for stage 7 optimized-full-model,
Qwen/Qwen3.6-27B on Blackhole TP4. Reviewed the live `main` worktree based on
`7dde27883c7cf72ed1268a9ef5845b19134549c3`. Final generator SHA256:
`9a924831c1d59fe9ffcb0f2addf839c25a4afef97a9d219a94beecb68d86da2f`.
This is an inspection verdict over completed evidence, not a new device run.

## Required Work

- None. No correctness defect, missing required gate, unsupported performance
  claim, policy drift, or unsafe trace-lifetime change was found in the inspected
  stage changes and evidence.

## Other Concerns

- Documentation handoff: the matched HF/TT story opening, `thinking thinking`,
  is correctly classified as controlled in `anomalies.md` and
  `qualitative_verdict.json`, but `../eval_readiness/limitations.json` does not
  exist. Record this narrowly observed shared behavior with
  `original_model: "same"` and the actual HF/TT evidence paths, as requested by
  stage-review. Do not turn the single-seed observation into a population-level
  quality claim. This is documentation housekeeping, not a model-quality failure
  or a reason to rerun hardware or independent review.
- Commit handoff: compact required CSVs exist locally, but `git check-ignore`
  confirms that `perf/decode.csv` and `perf/runtime_matmul_policy.csv` are ignored.
  Ensure the compact CSV evidence is explicitly included in the local checkpoint;
  retain the bulky raw capture locally as documented. Checkpoint commits and SHA
  receipts are intentionally pending review. The run plan authorizes local
  stage-owned commits only and explicitly forbids a push.
- Optional performance families already enumerated under `Not tried (stop rule)`
  remain handoff notes. The paired P128 trace improvement is 0.2365%; parameter
  caching eliminates an upload without a resolved latency gain. The documented
  last-two-kept-changes stop rule applies. No additional candidate sweep is
  required by this review.

## Hard-Check Gaps

- No blocking gap identified. The 20 acceptance receipts have exit code 0 and
  nonoverlapping start/end timestamps. Current runtime hashes reproduce from the
  files. The retained package-mode runs differ only by the exact relative-to-
  absolute helper import replacement; the old source, proof, original failing
  log, and refreshed standalone readiness receipts substantiate this exception.
- Strict tracking is enabled with program-cache allocations included in both
  the full-stack and broader reduced tests. The separate positive-control log
  contains the expected unsafe-buffer rejection. No new acknowledgment or broad
  allocation exemption appears in the helper.
- The stage check alone is a degeneracy check and is not proof of every stage
  contract. The separate runtime sentinels, tracked replay cases, numerical logs,
  B32/mixed/resume artifacts, and profile rows supply that additional evidence.
- The original baseline command receipt is retrospective and says so. Its
  generator hash independently matches the starting commit; the measured
  before/after harness code and timing boundaries are comparable. No invented
  baseline timestamps are needed to support the reported near-noise result.

## Anomaly Ledger

- Observed anomaly: readback throughput decreases slightly and sampler parameter
  caching has no measurable TTFT gain.
  Evidence: `baseline.json`, `readback_final.json`,
  `performance_final.json.sampling_param_ab`.
  Affected path: whole-request and caller-visible readback timing.
  Control or comparison: three same-harness repetitions; overlapping decode
  ranges; cached/uncached first tokens agree and redundant uploads change 1 to 0.
  Likely subsystem: normal timing variation and small request-boundary overhead.
  Investigation performed: independently recomputed medians, inspected timing
  code and resident counters.
  Resolution: controlled. Reported values are accurate and no material speedup
  is claimed.

- Observed anomaly: active-trace allocator warnings during setup/eager helpers.
  Evidence: `full_trace_tracking.log`, `trace_contract.json`,
  `trace_positive_control.log` and their environment receipts.
  Affected path: persistent buffers and temporary allocations around live traces.
  Control or comparison: allocation tracker enabled with skip-program-cache 0;
  positive control rejects an intentionally live unsafe buffer.
  Likely subsystem: generic allocator warning, requiring actual lifetime checks.
  Investigation performed: inspected preparation/capture ordering, helper buffer
  bindings, repeated eager/traced transitions and final replay, stable trace IDs,
  program counts, and addresses. Four full-stack traces and 24456 programs remain
  unchanged; reduced complete-shape coverage prepares lengths 1 through 512.
  Resolution: controlled. No unsafe allocation exemption or request-time
  recapture is used.

- Observed anomaly: the initial standalone readiness import failed.
  Evidence: `import_failure/prefill.log`, `import_failure/generator_before.py`,
  `import_resolution_proof.json`, current prefill/teacher-forcing receipts.
  Affected path: standalone generator loading before device evaluation.
  Control or comparison: original package-mode probes resolved the same helper;
  final readiness jobs use the repaired source.
  Likely subsystem: Python relative import without a package context.
  Investigation performed: independently verified the exact one-statement
  replacement and all final runtime hashes; inspected the loader proof.
  Resolution: fixed. The repaired all-layer prefill and teacher-forcing gates pass.

- Observed anomaly: one resumed-prefill row has PCC 0.985754633.
  Evidence: `prefill_resume.json`,
  `../full_model/prefill_resume_native_receipt.json`.
  Affected path: unchanged bounded-prefill partition arithmetic.
  Control or comparison: all four starts reproduce the previous accepted
  numerical values; case-mean PCC is at least 0.999431966, minimum top-1
  agreement is 0.992217898, and final tokens agree. Both eligible no-history and
  fresh-suffix controls fail as expected.
  Likely subsystem: numerical differences across prefill partitions.
  Investigation performed: compared fresh and prior per-case metrics and the
  unchanged runner thresholds; inspected mixed-order and preemption controls.
  Resolution: controlled under the existing explicit resume contract; no stage-7
  threshold reduction or numerical regression.

- Observed anomaly: qualitative windows mostly end during thinking, and the
  story opening repeats one word.
  Evidence: `qualitative_tt.json`, `../full_model/qualitative/hf_suite.json`,
  `../full_model/qualitative/tt_suite.json`, `autoregressive/`.
  Affected path: prompt-correct 128-token generation samples.
  Control or comparison: identical model/revision, rendered prompts, prompt IDs,
  seed and sampling settings; all 12 TT token histories equal stage 6. HF also
  contains the repeated story word. Separate readiness HF/TT output is identical
  over 128 tokens, including the leading closing-think token.
  Likely subsystem: matched checkpoint/prompt/token-cap behavior.
  Investigation performed: read all 24 shared-suite completions and both
  readiness texts; independently compared metadata and TT histories.
  Resolution: controlled observation. No TT-specific content defect or sustained
  mechanical collapse is evident. Final-answer success and statistical quality
  equivalence are not established; the documentation registry note is above.

- Observed anomaly: profiler gap/utilization annotations can be misleading.
  Evidence: raw `tracy/reports/2026_11_14_06_17_56/` CSV,
  `perf/{analysis.md,kernel_accounting.json,layer_accounting.json}`.
  Affected path: performance accounting, not model output.
  Control or comparison: per-rank raw kernel durations, explicit program configs,
  signpost spans, and separate uninstrumented all-layer wall measurements.
  Likely subsystem: profiler-drain gaps and generic tool geometry assumptions.
  Investigation performed: independently summed all four signposted windows per
  rank, checked every decomposition CSV range/sum and all 18 runtime-policy rows
  against the raw CSV, and recomputed physical matrix bytes.
  Resolution: controlled. The report excludes invalid drain gaps, does not sum
  TP ranks into latency, and labels the reduced-layer projection as an estimate.

## Scope Inspected

- Goal/skill paths: `review_contract.md`; installed stage-review, optimize,
  multichip, tt-enable-tracing, tt-device-usage, qualitative-check and preserved
  full-model requirements. Relevant optimize method, accounting, TTFT, terminal
  sampling and evidence-checklist references were inspected.
- Implementation: `../../tt/generator.py`, `../../tt/prefill_trace.py`, affected
  probe/profile files, `../../tests/optimized_full_path.py`,
  `../../tests/optimized_runtime_audit.py`, and relevant bounded-prefill and common
  sampler consumers. The decoder, model, flat-prefill and bounded-prefill source
  hashes remain unchanged from the accepted baseline.
- The resident API returns the exact persistent sampler output consumed by the
  next decode. Model and sample replays are nonblocking; positions/RoPE advance
  on device; request changes retain explicit page/input refreshes. Each measured
  127-step resident window records exactly 127 model and sample replays and zero
  per-token refreshes/readbacks, with one final synchronization. The new P128
  helper prepares before capture, binds existing storage, clears/imports fresh
  recurrent state and publishes it after prefill. General, resumed and multi-row
  paths remain on the prepared bounded implementation.
- Artifacts: model/stage README, work log, checklist, mesh plan, runtime/anomaly
  reports, hard gates, source/import proof, raw numerical logs, command receipts,
  tracking/Watcher/positive-control outputs, fixed-slot B32, mixed and resume
  cases, host compatibility, qualitative/autoregressive outputs, context contract,
  prior layer/head decisions, serving contract, profile tables/CSV/raw capture,
  and formatting/health receipts.
- Independently reproduced results: same-harness TTFT 87.031086 to 86.888559 ms;
  readback token-out 30.976371 to 30.956937 t/s/user; resident 32.258325 ms/token
  or 30.999750 t/s/user. Numerical logs give prefill top-1/5/100 100/100 each and
  teacher forcing 99/100, 100/100, 100/100. These pass the specified top-5/top-100
  gates and are not AIME problem-solving scores.
- Raw profile verification: selected GDN BFP8/HiFi4 attention and BFP8/HiFi2 MLP,
  full-attention BFP8/LoFi and BF16/HiFi4 vocab head appear in executed rows.
  The sampler uses the specialized large-indices TopK family and two width-32
  candidate gathers, with no generic TopK, argmax or full-vocabulary gather in
  the selected window. Per-rank decode kernel work is 2.978180 to 2.979297 ms for
  the two-layer probe. The all-layer projection is 31.182800 to 31.489595 ms;
  the older layer-wall reference is 31.055455 ms. Their workload differences are
  disclosed, and no unexplained 10-15% excess is evident. Physical matrix traffic
  recomputes to 7,899,709,440 bytes/rank/token; the stated analytical FUNCTIONAL
  floor is comfortably below the measured throughput.
- Commands run: read-only `git status/diff/show/check-ignore`, `rg`, bounded
  `cat/sed`, and small standard-library Python scripts over JSON/CSV/text for
  hashes, medians, receipt ordering, output comparison and profile arithmetic.
  No hardware, TTNN import, reset, server, experiment or implementation edit was
  performed. Only this report was written.

## Residual Risk

- The added prefill trace covers only B1/P128/start0. Other valid shapes retain
  the bounded eager path; broader trace coverage is an optional performance
  handoff, not a capability restriction.
- Borrowed device token/logit returns are overwritten by subsequent replay.
  The documented caller-consumption rule remains necessary; this stage does
  not certify a future asynchronous serving adapter.
- Context remains 262144. Stage 7 refreshes a nonaligned 32769-token smoke at
  that allocation and retains the stage-6 maximum-context proof because cache
  dtype/layout is unchanged. Memory snapshots are not allocator peaks.
- B32 evidence is a fixed-slot synthetic functional regression with 31 active
  decode rows, not a natural-answer or vLLM serving benchmark. No vLLM integration
  or broad datatype frontier is claimed or required here.
- The quality sample size supports regression inspection, not statistical
  equivalence or final-answer accuracy. The reduced profiler supports attribution
  and accounting estimates, not a measured all-layer device critical path.
