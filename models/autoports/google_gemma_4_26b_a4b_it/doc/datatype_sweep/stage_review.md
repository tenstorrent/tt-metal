# Stage Review

Verdict: clean-pass

Independent review of Stage 08, datatype-sweep, for `google/gemma-4-26B-A4B-it`.
Reviewed the live `gemma-4-26b-a4b-it` worktree at starting commit
`0cfab7802b734e9cb36f0500271e589519ba0679`, including the completed final capacity
artifacts. The reviewer did not modify implementation, run tests, open devices,
start servers, or delegate this review. This report is the only reviewer write.

## Required Work

None. The implementation and saved evidence satisfy the supplied stage contract.
The stage owner can now perform the explicitly post-review local checkpoint and
record its SHA; no push or vLLM integration is part of this verdict.

## Evidence Supporting Pass

- Independently rederived the JSON/CSV accuracy minima and timing medians from all
  23 raw `results/*.json` files. Each policy has 30 runtime layer summaries and
  four 100-position teacher-forcing requests, each with 99 model and sampling
  trace replays. Both accuracy phases pass 90%/98%/100%. Instrumented diagnostics
  are excluded from ranking.
- `head4_inner_all4_shared_down4` is the maximum observed passing median:
  52.30587745 traced teacher-forcing tokens/s/user, 149.24657 ms warmed TTFT,
  94%/100%/100% decode and 97%/100%/100% prefill. The complete selected JSON equals
  the measured winning policy. BFP4/LoFi and legal HiFi2 comparisons cover the
  material decode groups; uniform BFP8 LoFi/HiFi2, BF16 recovery, activation,
  collective, cache, individual-group and compatible combination controls exist.
- Inspected policy resolution and actual construction/dispatch code. Requested
  weight/fidelity, activation, CCL and cache settings reach the bound tensors and
  kernels; layer overrides take precedence. Fixed prefill/router/norm/logits and
  accumulator assumptions are explicit and unsupported changes fail closed.
  Upward recovery uses original checkpoint values. Explicit `{}` restores the
  baseline, while ordinary `build_generator` construction loads the selected file.
- All raw candidate runtime summaries match their resolved model/layer policies.
  Final normal-default and capacity summaries match the winning policy. Actual
  execution observations record BF16 embedding/residual/logits and 30 BFP8 cache
  pairs. Recomputed runtime hashes match `selected/source_manifest.json` and the
  winning candidate; selected policy SHA256 is
  `9af37c1309a8cfd3b6018b7cbade031a11646dc46d3282cd2be1a4d018fba3b5`.
- The normal-default final runner independently reproduces accuracy and
  52.18254 teacher-forcing tokens/s/user. Its separately measured warmed
  autoregressive benchmark is 51.64728059 tokens/s/user and 2115.24765 ms TTFT
  at exactly 4096 input / 128 output, B1/C1, median of three requests. Refreshed
  baseline is 49.35445763 tokens/s/user in the same regime. Buffered counters
  show 127 model/sampling/output replays, one final token readback, zero full-logit
  readbacks, and no per-token token/position/cache-position refreshes. Streaming
  and buffered tokens agree exactly.
- Inspected all six selected shared-suite completions, rendered chat prompts,
  HF controls, refreshed baseline outputs, and the separate sky explanation.
  Outputs are coherent within the recorded 128-token budget. The tokenizer,
  revision, roles, prompt tokens and generation settings are recorded. No
  unexplained corruption, wrong language, mechanical repetition or leakage is
  visible in this correctly formatted suite.
- Final capacity records prove all 30 layers run 262143-token nonaligned prefill
  with final-position decode and 262144-token aligned prefill, with finite logits.
  Twelve selected boundary cases pass under trace allocation tracking. The
  all-layer B32 check preserves mixed 31/127-token requests and gives PCC 1.0 for
  isolated slots 0 and 31 across prefill and two decode steps. The updated context
  contract retains 262144 with no logical alignment restriction. Independently
  summed selected memory accounting gives 26,652,351,488 bytes/device; it is
  correctly labeled a conservative source-derived bound, not allocator telemetry.
- Visually inspected both pyplot charts and their generator: every evaluated
  full-model policy is plotted, frontiers are non-dominated, the selected point
  is red, and the dotted thresholds are 90% and 98%. The BFP4/LoFi head geometry
  probe uses actual target weights/recorded activations: 19 rows, 15 passing and
  four explicit L1 rejections, with the production 11x10/K4 geometry fastest.
- Telemetry identifiers and workload exactly match the supplied template.
  Performance is sourced from the actual 4096/128 autoregressive result; unknown
  full-model device times and rooflines remain null. Saved host verification
  records eight passing policy tests, 16 Python syntax checks and passing
  pre-commit hooks. Reviewer `git diff --check` also passes. Changes are Python
  and evidence/documentation, so no C++ build is required.

## Other Concerns

- Selection is the fastest observed passing median, as explicitly requested.
  Close candidate differences are not statistically established. Three candidate
  initialization logs report 1343 MHz against requested 1350 MHz; this is
  disclosed in the work log and reinforces the timing-noise limitation.
- The selected precision artifact is required for reproducing the default policy
  and must accompany the local checkpoint. The reviewed stage-owned files can
  be isolated from the current worktree changes.

## Hard-Check Gaps

- No new full-model device-time or roofline evidence exists. No such values or
  claims are substituted from host-wall or reduced-model measurements.
- vLLM adapter propagation is outside the user's explicit stage scope. The shared
  factory is ready to consume the selected policy; no adapter or serving result
  is claimed here.
- Maximum-context checks establish execution/capacity and finite outputs, not
  full-model reference accuracy at 262144 positions. Numerical acceptance is the
  specified 100-position readiness reference.

## Anomaly Ledger

- Observed anomaly: BF16-cache long prefill exceeded L1.
  Evidence: `smoke_kv_bf16.log`, `AUTODEBUG_bf16_cache.md`, before/after lowering JSON.
  Affected path: 512-wide-head paged prefill with BF16 K/V.
  Control or comparison: all eight BFP8 lowering cases are identical before/after.
  Likely subsystem: SDPA circular-buffer allocation.
  Investigation performed: exact allocation arithmetic, 16 lowering cases,
  repaired 12-boundary device smoke and full-model BF16 candidate.
  Resolution: fixed by selecting the existing Q64/K128 program; no context reduction.

- Observed anomaly: the first repeated accuracy diagnostic reported 200/100 predictions.
  Evidence: `diagnostics/baseline_tracker_accumulator.json` and the final candidate runner.
  Affected path: reused host accuracy collector, after one valid 100-position request.
  Control or comparison: generator made 100 callbacks; final runs use a fresh collector.
  Likely subsystem: evidence harness accumulation.
  Investigation performed: inspected failure counts and all final repeated totals.
  Resolution: fixed; failed instrumented diagnostic is excluded from ranking.

- Observed anomaly: ordinary runs warn about allocation while a trace exists.
  Evidence: candidate/default logs, tracked baseline diagnostic and `selected_nonaligned` artifacts.
  Affected path: inherited split model/sampling/output trace lifecycle.
  Control or comparison: tracked full-baseline readiness passes; selected tracked
  boundary/reuse cases and all-layer B32 isolated-slot comparisons pass.
  Likely subsystem: conservative trace allocation warning.
  Investigation performed: inspected tracking evidence, replay counters and reuse outputs.
  Resolution: controlled; no contradictory corruption is visible in the reviewed evidence.

- Observed anomaly: throughput completion repeats a nine-token phrase.
  Evidence: selected/baseline `performance.json` and `selected/performance_output_control.json`.
  Affected path: repeated raw 4096-token synthetic throughput input.
  Control or comparison: all 128 selected tokens exactly equal the refreshed baseline.
  Likely subsystem: continuation of repetitive input.
  Investigation performed: independently checked token equality/period and read
  the separate prompt-correct chat suite and sky explanation.
  Resolution: controlled; throughput input is not used for a quality verdict.

- Observed anomaly: selected free-running text differs from controls; long answers stop mid-answer/code.
  Evidence: selected/baseline `qualitative_tt.json` and embedded HF outputs.
  Affected path: greedy generation after precision changes, with 128-token budget.
  Control or comparison: same-format HF/baseline also reach the fixed budget;
  normal turn terminators appear in controls because special tokens are retained.
  Likely subsystem: expected free-running divergence and output-length limit.
  Investigation performed: read every completion and checked format/metadata.
  Resolution: controlled for this smoke; completed-code correctness is not claimed.

- Observed anomaly: four early candidates have a different optimized-decoder source hash.
  Evidence: raw source manifests, `git show HEAD`, and final `git diff`.
  Affected path: baseline and early head policies, all using BFP8 cache.
  Control or comparison: old hash equals HEAD; final change is solely the BF16
  cache program branch, and all BFP8 lowering cases remain identical.
  Likely subsystem: source provenance across the isolated BF16 repair.
  Investigation performed: independently recomputed hashes and compared the branch/evidence.
  Resolution: controlled; the winning and final-default complete source hashes agree.

## Scope Inspected

- Goal/skill paths: supplied Stage 08 contract; `.agents/skills/{stage-review,datatype-sweep,qualitative-check,tt-device-usage,autofix}/SKILL.md`.
- Artifact paths: stage README/work log, configs/results, CSV/JSON/plots, baseline
  and selected readiness/performance/quality, source manifest, boundary/B32/capacity
  results, diagnostics, AutoDebug/AutoFix reports, memory accounting, host checks,
  and `doc/context_contract.json`.
- Telemetry: `bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/ca0657b9-2c9e-4aea-8f29-86abae93b938.json` and matching template.
- Code paths: `tt/{precision_policy,multichip_decoder,model,generator,optimized_decoder,fused_decoder}.py`;
  candidate/smoke/capacity/optimized-full-model runners, precision-memory and plot
  helpers, candidate construction, policy tests and touched trace/head probes.
- Commands run: read-only `git status`, `git diff`, `git show`, `cat`, `sed`,
  `grep`, `find`, standard-library Python artifact/hash/statistics analyses,
  image inspection and `git diff --check`. No tests or hardware commands were
  run by the reviewer.

## Residual Risk

Accuracy covers one AIME24 chat prompt and 100 continuation positions, not the
dataset. Qualitative evidence is a six-prompt 128-token smoke plus one explanation.
The measured winner is limited to the evaluated matrix/workload, and small timing
differences may change on repeat. Maximum-context support is B1; B32 evidence is
for short requests. BF16-cache maximum-context capacity is unproven and that
slower policy is not selected. Device behavior is assessed from retained stage
evidence; this independent review did not rerun hardware.
