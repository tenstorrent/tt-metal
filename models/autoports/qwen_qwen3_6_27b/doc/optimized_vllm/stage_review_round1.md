# Stage Review

Verdict: clean-pass

Independent round 1 review of Stage 10, optimized-vLLM, Qwen/Qwen3.6-27B, on 2026-11-17. Inspected live `main` at `8df7e416f5c8bf097ae84f1a8660cb0ef4e22951`, including staged evidence. Runtime is frozen at `d0a39b7debba48355589141ae43ef74c18db15eb`; comparison baseline is `d67807951e5411a4c1a79144da81b7fbc7852a43`. This review used source and existing artifacts only. No hardware, model import, server, benchmark, or reset was run. Only this report was written.

## Required Work

- None. No new correctness defect, failing required gate, or missing acceptance evidence was identified. The owner's final local evidence checkpoint and status updates follow this review; they are not yet completed by this verdict. The run plan says local commits only, no push.

## Other Concerns

- **No primary single-user speedup is established.** Raw primary results give TTFT 276.979049 → 291.277070 ms and TPOT-derived decode 37.632082 → 37.625576 tokens/s/user at 128 input / 128 output / one request, concurrency 1. The final TTFT is 5.162% higher. Each primary run measures one request, so neither a latency distribution nor statistical significance is established. The report correctly uses the final reproduced result instead of the earlier C3 result. The retained benefit is the separately measured mixed-continuation workload, with its explicitly reported tradeoffs. Preserve this distinction in the final handoff.
- **Mixed latency improves selectively.** The paired 17-request C1 trial has identical workload specifications and all 16 greedy token streams match. P99 TTFT falls from 7362.703 to 3461.149 ms and maximum ITL from 3639.968 to 1231.591 ms, while median TTFT and P99 ITL worsen. The final telemetry-disabled replay is a separate latency result, not another matched baseline comparison. These are useful measurements, not a general latency guarantee.
- **Quality remains a smoke-test conclusion.** I read all 12 final served texts. The inventor story changes pronouns in both modes; the greedy defect exactly reproduces in the current standalone TT control. Explanations/stories hit the 256-token cap, and thermodynamics does not cover all three laws. The HF controls also truncate, but do not reproduce the greedy pronoun defect. No HF-equivalence waiver or population-quality conclusion is justified; the current documents correctly refrain from one. Under the review skill, a relative content-quality failure requires at least ten matched blinded samples per side. This small suite does not establish such a failure.
- **P4 is load evidence.** Its saved answer tails include simulated empty-message conversations and role delimiters after generation was forced to 6000 tokens with `ignore_eos=True`. The runner retains only four tails, so the artifacts do not establish the point of the first natural EOS or the correctness of the initial answers. They establish completion, device routing, throughput and resource release. This does not invalidate the prescribed forced-length load gate, but it must not become a claim of 32 correct reasoning answers. Stage 11 accuracy evaluation remains necessary.
- **Optimization handoff is bounded by the Stop Rule.** Raw C1/C2/C3 primary decode rates reproduce the two sub-3% adjacent kept changes in `stop_rule.json`. Page sharing, intermediate decode buckets, further prefill traces, optional batching and selected-policy geometry/terminal alternatives are explicitly untried. They are handoff opportunities, not measured rejections or blockers under the authorized Stop Rule.

## Hard-Check Gaps

- The shared full sampling runner skips three opt-in chunk-recall cases because it does not supply `--tt-chunked-prefill-budget`. The 67 passes and 75 skips are accurately reported: 71 speculative dummy-server cases, those three recall cases, and one unsupported all-vocabulary-logprob case. The skipped cases are not counted as passes. Current reduced real-weight continuation checks, all-layer mixed token controls, and P5 timing cover the changed implementation; this gap is not a missing Stage 10 acceptance gate.
- The stage checker does not independently establish every claim in the serving contract. In particular, allocation tracking, trace retention, actual precision propagation, and phase-specific host routing require the supplementary artifacts. Those artifacts are present and were checked here. The telemetry-disabled mixed summary correctly marks these checks `NOT RUN` and `acceptance_ready=false`; its success alone would be insufficient.
- Greedy qualitative agreement can be verified only as text equality because the served suite does not retain completion token IDs. I independently reproduced all six exact text matches after removal of one terminal control token from the standalone control. Sampled serving requests are unseeded, while controls use seed 0, so sampled equality or RNG-stream identity cannot be inferred.
- Larger scheduler-budget measurements retain baseline-source provenance. I independently parsed the three byte-scoped baseline chunk logs: each has 103 chunks and zero chunks entering C1's newly eligible branch. C2/C3 alter redundant host validation/telemetry only. The final-source 512 confirmation remains under the bound; the unchanged-path 2048/4096 results are far above it. This is adequate for the selected budget, but should not be presented as a fresh final-source sweep of all three budgets.

## Anomaly Ledger

- Observed anomaly: initial global 32-token partition and multi-row extension failed numerical checks.
  Evidence: `AUTOFIX.md`, continuation probe artifacts and the baseline-to-current source diff.
  Affected path: GDN partition arithmetic and uniform multi-row continuation.
  Control or comparison: pinned baseline public calls with real layers 0/3/4, selected precision, logits, downstream recurrent state and cache ownership guards.
  Likely subsystem: changed chunk arithmetic, followed by multi-row numerical differences.
  Investigation performed: restored original 512 partitions, retained only physical single-row tile-aligned vector attention, and reran affected boundaries. The uniform final probe is bit-equal to baseline. Exact-max tie membership is separately tested against strictly lower candidates; no epsilon was introduced.
  Resolution: fixed by narrowing the implementation. The accepted `32→545` single-row logit PCC is 0.999977682, exact-reference-max top1 agreement 0.992202759 and top5 agreement 1.0; near-end `2016→2048` is within the same recorded gates. Prefix, outside-footprint and peer-state guards pass.

- Observed anomaly: untracked startup warns that allocations behind a live trace may be unsafe.
  Evidence: final server log line 167; `optimized_tracked/source.json`, raw lifecycle/log, and `adapter_optimized_final.log` line 68.
  Affected path: prepared model/sampling/reset traces and eager prefill between replays.
  Control or comparison: tracking enabled before server import, with `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; reduced full-preparation positive control rejects a deliberately live late device buffer.
  Likely subsystem: generic allocator warning when tracking is disabled.
  Investigation performed: inspected active capture/replay code and raw tracked events. The model-owned path adds no corruptible allocation scope or acknowledgment. It calls the common sampler's `_run_sampling` inside its own capture, not the common helper's broad capture scope. The all-layer workload ends with decode and records 759 decode forwards, five unique captures, zero retirements and 24888 programs throughout.
  Resolution: controlled by the required strict tracked workload and positive control; no unsafe surviving allocation demonstrated.

- Observed anomaly: final primary TTFT and some mixed/capacity metrics are worse.
  Evidence: raw baseline/final benchmark JSON and paired lifecycle artifacts.
  Affected path: measured serving latency.
  Control or comparison: identical actual server argv, selected policy, primary/CI workload configuration and telemetry/tracking regime; separate paired mixed trial.
  Likely subsystem: source of the small fixed-shape timing difference is not established by one request.
  Investigation performed: final default reproduced after sampling/quality checks and reported without substituting the earlier candidate number; mixed benefit and adverse percentiles retained.
  Resolution: controlled reporting boundary; no general or single-user speedup claim. Longer statistical characterization is residual performance work, not a new correctness gate.

- Observed anomaly: story continuity defects and incomplete capped answers.
  Evidence: final served outputs, current TT control and retained matched HF control.
  Affected path: shared qualitative prompts 1–3.
  Control or comparison: all six greedy served texts exactly match same-code TT; HF also reaches the cap for the long rows, but its greedy story uses consistent pronouns.
  Likely subsystem: content generation and token budget; no mechanical serving anomaly identified.
  Investigation performed: direct text inspection and exact greedy comparison; sampled comparison limited to observation.
  Resolution: controlled for serving-regression attribution; content observations remain in limitations, without an HF waiver or comparative population claim.

- Observed anomaly: P4 tails leave the original reasoning questions and include role syntax.
  Evidence: `doc/eval_readiness/long_answer_load/answers_sample.json`, 32 raw request receipts and runner's `ignore_eos` behavior.
  Affected path: forced 6000-token load generation.
  Control or comparison: prompt-correct 256-token qualitative suite separately passes; P4 explicitly forces output past ordinary stopping behavior.
  Likely subsystem: artificial continuation beyond normal response termination; the retained tails cannot localize its onset.
  Investigation performed: inspected all saved tails, the raw request rows and the load collector's retention/termination behavior.
  Resolution: controlled scope of the load gate only. No useful-answer quality conclusion is accepted from P4.

- Observed anomaly: first final acceptance driver returned 1, and upstream shutdown force-killed EngineCore.
  Evidence: `final_pipeline_commands.json`, final acceptance receipts, server-log shutdown lines and `cleanup.json`.
  Affected path: phase drain handoff and teardown.
  Control or comparison: the failed handoff followed cancellation of the completed chunk probe; bounded drain handling was added, and resumed timing/load, runtime audit and final stop all returned 0.
  Likely subsystem: orchestration handoff and upstream best-effort shutdown.
  Investigation performed: checked retained failure/resume history and final cleanup reporting. Final API/EngineCore exit and empty device-holder receipt are recorded; no reset was used. The implementation does not claim graceful trace teardown.
  Resolution: fixed handoff and controlled shutdown behavior; no remaining request failure or holder is concealed.

- Observed anomaly: stage gate returns 1 rather than 0.
  Evidence: `stage_gate_final.log` and current context/runtime records.
  Affected path: advisory scanner and optional batched-prefill targets.
  Control or comparison: full context remains 262144; the cap warnings quote historical process-return-code prose. Burst is serial/off before and after.
  Likely subsystem: gate advisories, not current capability reduction.
  Investigation performed: checked warning text against actual runtime context and the user contract making batching optional.
  Resolution: controlled; no critical/checker failure is present and no batched adoption is claimed.

## Scope Inspected

- Goal/skill paths: `goal_contract.md`; installed 0.1.24 `stage-review`, `model-bringup` startup, `vllm-integration`, its eval-readiness-load reference, `optimize`, `tt-enable-tracing`, `qualitative-check`, and applicable `tt-device-usage` requirements. The explicit no-profiler and optional-batching user overrides were applied.
- Artifact paths: this directory's README, work log, checklist, trace/async decisions, manifest, stack/commit/Stop Rule records, metrics, runtime summary code, functional-floor estimate, AutoDebug/AutoFix and change audit; continuation/SDPA/adapter receipts; baseline512 and final primary/CI/burst raw JSON; paired C1 mixed runs; final sync/tracked raw lifecycle JSON and server logs; final byte-scoped audits, actual argv/source and cleanup; raw sampling log; final qualitative text, prompt metadata and TT/HF controls; P4/P5 summaries, raw request/timeline and answer-tail artifacts; stage-gate logs and prior plugin-change records.
- Code paths: baseline diff of `tt/bounded_prefill.py`, `tt/generator.py`, `tt/generator_vllm.py`; relevant surrounding preparation, cache binding, state publication, split sampling and deferred-read code; continuation, SDPA, adapter, lifecycle and host diagnostic tests; launch/check and evidence scripts. Also inspected the common sampler's called boundary and pinned plugin scheduler inheritance. No Stage 10 changes occur in common sampling, TTNN, plugin or vLLM source. Plugin checkout is clean at `da1587235b84b13f2826cfd7e8b448b340322a54`.
- Commands run: read-only `rg`, `sed`, `cat`, `tail`, `git diff/status/rev-parse`, and Python-standard-library artifact analysis. Independently recomputed TPOT-derived rates, checked paired workload/token equality, parsed trace/program events, counted phase-scoped device/host forwards, parsed actual runtime precision, checked manifest existence/sizes, and confirmed final compressed server log equals its retained raw bytes. No test suite was rerun during review.
- Key re-derived facts: final benchmark has 282 device/0 host forwards; qualitative 1979/0; timing 494/0; long load 6522/0. Full sampling compatibility has 1947 device/477 host forwards and 67 passes/75 skips. Actual runtime records contain all 64 layers and 256 weights: 124 BFP4/LoFi, 98 BFP8/HiFi2 and 34 BFP8/LoFi, exactly matching selected policy; context/page table is 262144/[32,8192]. Tracked mixed output matches 16 synchronous greedy controls. P4 completes all 192000 output tokens and all 11 checks; P5 completes all four checks. The final selected budget's worst ITL is 974.3 ms against 1000 ms.

## Residual Risk

- This is an inspection-based acceptance of finite existing tests, not an exhaustive proof over prompts, request schedules or future plugin revisions. Tracked mixed traffic uses a small concurrency set; CI and P4 extend functional/load coverage to 32 users.
- Changed near-capacity tile fill is covered numerically at the explicit reduced 2048 allocation, with a full-context adapter and inherited maximum-context serving evidence. This stage does not repeat a maximum-length request, consistent with unchanged KV dtype/layout and the stage hardware budget rules.
- Prefix caching retains the documented recurrent-state exclusion. Batched prefill remains disabled. Neither feature is newly advertised as accepted by this review.
- Forecasts use planning defaults and extrapolate GPQA beyond the measured 6000-token window. They do not replace Stage 11 accuracy measurements or a release-specific eval plan.
- Final evidence commits, manifest/status refresh and local SHA recording remain the stage owner's post-review closure actions. No push is authorized or performed by this review.
