# Stage Review

Verdict: more-work-needed

Stage 10, optimized-vLLM, Qwen/Qwen3.6-35B-A3B. Independent review, round 1 of at most 3. Inspection only; no devices, servers, tests, benchmarks, resets or process signals were used. This report is the reviewer's only write.

The selected capacity-32 serving evidence is substantial and the reported performance improvement is reproducible from the saved raw results. It does not pass either unwaived performance floor. Source inspection also finds an independently actionable serving-capacity regression: the adapter enables a helper restricted to 2/4/32 rows for every accepted capacity from 2 through 32.

Paths below are relative to `models/autoports/qwen_qwen3_6_35b_a3b` unless explicitly checkout-relative. `D` denotes `doc/optimized_vllm`; `R` denotes `readiness_vllm`.

## Required Work

- **P1 — RW1: Preserve the accepted serving capacities outside 1/2/4/32.**
  Evidence: `tt/generator_vllm.py:32` accepts every capacity 1 through 32; lines 36–38 include that capacity in the prepared decode buckets; line 58 enables `fuse_hidden_row_mask` whenever capacity is at least 2. `tt/generator.py:536` prepares every such bucket with masked decode. `tt/model.py:440` consequently passes the fused mask to `decode_batch`, whose new guard at `tt/multichip_decoder.py:1916` rejects every row count except 2, 4 or 32. `tt/hidden_row_mask.py:76` has the same restriction. For example, capacity 8 prepares `[2, 4, 8]` and fails on the 8-row warmup; capacity 3 prepares `[2, 3]` and fails on 3. The affected accepted capacities are 3 and 5–31, including common capacities 8 and 16. The base implementation did not have these fused-mask restrictions. This conclusion follows directly from the construction and preparation branches; no hardware reproduction is claimed.
  Why this matters: these configurations fail during preparation, before serving requests. A successful capacity-32 workload containing three or eight active requests does not cover a server constructed with capacity 3 or 8. This is a capability regression, not an optional optimization opportunity.
  Required next step: keep the existing generic row-extraction/masking path for row counts the fused helper does not support, or extend the helper with appropriate exactness and trace coverage. Preserve the accepted capacity range. Check dispatch selection across 1–32 and exercise one representative previously rejected capacity through preparation and decode; use focused affected-path checks rather than repeating unrelated long-context/load measurements. The selected capacity-32 path can retain its existing evidence if its executable path is unchanged.

- **P1 — RW2: The primary single-user FUNCTIONAL floor remains failed.**
  Evidence: the raw `R/optimized_acceptance_perf/vllm_result.json` gives `mean_tpot_ms = 31.426704362196126`, or **31.8200721423 tokens/s/user**, for 128 input / 128 output / concurrency 1, served capacity 32 and context 262144. The unchanged useful-work ceiling is **21.63129875 ms/token**, or **46.2293092781 tokens/s/user**. Recomputing `1,107,522,496 bytes/rank / 512e9 bytes/s / 0.10` yields that ceiling. Its source is `tests/multichip_functional.py`, the acceptance calculation in `tests/run_full_model_performance.py:737`, and the inherited full-model contract. The result requires another **31.1691% latency reduction**. `D/AUTOFIX.md`, `D/autofix_failure_audit.md` and both limitations records explicitly preserve this failure without a waiver.
  Why this matters: same-configuration TPOT improved **41.582762%**, from 53.7969706929 ms, but improvement and successful HTTP completion do not pass the floor. Matched native serial 31.5064299133 ms and queued adapter 31.4449860906 ms correctly localize the remaining cost largely outside serving overhead; they do not change the required useful-work numerator. The prior Stage 5 owner acceptance does not extend to this stage.
  Required next step: retain this as an unresolved acceptance requirement. Passing needs a correct selected-path measurement meeting the unchanged floor, or an explicit applicable owner revision of this requirement. The recorded thirteen-hypothesis AutoFix failure is credible evidence of that campaign's failure; do not relabel it as acceptance, an impossibility proof, or authorization for arbitrary optional searches. The review does not require another unchanged benchmark solely to reconfirm this already established failure.

- **P1 — RW3: The newly measured P4 batch-32 functional floor is critical and unwaived.**
  Evidence: `doc/eval_readiness/long_answer_load.json` records first-bucket median **2.36 tokens/s/user**, below the conservative necessary bound **4.764924390924285**. The raw `long_answer_load/requests.jsonl` has 32 error-free records, each with 6000 output tokens, with maximum prompt length 74; this validates the bound's stated prompt scope. Re-summing `D/eval_batch32_necessary_floor.json` gives **10,745,186,240 maximum useful bytes/rank/step** and `0.10 * 512e9 / bytes = 4.764924390924285`. Pricing the maximum possible expert union makes this a relaxed necessary bound, not a claim about measured expert coverage. Even this relaxed bound is missed. P4 exits **2** in `R/optimized_acceptance_eval_final/records/p4_long_answer.execution.json`; the installed stage check exits **2** and explicitly reports `CRITICAL long_answer_load/functional_floor`. The matching `items` entry in `doc/eval_readiness/limitations.json` says `original_model: "not_run"`, `fix_hours: 0`, and `gate_waived: false`.
  Why this matters: 192000 completed output tokens, stable traces, on-device sampling, health and resource release establish execution correctness, not this performance gate. Merely listing the finding in limitations does not satisfy the named probe's critical-check waiver rules. The earlier single-user AutoFix campaign is not evidence of time spent repairing this newly measured P4 failure.
  Required next step: treat this exact P4 failure as remediation work under the probe/AutoFix rules. Resolve it, or complete and accurately record an applicable evidence-backed disposition allowed by the governing contract; do not claim an existing time-box or owner waiver. Rerun the affected P4 check only if a change or valid new disposition warrants it. RW2's prior AutoFix failure does not by itself close this item.

Not every remaining finding requires human authorization: RW1 has a model-owned repair path, and RW3 has no completed repair campaign in the evidence. This verdict is not permission to mark the stage complete or to treat the review itself as a terminal blocking condition.

## Other Concerns

- **No chunk budget meets the mixed-traffic ITL target.** The three current canonical chunk artifacts record worst background gaps of **25.6824 / 92.4546 / 186.7118 seconds** at budgets **512 / 2048 / 4096**, respectively, against 1 second. The admission audits establish all 31 background prefills before the long requests. Budget 2048 is accurately labeled a retained diagnostic setting, not a qualifying choice. The named `eval-readiness-load.md` and emitted P5 check classify this result as advisory; the current limitations record preserves it. Do not describe the service as meeting the mixed-traffic latency bound. Background 16384-token values are caps: the packaged chunk probe stops background streams after the foreground measurements, so its completed probe does not establish 31 completed 16384-token answers.
- **The load forecast remains advisory planning evidence.** The recorded combined default plan is 15.08 hours against a 14.4-hour allowance. GPQA extrapolates beyond the measured 6000 output tokens. Replace defaults with the actual release plan in the next stage; this is not measured evaluation accuracy or a release-job duration.
- **Quality evidence is bounded.** I read all twelve current served shared-suite completions, the six distinct sampled native controls, and the four P4 answer tails. Direct comparison establishes that all six greedy native/served outputs are equal, and all twelve outputs on each current path equal their corresponding Stage 9 controls. The shared outputs are coherent task-directed reasoning at a 256-token cap, not completed-answer accuracy. Three forced-ignore-EOS P4 tails contain repeated turn markers or invented dialogue; the retained tails do not locate the first EOS. Four tails without ten blind matched samples per side do not establish a TT-specific quality regression. Keep that observation separate from a quality pass or HF-equivalence waiver. If investigated later, use the skill's matched blind control rule.
- **Untried optional optimization stays handoff work.** The four-hour optional-search stop rule and owner steering justify leaving the documented geometry, projection/transport, head-to-sampler, embedding and optional batched-prefill ideas untried. Their absence is not additional Required Work. Batched prefill was not adopted; the 0.9827766667 measured burst ratio is correctly a serial-before/serial-after ratio, not batched-equals-solo evidence.

## Hard-Check Gaps

- The completed checks exercised a server constructed at capacity 32 and its 2/4/32/solo0 traces. They did not cover the construction-time capacity/helper mismatch in RW1. Include that boundary in focused dispatch checks.
- The installed stage script catches P4's floor failure, but does not independently enforce the inherited primary single-user useful-work ceiling. RW2 must remain an explicit acceptance condition even if a later script invocation stops reporting a critical P4 row.
- P5's interrupted wrapper has no recoverable exit code. This is transparently retained as `null`, not changed to zero. The fresh canonical P5 result, printed completion, dedicated server audit and subsequent idle boundary are adequate evidence that the timing requests finished. No duplicate hardware run is requested for this bookkeeping limitation.
- HTTP logit evidence compares returned top-20 logprobs/tokens; the separate source-matched standalone comparison covers full-vocabulary logits. Preserve that distinction. No new profiler or runtime counter is required by this review.

## Anomaly Ledger

- Observed anomaly: accepted server capacities 3 and 5–31 reach a rejected fused row count.
  Evidence: the source chain and static bucket evaluation in RW1.
  Affected path: adapter initialization and generator-wide preparation.
  Control or comparison: the base source lacks the new fused-mask guard; capacity 32 is the measured passing path.
  Likely subsystem: model-owned optimization dispatch.
  Investigation performed: followed accepted capacity, bucket construction, mask selection and decoder/helper guards.
  Resolution: more-work-needed, RW1.

- Observed anomaly: single-user useful-work floor still fails after optimization.
  Evidence: raw before/after benchmark JSON, native comparison and AutoFix artifacts cited in RW2.
  Affected path: 128/128/1 token-out serving.
  Control or comparison: identical actual server argv; same-family native/serial/queued-async controls.
  Likely subsystem: retained grouped model execution, rather than a substantial adapter penalty.
  Investigation performed: rederived raw TPOT, reciprocal throughput, improvement and floor; inspected the thirteen-hypothesis campaign and its rejected/retained decisions.
  Resolution: more-work-needed, RW2; AutoFix failure is recorded, not waived.

- Observed anomaly: 32-user long-answer floor fails despite healthy complete execution.
  Evidence: P4 raw request records, result, necessary-bound artifact and exit-2 check.
  Affected path: 32-user eval-profile decode, first 1000-token bucket.
  Control or comparison: plain/eval TPOT 421.53/421.20 ms; device-only decode in the raw log; valid prompt-length bound.
  Likely subsystem: batch model execution performance; no precise kernel attribution is established.
  Investigation performed: checked request counts, output lengths, prompt scope, bound arithmetic, route counts and waiver fields.
  Resolution: more-work-needed, RW3.

- Observed anomaly: original concurrent raw-completion recall test fails at its 24-token cap.
  Evidence: original JUnit has 70 passes, 71 skips and one failure; ten failed responses in `D/recall_original_failure_attribution.json` exactly match raw HF controls. Configured final JUnit has 71 passes, 71 expected speculative skips and no failures, with the same 142 test IDs.
  Affected path: short-answer recall test request format.
  Control or comparison: fourteen prompt-correct nonthinking HF/native/served fixtures, unchanged cap/assertions/concurrency, actual ragged continuation and intervening decode evidence.
  Likely subsystem: test prompt mode and reasoning budget.
  Investigation performed: inspected the six narrowly named request overrides, original failure attribution, configured JUnit and recall partition audit.
  Resolution: controlled for this configured suite. This is not unchanged-default execution or a broad quality waiver.

- Observed anomaly: sampled native/served texts differ; shared q03 begins with “thinking thinking”.
  Evidence: current shared outputs, native compressed artifact and matching HF q03 controls.
  Affected path: bounded shared qualitative suite.
  Control or comparison: exact per-path Stage 9 reproduction; six exact current greedy pairs; HF also has the q03 phrase. `doc/eval_readiness/limitations.json` preserves the narrowly scoped `original_model: "same"` entry.
  Likely subsystem: expected sampler-stream differences across APIs and checkpoint wording.
  Investigation performed: direct text comparisons and inspection of actual completions, prompt metadata and the HF control.
  Resolution: controlled within the recorded scope; no sampled-trajectory identity or statistical quality claim.

- Observed anomaly: mixed-traffic pauses exceed the bound and P4 forced-ignore-EOS tails contain repeated dialogue.
  Evidence: canonical chunk rows and `long_answer_load/answers_sample.json`.
  Affected path: long prefill admission latency; post-answer forced generation.
  Control or comparison: three budgets with admission audits; no ten-sample matched quality control for the four P4 tails.
  Likely subsystem: serial/chunked prefill cost; output beyond normal stopping is a hypothesis for the tails, not proven first-EOS attribution.
  Investigation performed: read probe implementation, measured values and all retained tails; applied the content-judgment sample rule.
  Resolution: controlled as disclosed advisory/quality-evidence limits, not an ITL pass or quality waiver; no unsupported TT-regression finding.

- Observed anomaly: renamed EngineCore `/proc` environment can be empty; P5 wrapper exit status is lost.
  Evidence: runtime identity, collector negative controls, confirmation execution and P5 recovery record.
  Affected path: evidence collection.
  Control or comparison: verified inherited API-parent environment; tracked negative control rejected; untracked confirmation accepted; fresh P5 completion and idle boundary.
  Likely subsystem: process-title replacement and interrupted orchestration.
  Investigation performed: compared actual server argv/environment and raw completion/idle records.
  Resolution: controlled. Earlier empty worker environment is not treated as proof of disabled tracking.

- Observed anomaly: interpreter shutdown prints nanobind reference-leak warnings.
  Evidence: final eval server log and shutdown record.
  Affected path: interpreter teardown.
  Control or comparison: shutdown events release all six traces; saved cleanup has no remaining owned PIDs and no device holders.
  Likely subsystem: Python/native binding teardown references.
  Investigation performed: checked final events and cleanup records separately from warning text.
  Resolution: controlled as a residual teardown warning, without claiming a live engine/device leak or proving the references harmless in every embedding environment.

## Scope Inspected

- Goal/skill paths: `D/stage_review_contract.md`; original `/home/ttuser/qwen36-35b-a3b/multigoal_logs/10-10-optimized-vllm.prompt.txt`; installed 0.1.24 `stage-review`, `optimize`, `vllm-integration`, `tt-enable-tracing`, `tt-device-usage`, `qualitative-check`, model-bringup startup instructions, eval-readiness-load and plugin-change references; checkout `TraceCorrectness.md`.
- Artifact paths: `D/README.md`, work log, manifest, both limitations schemas, checklist, cumulative contracts, AutoDebug/AutoFix floor records, source match, performance comparisons, async contract/comparison, shared quality controls, recall repair and original attribution, stage check; canonical context/serving contracts; `R/optimized_before`, current `optimized_acceptance_*` sampling/lifecycle/growth/native/performance/eval artifacts; canonical P4/P5 results, P4 requests/metrics/tails and three current chunk results; component row/state/native-QK/postreduce evidence and positive tracker control.
- Code paths: stage diff from base `5e2dca2e09485be6c4f18c688a1b7b532e45a99b` through live tree at HEAD `8fbe38e7ce079a19153b1baad80f270f3d0c3025`; `tt/generator.py`, `generator_vllm.py`, `model.py`, `multichip_decoder.py`, `hidden_row_mask.py`, `moe_route_mask.py`, `mesh_state_slab_recurrence.py`; checkout-relative native `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/` changes; serving/output-lease code and request-mode helper. The native valid-prefix default remains zero, with the original full-chunk path retained.
- Commands run: read-only `git status`, `git diff`, `git rev-parse`, `rg`, `cat`, `sed`, `nl`, and small standard-library Python artifact analyses. Analyses parsed JUnit, compared raw token/text arrays and actual argv, recomputed metric/floor arithmetic, counted raw log route/trace events, and rehashed all 73 production source entries against the current full-40 artifact. All 73 matched. No TTNN/model import or target execution was used.

Independent raw checks reproduced: final configured JUnit **142 total / 71 passed / 71 skipped / 0 failed** with identical test IDs to the original; all six async/sync request token arrays equal; raw final sampling log **1474 device / 988 host diagnostic steps**, six captures and one six-trace identity set over **5157 traffic events**; final P5/P4 log **6657 device / 0 host steps**, six captures and one identity set over **13411 traffic events**, maximum pending outputs 2 and six shutdown releases. The tracked serving identity records tracker=1 and skip-program-cache=0; the positive control rejects a deliberately unsafe allocation before a clean replay. Production paths add no broad corruptible-allocation exemption. Component acknowledgments name exact outputs and consume them before subsequent replay.

The performance confirmation has identical actual server argv to the baseline and records audit/tracker/profiler disabled. Its primary, CI and serial-prefill raw result numbers agree with `D/confirmation_comparison.json`. Full-40 exact comparisons, native comparison, tracked correctness and untracked performance are kept distinct. Served context remains **262144**; the recurrent-state prefix-cache exclusion is documented, and batched prefill is explicitly optional.

Production checkpoint `08384c1007d1a1065955221284f6d2c197d85260` and documentation checkpoint `8fbe38e7ce079a19153b1baad80f270f3d0c3025` are local WIP checkpoints, not stage acceptance. The plugin checkout is clean at `5d1785b76d17a8989c9495f6939abea6cef62f09`; its pre-existing repair is recorded in `doc/vllm_integration/plugin_changes.md`. The run plan states no push was authorized or performed. This reviewer made no commit or push.

## Residual Risk

- This was a source/artifact review. RW1 is a deterministic source-path finding, not a new hardware result. Recorded hardware evidence is strongest for the selected capacity-32 configuration and its prepared signatures.
- Warmed headline numbers are single-request measurements, not confidence intervals. The confirmation closely reproduces the selected result, but neither it nor native agreement closes the failed floors.
- Shared qualitative controls establish bounded behavior and lack of an introduced regression in those fixtures. They do not establish long-answer evaluation accuracy, unrestricted constrained-generation quality or behavior beyond the measured output lengths.
- Optional untried changes, broader native-op consumers and long-running service behavior outside the saved workloads remain outside this review. No additional generic sweep, profiler run or soak is required solely to enlarge this evidence format.
