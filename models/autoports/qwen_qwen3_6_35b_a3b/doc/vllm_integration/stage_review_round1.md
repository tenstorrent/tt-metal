# Stage Review

Verdict: more-work-needed

Fresh independent review, round 1 of at most 3, Stage 9 vLLM integration for `Qwen/Qwen3.6-35B-A3B`. Paths below are relative to `models/autoports/qwen_qwen3_6_35b_a3b` unless stated otherwise. This review made no implementation changes and ran no device, server, or inference jobs.

## Required Work

- P1: The matched constrained-thinking comparison still shows materially worse TT output; native TT reproduction localizes this failure but does not close it.
  Evidence: `readiness_vllm/sampling_controls/{hf_bad_words_card_1024.json,http_bad_words_card_1024.json,card_blind_review.json,card_blind.key.json}`. I rejoined the frozen blind scores to the key: HF has 2/10 secondary failures and no sustained loops; served TT has 6/10 secondary failures and two sustained loops. The excess is 40 percentage points against the predeclared greater-than-20-point rejection rule. The prompt, revision, rendered prompt tokens, bans, sampling settings, presence penalty, 1024-token cap, and seeds 0–9 match. I read the actual HTTP outputs: seed 0 repeatedly drafts `Say greeting!` and finally echoes that instruction; seed 3 repeatedly drafts `Say_hello_` through the cap without final content; seeds 4 and 8 give farewells. These are actual output defects, not merely a missing EOS assertion. I independently compared all native `tokens` against actual HTTP `choices[0].token_ids` and finish reasons: all 10 sequences, totaling 4620 tokens, match exactly. The native precision dictionary equals the selected policy.
  Why this matters: `qualitative-check/SKILL.md:42,55` requires work when TT is materially worse than HF or exceeds the chosen matched-comparison margin. The 10 blind samples per side satisfy the stage-review minimum. The valid native control establishes inherited full-model behavior for this corpus and argues against an additional serving-only regression. It does not show HF-equivalent behavior or make the bad output acceptable. The six-prompt thinking-off suite and the separate HF-equal thinking-off constrained comparison do not cover this explicit-thinking recipe. Neither the earlier resumed-prefill owner acceptance nor the later wall-cap/resume messages waive this finding. The contract supplement correctly preserves the failure but its distinction between a diagnostic and a named numerical gate does not override the qualitative skill's explicit work requirement. This is an output-quality finding, so it cannot become a non-correctness carried item merely by reaching round 3.
  Required next step: Resolve the inherited full-model quality finding, using the saved shared-prefix boundary evidence to localize a justified repair, then recheck the affected matched comparison without changing its frozen rubric. `doc/vllm_integration/sampling_debug/{AUTOFIX.md,QUALITY_AUTOFIX_CLOSURE.md}` already document an unsuccessful repair investigation; that may justify stopping under the original goal's AutoFix-failure condition, but it does not justify completing the stage or issuing a passing review. If the owner chooses to accept this specific limitation instead, record that explicit new decision and its exact scope. No repeat maximum-context or P5 run is requested for this finding.

- P2: The final launcher adopts a direct-answer default without a recorded operator acceptance of that choice.
  Evidence: `tests/run_vllm_server.sh:26` unconditionally supplies `--default-chat-template-kwargs '{"enable_thinking":false}'`; the final server launch records confirm it. The pinned checkpoint's `chat_template.jinja:151–156` starts open thinking unless `enable_thinking` is explicitly false. `doc/vllm_integration/work_log.md:208` introduces this as a candidate operator policy, explicitly not yet accepted, to avoid exhausting short test budgets. Later entries record successful tests and call it the final policy, but provide no operator decision accepting the default change. I read the original goal and all five `09-09-vllm.resume-*.message.txt` files: they authorize continuation, recovery, and relaxed time limits, not this specific policy.
  Why this matters: The default behavior of ordinary chat requests changes, and the final sampling success depends on the selected direct-answer configuration. Explicit thinking still works in P3; this is not a claim that thinking capability was removed or that the implementation of the flag is defective. It is a concrete served-default choice that the stage itself identifies as operator policy. `stage-review/SKILL.md:280–283` requires an operator decision for such a choice; passing tests and agent-authored policy prose are not that decision.
  Required next step: operator decision. Obtain and record acceptance of the already implemented, measured direct-answer default described in `work_log.md:208`, or restore the checkpoint default and validate only the affected sampling/qualitative behavior. Accepting this default alone does not close the separate explicit-thinking quality finding above.

## Other Concerns

- Admission-dependent seeded host/device routing remains a real limitation. The saved `readiness_vllm/mixed_route_probe.json`, `readiness_vllm/mixed_route_analysis_legacy_parsed.json`, and `readiness_vllm/mixed_route_server/server.log` establish different first-token routes and a subsequent device-to-host transition for the same seeded `Random: ` request when penalty-bearing companions arrive. The final full-profile pass is real, but it does not repair that scheduling-dependent behavior. The older uninstrumented `TopP: ` mismatch remains individually unattributed. This is appropriately separated as reproducibility-only transport behavior under the sampling skill exception; it is not evidence of corrupt logits or a reason to demand another broad sampling rerun. It cannot waive the independent quality finding.
- Some inherited documentation is easy to misread as current serving behavior. `tt/serving_capabilities.py::serving_limits` still describes a top-k ceiling of 32 and a single fixed decode bucket. The serving-contract baseline rows similarly retain old standalone semantics, while `vllm_stage9_adapter` accurately describes 64 candidates and buckets 2/32. The `seed` row's broad requirement about stable streams across host/device transitions also needs an explicit pointer to the serving routing limitation. Reconcile these descriptions during ordinary owner documentation cleanup; the measured paths and adapter declarations are sufficiently explicit, so this is not a request for new hardware evidence.
- The native SwiGLU completion repair is justified as a source-contract repair with bounded regression evidence. The causal explanation for the original recurrent hang remains provisional because both unpatched small native controls passed and no next-program ACK overshoot was captured. Keep that qualification in the handoff.
- Primary performance uses one request, and the matched HTTP comparison has two measured repetitions. Their percentile labels are correctly reported but are descriptive rather than reliable tail estimates. The evidence supports no dominant adapter overhead in the measured B2 path, not a universal claim about every workload or the faster, numerically different B1 path.

## Hard-Check Gaps

- `doc/vllm_integration/logs/stage_check_final64.log` really reports exit 1 for the missing prefill `serial_reference`; it also reports passing context, serving-contract, degeneracy, and eval-readiness checks. The more specific current prefill-features reference permits serial prefill and makes this comparison advisory when no batched implementation exists. The completed 32-request serial burst is present. I do not require a redundant serial-versus-itself run or batched implementation.
- The canonical full sampling JUnit contains 142 cases: 68 passed, 74 skipped, zero failures/errors. Seventy skips require speculative serving, one requires speculative artifacts, and three require the omitted `--tt-chunked-prefill-budget` test argument. Thus “canonical skips” does not mean chunked-prefill coverage came from that suite. The separate final same-server lifecycle, plugin slot regressions, and async growth controls supply the relevant serving coverage; no extra suite run is requested solely for those skips.
- The degeneracy check passes on the shared-suite artifacts. It is not a quality verdict over the separately named constrained-thinking files and does not detect or dispose of the two loops described above.
- P5's original process exit status is unknown. Its nine completed raw cases, matching returned needles, stop finishes, completed normalized result, server log, and later idle/cleanup evidence establish the required result without inventing an exit code. Preserve `PASS_RECOVERED`; no rerun is requested.

## Anomaly Ledger

Evidence filenames in this ledger are shorthand for the evidence paths mapped above and in `doc/vllm_integration/review_brief.md`: runtime and quality files are under `readiness_vllm/`; source-audit Markdown reports are under `doc/vllm_integration/`.

- Observed anomaly: constrained-thinking loops and wrong-task answers occur more often on TT than HF.
  Evidence: `sampling_controls/card_blind_review.json`, the paired raw outputs, and `tt_native_host_card_all10.json`.
  Affected path: explicit thinking with the recorded bad-word constraint and model-card recipe, using explicit host compatibility sampling.
  Control or comparison: 10 matched blind HF/HTTP samples; all 10 native TT/HTTP trajectories independently verified exact.
  Likely subsystem: inherited TT full-model numerics upstream of sampling; exact operator and causal chain to later loops are not established.
  Investigation performed: larger budget, parsed-field controls, shared host sampler/RNG checks, shared-prefix logit/draw comparisons, native reproduction, and source audits.
  Resolution: more-work-needed, Required Work P1; not an HF-equal limitation.

- Observed anomaly: default chat mode changed to direct answers to satisfy short-budget serving tests.
  Evidence: launcher line 26, final launch records, work-log candidate policy, pinned template default.
  Affected path: requests omitting `enable_thinking`.
  Control or comparison: final sampling and shared suite pass in the new mode; P3 validates explicit thinking overrides.
  Likely subsystem: serving policy, not tensor execution.
  Investigation performed: paired thinking-off controls, original focused assertions, full suite, and P3.
  Resolution: more-work-needed pending the operator decision in Required Work P2.

- Observed anomaly: seeded outputs can differ across mixed-batch admission patterns.
  Evidence: `mixed_route_probe.json`, `mixed_route_analysis_legacy_parsed.json`, and the preserved historical sampling failure.
  Affected path: optional host compatibility when penalty requests join or leave.
  Control or comparison: the captured pair differs at the first token after different prefill samplers; nine other cases match; final full suite passes on its observed schedule.
  Likely subsystem: plugin per-step sampler routing and distinct host/device RNG streams.
  Investigation performed: request-to-slot/admission reconstruction, exact response tokens, source routing inspection, full-vocabulary standalone and exposed HTTP logit controls.
  Resolution: controlled as a documented reproducibility limitation, not repaired and not a quality waiver; the older individual TopP cause remains unproven.

- Observed anomaly: the shared suite truncates six responses at 256 tokens; the separate nonthinking constrained greeting comparison is poor on both implementations.
  Evidence: final 12 raw shared outputs and matching native/HF controls; `off_card_unblinded_summary.json`; `doc/eval_readiness/limitations.json`.
  Affected path: bounded qualitative checks.
  Control or comparison: q02/q03/q04 also lack complete answers in controls; the constrained nonthinking comparison reports HF 10/10 and TT 10/10 secondary defects.
  Likely subsystem: token budget and original-model behavior under the recorded constraints.
  Investigation performed: actual output reading, matching prompt/template/settings, frozen blind comparison for the constrained case.
  Resolution: controlled for these exact configurations; no complete-answer or general quality claim and no waiver of explicit thinking.

- Observed anomaly: continuation state ownership could change when a prefill row moved.
  Evidence: plugin fail-before log with three failing regressions, repair `5d1785b`, 2306 passing host tests/16 skips, and final lifecycle.
  Affected path: recurrent-state slots across chunked-prefill admission/compaction.
  Control or comparison: same regressions pass after preserving live owners; final actual chunk offsets interleave with decode.
  Likely subsystem: pinned shared plugin slot allocator.
  Investigation performed: source contract comparison, exhaustive small-capacity CPU reproduction, minimal model-independent repair, host suite, and serving checks.
  Resolution: fixed; the eight-part plugin change record is complete and the change adds no shared interface.

- Observed anomaly: native reader completion could leave ACKs outstanding; full-model runs previously hung.
  Evidence: `AUTOTRIAGE_RECURRENT_ZERO.md`, `NOC_COMPLETION_AUDIT.md`, `noc_boundary_probe/{synthetic_comparison,native_baseline,native_drained}.json`, repaired full-model controls, and final serving runs.
  Affected path: native fused SwiGLU reader followed by subsequent kernels.
  Control or comparison: synthetic outstanding ACK count becomes zero with the drain; empty/nonempty native controls pass before and after; repaired full-model and serving runs finish.
  Likely subsystem: NoC completion contract; original hang causality remains unproven.
  Investigation performed: firmware/source audit, bounded synthetic/native comparisons, one-line unconditional barrier, regression runs and authorized recovery.
  Resolution: fixed completion-contract gap; controlled residual uncertainty about the original hang.

- Observed anomaly: top-k40 previously used the host route and failed P2's 32-user latency ratio; an initial width selector also overlooked dormant wide rows.
  Evidence: preserved P2 failure, native32/64 probes, TP4 mesh probe, final-source `adapter_sampling64_dormant/result.json`, and final P2 raw/normalized results.
  Affected path: canonical device sampler width and persistent parameter ownership.
  Control or comparison: unchanged default32 fixture identities; 64-candidate tail membership; dormant k40/k64 preservation; final reasoning ratio 1.007410 at 32 users with 2547 device/0 host decode steps.
  Likely subsystem: native sampling draw width and generator trace selection.
  Investigation performed: native extension, two retained prepared sampling traces, all-physical-row selector, source-hash-matching reduced adapter checks, lifecycle, full sampling and P2.
  Resolution: fixed in the measured final implementation.

- Observed anomaly: initial native-library deployment and probe serialization failures, performance preflight disagreement about unset tracker flags, and checker context-cap false positives.
  Evidence: preserved failed attempts and corrected artifacts under `sampling64_probe`, `final64_perf_preflight_attempt`, and stage-check logs.
  Affected path: build deployment, test collection, and metadata interpretation.
  Control or comparison: installed native reruns pass; runtime treats unset/0 as tracking disabled; actual served context remains 262144.
  Likely subsystem: harness/environment bookkeeping.
  Investigation performed: corrected installation/serialization and metadata interpretation without changing thresholds, requested context, or benchmark workloads.
  Resolution: fixed/controlled; these attempts are not counted as passing runtime evidence.

## Scope Inspected

- Goal/skill paths: original `/home/ttuser/qwen36-35b-a3b/multigoal_logs/09-09-vllm.prompt.txt` and all five Stage 9 resume messages; installed `stage-review`, `vllm-integration`, `qualitative-check`, `tt-device-usage`, model-bringup startup, relevant tracing requirements, and prefill/plugin-change/eval-request references. Prior resumed-prefill owner acceptance was treated as limited to its recorded scope.
- Artifact paths: Stage 9 README/work log/manifest/inventory/brief/checker; serving/context/precision contracts; final64 gate and performance artifacts/raw results/logs; native64 and real-adapter probes; sampling JUnit; raw shared qualitative outputs and controls; constrained-quality blind/raw/native/boundary evidence; P3 raw responses; all nine P5 raw cases; maximum-context result; async/logit controls; plugin regression logs; NoC repair evidence; both final shutdown records.
- Code paths: `tt/{generator_vllm,generator,serving,serving_state,serving_capabilities}.py`, associated cache/model/slot-state and lifecycle-test paths; common sampler and native sampling diffs; fused SwiGLU reader epilogue; pinned plugin model interface, execution/reload documentation, slot allocator diff, async finalization and routing source; model-owned server launcher; pinned checkpoint chat template.
- Commands run: read-only `rg`, `cat`, `sed`, `tail`, `git diff/status/log`, and short Python JSON/XML/source-analysis scripts. Independently recomputed frozen blind counts, all 4620 native/HTTP token identities and finish reasons, selected-policy equality, final reduced-probe source identities, raw benchmark arithmetic, JUnit outcomes/skips, raw routing counts, and P3/P5 output evidence. No hardware experiment, server, model import/inference, test suite, reset, or process termination was run.
- Verified positive evidence: adapter delegates to the canonical generator and uses vLLM's exact cache; documented capability names; generator-owned preparation; nonblocking retained model/sampling traces; real async overlap/page growth; final lifecycle 18/18; final full sampling 68/74 skips/0 failures; P2/P3/P5 pass; full 262144 context and nonaligned prompt acceptance; documented recurrent-state prefix exclusion; completed primary/CI/prefill bursts. Primary raw TTFT is 3006.653 ms and TPOT-derived decode is 18.588 tokens/s/user. Performance intervals contain 640 device/0 host decode steps. Final shutdown records show four captures/four releases, no remaining owned PIDs, and no device holders.
- Repository state: live `main` worktree, tt-metal baseline `47c0bbd606a691e6111b46c5ee1eb9e01fd423d8`; stage-owned changes await a local checkpoint. Plugin base `042a94863ad27d88207bcf71abdf67122c6f4c27`, local repair `5d1785b76d17a8989c9495f6939abea6cef62f09`, clean plugin worktree; pinned install recommends upstream vLLM 0.31.0. Run plan forbids stage-agent pushes. Local tt-metal commit follows a passing review, so its current absence is not an additional pre-review finding.

## Residual Risk

- This review establishes a sound body of serving transport and performance evidence, but cannot approve completion while the output-quality finding and operator default decision remain open. The remaining findings are not exclusively plugin authorization requests.
- Exact native reproduction is limited to the captured ten-seed constrained corpus and does not establish general absence of serving regressions. The failed HF comparison is also limited to that corpus and does not establish global model accuracy degradation or an unavoidable quantization cause.
- P3, P5, maximum context, and historical logit/async controls predate the final sampler widening. Their relevant model/cache geometry and greedy semantics are unchanged; final narrow-path identity, final-source adapter checks, and final serving gates support scoped reuse. I request no blanket reruns.
- The inherited resumed-prefill PCC limitation, serialized prefill latency, lack of automatic recurrent-state prefix caching, limited supported hardware for the native64 draw, and unrepaired admission-sensitive reproducibility remain documented boundaries for subsequent stages.
