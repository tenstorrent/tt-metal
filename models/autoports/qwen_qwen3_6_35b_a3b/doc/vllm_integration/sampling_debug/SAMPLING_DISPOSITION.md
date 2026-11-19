# Sampling disposition for the measured stage-9 path

2026-11-17, source/artifact review only. This is a sampling classification, not independent stage review or stage acceptance. No tests, server requests, or production changes were made.

**The evidence is sufficient to classify the residual sampling assertion as reproducibility-only and to document a demonstrated admission-dependent routing defect. It does not justify rewriting raw FAIL as PASS or granting an unconditional stage waiver.** The skill exception removes the need for further serving-correctness investigation of a reproducibility-only assertion once its other prerequisites are satisfied; separate quality/gate findings retain their own disposition.

## Preserved results and exact attribution

- Full profile: **67 passed, 74 canonical skips, 1 failed**, pytest/runner exit 1, 2299.30 seconds. Preserve `readiness_vllm/sampling_tests.log`, `sampling_execution.json`, and `sampling_final_summary.json`. The only assertion failure is `test_mixed_params_batch`, comparing two seeded strings. In that historical run it stopped at `TopP: `, seed 7. No per-request admission trace exists for that historical failure; its individual cause remains unproven.
- The targeted canonical-config pair remains **FAIL**, with nine exact text/token matches and one unequal `Random: ` case. `mixed_route_analysis_legacy_parsed.json` attributes all 20 requests and 18 decode submissions without missing/ambiguous evidence. `Random: ` used host prefill plus four host decodes in batch 0; device prefill plus four host decodes in batch 1. First unequal token is output index 0, IDs **95772 versus 102038**. Different first-token samplers precede the later observed device-to-host transition.
- Actual server admission, not client submission order, supports this classification: `mixed_route_server/server.log:2757` places batch-0 `Random: ` with penalty-bearing companions on host; line 2796 admits batch-1 `Random: ` alone on device; lines 2798 and 2801–2802 show penalty companions arriving and subsequent host decode. This demonstrates the routing mechanism in the same canonical test family. It does not retroactively recover the old `TopP: ` timeline; that case matches exactly, host throughout, in the new pair.

## Evidence for the exception's prerequisites

| Prerequisite | Evidence and limit |
| --- | --- |
| Sampling correctness beyond reproducibility | All remaining full-profile tests pass: top-k membership/greedy checks, penalties, host-only controls, structured/plain coexistence and reused slots, plus other seed tests. This is not a claim that every possible distribution is proven correct. |
| Logprobs | All 20 parameterized logprob tests and full-vocabulary chat-logprob test pass. `logit_determinism/http.json` has 12 passing requests with exact exposed top-20/emitted-token logprobs versus standalone, max absolute difference 0. |
| Repeated/cross-position logits | `logit_determinism/standalone.json`: eight full-vocabulary exact cases across B2/B32 and reordered rows. HTTP controls exercise repeated/reordered requests, but expose top 20 only and use explicit host compatibility. These are representative controls, not an exact-prefix logit capture for historical `TopP: ` or new `Random: `. |
| Serving state and crash freedom | `sampling_execution.json` records no runtime errors for the completed full suite. `full_buckets_reader_drain/lifecycle_summary.json` passes 16 checks, including identical repeated greedy tokens, overlapping decode, allocator page growth, preserved token/position state and stable traces. `async_control_comparison.json` records exact async/nonasync outputs. This does not erase earlier repaired hangs or claim arbitrary-duration reliability. |
| Standard qualitative acceptance | `qualitative_verdict.json` reviews all 12 shared outputs: coherent/on-topic, no mechanical loops, gibberish, language drift or request contamination, with documented 256-token truncation. All six greedy texts match standalone TT before its first EOS marker. Status remains `PASS_WITH_TOKEN_CAP_LIMITATION`. |

The logit controls are versioned historical evidence. Their `tt/serving.py` and `tt/model.py` hashes match the reviewed current files, while generator/adapter hashes differ. Do not silently describe them as a fresh exact-source test of any newly wired sampler. Later lifecycle/native controls support the recorded serving path; a material sampler change needs its own affected validation.

## Constrained-quality control: attribution is complete, acceptance is separate

`sampling_controls/native_card_all10_attribution.json` and `native_card_all10_quality_review.json` establish exact equality of all ten native full-model and HTTP trajectories, **4620 tokens**, including all six quality defects and both loops. Thus no added serving regression is demonstrated for this corpus. This resolves the prior attribution gap; it is not evidence that the outputs became good.

The frozen HF comparison still fails: HF 2/10 versus TT 6/10 quality failures, 40 percentage points above HF against a 20-point margin. It is neither an HF-matched known limitation nor an automatic carried item. The skill's literal prerequisite that qualitative output pass cannot be expanded into “every diagnostic passed” here. For narrow serving-regression assessment, the standard suite passes and the additional defects are exactly inherited from the full model; independent stage review must explicitly dispose of that inherited full-model finding. Sampling classification alone cannot provide that disposition.

## Decision and boundaries

Under `vllm-integration/SKILL.md:257–261`, this is the named reproducibility-only case with a real documented host-compatibility boundary and observed routing evidence. There is no basis to demand a broad sampling rerun solely to force the historical 67/74/1 count to become 68/74/0, or to manufacture a pass by forcing all seeds to host, suppressing penalties, changing top-k declarations, or weakening the assertion. The model-owned penalty feasibility/remaining resume-history boundary is documented in `DEVICE_PENALTIES_DESIGN_AUDIT.md`.

**What remains missing for unconditional acceptance is not another admission trace:** it is the explicit review disposition of the still-failed inherited HF-relative quality finding and satisfaction of the other stage gates. The user's literal runner-success requirement also cannot be represented as met by changing an exit code or summary label. Keep measured runner FAIL and the reproducibility-only classification side by side for review/gate interpretation. No blanket waiver is awarded by this report.

The isolated native-64 sampling fixtures are separate experimental evidence. They neither repair this penalty-induced route transition nor validate a final canonical serving integration. If that implementation is adopted, rerun only its affected gates first; the present disposition must not be used to excuse new correctness, logprob, crash or qualitative regressions.
