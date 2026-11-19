# Stage 9 owner decisions pending

Status: **resolved by explicit owner decisions on 2026-11-17**; see `owner_decisions_20261117.md`. The text below preserves the original decision requests. Independent review round1 is `more-work-needed`. This record makes the two existing async questions durable; it is not a new waiver, new plugin interface proposal, or authorization.

## 1. Inherited constrained-thinking quality

The frozen matched ten-seed comparison reports HF2/10 versus TT6/10 secondary failures, including two sustained TT loops. All ten served token arrays and finish reasons, totaling4620tokens, exactly match standalone TT using the selected precision. This localizes the captured behavior to the full model; it does not resolve the failed HF comparison. AutoFix has not produced a justified verified repair. Native equality, successful transport gates and a wall-cap extension do not accept this defect.

Decision requested: explicitly accept **this recorded constrained-thinking limitation for closing stage9**, keeping the failed scores and deferring full-model numerical localization; or direct continued repair before completion. Any acceptance is limited to the recorded prompt, constraints, sampling recipe,1024-token cap andseeds0–9. It does not waive other quality failures, bless loops generally, change accuracy thresholds, or claim HF-equivalent quality.

Evidence:
- `stage_review_round1.md`, RequiredWorkP1.
- `sampling_debug/QUALITY_AUTOFIX_CLOSURE.md` and `QUALITY_CONTROL_CONTRACT_SUPPLEMENT.md`.
- `../../readiness_vllm/sampling_controls/card_unblinded_summary.json` and the raw HF/HTTP outputs beside it.
- `../../readiness_vllm/sampling_controls/native_card_all10_attribution.json` and `native_card_all10_quality_review.json`.

## 2. Default chat behavior

The implemented and measured candidate launcher supplies `--default-chat-template-kwargs '{"enable_thinking":false}'`. Requests omitting that option therefore receive direct answers. Explicit thinking requests remain available and pass P3. The checkpoint ordinarily defaults to thinking on. Temperature1/top-k20/top-p.95 checkpoint sampling defaults are unchanged.

Decision requested: approve **this exact direct-answer serving default**, or restore the checkpoint thinking-on default and validate the affected sampling/qualitative behavior. The candidate has complete final full-sampling/shared-suite/performance evidence; no server is currently running. Approving this choice alone does not accept the separate quality finding above.

Evidence:
- `../../tests/run_vllm_server.sh` and `../../readiness_vllm/final64_performance_server/launch_record.json`.
- `stage_review_round1.md`, RequiredWorkP2.
- `../../readiness_vllm/final64_gate/execution.json`, fullsampling68passed/74skipped/0failed.
- `../../readiness_vllm/qualitative_verdict.json` and `../eval_readiness/chat_conformance.json`.

## Remaining sequence

Record the owner's exact decisions and their source. Resolve any unaccepted findings. Obtain a fresh diff-only stage-review round2 after required affected work finishes. A passing review precedes local tt-metal stage checkpoints and logged SHAs. Never push. Existing maximum-context/P5 acceptance and unaffected runtime results remain valid; neither is queued for a repeat.

The current quality AutoFix failure permits stopping under the original goal, but does not permit marking the goal complete. No answer or approval is inferred from automatic continuation messages or elapsed time.
