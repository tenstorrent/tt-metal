# Stage 9 owner decisions — 2026-11-17

Source: Vladan, standing overnight instruction relayed by the hub; user message and `/home/ttuser/qwen36-35b-a3b/multigoal_logs/09-09-vllm.resume-6.message.txt`.

## Verbatim owner instruction

Owner decisions for the two items in doc/vllm_integration/owner_decisions_pending.md (Vladan, standing overnight instruction relayed by the hub, 2026-11-17):

1. Constrained-thinking quality: ACCEPTED as a recorded known limitation for closing stage 9, exactly as scoped in owner_decisions_pending.md (that prompt, constraints, sampling recipe, 1024-token cap, seeds 0-9; failed scores kept; numerical localization deferred). The owner will investigate precision separately after the run.
2. Default chat behavior: do NOT change the serving default. Restore the checkpoint default (thinking on; requests that omit enable_thinking get the model's normal thinking behavior) and run the affected validation. The direct-answer default is not approved. The owner can revisit this in the morning.

Record both decisions with this source, run the diff-only stage review round 2, commit locally and complete stage 9. Stage cap is loosened for this run.


## Application

The constrained-thinking limitation is accepted only in the scope named in `owner_decisions_pending.md`. Failed HF/TT scores and raw outputs remain unchanged; this does not assert HF-equivalent quality. Numerical localization is deferred to the owner.

The direct-answer server default is rejected. The launcher no longer overrides checkpoint chat-template defaults. Requests without `enable_thinking` retain normal thinking; explicit request overrides remain supported. Affected validation and diff-only review round 2 follow. Existing completions-endpoint benchmarks, maximum-context and P5 evidence are unchanged and will not be repeated.
