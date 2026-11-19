# Qwen/Qwen3.6-35B-A3B TTNN autoport

**Warmed TTFT: 2919.990 ms; trace-verified batch-1 caller-observed decode:
39.79582 t/s/u (25.12827 ms/token)** on 128 input / 128 output tokens.
Queued device production: 39.86941 t/s/u. True HF teacher forcing to resident
logits: 40.64963 t/s/u over 99 steps after a 161-token prefill, excluding sampling
and readback. [Final isolated performance](doc/full_model/performance/lifetime_fixed.json)
passes with on-device traced greedy sampling and resident token feedback.

**Stage 6 complete under the owner-revised contract; [independent review](doc/full_model/stage_review_round2.md): clean-pass.** Final-source all-40-layer
[prefill](doc/full_model/prefill_accuracy_lifetime_fixed.json) and
[teacher-forcing](doc/full_model/teacher_accuracy_lifetime_fixed.json) accuracy
both pass at 98% top-1, 100% top-5 and 100% top-100 over 100 predictions each.
Batch-32 capability passes in the recorded probe. Resumed prefill fails three
PCC checks; the owner accepts this HF-reproduced numerical sensitivity as a known limitation. All seven TT top-1 checks and ten applicable negative controls pass. Raw failures and the unsuccessful AutoFix report are preserved. All-40-layer
maximum-context acceptance **passes at 262144 tokens**, with no capability
reduction: 262143-token prefill plus one traced decode at the final valid position.
[Capacity evidence](doc/full_model/context_acceptance/final.json) records
5883.448 s prefill and 33.730 ms decode; these are capacity diagnostics, not
headline performance. [Reduced generator Watcher](doc/full_model/watcher/summary.json)
passes on real layers 0/3, with 90 clean dumps across all four chips; this is
not an all-40-layer Watcher run. Native profiler v3 was rejected for marker loss;
[V4 native profiling](doc/full_model/performance/profile_reduced_v4_drains/component_summary.json)
passes, with 32,896 native executions and 16 sessions per rank/component.
The [owner decision](doc/full_model/owner_decision.md) resolves the previous continuation-contract blocker. The serving inventory now records 56 supported and 30 explicitly unsupported rows, with no missing rows. Automatic KV-only prefix caching remains disabled for recurrent state. No further device work on the resumed-prefill gate is requested.

The accepted optimized multichip decoder runs on a 1x4 Blackhole mesh. Its
layer-stack lower bound is 24.9935 ms/token, excluding the full-model terminal
and generator work. The owner-accepted stage-5 linear latency limitation is
preserved in the decoder evidence.

See [full-model work](doc/full_model/README.md),
[work log](doc/full_model/work_log.md), and
[accepted decoder](doc/optimized_multichip_decoder/README.md).
No vLLM integration work has begun.

The prior [round1 review](doc/full_model/stage_review.md) is retained as history; the independent round2 review closes its sole blocker under the owner decision. Local acceptance SHAs are recorded in the work log.
