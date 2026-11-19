# Stage 7 review contract

Model: `Qwen/Qwen3.8-Flash-Next`. Stage: optimized-full-model. Start from the
completed full model in this checkout. Do not begin vLLM integration.

The user explicitly authorizes subagents when a selected skill requests or
recommends them. Review the live worktree against the original stage contract:

- Optimize the complete model and generator across the target mesh: embeddings,
  norms, LM head, logits, sampling, cache, trace replay, collectives, residual
  layouts, host boundaries, and orchestration.
- Preserve canonical split-sampling token output: sampler-ready logits,
  `tt_out_tok` feedback, device-side position/RoPE advance, changed-only page
  tables, and traced greedy plus top-k/top-p sampling.
- Preserve explicit cache/page-table/position/prompt-length/batch state, mixed
  prompts, fixed slots, and inactive rows. Greedy decode stays on device.
- Benchmark correct greedy choices for this mesh. Fix a malformed generic
  sampled path rather than selecting force-argmax for that reason; otherwise
  leave a reproduction.
- Preserve the decoder stack's selected dtype, fidelity, KV, activation and CCL
  policies, rejection ledger and inter-layer residual layout. Do not reinstate
  rejected policies or a replicated residual stream. No broad datatype frontier
  search: stage 8 owns Pareto selection.
- Preserve or improve `doc/context_contract.json`; reduce capability only for
  a proven hard physical limit with the largest feasible value. Run a 32K–64K
  smoke; rerun maximum context only if KV dtype/layout changes.
- Preserve non-aligned prompt lengths through the optimized generator.
- Refresh AIME24 chat-template prefill, teacher-forcing decode and autoregressive
  evidence. Prefill and teacher forcing require top-5 >=98% and top-100 =100%.
- Put before/after warmed batch-1 TTFT and trace-verified decode tokens/s/user at
  the top of README. Keep traced teacher forcing separate from token-out timing
  that includes sampling, feedback and readback.
- Compute the decoder stack budget from optimized multichip layer latencies.
  Token-out uses persistent token/position/RoPE/page-table inputs, nonblocking
  replay, no per-token host synchronization/readback, and applicable persistent
  buffer/CCL optimizations. If token-out exceeds stack plus terminal work by
  >10–15%, split and close the largest avoidable gap.
- If force-argmax, full-vocabulary all-gather, generic TopKDeviceOperation or
  another sampler operation dominates, fix the head/sampling contract.
- Leave full-path tt-perf-report tables, CSV and provenance logs. Applicable
  multichip/optimize checklist items need sharding, CCL, matmul, config and
  kernel evidence. Stop optimization under the skill Stop Rule; untried items
  are handoff notes. The Stop Rule does not waive required gates.
- Runtime fallback audit is clean for the measured fully traced split-sampling
  token-out path, without an interior host boundary.
- README and work log record commands, before/after accuracy and performance,
  mesh decisions, profiling conclusions, limitations and exact artifacts.
- Obtain fresh independent stage-review: clean-pass, or pass-with-carried only
  after three rounds with the skill's conditions. Fix and rereview findings.
  Commit stage-owned changes locally and record SHAs. Never push.
- Unmet requirements, failed gates and review findings remain work. Stop only
  after AutoFix fails.

Selected installed skill package:
`/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24`.
Read `skills/multichip/SKILL.md`, `skills/optimize/SKILL.md`,
`skills/tt-device-usage/SKILL.md` and `skills/stage-review/SKILL.md`, plus applicable
linked references. The optimize FUNCTIONAL floor remains required: batch 1,
128 input/128 output, at least 10% of theoretical. The user's explicit policy
preservation overrides the skill's generic starting precision advice.

Reviewer rules: read-only, no TT device use, no servers or long tests. Treat
reports as claims and rederive important numbers from raw artifacts. Read actual
generated text. Report concrete required work, concerns, hard-check gaps,
anomalies, inspected scope and residual risk in the stage-review format.

Latest user wrap-up instruction: "Stage time/token cap reached. Keep the best version that passes the required gates, run only the remaining required gates once, write untried ideas as hand-off notes in the stage work log, and complete the stage (pass-with-carried is acceptable for non-blocking items)." No new optimization or optional hardware experiments are permitted.

## Owner amendment, 2026-11-16

Owner decision: accept the current decode speed (about 9.7 tok/s/user vs the 15.6 FUNCTIONAL floor) as a known limitation for this run. Record it in limitations with this decision and the evidence, finish the review and complete stage 7. No further tuning for this floor.

This explicit owner instruction supersedes only the stage7 performance-floor acceptance blocker for this run. Keep the failed numerical result and original review on disk; record the accepted limitation and finish independent follow-up review. All other requirements remain unchanged. Decision: `owner_decision_2026_11_16.json`; limitations: `../eval_readiness/limitations.json`. The original goal prompt copy is not the source of the later owner decision. Supplied datatype-sweep/full-model/tracing skills remain reference contracts; this closure does not start stage8 or vLLM.
