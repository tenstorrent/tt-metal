# Stage7 original scope and review contract

Model: `Qwen/Qwen3.6-27B`, current tt-metal checkout, starting completed full
model commit `7dde27883c7cf72ed1268a9ef5845b19134549c3`. Target Blackhole TP4.
The user selected `$multichip`, `$optimize`, `$tt-device-usage`, authorized
skill-requested subagents, and explicitly excluded vLLM integration.

User completion requirements:

- Optimize the complete model/generator path: embeddings, norms, head, logits,
  sampling, cache, trace replay, collectives, residual layouts, host boundaries
  and generator orchestration.
- Preserve canonical split sampling: sampler-ready logits, `tt_out_tok`
  feedback, device position/RoPE advance, changed-only tables, traced greedy
  and top-k/top-p-capable sampling. Benchmark correct greedy choices, not a
  malformed sampled path versus force-argmax.
- Preserve explicit cache/page/position/prompt-length/batch state, mixed prompts,
  fixed slots and inactive rows. Preserve nonaligned logical prompt lengths.
- Preserve the selected decoder dtype/fidelity/KV/activation/CCL policy,
  rejection ledger and inter-layer sharded residual. No rejected faster policy
  or replicated residual stream; no broad datatype frontier here.
- Preserve or improve `doc/context_contract.json`; reduce capability only for a
  hard physical limit with evidence and largest feasible value.32K–64K smoke;
  max retest only for changed KV dtype/layout.
- Refresh chat-template AIME24 prefill, teacher forcing and autoregression;
  run_prefill_check and run_teacher_forcing top5>=98%, top100=100%.
- README starts with before/after warmed B1 TTFT and trace-verified decode t/s/u.
  Separate teacher-forcing, logits-only and token-out boundaries, including
  sampling, feedback and readback when present.
- Compute optimized layer-stack lower bound. Token-out uses persistent inputs,
  nonblocking replay, no per-token host read/sync, enabled persistent CCL.
  Close largest avoidable gap if>10–15% over stack plus terminal work.
- Fix a dominant force-argmax/full-vocab gather/generic TopK sampler contract
  before completion. Leave full-path tt-perf-report tables, CSV and provenance.
- Applicable multichip/optimize checklist evidence covers sharding, CCL,
  matmuls, program configs and kernels. Stop by optimize's checklist, last two
  kept gains<3%, or wall cap; untried items are handoffs, never correctness waivers.
- Runtime fallback audit clean; token-out fully traced with split sampling and
  no host boundary inside the resident loop.
- README/work_log record commands, before/after accuracy/perf, mesh decisions,
  profiling conclusions, limitations and exact artifacts.
- Fresh independent stage-review clean-pass, or pass-with-carried after3 rounds;
  fix/rereview findings. Local stage-owned commits and SHA receipts; never push.
- Unmet requirements or failed gates are work. Use AutoFix for nonobvious or
  persistent failures rather than stopping at a disclosure.

Applicable installed skills (root
`/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills`):
`multichip/SKILL.md`, `optimize/SKILL.md`, `tt-device-usage/SKILL.md`,
`tt-enable-tracing/SKILL.md`, `qualitative-check/SKILL.md`,
`stage-review/SKILL.md`; preserve `full-model/SKILL.md` contracts.

Reviewer works read-only against the live workspace. No hardware, services,
server startup, reset or implementation edits. Treat this directory's prose as
claims and rederive from code, raw logs, JSON/CSV and generated text. Review only
after all hard gates finish. Reporting a missing/contradicted gate means work.
