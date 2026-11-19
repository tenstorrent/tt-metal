# Stage 6 review contract

Target: `ornith-ai/Ornith-1.5-9B`.
Autoport: `models/autoports/ornith_ai_ornith_1_5_9b`.
Base checkout: main, `c2254f91c38840cd4069a1ecbbd3f4874e27d557`.
Review the live worktree. Stage changes are uncommitted until a passing review;
local checkpoint commits follow, never a push. No vLLM implementation or server
is authorized in this stage. Read-only pinned plugin inspection is required.

## Original user completion contract

- Complete `tt/model.py` and `tt/generator.py` for the full HF autoregressive
  path and standard Metal readiness generator contract.
- Preserve the completed optimized multichip decoder strategy, dtype/fidelity,
  activation/CCL policy, paged KV storage/update dtype split, rejection ledger,
  and inter-layer residual layout; no single-chip, replicated projection,
  host-compute, or less-optimized fallback.
- Recompute full weights plus KV DRAM in `doc/context_contract.json`; reduce
  advertised capability only for a proved hard physical limit and record the
  largest feasible value.
- Accept valid prompt lengths through the supported context, including
  non-aligned lengths. Public generation owns padding, masks, cache fill,
  positions and logical slicing.
- Expose serving-ready prefill/decode with explicit cache, page table, position,
  prompt lengths, batch state, mixed prompts, fixed slots and inactive rows.
- `serving_contract.json` maps all pinned plugin calls, capabilities and tested
  request features. Missing rows are stage work.
- Generate an AIME24 original-tokenizer chat-template reference, 100 tokens,
  using `--model-dir` and exact-match cache reuse only.
- Prefill and teacher forcing report top-1/top-5/top-100. Required top-5 >=98%,
  top-100=100%. Read HF and TT autoregressive outputs for coherence, repetition,
  wrong language and early divergence.
- Compare both common samplers, choose one and record rejection. Custom sampling
  mathematics only if neither fits. Greedy must remain semantically greedy.
- Traced split sampling writes device tokens back to the persistent decode
  input. No host argmax, full-logit readback, untraced decode sampling or Python
  token-feedback loop in the measured device path. Explicit host compatibility
  is available separately.
- Resumed prefill passes; prefix caching is recorded per full-model rules.
  Trace evidence proves feedback, position coherence, changed/unchanged page
  handling and zero per-token host reconstruction.
- Top of model README reports full-model TTFT and traced batch-1 decode t/s/u,
  separating teacher-forcing and token-out boundaries when they differ.
- Audit runtime fallback, cache ownership, host-logit boundaries, sampling and
  reset. If terminal sampling dominates, repair before closure.
- Apply matched, blinded original-HF controls before diagnosing content/eval
  failures. Shared HF behavior is a documented model limitation.
- Record exact commands, accuracy, quality, performance, limits and artifacts
  in `doc/full_model/README.md` and `work_log.md`.
- Independent stage-review must pass cleanly, or pass with carried items after
  three rounds; fix and rereview findings. Commit only local stage-owned changes,
  never push, and log SHAs.

## Skills

Installed package root:
`/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23`.
Compare `skills/full-model/SKILL.md`, `tt-device-usage/SKILL.md`,
`tt-enable-tracing/SKILL.md`, `qualitative-check/SKILL.md`, and the applicable
full-model performance requirements in `optimize/SKILL.md`.
Follow `stage-review/SKILL.md` as the independent reviewer.

## Inspection

READMEs and work logs are claims, not evidence. Re-derive important claims from
source, JSON, commands, logs and actual generated text. Inspect the accepted
final artifacts and retained failed controls, not only the final prose. Read
`final_gate_commands.json` and `final_performance_commands.json` for source
identity and completion. The evidence inventory is linked from the stage README.

Read-only review: no device open, reset, server, hardware experiment, long test,
or implementation edit. A report file is authorized. Return the skill's full
verdict format, lead with findings, and record actual scope and residual risk.
