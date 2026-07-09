---
description: DiffusionGemma stage 06 — validate the QB2 256K memory budget and long-context fit on the (1,4) mesh.
---

Read `models/experimental/diffusion_gemma/.agent/skills/diffusion-gemma/SKILL.md`, `models/experimental/diffusion_gemma/.agent/skills/multichip/SKILL.md`, and `models/experimental/diffusion_gemma/.agent/skills/tt-device-usage/SKILL.md` first. This is the HW-enablement stage (#47487). The backbone is already TP=4 on QB2 (P150x4) via the untouched gemma4 CCL/expert sharding — do NOT re-plan or re-shard the backbone and do NOT edit models/demos/gemma4/. This stage sizes and validates the FULL diffusion memory budget and long-context fit on the (1,4) mesh.

Goal completion requirements:
- The QB2 memory budget (models/experimental/diffusion_gemma/QB2_MEMORY_BUDGET.md, memory_budget.py) accounts for ALL diffusion terms, not just weights + persistent KV: the per-step canvas K/V scratch zone (#47474 storage class ii) and the non-causal long-context mask buffers (#47462), plus trace/activation reserve. A clean short-prompt causal run does NOT count as 256K de-risking — state that explicitly.
- The largest feasible context for prompt+generated on QB2 is measured with a byte calculation AND a capacity probe (the near-262K single-shot prefill is a known hang/cleanup risk — reset the board between attempts and record the exact failure signature rather than silently reducing context). doc/context_contract.json records hf_advertised_context (262144), the largest validated context, and any hard-DRAM reduction with evidence.
- The mesh plan for the diffusion-specific tensors (canvas activations, canvas K/V scratch, [P+C] concatenated KV per layer, non-causal mask) is recorded with per-device shapes on the (1,4) mesh; it preserves the backbone's inter-layer residual/CCL contract without modifying the backbone.
- Long-context denoise attention (the >32768 non-causal path) is validated to fit and run at the target context on QB2; batch=1 first, then the largest batch that fits given context (batched decode (#47557) is out of scope for this 10-stage pipeline — record the batch ceiling here).
- Watcher-clean run; runtime fallback audit; Blackhole/P150x4 fabric config confirmed.
- `models/experimental/diffusion_gemma/doc/qb2_longcontext/README.md` and `work_log.md` record the budget breakdown, byte math, capacity probes, mesh shapes, batch ceiling, limitations, and exact artifacts.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); then push — invoking this stage command explicitly authorizes both actions; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
