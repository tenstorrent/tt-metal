---
description: DiffusionGemma stage 10 — release handoff (tt-inference-server) + tiered models-CI wiring.
---

Read `models/experimental/diffusion_gemma/.agent/skills/diffusion-gemma/SKILL.md`, `models/experimental/diffusion_gemma/.agent/skills/tti-release/SKILL.md`, and `models/experimental/diffusion_gemma/.agent/skills/models-ci/SKILL.md` first. This is the RELEASE + CI stage (#47489). Evaluate the experimental DiffusionGemma implementation through the TT-plugin server, not a stock implementation. Do NOT edit models/demos/gemma4/.

Goal completion requirements:
- The TTI release workflow (or client-side equivalent against the already-working DiffusionGemma vLLM server) evaluates models/experimental/diffusion_gemma. The run spec / report proves the evaluated code path is models/experimental/diffusion_gemma and does NOT identify stock models/tt_transformers or models/demos. Prefer client/server topology; Docker only as an investigated fallback.
- The eval bar is diffusion-appropriate. meta_ifeval / meta_gpqa_cot are autoregressive instruct-LLM gates — do NOT treat them as mandatory unless a diffusion-appropriate rationale exists; substitute or supplement with a diffusion decision / qualitative bar and record the choice. Under RUN-first, degraded output may be expected until #48291 is resolved — record the current fidelity state rather than forcing a passing eval.
- Benchmark metrics are block-granular (per-block latency, tokens-per-block, prefill TTFT), not per-token TPOT. Context is preserved (never capped to hide a model bug); the intrinsic 256-token output-block granularity is not an alignment failure.
- CI: add DiffusionGemma to tiered models-CI with a representative diffusion unit and `models/experimental/diffusion_gemma/tests/test_device_text_demo_run.py` in the BH QB2 pipeline. Tests do not generate or read a TTNN cache file.
- Recover ARC/ERISC/reset failures before stopping; use `autofix` for release test/spec/API/harness failures, not pure infra. Leave no servers, tmux sessions, or containers.
- `models/experimental/diffusion_gemma/doc/tti_release/README.md` and `work_log.md` record server mode, context, TTI tag/SHA, command, key environment, recovery, report path, pass/fail summary, and implementation-path check.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); then push — invoking this stage command explicitly authorizes both actions; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
