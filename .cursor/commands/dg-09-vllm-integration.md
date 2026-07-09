---
description: DiffusionGemma stage 09 — block-granular serving through the tenstorrent/vllm TT plugin.
---

Read `.cursor/skills/diffusion-gemma/SKILL.md`, `.cursor/skills/vllm-integration/SKILL.md`, and `.cursor/skills/qualitative-check/SKILL.md` first. This is the SERVING stage (#47466 / #47488). Serve via the tenstorrent/vllm TT plugin fork. The whole denoise loop lives inside the model and emits a 256-token block per step. Do NOT edit models/demos/gemma4/.

Goal completion requirements:
- A DiffusionGemma vLLM adapter (models/experimental/diffusion_gemma/tt/generator_vllm.py) delegates to tt/generate.py low-level methods and reuses the on-device canvas sampling path — no separate host sampling / full-logits readback path. The tt-metal model owns forward + attention + KV; the runner passes only tokens/page_table/kv_cache/start_pos/prompt_lens/sampling.
- The block-granular emission contract is explicit and validated for block 0 plus a subsequent block. The scoped runner/scheduler patches accept N-token output and advance state by 256. Keep `supports_async_decode=False` until a separately tested per-block async contract exists; do not adapt per-token stale-input logic.
- Registered in the tenstorrent/vllm TT plugin's register_tt_models() with a diffusion-appropriate architecture name (copy the gemma4 TT bridge as the template). The plugin's spec-decode block / chunked-prefill-unsupported / phase-based-batching / APC-off facts are documented in the stage evidence.
- Served max_model_len matches `models/experimental/diffusion_gemma/doc/context_contract.json`; serving accepts non-aligned prompt lengths.
- Serving metrics are reported per-block, not per-token TPOT: prefill TTFT, per-block latency, tokens-per-block throughput; do NOT report 1000/mean_tpot_ms. Qualitative outputs judged via `diffusion-gemma` / `qualitative-check` with an HF-vs-TT control (RUN-first: degenerate output may still be expected until #48291).
- Runtime fallback + process cleanup audited (no leftover vLLM/EngineCore processes holding devices).
- `models/experimental/diffusion_gemma/doc/vllm_integration/README.md` and `work_log.md` record server command, TT config, max model len, block contract, per-block metrics, qualitative verdict, limitations, and exact artifacts.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); then push — invoking this stage command explicitly authorizes both actions; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
