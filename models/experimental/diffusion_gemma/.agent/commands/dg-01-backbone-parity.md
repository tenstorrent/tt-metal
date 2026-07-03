---
description: DiffusionGemma stage 01 — validate the reused Gemma-4 backbone against the DiffusionGemma checkpoint (weight mapping + causal PCC on QB2).
---

Load `diffusion-gemma` first. This is the FOUNDATION stage (#47461 / #47468). The backbone is the in-repo Gemma-4 26B-A4B MoE at models/demos/gemma4/ — do NOT re-author it and do NOT edit it. This stage points the existing DiffusionGemma4Model (models/experimental/diffusion_gemma/tt/model.py, weight_mapping.py) at the DiffusionGemma checkpoint and validates the causal (encoder/prefill) pass, before any diffusion delta is exercised. Work only under models/experimental/diffusion_gemma/.

Goal completion requirements:
- weight_mapping.remap_state_dict loads the real DiffusionGemma checkpoint into the gemma4-keyed backbone state + the self-conditioning state, with every missing/renamed key and config diff (v-norm, K=V for full-attn layers, canvas_length, dual RoPE, softcap) reconciled and recorded. `git diff main -- models/demos/gemma4/` is EMPTY.
- Causal (prefill/encoder) HF-vs-TTNN PCC is validated on QB2 on the real checkpoint per meaningful layer kind (sliding vs full attention). Record measured PCC with exact commands. Use the diffusion decision-fidelity framing from `diffusion-gemma`: also record the causal argmax-agreement-vs-HF baseline (the ≈50% figure that #48291 tracks), since diffusion commits the clean argmax.
- The torch/HF reference (reference/hf_reference.py) and the PCC harness are exercised and their provenance recorded.
- models/experimental/diffusion_gemma/doc/context_contract.json records hf_advertised_context (262144), the current validated causal prefill context, and any hard-physical-limit reduction with byte/probe evidence. It parses under models/experimental/diffusion_gemma/.agent/scripts/check_context_contract.py.
- At least one real-weights QB2 causal test passes and evidence shows it. Watcher-clean run recorded (or the exact env failure + replacement evidence).
- No torch / ttnn.from_torch / ttnn.to_torch / host fallback inside a single measured prefill pass (setup/test boundaries excepted).
- doc/backbone_parity/README.md and work_log.md record commands, PCC, argmax-agreement baseline, weight-mapping reconciliation, limitations, exact artifacts.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit stage-owned touched changes under models/experimental/diffusion_gemma/ (no Co-Authored-By trailer); never push; never touch models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
