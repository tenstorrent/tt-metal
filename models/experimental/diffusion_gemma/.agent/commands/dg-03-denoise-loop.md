---
description: DiffusionGemma stage 03 — validate the discrete-diffusion denoise-loop control flow and keep it trace-safe.
---

Read `models/experimental/diffusion_gemma/.agent/skills/diffusion-gemma/SKILL.md` and `models/experimental/diffusion_gemma/.agent/skills/tt-enable-tracing/SKILL.md` first. This stage covers the discrete-diffusion decode loop (#47463): per-step entropy → Gumbel-max → entropy-budget acceptance (sort → cumsum → scatter/inverse-permutation) → renoise → halt. The loop exists (tt/denoise_forward.py, tt/denoise_loop.py). The load-bearing risk is data-dependent control flow vs Metal Trace. Do NOT resolve the #48291 fidelity bar here — this stage proves the control flow runs correctly and remains trace-safe.

Goal completion requirements:
- The entropy-budget accept path (sort → cumsum → scatter-back) is validated on synthetic and reference logits vs reference/denoise_loop.py: entropy values, accept/renoise masks, and the scatter-back/inverse-permutation mapping to original canvas positions.
- Every captured trace has fixed shapes and operations, with an on-device tensor cutoff mask, tensor-valued scatter/gather indices, and exact-shape program-cache warmup. Validate both the shipping fixed-48 replay and the landed opt-in early-halt controller, which replays a one-step/window trace and reads one halt scalar between replays. Record its no-halt overhead and #48291 behavior; never branch inside capture.
- A canvas-feedback trace test proves the accepted canvas from step N is the persistent trace input consumed by step N+1 with no host reconstruction of accepted tokens or the entropy cutoff.
- If any op (sort/cumsum over the 256 axis) triggers a mid-capture recompile / write / host sync, it is diagnosed with the `tt-enable-tracing` flushed-marker technique and fixed or documented with a minimal repro; use `autofix` for tricky failures.
- Watcher-clean traced run recorded; determinism across repeated replay proven with injected reference noise.
- `models/experimental/diffusion_gemma/doc/denoise_loop/README.md` and `work_log.md` record commands, decision agreement, trace scheme, halt-scalar cost, limitations, and exact artifacts.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); then push — invoking this stage command explicitly authorizes both actions; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
