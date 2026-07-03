---
description: DiffusionGemma stage 03 — validate the discrete-diffusion denoise-loop control flow and keep it trace-safe.
---

Load `diffusion-gemma` first. This stage covers the discrete-diffusion decode loop (#47463): per-step entropy → Gumbel-max → entropy-budget acceptance (sort → cumsum → scatter/inverse-permutation) → renoise → halt. The loop exists (tt/denoise_forward.py, tt/denoise_loop.py). The load-bearing risk is data-dependent control flow vs static Metal Trace. Do NOT resolve the #48291 fidelity bar here (that is the correctness stage) — this stage proves the control flow RUNS correctly and stays trace-safe. Work only under models/experimental/diffusion_gemma/.

Goal completion requirements:
- The entropy-budget accept path (sort → cumsum → scatter-back) is validated on synthetic and reference logits vs reference/denoise_loop.py: entropy values, accept/renoise masks, and the scatter-back/inverse-permutation mapping to original canvas positions.
- The loop is trace-safe by construction: a FIXED step budget (always run the ≤48 max), an on-device tensor mask for the cutoff (no host branch, no variable-length slice), tensor-valued index arguments for scatter/gather (device-resident indices), and program-cache warmup for sort/cumsum/scatter/gather/entropy at the exact fixed canvas shape and argument values. Any data-dependent early-halt is expressed as an on-device mask or a bounded host readback (≤256-element), with the cost of a per-step host readback explicitly measured against the trace budget (the loop runs ≤48×/block).
- A canvas-feedback trace test proves the accepted canvas from step N is the persistent trace input consumed by step N+1 with no host reconstruction of accepted tokens or the entropy cutoff.
- If any op (sort/cumsum over the 256 axis) triggers a mid-capture recompile / write / host sync, it is diagnosed with the `tt-enable-tracing` flushed-marker technique and fixed or documented with a minimal repro; use `autofix` for tricky failures.
- Watcher-clean traced run recorded; determinism across repeated replay proven with injected reference noise.
- doc/denoise_loop/README.md and work_log.md record commands, decision-agreement, trace-safety decisions (fixed budget, device mask, tensor indices), any host-readback fallback with cost, limitations, exact artifacts.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); never push; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
