---
description: DiffusionGemma stage 04 — harden the end-to-end block-diffusion RUN on QB2 (RUN-first; degenerate output OK).
---

Read `models/experimental/diffusion_gemma/.agent/skills/diffusion-gemma/SKILL.md` and `models/experimental/diffusion_gemma/.agent/skills/full-model/SKILL.md` first. This is the FUNCTIONAL e2e RUN stage (#47464): prompt → prefill → multi-block (denoise → commit → advance 256) → detokenized text on QB2 with real 26B weights. The entry path exists (tt/generate.py::generate_text / generate_text_from_checkpoint_state; demo/text_demo.py). RUN-FIRST: degenerate/EOS-heavy output is ACCEPTABLE for this milestone — do NOT gate on text quality or #48291 fidelity. Do NOT begin datatype-sweep or vLLM work.

Goal completion requirements:
- The RUN is reproducible on QB2: short-prompt and long-prompt (>768 tok, maskless denoise path) multi-block runs each commit ≥2 blocks and detokenize, exiting 0. Grep DG_TEXT_DEMO_SUCCESS / DG_TEXT_DEMO_FAILURE for outcome (the RUN-first denoise path emits expected TT_THROW fallback noise even on success).
- The pinned device-gated regression `models/experimental/diffusion_gemma/tests/test_device_text_demo_run.py` passes on QB2 for short- and long-prompt variants and asserts parsed `generated_tokens`, `blocks`, `prompt_len`, and `next_pos`.
- Per-block position/RoPE advancement (block N at prompt_len + N·256) and commit-append are exercised across ≥2 blocks. Commit-append uses DiffusionGemma-local code (`tt/commit_decode.py`), and the shared-directory gate passes with the actual `DG_BASE_REF`.
- Any footprint work needed to fit the real denoise→commit block on the (1,4) mesh is DiffusionGemma-local and gated; the shared backbone is byte-for-byte unchanged. If a fit reduction is required, record the byte calculation / capacity probe and the largest feasible value in doc/context_contract.json.
- The generation entry validates length/block/canvas/vocab/logits inputs; host-seeded noise + bounded host readback are acceptable for the functional run (production device RNG and intra-block device early-stop may remain deferred, recorded as such).
- Runtime fallback audit for the measured path; watcher discipline recorded.
- `models/experimental/diffusion_gemma/doc/block_diffusion_run/README.md` and `work_log.md` record commands, prefill TTFT, per-block/denoise-step timing, DRAM post-build/post-prefill, degeneracy note, limitations, and exact artifacts.
- `stage-review` returns clean-pass; findings fixed/rereviewed. Locally commit under models/experimental/diffusion_gemma/ (no Co-Authored-By); then push — invoking this stage command explicitly authorizes both actions; never edit models/demos/gemma4/; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `autofix` fails.
