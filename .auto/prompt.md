# Autoresearch: ACE-Step v1.5 — LM planner bring-up + batch>1 support

## Objective
Bring up the 5Hz LM planner ("Songwriter", acestep-5Hz-lm-1.7B) and add batch>1 support to the
ACE-Step TT pipeline, while MAINTAINING all existing PCC gates (CFG e2e, no-CFG e2e, VAE, condition
encoder). The LM planner is a 28-layer causal Qwen3Model (hidden 2048, vocab 217204, tied embeddings)
that generates audio semantic tokens giving the song verse/chorus STRUCTURE (dynamics over time),
which the DiT then fills. `build_lm_planner` already builds it (reuses the Qwen3 text-encoder path),
but it was never wired in and has NO passing PCC — the bf16 forward only reaches ~0.58-0.78 PCC vs
the HF reference because of MASSIVE ACTIVATIONS (absmax ~205) that bf16 mis-represents.

Two deliverables:
  1. **LM planner PCC** (PRIMARY) — raise lm_pcc (MIN over seqs) toward the 0.97 gate. The core
     problem is the massive-activation outlier under bf16. Levers to try: fp32/HiFi4 on specific
     layers, per-layer activation scaling, fp32 accumulate, RMSNorm precision, mixed precision on the
     outlier-heavy final layers. MUST generalize (metric is the MIN across seqs, not a single tuned
     seq) — do NOT overfit to one sequence length.
  2. **batch>1 support** (SECONDARY) — make the TT forward accept batch>1 correctly (batch_pcc: a
     batch-2 forward must match two independent batch-1 forwards at PCC ~1.0). Currently the encoder
     path assumes batch-1 reshapes.

## Metrics
- **Primary**: `lm_pcc` (unitless, HIGHER better) — MIN over seq∈{32,64,128} of TT-vs-HF
  last_hidden_state PCC. Baseline ~0.58.
- **Secondary**: `lm_pcc_mean`, `lm_pcc_seq32/64/128` (per-seq, watch generalization),
  `batch_pcc` (batch-2 == 2×batch-1 correctness; 0 = unsupported/crash, ~1.0 = correct).

## How to Run
`./.auto/measure.sh` — builds the LM planner, outputs `METRIC lm_pcc=...` etc. Runs on the remote
Tenstorrent device (~2-4 min incl the 1.7B build). Uses ssh implicitly (already on the host).

## Files in Scope
- `models/experimental/acestep/tt/model_config.py` — `build_lm_planner`, `_build_qwen3_encoder`,
  `_text_encoder_layer_config`, compute-kernel-config helpers.
- `models/experimental/acestep/tt/text_encoder.py` — `AceStepTextEncoder` (the shared Qwen3 forward:
  vocab embed, causal mask, 28 layers, final norm). LM planner reuses this.
- `models/experimental/acestep/tt/encoder_layer.py` (if present) — the per-layer Qwen3 block
  (attn q/k/v/o + q/k-norm, SwiGLU MLP, pre-norms).
- `models/experimental/acestep/tt/pipeline.py` — wiring the LM planner into generate_song (later).
- New PCC test: `models/experimental/acestep/tests/pcc/test_lm_planner.py`.
- `.auto/measure_lm.py` — the benchmark harness (editable for more signal).

## Off Limits
- Do NOT break any existing gate in `.auto/checks.sh` (CFG e2e 0.888-vs-floor, no-CFG e2e 0.9695,
  VAE decoder, condition encoder, generate_song fidelity). Re-run checks before every `keep`.
- Do NOT touch `models/common/` or `tt_dit` internals.
- Do NOT overfit: no per-seq-length special-casing; the metric is the MIN across seqs.

## Constraints
- All existing checks must pass (`.auto/checks.sh`). A change that raises lm_pcc but breaks any
  shipped gate is a `checks_failed`, not a keep.
- Prefer bf16 (deployment dtype). fp32 activations are allowed ONLY if scoped to the LM planner path
  and justified (the LM is a separate model from the DiT; unlike the DiT-vs-reference case, higher
  precision here should move TOWARD the fp32 reference since the reference IS fp32).
- No new heavyweight deps.

## What's Been Tried
- Baseline (measured 2026-07-07): lm_pcc = 0.58 (seq32=0.647, seq64=0.576, seq128=0.776).
  ref_absmax ~90-205 (massive activations confirmed). batch_pcc unknown (untested).
- The DiT-vs-reference precision lessons DO NOT directly transfer: there, higher precision moved AWAY
  from the fp32 reference (bf16-vs-bf16 regime). Here the reference IS fp32 last_hidden_state, so
  raising TT precision should move TOWARD it — fp32/HiFi4 are expected to HELP (opposite of the DiT).
- IDEAS: (1) HiFi4 math fidelity on the LM layers (text encoder uses default — check). (2) fp32
  accumulate. (3) the massive-activation channel: RMSNorm in fp32, or scale the residual stream. (4)
  the final few layers likely hold the outlier — mixed precision there. (5) bf16 embed vs fp32 embed.
