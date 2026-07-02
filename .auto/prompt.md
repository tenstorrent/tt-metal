# Autoresearch: ACE-Step v1.5 FULL text-to-music pipeline — TTNN bring-up on Blackhole p150

## Objective
Extend the validated DiT bring-up into the **complete generation pipeline** running on TTNN, then
score generated audio with **SongEval** (the aesthetic evaluator used in the ACE-Step paper).

The full compute graph (reference is genuine HF checkpoints in `/local/ttuser/gtobar/acestep_pipeline`):

    text/lyric/timbre embeddings
        │
        ▼  ConditionEncoder      ✅ DONE (text_projector + lyric + timbre encoders)
    encoder_hidden_states + context_latents
        │
        ▼  DiT denoise loop      ✅ CORE DONE (24-layer AceStepDiTModel; loop = FlowMatchStep)
    clean audio latents  [B, 64, T_latent]
        │
        ▼  VAE decoder (Oobleck) ⬜ NEW — Conv1d + ConvTranspose1d + Snake1d, ~169M params
    48 kHz stereo waveform  [B, 2, samples]
        │
        ▼  SongEval scorer       ✅ harness works (MuQ SSL + Generator -> 5 aesthetic scores)
    Coherence / Musicality / Memorability / Clarity / Naturalness

Pipeline components on disk (`/local/ttuser/gtobar/acestep_pipeline`, ALL gitignored — never commit weights):
- `acestep-5Hz-lm-1.7B/`  LM planner (prompt -> song blueprint), 3.7 GB — standard Qwen3 causal LM
- `Qwen3-Embedding-0.6B/` text encoder, 1.2 GB
- `acestep-v15-turbo/`    turbo DiT (distilled, few-step), 4.8 GB
- `vae/`                  diffusers AutoencoderOobleck (latents->audio), 0.34 GB  ← PRIMARY VAE
- `vae_stable_audio/`     same VAE in stable-audio-tools format, 0.67 GB
Base DiT (`acestep-v15-base`) already cached + TTNN-validated.

## Metrics
- **Primary**: `modules_passing` (count of PCC tests passing, higher is better) — continues the
  bring-up count. Each new pipeline component (VAE conv block, Snake1d, VAE decoder, text-encoder
  layer, LM layer, full e2e pipeline, SongEval integration) adds validated modules.
- **Secondary**: `min_pcc` (lowest module PCC — keep modules ≥0.97), `e2e_pcc` (full-pipeline PCC
  vs reference — STRICT GATE ≥0.95), `tests_failing`, `suite_seconds`.

## STRICT CORRECTNESS RULES (do not violate)
1. **Per-module PCC high**: aim 0.99, accept ≥0.97 for deep stacks. A module below its honest
   achievable PCC is a bug to fix, not to paper over.
2. **E2E pipeline PCC ≥ 0.95** — the whole TTNN pipeline vs the genuine reference pipeline. This is
   a HARD requirement. If e2e drops below 0.95, that iteration fails.
3. **Never cheat / never overfit**: compare against genuine HF reference outputs, measure real PCC,
   use honest thresholds. Do not tune tolerances to pass. Do not special-case test inputs.
4. **Reuse TTTv2** (`models/tt_dit`, `models/common`) wherever applicable; if no TTTv2 module fits,
   write a thin custom module following the TTTv2 config+LazyWeight contract (see existing acestep tt/).

## How to Run
`./.auto/measure.sh` — runs the acestep PCC suite, emits `METRIC name=value` lines.
Fast dev subset: `pytest models/experimental/acestep/tests/pcc -m "not slow"`.

## Files in Scope
- `models/experimental/acestep/tt/*.py` — TTNN modules (add vae_decoder.py, snake.py, text_encoder.py, lm/*, pipeline.py)
- `models/experimental/acestep/tt/model_config.py` — AceStepModelConfig + create_tt_model factory (EXTEND: add VAE/text-enc/LM builders + a create_tt_pipeline)
- `models/experimental/acestep/tests/pcc/test_*.py` — PCC tests (add per new module + e2e)
- `models/experimental/acestep/demo/**` — SongEval scorer + demo eval (add ttnn pipeline demo)
- `models/experimental/acestep/reference/*.py` — reference loaders/weight_utils (extend for new checkpoints)

## Off Limits
- `models/tt_dit/`, `models/common/` core (reuse, do not modify their internals)
- Any weights / audio / large binaries — NEVER commit (gitignored). Pipeline lives outside repo tree.
- Do not weaken existing passing tests to make new ones pass.

## Constraints
- All existing PCC tests must keep passing (no regressions).
- New device ops must actually run on the p150 (single Blackhole device).
- bf16 weights; PCC measured vs fp32/bf16 reference as the existing suite does.

## Build Order (suggested — each a keepable module)
1. **Snake1d** activation (x + sin²(αx)/α) — elementwise, TTNN ops. PCC 0.99.
2. **VAE Conv1d / ConvTranspose1d** wrappers (weight-norm folded) — reuse ttnn.conv1d or fold to matmul.
3. **Oobleck residual unit** (snake+conv x2 + residual) — composition.
4. **VAE decoder block** (upsample ConvTranspose + 3 res units).
5. **Full VAE decoder** (proj + 5 blocks + final snake+conv) — latents -> waveform. PCC ≥0.97.
6. **create_tt_pipeline** in model_config: DiT-loop + VAE decode -> audio (like DiT's factory).
7. **E2E pipeline PCC** vs reference generate_audio+VAE (STRICT ≥0.95).
8. **SongEval on TTNN audio**: decode ttnn latents -> waveform -> SongEval 5 scores; compare to
   reference-pipeline audio scores (delta should be tiny — same weights, only bf16/device noise).
9. (stretch) text encoder + LM planner layers on device.

## What's Been Tried
- (session start) Full pipeline downloaded + verified. VAE loads (diffusers AutoencoderOobleck):
  `[1,64,50] -> [1,2,96000]` @ 48kHz, 1920 samples/latent-frame. SongEval scorer runs end-to-end on
  real audio (example.mp3 -> Coherence 4.23 etc). DiT + ConditionEncoder already validated (67 PCC
  tests, real-weight e2e 0.9627). model_config.py has AceStepModelConfig + create_tt_model (DiT).
