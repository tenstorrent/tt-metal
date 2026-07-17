# XTTS-v2 Bringup in TTNN / tt-metal — MASTER

Master coordination file for implementing **Coqui XTTS-v2** as a hand-written **TTNN**
implementation in this repo (tt-metal), NOT the compiler path.

**This is a MASTER/index file.** Each pipeline block is brought up **independently** and
has its **own `.md` file** so parallel agents can work without stepping on each other.
This file holds only what is *shared*: cross-block decisions, the integration contract
(the tensor interfaces between blocks), the block registry, and references. **Per-block
detail — build steps, findings, PCC logs, open questions — lives in that block's file, not
here.**

> Sibling effort: a compiler-based bringup (tt-xla, PJRT → tt-mlir, via tt-forge-models
> PR #784) is logged in `/localdev/acicovic/tt-xla/CLAUDE_XTTS_FORGE.md` (owned by another
> agent — read for model-internals detail, don't modify). This file is the TTNN/Metal path.

---

## Workflow: parallel per-block bringup

- Each block/submodel is developed and PCC-validated **independently against golden
  tensors** (produced by the PyTorch reference mirror), so blocks do not need to wait on
  each other. Integration happens last, once each block passes PCC standalone.
- **One `.md` file per block** (registry below). An agent picking up a block:
  1. Reads THIS master first (shared decisions + integration contract).
  2. Works only in its own block file + that block's code under
     `models/experimental/xtts_v2/`.
  3. Treats the **Integration Contract** below as fixed. If a block's real input/output
     shape or dtype differs from the contract, update the contract row here (and ping the
     downstream block's owner) — do NOT silently diverge.
  4. Records all findings/PCC/decisions in its own block file, dated.
- **Golden-tensor rule:** every block is fed the SAME real end-to-end tensors (from the
  reference mirror), so an agent can develop its block in isolation using a saved golden
  input and check its output against the golden output — no other block required.
- Keep per-block `.md` files to their own scope; put anything cross-cutting HERE.

---

## Block registry (one `.md` per block)

| # | Block | File | Foundation | Status | Owner |
|---|-------|------|-----------|--------|-------|
| 0 | Text tokenizer / normalization | `CLAUDE_XTTS_TOKENIZER.md` | net-new (CPU) | not started | — |
| 1 | Conditioning encoder + Perceiver resampler | `CLAUDE_XTTS_CONDITIONING.md` | net-new TTNN | not started | — |
| 2 | ResNet speaker encoder (d-vector) | `CLAUDE_XTTS_SPEAKER_ENCODER.md` | net-new (CPU first) | not started | — |
| 3 | GPT (decoder-only, VQ codes) | `CLAUDE_XTTS_GPT.md` | HF GPT2 core (hand-written TTNN) | **prefill + KV-cache decode PCC passing (0.9997 bf16 / 0.99996 fp32)** | acicovic |
| 4 | HiFi-GAN vocoder | `CLAUDE_XTTS_HIFIGAN.md` | `speecht5_tts` pattern (CPU first) | not started | — |
| — | Integration / top-level model + demo | this file + `tt/ttnn_xtts_model.py` | — | not started | — |

Each block file follows the same template — see **Per-block file template** at the bottom.

---

## Integration Contract (fixed interfaces between blocks)

These are the tensor hand-offs. Shapes from the FORGE log's real-input chain; confirm
exact dims against coqui `TTS/tts/models/xtts.py` when building each block. **Do not change
a row without updating downstream owners.**

| Producer | Output tensor | Shape | dtype | Consumer |
|----------|--------------|-------|-------|----------|
| (input) reference clip | `reference_audio_22k` | waveform @ 22050 Hz | f32 | Block 1 |
| (input) reference clip | `reference_audio_16k` | waveform @ 16000 Hz | f32 | Block 2 |
| (input) text | raw string | — | — | Block 0 |
| Block 0 tokenizer | `text_tokens` | (1, N_text) | int | Block 3 |
| Block 1 conditioning+Perceiver | `gpt_cond_latent` | (1, 32, 1024) | f32 | Block 3 (GPT prefix) |
| Block 2 speaker encoder | `speaker_embedding` (d-vector) | (1, 512, 1) | f32 | Block 4 (vocoder `g=`) |
| Block 3 GPT (`return_latent`) | `gpt_latents` | (1, T_code, 1024)* | f32 | Block 4 |
| Block 4 HiFi-GAN | `waveform` | 24 kHz mono | f32 | (output .wav) |

\* `T_code` = number of autoregressively generated VQ frames (~21.53 Hz). Confirm the
latent channel dim (1024) against the checkpoint.

**Two separate conditioning branches — do NOT conflate:** Block 1 (A) feeds the GPT prefix;
Block 2 (B) feeds the vocoder d-vector. They come from the same reference clip at different
sample rates.

---

## Decisions locked in (2026-07-17) — shared across all blocks

1. **Path:** hand-written TTNN in tt-metal (op-by-op, PCC-validated), NOT tt-xla compiler.
2. **GPT core → build on `models/tt_transformers/`.** XTTS's GPT is a decoder-only LLM;
   reuse attention/RoPE/KV-cache/traced-decode/sampling-head. Adapt only the input side
   (32 conditioning latents as prefix + text tokens) and the head (VQ code logits).
3. **Convs on CPU first, port to TTNN later.** HiFi-GAN vocoder (Block 4) and ResNet
   speaker encoder (Block 2) are conv-heavy; ship a correct e2e pipeline with them on CPU
   (as `speecht5_tts` does), then port. De-risks group_norm/conv tile-alignment (see below).
4. **Golden-tensor-driven, per-block PCC** at ≈ 1.0 vs the PyTorch reference mirror before
   any integration.

---

## Model Overview (XTTS-v2)

Multilingual (17 lang) zero-shot voice-cloning TTS from a ~6s reference clip; 24 kHz out.
Inference path (confirmed against coqui `TTS/tts/models/xtts.py`, per FORGE log):

- **Branch A — conditioning encoder + Perceiver** (`gpt.get_style_emb`) → 32×1024 latents
  that **prefix the GPT**.
- **Branch B — ResNet speaker encoder** → d-vector, injected into HiFi-GAN (`g=`).
- **Text** → normalization + custom XTTS BPE → tokens.
- **GPT** (decoder-only, 30 blocks, 16 heads, ~443M): 32 latents + text tokens →
  autoregressive **discrete VQ audio codes** (~21.53 Hz); with `return_latent` emits
  **latents** for the vocoder.
- **HiFi-GAN** (~26M) → 24 kHz. Consumes GPT latents **directly** (no mel), d-vector
  injected via linear projections at each upsample layer.
- **DVAE:** training-only (audio→codes); stripped from inference checkpoints. Not needed.

Loaded via **coqui-tts** (`TTS.tts.models.xtts.Xtts`), NOT transformers-native; no single
traceable forward → hence the per-block split.

---

## `speecht5_tts` as a partial template (shared findings)

`models/experimental/speecht5_tts/` is the closest in-repo TTS model. Copy its **skeleton
and methodology**, not its model code.

**Transfers:** directory layout (`reference/` op-by-op mirrors with `load_from_*` + `tt/`
TTNN modules each with a `TTNN*Config` + `preprocess_*_parameters()` + `tests/` PCC +
`demo_ttnn.py`); the reference-mirror + per-component PCC discipline; CPU vocoder fallback;
component-split top-level model.

**Does NOT transfer:** single-vector conditioning (XTTS has two branches); mel-regression
decoder (XTTS GPT does discrete-token decode → use `tt_transformers`); postnet/mel stage
(XTTS has none); transformers/HF loading (XTTS uses coqui-tts + custom BPE).

---

## Proposed directory layout — `models/experimental/xtts_v2/`

```
reference/            op-by-op PyTorch mirrors, PCC=1.0 vs coqui-tts (golden tensors)
  xtts_config.py
  tokenizer.py               # Block 0
  conditioning_encoder.py    # Block 1 (conditioning encoder + Perceiver)
  speaker_encoder.py         # Block 2 (ResNet → d-vector)
  gpt.py                     # Block 3 (decoder-only GPT reference)
  hifigan.py                 # Block 4 (vocoder reference + CPU runtime impl)
tt/                   hand-written TTNN + tt_transformers-based GPT
  ttnn_conditioning_encoder.py   (+ Config + preprocess_*_parameters)
  ttnn_speaker_encoder.py
  ttnn_xtts_gpt.py               # wraps tt_transformers primitives
  ttnn_xtts_model.py             # integration: wires branches A/B → GPT → vocoder
tests/               per-block layer-by-layer PCC + e2e PCC
demo_ttnn.py
```

---

## Recommended bringup order (blocks are otherwise independent)

Parallelizable, but if sequencing: **0/1/2 (golden tensors) → 3 GPT (biggest risk) → 4
vocoder port → integration.** Blocks 2 and 4 can stay on CPU until the core works.

---

## Shared dependencies / setup

- **coqui-tts** for reference mirrors + golden tensors (pins in the FORGE log: coqui-tts
  0.27.5, torchaudio matched to repo torch, + NLP deps: num2words, anyascii, inflect,
  pysbd, monotonic-alignment-search, …).
- Checkpoint from HF at load time; CPML-gated → loader auto-sets `COQUI_TOS_AGREED=1`
  (HF repo itself ungated).
- `import TTS` alone can fail (isin_mps_friendly + NLP deps); the real loader path handles
  it once deps are installed.

---

## Compiler-path failure to keep in mind (FORGE log, 2026-07-16)

The tt-xla bringup hit `ttnn.group_norm op flattened height must be tile-aligned, got 505`
in the **conditioning encoder** (mel/sequence time length not a multiple of 32; next tile
boundary 512) — a tt-mlir/ttnn limitation on non-tile-aligned group_norm heights. A
hand-written TTNN impl faces the same class of conv/group-norm alignment issues → another
reason to keep convs (Blocks 2 & 4) on CPU until the core pipeline works. Relevant to
Blocks 1, 2, 4.

---

## References

- **Official model (HF):** https://huggingface.co/coqui/XTTS-v2
- XTTS paper: https://arxiv.org/abs/2406.04904
- Coqui XTTS source: `TTS/tts/models/xtts.py`
- Compiler-path log (sibling): `/localdev/acicovic/tt-xla/CLAUDE_XTTS_FORGE.md`
- tt-forge-models PR #784: https://github.com/tenstorrent/tt-forge-models/pull/784
- In-repo template model: `models/experimental/speecht5_tts/`
- LLM stack for the GPT: `models/tt_transformers/`

---

## Per-block file template

Every `CLAUDE_XTTS_<BLOCK>.md` should follow this shape so agents stay consistent:

```
# XTTS-v2 <Block Name> — TTNN bringup
Parent: CLAUDE_XTTS_TTNN.md (read it first for shared decisions + integration contract)

## Status / Owner / Started
## Role in pipeline           (one line + which contract rows it produces/consumes)
## Interface contract         (this block's inputs/outputs — copy the relevant rows)
## Foundation / template      (net-new / tt_transformers / speecht5 pattern)
## Reference source           (coqui file + class/function to mirror)
## Build steps
## PCC validation plan         (golden tensors, target PCC, test location)
## Findings log (dated)
## Open questions / TODO
```
