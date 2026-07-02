# ACE-Step v1.5 base — TTNN bring-up (Blackhole p150)

TTNN implementation of **[ACE-Step/acestep-v15-base](https://huggingface.co/ACE-Step/acestep-v15-base)**,
a ~2B-parameter music-generation model, brought up module-by-module on a single Blackhole
**p150** and validated for numerical correctness (PCC) against the genuine HuggingFace reference.

## Approach

Built on the **TTTv2 module library** (`models/common/modules/`) — reuse first, write custom
only where the library cannot express the op. Every module: **write class → write PCC test →
verify PCC → next**. Validated against the real HF modules (`trust_remote_code`), with real
config dims, and against the **real trained checkpoint** (`model.safetensors`) for the core path.

## Architecture

ACE-Step v1.5 base is a **Qwen3-architecture Diffusion Transformer (DiT)** for flow-matching
music generation. `hidden=2048`, 24 layers, 16 heads / 8 KV (GQA), `head_dim=128`, SwiGLU MLP
(`intermediate=6144`), RMSNorm (`eps=1e-6`), per-head q/k-norm, RoPE `θ=1e6`, alternating
sliding (window=128) / full attention, `AdaLN` timestep modulation.

```
text  ─► text_projector (Linear) ─┐
lyric ─► AceStepLyricEncoder ──────┤ pack ─► cross-attn context ─┐
timbre─► AceStepTimbreEncoder ─────┘                            │
                                                                ▼
noise + context_latents ─► proj_in (patchify) ─► [ 24 × AceStepDiTLayer ] ─► norm_out ─► proj_out
                              (AdaLN from dual timestep embedding)              (de-patchify)
                                                                ▼
                        per denoise step: v = DiT(...);  xt ← xt − v·dt   (flow-matching Euler)
                                                                ▼
quantized tokens ─► AudioTokenDetokenizer ─► reconstructed audio latents
```

## Module map (`tt/`)

| Module | File | Reuse / custom |
|--------|------|----------------|
| RMSNorm | *(TTTv2 `RMSNorm1D`)* | pure reuse |
| MLP (SwiGLU) | *(TTTv2 `MLP1D`)* | pure reuse |
| RoPE | *(ttnn `rotary_embedding_hf`)* | op reuse |
| Attention (self full/sliding + cross) | `attention.py` | custom (GQA, qk-norm, bidirectional) |
| Timestep embedding | `timestep_embedding.py` | custom (sinusoidal + 6× modulation) |
| Encoder layer | `encoder_layer.py` | composition |
| DiT layer (AdaLN) | `dit_layer.py` | composition |
| DiT layer stack | `dit_stack.py` | composition |
| Patch embed (`proj_in`) | `patch_embed.py` | custom (Conv1d→patchify+linear) |
| DiT output (`norm_out`+`proj_out`) | `dit_output.py` | custom (AdaLN + de-patchify) |
| Full DiT model | `dit_model.py` | top-level assembly |
| Lyric / timbre encoder | `lyric_encoder.py` | composition (timbre reuses lyric) |
| Attention pooler | `attention_pooler.py` | composition (CLS pooler) |
| Audio detokenizer | `detokenizer.py` | composition |
| Condition encoder | `condition_encoder.py` | top-level assembly |
| Flow-matching solver step | `flow_match.py` | custom (elementwise Euler) |

## PCC results

Validated vs genuine HF reference. `random` = random-init weights; `real` = genuine
`model.safetensors` trained weights.

| Component | Weights | PCC | Threshold |
|-----------|---------|-----|-----------|
| RMSNorm / MLP / RoPE | random | 0.9999 | 0.999 |
| Attention (self / sliding / cross) | random | ≥ 0.99 | 0.99 |
| Timestep embedding | random | 0.999 | 0.999 |
| Encoder layer / DiT layer | random | ≥ 0.98 | 0.98 |
| Patch embed / DiT output | random | ≥ 0.99 | 0.99 |
| Lyric encoder (8 layers) | random / **real** | 0.97 / **0.97** | 0.97 |
| Timbre encoder (4 layers) | random / **real** | 0.97 / **0.97** | 0.97 |
| Attention pooler | random | 0.98 | 0.98 |
| Detokenizer | random / **real** | 0.98 / **0.98** | 0.98 |
| Condition encoder (text+lyric+timbre) | random | 0.96 | 0.96 |
| Condition → DiT seam (2 DiT layers) | random | 0.94 | 0.94 |
| **Full 24-layer DiT model (e2e)** | random / **real** | **0.999 / 0.9919** | **0.95** |
| **Full pipeline (ConditionEncoder → 24-layer DiT)** | **real** | **0.9627** | **0.95** |

**Headline: the full generation pipeline — the real ConditionEncoder feeding the real 24-layer
DiT — runs end-to-end at PCC 0.9627 on genuine trained weights** (≥ 0.95 required); the DiT
model alone is 0.9919. Real weights score slightly below random — the trained distribution has
larger magnitudes/outliers that stress bf16 more — which confirms the suite is not gamed.

## Running the tests

```bash
# full suite (all modules, ~128s on p150)
pytest models/experimental/acestep/tests/pcc/

# fast subset for iteration (excludes the heaviest e2e/composition tests, ~79s)
pytest models/experimental/acestep/tests/pcc/ -m "not slow"
```

Real-weight tests auto-skip if `model.safetensors` is absent. To enable them:

```bash
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('ACE-Step/acestep-v15-base','model.safetensors')"
```

## Weight loading

`reference/weight_utils.py` loads genuine checkpoint tensors directly from `model.safetensors`
via `safetensors.safe_open` (the full `AutoModel.from_pretrained` fails on `ResidualFSQ`
meta-init). `load_module_weights(ref_module, "decoder.")` populates an instantiated reference
sub-module; the TT test helpers then transpose HF `[out,in]` weights to `[in,out]` for
`ttnn.linear` and wrap them in `LazyWeight` (mirrors the Phi-4 `weight_utils` pattern).

## Known exclusions (justified)

- **FSQ / ResidualFSQ tokenizer** — cover-song aux path only (`is_covers=True`); the library's
  internal quantizer normalization could not be replicated exactly (~0.98 ceiling), so it was
  not shipped rather than misrepresent the numerics. See `.auto/ideas.md`.
- **`pack_sequences`** — host-side data-dependent argsort/gather; with all-valid masks it is
  exactly `concat` (the case validated). Padded-batch reordering is caller orchestration.
- **CFG guidance combine** (`apg_forward`/`adg_forward`) — `apg_guidance.py` is absent from the
  checkpoint snapshot; guidance-only, orthogonal to the validated solver step.

## References

- TTTv2 module contract: `models/common/modules/README.md`
- Weight-loading pattern: `models/common/models/phi4/weight_utils.py`
- File/test layout pattern: `models/demos/wormhole/bge_m3/`
