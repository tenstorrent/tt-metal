# DiffusionGemma in tt-dit

TT-Metal implementation of [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it):
a 30-layer Gemma 4 MoE (128 experts, top-8) with an autoregressive encoder + bidirectional
diffusion decoder. Generates text in 256-token canvases by iterative denoising.

## Module layout

```
models/tt_dit/
├── encoders/gemma4/                  # Shared Gemma 4 primitives (text + vision)
│   ├── rope.py                       # Dual-flavor 1D RoPE (default sliding + proportional full)
│   ├── attention.py                  # Sliding/full GQA, K=V quirk, KV-head replication
│   ├── vision_rope.py                # 2D multidim RoPE (with head_dim padding)
│   ├── vision_attention.py           # Vision MHA with head_dim padding (72 → 96)
│   ├── vision_mlp.py                 # (removed; now uses layers/feedforward.py:GatedMLP)
│   ├── vision_layer.py               # Vision encoder layer (4 norms)
│   ├── vision_patch_embedder.py      # Patch projection + 2D position lookup
│   ├── vision_pooler.py              # Host-side avg pool + fp32 sqrt scale
│   └── vision_model.py               # Full vision tower
├── models/transformers/diffusion_gemma/
│   ├── self_conditioning.py          # Pre/post-norm wrapper around GatedMLP
│   ├── embedding.py                  # Scaled word embedding (×sqrt(hidden))
│   ├── moe.py                        # Thin wrapper around demos/gemma4 MoEBlock
│   ├── layer.py                      # Encoder/decoder text layer (combined; encoder_kv triggers decoder mode)
│   ├── text_encoder.py               # Embedding → 30 layers → norm; returns KV cache
│   ├── text_decoder.py               # Embedding → self_conditioning → 30 layers (cross-attn) → norm
│   ├── multimodal_embedder.py        # Vision-soft-token → text-hidden projection
│   ├── encoder_model.py              # vision_tower + text_encoder + image-token substitution
│   └── model.py                      # DiffusionGemmaModel (enc+dec) + ForBlockDiffusion (lm_head + softcap)
├── layers/feedforward.py             # GatedMLP (shared across text/vision)
└── pipelines/diffusion_gemma/
    └── pipeline_diffusion_gemma.py   # Outer (canvas) + inner (diffusion) generation loops
```

## Reuse of existing tt-metal infrastructure

| Component | Reused from |
|---|---|
| MoE expert kernel (sparse_matmul) | `models/demos/gemma4/tt/{moe,router,experts}` |
| Linear projections, RMSNorm | `models/tt_dit/layers/` |
| CCL primitives, mesh management | `models/tt_dit/parallel/` |
| Sampler / logits processor / stopping criteria | `transformers.models.diffusion_gemma.generation_diffusion_gemma` (imported, not duplicated) |
| Test mesh fixtures | `models/tt_dit/utils/test.py` |

## Supported device meshes

Tests parametrize across three mesh families:

| Test id | Mesh shape | Topology | Notes |
|---|---|---|---|
| `bh_qb2_tp2` | (2, 4) | Linear | Blackhole QB2 board, TP=2 on mesh axis 0 |
| `bh_galaxy_tp4` | (4, 8) | Linear | Blackhole galaxy, TP=4 on mesh axis 0 |
| `wh_t3k_tp2` | (2, 4) | Ring | Wormhole T3K, TP=2 on mesh axis 0 with ring fabric |

The pipeline e2e/perf test skips `wh_t3k_tp2` by default because the 51 GB
checkpoint plus activations is tight on 8 × ~12 GB. Override with
`DIFFUSIONGEMMA_FORCE_T3K=1` once you've confirmed it fits.

## Tile-alignment / sharding details

- `intermediate_size = 2112` (dense MLP) is tile-aligned only for TP ∈ {1, 2}.
  `GatedMLP._padded_intermediate` pads to the next multiple at load time when needed.
- `moe_intermediate_size = 704` (per expert) is handled by demos/gemma4's expert
  weight loader, which pads automatically.
- Vision `head_dim = 72` is not tile-aligned. The vision attention pads each head
  to 96 with zero-fill (`Option A`), scales the Q/K norm weight by `sqrt(96/72)`,
  and multiplies the v_norm output by the same factor.
- Text `num_global_key_value_heads = 2` (full-attention layers) historically limited
  TP to {1, 2}. The attention module now supports KV-head replication
  (`Gemma4Attention._kv_replication`): when `tp_factor > num_kv_heads`, each KV
  head's weight is repeated `tp_factor // num_kv_heads` times so every TP rank
  gets exactly one (replicated) head.

## Running tests

All tests use the **real `transformers` HF classes as the reference baseline**
(no hand-ported math). They require `transformers >= 5.8.0.dev0` (the version
DiffusionGemma shipped in).

Bottom-up correctness order — recommended sequence for first hardware iteration:

```bash
# Foundational primitives
pytest models/tt_dit/tests/encoders/gemma4/test_rope.py -s
pytest models/tt_dit/tests/encoders/gemma4/test_vision_rope.py -s
pytest models/tt_dit/tests/unit/test_gated_mlp.py -s
pytest models/tt_dit/tests/models/diffusion_gemma/test_self_conditioning.py -s

# Attention and MoE
pytest models/tt_dit/tests/encoders/gemma4/test_attention.py -s
pytest models/tt_dit/tests/encoders/gemma4/test_vision_attention.py -s
pytest models/tt_dit/tests/models/diffusion_gemma/test_moe.py -s

# Layer composition
pytest models/tt_dit/tests/models/diffusion_gemma/test_layer.py -s
pytest models/tt_dit/tests/models/diffusion_gemma/test_layer_decoder.py -s
pytest models/tt_dit/tests/encoders/gemma4/test_vision_layer.py -s

# Full text models
pytest models/tt_dit/tests/models/diffusion_gemma/test_text_encoder.py -s
pytest models/tt_dit/tests/models/diffusion_gemma/test_text_decoder.py -s

# End-to-end pipeline (requires the full 51 GB checkpoint)
pytest models/tt_dit/tests/models/diffusion_gemma/test_pipeline_diffusion_gemma.py -s
```

Select a specific mesh with `-k`:

```bash
pytest .../test_attention.py -s -k bh_qb2_tp2     # BH QB2
pytest .../test_attention.py -s -k bh_galaxy_tp4  # BH galaxy
pytest .../test_attention.py -s -k wh_t3k_tp2     # WH T3K (ring)
```

Environment variables:

| Var | Default | Purpose |
|---|---|---|
| `DIFFUSIONGEMMA_MODEL_ID` | `google/diffusiongemma-26B-A4B-it` | HF checkpoint path/id for the e2e test |
| `DIFFUSIONGEMMA_FORCE_T3K` | `0` | Set to `1` to run the e2e test on WH T3K (the model is tight on memory there) |

## Test threshold conventions

| Granularity | PCC threshold | Allclose tolerance |
|---|---|---|
| Single primitive (RoPE, RMSNorm) | ≥ 0.9999 | atol 1e-2, rtol 1e-2 |
| Single submodule (attention, MLP) | ≥ 0.999 | atol 5e-2, rtol 5e-2 |
| Full layer (norms + attn + MoE) | ≥ 0.99 | atol 5e-2, rtol 5e-2 |
| Full text encoder / decoder model | ≥ 0.99 | atol 5e-2, rtol 5e-2 |
| MoE (sparse_matmul bf16 paths) | ≥ 0.99 | atol 5e-2, rtol 5e-2 |
| E2E pipeline (logit PCC + first-N token match) | ≥ 0.99 + exact token match | — |

## Known limitations / follow-ups

1. **Per-canvas re-encoding** — the pipeline re-encodes the full prefix each canvas
   rather than rolling the KV cache. Correct but wasteful. Optimization follow-up.
2. **Host-side image-token scatter** — `encoder_model.forward` round-trips merged
   text+vision embeddings to host. Moveable to on-device `ttnn.scatter` when the
   basic path is validated.
3. **Vision patch-embedder position table** — read back to host every forward.
   On-device `ttnn.embedding` lookup is a follow-up.
4. **Vision pooler** — host-side 2-D average pool. Convertible to `ttnn.matmul`
   on one-hot weights.
5. **`lm_head` is replicated** (~1.4 GB bf16). TP'ing on vocab dim halves the
   per-device memory; worth it when WH T3K is tight.
6. **Multi-batch (B > 1)** — pipeline assumes B=1. Multi-batch needs per-row
   finished masking in the outer loop.
7. **Real-config tests not validated on hardware yet** — every TT module here is
   best-effort against tt-dit conventions; first hardware iterations will surface
   issues that need fixing.

## Code provenance

| File | Source / reference |
|---|---|
| `attention.py`, `vision_attention.py` | `transformers.models.gemma4.modeling_gemma4` |
| `rope.py`, `vision_rope.py` | `transformers.models.gemma4.modeling_gemma4`'s `Gemma4TextRotaryEmbedding` + `Gemma4VisionRotaryEmbedding` |
| `moe.py` | Wraps `models.demos.gemma4.tt.moe.MoEBlock` |
| `layer.py`, `text_encoder.py`, `text_decoder.py`, `self_conditioning.py`, `embedding.py`, `multimodal_embedder.py`, `encoder_model.py`, `model.py` | `transformers.models.diffusion_gemma.modeling_diffusion_gemma` |
| Vision tower (patch embedder, pooler, layer, model) | `transformers.models.gemma4.modeling_gemma4` |
| Sampler / stopping / processor | Imported directly from `transformers.models.diffusion_gemma.generation_diffusion_gemma` |

All TT implementations are validated against the HF classes as the parity baseline
(no hand-ported math in the tests).
