# FIBO on TTNN — Model Documentation

## Overview

This document covers the Tenstorrent tt_dit implementation of Bria's **FIBO** text-to-image model. FIBO is a flow-matching MMDiT (Flux-shaped) model conditioned on an LLM text encoder. The implementation is decomposed into four sub-projects, built in data-flow order. See the design spec for full detail:
[`docs/superpowers/specs/2026-07-06-fibo-smollm3-encoder-design.md`](../../../docs/superpowers/specs/2026-07-06-fibo-smollm3-encoder-design.md)

## 4-Sub-Project Decomposition

| # | Sub-project | Status | Strategy |
|---|---|---|---|
| **1** | **SmolLM3 text encoder** | **Done** | New `encoders/smollm3/`; decoder layer from Qwen25VL, all-hidden-states shell from Gemma |
| **2** | **BriaFibo transformer** | **Done** | New `transformer_bria_fibo.py` from Flux1 + per-layer "concat-halves" text injection |
| **3** | **Wan VAE + solver wiring** | **Done** | Reuse `vae_wan2_1.py` (T=1 decode) + `EulerSolver` + dynamic-shift scheduler |
| 4 | Pipeline + Blackhole bringup | TODO | New `pipelines/bria_fibo/`; CFG batched=2; 2×2 mesh (`cfg=(1,0) sp=(2,0) tp=(2,1)`) |

---

## Sub-project 1: SmolLM3 Text Encoder

### Architecture

SmolLM3-3B serves as the text encoder for FIBO. Key configuration:

| Parameter | Value |
|---|---|
| Layers | 36 |
| Hidden size | 2048 |
| Attention heads / KV heads | 16 / 4 (GQA) |
| Intermediate size | 11008 |
| Activation | SiLU |
| Norm | RMSNorm, eps=1e-6 |
| RoPE theta | 5,000,000 |
| Max position embeddings | 65,536 |
| NoPE (no positional embedding) | every 4th layer (0-indexed 3, 7, 11, ..., 35); 9 NoPE layers total |
| Vocab size | 128,256 |
| Attention bias | False |

### NoPE Layers

`no_rope_layers[i] = int((i + 1) % 4 != 0)` — value `1` = apply RoPE, value `0` = NoPE (no positional embedding applied). The 9 NoPE layers are at indices 3, 7, 11, 15, 19, 23, 27, 31, 35.

### HF-Exact Output Contract

The encoder replicates `SmolLM3ForCausalLM(..., output_hidden_states=True)` exactly:

- **All hidden states**: length `num_hidden_layers + 1 = 37`. `hidden_states[0]` is the embedding output (input to layer 0); `hidden_states[i]` is the output of layer `i-1`; `hidden_states[-1]` is the final RMSNorm output.
- **`prompt_embeds`**: `torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)` → shape `[B, T, 4096]` (2 × 2048). This is the primary FIBO conditioning signal consumed by the transformer.
- **All 37 states** are also returned for per-block injection into the BriaFibo transformer (sub-project 2).

### Files

```
models/tt_dit/
  encoders/smollm3/
    __init__.py
    config.py              # SmolLM3Config + from_hf_config()
    model_smollm3.py       # SmolLM3TextEncoder, SmolLM3DecoderLayer, SmolLM3Attention, SmolLM3Mlp
  tests/encoders/smollm3/
    test_smollm3.py        # Unit tests + full-mesh (2×2) validation
```

### Running the Tests

```bash
# All SmolLM3 encoder tests (needs FIBO weights + login, see below)
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py -v

# Full-mesh (2×2 Blackhole) validation only
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_full_mesh" -v
```

**Prerequisites:**

- **Weights**: `briaai/FIBO` is a gated HuggingFace repo. Accept the license on the HF model page, then authenticate: `huggingface-cli login`. The test will `pytest.skip` with a clear message if weights are unavailable.
- **`FIBO_PATH`** (optional): Override the HF model path, e.g. `FIBO_PATH=/data/models/FIBO pytest ...`. Defaults to `briaai/FIBO`.
- **`N_LAYERS`** (optional): Truncate the encoder to `N` layers for the `test_smollm3_encoder_all_layers` and `test_smollm3_encode_contract` tests (default: 6). Use `N_LAYERS=36` for a full-depth truncated run. The `test_smollm3_encoder_full_mesh` test always runs all 36 layers.
- **Devices**: The full-mesh test requires **4 Blackhole devices** (2×2 mesh) with `FABRIC_1D` fabric config. Single-device tests (`tp=1`) use any single Blackhole.

### Measured PCCs (real `briaai/FIBO` weights)

| Test | PCC |
|---|---|
| MLP (single layer) | 99.998% |
| Attention, RoPE path | ~99.99% |
| Attention, NoPE path | ~99.99% |
| Decoder layer (single) | 99.9996% |
| Encoder all-layers (6L, seq=128, tp=1) — per-layer min | 99.957% |
| encode() contract (6L, seq=128, tp=1) | 99.98% |
| **Full 36L mesh (seq=128, tp=2 on 2×2)** | **99.9362%** |
| **Full 36L mesh (seq=2048, tp=2 on 2×2)** | **99.9597%** |

All tests pass PCC ≥ 0.99. No bf16 depth drift observed even at seq=2048 over all 36 layers. CCL (`all_gather_async`) on the mesh axis was exercised without hangs.

---

## Sub-project 2: BriaFibo Transformer

### Architecture

The BriaFibo transformer is an MMDiT denoiser following the Flux1 pattern, combining dual and single transformer blocks:

| Parameter | Value |
|---|---|
| Total blocks | 46 |
| Dual transformer blocks | 8 |
| Single transformer blocks | 38 |
| Inner dim (`hidden_size`) | 3072 |
| Attention heads | 24 |
| Head dim | 128 |
| In channels | 48 |
| Out channels | 48 |
| Timestep modulation | Timestep only (no pooled embedding, no guidance scaling) |
| Context embedding dim | 4096 (from SmolLM3 final + penultimate hidden states) |
| Text encoder hidden dim | 2048 |
| RoPE configuration | Axial on spatial dims [16, 56, 56]; θ = 10000 |

### Per-Block Text Injection: "Concat-Halves"

Before each of the 46 transformer blocks, the context is updated via concat-halves injection. The implementation maintains a 46-entry list of text encoder hidden states (padded from SmolLM3's 37 states; padding is a pipeline concern). At each block `i`, the second half of the running `encoder_hidden_states` is replaced:

```
encoder_hidden_states[..., hidden_size/2:] = caption_projection[i](text_encoder_layers[i])
```

where `caption_projection[i]` is a learned Linear layer projecting from 2048 (text encoder dim) to 1536 (half of 3072).

**Tensor-parallel efficiency (tp=2):** On a 2×2 mesh with tp=2, the feature dimension is sharded at the 1536 boundary. This means the half-replacement is a gather-free per-device shard select — each device receives its own text projection and performs the replacement on its local shard. At tp=1, the operation is a standard concatenation across the full feature dim.

### Component Reuse

The transformer reuses two core blocks from `tt_dit.Flux`:

- **`TransformerBlock` (dual block):** Combines both image (latent) and context (text) streams with interleaved self- and cross-attention. Used for the first 8 blocks.
- **`Flux1SingleTransformerBlock` (single block):** Image-only, processing latent → latent with self-attention. Used for the remaining 38 blocks.

Both blocks use the same attention machinery, RoPE, and checkpoint (gradient ckpt) support unchanged from Flux1. Net-new components:

- **`BriaFiboTextProjection`:** 46-entry list of Linear(2048 → 1536) layers, one per block.
- **`BriaFiboTimestepEmbed`:** Timestep embedding (scalar t → 3072-dim, no guidance scaling).
- **`context_embedder`:** Linear(4096 → 3072), applies to the initial context vector (SmolLM3 prompt_embeds).
- **`inject_text(context, text_encoder_layers, block_id)`:** Replaces the second half of context with the current block's projected text.

### Text Encoder Layer Indexing

SmolLM3 yields 37 hidden states (input embedding + 36 layer outputs). The transformer requires 46 (one per block). The mismatch is resolved at the **pipeline level (Sub-project 4):** the pipeline pads the 37-state list with 9 copies of the final state, producing a 46-entry list. The transformer receives this list and indexes it directly; no padding occurs in the transformer itself.

### Files

```
models/tt_dit/
  models/transformers/
    transformer_bria_fibo.py        # BriaFiboTransformer, inject_text, BriaFiboTextProjection, BriaFiboTimestepEmbed
  tests/models/bria_fibo/
    test_transformer.py             # Unit tests + full-mesh (2×2) validation
```

### Running the Tests

```bash
# All BriaFibo transformer tests (needs FIBO weights + login, see below)
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest models/tt_dit/tests/models/bria_fibo/test_transformer.py -v

# Reduced-depth iteration (2 dual + 2 single blocks only, for fast development)
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  FIBO_DUAL=2 FIBO_SINGLE=2 \
  python_env/bin/python -m pytest models/tt_dit/tests/models/bria_fibo/test_transformer.py -v
```

**Prerequisites:**

- **Weights**: `briaai/FIBO` transformer weights must be pre-downloaded. Accept the license on the HF model page, authenticate (`huggingface-cli login`), then download the `transformer/` folder. The test will skip with a clear message if weights are unavailable.
- **`FIBO_PATH`** (optional): Override the HF model path, e.g. `FIBO_PATH=/data/models/FIBO pytest ...`. Defaults to `briaai/FIBO`.
- **`FIBO_DUAL`** (optional): Truncate to `N` dual blocks (default: 8). Use for reduced-depth iteration.
- **`FIBO_SINGLE`** (optional): Truncate to `N` single blocks (default: 38). Use for reduced-depth iteration.
- **Devices**: Full-mesh (tp=1 and sp=2, tp=2 on 2×2) tests require **4 Blackhole devices** (2×2 mesh) with `FABRIC_1D` fabric config. Single-device tests (tp=1) use any single Blackhole.

### Measured PCCs (real `briaai/FIBO` weights, bf16)

| Test | Configuration | PCC |
|---|---|---|
| Text projection layer (single) | tp=1 | 99.95% |
| Timestep embedding | tp=1 | 99.95% |
| Dual block, spatial attention | tp=1 | 99.996% |
| Dual block, context attention | tp=1 | 99.97% |
| Full transformer (8+38 blocks) | tp=1 | 99.97% |
| Full transformer (8+38 blocks) | 2×2 mesh (sp=2, tp=2) | 99.53% |
| Full transformer, reduced (2+2) | tp=1 | 99.998% |
| Full transformer, reduced (2+2) | 2×2 mesh (sp=2, tp=2) | 99.94% |

All tests pass PCC ≥ 0.99. The full-depth tp=1 and reduced-depth 2×2 iterations both validate HF-exact behavior. The full-depth 2×2 result (99.53%) reflects accumulated bf16 quantization across the longer block chain with spatial sharding; reduced iterations show bf16 accumulation is stable below that depth.

---

## Sub-project 3: Wan VAE + Solver Wiring

### Architecture

FIBO's VAE is `AutoencoderKLWan` configured as the Wan 2.2 high-compression ("TI2V") variant, distinct from the Wan 2.1 `z_dim=16` VAE already supported by tt_dit:

| Parameter | Value |
|---|---|
| `is_residual` | `True` (Wan 2.2 residual decoder) |
| `z_dim` | 48 |
| `base_dim` | 160 |
| `decoder_base_dim` | 256 |
| `dim_mult` | `[1, 2, 4, 4]` |
| `out_channels` | 12 |
| `scale_factor_spatial` | 16 |

`z_dim=48` is also the BriaFibo transformer's `in_channels`/`out_channels` (sub-project 2), so the diffusion latent feeds the transformer directly with no 2×2 packing step (unlike Flux's VAE/transformer channel relationship).

`tt_dit`'s existing `vae_wan2_1.py` previously hard-blocked this configuration with `assert not is_residual` in `WanDecoder3d`, `WanDecoder`, and `WanEncoder3D`. This sub-project removes the two **decode-side** asserts and implements the residual decoder path below; the **encode-side** assert (`WanEncoder3D`) is left in place — encode is out of scope for FIBO, which only needs decode. The residual decoder path adds:

- **`WanDupUp3D`**: a parameter-free residual-shortcut upsampler, a BTHWC port of diffusers' `DupUp3D`. Repeats/duplicates the input along the upsample factor and optionally drops the leading `factor_t - 1` frames when `first_chunk=True`.
- **`WanResidualUpBlock`**: the Wan 2.2 up-block. Wraps the existing resnet-block path and adds the `avg_shortcut` (a `WanDupUp3D`) applied to the block's input and summed into its output: `x = x + avg_shortcut(x_copy, first_chunk=first_chunk)`.
- **`WanDecoder3d`**'s up-block construction loop now branches on `is_residual`, building `WanResidualUpBlock`s instead of the standard `WanUpBlock`/`WanResample` path when set. `first_chunk` (true only for the first temporal chunk, matching the reference `_decode`) is threaded from `WanDecoder.forward` through the up-block loop into `WanResidualUpBlock`/`WanDupUp3D`.
- The Wan 2.1 (`is_residual=False`) decode path — `WanUpBlock`, `WanResample`, `WanDecoder3d`, `WanDecoder` — is unchanged; `test_wan_decoder` (Wan 2.1) continues to pass at the same PCC as before this change.

### Flow-Match Solver + Dynamic Shift

No new solver code was needed — FIBO reuses the existing tt_dit `solvers/euler.py` (`EulerSolver`) unchanged. FIBO's `FlowMatchEulerDiscreteScheduler` config (`use_dynamic_shifting=True`, `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=4096`, exponential shifting) matches the shift math tt_dit already implements (`_calculate_shift`). The diffusers host scheduler computes `mu` and the per-step sigma schedule (via `sigmas`/`mu` passed to `set_timesteps`); the device-side `EulerSolver.step` consumes that schedule directly. This was validated by reproducing the diffusers scheduler's sigma schedule against `EulerSolver`'s internal schedule (see Measured PCCs / results below) — no transformer or VAE weights are needed for this check, only the scheduler config.

### Files

```
models/tt_dit/
  models/vae/
    vae_wan2_1.py                   # modified: WanResidualUpBlock, WanDupUp3D, is_residual branch in WanDecoder3d
  tests/models/bria_fibo/
    test_vae.py                     # FIBO VAE config check + reduced/production decode PCC vs HF reference
    test_solver.py                  # EulerSolver schedule vs diffusers FlowMatchEulerDiscreteScheduler
```

### Running the Tests

```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest models/tt_dit/tests/models/bria_fibo/test_vae.py models/tt_dit/tests/models/bria_fibo/test_solver.py -v
```

**Prerequisites:**

- **Weights**: `briaai/FIBO` is a gated HuggingFace repo. Accept the license on the HF model page, authenticate (`huggingface-cli login`), then pre-download the `vae/*` files (and `scheduler/*` for the solver test), e.g. `huggingface-cli download briaai/FIBO --include "vae/*" "scheduler/*"`. Tests run fully offline (`HF_HUB_OFFLINE=1`) against the local snapshot cache and `pytest.skip` with a clear message if weights are unavailable.
- **`FIBO_PATH`** (optional): Override the HF model path, e.g. `FIBO_PATH=/data/models/FIBO pytest ...`. Defaults to `briaai/FIBO`.
- **Devices**: All sub-project 3 tests run single-device (`mesh_device=(1,1)`); no multi-chip mesh required.

### Measured PCCs (real `briaai/FIBO` VAE weights, bf16 tt vs f32 HF reference, Blackhole)

| Test | Configuration | PCC |
|---|---|---|
| VAE decode, reduced | T=1, 16×16 latent | 99.97% |
| VAE decode, production | T=1, 64×64 latent → 1024×1024 image, single chip | 99.96% |
| Solver schedule vs diffusers | dynamic shift, 30 steps, seq_len=4096 | sigma schedule matches exactly (< 1e-6) |
| Wan 2.1 `test_wan_decoder` (regression) | unchanged (`is_residual=False`) | 99.998% |

All tests pass PCC ≥ 0.99. The production-resolution decode (64×64 latent, matching FIBO's default 1024×1024 output image via `scale_factor_spatial=16` × unpatchify) runs unconditionally on a single Blackhole chip — no OOM-fallback/retry loop is needed at this resolution.

### Out of Scope / Deferred

- **VAE encode** for `is_residual=True` remains blocked (`assert not is_residual` in `WanEncoder3D`) — FIBO's pipeline only needs decode.
- **Pipeline wiring and end-to-end Blackhole bringup** (denoise loop calling the transformer + `EulerSolver`, then this VAE decoder) is sub-project 4, still TODO. Until that lands, the host (`torch`/diffusers) `AutoencoderKLWan` decode is an available stopgap for end-to-end testing.
