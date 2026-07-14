# FIBO on TTNN — Model Documentation

## Overview

This document covers the Tenstorrent tt_dit implementation of Bria's **FIBO** text-to-image model. FIBO is a flow-matching MMDiT (Flux-shaped) model conditioned on an LLM text encoder. The implementation is decomposed into four sub-projects, built in data-flow order. See the design spec for full detail:
[`docs/superpowers/specs/2026-07-06-fibo-smollm3-encoder-design.md`](../../../docs/superpowers/specs/2026-07-06-fibo-smollm3-encoder-design.md)

**Status:** all four sub-projects are done and PCC-validated on the 2×2 Blackhole mesh; the pipeline runs **fully on-device** end-to-end, including the Wan 2.2 VAE decode (see *Sub-project 4 → On-device VAE decode*). FIBO's intended input is a structured JSON caption produced by a host-side VLM — the full natural-language **text → JSON → image** path is documented in the *VLM Front-end* section at the end.

## 4-Sub-Project Decomposition

| # | Sub-project | Status | Strategy |
|---|---|---|---|
| **1** | **SmolLM3 text encoder** | **Done** | New `encoders/smollm3/`; decoder layer from Qwen25VL, all-hidden-states shell from Gemma |
| **2** | **BriaFibo transformer** | **Done** | New `transformer_bria_fibo.py` from Flux1 + per-layer "concat-halves" text injection |
| **3** | **Wan VAE + solver wiring** | **Done** | Reuse `vae_wan2_1.py` (T=1 decode) + `EulerSolver` + dynamic-shift scheduler |
| **4** | **Pipeline + Blackhole bringup** | **Done** | New `pipelines/bria_fibo/`; unpadded per-branch CFG; 2×2 mesh (`cfg=(1,0) sp=(2,0) tp=(2,1)`) |

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

---

## Sub-project 4: End-to-end Pipeline

### Architecture

`BriaFiboPipeline` (`pipelines/bria_fibo/pipeline_bria_fibo.py`) wires the three validated components into one text→image flow-match denoise loop, mirroring `pipelines/flux1/pipeline_flux1.py` with the FIBO deltas below. It runs on the 2×2 Blackhole mesh as a single submesh: `DiTParallelConfig.from_tuples(cfg=(1,0), sp=(2,0), tp=(2,1))`, one `CCLManager`, one `BriaFiboTransformer`, one `EulerSolver`. The encoder runs fully replicated (`tp=1`, no CCL); the VAE decoder is configured hw-parallel `(2,2)`. No inter-stage eviction (all of encoder + transformer + VAE stay resident; no OOM observed).

`__call__(prompt, *, negative_prompt="", height=1024, width=1024, num_inference_steps=30, guidance_scale=5.0, seed=0, latents=None, output_type="pil", force_device_decode=False)`:

1. **Encode** the positive and negative prompts **separately** via the SmolLM3 wrapper (below); build two 46-entry `text_encoder_layers` lists (the 37→46 pad, below); move each branch's prompt + layers + RoPE to the submesh once (reused across all steps).
2. **Latents (no 2×2 pack):** `(1, 48, h, w) → (1, h*w, 48)` with `h=w=height/16` (`in_channels == z_dim == 48`), sequence-sharded on the `sp` axis. `latents=` injects a fixed initial noise in this packed layout (used by the reference-PCC test).
3. **RoPE:** flux-style ids `cat([zeros(T,3), _latent_image_ids(h,w)])` → `pos_embed` → split into a replicated prompt part (`[:T]`) and a sequence-sharded spatial part (`[T:]`); `head_dim=128 = 16+56+56`.
4. **Schedule:** `mu = _calculate_shift(h*w, scheduler)`; `scheduler.set_timesteps(sigmas=linspace(1, 1/N, N), mu=mu)`; `solver.set_schedule(scheduler.sigmas.tolist())`.
5. **Denoise loop:** per step, run the transformer **twice** (once per CFG branch), combine `v = uncond + guidance_scale·(cond − uncond)` (`ttnn.lerp`), then `solver.step`. Per-step deallocations + `synchronize_device` keep DRAM bounded across the untraced loop.
6. **Decode:** all-gather the sp-sharded latent → `(1, 48, 1, h, w)` BCTHW → **on-device** Wan 2.2 residual VAE decode (full 2×2 submesh, hw-parallel) → `unpatchify(patch_size=2)` + `clamp(-1,1)` → `VaeImageProcessor.postprocess`. Pass `force_device_decode=True` to require the on-device path (raise on failure instead of falling back to the host reference decode — see *On-device VAE decode* below). `output_type="latent"` instead returns the pre-VAE latent (for the reference-PCC gate).

### Text Encoder Wrapper + 37→46 Layer Build

`text_encoder.py`'s `SmolLM3TextEncoderWrapper` adapts the sp1 encoder for the pipeline: it tokenizes with `AutoTokenizer` (replicating the reference's empty-prompt special case — a lone `bot_token_id=128000` for `""`), builds RoPE via the encoder's `create_rope_tensors`, calls `.encode()`, and returns `(prompt_embeds[1,T,4096], all_hidden_states)`. `build_text_encoder_layers` implements the reference's list rule exactly: SmolLM3 emits 37 hidden states but the transformer indexes 46 (8 dual + 38 single), so it pads with 9 copies of the last state (and right-trims if ever longer).

### CFG: Unpadded Per-Branch Forwards (key design decision)

The reference batches `cat([negative, positive])` and passes a padding **attention mask**. The tt transformer has no mask parameter, so instead the pipeline runs **batch=1, unpadded, per CFG branch** — each branch encodes at its *true* token length and gets its own transformer forward per step. With no padding there is no mask to apply, making this bit-faithful to the reference (whose mask is a no-op on unpadded tokens). This was confirmed by the end-to-end latent PCC below. Cost: two forwards per step (batched+masked CFG is a perf follow-up).

### On-device VAE decode

The Wan 2.2 residual VAE decode runs **on-device** on the full 2×2 submesh (hw-parallel; the decode's activations spread across all 4 devices, reusing the transformer's `CCLManager`), producing the 1024×1024 image with no host fallback.

Reaching this required a one-line fix in `models/tt_dit/models/vae/vae_wan2_1.py`: `WanVAEDecoderAdapter` built its `WanDecoder` **without** `decoder_base_dim`, so it defaulted to `base_dim` (160) instead of FIBO's *asymmetric* `decoder_base_dim` (256). That made `decoder.conv_in` a `(1728, 640)` Parameter while the real weight is `(1728, 1024)` → a `LoadingError` at weight load, **on every mesh size** (this was originally mis-attributed to a hw-parallel limitation; sp3's `test_vae` never hit it because it constructs `WanDecoder` directly, bypassing the adapter). The adapter now passes `decoder_base_dim=getattr(config, "decoder_base_dim", None)`; Wan 2.1 configs omit the field, so `WanDecoder` defaults `None → base_dim` and the Wan 2.1 path is unchanged.

`_decode_vae` still keeps the host-torch `AutoencoderKLWan.decode` as a *defensive* fallback on `LoadingError`, but it no longer fires. `force_device_decode=True` re-raises instead of falling back, so a regression fails loudly (used by the smoke + the on-device decode test).

### Files

```
models/tt_dit/
  pipelines/bria_fibo/
    __init__.py
    text_encoder.py                 # SmolLM3TextEncoderWrapper + build_text_encoder_layers (37->46)
    pipeline_bria_fibo.py           # BriaFiboPipeline, __call__, _decode_latents, CFG, on-device VAE decode
  tests/models/bria_fibo/
    test_pipeline.py                # layer-build (host), wrapper-encode PCC, latent PCC, image smoke,
                                    #   e2e-image golden, on-device VAE decode golden
    test_vlm_pipeline.py            # full product path: text -> FIBO-vlm (CPU) -> JSON -> TT pipeline -> image
```

### Running the Tests

```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest models/tt_dit/tests/models/bria_fibo/test_pipeline.py -v
```

**Prerequisites:**

- **Weights**: `briaai/FIBO` (gated). Pre-download all components used by the pipeline: `huggingface-cli download briaai/FIBO --include "text_encoder/*" "tokenizer/*" "transformer/*" "vae/*" "scheduler/*"`. Tests run offline (`HF_HUB_OFFLINE=1`) and `pytest.skip` if weights are absent.
- **Devices**: the pipeline tests run on the 2×2 Blackhole mesh. The end-to-end latent PCC also loads the diffusers reference transformer on CPU as the oracle, so it uses a reduced resolution (512²) to stay tractable — the smallest resolution whose spatial token count is a clean multiple of the ring joint-SDPA constraint `k_chunk_size(512) · sp(2) = 1024` (512² → 32×32 = 1024 tokens), so no attention-padding code is needed.

### Measured PCCs (real `briaai/FIBO` weights, bf16 tt vs f32 HF reference, 2×2 Blackhole)

| Test | Configuration | Result |
|---|---|---|
| Text-encoder wrapper vs reference `get_prompt_embeds` | representative prompt | 99.79% |
| End-to-end latent trajectory vs diffusers `BriaFiboPipeline` | 512² (32×32 latent, 1024 tokens), 2 steps, identical injected noise | 99.69% |
| End-to-end latent trajectory (higher-confidence, committed test) | 512², 4 steps | 99.12% |
| On-device VAE decode | 1024², synthetic latent | runs on-device; correct-range `(1,3,1,1024,1024)` image (golden PCC-vs-host is an on-demand test) |
| Full image smoke (on-device decode) | 1024², 30 steps, `force_device_decode=True` | runs; finite, non-degenerate `(1024,1024,3)` image |

The end-to-end latent PCC is measured against a full-precision (fp32) diffusers reference fed *identical* injected noise (`latents=`), on the pre-VAE latent. PCC declines slightly with step count (99.69% @ 2 steps → 99.12% @ 4 steps) — accumulated bf16 drift over solver steps, not a wiring defect (a glue bug would crater PCC far below 0.99 at any step count). The glue under test (noise → no-patch pack → RoPE → per-branch CFG → Euler solver → latent) is resolution-independent, so the reduced-resolution gate transfers to production resolution, where each component carries its own production PCC (sp1–sp3). The committed test runs 4 steps.

### Out of Scope / Deferred

- **Perf**: the bringup above was functional-first (untraced, DRAM-interleaved). A subsequent optimization pass (opt-in denoise tracing, per-shape matmul + conv3d blockings, `num_links=4` CCL, CFG gating) is documented in its own section — see **[Performance Optimization History](#performance-optimization-history)** — and took the traced pipeline to ~13.84 s. The remaining large lever is **batched+masked CFG** (~2× by halving the per-step forwards; needs transformer attention-mask support), still deferred.
- **On-demand golden tests**: `test_fibo_pipeline_vae_decode_on_device` (native-res 1024² VAE golden vs host) and `test_fibo_pipeline_e2e_image_golden` (reduced-res image golden vs the diffusers reference) are committed but slow (host reference decode), so they run on-demand rather than in fast CI.

---

## Sharding & Parallelism

FIBO runs on the **2×2 Blackhole mesh as a single CFG submesh** (`cfg=(1,0)`) spanning both axes: `sp` (sequence-parallel) on mesh axis 0, `tp` (tensor-parallel) on mesh axis 1. The three stages use **three different sharding regimes** on that same mesh:

| Stage | Regime | What is split | What is replicated |
|---|---|---|---|
| **Encoder** (SmolLM3) | fully replicated (`tp` factor 1) | nothing | the whole encoder — every device runs it, no CCL |
| **Transformer** (denoise) | `sp=2` × `tp=2` | spatial sequence (sp) + feature/hidden dim + weights (tp) | prompt tokens (across sp), prompt RoPE |
| **VAE decode** (Wan 2.2) | hw-parallel `(2,2)` | image height (tp axis) + width (sp axis) — a spatial quadrant per device | — |

### Transformer sharding (`sp=2` × `tp=2`)

- **Sequence parallel (`sp=2`, axis 0):** the spatial latent `(1, h·w, 48)` is sequence-sharded on dim 1 (`mesh_axes=[None, sp_axis, None]`) — each sp device holds *half the spatial tokens*. FeedForward and the per-token ops run independently on each half. The spatial RoPE tables are sharded the same way (`mesh_axes=[sp_axis, None]`).
- **Tensor parallel (`tp=2`, axis 1):** the `inner_dim=3072` (= 24 heads × 128) feature dimension and the block weights are fractured across the tp axis — each tp device holds a *1536-wide feature shard*. `context_embedder`/`x_embedder` are `ColParallelLinear` (output feature-sharded on tp); attention/FF gather and scatter activations across the tp axis with `all_gather`/`reduce_scatter`. Ring attention (`RingJointSDPA`) overlaps the KV all-gather with compute.
- **Replicated:** the prompt tokens and prompt RoPE are not sharded across sp (attention needs the full prompt on every sequence shard).
- At the end of the forward, the tp-sharded spatial activation is `all_gather`-ed across the tp axis (dim 2) before the final `proj_out`.

### The `tp=2` injection masking (splitting the concat across two devices)

This is the masking that arose from splitting work over the two tp devices. Before every block, FIBO's "concat-halves" injection replaces the **upper half** of the context features (`[1536:3072]`) with the per-block projected text, keeping the lower half (`[0:1536]`) as the running prompt:

```
out[..., :1536] = context[..., :1536]     # kept prompt half
out[..., 1536:] = projected_text          # replaced half
```

At **tp=2 the 3072 feature dim is sharded exactly at the 1536 boundary**, so tp-device 0 owns `[0:1536]` (the kept half) and tp-device 1 owns `[1536:3072]` (the replaced half). A naive concat across a *sharded* feature dim has no cheap primitive (it would need an all-gather). Instead FIBO precomputes two per-device `bf16` masks once (`_InjectionMask`):

- `keep` — `1` on the kept-prompt features, `0` elsewhere; sharded on the tp axis → all-ones on device 0, all-zeros on device 1.
- `take` — the complement → all-zeros on device 0, all-ones on device 1.

The injection is then a **gather-free, per-device select** — `context_local * keep + projected * take` (`inject_text`) — with **no CCL**. `projected` comes from a *replicated* `Linear`, so it's the full 1536-wide vector on every device; the `take` mask keeps only the slice that device owns. This is only valid at `tp==2` (asserted), because it requires the per-device feature-shard width (1536) to equal exactly half the inner dim. At `tp=1` the injection falls back to a plain `ttnn.concat`.

### VAE hw-parallel decode `(2,2)` + halo exchange

The Wan 2.2 residual decoder is **height/width parallel** (`VaeHWParallelConfig`: height on the tp axis, width on the sp axis), so the 64×64→1024×1024 spatial decode is split into **quadrants across all 4 devices** — each device decodes ~1/4 of the spatial extent. It uses its **own** `CCLManager` (`_vae_ccl_manager`, separate from the transformer's so it can't disturb the resident denoise trace). Each conv only needs its neighbors' boundary pixels, so instead of gathering full activations the decoder does a **halo exchange** (`neighbor_pad_persistent_buffer` → `NeighborPadAsync`): it swaps only the `kernel//2`-wide border rows (H) / columns (W) with neighbor devices. Two kinds of padding-masking keep this correct at shard edges:

- **Height padding** — zeroing rows at/beyond `logical_h` is *fused into* `neighbor_pad` via the `logical_h` parameter (no separate mul-mask op).
- **Width padding** — a pre-conv `mul` by `_get_w_mask` zeros the width-padding columns before the halo (no C++ fusion for `logical_w` yet), so the exchange doesn't propagate non-zero padding.

### Why the Wan 2.2 VAE decode is fast

The decode dropped to **0.33 s** (from 2.33 s) for five compounding reasons:

1. **Single-frame (`T=1`) path.** Wan is a *video* VAE, but FIBO decodes one latent frame, so the expensive temporal machinery (multi-frame chunking, causal-T convolutions, temporal upsample) degenerates to `T=1` — the real temporal-upsample convs emit a single output frame.
2. **hw-parallel `(2,2)` quadrant split.** Each of the 4 devices does only ~1/4 of the spatial work (see above).
3. **Halo exchange instead of full-activation gather.** Only the thin conv borders cross device boundaries, and that halo is bandwidth-bound so it rides `num_links=4` (the W-halo uses all 4 links; the H-halo self-caps to 1 since its upper dims `B·T=1`). This makes the spatial parallelism nearly free — CCL is only ~5% of decode.
4. **Cheap, mostly-fused padding masking** (fused H-mask, single W pre-mask) — no per-conv mask overhead.
5. **Per-shape conv3d blocking tuning — the dominant win.** This is the actual optimization (see *Performance Optimization History → Conv3d blocking tuning*): all 13 decode conv3d shapes were missing the `_BLOCKINGS` table and fell back to `H_block=W_block=1` (one output pixel per core). The tuned blockings tile the output across the full compute grid, giving **15–86× per-op** and driving the 2.33 s → 0.33 s result.

---

## Performance Optimization History

Sub-projects 1–4 above were **functional-first**: correct on the 2×2 Blackhole mesh, but untraced and DRAM-interleaved. This section records the optimization work layered on top afterwards. All numbers are the 1024×1024, 30-step, `guidance_scale=5.0` pipeline on the 2×2 Blackhole (P150) mesh unless noted; commit hashes reference the `fibo-pipeline` branch.

**Net effect of the shipped work: traced end-to-end pipeline ~15.84 s → ~13.84 s, with VAE decode 2.33 s → 0.33 s and denoise 2.00 → 2.33 it/s — all PCC-neutral.**

### What FIBO inherits from Flux1 (the shared DiT baseline)

The FIBO transformer was built *on top of* the existing tt_dit Flux1 DiT (`BriaFiboTransformer` is "adapted from `transformer_flux1.Flux1Transformer`"), so most of the heavy device-side performance machinery is **reused, not re-written**. What is reused unchanged vs. overridden/added:

| Component | Source | FIBO treatment |
|---|---|---|
| Single transformer block (×38) | `transformers/transformer_flux1.py::Flux1SingleTransformerBlock` | **reused unchanged** (imported directly) |
| Dual transformer block (×8) | `blocks/transformer_block.py::TransformerBlock` | **reused unchanged** (shared MMDiT dual block, also used by SD3.5 etc.) |
| Top-level DiT | `transformer_flux1.py::Flux1Transformer` | **adapted** → `BriaFiboTransformer` (deltas below) |
| Sequence-parallel + tensor-parallel scheme | `parallel/` config + blocks | **reused**; FIBO selects `cfg=(1,0) sp=(2,0) tp=(2,1)` |
| Ring attention / `RingJointSDPA` (overlaps KV all-gather with compute) | inside the shared blocks | **reused**; FIBO adds one `(is_blackhole, sp, tp)` chunk-size entry `(True,2,2)→(128,512)` |
| CCL collectives (`all_gather_persistent_buffer`, `reduce_scatter_minimal_async`) + `CCLManager` | shared `parallel/manager.py` | **reused** (persistent pre-allocated buffers) |
| `minimal_matmul` fast path (via `Linear`/`ColParallelLinear`) | shared `layers/linear.py` | **reused**; FIBO registers per-shape block configs (below) |
| `Tracer` (capture/replay a resident device trace) | `utils/tracing.py` | **reused** (opt-in denoise trace; below) |
| Timestep embedding | Flux uses pooled + guidance embeds | **overridden** → `BriaFiboTimestepEmbed` (timestep-only) |
| `context_embedder` / `x_embedder` dims | Flux dims | **overridden** (4096→3072 / 48→3072) |
| Per-block text injection | none in Flux | **net-new** → `inject_text` + `_InjectionMask` (concat-halves) |
| `caption_projection` (46× `Linear(2048→1536)`) | none in Flux | **net-new** |

Contrast worth noting: the Flux1 pipeline traces **all three** stages (denoise + VAE + encoder) and defaults `num_links=2` on its (Wormhole) 2×4 mesh; FIBO traces **only denoise** (encoder tracing corrupts the image, VAE decode is compute-bound — see *Tried and reverted*) and defaults `num_links=4` (the Blackhole 2×2 hardware max). Flux also avoids CFG entirely via learned guidance embeds, whereas FIBO runs real CFG — which is why the *CFG gating* optimization below is FIBO-specific.

### Shipped optimizations (committed on `fibo-pipeline`)

Denoise (~95% of the pipeline):

- **Per-shape matmul block-size tuning** (`efc2dd2d586`, `1b829d82ae3`). Every FIBO block matmul takes the non-AGMM `minimal_matmul` path, which was falling back to a generic `(8,8,8)` blocking. Added a `bh_2x2` device config + FIBO's 19 matmul shapes to `utils/sweep_mm_block_sizes.py`, swept on the profiler, and registered the winners as per-`(M,K,N)` `MinimalMatmulConfig`s at import (`transformer_bria_fibo.py::_register_fibo_matmul_configs`, keyed under grid `12x10`, additive so other models are unaffected). → matmul device-time **91.4 → 86.2 ms (−5.8%)**, whole forward 235 → 230 ms (−2.2%), PCC 99.53% (unchanged).
- **Denoise transformer tracing** (`f642cd95d6e`, `1eef9fcd8a5`, opt-in `traced=`). Capture one resident device trace of the transformer forward and replay it each of the 30 steps, removing per-step host dispatch. Prototype measured ~1.28× on the denoise step; the follow-up fix verified it captures exactly one trace/step. Trace region needs ≥71 MB (default 50 MB).
- **CCL `num_links` 1 → 4** (`bc8db8971ed`, `e07c05a65a3`). The sp/tp collectives are bandwidth-bound; striping each across all 4 physical ethernet links (the 2×2 P150 hardware max) sped denoise **2.00 → 2.33 it/s** and total pipeline **15.84 → 13.84 s (−13%)**, correctness-neutral (transport-only). Set as the default in `BriaFiboPipelineConfig`.

VAE decode (was ~98% on-device Wan-VAE conv3d):

- **Conv3d blocking tuning** (`fbaac226ef6`). All 13 decode conv3d shapes were *missing* the `utils/conv3d.py::_BLOCKINGS` table and fell back to a pathological `H_block=W_block=1` (one spatial element per core). Swept each shape (`tests/models/wan2_2/bruteforce_conv3d_sweep.py`) and registered tuned blockings → **decode 2.33 s → 0.33 s (−86%)**, PCC 0.99997; per-op speedups 15–86×.
- **Temporal-upsample conv3d blockings** (`191756ca539`). The last 2 `(3,1,1)` tconvs were still on the fallback; registered their blockings (per-op 61–80×). End-to-end decode already dominated by other stages, so this was for table completeness/consistency.

Pipeline-level:

- **CFG gating on `guidance_scale > 1`** (`5d78fcf398f`). At `gs ≤ 1` the uncond branch is mathematically dead (`noise = uncond + 1·(cond − uncond) = cond`), so the uncond encode/prepare/forward are skipped → **~2× cheaper denoise on the `gs ≤ 1` path**; the default `gs=5` path is byte-for-byte unchanged. Mirrors the diffusers reference.
- **Tokenizer `max_length=3000` clamp** (`dc8aece0110`). Correctness fix (matches the diffusers reference default) that also bounds encoder input length instead of the tokenizer's 131072 default.

Measurement infrastructure (enabled everything above):

- **Tracy per-op device-profiling tests + per-stage wall-clock harness** (`77e42dca28e`, `874654575d5`, `e447f3d265e`, `2dbf3e5e563`). Profile the **DIT transformer** test for per-forward denoise cost (not the full pipeline — that overflows the profiler marker buffer and asserts on device asymmetry). Recipe and profiler knobs are in the team's profiling notes.

### Tried, measured, and reverted — do NOT redo

These represent real spent effort; each was reverted (not in git) after measurement:

- **L1 activation residency ("put everything in L1")** — reverted. The forward is only ~15.7% of DRAM roofline (compute-bound, not bandwidth-bound). SDPA is hardware DRAM-only; interleaved L1 clashes with static CBs; the subset that *fits* is bit-exact but perf-neutral. Broad L1 would need a full sharded-matmul rewrite for a capped payoff.
- **Op-count fusion (spatial+prompt stream-concat)** — reverted. Cut matmul ops 398 → 322 (−15.6% matmul time) but only −5.1% whole denoise; the added concat/slice TM ops ate most of the win.
- **`RingJointSDPA` chunk-size tuning** — dead end. Best valid config was ~noise-floor (+0.5%) and drifted the image to PCC 0.97 (accumulation-order change compounds over 30×46 SDPA applications).
- **Encoder (SmolLM3) tracing** — reverted. Real −20% on encode, but a naive 2nd resident trace corrupts the image (PCC 0.18); correct multi-trace pre-allocation is real work, and encode is only ~2.6% of the pipeline.
- **Small-M matmul core-grid fix** — empirically refuted; the full 12×10 grid is already optimal (smaller grids 34–369% slower).
- **Full-spatial conv3d re-sweep** — found 5–8% per-op gains but no measurable end-to-end decode gain (decode already below the noise floor); reverted.

---

## VLM Front-end: Full Text → JSON → Image Path

FIBO is a **two-stage** system. Sub-projects 1–4 above are the diffusion half (**structured JSON → image**). FIBO was trained on structured JSON captions, not free-form text, so its intended input is a JSON string; the front-end that turns a user's natural-language prompt (or a reference image) into that JSON is a separate **VLM**.

### The VLM

- **`briaai/FIBO-VLM-prompt-to-JSON`** — a tiny (~5-file) remote-code `ModularPipelineBlocks` that wraps the model. Despite the "Gemini" title in its README, the code loads a **local** model, not an API.
- **`briaai/FIBO-vlm`** — the actual weights: a **Qwen3-VL** model (~8.9 GB, 2 safetensors shards, public/not-gated). `transformers` ≥ 5.10 provides `Qwen3VLForConditionalGeneration`.
- It runs **on host CPU** (bf16): the model loads in ~0.8 s and generates the JSON caption in ~87 s (greedy). Output is a minimal JSON string with FIBO's caption schema: `short_description`, `objects`, `background_setting`, `lighting`, `aesthetics`, `photographic_characteristics`, `style_medium`, `text_render`, `context`, `artistic_style`.
- **Gotcha:** the block's `__init__` hardcodes `self.engine.model.to("cuda")` (no CUDA here). We bypass the `ModularPipeline` and call its `TransformersEngine("briaai/FIBO-vlm")` + `generate_json_prompt(...)` helpers **directly**, which load/run on CPU.

The VLM is **not** ported to Tenstorrent (out of scope) — it runs on the host and its compute is small relative to the diffusion pipeline. Running the diffusion pipeline does not *require* the VLM: you can hand-write a JSON prompt, or pass free text (out-of-distribution — still produces an image, just without FIBO's structured control).

### Files

```
models/tt_dit/tests/models/bria_fibo/
  test_vlm_pipeline.py              # text -> FIBO-vlm (CPU) -> JSON -> BriaFiboPipeline (TT) -> 1024x1024 image
```

The pipeline itself is unchanged — it already takes a string prompt, so the JSON string flows straight through `SmolLM3TextEncoderWrapper`.

### Running the Test

```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_vlm_pipeline.py -v -s
```

**Prerequisites** (in addition to the sub-project 4 ones):

- **Weights**: `briaai/FIBO-vlm` (~8.9 GB) and `briaai/FIBO-VLM-prompt-to-JSON` cached (both public). The test `pytest.skip`s if absent.
- **Deps**: the block needs `ujson` + `boltons` (`python_env/bin/python -m pip install ujson boltons`; if the venv has no pip, `python_env/bin/python -m ensurepip --upgrade` first).
- **Slow**: host CPU autoregressive decode of the JSON (~87 s) + the TT generation (~90 s) ≈ 3 min. On-demand, not fast CI.

### Result

`test_fibo_vlm_to_image_e2e` **passes**: free text → valid structured JSON (3490 chars) → non-degenerate 1024×1024 image on the 2×2 Blackhole mesh (on-device decode). Artifacts: `fibo_vlm_prompt.json` (the intermediate JSON) + `fibo_vlm_e2e.png`.
