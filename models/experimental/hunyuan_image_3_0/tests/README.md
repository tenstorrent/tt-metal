# HunyuanImage-3.0 tests

Layout and run commands for the consolidated test suite.

## Directory map

| Directory | Purpose |
|-----------|---------|
| `pcc/` | On-device PCC gates vs `ref/` — transformer, MoE, pipeline, denoise, recaption, embeddings |
| `tokenizer/` | Host-side tokenizer / input-bundle parity (no device required for most tests) |
| `vision/` | SigLIP2, cond-image preprocess/inject, image processor |
| `vae/` | VAE encoder/decoder block PCC, conv3d, spatial helpers |
| `perf/` | Tracy profiling and conv3d blocking sweeps (optional, long-running) |

## Shared conventions

- **ISL:** S=1, 32 fast; S=4096, 4160 slow; S=22784 max-context slow
- **Latent grid:** GRID=8 (S=128) fast; GRID=64 (S=4160) production slow
- **Layers:** `HY_NUM_LAYERS` env (defaults vary by test module)
- **Weights:** `HUNYUAN_MODEL_DIR`, `HUNYUAN_INSTRUCT_MODEL_DIR` via `resolve_base_model_dir()`
- **Upstream HF parity:** set `HUNYUAN_UPSTREAM` to a local clone of the `hunyuan_image_3` package
- **Device:** function-scoped `device` in `pcc/conftest.py`; mesh tests use `mesh_device` fixture
- **Marks:** `@pytest.mark.slow` for production scale, mesh, full-depth, I2I, recaption trace

## Smoke vs production

| Tier | Command | Scale | Purpose |
|------|---------|-------|---------|
| **Smoke (fast CI)** | `pytest tests/pcc/ -m "not slow and not unit_host and not e2e_random_inputs"` | 2–8L, GRID=8, S≤32 | Submodule gates, quick regression |
| **Production (slow CI)** | `bash tests/run_pcc_production_slow.sh` | **32L**, GRID=64, S=4160, S=22784, decode S=1, denoise loop ×50 | Full HF width + production ISL |

Production slow gate covers: teacher-forced 32L (decode + prefill), chained backbone decode,
full logit stack (including **max-context S=22784**), MoE, patch_embed GRID=64, denoise step 32L,
**I2I denoise step 32L** (instruct checkpoint), submodule gates for **RMSNorm**, **RoPE**,
**attention mask**, **attention**, **decoder layer**, **WTE @ S=4160**, **lm_head @ S=4160
last-token**, and **32L recaption** greedy token parity.

Vision (SigLIP), VAE, **scheduler** (`test_scheduler.py`), and **timestep embedders**
(`test_embeddings.py` — `timestep_emb`, `time_embed`, `time_embed_2`) have **smoke** cases
plus **full-dim slow** cases: vision `test_siglip2_full_dim.py` (S=1024 / 27L),
scheduler/CFG at 64×64 latent, timestep over the 50-step FlowMatch schedule, and
on-device AR `test_generate_device.py` (full vocab × H=4096).

## Compliance checklist (backbone + denoise production CI)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | All backbone modules + PCC at full size (32L, GRID=64, S=4160, S=22784) | **Pass** — 36 tests in `run_pcc_production_slow.sh` |
| 2 | Tests adhere to PCC; exceptions documented in README | **Pass** — table + gated `unit_host` / `e2e_random_inputs` |
| 3 | Each module isolated at full HF config | **Pass** — layer-0 gates incl. text + image layout @ S=4160 |
| 4 | Real weights; single-layer + all-layer prefill + decode | **Pass** — random activations only; WTE + lm_head + logit stack @ S=4160 |

Vision S=1024/27L, on-device AR full-vocab, scheduler 64×64, and timestep×50 are in the
production script’s second section (`HY_SKIP_FULL_DIM_SUPPORT=1` to skip). VAE and recaption
trace/2CQ remain opt-in outside that gate (recaption **32L greedy** is in backbone CI).

## PCC thresholds

See the full exception table in [`../README.md`](../README.md#pcc-thresholds-and-exceptions).
Key constants from `pcc/pcc_common.py`:

| Constant | Value | Typical use |
|----------|------:|-------------|
| `PCC_STRICT` | 0.999 | RMSNorm, RoPE, MoE |
| `PCC_BLOCK` | 0.99 | Attention, decoder layer, teacher-forced final |
| `PCC_CHAINED` | 0.86 | 32L chained free-running final |
| `PCC_DECODE_STACK` | 0.96 | Backbone free-running decode at S=1 |
| `PCC_LOGIT_DECODE` | 0.96 | Logit stack **teacher-forced** decode at S=1 |
| `PCC_LOGIT_PREFILL` | 0.85 | 32L last-token logits at S=4160 |
| `PCC_LOGIT_MAX_CONTEXT` | 0.85 | 32L last-token at S=22784 (production slow CI) |

Denoise step uses dynamic thresholds from `pipeline_helpers.pipeline_pcc_threshold`
(0.99 for ≤8L bf16, **0.85 for 32L bf16** — observed ~0.983). E2e uses `HY_LATENT_PCC=0.98`,
`HY_RGB_PCC=0.97`.

## PCC tests (`pcc/`)

| File | Covers |
|------|--------|
| `test_attention_modules.py` | RoPE, mask, attention blocks |
| `test_transformer.py` | Backbone layer/stack + mesh SP/TP/EP |
| `test_embeddings.py` | patch_embed, final_layer, timestep (smoke + **50-step full schedule**), WTE |
| `test_moe.py` | router, expert FFN, parallel MoE |
| `test_pipeline.py` | on-device denoise step, e2e pipeline, model forward |
| `test_denoise.py` | denoise loop, I2I denoise step/loop |
| `test_recaption.py` | on-device recaption, trace, 2CQ |
| `test_teacher_forced.py` | teacher-forced backbone (full depth) |
| `test_logit_stack.py` | 32L wte→layers→ln_f→lm_head logit PCC |
| `test_scheduler.py` | FlowMatch + CFG (smoke 8×8; **slow 64×64** latent) |
| `test_lm_head.py` | LM head PCC (smoke @ S=32; production last-token @ S=4160) |
| `test_generate.py` | host generate helpers (`@pytest.mark.unit_host`; excluded from smoke) |
| `test_generate_device.py` | On-device AR at **full vocab × H=4096** (token parity vs host; 32L slow) |
| `test_full_dim_moe_denoise.py` | MoE @ **S=22784** + denoise loop **32L × GRID=64 × 50 steps** |

Helpers: `pcc_common.py`, `pipeline_helpers.py`, `denoise_helpers.py`, `recaption_helpers.py`, `i2i_helpers.py`, `mesh_helpers.py`

### Fast PCC sweep (no slow marks)

```bash
cd tt-metal
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/ \
  -m "not slow and not unit_host and not e2e_random_inputs" -v
```

### Production slow CI gate (32L, GRID=64, S=4160, S=22784)

Runs teacher-forced **32L decode (S=1)** + **prefill (S=4160)**, **chained backbone decode**,
**full logit stack** (including max-context), **MoE**, **patch_embed GRID=64**, **32L denoise step**,
**I2I denoise step 32L** (instruct weights), submodule gates (**RMSNorm**, **RoPE**, **mask**,
**attention**, **decoder layer** at text + image layout), **WTE @ S=4160**, **lm_head last-token
@ S=4160**, and **32L recaption**; then full-dim support (**vision S=1024/27L**, **AR full vocab**,
**scheduler 64×64**, **timestep ×50**). Requires base + instruct checkpoint weights.

```bash
cd tt-metal
bash models/experimental/hunyuan_image_3_0/tests/run_pcc_production_slow.sh
```

Opt-in E2E with random latent/text inputs (not part of production gate):

```bash
HY_RUN_E2E_RANDOM=1 python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py -k e2e_pipeline -v -s
```

Equivalent manual invocation:

```bash
HY_NUM_LAYERS=32 python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_teacher_forced.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_logit_stack.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_moe.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_embeddings.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_attention_modules.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_recaption.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_lm_head.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise.py \
  -m slow \
  -k "production_32l or production_64 or logit_stack_production or logit_stack_max_context or moe_module_production or moe_router_production or all_layers_production or final_production or denoise_step_production_32l or rms_norm_production or rms_norm_max_context or rope_2d_production or mask_production or attention_production or attention_max_context or decoder_layer_production or decoder_layer_max_context or wte_production or recaption_production or lm_head_production or i2i_denoise_step_production" \
  -v -s --timeout=10800
```

### Production / mesh (slow)

```bash
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer.py -k mesh -v -s

python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise.py -v -s
```

Run one device pytest job at a time — concurrent runs cause device timeout.

## Vision tests (`vision/`)

| File | Covers |
|------|--------|
| `test_siglip2_ttnn.py` | Smoke SigLIP2 + aligner (default S=64, 1L; `HY_VIT_NUM_LAYERS=27` optional) |
| `test_siglip2_full_dim.py` | **Full dim** S=1024 patches (32×32) + 27L vision / e2e aligner (`@pytest.mark.slow`) |
| `test_cond_*` / `test_image_processor.py` | Preprocess, inject, processor |

```bash
# Full vision dimension
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/vision/test_siglip2_full_dim.py -v -s --timeout=10800
```

## Tokenizer tests (`tokenizer/`)


| File | Covers |
|------|--------|
| `test_model_inputs.py` | chat template, prepare_model_inputs parity, attention mask |
| `test_recaption_inputs.py` | recaption inputs, AR bundle, stage params |
| `test_cond_vae_encode.py` | cond VAE encode path |

```bash
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/tokenizer/ -v
```

## VAE tests (`vae/`)

| File | Covers |
|------|--------|
| `test_decode_pipeline.py` | decode_latent glue + spatial full-res decode |
| `test_encoder.py` / `test_decoder.py` | block PCC |
| `test_conv3d_*.py` | conv3d sharding/chunk |

```bash
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/vae/test_decode_pipeline.py -k glue -v -s
```

## Perf / profiling (`perf/`)

| File | Purpose |
|------|---------|
| `test_encoder_perf_tracy.py` | Tracy encoder sweep |
| `test_encoder_conv3d_sweep.py` | conv3d blocking brute-force |
| `test_vae_decode_perf.py` | VAE decode perf |

These are optional optimization harnesses — not CI gates.
