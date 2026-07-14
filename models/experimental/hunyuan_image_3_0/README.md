# HunyuanImage-3.0 (tt-metal)

Experimental TTNN port of [Tencent HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0).

## Tokenizer

Host-side tokenizer code and assets live under `ref/tokenizer/`:

| Path | Description |
|------|-------------|
| `ref/tokenizer/hunyuan_tokenizer.py` | Public API (`HunyuanTokenizer`) |
| `ref/tokenizer/gen_image_inputs.py` | Host preprocess bundle for device upload |
| `ref/tokenizer/assets/config.json` | Model config used by the tokenizer stack |
| `ref/tokenizer/assets/tokenizer_config.json` | HF tokenizer config |
| `ref/tokenizer/assets/tokenizer.json` | BPE vocab (~24 MB; not in git) |

Download `tokenizer.json` (and refresh `tokenizer_config.json`) from Hugging Face:

```bash
cd ~/ign-tt/tt-metal
mkdir -p models/experimental/hunyuan_image_3_0/ref/tokenizer/assets

hf download tencent/HunyuanImage-3.0 \
  tokenizer.json tokenizer_config.json \
  --local-dir models/experimental/hunyuan_image_3_0/ref/tokenizer/assets
```

If `hf` is not installed, use `huggingface-cli download` with the same arguments.

Verify:

```bash
ls -lh models/experimental/hunyuan_image_3_0/ref/tokenizer/assets/tokenizer.json
```

Load check:

```bash
python3 -c "from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer; HunyuanTokenizer.from_pretrained(); print('OK')"
```

Sanity validation:

```bash
python3 -m models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer
```
# HunyuanImage-3.0 — TTNN port

TTNN (Tenstorrent) port of Tencent's **HunyuanImage-3.0**, a unified autoregressive
multimodal model that generates images by running a diffusion (flow-matching)
denoise loop *inside* a 32-layer MoE transformer backbone.

Reference (PyTorch) lives under `HunyuanImage-3.0/hunyuan_image_3/` and is mirrored,
block-for-block, under `ref/`. The device port lives under `tt/`. Every block has a
PCC test under `tests/pcc/` (and `tests/vae/`) that gates the TT block against `ref/`.

---

## The three variants

All three share the **same MoE transformer backbone** (32 layers, 64 experts, top-8,
hidden 4096). They differ only at the edges:

| Variant | Adds over base | Incremental port cost |
|---|---|---|
| **HunyuanImage-3.0** (base, text→image) | — | runs end-to-end on the 2×2 mesh (`demo/demo.py`); accuracy/scale hardening remains |
| **HunyuanImage-3.0-Instruct** (image→image, reasoning) | SigLIP2 vision encoder, image-input preprocessing, recaption/think token generation | **large** — net-new vision + AR-text path |
| **HunyuanImage-3.0-Instruct-Distil** | 8-step meanflow + cfg_distilled (no CFG) | **small** — distill embedders + scheduler reuse |

**Strategy:** finish base T2I end-to-end first (it unblocks all three), add Distil
(nearly free), then build the Instruct I2I stack.

---

## Running demos

Run from the `tt-metal` repo root (`cd ~/ign-tt/tt-metal`). All three use the 2×2 mesh
and default to 32 backbone layers (`HY_NUM_LAYERS=32`).

| Variant | Demo | Steps | CFG | Checkpoint env / default path |
|---|---|---:|---|---|
| **Base** T2I | `demo/demo.py` | 50 | yes (`HY_GUIDANCE=5.0`) | `HUNYUAN_MODEL_DIR` (or HF `tencent/HunyuanImage-3.0`) |
| **Instruct** I2I | `demo/demo_i2i.py` | 50 | yes (`HY_GUIDANCE=2.5`) | `HUNYUAN_INSTRUCT_MODEL_DIR` (or HF `tencent/HunyuanImage-3.0-Instruct`) |
| **Instruct-Distil** I2I | `demo/demo_i2i.py --distil` | 8 | no (distilled) | `HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR` (HF `tencent/HunyuanImage-3.0-Instruct-Distil`) |

### Base text-to-image

```bash
HY_NUM_LAYERS=32 HY_GUIDANCE=5.0 python_env/bin/python \
  models/experimental/hunyuan_image_3_0/demo/demo.py \
  "a photo of a cat, studio lighting"
```

Base T2I defaults to **50 denoise steps** and **CFG 5.0**, matching HF
`tencent/HunyuanImage-3.0` `generation_config.json` (`diff_infer_steps=50`,
`diff_guidance_scale=5.0`). Override with `HY_STEPS` / `HY_GUIDANCE`. Matches upstream
`generate_image` with `bot_task=image`: **no recaption** by default (prompt used verbatim).
Optional AR recaption (`HY_RECAPTION=1`) rewrites the prompt via the text-sampling loop
(`ref/generate.py`, re-exported by `tt/generate.py`) before the gen-image block.

| Env | Default | Meaning |
|---|---|---|
| `HY_STEPS` | `50` | Denoise steps (HF base default) |
| `HY_GUIDANCE` | `5.0` | Classifier-free guidance scale (HF base default) |
| `HY_RECAPTION` | `0` | `1` enables optional recaption/think before image gen |
| `HY_BOT_TASK` | `recaption` | `recaption` / `think` / `think_recaption` |
| `HY_MAX_NEW_TOKENS` | `512` | AR token budget — caps recaption latency |
| `HY_TEMPERATURE` | `0.6` | sampling temperature |
| `HY_TOP_K` | `1024` | top-k filter (0 disables) |
| `HY_TOP_P` | `0.95` | nucleus top-p (1.0 disables) |
| `HY_REP_PENALTY` | `1.0` | repetition penalty (1.0 disables) |
| `HY_DO_SAMPLE` | `1` | `0` = greedy argmax |
| `HY_TRACE` | `1` | **Master switch.** `1` = 2CQ mesh + recaption AR trace (when recaption runs) + denoise CFG ``execute_trace`` when steps > ``HY_DENOISE_TRACE_MIN_STEPS``. `0` = eager 1CQ everywhere |
| `HY_DENOISE_TRACE` | auto | `1` / `0` force denoise trace on/off. Default **auto**: off when denoise steps ≤ ``HY_DENOISE_TRACE_MIN_STEPS`` (capture overhead does not amortize on short loops) |
| `HY_DENOISE_TRACE_MIN_STEPS` | `8` | Auto-disable denoise ``execute_trace`` when step count is at or below this (Instruct-Distil 8-step path) |
| `HY_VAE_DECODE_TRACE` | `0` | `1` = CQ0 ``execute_trace`` for final RGB VAE decode (opt-in; see below) |
| `HY_COND_ENCODE_TRACE` | `0` | `1` = CQ0 ``execute_trace`` for I2I cond **VAE encoder + ViT/aligner** (opt-in; recap stage stays eager) |
| `HY_TRACE_REGION_MB` | auto | Trace region MiB (default scales with `HY_NUM_LAYERS`, 128–512 MiB) |
| `HY_RECAPTION_KV` | `1` | `0` disables KV incremental decode on recaption path (required for recaption trace) |
| `HY_RECAPTION_PREFILL_CHUNK` | `1024` | Chunk size for long-prefix KV prefill (`0` = one shot) |
| `HY_KEEP_BACKBONE` | `1` | I2I: cache cond VAE/ViT tokens on host and **reuse** the resident backbone for denoise (skip ~140s reload). `0` = old free/rebuild sandwich |

> **``HY_TRACE=1``** (default): recaption AR uses CQ0 ``execute_trace`` when recaption runs.
> Denoise CFG ``execute_trace`` is **auto-enabled only when steps > 8** (default
> ``HY_DENOISE_TRACE_MIN_STEPS``): step-1 capture + warmup on an 8-step Distil loop costs
> more than eager replay. Instruct 50-step and base 50-step T2I keep denoise trace on.
> Force with ``HY_DENOISE_TRACE=1``; disable with ``HY_DENOISE_TRACE=0``.
> CQ1 async I/O for denoise latent / VAE RGB transfers when 2CQ is active.
>
> **``HY_TRACE=0``**: single command queue, no trace replay, no async I/O overlap.
>
> **VAE / vision cond encode trace is opt-in (default off).** ``HY_VAE_DECODE_TRACE``
> and ``HY_COND_ENCODE_TRACE`` default to ``0`` so first-run latency stays predictable:
> trace capture on VAE encode/decode often costs more than eager on a single image,
> cond traces are invalidated across backbone stage boundaries, and other tt-metal
> pipelines (tt_dit SD3.5/Flux) keep ``vae_traced=False`` by default. Enable when
> benchmarking steady-state replay (batch runs, repeated resolution) or profiling
> capture vs eager.
>
> Text-only and I2I recaption use KV + trace decode when ``HY_TRACE=1`` and recaption runs.
> Long prefixes (>``HY_RECAPTION_PREFILL_CHUNK``) use chunked eager KV prefill.
> Prefill cannot be trace-captured (KV ``replace()`` writes are illegal inside a trace).

### Instruct image-to-image (50-step CFG)

Replace `/path/to/input.png` with your cond image. `--bot-task image` skips the AR
recaption stage; use `think_recaption` for the full upstream flow.

Trace defaults match ``demo.py``: ``HY_TRACE=1`` for recaption AR; denoise
``execute_trace`` when steps > 8 (50-step Instruct keeps trace on). VAE decode and cond
VAE/ViT encode trace off unless ``HY_VAE_DECODE_TRACE=1`` / ``HY_COND_ENCODE_TRACE=1``.

```bash
HY_STEPS=50 HY_NUM_LAYERS=32 HY_GUIDANCE=2.5 python_env/bin/python \
  models/experimental/hunyuan_image_3_0/demo/demo_i2i.py \
  --prompt "make the sky more dramatic" \
  --cond /path/to/input.png \
  --bot-task image \
  --out hy_instruct.png
```

### Instruct-Distil image-to-image (8-step meanflow)

Denoise ``execute_trace`` is **off by default** (8 steps ≤ ``HY_DENOISE_TRACE_MIN_STEPS``);
recaption AR trace (if used) and 2CQ mesh stay on under ``HY_TRACE=1``.

```bash
HY_DISTIL=1 HY_NUM_LAYERS=32 HY_GUIDANCE=2.5 python_env/bin/python \
  models/experimental/hunyuan_image_3_0/demo/demo_i2i.py \
  --distil \
  --prompt "make the sky sunset orange" \
  --cond /path/to/input.png \
  --bot-task image \
  --out hy_instruct_distil.png
```

Without `HY_STEPS`, Instruct defaults to 50 steps and Distil to 8 (from each checkpoint's
`generation_config.json` when present).

---

## Status

### Ported, with passing PCC tests
- RMSNorm — `tt/attention/rms_norm.py`
- 2D RoPE — `tt/attention/rope_2d.py`
- Attention (GQA, qk-norm) — `tt/attention/attention.py`
- Attention mask (causal text + bidirectional image span) — `tt/attention/mask.py`
- MoE: router/gate, expert MLP, shared expert — `tt/moe/`
- Decoder layer — `tt/transformer_layer.py`
- Full 32-layer backbone (teacher-forced) — `tt/model.py`
- Patch embed `UNetDown` / final layer `UNetUp` — `tt/image_gen/patch_embed.py`
- Timestep embedder — `tt/image_gen/timestep_embedder.py`
- Flow-matching scheduler — `tt/scheduler.py`
- VAE encoder / decoder blocks (Conv3D) — `tt/vae/`
- **Full-res VAE decode on device** — `tt/vae/spatial.py` H/W-spatial-parallel
  (each device a 512² quadrant of 1024²); `tests/vae/test_decode_pipeline.py`
  (`test_decode_latent_spatial_vs_reference`) vs fp32 reference: PCC 0.999489, no OOM. **Optional trace VAE decode**
  (``HY_VAE_DECODE_TRACE=1`` with ``HY_TRACE=1``): CQ0 ``execute_trace`` via
  ``tt/stage_trace.py``; 2CQ async RGB D2H — see ``tt/vae_dual_cq.py``.
- **On-device single denoise step** — `tt/pipeline.py` `HunyuanTtDenoiseStep`
  (`tests/pcc/test_pipeline.py`): 4-layer PCC 0.99999, full 32-layer PCC 0.983.
  Device-side scatter (concat) + image-span slice; no host round-trips.
- **Multi-step denoise loop** — `tt/pipeline.py` `denoise_loop`
  (`tests/pcc/test_denoise.py`, PCC 0.99999): per-step timestep embedding,
  scheduler Euler update, CFG. **Trace denoise** (``HY_TRACE=1`` and steps >
  ``HY_DENOISE_TRACE_MIN_STEPS``): CQ0 ``execute_trace`` CFG loop via ``tt/stage_trace.py``;
  2CQ latent D2H fallback — see ``tt/denoise_dual_cq.py``.
- **`decode_latent` glue** — `tt/pipeline.py` (`tests/vae/test_decode_pipeline.py`):
  scaling / temporal-dim / denormalize wiring verified with an injected decoder.

### Known issues / blockers
- ~~**VAE full-res decode OOMs.**~~ RESOLVED — the decoder is now H/W-spatial-parallel
  across the 2×2 mesh (`tt/vae/spatial.py`), so each device holds a 512² quadrant and the
  conv im2col is 4× smaller per shard. Full 1024² decode runs with no OOM, PCC 0.999489 vs
  fp32 reference (`tests/vae/test_decode_pipeline.py`). See MEMORY_FIT_PLAN.md.
- **32-layer backbone drift:** free-running PCC ≈ 0.88 (bf16) for the standalone backbone;
  the full denoise step composes to 0.983. Audit before final image fidelity.
- **Host RAM:** 32 layers resident with `stream_experts=True` ≈ 150 GB host RAM
  (see note in `tt/model.py`). Needs on-demand disk streaming for the full model.
- **Not device-resident:** expert weights re-upload from host RAM every forward
  (`tt/moe/moe.py`), and MoE runs dense over all 64 experts (correct but not sparse).

---

## Pending work

### Phase 1 — Base T2I end-to-end (critical path) — COMPLETE
1. ~~**On-device single denoise step**~~ — DONE (`HunyuanTtDenoiseStep`).
2. ~~**Multi-step denoise loop** with CFG~~ — DONE (`denoise_loop`).
3. ~~**VAE decode**~~ — DONE on device, H/W-spatial-parallel (`tt/vae/spatial.py`,
   `decode_latent`); full-res 1024² runs with no OOM (PCC 0.999489).
4. ~~**Tokenizer + input construction**~~ — DONE. `HunyuanTokenizer` +
   `prepare_gen_image_inputs` build input_ids and the `[text | image span | text]`
   sequence; `demo/demo.py` runs from a real prompt (host wte embed; on-device embed via
   `HunyuanTtModel(embed_state_dict=...)` is also supported).
5. ~~**Runnable `demo/demo.py`**~~ — DONE: prompt → tokenizer → resident bf8 2×2 backbone
   → on-device VAE → PNG (`HY_NUM_LAYERS=32 demo/demo.py "a photo of a cat"`; default 50 steps).

### Phase 2 — Accuracy & scale
6. **VAE upsample memory** — chunk/shard/stream the DCAE upsample (the Phase-1 VAE
   blocker is really this work).
7. Resolve 32-layer bf16 drift (precision audit; candidate ops: MoE accumulation,
   RMSNorm, attention softmax). **Instrument:** `tests/pcc/test_teacher_forced.py::
   test_bf8_mixed_precision_audit` sweeps each layer at bf16 vs bf8 against the fp32 golden
   and prints the recommended `bf16_layers` set (layers whose bf8 per-layer PCC < 0.99).
   Run it on the box, then feed the result into `HunyuanTtModel(bf16_layers=...)` / `demo.py`.
8. On-demand layer/expert weight streaming from disk.
9. **Device-resident experts** (stop re-uploading per forward) + **sparse top-8 routed
   MoE** (replace the dense-over-64 correctness path) — `tt/moe/moe.py`.

### Phase 3 — Distil variant — DONE
10. ~~8-step meanflow sampling + cfg_distilled plumbing~~ — DONE:
    `scatter_distill_step_embeds`, `denoise_loop` guidance/timestep_r scatter,
    `demo/demo_i2i.py --distil` / `HY_DISTIL=1` with `HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR`.

### Phase 4 — Instruct (I2I) variant
Vision input path — device pieces + host glue DONE; full on-box AR decode remains:
11. ~~**SigLIP2 vision encoder** (`tt/vision/siglip2.py`) + vision→4096 `LightProjector`~~ —
    ported + PCC-tested (`tests/vision/test_siglip2_ttnn.py`, `forward_vision_with_aligner`).
12. ~~**Cond-vision sequence injection**~~ — DONE: `tt/vision/inject.py`. Three paths,
    all vs the reference masked scatter (`tests/vision/test_cond_vision_inject.py`, PCC 0.999998):
    `scatter_cond_vision_embeddings` (single contiguous TILE-aligned span),
    `scatter_cond_vision_embeddings_multi` (several contiguous spans — multi-image, device
    concat), and `scatter_cond_vision_embeddings_host` (arbitrary / ragged / non-aligned mask).
13. ~~**Host glue**~~ — DONE. `ref/vision/preprocess.py` extracts
    `HunyuanImage3ImageProcessor.vit_process_image` (verified **bitwise-equal** to upstream —
    `test_cond_image_preprocess.py::test_ref_preprocess_bitwise_matches_upstream`). `tt/vision/
    preprocess.py` adds the device bridge (`to_vision_inputs`) + `<img>` span lookup
    (`find_image_token_spans`). `tt/vision/i2i.py` assembles the pipeline: image → processor →
    `Siglip2VisionInputs` → vision+aligner (`encode_cond_vision`) → `inject_cond_vision`
    (auto device-concat vs host-scatter) → `model.forward(inputs_embeds=)`.
14. ~~**Autoregressive text generation**~~ — DONE (host loop + device head). `tt/lm_head.py`
    `HunyuanTtLMHead` projects backbone hidden → vocab logits (real `lm_head.weight`,
    `test_lm_head.py` PCC 0.99999, both full + last-token paths). `tt/generate.py` is the
    sampling loop: repetition-penalty → temperature → top-k → top-p → sample, plus a
    `StageTransitionLogitsProcessor` (recaption/think phase forcing) verified **bitwise-equal**
    to upstream `_StageTransitionLogitsProcessor` (`test_generate.py`, 7 unit tests). The loop
    is decoupled via a `forward_logits_fn` callback; `make_backbone_logits_fn` wires the
    resident backbone + LM head as the device adapter.
    **Host recaption orchestration:** `ref/recaption.py` + `tt/recaption.py`
    (`run_recaption_on_device` with `make_recaption_logits_fn` for I2I cond embeds).
    `demo/demo_i2i.py` runs the full ``think_recaption`` → denoise chain.
    **Remaining:** default ``HY_RECAPTION_LAYERS`` matches ``HY_NUM_LAYERS``. **2CQ AR**
    (``HY_RECAPTION_2CQ=1``): CQ0 forward + CQ1 async logits D2H — see ``tt/ar_dual_cq.py``.
    **Trace decode** (``HY_RECAPTION_TRACE=1``): requires ``HY_RECAPTION_KV=1`` and
    ``sp_factor=1`` on the recaption backbone; captures one KV single-token decode step on
    CQ0 (``tt/ar_trace.py``) and replays it in the host ``generate_text`` loop while stage
    forcing stays on host (Whisper-style). Open the device with an enlarged trace region
    (``open_recaption_mesh`` auto-sizes trace region from layer count, or set
    ``HY_RECAPTION_TRACE_REGION_MB=128``). Logs:
    ``capturing decode trace on CQ0``, ``decode trace captured``, ``trace replay steps=N``.

---

## PCC thresholds and exceptions

Default gates live in `tests/pcc/pcc_common.py` and `tests/pcc/pipeline_helpers.py`.
Tests compare TT output against the PyTorch `ref/` path. **Activations** (hidden states,
latents, text embeds) are randomly generated; **weights** come from the HF checkpoint.

| Threshold | Value | Test / module | Notes |
|-----------|------:|---------------|-------|
| `PCC_STRICT` | 0.999 | RMSNorm, RoPE, MoE expert FFN, WTE, **lm_head @ S=4160** | Per-block strict gate |
| `PCC_BLOCK` | 0.99 | Attention, decoder layer, teacher-forced final, 32L production backbone | Default block gate; production gates at S=1 + S=4160 |
| `PCC_CHAINED` | 0.86 | 32L chained free-running final hidden | Accumulated drift across layers |
| `PCC_DECODE_STACK` | 0.96 | Backbone at **S=1** (decode smoke) | Lower bar for single-token stack |
| `PCC_LOGIT_DECODE` | 0.96 | `test_logit_stack_production_pcc` at S=1 | 32L **teacher-forced** logits (free-running S=1 ~0.59 — known MoE drift) |
| `PCC_LOGIT_PREFILL` | 0.85 | `test_logit_stack_production_pcc` at S=4160 | 32L chained last-token logits |
| `PCC_LOGIT_MAX_CONTEXT` | 0.85 | `test_logit_stack_max_context_pcc` | 32L last-token at S=22784 (production slow CI) |
| `PCC_PIPELINE` | 0.98 | Reserved pipeline constant | See denoise/e2e rows below |
| Denoise step (≤8L, bf16) | 0.99 | `test_pipeline.py` / `pipeline_pcc_threshold` | |
| Denoise step (32L, bf16) | 0.85 | `test_denoise_step_production_32l_pcc`, `test_i2i_denoise_step_production_32l_pcc` | **Observed ~0.983** (T2I); threshold is conservative |
| Denoise step (bf8) | 0.90 | `pipeline_pcc_threshold(bf8)` | |
| Resident mesh denoise | 0.98 | `test_denoise_step_resident_mesh` | |
| E2E latent | 0.98 | `test_e2e_pipeline` (`HY_LATENT_PCC`) | Random latent/text inputs; **opt-in** (`HY_RUN_E2E_RANDOM=1`) |
| E2E RGB | 0.97 | `test_e2e_pipeline` (`HY_RGB_PCC`) | |
| `test_generate.py` | — | Host unit tests (`@pytest.mark.unit_host`) | Mock logits (V=64); excluded from smoke/PCC sweeps |
| Recaption AR | greedy token match | `test_recaption_production_greedy_tokens` | 32L instruct; not PCC (token parity) |
| Scheduler | 0.99 | `test_scheduler.py` | **Smoke tier only** — deterministic ref match; not in production script |
| Timestep embedders | 0.999 | `test_timestep_embedder_pcc` | **Smoke tier only** — scalar/batch timesteps @ S≤32; not in production script |
| lm_head (standalone) | 0.99 / 0.999 | `test_lm_head.py` | Smoke @ S=32 (`PCC_THR=0.99`); **production last-token @ S=4160** (`PCC_STRICT`); integrated path in logit stack |
| I2I denoise step | 0.85 | `test_i2i_denoise_step_production_32l_pcc` | 32L instruct bundle @ 1024²; in production slow CI |
| Vision / VAE | varies | `tests/vision/`, `tests/vae/` | Separate subtrees; not in backbone production script |
| VAE spatial decode | 0.99 | `test_decode_latent_spatial_vs_reference` | **Observed ~0.999489** |
| Free-running 32L backbone | ~0.88 | Known issue (not a CI gate) | See blockers below |
| 32L chained decode hidden | 0.96 | `test_backbone_production_32l_decode_pcc` | Chained, not teacher-forced |
| Attention mask | bitwise | `test_mask_production_pcc` | Exact match @ S=4160 image layout |

Override e2e thresholds with `HY_LATENT_PCC` / `HY_RGB_PCC`. Denoise weight dtype via
`HY_WEIGHT_DTYPE=bf8` and layer precision via `HY_BF16_LAYERS=0,1,...`.

---

## Running tests

```bash
# Single block (example)
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer.py -v -s

# On-device denoise step + multi-step loop
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise.py -v -s

# Full 32-layer single step on real weights (slow, ~6 min)
HY_NUM_LAYERS=32 python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py -k denoise_step -v -s

# Production slow CI gate (32L, GRID=64, S=4160 + S=22784 max-context, submodule gates)
bash models/experimental/hunyuan_image_3_0/tests/run_pcc_production_slow.sh
```

Checkpoint weights (sharded safetensors + `*.index.json` + `config.json`):

- **Base:** `HUNYUAN_MODEL_DIR` — or newest HF hub snapshot for `tencent/HunyuanImage-3.0`
- **Instruct:** `HUNYUAN_INSTRUCT_MODEL_DIR` — or HF `tencent/HunyuanImage-3.0-Instruct`
- **Upstream parity:** `HUNYUAN_UPSTREAM` — local clone of the `hunyuan_image_3` Python package (tokenizer tests)
- **Distil:** `HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR` — HF `tencent/HunyuanImage-3.0-Instruct-Distil`

Demos call `ensure_*_weights()` in `ref/weights.py`: env override first, then HF hub cache, auto-download on first run.
