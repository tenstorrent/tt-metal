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
cd ~/tt-ign/tt-metal
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
| **HunyuanImage-3.0** (base, text→image) | — | the bulk of the work; in progress |
| **HunyuanImage-3.0-Instruct** (image→image, reasoning) | SigLIP2 vision encoder, image-input preprocessing, recaption/think token generation | **large** — net-new vision + AR-text path |
| **HunyuanImage-3.0-Instruct-Distil** | 8-step sampling schedule only | **small** — config/scheduler reuse |

**Strategy:** finish base T2I end-to-end first (it unblocks all three), add Distil
(nearly free), then build the Instruct I2I stack.

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
- VAE encoder / decoder blocks (Conv3D) — `tt/vae/` (block-level PCC only — see blocker)
- **On-device single denoise step** — `tt/pipeline.py` `HunyuanTtDenoiseStep`
  (`tests/pcc/test_pipeline_step.py`): 4-layer PCC 0.99999, full 32-layer PCC 0.983.
  Device-side scatter (concat) + image-span slice; no host round-trips.
- **Multi-step denoise loop** — `tt/pipeline.py` `denoise_loop`
  (`tests/pcc/test_denoise_loop.py`, PCC 0.99999): per-step timestep embedding,
  scheduler Euler update, CFG.
- **`decode_latent` glue** — `tt/pipeline.py` (`tests/vae/test_decode_latent.py`):
  scaling / temporal-dim / denormalize wiring verified with an injected decoder.

### Known issues / blockers
- **VAE full-res decode OOMs (BLOCKER).** The real decoder runs only at its built-in
  resolution (64×64 latent → 1024×1024 image); the DCAE upsample needs a ~16 GB DRAM
  intermediate, but a single Blackhole device has ~2 GB free per the allocator. The
  upstream `tests/vae/test_decoder.py::test_full_decoder_vs_pytorch` OOMs identically,
  and `GroupNorm3D` bakes in `input_nhw=4096` at construction so it can't be shrunk.
  Needs chunked/sharded/streamed upsample (or a multi-device mesh). Blocks latent→image.
- **32-layer backbone drift:** free-running PCC ≈ 0.88 (bf16) for the standalone backbone;
  the full denoise step composes to 0.983. Audit before final image fidelity.
- **Host RAM:** 32 layers resident with `stream_experts=True` ≈ 150 GB host RAM
  (see note in `tt/model.py`). Needs on-demand disk streaming for the full model.
- **Not device-resident:** expert weights re-upload from host RAM every forward
  (`tt/moe/moe.py`), and MoE runs dense over all 64 experts (correct but not sparse).

---

## Pending work

### Phase 1 — Base T2I end-to-end (critical path)
1. ~~**On-device single denoise step**~~ — DONE (`HunyuanTtDenoiseStep`).
2. ~~**Multi-step denoise loop** with CFG~~ — DONE (`denoise_loop`).
3. **VAE decode (BLOCKED on memory)** — `decode_latent` glue is done and verified, but
   the real decoder OOMs at full res (see blockers). Unblocking it = chunked/sharded
   upsample so latent → image actually runs on device.
4. **Tokenizer + input construction** — prompt → input_ids and the
   `[text | image-token span | text]` sequence (`prepare_model_inputs` /
   `prepare_message_list`). Current tests use synthetic embeddings. (Independent of the
   VAE blocker — can proceed in parallel.)
5. **Runnable `demo/demo.py`** (currently empty) producing an image from a prompt.

### Phase 2 — Accuracy & scale
6. **VAE upsample memory** — chunk/shard/stream the DCAE upsample (the Phase-1 VAE
   blocker is really this work).
7. Resolve 32-layer bf16 drift (per-op precision audit; candidate ops: MoE accumulation,
   RMSNorm, attention softmax).
8. On-demand layer/expert weight streaming from disk.
9. **Device-resident experts** (stop re-uploading per forward) + **sparse top-8 routed
   MoE** (replace the dense-over-64 correctness path) — `tt/moe/moe.py`.

### Phase 3 — Distil variant
10. 8-step sampling schedule + config plumbing (`--diff-infer-steps 8`). Reuses Phase 1.

### Phase 4 — Instruct (I2I) variant
11. **SigLIP2 vision encoder** port (`ref/`+`tt/`) — reference `siglip2.py` (570 lines).
12. Image preprocessing (`image_processor.py`) and input-image embedding into the sequence.
13. Reasoning/recaption **autoregressive text generation** path (token sampling) for the
    `think_recaption` bot-task.

---

## Running tests

```bash
# Single block (example)
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer_layer.py -v -s

# On-device denoise step + multi-step loop
python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline_step.py \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise_loop.py -v -s

# Full 32-layer single step on real weights (slow, ~6 min)
HY_NUM_LAYERS=32 python_env/bin/python -m pytest \
  models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline_step.py -v -s
```

Checkpoint weights expected at `/home/iguser/Christy/HunyuanImage-3` (sharded safetensors
+ `*.index.json` + `config.json`).
