# Qwen3.6-27B VL (image + video) serving via tt-inference-server + vLLM

**Date:** 2026-06-14
**Author:** ssinghal
**Status:** Design — pending review

## Goal

Serve the Qwen3.6-27B vision-language model (text **+ multi-image + video**) through
tt-inference-server using vLLM, on Blackhole Galaxy. The text-only path already serves.
The VL model path (vision encoder, M-RoPE, multimodal prefill, multimodal decode) is
**already implemented and validated in the demo** (`models/demos/qwen3_6_galaxy_v2/demo/mm_demo_qwen36.py`
and the `Qwen36MM*` / `vision_encoder` / `qwen36_mrope` modules). This effort is **purely the
serving/wiring layer** — no model-side vision work, no changes to the vision math.

**Non-goals:** changing the vision encoder, M-RoPE, or decode internals; re-deriving or
re-verifying the vision attention math (the demo already proves it); video *generation*
(tt-media-server / Wan / mochi is unrelated).

## Success criteria

- Server accepts OpenAI-style multimodal requests (image URL(s)/base64 and video) and returns
  coherent output.
- **Parity:** server output matches `mm_demo_qwen36` for identical inputs — single image,
  multi-image, and video — within the usual decode-coherence tolerance.
- `limit-mm-per-prompt` reflects what the model supports (multi-image + 1 video).
- No device hang across requests; text-only requests on the same server still work.

## Background / what already exists

### Demo (model side — DONE, reused as-is)
- `Qwen36MMGenerator` / `Qwen36MMPipeline` / `Qwen36MMPreprocessor` (`tt/qwen36_mm_*.py`):
  HF `Qwen3VLProcessor` → on-device seq-parallel `Qwen36VisionEncoder` → splice vision
  features into text embeddings → M-RoPE.
- `Qwen36VisionEncoder.forward(pixel_values, grid_thw, seq_parallel=True)` (`tt/vision_encoder.py`):
  27 TP=8 blocks + patch merger; **multi-image + video correct by construction** via the
  `cu_seqlens` block-diagonal additive mask built in `build_vision_seq_parallel_tensors`
  (`tt/vision_attention_tp.py:366` — "Blocks cross-segment attention (cu_seqlens) and
  attention to padded keys").
- M-RoPE: `get_rope_index()` + `build_mrope_tt_tensors()` (`tt/qwen36_mrope.py`).
- Vision/image/video token IDs: vision_start 248053, vision_end 248054, image_pad 248056,
  video 248057.
- Processor config (checkpoint): `Qwen3VLProcessor` + `Qwen3VLVideoProcessor`, patch_size 16,
  temporal_patch_size 2, merge_size 2, default video sampling 2 fps + `do_sample_frames`.
  **These are stock Qwen3VL processors** → vLLM's native Qwen3VL multimodal processor produces
  demo-matching sampling; no custom video backend needed (contrast: molmo2 needed one).

### Generator plumbing (model side — DONE, reused)
- `Generator.prefill_forward_text_embeds(tokens, inputs_embeds, rot_mats, page_table, kv_cache, prompt_lens)`
  (`tt/generator.py:825`): multimodal prefill entry — skips token-embed, accepts pre-fused
  vision+text embeddings + external M-RoPE.
- `Generator.set_decode_rope_offset(offset)` (`tt/generator.py:175`): decouples decode RoPE
  position from KV index for the post-vision span.

### Server side (text-only TODAY — what we extend)
- `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py`:
  `Qwen3_5ForConditionalGeneration(Generator)` — text only; `initialize_vllm_text_transformer_qwen36`
  builds the text `TtTransformer`; `allocate_vllm_kv_cache` row-shards KV across 8 rows.
- `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/qwen3_5_config.py`: `Qwen3_5Config`
  **drops `vision_config` and `text_config`** to keep `is_multimodal` off; sets architectures
  to `Qwen3_5ForConditionalGeneration`.
- Plugin `__init__.py`: registers the `qwen3_5` AutoConfig at import; `register_models()`
  registers the architecture → class mapping. `platform.py` resolves the `TT`-prefixed arch.
- Model spec: `tt-inference-server/workflows/model_specs/dev/llm.yaml` Qwen3.6 entry
  (`hf_overrides` currently forces a text architecture).

### Reference patterns (templates)
- **Image-via-vLLM:** `models/demos/qwen3_vl/tt/generator_vllm.py` — `SupportsMultiModal`,
  `@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor, info=TT_Qwen3VLProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder)`,
  `prefill_forward(..., **kwargs)` consuming `pixel_values`/`image_grid_thw`, returns `(logits, rope_deltas)`,
  `decode_forward` updates rope_deltas.
- **Video-via-vLLM:** molmo2 (branch `ssinghal/molmo2_new_glx`) — registered in the vLLM plugin
  (`models.demos.molmo2.tt.generator_vllm:Molmo2ForConditionalGeneration`), spec `model_type=VLM`,
  `supported_modalities=[text,image,video]`, `vllm_args` with
  `limit-mm-per-prompt: {"image": N, "video": 1}` and `media_io_kwargs` for the video backend.
  Confirms: **video understanding goes through vLLM natively** (no media-server).

## Design

### Component 1 — Multimodal config (`qwen3_5_config.py`)
Stop dropping `vision_config` when serving VL; surface a multimodal-capable config so vLLM
treats Qwen3.6 as a multimodal model and its Qwen3VL processor can expand image/video tokens.

- Keep `vision_config` (and whatever fields vLLM's `Qwen3VLProcessingInfo` reads: image/video
  token IDs, vision_start/end, spatial_merge_size, patch/temporal sizes).
- Set the architecture name to a distinct VL arch (e.g. `Qwen3_6VLForConditionalGeneration`)
  so `platform.py` resolves the new VL class and does not collide with the text-only path.
- Preserve text-only behavior: gate the VL config behind the spec (text-only spec keeps the
  current drop). Decision point in plan: separate config class vs. a flag on `Qwen3_5Config`.

**Risk (primary):** vLLM's `Qwen3VLProcessingInfo`/`Qwen3VLMultiModalProcessor` expect a
qwen3_vl-shaped config + an `AutoProcessor`-loadable `Qwen3VLProcessor`. Our config is
`qwen3_5`. Mitigation: a `TT_Qwen36VLProcessingInfo(Qwen3VLProcessingInfo)` subclass that reads
the right fields/token-IDs from our config and points at the checkpoint's `Qwen3VLProcessor`
(`trust_remote_code`). Validate token-expansion against the demo's `Qwen36MMPreprocessor` output
on the same prompt before anything else.

### Component 2 — VL generator class (`generator_vllm.py`)
Add `Qwen3_6VLForConditionalGeneration(Generator, SupportsMultiModal)`, registered via
`@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor, info=TT_Qwen36VLProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder)`.

- `TT_Qwen36VLProcessingInfo.get_supported_mm_limits()` → `{"image": N, "video": 1}`
  (N = model's practical multi-image limit; confirm a concrete number in plan).
- `initialize_vllm_model`: build the text `TtTransformer` (reuse `initialize_vllm_text_transformer_qwen36`)
  **plus** the on-device `Qwen36VisionEncoder` + `CCLManager` (reuse the demo's construction);
  load the HF reference `embed_tokens` for text-embedding lookup.
- `allocate_kv_cache` / `cache_path`: unchanged (reuse the row-sharded `allocate_vllm_kv_cache`).
- `prefill_forward(tokens, page_table, kv_cache, prompt_lens, enable_trace, **kwargs)`:
  1. Reconstruct `input_ids` + per-user `attention_mask` (fix padding to `pad_token_id`),
     matching the qwen3_vl precedent.
  2. If `pixel_values` and/or `pixel_values_videos` present → run `Qwen36VisionEncoder` to get
     image/video features (concat per the precedent's per-user unwrap).
  3. Text-embed lookup + splice vision features at image/video token positions
     (reuse `splice_vision_into_embeddings` / `Qwen36MMPipeline` helper).
  4. Build M-RoPE cos/sin from 3D positions via `get_rope_index` + `build_mrope_tt_tensors`
     using `image_grid_thw` / `video_grid_thw`; compute `rope_deltas`.
  5. Call `prefill_forward_text_embeds(inputs_embeds, rot_mats=(cos,sin), page_table, kv_cache, prompt_lens=decoding_pos)`.
  6. Return `(logits, rope_deltas)`.
  - Prefill tracing stays disabled (GDN/DeltaNet circular-buffer clash — already the case).
- `decode_forward(*args, **kwargs)`: pop `rope_deltas_all_users`, apply via
  `set_decode_rope_offset` / rope-delta update, then defer to the existing text `decode_forward`
  (M-RoPE degenerates to 1D after the vision span, so the validated text decode is exact).

### Component 3 — Plugin registration + model spec
- Plugin `__init__.py` `register_models()`: `ModelRegistry.register_model("TTQwen3_6VLForConditionalGeneration",
  "models.demos.qwen3_6_galaxy_v2.tt.generator_vllm:Qwen3_6VLForConditionalGeneration")`.
- `workflows/model_specs/dev/llm.yaml` Qwen3.6 entry: `model_type=VLM`,
  `supported_modalities=[text,image,video]`, `hf_overrides` architecture → the VL arch,
  `vllm_args` with `limit-mm-per-prompt: {"image": N, "video": 1}`, `trust_remote_code: true`,
  and `mm_processor_kwargs`/`media_io_kwargs` only if non-default fps/longest_edge tuning is
  needed (default 2 fps matches the demo). Keep the existing `QWEN36_*` decode/trace flags.

### Component 4 — Validation
- **Token-expansion parity:** vLLM processor output (`input_ids`, `image_grid_thw`,
  `video_grid_thw`, pixel tensors) == `Qwen36MMPreprocessor` output for the same prompt+media.
- **Prefill/decode parity vs demo:** server logits / generated text == `mm_demo_qwen36` for
  (a) single image, (b) multi-image, (c) video — same prompt, greedy.
- **Server smoke:** `curl` an image request and a video request; confirm coherent output and
  no hang; confirm text-only requests still work on the same server.
- **Eval suite:** run the multi-video / VLM eval suite per CLAUDE.md completion criteria;
  document results in `BRINGUP_LOG.md` and the README.

## Risks & open questions

1. **Config/processor compat (primary):** does vLLM's Qwen3VL processor accept the `qwen3_5`
   config + checkpoint `Qwen3VLProcessor`? Resolve first via the token-expansion parity check;
   fall back to a `TT_Qwen36VLProcessingInfo` override.
2. **On-device vision in server batch-1:** the demo runs batch-1 prefill; confirm the
   `CCLManager`/seq-parallel vision encoder coexists with the server's sub-device/trace setup
   (prefill trace already off).
3. **Mixed text-only + MM in one batch step:** precedents punt (assert single-modality batch);
   match that initially.
4. **Multi-image concrete limit N:** confirm a supported value (model card implies multi-image;
   pick a tested N, document it).
5. **Decode coherence on hard content** remains a known model-level limitation (see memory);
   out of scope here but note it in parity tolerance.

## Out of scope
- Vision encoder / M-RoPE / decode internals (done in demo).
- Multi-image vision-attention correctness work (already block-diagonal; only an empirical
  parity check is in scope).
- Video generation models / tt-media-server.
- Performance optimization of the vision path.
