# MiMo-V2.6-Flash multimodal input (image, video, audio): plan

Parent: tenstorrent/tt-metal#60807 ("Later phases: Multimodal: image / video / audio"). Phase 0 is text-only; this
document plans the multimodal phase. Integration branch: `mimo-v2.6/multimodal` (off `mimo-v2.6/dev`); workstream
branches `mimo-v2.6/mm-<workstream>` merge into it, and it merges into `mimo-v2.6/dev`. Plugin branch:
`mimo-v2.6-flash/multimodal` (off `mimo-v2.6-flash/vllm-plugin-dev`).

**Goal.** Serve MiMo-V2.6-Flash-MOPD on Blackhole Galaxy with image, video and audio input through the OpenAI chat API
in the formats OpenRouter sends (`image_url` URL or data URI, `input_audio` base64, `video_url` URL or data URI),
with accuracy on standard multimodal benchmarks matching a GPU / OpenRouter reference.

## What the checkpoint contains

All encoder weights are BF16 and ship in `model_pp0_ep0_shard0.safetensors`, except the audio tokenizer
(`audio_tokenizer/model.safetensors`; encoder BF16, quantizer F32).

| Part | Shape | Weights |
|---|---|---|
| ViT `visual.*` (681M) | 28 blocks, hidden 1280, 32 Q / 8 KV heads, head dim 64, SwiGLU 4608 with bias, RMSNorm. Blocks 0/9/18/27 full attention; the rest window attention (|i−j| ≤ 64 positions of the patch sequence, `visual_token_window_size`) with a learned sink per head; `vit_window_attn_types` switches blocks 5-8, 14-17, 23-26 to column-major token order. 2D RoPE on (h, w). Patch 2×16×16 (Conv3d, no bias), merger: norm + MLP 5120→5120→4096 on 2×2 merged patches | `visual.blocks.N.{norm1,norm2,attn.qkv,attn.proj,attn.sinks,mlp.*}`, `visual.patch_embed.proj`, `visual.merger.{ln_q,mlp.0,mlp.2}` |
| Audio tokenizer encoder (308M) | mel (24 kHz, hop 240, 128 mels) → 2 convs (stride 2) → 24 layers (d 1024, 16 heads, window 128 / full hybrid) → avg-pool 2 → 20-codebook RVQ → codes at 25 Hz | `audio_tokenizer/model.safetensors` `encoder.*` (the `decoder.*` vocoder is unused for input) |
| Audio patch encoder (127M) | 20 code embeddings summed (`speech_embeddings.0..19`, dim 1024), groups of 4 frames → 6-layer Qwen2-style transformer (full attention, RoPE θ 640000) → MLP 4096→16384→4096 per group: 6.25 LLM tokens per second of audio | `speech_embeddings.*`, `audio_encoder.input_local_transformer.*`, `audio_encoder.projection.mlp.{0,2}` |
| LLM | Unchanged. Multimodal items replace their placeholder token embeddings (`image_token_id` 151655, `video_token_id` 151656, `audio_token_id` 151669); positions stay 1D (no M-RoPE) | — |

Placeholder expansion, pixel / mel preprocessing, video frame sampling and timestamp tokens are done by vLLM 0.30.0's
upstream processor (`vllm/model_executor/models/mimo_v2_omni.py`: `MiMoV2OmniMultiModalProcessor`,
`MiMoV2OmniProcessingInfo`, `MiMoV2OmniDummyInputsBuilder`; `vllm/transformers_utils/processors/mimo_v2_omni.py`).
We reuse it unchanged. The HF reference is `modeling_mimo_v2.py` in the checkpoint (`MiMoVisionTransformer`,
`MiMoAudioEncoder`, `MiMoAudioTokenizer`, `MiMoV2ForCausalLM._get_multimodal_embeds`).

Per item, the processor emits (field names as in `_get_mm_fields_config`):

* image: `pixel_values` [patches, 3·2·16·16], `image_grid_thw` [3]; `t·h·w / 4` placeholder tokens.
* video: `pixel_values_videos`, `video_grid_thw`, `second_per_grid_ts`, `video_start_times`. The placeholder range
  interleaves timestamp text tokens and `<|vision_start|>…<|vision_end|>` per temporal grid, so the item carries an
  `is_embed` mask. Video with an interleaved audio track (`video_audio_*`, `va_audio_features`) is out of scope for
  the first milestone.
* audio: `audio_features` (mel [T, 128]) and `audio_token_lens`.

## Architecture on TT

```
request ─► vLLM processor (CPU, upstream) ─► plugin: per-request MM items {modality, fields, offset, length, is_embed}
        ─► TT model class TTMiMoV2OmniForCausalLM.prefill_forward
              ├─ encoder backend: ViT (device, TP=8 on the request's row; host fallback), audio (host first)
              ├─ per-request embedding cache (keyed by item identifier) so chunked prefill and realigned chunks reuse it
              └─ chunk embeddings = embed(tokens) with rows at offset+is_embed replaced ─► existing 48-layer prefill
        ─► decode unchanged
```

Decisions:

1. **LLM input.** Add an embeddings input to the prefill chunk (`_prefill_chunk(..., embeds=...)`): a host bf16
   [padded, 4096] tensor uploaded with the same layout `ttnn.embedding` produces, used only for chunks that contain
   multimodal rows. Text-only chunks keep the device embedding and the traced chunk path. Chunks with multimodal rows
   run eagerly at first; tracing them (an embeddings input buffer) is a later optimization.
2. **Model class.** A new `MiMoV2OmniForCausalLM` TT class (subclass of the text `MiMoV2ForCausalLM` in
   `tt/generator_vllm.py`, registered with the upstream processor) selected by
   `--hf-overrides '{"architectures": ["MiMoV2OmniForCausalLM"]}'`. The text-only release path stays byte-identical.
3. **Encoders.** ViT on device (it dominates: a 1024² image is 4096 patches × 28 layers; video multiplies that by
   frames). It runs TP=8 on the requesting row's 1×8 submesh, so no cross-row traffic and each row keeps its own
   copy (~170 MB/chip). Audio starts on the host (fp32/bf16 torch): its RVQ step is an argmin over codebooks, so
   device numerics could flip codes; it moves to device only if host latency is a problem.
4. **Speculative decoding.** MTP / DFlash drafters see placeholder-token embeddings at multimodal positions in the
   first milestone (affects acceptance only, never output). Fix later if acceptance drops measurably.
5. **Chunked prefill and APC.** The plugin already sets `disable_chunked_mm_input`, so an item never straddles a
   chunk. With APC, an item fully inside the cached prefix arrives without tensors; MiMo's chunk realignment can move a
   chunk start back into such an item, which the embedding cache covers (else recompute from the request's features).

## Plugin contract (vllm-tt-plugin → model)

`prefill_forward(..., **kwargs)` receives, when the model declares multimodal support, `mm_items`: a list aligned with
the prefill rows; entry `i` is `None` (text-only) or a list in prompt order of

```python
{"modality": "image" | "video" | "audio", "identifier": str,      # vLLM mm hash, stable across chunks / requests
 "offset": int, "length": int,                                   # PlaceholderRange in the full prompt
 "is_embed": torch.BoolTensor | None,                            # None = all `length` rows are embeddings
 "data": dict[str, torch.Tensor] | None}                         # processor fields; None when stripped by APC
```

The existing image-only `pixel_values` / `image_grid_thw` lists stay for the other models. The model declares the
modalities it accepts (`model_capabilities()["mm_modalities"]`); the plugin rejects others at request time with a
clear error instead of `NotImplementedError` in the engine (an exception in prefill kills the engine).

## Workstreams

| Id | Branch | Scope | Device |
|---|---|---|---|
| vision | `mimo-v2.6/mm-vision` | `reference/vision.py` (fp32, bit-identical to HF), `tt/vision/` TTNN ViT on a 1×8 row, PCC tests per block and end to end, latency for 448², 1024², and a 32-frame video | bhg2 (short jobs), bhg4 |
| audio | `mimo-v2.6/mm-audio` | `reference/audio.py` (mel → tokenizer encoder → RVQ codes → patch encoder → [n, 4096]), bit-identical to HF and checked against vLLM `mimo_audio.py`; host backend with latency numbers; device port of the patch encoder only if needed | host |
| llm | `mimo-v2.6/mm-llm` | embeddings prefill input, `MiMoV2OmniForCausalLM` TT class, encoder-backend interface + embedding cache, multimodal golden (host-streamed 48 layers with `inputs_embeds`) and a device test, vLLM serving smoke for each modality | bhg4 |
| plugin | `mimo-v2.6-flash/multimodal` | the contract above, capability gate, lane / row-engine / async paths with `mm_features`, host tests | host |
| evals | (scripts under `tests/perf/` or the TTI branch) | benchmark choice and harness, OpenRouter-format API conformance suite, reference scores, tt-inference-server spec + eval config | bhg4 for runs |

Encoder backend interface (`tt/multimodal.py`), shared by vision / audio / llm:

```python
class EncoderBackend(Protocol):
    def embed_image(self, pixel_values, grid_thw) -> torch.Tensor  # [sum(t*h*w)/4, 4096] bf16, host
    def embed_video(self, pixel_values_videos, grid_thw) -> torch.Tensor
    def embed_audio(self, mel) -> torch.Tensor                     # [n_tokens, 4096] bf16, host
```

## Tests and gates

Tests land with the code (as for Phase 0, #60808):

* Reference: `reference/vision.py` and `reference/audio.py` match HF `modeling_mimo_v2.py` exactly (fp32, same
  inputs); token counts match the vLLM processor for every test item.
* ViT on device: per-block PCC ≥ 0.99 and final embeddings PCC ≥ 0.99 vs the fp32 reference, real weights, for an
  image, a non-square image, multiple images in one batch and a video.
* Audio host path: codes identical to the HF reference on the test clips; embeddings PCC ≥ 0.999.
* Full model: a multimodal golden (image + text, audio + text) from the host-streamed fp32 reference; device top-1
  and logits PCC at the same bar as the text golden (#60808: top-1 ≈ 0.93-0.95 with BFP4 experts).
* Serving: OpenRouter-format requests for every modality and input form (URL, data URI, multi-image, mixed), chunked
  prefill across an item boundary, APC hit with images, concurrent text + multimodal traffic, no engine death.
* Accuracy: per-modality benchmarks (see the evals workstream) within noise of the reference.

## Machines

* bhg4 (`sjc4-glx-r02u27`, shared with ipastaltzidis' TTI docker runs): main device box. Every device job goes
  through `/data/ttuser-agentx/mm/devlock.sh <timeout_s> <cmd>` (flock `/data/ttuser-agentx/.galaxy.lock`, refuses while
  any docker container runs). Workspace `/data/ttuser-agentx/mm/`.
* bhg2 (`g12blx02`, the user's own benchmark Galaxy): device jobs only through `/home/zni/mimo/mm/devlock.sh`
  (flock `/home/zni/mimo/.galaxy.lock`, capped at 45 min per job). Checkpoint `/home/zni/models/MiMo-V2.6-Flash-MOPD`,
  goldens in `/home/zni/models/mimo-golden`. Workspace `/home/zni/mimo/mm/`.
