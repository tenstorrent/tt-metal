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

1. **LLM input.** An embeddings input to the prefill chunk (`TtMiMoModel.prefill_enqueue(..., embeds=, embed_rows=)`):
   host bf16 rows for the chunk's multimodal positions, uploaded row-major and merged into the device embedding of the
   chunk's tokens by one exact gather (`ttnn.embedding` over [token rows; uploaded rows]), so the pass starts from
   exactly the layout `ttnn.embedding` produces and pad rows stay the tokens'. Used only for chunks that contain
   multimodal rows; text-only chunks keep the device embedding, prefill batches and the traced chunk path.
   Traced passes take multimodal rows too (`TtMiMoModel.enable_trace_embeds`, `MIMO_MM_TRACE`, default on in
   `MiMoV2OmniForCausalLM`): every traced prefill pass embeds its tokens and merges host rows by the same gather,
   from persistent inputs a replay rewrites (its gather index, and the host rows packed into a segmented row buffer
   the passes share, of which a replay uploads only the segments its rows reach). So a chunk with multimodal rows
   batches with the lane's other first chunks and replays their trace, or replays a traced later chunk, or a traced
   first chunk (start 0) of 640-4096 tokens; the multimodal traces are optional (captured last, while the trace
   region has room: ~14 MB each). A text pass computes the same values either way; traced and eager multimodal
   passes are bit-identical (`tests/test_prefill_embeds_tp8.py`, `tests/test_mm_prefill_adapter_tp8.py`).
2. **Model class.** A new `MiMoV2OmniForCausalLM` TT class (subclass of the text `MiMoV2ForCausalLM` in
   `tt/generator_vllm.py`, registered with the upstream processor) selected by `TT_MODEL_CLASS_OVERRIDES` (see the
   plugin contract). The text-only release path stays byte-identical.
3. **Encoders.** ViT on device (it dominates: a 1024² image is 4096 patches × 28 layers; video multiplies that by
   frames). It runs TP=8 on the requesting row's 1×8 submesh, so no cross-row traffic and each row keeps its own
   copy (~170 MB/chip). Audio starts on the host (fp32/bf16 torch): its RVQ step is an argmin over codebooks, so
   device numerics could flip codes; it moves to device only if host latency is a problem. As built: the tokenizer
   encoder (mel -> codes) runs TP=8 on the requesting row (tt/audio/tokenizer.py, split bf16 activations, fp32
   elementwise; ``MIMO_AUDIO_TOKENIZER=device``, the default): ~4 % of its code frames differ from the fp32 reference
   (Blackhole matmul accumulation), at no cost in accuracy (bhg2 serving A/B: LibriSpeech test-clean WER 1.79 % vs
   1.97 % for the bit-identical fp32 host encoder in vLLM's front end, ``frontend``), and the front end no longer
   queues audio behind long clips. The patch encoder (codes -> embeddings) runs TP=8 on the row too
   (tt/audio/patch.py, 57 MB/chip; ``MIMO_AUDIO_PATCH=device``, the default) or on the engine's host CPU
   (``engine``). Both take 0.1 GB more of MIMO_MM_VIT_RESERVE_GB by default.
4. **Speculative decoding.** MTP / DFlash drafters see placeholder-token embeddings at multimodal positions in the
   first milestone (affects acceptance only, never output). Fix later if acceptance drops measurably.
5. **Chunked prefill and APC.** Items may straddle chunks (vLLM's chunked multimodal input stays on): each chunk
   takes the rows of its range from every item, and the embedding cache keeps an item encoded once across chunks.
   With APC the plugin restores an item's tensors, so a chunk start MiMo's realignment moves back into a cached item
   either hits the embedding cache or recomputes it.

## Plugin contract (vllm-tt-plugin → model)

`prefill_forward(..., **kwargs)` receives, when the model declares `model_capabilities["mm_modalities"]` (class
attribute; the plugin reads it from the class), `mm_items` and no `pixel_values` / `image_grid_thw`: a list aligned
with the prefill rows; entry `i` is `None` (text-only) or ALL of that request's items in prompt order, sent with every
prefill chunk of the request, each

```python
{"modality": "image" | "video" | "audio", "identifier": str,      # vLLM mm hash, stable across chunks / requests
 "offset": int, "length": int,                                   # PlaceholderRange in the full prompt
 "is_embed": torch.BoolTensor | None,                            # None = all `length` rows are embeddings
 "data": dict[str, torch.Tensor]}                                # processor fields; always present
```

* `data` is always present, also for items inside an APC-cached prefix (the TT scheduler restores the fields vLLM
  strips), so an item can always be recomputed.
* Items may straddle chunks: a model taking `mm_items` keeps vLLM's chunked multimodal input (forcing
  `disable_chunked_mm_input` made vLLM refuse to start: images up to 8192 tokens, videos up to 114688, against a 4096
  budget). The model embeds the rows of `[start_pos, prompt_lens)` from each item's `offset` / `is_embed`, including
  rows of a chunk start it moved back itself (MiMo's 128-token alignment), and caches each item's embeddings so a
  straddling item is encoded once (`tt/multimodal.py`).
* The plugin never says when an item or request is done: the model's embedding cache evicts on its own (an LRU by
  bytes, `MIMO_MM_CACHE_MB`; a slot that takes another request drops its old request's items that no other slot holds).
* The plugin refuses modalities the model does not declare at request time (a 4xx, never an exception in the engine:
  an exception in prefill kills it). The model validates what remains as early as it can. A row it still cannot serve
  (malformed items, an encoder failure, a cached prefix holding an item that failed) is logged as an ERROR with a
  running count and, with `MIMO_MM_STRICT=1` (the default), never answered from placeholder embeddings: its first
  token is forced to the end of the turn, and the model, which declares `model_capabilities["reports_prefill_errors"]`,
  returns `{row: reason}` from `take_prefill_errors()` after `prefill_forward`; **the plugin is to finish those
  requests with an error** (vLLM's `FINISHED_ERROR`), so a client never scores a silent wrong answer. Validation
  belongs in the processor / request stage wherever possible.
* DRAM: the model takes `MIMO_MM_VIT_RESERVE_GB` off the KV pool (default 0.25 with speculative decoding, 0
  without) and refuses to start when a replica keeps less free DRAM than its largest ViT pass needs after every trace
  is captured (`VIT_PASS_DRAM_MB`, measured by `tests/test_vit_dram_headroom_tp8.py`); the ViT compiles all its row
  buckets in the prefill warm-up, before any capture.
* Class selection: vLLM resolves the MiMo-V2.6-Flash checkpoints (they carry a `vision_config`) to
  `MiMoV2OmniForCausalLM`, so `--hf-overrides '{"architectures": [...]}'` changes nothing, and the plugin's built-in
  map serves both `TTMiMoV2ForCausalLM` and `TTMiMoV2OmniForCausalLM` with the text class (the release path, unchanged).
  Serve multimodal with
  `TT_MODEL_CLASS_OVERRIDES=TTMiMoV2OmniForCausalLM=models.experimental.mimo_v2_6_flash.tt.generator_vllm:MiMoV2OmniForCausalLM`.
  The class registers vLLM's upstream processor (`@MULTIMODAL_REGISTRY.register_processor` with
  `MiMoV2OmniMultiModalProcessor` / `MiMoV2OmniProcessingInfo` / `MiMoV2OmniDummyInputsBuilder`) and implements
  `get_placeholder_str` (chat rendering calls it on the TT class).
* Models that do not declare `mm_modalities` keep the image-only `pixel_values` / `image_grid_thw` lists.
  `MiMoV2OmniForCausalLM` also accepts them (items found by their placeholder runs).

## Video input (frames, times, audio)

vLLM 0.30's MiMo processor labels the loader's frames 1 s apart and takes 32 of them at full resolution by default, so
a 2-minute clip was announced as 00:00 .. 00:30 at 14K tokens, without its audio. Xiaomi's API samples ~2 frames a
second at ~280 merged tokens a frame pair (16-34K tokens on Video-MME short) and interleaves the audio track.

* `tt/video_input.py`: frames labelled with the loader's real times (`MIMO_VIDEO_TIMESTAMPS=loader`, default; the
  loader metadata travels with the frames through vLLM's processor cache), per-frame / per-clip pixel caps
  (`MIMO_VIDEO_MAX_PIXELS`, `MIMO_VIDEO_TOTAL_MAX_PIXELS`; unset: the checkpoint's 8388608 / 268435456).
* `tt/video_audio.py`: a video's audio track decoded from the video's bytes and interleaved per temporal grid at 6.25
  tokens a second, as the checkpoint's processor does (`video_audio` items; `MIMO_VIDEO_AUDIO`, default on when audio
  is served; per request / server `mm_processor_kwargs: {"mimo_video_audio": true|false}`); the engine interleaves the
  ViT rows and the audio patch encoder's rows (`tt/multimodal.py interleave_video_audio`). The track follows
  `MIMO_AUDIO_TOKENIZER` like an audio item: `device` (default) sends its fp32 mel bit for bit (`va_audio_mel_bits`)
  and the request's row tokenizes it (device tokenizer encoder, host RVQ, codes remembered by item identifier:
  `MIMO_MM_TRACK_CODES_MB`); `frontend` tokenizes it in the API server (`va_audio_codes`).
* `tt/video_prompt.py`: the checkpoint processor's video prompt (`MIMO_VIDEO_PROMPT=hf`, default) instead of
  upstream's extra outer `<|vision_start|> .. <|vision_end|>`.

Frame sampling is vLLM's. Xiaomi parity: `--media-io-kwargs '{"video": {"num_frames": -1, "fps": 2}}'` (2 fps, every
second of the clip), with audio served (a checkpoint with `audio_tokenizer/`, torchaudio); the frame sizes are the
defaults (`MIMO_VIDEO_MAX_PIXELS=307200` a frame, `MIMO_VIDEO_TOTAL_MAX_PIXELS=268435456` spread over the frames, odd
frame counts cut to whole pairs: `MIMO_VIDEO_EVEN_FRAMES=1`). With them the prompt token counts equal Xiaomi's API's on
the evals agent's probes (120 s 720p 33978, a VGI-Bench clip 124254; with its audio track within 6 tokens).
Long videos (VGI-Bench, 10-40 min, 130-146K tokens, up to 315 MB): vLLM's per-item encoder bound is the context
(`get_mm_max_tokens_per_item`; upstream's 114688 refused them); the proxy needs `TT_PROXY_FETCH_MAX_MB=400`,
`TT_PROXY_FETCH_TIMEOUT_S=300` and `TT_PROXY_MAX_REQUEST_MB=512` (a 315 MB file is a 420 MB data URI), vLLM fetching a
URL itself `VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB=400` and `VLLM_VIDEO_FETCH_TIMEOUT=300`. Such a request spends 1-4 minutes
in the API server, where vLLM processes multimodal items one at a time. On bhg2's CPU (10.5 / 27.3 / 40.3 min videos,
129.5 / 133.2 / 145.8K tokens; 2026-11-07): vLLM's OpenCV decode at 2 fps 6-20 s, the audio track's decode 1-4 s, the
frame resize 1.2-4.4 s (OpenCV, `MIMO_VIDEO_RESIZE=cv2`, default; torch's 10-25 s), the checkpoint processor and mel
3-4 s, and the fp32 CPU audio tokenizer 56-204 s (~5 s a minute of audio, 75-84 % of the front end). That tokenizer now
runs on the request's row (`MIMO_AUDIO_TOKENIZER=device`, the default; `tests/perf/video_audio_device.py`, bhg2 row 1,
27.3 / 40.3 min videos): the front end takes 21 / 40 s instead of 151 / 230 s (vLLM's decode 13 / 29 s of it; the
40 min host run shared the CPU with a host test), and the engine's encode of the item 16.4 / 21.5 s instead of
6.8 / 7.0 s (ViT 6.0-6.2 s, device tokenizer encoder 8.9 / 13.2 s, host RVQ 0.9 / 1.4 s, patch encoder 0.3 / 0.5 s;
the row's other requests wait meanwhile, as for the ViT), with the same prompt (133188 / 145825 tokens), 95.8 / 96.2 %
of the code frames equal to the front end's fp32 codes, the audio rows at PCC 0.999 and the frame rows identical. An
item encoded again (embeddings evicted from the engine's cache) takes 6.7 / 6.9 s: its codes are remembered. The
next question on
a video hits vLLM's processor cache (0.1-0.3 s; the track's decode is cached too, `MIMO_VIDEO_AUDIO_CACHE_MB`) and the
prefix cache (129024 of 129538 tokens): 9.9 s end to end on the 10.5 min video, nearly all of it vLLM decoding the video
again. A client that sends `"uuid": <video id>` in the video part of every request on a video, and `"video_url": null`
with that uuid after the first, skips the fetch and decode too: 0.4-0.5 s. The uuid-keyed hash is not the content
hash (pick one per video), and the item must still be in the processor cache (`--mm-processor-cache-gb`, default 4; a
long video's item is 1.3-1.4 GB), else vLLM raises "Cache miss ... but data is not provided".
Video-MME short docs 0-59, one row engine at concurrency 8 (Xiaomi's API: 75.0 %; 2026-11-07):

| Video input | Score | Verbose / 60 | Prompt tokens p50 | Latency c=8 / c=1 |
|---|---|---|---|---|
| 32 frames, full resolution, 1 fps labels (before) | 63.3 / 63.3 / 60.0 | 13-15 | 14.3K | 11.8 s / 3.4 s |
| 32 frames, real times | 58.3 | 15 | 14.3K | 9.9 s / 3.4 s |
| 2 fps, 286720 px a frame, real times | 63.3 / 65.0 / 65.0 | 5-6 | 28.0K | 18.0 s / 6.2 s |
| + audio track (upstream prompt) | 70.0 / 68.3 | 9-10 | 28.9K | 22.6 s / 8.7 s |
| + audio track, `MIMO_VIDEO_PROMPT=hf` | 71.7 | 0 | 28.9K | 23.3 s / - |
| Xiaomi's sizes (307200 a frame, every second at 2 fps) + audio + `hf` (the defaults) | 71.7 | 0 | 30.1K | 25.9 s / - |

At 2 fps a request's ViT takes ~0.5 s and its prefill ~3.9 s (c=1, 26K tokens); the front end's audio tokenizer adds
~2.5 s a request (CPU, 8 threads; these runs predate the track's device tokenizer, so they tokenized in the front end).

### Greedy decoding on long videos (VGI-Bench)

At temperature 0 with thinking on, MiMo's reasoning on VGI-Bench's long videos often never answers: on a question it
cannot settle (the smoke questions are false-premise ones, "This does not happen in the video" is right) it re-checks
the same scenes until the token limit. That comes with greedy decoding; nothing points at TT's numerics (no GPU
provider decodes greedily with thinking on, so there is no greedy GPU run to compare with). One row engine on bhg2, smoke docs
0-4 (128-147K tokens) and the 7 other public questions on the same 7 videos, max_tokens 16384 (2026-11-07):

| Decoding / input | No answer by 16384 tokens |
|---|---|
| greedy (the bhg4 eval at 9c014d841, 131072 limit: 4/5) | 2/5 + 1/7 |
| greedy, no audio track / 1 fps (same token count) / 0.25 fps (40-91K tokens) | 3/5 / 3/5 / 1/5 |
| greedy, GPTQ experts and BFP8 experts in the last layer | 3/5 + 4/7 |
| temperature 0.05 / 0.2 | 3/5 / 4/5 |
| temperature 1.0, top-p 0.95 (the model card's), two seeds | 0/5 + 0/7, 0/5; 404-9323 tokens |
| text only: LongBench v2, 118-139K tokens, greedy | 0/6 |

The GPU providers' temperature-0 runs that answer (GMICloud: 0 of 17 loop) are sampled: with thinking on, GMICloud
answers one temperature-0 "pick a random integer" request four ways (734 four times with thinking off), as Xiaomi's
endpoint does. And TT's next-token distribution matches theirs: teacher-forcing GMICloud's own reasoning for docs 3
and 4 (its prompts within 2 tokens of ours) through TT, TT's argmax is GMICloud's token at 85.3 / 83.4 % of the
positions against a mean TT top-1 probability of 84.6 / 81.6 %, in every probability bin, which is what samples from
TT's own distribution would give (the same with GPTQ and last-layer BFP8 experts). To serve temperature-0 requests the way those providers do, start lb.py with
`--thinking-greedy 1.0,0.95` (`TT_PROXY_THINKING_GREEDY`): such a request with thinking on is sampled at the model
card's settings; evals take `MM_TEMPERATURE=1.0 MM_TOP_P=0.95`.

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
