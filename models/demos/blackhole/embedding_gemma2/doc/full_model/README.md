# EmbeddingGemma 2: full multimodal model (stage E4)

Host sjc-snva-tp100, one Blackhole P150a (device 0), branch `gtobarTT/embedding_gemma2_bringup`.
Model `google/embeddinggemma-2`, snapshot `914f7f89`. HF reference: transformers 5.20.2 (overlay on
python_env, see the top-level README) for the tests; sentence-transformers 6.2.0 with transformers
5.21.0 (separate py3.11 venv) for the formatting check.

Labels: **[M]** measured (command and value), **[I]** inferred from file:line, **[G]** guess.
`$HF` = `transformers/models/embedding_gemma2/modeling_embedding_gemma2.py` (5.20.2),
`$MU` = `transformers/masking_utils.py`, `$ST` = `sentence_transformers/` (6.2.0).
`$E4` = `tt-ops/runs/embeddinggemma-2/e4`.

## 1. HF merge semantics

Read in the HF source before the TT merge was written.

| question | answer | source |
|---|---|---|
| Which tokens are placeholders | `input_ids == image_token_id (258880)`, `video_token_id (258884)`, `audio_token_id (258881)`. Each media item is `<|image>` (255999) + n x `<|image|>` + `<image|>` (258882); audio uses `<|audio>` (256000) / `<audio|>` (258883); a video is one such image block per frame. The begin/end markers are ordinary text tokens. | [I] `$HF:714-716` (`get_placeholder_mask`); processor `replace_image_token`, `replace_video_token`, `replace_audio_token`; `config.json` |
| What the text embedding does for placeholder ids | Placeholder ids are replaced by `pad_token_id` (0) and looked up like any token. Those rows are then overwritten. | [I] `$HF:778-779` |
| Embedding scale before or after the scatter | The `sqrt(512)` scale is inside `embed_tokens` (`EmbeddingGemma2TextScaledWordEmbedding.forward`), so it applies to the text lookup only. The soft tokens are scattered **after** it and are **not** scaled. | [I] `$HF:153`, `$HF:779`, `$HF:795`, `:814`, `:835` |
| Scatter order | `masked_scatter` fills the placeholder positions in flattened (batch-major, then position) order from the concatenated features: images in processor order, video frames in order (per-frame soft tokens concatenated), audio clips in order with padded audio tokens dropped first. | [I] `$HF:783-797`, `:801-816`, `:821-837`; `get_video_features` split `:916-927` |
| Soft-token producers | image/video: vision tower + `embed_vision` (RMSNorm without scale, Linear 768->512); audio: audio tower + `embed_audio` (RMSNorm without scale, Linear 1536->512). Implemented and tested in E2/E3. | [I] `$HF:565-591`, `:666-692`, `:858-875` |
| Per-layer inputs (PLE) | Computed from the **merged** `inputs_embeds`: the full model calls `language_model(inputs_embeds=...)`, and the text model runs `per_layer_inputs = self.ple(inputs_embeds)` on what it is given. Soft-token rows therefore also get per-layer inputs. There is no token-id table. | [I] `$HF:839-845`, `$HF:524-527` |
| Positions | `arange(S)` for every token, media tokens included. | [I] `$HF:529-530` |
| Masks over multimodal spans | Nothing modality-specific. Full layers: `create_bidirectional_mask` (padding only). Sliding layers: `create_bidirectional_sliding_window_mask` (`abs(q - kv) <= 512` and padding). All layers are non-causal (`is_causal = False`). A media span is attended like text. | [I] `$HF:533-542`, `$HF:331`, `$MU:137-155` |
| Pooling | ST `1_Pooling`: mean over all tokens with `attention_mask == 1` (`include_prompt: true`), so BOS/EOS, prompt, boi/eoi and every soft token are pooled. Then `2_Normalize` (L2). The 512->768 projection runs per token inside HF before pooling. | [I] `1_Pooling/config.json`, `$ST/sentence_transformer/modules/pooling.py:127-147`, `$HF:559-560` |
| MRL | Keep the first 512/256/128 values and re-normalize. | [I] model card "Matryoshka Dimension Truncation" |

## 2. Input formatting (sentence-transformers)

`modules.json` loads a sentence-transformers `Transformer` module whose `modality_config` includes
`message` (`sentence_bert_config.json`). ST therefore converts every input to a chat message
(`$ST/base/modules/transformer.py:1363-1377`):

- A string becomes `[{"role": "user", "content": [{"type": "text", "text": s}]}]`.
- A dict `{"text": ..., "image": ..., "audio": ..., "video": ...}` becomes one user message whose
  content lists the modalities in **dict key order** (`$ST/base/modality.py:541-579`).
- A prompt (`prompt_name` from `config_sentence_transformers.json`) becomes a system message
  (`modality.py:679-697`). `chat_template.jinja` renders system text first and joins everything
  without separators, so the string equals `prompt + text`.
- The processor (`EmbeddingGemma2Processor`, `apply_chat_template(tokenize=True)` with
  `text_kwargs={"padding": True, "truncation": "longest_first"}`, `audio_kwargs={"padding": True}`
  and the video metadata, `transformer.py:1958-1982`) renders the template, inserts `<|image|>` /
  `<|audio|>` / `<|video|>` for media without manual placeholders, expands every placeholder to
  its soft-token count, and tokenizes with BOS (2) and EOS (1).

`tt/embedding_model.py` (`to_messages`, `process_item`) does the same calls. Evidence:

- [M] `$E4/st/st_dump.py` ran `SentenceTransformer("google/embeddinggemma-2")` (ST 6.2.0) on all 28
  test inputs. `$E4/st/check_formatting.py` ran the wrapper's `process_item` in python_env on the
  same inputs. The input ids are **identical for 28 / 28** inputs (`doc/full_model/st_reference.json`,
  `all_input_ids_equal: true`).
- [M] HF `EmbeddingGemma2Model` fp32 (overlay) + mean pool + L2 normalize vs ST fp32 embedding:
  cosine 0.9999999-1.0000002 on all 28 inputs. So the test reference equals ST.
- [M] ST bf16 vs ST fp32: cosine 0.99962 (tone3s audio) to 0.99998.
- `tests/pcc/test_full_model.py::test_input_ids_match_sentence_transformers` re-checks the ids
  against the sha256 values in `st_reference.json` on every run.

## 3. Device implementation

`tt/model.py` `EmbeddingGemma2Model.forward(x, truncate_dim)`, all on device:

```
text   = TextEmbedding(input_ids)               ids with placeholders -> pad (host, as HF :778)
soft_i = VisionModel(image inputs)              E2 tower, [Bi,1,288,512] per 2560-patch bucket
soft_v = VisionModel(video-frame inputs)        E2 tower, [F,1,160,512] per 1280-patch bucket
soft_a = AudioModel(audio inputs)               E3 tower, [Ba,1,Sa,512]
table  = concat(text rows, soft_i rows, soft_v rows, soft_a rows) -> ROW_MAJOR [R, 512]
embeds = ttnn.embedding(gather, table)          the masked_scatter, as one gather
h      = PLE(embeds); 24 encoder layers         E1 modules (masks, RoPE, layers) unchanged
emb    = L2norm(masked_mean(norm(h) @ W_768))   E1 head + Pooler
emb_d  = L2norm(emb[..., :d])                   only when truncate_dim is set
```

The host builds `gather [B, S]` from the token ids before upload
(`EmbeddingGemma2.gather_index`): `b*S + s` for text and padding positions, and for the k-th
placeholder of a modality (flattened batch order) the table row of the k-th valid soft token of
that modality. A gather of bf16 rows is an exact copy, so the merge adds no error. The towers'
padded rows are never referenced. Text-only batches skip the gather (`gather=None`) and run the E1
path unchanged.

The E1/E2/E3 modules are reused unchanged; this stage added only new files (`tt/model.py`,
`tt/embedding_model.py`, `reference/hf_full_reference.py`, `reference/full_model_inputs.py`, the
tests, the demo).

## 4. Public API

```python
from models.demos.blackhole.embedding_gemma2.tt.embedding_model import EmbeddingGemma2

device = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
model = EmbeddingGemma2.from_pretrained(device, "google/embeddinggemma-2", max_seq_len=8192)

emb = model.encode(
    [
        "What causes the northern lights?",                              # text
        pil_image,                                                       # image
        {"text": "A dog.", "image": pil_image},                          # image + text
        {"audio": {"array": wave_16k, "sampling_rate": 16000}},          # audio
        {"text": "A dashboard.", "video": {"array": frames_uint8,        # video [F,H,W,3]
                                           "video_metadata": {"fps": 1.0}}},
        {"text": "Two photos: <|image|> then <|image|>.", "image": [img_a, img_b]},   # mixed
    ],
    prompt_name="SearchQuery",          # or prompt="task: ... | query: ", applied to every input
    truncate_dim=None,                  # 512 / 256 / 128: MRL on device
)                                       # -> torch.float32 [N, 768], unit L2 norm

keys = model.shape_keys(inputs)          # ShapeKeys encode() will run for these inputs
model.capture_all(keys)                  # one trace per key, all captured in one call
emb = model.encode(inputs, traced=True)  # replays; fails for a shape that was not captured
```

- An input is a `str`, a PIL image, or a dict in the sentence-transformers format (values may be
  lists; image values may be paths; audio arrays are 16 kHz mono). `video_metadata` with `fps`
  tells the processor the frame rate; the processor samples 1 frame per second, at most 32.
- `vision=False` / `audio=False` in `from_pretrained` skip loading a tower.
- Lower-level pieces: `process` (processor output of one input), `plan` (grouping), `prepare_batch`
  (host tensors + `ShapeKey`), `run` (one forward), `replay` / `execute_captured` (trace).

### Batching and buckets

`encode` groups inputs with the same media structure (number of images, number of video frames,
number of audio clips, audio token bucket) and runs each group in batches of up to `batch_size`
(default 8). One forward per batch:

| part | physical shape | buckets |
|---|---|---|
| text | `[B, S]` | S = smallest power of two >= longest item, 32 ... `max_seq_len` (E1 buckets) |
| images | `[Bi, 2560]` patches | Bi = images in the batch (processor: 2520 patches, at most 280 soft tokens) |
| video frames | `[F, 1280]` patches | F = frames in the batch (processor: 1260 patches, at most 140 soft tokens) |
| audio clips | `[Ba, Sa]` tokens | Sa in 32, 64, 128, 256, 512, 768 (25 tokens/s, at most 750) |

The `ShapeKey` (`b{B}_s{S}[_image{Bi}x2560][_video{F}x1280][_audio{Ba}x{Sa}][_d{dim}]`) names one
forward. Any logical length inside a bucket reuses that key's trace: inputs are padded with pad
ids and `keep = 0`, which the masks and the mean pool exclude. The batch size is not padded, so a
batch of 3 texts is its own key.

Trace policy (as in E1-E3): `capture_all(keys)` first allocates every persistent input and runs
the per-shape setup (text band masks, audio conv-weight layout), then compiles every key with one
eager pass, and only then captures. E1 measured with `TT_METAL_TRACE_ALLOC_TRACKING=1` that a new
program's cache buffers, allocated after a capture, can land inside that trace's buffers.

## 5. Accuracy

`tests/pcc/test_full_model.py::test_full_model_item` runs each input through the public API
(`encode`, eager) and through HF fp32 `EmbeddingGemma2Model` + ST pooling with the same processor
output. Bar: cosine >= 0.99 per input (`tests/common.py COSINE_EMBEDDING`, unchanged). Inputs are
in `reference/full_model_inputs.py`; all media is in-tree (images from `models/sample_data`, the
resnet50 demo and retinanet resources; speech from the Whisper demo; the video is
`tech_reports/ttnn/images/ttnn-dashboard.mp4`; `tone3s` is a synthetic tone + chirp).

[M] `EMBEDDING_GEMMA2_RESULTS=$E4/pcc_run1.jsonl pytest tests/pcc/test_full_model.py -q`: 27 passed.

| input | tokens | ShapeKey | cosine vs HF fp32 | cosine at MRL 256 | ST bf16 vs ST fp32 |
|---|---|---|---|---|---|
| text_query | 15 | b1_s32 | 0.99997 | 0.99998 | 0.99997 |
| text_doc | 27 | b1_s32 | 0.99995 | 0.99996 | 0.99996 |
| text_de (German) | 22 | b1_s32 | 0.99998 | 0.99998 | 0.99998 |
| text_ja (Japanese) | 22 | b1_s32 | 0.99997 | 0.99998 | 0.99998 |
| text_es (Spanish) | 17 | b1_s32 | 0.99998 | 0.99998 | 0.99998 |
| text_long | 2080 | b1_s4096 | 0.99997 | 0.99997 | 0.99998 |
| image_house | 268 | b1_s512_image1x2560 | 0.99989 | 0.99992 | 0.99994 |
| image_pajamas | 264 | b1_s512_image1x2560 | 0.99995 | 0.99995 | 0.99995 |
| image_dog | 260 | b1_s512_image1x2560 | 0.99996 | 0.99997 | 0.99997 |
| image_handwriting | 274 | b1_s512_image1x2560 | 0.99997 | 0.99997 | 0.99998 |
| image_house_caption | 277 | b1_s512_image1x2560 | 0.99989 | 0.99990 | 0.99994 |
| image_dog_caption | 267 | b1_s512_image1x2560 | 0.99993 | 0.99993 | 0.99996 |
| image_pajamas_caption | 269 | b1_s512_image1x2560 | 0.99993 | 0.99994 | 0.99994 |
| video_text (8 frames + text) | 986 | b1_s1024_video8x1280 | **0.99957** | 0.99963 | 0.99981 |
| audio_speech2s | 54 | b1_s64_audio1x64 | 0.99990 | 0.99990 | 0.99993 |
| audio_speech10s | 254 | b1_s256_audio1x256 | 0.99996 | 0.99996 | 0.99998 |
| audio_tone3s (synthetic) | 79 | b1_s128_audio1x128 | 0.99966 | 0.99975 | 0.99962 |
| audio_speech2s_text | 57 | b1_s64_audio1x64 | 0.99984 | 0.99985 | 0.99993 |
| audio_speech10s_text | 260 | b1_s512_audio1x256 | 0.99995 | 0.99996 | 0.99995 |
| mixed_image_audio (text + image + audio) | 323 | b1_s512_image1x2560_audio1x64 | 0.99975 | 0.99976 | 0.99992 |
| mixed_two_images (text + 2 images) | 537 | b1_s1024_image2x2560 | 0.99990 | 0.99991 | 0.99989 |

By modality (cosine vs HF fp32, min / mean): text 0.99995 / 0.99997 (6); image 0.99989 / 0.99994
(4); image + caption 0.99989 / 0.99992 (3); video + text 0.99957 (1); audio 0.99966 / 0.99984 (3);
audio + text 0.99984 / 0.99990 (2); mixed 0.99975 / 0.99983 (2). All 21: **min 0.99957, mean
0.99990**. MRL 256: min 0.99963, mean 0.99991. The last column is HF's own bf16 error on the same
inputs (sentence-transformers bf16 vs fp32, `st_reference.json`): TT is within 0.0003 of it on every
input and above it on tone3s and mixed_two_images.

Batching (`test_batched_mixed_modalities`): one `encode` call over 8 inputs of 5 structures ran 5
forwards (`b3_s32`, `b2_s512_image2x2560`, `b1_s64_audio1x64`, `b1_s512_audio1x256`,
`b1_s512_image1x2560_audio1x64`); cosine vs HF fp32 min 0.99975, mean 0.99991 [M].

## 6. Retrieval

`test_retrieval` embeds text captions (rows, no prompt) and media-only inputs (columns) on TT and
on HF fp32, and compares the per-row argmax of the cosine matrices. Committed outputs:
`retrieval_image.json`, `retrieval_audio.json` (both matrices, argmaxes, HF top-1 minus top-2
margin, max |TT - HF|).

[M] same run, 2 passed:

| rows x columns | TT argmax == HF argmax | max abs diff of the matrices | smallest HF margin |
|---|---|---|---|
| 4 captions x 4 images | 4 / 4 rows; every caption picks its own image | 0.00165 | 0.090 (dog) |
| 3 captions x 3 audio clips | 3 / 3 rows | 0.00166 | **0.0046** (speech2s caption) |

The audio matrix is a weak semantic test: both speech captions pick `speech10s` on TT and on HF,
and the "short clip" caption beats the 2 s clip by only 0.0046 on HF. TT agrees with HF on every
row, which is what this test checks. The demo prints the same matrices from traced forwards.

## 7. Trace, determinism, fallback audit, watcher

| check | command | result |
|---|---|---|
| trace replay == eager | `TT_METAL_TRACE_ALLOC_TRACKING=1 pytest tests/pcc/test_full_model_trace.py`: 8 ShapeKeys captured by one `capture_all` (`b1_s32`, `b1_s4096`, `b1_s512_image1x2560`, `b2_s512_image2x2560`, `b1_s512_image1x2560_d256`, `b1_s1024_video8x1280`, `b1_s512_audio1x256`, `b1_s512_image1x2560_audio1x64`) | [M] 11 passed: bit-identical embeddings (max abs diff 0.0) on 7 single inputs, the batch of 2 and MRL 256; traced cosine vs HF fp32 0.99957-0.99997. The allocation tracker (fails on an unsafe allocation) did not fire. Two images with the same key replay one trace and give different embeddings (`test_trace_reuse`). |
| determinism | `test_determinism` (`mixed_image_audio`, `video_text`) | [M] 5 replays + 2 eager runs bit-identical |
| fallback audit | `test_full_model.py::test_no_host_fallback_in_full_forward` (`mixed_image_audio`: both towers + merge; `video_text`) | [M] 0 torch calls in `EmbeddingGemma2Model.forward` (TorchFunctionMode), no ttnn host-transfer entry point called, `throw_exception_on_fallback=True` did not raise. Tracy of one image item: **1194 device ops, 0 host ops** (section 8). |
| merge placement | `test_full_model.py::test_merge_matches_hf` (`mixed_image_audio`, `mixed_two_images`, `video_text`): the on-device merged `inputs_embeds` vs HF's (forward pre-hook on `language_model`), row by row | [M] 3 passed (`$E4/merge_run1.jsonl`). Mean row cosine at the placeholder rows: image 0.99951 / 0.99898, audio 0.99990, video 0.99868 (rows 256 / 524, 50, 960); text rows max relative diff 0.0025 (bf16 rounding of the table). |
| watcher | `TT_METAL_WATCHER=10 pytest tests/pcc/test_full_model.py tests/pcc/test_full_model_trace.py` (includes the trace tests) | [M] 38 passed in 140 s; `$E4/watcher/generated/watcher/watcher.log` (1293 lines) has 0 lines matching `error|assert|exception|overflow|sanitiz|fault` (`grep -ciE`). The merge-placement test was added after this run. |

Why a separate placement test: the pooled cosine alone could miss a misplaced soft token. Mean
pooling is order-free, and only RoPE positions and the 512-radius sliding window make the order
visible. A wrong gather index (for example the rows of two images swapped) pairs unrelated TT and
HF rows and pulls the mean row cosine far below the 0.99 bar. The minimum row cosines (0.927 on
the house image and on a video frame) are the E2 worst tokens, not placement errors. The gather
index uses HF's flattened `masked_scatter` order, and `gather_index` raises on a placeholder /
soft-token count mismatch, like HF's `torch_compilable_check`.

## 8. Performance

`tests/pcc/test_full_model_perf.py::test_full_perf_traced`: all six traces captured together,
3 warm-up replays, then 20 iterations. `device` = 20 back-to-back `execute_trace` + one
`synchronize_device`, per forward: inputs already on device -> normalized embedding, both towers,
the merge, the text encoder and pooling included. `e2e` = `copy_host_to_device_tensor` of every
input + replay + embedding readback, per request (the HF processor on the host is excluded).

[M] `EMBEDDING_GEMMA2_RESULTS=$E4/perf_run1.jsonl pytest tests/pcc/test_full_model_perf.py -k traced -s`:

| case | ShapeKey | device ms / forward | e2e ms |
|---|---|---|---|
| text 1 x 128 | b1_s128 | 8.73 | 8.79 |
| text 1 x 512 | b1_s512 | 18.30 | 18.37 |
| text 8 x 512 | b8_s512 | 74.48 | 74.63 |
| 1 image + caption (267 tokens) | b1_s512_image1x2560 | 73.90 | 74.37 |
| 10 s audio + text (260 tokens) | b1_s512_audio1x256 | 41.99 | 45.20 |
| 8-frame video + text (986 tokens) | b1_s1024_video8x1280 | 230.11 | 232.03 |

The full-model latency is the sum of its parts: image item 73.9 ms = vision tower 55.6 ms (E2,
1 x 2560) + text 1 x 512 18.3 ms; audio item 42.0 ms = audio tower 23.7 ms (E3, 10 s) + 18.3 ms;
video 230.1 ms is about the E2 8 x 1280 vision trace (199.6 ms) + text at 1024. [I] from the E2/E3
READMEs.

tt-perf-report of one warmed eager forward of `image_dog_caption`, signposts `PERF_FULL` /
`PERF_FULL_END` (`tracy/b1_s512_image1x2560/`): **1194 device ops, 0 host ops**, 73,590 us device
time + 949 us gaps. SDPA is 53 % (40 ops: 16 vision + 24 text layers), matmul 23 %. The merge ops
(concat, untilize, the gather `EmbeddingsDeviceOperation`, untilize/tilize of its output; report
rows 1646-1653) take 28 us in total.

```bash
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 PATH=$PWD/python_env/bin:$PATH python_env/bin/python -m tracy -r -p -v -m pytest \
  models/demos/blackhole/embedding_gemma2/tests/pcc/test_full_model_perf.py -k tracy -q
CSV=generated/profiler/reports/2026_10_27_14_36_02/ops_perf_results_2026_10_27_14_36_02.csv
D=models/demos/blackhole/embedding_gemma2/doc/full_model/tracy/b1_s512_image1x2560
tt-perf-report $CSV --start-signpost PERF_FULL --end-signpost PERF_FULL_END --csv $D/forward_perf_report.csv --no-advice \
  > $D/forward_perf_report.console.log
tt-perf-report $CSV --start-signpost PERF_FULL --end-signpost PERF_FULL_END --no-advice --no-color > $D/forward_perf_report.txt
```

The first Tracy attempt (`$E4/logs/tracy_run1.log`) failed in post-processing with `Device data
missing` and 250 "Profiler DRAM buffers were full" warnings, the same failure as E3: the profiler
flushes every 1000 programs by default and one image item is about 1200 programs.
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000` fixed it (`tracy_run2.log`). The full all-layer
stack was profiled, not a reduced-layer variant: one forward is 1194 ops (a 35 MB Tracy file for the whole test), so
the skill's reason for a reduced variant (multi-GB dumps) does not apply here.

No optimization was done in this stage (E5).

## 9. Limits

`doc/context_contract.json` (`full_model`) has the numbers and their sources. Summary:

| limit | value | why | largest tested here |
|---|---|---|---|
| tokens per input | 8192 by default (`max_seq_len`), up to 32768 | model card context, shared by all modalities; 32768 is the E1 bucket limit (dense O(S^2) masks) | 2080 (text), 986 (video) |
| images per input | about 29 at 8192 tokens | 280 soft tokens + 2 markers per image | 2 (`mixed_two_images`) |
| audio per clip | 30 s (750 tokens) | the feature extractor caps a clip at 2999 frames (E3); several clips share the context | 10 s in the full model; 30 s in E3 |
| video frames | 32 | `processor_config.json` `max_frames` at 1 fps; 32 x 142 tokens fits 8192 | 8 |

Other limitations:

- **Batch size is not padded.** Each distinct batch size is a separate ShapeKey and trace. A
  serving caller should batch to a fixed set of sizes (or pad with dummy inputs) and capture
  those. Batches above 8 texts and above 2 images were not tested in the full model.
- **One trace per media structure.** The number of images, frames and clips in a batch is part
  of the key; any image size, aspect ratio or clip length within the buckets reuses the trace.
- **Video input format.** The wrapper passes video frames to the HF processor like ST does. With
  `video_metadata` (`fps`) the processor samples 1 frame per second; without it, transformers
  infers defaults. Video paths are decoded by transformers (PyAV), not by the wrapper; only
  in-memory frames were tested.
- **Inherited from E1-E3:** dense additive attention masks (memory grows as S^2), vision key-padding
  mask, audio per-shape conv weight preparation (setup time per audio ShapeKey), bf16 activations
  (worst single audio soft token cosine 0.76 vs HF fp32 in E3; its effect on the pooled embedding is
  below 0.0004 in cosine here).
- **Untested:** more than 2 images per input, more than 1 video or clip per input, videos longer
  than 8 frames, audio longer than 10 s in the full model, non-default vision token budgets (70 -
  1120), `max_seq_len` 16384/32768 with media, image paths and video paths through the API.

## 10. Out of scope

This is an encoder that returns one embedding per input. The decoder items of the full-model
skill do not apply and were not built: KV cache, page tables, decode step, sampling (split
sampling, top-k/top-p, `tt_out_tok`), generator `prefill_forward` / `decode_forward` /
`generate`, teacher forcing, the AIME24 readiness reference, the degenerate-output check, and the
vLLM serving contract (`serving_contract.json`). The person excluded AIME and evaluation
benchmarks (MTEB, MIEB, MMEB, MSEB) for this stage. Optimization (dtype policy, sharding,
multi-CQ, the dense O(S^2) masks) belongs to E5.
