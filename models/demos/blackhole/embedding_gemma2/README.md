# EmbeddingGemma 2

TTNN implementation of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
on one Blackhole P150a: text, image, video, audio and mixed inputs to one 768-dim embedding. The
output matches `SentenceTransformer("google/embeddinggemma-2").encode(input)`
(`[Transformer, Pooling(mean), Normalize]`), with optional MRL truncation to 512/256/128.

The text tower is token embedding, projection-only per-layer inputs (PLE), 24 bidirectional encoder
layers, final norm, 512->768 projection, masked mean pooling and L2 normalization. Image and video
soft tokens come from the vision tower, audio soft tokens from the audio tower; they are placed
into their placeholder positions on device before the text tower.

Status: stage E1 (functional text encoder), E2 (vision tower, see [Vision tower](#vision-tower)),
E3 (audio tower, see [Audio tower](#audio-tower)) and E4 (full multimodal model and public API,
see [Full model](#full-model)).

## How to run

```bash
source python_env/bin/activate          # plus the Setup exports below
pytest models/demos/blackhole/embedding_gemma2/demo/demo.py -s          # text + multimodal demos
python models/demos/blackhole/embedding_gemma2/demo/demo.py             # same, as a script
python models/demos/blackhole/embedding_gemma2/demo/demo.py --no-hf     # skip the HF fp32 comparison
```

The demo embeds in-tree media on device with traced forwards: text retrieval, image <-> text,
audio <-> text and video <-> text similarity matrices, and prints the TT vs HF fp32 cosine of
every embedding.

## Setup

Run from the tt-metal root inside `python_env`. The HF reference needs transformers >= 5.20
(`models/embedding_gemma2`); `python_env` pins 5.12.1, so the tests put an overlay directory first
on `PYTHONPATH`. The overlay holds only `transformers==5.20.2`, `huggingface_hub==1.33.0` and
`tokenizers==0.23.3`, installed with `--no-deps --target` (python_env is unchanged):

```bash
uv pip install --python python_env/bin/python --no-deps --target /local/ttuser/gtobar/venvs/tf_overlay \
    transformers==5.20.2 "huggingface_hub>=1.31,<2" tokenizers==0.23.3

export TT_METAL_HOME=$PWD PYTHONPATH=/local/ttuser/gtobar/venvs/tf_overlay:$PWD
export TT_VISIBLE_DEVICES=0 HF_MODEL=google/embeddinggemma-2
```

The overlay reference was checked against a separate python 3.11 / transformers 5.21.0 venv: the
two `EmbeddingGemma2Model` outputs on a 2-prompt padded batch have max abs difference 0.0 (fp32).
The tests compute the reference live (no stored goldens); the HF model code is used unmodified.

TTNN weights are cached under `~/.cache/ttnn/models/embedding_gemma2/<snapshot>/`
(override with `EMBEDDING_GEMMA2_CACHE`).

## Use

```python
from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder

enc = TextEncoder.from_pretrained(device, "google/embeddinggemma-2", max_seq_len=8192)
batch = tokenizer(["task: search result | query: northern lights"], padding=True, return_tensors="pt")
emb = enc.encode(batch["input_ids"], batch["attention_mask"])               # eager, [B, 768], unit norm

enc.capture_all([(1, 32), (8, 512)])          # every (batch, bucket) the session will replay
emb = enc.encode(batch["input_ids"], batch["attention_mask"], traced=True)  # trace replay
```

Open the device with a trace region (`trace_region_size=128 MiB` is what the tests use).
Logical lengths run in the smallest power-of-two bucket >= L (32 ... 32768); `capture_all`
explains why every trace is captured in one call.

## Layout

| path | content |
|---|---|
| `tt/config.py` | config dataclasses built from the HF config; buckets; compute configs |
| `tt/embedding.py` | `TextEmbedding` (scaled token embedding) |
| `tt/per_layer_input.py` | `PerLayerInput` (PLE projection + norm), `PerLayerInputBlock` (PLE gate in each layer) |
| `tt/rms_norm.py` | `RMSNorm` (weight used as-is; `with_scale=False` for v_norm) |
| `tt/rope.py` | `RotaryTables` (cos/sin per layer type) |
| `tt/masks.py` | `AttentionMasks` (band + key-padding additive masks) |
| `tt/attention.py` | `Attention` (sliding: hd 256 / 2 KV; full: hd 512 / 1 KV) |
| `tt/mlp.py` | `MLP` (gated gelu-tanh) |
| `tt/encoder_layer.py` | `EncoderLayer` (sandwich norms, PLE block, layer_scalar) |
| `tt/pooler.py` | `Pooler` (masked mean + L2 normalize) |
| `tt/text_model.py` | `TextModel` (device-only forward) |
| `tt/text_encoder.py` | `TextEncoder` (host: bucketing, upload, trace capture/replay, readback) |
| `reference/hf_reference.py` | HF fp32 reference loader + activation capture |
| `tests/pcc/` | PCC tests per layer kind, full stack, trace, determinism, fallback audit |
| `tests/perf/perf.py` | traced latency + Tracy forward |
| `demo/demo.py` | retrieval demo |
| `doc/` | `context_contract.json`, `functional_text/` (stage report, work log, perf artifacts) |

Every module has a config dataclass, a `from_state_dict(...)` loader that writes the ttnn weight
cache, and a `forward` without host work. Contracts (shapes, layouts, dtypes, mask semantics) are
in each module docstring and in `doc/functional_text/README.md`.

## Tests

```bash
pytest models/demos/blackhole/embedding_gemma2/tests/pcc -q          # all PCC / trace / audit tests
pytest models/demos/blackhole/embedding_gemma2/tests/perf/perf.py -k traced -s
pytest models/demos/blackhole/embedding_gemma2/demo/demo.py -s
```

## Vision tower

`tt/vision/` implements HF `Gemma4VisionModel` + `embed_vision`: patch embedder with the learned
2D position table, 16 bidirectional encoder layers (q/k/v norms, 2D axial RoPE, key-padding mask,
sandwich norms, gated gelu-tanh MLP), 3x3 position pooling x sqrt(768), and the scale-free RMSNorm
+ Linear 768->512 that maps soft tokens into the text hidden size. Images use 2520 patches (280
soft tokens at most) and run in the 2560-row bucket. Video frames use 1260 patches (140 at most)
and run in the 1280-row bucket.

```python
from transformers import AutoProcessor
from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder

proc = AutoProcessor.from_pretrained("google/embeddinggemma-2")
venc = VisionEncoder.from_pretrained(device, "google/embeddinggemma-2")
px = proc.image_processor(images=[image], return_tensors="pt")
soft = venc.encode(px["pixel_values"], px["image_position_ids"])        # list of [n_i, 512], eager

venc.capture_all([(1, 2560), (8, 1280)])     # every (batch, bucket) the session will replay
soft = venc.encode(px["pixel_values"], px["image_position_ids"], traced=True)
vid = proc.video_processor(videos=[frames], return_tensors="pt")     # frames: [F, H, W, 3] uint8
soft = venc.encode(vid["pixel_values_videos"], vid["video_position_ids"])  # one entry per frame
```

| path | content |
|---|---|
| `tt/vision/config.py` | `VisionConfig` from the HF vision config; patch buckets |
| `tt/vision/weights.py` | vision / embed_vision tensor loading (`.linear.weight` folded) |
| `tt/vision/patch_embedder.py` | `VisionPatchEmbedder` (input projection + 2D position table) |
| `tt/vision/rope.py` | host `axial_cos_sin`; device `AxialRope` |
| `tt/vision/attention.py` | `VisionAttention` (12 x 64 MHA, q/k/v norms, scaling 1.0) |
| `tt/vision/encoder_layer.py` | `VisionEncoderLayer` (sandwich norms; E1 `MLP`) |
| `tt/vision/pooler.py` | host `pooling_matrix`; device `VisionPooler` (one batched matmul) |
| `tt/vision/embedder.py` | `VisionEmbedder` (`embed_vision`) |
| `tt/vision/vision_model.py` | `VisionModel` (device-only forward) |
| `tt/vision/vision_encoder.py` | `VisionEncoder` (host: inputs, upload, trace capture/replay, readback) |
| `reference/hf_vision_reference.py` | HF fp32 vision reference + activation hooks |
| `tests/pcc/vision/` | PCC per module, full stack, soft tokens, trace, determinism, audit, perf |
| `doc/functional_vision/` | stage report, work log, tt-perf-report artifacts |

```bash
pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision -q                      # PCC + trace + audit + perf
pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision/test_vision_perf.py -k traced -s
```

Measured on real images and an 8-frame video: per-module PCC >= 0.99995 and soft-token PCC
0.9977-0.9997 per image vs the HF fp32 reference. HF's own bf16 run scores 0.9970-0.9998 against
the same reference. Traced latency is 55.6 ms for 1 image and 294.5 ms for a batch of 4. Details
and limitations are in `doc/functional_vision/README.md`.

## Audio tower

`tt/audio/` implements HF `Gemma4AudioModel` + `embed_audio`:

- subsampling: two on-device `ttnn.conv2d` (k3 s2) with channel LayerNorm, ReLU and input-mask
  zeroing, then flatten and `input_proj`
- 12 Macaron conformer layers: FFN1, chunked causal-local attention with the relative-position
  term, per-dim query scale and logit softcap 50, LightConv1d (GLU, causal depthwise k5,
  RMSNorm, SiLU), FFN2, norm_out
- the checkpoint's per-tensor clip bounds folded into `ttnn.clamp`
- `output_proj` 1024->1536 (+bias), then `embed_audio` (scale-free RMSNorm + Linear 1536->512)

Clips run in token buckets of 32 to 768 (25 tokens per second, at most 750 tokens for 30 s).

```python
from models.demos.blackhole.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

aenc = AudioEncoder.from_pretrained(device, "google/embeddinggemma-2")
feats, mask = aenc.extract_features([waveform_16k])                 # HF feature extractor (host)
soft = aenc.encode(feats, mask)                                     # list of [n_clip, 512], eager

aenc.capture_all([(1, 256), (1, 768)])       # every (batch, token bucket) the session will replay
soft = aenc.encode(feats, mask, traced=True)
```

| path | content |
|---|---|
| `tt/audio/config.py` | `AudioConfig` from the HF audio config; token buckets |
| `tt/audio/weights.py` | audio / embed_audio tensor loading; clip bounds |
| `tt/audio/clippable_linear.py` | `ClippableLinear` (clamp -> linear -> clamp, bounds folded) |
| `tt/audio/subsample.py` | `AudioSubsample` (2 x conv2d + LN + ReLU + masks, input_proj) |
| `tt/audio/feed_forward.py` | `AudioFeedForward` (Macaron half-step FFN) |
| `tt/audio/attention.py` | `AudioAttention` (band attention on 32-row blocks, rel-pos shift, softcap); host `attention_mask` |
| `tt/audio/light_conv.py` | `AudioLightConv1d` (GLU, causal depthwise conv by selection matmul) |
| `tt/audio/conformer_layer.py` | `AudioConformerLayer` |
| `tt/audio/embedder.py` | `AudioOutputProj`, `AudioEmbedder` (`embed_audio`) |
| `tt/audio/audio_model.py` | `AudioModel` (device-only forward; `prepare` lays out conv weights per shape) |
| `tt/audio/audio_encoder.py` | `AudioEncoder` (host: features, masks, upload, trace capture/replay, readback) |
| `reference/hf_audio_reference.py` | HF fp32 audio reference + activation hooks |
| `tests/pcc/audio/` | PCC per layer kind, crops around the 12-token chunk, 12-layer stack, soft tokens, trace, determinism, audit, perf |
| `doc/functional_audio/` | stage report, work log, tt-perf-report artifacts |

```bash
pytest models/demos/blackhole/embedding_gemma2/tests/pcc/audio -q -k "not tracy"          # PCC + trace + audit + perf
pytest models/demos/blackhole/embedding_gemma2/tests/pcc/audio/test_audio_perf.py -k traced -s
```

Measured on real in-tree speech (0.64 s to 30 s, and a padded batch of 2): per-layer-kind PCC
>= 0.9997, and soft-token PCC 0.9984-0.9999 per clip vs the HF fp32 reference. HF's own bf16 run
scores 0.9985-0.9999 against the same reference. The worst single soft token is much lower:
per-token cosine 0.764 (s10 token 126) and 0.773 (s30 token 267), against 0.851 and 0.820 for
HF-bf16 on the same tokens. Traced latency is 23.7 ms for a 10 s clip,
47.8 ms for 30 s, and 80.5 ms for 4 x 10 s. Details, the fp32-activation precision lever and
limitations are in `doc/functional_audio/README.md`.

## Full model

`tt/model.py` (`EmbeddingGemma2Model`, device forward) and `tt/embedding_model.py`
(`EmbeddingGemma2`, public host API) join the three towers:

```python
from models.demos.blackhole.embedding_gemma2.tt.embedding_model import EmbeddingGemma2

model = EmbeddingGemma2.from_pretrained(device, "google/embeddinggemma-2", max_seq_len=8192)
inputs = [
    "task: search result | query: northern lights",
    {"text": "A dog.", "image": pil_image},
    {"audio": {"array": wave_16k, "sampling_rate": 16000}},
    {"text": "A dashboard.", "video": {"array": frames, "video_metadata": {"fps": 1.0}}},
]
emb = model.encode(inputs)                      # [4, 768] float32, unit norm (eager)
model.capture_all(model.shape_keys(inputs))     # one trace per ShapeKey, all in one call
emb = model.encode(inputs, traced=True)         # replays the captured traces
# MRL: encode(..., truncate_dim=256); traced MRL needs shape_keys(inputs, truncate_dim=256) captured too
```

Inputs use the sentence-transformers format and are formatted exactly as ST 6.2.0 does (same
token ids on all 28 test inputs). The host runs the HF processor and builds a gather index from
the token ids. On device, one forward runs the towers, places the soft tokens into the
`<|image|>` / `<|video|>` / `<|audio|>` slots with one `ttnn.embedding` gather, then runs the text
encoder, pooling and normalization; it is captured as one trace per shape.

| path | content |
|---|---|
| `tt/model.py` | `EmbeddingGemma2Model` (towers + on-device merge + text encoder + pool + MRL), `ShapeKey` |
| `tt/embedding_model.py` | `EmbeddingGemma2` (ST formatting, batching, buckets, gather index, trace, readback) |
| `reference/hf_full_reference.py` | HF fp32 `EmbeddingGemma2Model` + ST pooling |
| `reference/full_model_inputs.py` | in-tree test / demo inputs (text, images, video, audio, mixed) |
| `tests/pcc/test_full_model*.py` | accuracy per input, batching, retrieval, audit, trace, perf |
| `doc/full_model/` | merge semantics with HF file:line, API, buckets, results, limits |

```bash
pytest models/demos/blackhole/embedding_gemma2/tests/pcc/test_full_model.py -q        # accuracy, retrieval, audit
pytest models/demos/blackhole/embedding_gemma2/tests/pcc/test_full_model_trace.py -q  # trace == eager, determinism
pytest models/demos/blackhole/embedding_gemma2/tests/pcc/test_full_model_perf.py -k traced -s
```

Measured on 21 real inputs (6 texts incl. German, Japanese, Spanish and a 2080-token document;
4 images; 3 images with captions; an 8-frame video with text; 3 audio clips; 2 audio clips with
text; 2 mixed inputs): embedding cosine vs HF fp32 min 0.99957, mean 0.99990; at MRL 256 min
0.99963. TT and HF pick the same best match on every row of the caption x image and
caption x audio similarity matrices. Traced latency (inputs on device -> normalized embedding):
text 1 x 512 18.3 ms, 1 image + caption 73.9 ms, 10 s audio + text 42.0 ms, 8-frame video
230.1 ms. Details are in `doc/full_model/README.md`.

## Out of scope (decoder-only items of the bring-up skill)

This model is an encoder that returns one embedding per input. It has no paged KV cache, decode
step, page table, current-position tensor or sampling, so none is built. Chunked prefill is not
needed either: the whole sequence runs in one pass per bucket, up to 32768 tokens. The vision
tower is also a one-pass bidirectional encoder per image (or video frame), and the audio tower is
a one-pass causal-local encoder per clip; the same items do not apply to them.

## Limitations

See `doc/functional_text/README.md` (dense O(S^2) masks, power-of-two buckets, supported context
32768 of the advertised 262144), `doc/functional_vision/README.md` (dense key-padding mask,
batch scaling, video tested up to 8 frames) and `doc/functional_audio/README.md` (bf16
activation drift vs HF-bf16, per-shape conv weight preparation, batch tested up to 4).
