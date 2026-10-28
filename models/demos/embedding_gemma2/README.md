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
pytest models/demos/embedding_gemma2/demo/demo.py -s          # text + multimodal demos
python models/demos/embedding_gemma2/demo/demo.py             # same, as a script
python models/demos/embedding_gemma2/demo/demo.py --no-hf     # skip the HF fp32 comparison
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
from models.demos.embedding_gemma2.tt.text_encoder import TextEncoder

enc = TextEncoder.from_pretrained(device, "google/embeddinggemma-2", max_seq_len=8192)
batch = tokenizer(["task: search result | query: northern lights"], padding=True, return_tensors="pt")
emb = enc.encode(batch["input_ids"], batch["attention_mask"])               # eager, [B, 768], unit norm

enc.capture_all([(1, 32), (8, 512)])          # every (batch, bucket) the session will replay
emb = enc.encode(batch["input_ids"], batch["attention_mask"], traced=True)  # trace replay
```

Open the device with a trace region (`trace_region_size=128 MiB` is what the tests use).
Logical lengths run in the smallest power-of-two bucket >= L (32 ... 65536); `capture_all`
explains why every trace is captured in one call. The HF-validated maximum is 32768 tokens
(`test_text_model[b1_s32768]` vs the HF fp32 reference). The 65536 bucket has no HF reference: it
is validated only by padding equivalence against 32768 (a 32768-token prompt in bucket 32768 and
padded into 65536 give the same embedding, cosine 1.000000).

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
| `tests/perf/perf.py` | benchmark: traced latency / throughput of every tower and the full model (Benchmark below) + Tracy forward |
| `demo/demo.py` | retrieval demo |
| `doc/` | `context_contract.json`, `functional_text/` (stage report, work log, perf artifacts) |

Every module has a config dataclass, a `from_state_dict(...)` loader that writes the ttnn weight
cache, and a `forward` without host work. Contracts (shapes, layouts, dtypes, mask semantics) are
in each module docstring and in `doc/functional_text/README.md`.

## Tests

```bash
pytest models/demos/embedding_gemma2/tests/pcc -q          # all PCC / trace / audit tests
pytest models/demos/embedding_gemma2/tests/perf/perf.py -s             # quick benchmark (Benchmark below)
pytest models/demos/embedding_gemma2/demo/demo.py -s
python models/demos/embedding_gemma2/tests/perf/bench_sustained.py   # burst + sustained
```

Compute grid: `EMBEDDING_GEMMA2_GRID=<columns>x<rows>` (e.g. `12x10`, the Galaxy grid) runs any
entry point (tests, demo, `perf.py`, `prefill.py`, `dtype_sweep.py`, `dtype_probe.py`) on that
grid. Defaults: the prefill harness `tests/perf/prefill.py` runs on 12 x 10 (E9 contract); every
other entry point uses the device's grid (13 x 10 on a P150a). On a P150a, 12 x 10 is reached with
the runtime option `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=11,9`, which `tt/config.py` sets before
the first device open (it is read once per process; mixing grids needs separate processes). Column
dispatch is already the Blackhole default and gives 13 x 10; details in
`doc/prefill/01-baseline/work_log.md` (E9.1-r1).

## Benchmark (E8)

Full report: `doc/benchmark/REPORT.md` (tables per tower and shape, protocol, comparison with the
earlier stages, limitations); identity of the run: `doc/benchmark/identity.json`; raw records:
`doc/benchmark/run/`. One P150a, selected precision config, traced, run `20261028-063930` on
commit `1bcb04d2f0` **[M]**:

| workload | shape | forward p50 ms | request (h2d_d2h) p50 ms, excl. host prep | public API p50 ms |
|---|---|---|---|---|
| text | 1 x 512 tokens | 6.869 (146 emb/s) | 6.951 | |
| text | 32 x 128 tokens | 22.227 (1,440 emb/s) | 22.756 | |
| text | 1 x 8192 tokens | 93.041 | 93.178 | |
| image (vision tower) | 1 x 2560 patches | 17.395 (57.5 images/s) | 18.151 | |
| audio 30 s (audio tower) | 1 x 768 tokens | 11.961 (83.6 clips/s) | 12.775 | |
| full model, image + text | 267 tokens + 1 image | 24.229 | 24.698 | 33.730 |
| full model, audio 10 s + text | 260 tokens + 1 clip | 14.170 | 14.353 | 46.994 |

`forward` = trace replay + one sync per iteration (inputs resident); `h2d_d2h` = input upload +
replay + output readback, excluding host input preparation (timed separately as `host_prep`: 3.8 ms
for 1 x 2560 vision, 6.7 ms for 1 x 768 audio, under 0.5 ms for text up to 8192; serial
throughput with host prep in `doc/benchmark/REPORT.md`); public API = `EmbeddingGemma2.encode([item], traced=True)` including the
HF processor. 2cq is not supported (one command queue). Vision sustains 5.4 % slower than burst
after 30 s back to back. Text throttles from 1 x 4096 up: in E9 (`doc/prefill/01-baseline/README.md`)
the median AICLK fell to 1312 MHz at 1 x 32768 and 1306 MHz at 1 x 65536 (minimum 1293 / 1268 MHz),
and sustained was 0.6-1.9 % slower than burst at 4096 and above; text up to 2048 held 1350 MHz.
Audio 1 x 768 showed no drift in E8.

How to run `tests/perf/perf.py` (from the tt-metal root, environment as in Setup). A plain run is
quick; every extra is an opt-in env switch (`"1"` = on). They combine, and `-k` picks single cases.
Costs with a warm weight / kernel cache: the plain run and the all-on run are measured whole
(16 s and 318 s wall); the per-option deltas are estimated from the per-class log timestamps of the
full-sweep run, rounded, and sum to the measured 318 s:

| option | adds | about |
|---|---|---|
| (none) | headline shapes, `forward` mode: text 1 x 512, vision 1 x 2560, audio 1 x 768, full model text 472 tokens; finite-output asserts | 0.3 min |
| `EMBEDDING_GEMMA2_PERF_SHAPES=1` | every shape: text 1 x 128 / 512 / 2048 / 8192, 8 x 512, 32 x 128; vision 1 x 2560, 4 x 2560, 8 x 1280; audio 1 x 256, 1 x 768, 4 x 256; full text, image + text, audio + text | +0.5 min |
| `EMBEDDING_GEMMA2_PERF_MODES=1` | `h2d_d2h`, `api` (full model) and `2cq` (records "not supported") besides `forward` | +0.3 min |
| `EMBEDDING_GEMMA2_PERF_LONG=1` | text 1 x 32768 in its own device session | +0.8 min |
| `EMBEDDING_GEMMA2_PERF_PROBE=1` | text batch-capacity probe at 128 tokens (32 ... 8192, stops at the first OOM) | +0.8 min |
| `EMBEDDING_GEMMA2_PERF_SUSTAINED=1` | 30 s sustained vs burst per tower at its headline shape | +2.3 min |
| `EMBEDDING_GEMMA2_PERF_HF_CHECK=1` | HF fp32 comparison of the timed output (one shape per tower, every selected full case) | +0.3 min |
| `EMBEDDING_GEMMA2_PERF_ALL=1` | all of the above: the full sweep of `doc/benchmark/REPORT.md` | 5.3 min |

```bash
pytest models/demos/embedding_gemma2/tests/perf/perf.py -s                                   # quick
EMBEDDING_GEMMA2_PERF_ALL=1 pytest models/demos/embedding_gemma2/tests/perf/perf.py -s       # full sweep
EMBEDDING_GEMMA2_PERF_SHAPES=1 pytest models/demos/embedding_gemma2/tests/perf/perf.py -k "TestVision and b4x2560" -s
```

Other env: `EMBEDDING_GEMMA2_PERF_ITERS` (20 timed iterations), `EMBEDDING_GEMMA2_PERF_WARMUP` (3,
minimum 3), `EMBEDDING_GEMMA2_SUSTAINED_S` (30), `EMBEDDING_GEMMA2_BENCH_DIR` (where `perf.jsonl` /
`perf.csv` are appended; default `doc/benchmark/run`).

## Text tower performance (E5a optimized, E6a precision)

Traced device time per forward on one P150a (burst median of 5 windows; `tests/perf/bench_sustained.py`
through the default construction path). E5a details, the change ledger and tt-perf-report tables:
`doc/optimized_text/README.md`; E6a precision selection: `doc/datatype_sweep/README.md`.

| batch x bucket | E4 functional ms | E5a optimized ms | E6a default ms (sustained) | speedup vs E4 |
|---|---|---|---|---|
| 1 x 128 | 8.75 | 4.92 | 4.81 (4.81) | 1.82x |
| 1 x 512 | 18.31 | 7.32 | 6.83 (6.83) | 2.68x |
| 1 x 2048 | 58.98 | 22.05 | 20.05 (20.05) | 2.94x |
| 8 x 512 | 74.50 | 29.42 | 24.24 (24.62) | 3.07x |
| 1 x 8192 | 411.70 | 109.81 | 93.05 (93.81) | 4.42x |

E6a default measured on code commit `9adacfe162` **[M]**; the E5a column is the E5a report (the
same policy re-measured in the E6a session through `EMBEDDING_GEMMA2_PRECISION=baseline_precision_config.json`:
4.92 / 7.31 / 22.04 / 29.39 / 105.69 ms).

Policy (E6a, `doc/datatype_sweep/selected_precision_config.json`, config `p3_qkv_gu`): bfp8
weights + HiFi2 matmuls with fp32 accumulation and measured program configs; the QKV and MLP
gate / up projection outputs in bfp8, every other activation and the residual stream bf16; HiFi2
SDPA with fp32 accumulation and per-shape chunks; HiFi4 norms; bfp4 additive masks (with an SDPA
block map at buckets >= 4096); intermediates and the residual stream in L1 up to 4096 tokens.
Accuracy margin over the gate cases plus 90 extra short / padded / mid-length inputs: last-hidden
PCC >= 0.99106 (gate 0.99), embedding cosine >= 0.99382 (gate 0.99). The E5a policy is
`EMBEDDING_GEMMA2_PRECISION=<doc/datatype_sweep/baseline_precision_config.json>`; it works on the
P150's 13 x 10 grid but not yet on the 12 x 10 Galaxy grid: at 12 x 10 it crashes in multi-bucket
device sessions of 8192 tokens and below (`TT_THROW`: static circular buffers clash with L1
buffers, first at the 1 x 4096 full-attention SDPA). That is an open item for E9 stage 5 (memory
placement): `doc/prefill/target_contract.json` `accuracy.open_items.baseline_precision_12x10`,
`doc/datatype_sweep/README.md` "Open item for E9 stage 5".
The largest bucket is now 65536 (was 32768); 65536 has no HF reference (validated by padding
equivalence vs 32768 only), so the HF-validated maximum stays 32768.

**E9.1-r re-selection (2026-10-28): the text default is now `sdpa_nofp32acc`** (E5a policy with
SDPA fp32 accumulation off, exact exp; QKV and gate / up outputs back to bf16). `p3_qkv_gu` fails
the E9 b1_s256 contract input (last-hidden PCC 0.98872), and the re-selection ran on the 12 x 10
grid. E9.1-r first picked `p2_sdpa_fast` (the same policy plus SDPA `exp_approx`); the two tie
within noise on time, and amendment E9.1-r2a switched to `sdpa_nofp32acc` for its wider
worst-case probe margin (last-hidden PCC 0.99240 vs 0.99094). Cost vs `p3_qkv_gu` **[M]**: text
1 x 512 forward at 13 x 10, 7.078 / 7.091 ms vs 6.861 ms (+3.3 %, new / control / new in one
session); 12 x 10 ranking aggregate (1x512 + 8x512 burst) 41.845 vs 38.234 ms (+9.4 %). The table
and the policy paragraph above describe `p3_qkv_gu`. Selection, numbers and the E8 numbers it
supersedes: `doc/datatype_sweep/README.md` ("E9.1-r re-selection", "Amendment E9.1-r2a") and
`doc/benchmark/REPORT.md` (addendum). The serving wrapper (`demo/generator_vllm.py`) reads the
same file, so served embeddings change with it.

## Vision tower

`tt/vision/` implements HF `Gemma4VisionModel` + `embed_vision`: patch embedder with the learned
2D position table, 16 bidirectional encoder layers (q/k/v norms, 2D axial RoPE, key-padding mask,
sandwich norms, gated gelu-tanh MLP), 3x3 position pooling x sqrt(768), and the scale-free RMSNorm
+ Linear 768->512 that maps soft tokens into the text hidden size. Images use 2520 patches (280
soft tokens at most) and run in the 2560-row bucket. Video frames use 1260 patches (140 at most)
and run in the 1280-row bucket.

```python
from transformers import AutoProcessor
from models.demos.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder

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
| `tt/vision/rope.py` | host `axial_cos_sin`; device `AxialRope` (fused `rotary_embedding_llama`) |
| `tt/vision/attention.py` | `VisionAttention` (12 x 64 MHA, q/k/v norms, scaling 1.0; SDPA table, windowed path) |
| `tt/vision/linear.py` | `vision_linear` (vision program-config rule over `tt/linear.py`) |
| `tt/vision/mlp.py` | `VisionMLP` (weights per the precision policy, gelu as the multiply's pre-activation) |
| `tt/vision/encoder_layer.py` | `VisionEncoderLayer` (sandwich norms, L1 residual for one image) |
| `tt/vision/pooler.py` | host `pooling_matrix`; device `VisionPooler` (one batched matmul) |
| `tt/vision/embedder.py` | `VisionEmbedder` (`embed_vision`) |
| `tt/vision/vision_model.py` | `VisionModel` (device-only forward) |
| `tt/vision/vision_encoder.py` | `VisionEncoder` (host: inputs, upload, trace capture/replay, readback) |
| `reference/hf_vision_reference.py` | HF fp32 vision reference + activation hooks |
| `tests/pcc/vision/` | PCC per module, full stack, soft tokens, trace, determinism, audit, perf |
| `doc/functional_vision/` | E2 stage report, work log, tt-perf-report artifacts |
| `doc/optimized_vision/` | E5b stage report, work log, tt-perf-report before / after |
| `tests/perf/bench_vision.py` | traced burst / sustained latency per (batch, bucket) |

```bash
pytest models/demos/embedding_gemma2/tests/pcc/vision -q                      # PCC + trace + audit + perf
pytest models/demos/embedding_gemma2/tests/pcc/vision/test_vision_perf.py -k traced -s
```

Measured on real images and an 8-frame video (E5b path): per-layer PCC >= 0.99993 and soft-token
PCC 0.9965-0.9997 per image vs the HF fp32 reference (bar 0.99; 0.9972-0.9997 before the
per-image MLP review fix, which reshuffles the batched cases). The E5b optimizations cost a small,
systematic amount of accuracy against E2. On the final E5b code the mean soft-token PCC drops by
4.0e-4 over 15 images and frames, and by up to 1.34e-3 on video frame 7. The worst-token cosine
drops on the portrait image (0.943 to 0.864; HF bf16 0.866) and on video frame 3 (0.952 to 0.866;
HF bf16 0.965). E5b also scores below HF's own bf16 run on 12 of the 15 images (portrait 0.99804
vs 0.99843, frame 3 0.99653 vs 0.99841, frame 7 0.99769 vs 0.99927). The E5b ablation (run before
the per-image MLP fix) attributes most of the loss to the gelu-multiply change and the
bf16-weight projection / MLP config change. Both are to be revisited in the E6 datatype sweep. See
`doc/optimized_vision/README.md` ("Precision").

### Vision tower performance (E5b, optimized)

Traced device ms per forward (`tests/perf/bench_vision.py`, burst median of 5 windows; sustained
after a 10 s soak is within 0.1 ms). "E2 code today" was re-measured on the same day and card as
E5b. It is slower than the E2 report (55.60 / 294.47 / 199.61 ms). The extra time is in the
default-config HiFi4 bf16 matmuls, and the cause is unexplained: the libraries predate E2 and the
E5a text control (1 x 512, 7.315 ms) reproduces. See `doc/optimized_vision/README.md`.

| batch x bucket | E2 code today ms | E2 published ms | E5b ms | speedup vs today / vs published | ms per image / frame |
|---|---|---|---|---|---|
| 1 x 2560 (image) | 62.80 | 55.60 | 26.27 | 2.39x / 2.12x | 26.3 |
| 4 x 2560 (images) | 323.26 | 294.47 | 116.02 | 2.79x / 2.54x | 29.0 |
| 8 x 1280 (video frames) | 228.31 | 199.61 | 88.14 | 2.59x / 2.26x | 11.0 |

E5b numbers include the review fix (per-image MLP for batches, `tt/vision/mlp.py`). Before it, E5b
(`47824a6e41`) measured 26.26 / 121.62 / 94.64 ms: 2.39x / 2.66x / 2.41x vs the E2 code today and
2.12x / 2.42x / 2.11x vs E2 published.

Policy: SDPA per-shape chunk/grid table with HiFi2 + fp32 accumulation and a bfp4 key-padding
mask (windowed SDPA without a mask for 8+ video frames), fused axial RoPE, bf16 weights with
HiFi2 2D-multicast / minimal_matmul configs, gelu fused into the gate * up multiply, activations,
the QKV and MLP up / gate outputs and the residual stream in L1 for one image, and the MLP run one
image / frame at a time for batches so it reuses that L1 path. bfp8 weights and SDPA without fp32
accumulation were measured and left for E6 (accuracy). Details: `doc/optimized_vision/README.md`.

### Vision tower precision (E6b default)

`doc/datatype_sweep/selected_precision_config.json`, config `v_exp_bfp8w4`: SDPA `exp_approx` on,
bfp8 weights for the q/k/v, o, gate/up and down projections (HiFi2 + fp32 accumulation, bf16
outputs), patch embedder and `embed_vision` bf16 HiFi4, HiFi4 norms, bfp4 mask. Traced device ms,
normal construction path, burst / sustained **[M]** (`doc/datatype_sweep/README.md`, Gates):
1 x 2560 17.36 / 18.24, 4 x 2560 88.57 / 89.62, 8 x 1280 69.20 / 69.22 (E5b policy in the same
session: 26.12 / 115.06 / 87.20 burst). Soft-token PCC on the tests >= 0.99477 (bar 0.99). Back to
E5b: `EMBEDDING_GEMMA2_PRECISION=<doc/datatype_sweep/baseline_precision_config.json>`.

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
from models.demos.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

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
| `tt/audio/clippable_linear.py` | `ClippableLinear` (clamp -> linear with the output clamp fused, bounds folded) |
| `tt/audio/linear.py` | `audio_linear`: 2D-multicast program configs, fused activation (E5c) |
| `tt/audio/norm.py` | `AudioRMSNorm`: RMSNorm with the policy's norm options (block-sharded for short inputs, E6b) |
| `tt/audio/program_configs.py` | block-matmul configs for the batched attention / lconv matmuls (E5c) |
| `tt/audio/subsample.py` | `AudioSubsample` (2 x conv2d + LN + ReLU + masks, input_proj) |
| `tt/audio/feed_forward.py` | `AudioFeedForward` (Macaron half-step FFN) |
| `tt/audio/attention.py` | `AudioAttention` (fused QKV, band attention on 32-row blocks, rel-pos gather shift, softcap); host `attention_mask` |
| `tt/audio/light_conv.py` | `AudioLightConv1d` (GLU, causal depthwise conv by exact 0/1 shift matmuls) |
| `tt/audio/conformer_layer.py` | `AudioConformerLayer` |
| `tt/audio/embedder.py` | `AudioOutputProj`, `AudioEmbedder` (`embed_audio`) |
| `tt/audio/audio_model.py` | `AudioModel` (device-only forward; `prepare` lays out conv weights per shape) |
| `tt/audio/audio_encoder.py` | `AudioEncoder` (host: features, masks, upload, trace capture/replay, readback) |
| `reference/hf_audio_reference.py` | HF fp32 audio reference + activation hooks |
| `tests/pcc/audio/` | PCC per layer kind, crops around the 12-token chunk, 12-layer stack, soft tokens, trace, determinism, audit, perf |
| `doc/functional_audio/` | E3 stage report, work log, tt-perf-report artifacts |
| `doc/optimized_audio/` | E5c stage report, work log, tt-perf-report before / after |
| `tests/perf/bench_audio.py` | traced burst / sustained latency per (batch, bucket) |
| `tests/perf/dtype_sweep_towers.py` | E6b vision / audio datatype sweep: screen, probe, report |

```bash
pytest models/demos/embedding_gemma2/tests/pcc/audio -q -k "not tracy"          # PCC + trace + audit + perf
pytest models/demos/embedding_gemma2/tests/pcc/audio/test_audio_perf.py -k traced -s
python models/demos/embedding_gemma2/tests/perf/bench_audio.py --cases 1x256,1x768,4x256
```

Measured on real in-tree speech (0.64 s to 30 s, and a padded batch of 2), E5c path: per-layer
PCC >= 0.99998, and soft-token PCC 0.9986-0.9999 per clip vs the HF fp32 reference (bar 0.99;
E3: 0.9984-0.9999). HF's own bf16 run scores 0.9985-0.9999 against the same reference. The worst
single soft token is much lower: per-token cosine 0.834 (s10 token 126; E3 0.764) and 0.773 (s30
token 267), against 0.851 and 0.820 for HF-bf16 on the same tokens. The synthetic tone clip
scores 0.99769 (E3 0.99806). Details, the fp32-activation precision lever and limitations are in
`doc/functional_audio/README.md` and `doc/optimized_audio/README.md`.

### Audio tower performance (E5c, optimized)

Traced device ms per forward (`tests/perf/bench_audio.py`, burst median of 5 windows; sustained
after a 10 s soak is within 0.01 ms). "E3 code today" was re-measured on the same day and card.
Like the E2 vision code in E5b, it is slower than its own report (the default-config HiFi4 bf16
matmuls; cause unexplained). See `doc/optimized_audio/README.md`.

| batch x bucket | E3 code today ms | E3 published ms | E5c ms | speedup vs today / vs published | ms per clip |
|---|---|---|---|---|---|
| 1 x 256 (10 s) | 25.78 | 23.67 | 8.74 | 2.95x / 2.71x | 8.74 |
| 1 x 768 (30 s) | 54.28 | 47.79 | 15.02 | 3.61x / 3.18x | 15.02 |
| 4 x 256 (4 x 10 s) | 88.34 | 80.48 | 19.15 | 4.61x / 4.20x | 4.79 |

Policy (precision unchanged from E3: bf16 weights, HiFi4 + fp32 accumulation, fp32 q / k / v /
logits / softmax / taps): block-matmul configs for the batched band-attention and lconv matmuls,
2D-multicast configs for the projections with the clip clamp fused into the matmul, fused QKV with
`nlp_create_qkv_heads` / `nlp_concat_heads`, unary steps fused into neighbouring ops, the rel-pos
shift as one gather, the lconv causal shift as exact 0/1 shift matmuls at batch 1, the mel input
in tile layout, and intermediates in L1 when they fit (64 KB per bank; 96 KB for the fused QKV
output). All but two changes are bit-identical to E3; those two (projection and output-head
program configs) change only the accumulation order. HiFi2 projections (-1.4 ms at 1 x 768),
LoFi shift matmuls and a block-sharded RMSNorm were measured and left for E6 (accuracy).

### Audio tower precision (E6b default)

`doc/datatype_sweep/selected_precision_config.json`, config `a_h2b8_qkvbf16_shift2_sharded`: bfp8
weights for every projection with HiFi2 + fp32 accumulation, fused QKV output bf16 (was fp32),
lconv shift matmuls HiFi2, block-sharded RMSNorm for <= 24 tile rows; subsampling convs, band
attention and norms HiFi4. Traced device ms, normal construction path, burst / sustained **[M]**:
1 x 256 7.31 / 7.31, 1 x 768 11.95 / 11.95, 4 x 256 15.38 / 15.38 (E5c policy in the same session:
8.72 / 14.92 / 19.07 burst). Soft-token PCC on the tests >= 0.99665 (bar 0.99). The low worst-token
cosines of s10 token 126 / s30 token 267 are a property of the model (measured,
`doc/datatype_sweep/README.md`, "Audio worst-token sensitivity").

## Full model

`tt/model.py` (`EmbeddingGemma2Model`, device forward) and `tt/embedding_model.py`
(`EmbeddingGemma2`, public host API) join the three towers:

```python
from models.demos.embedding_gemma2.tt.embedding_model import EmbeddingGemma2

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
pytest models/demos/embedding_gemma2/tests/pcc/test_full_model.py -q        # accuracy, retrieval, audit
pytest models/demos/embedding_gemma2/tests/pcc/test_full_model_trace.py -q  # trace == eager, determinism
pytest models/demos/embedding_gemma2/tests/pcc/test_full_model_perf.py -k traced -s
```

Measured on 21 real inputs (6 texts incl. German, Japanese, Spanish and a 2080-token document;
4 images; 3 images with captions; an 8-frame video with text; 3 audio clips; 2 audio clips with
text; 2 mixed inputs): embedding cosine vs HF fp32 min 0.99957, mean 0.99990; at MRL 256 min
0.99963. TT and HF pick the same best match on every row of the caption x image and
caption x audio similarity matrices. Traced latency at E4, before the E5a text and E5b vision
optimizations (inputs on device -> normalized embedding): text 1 x 512 18.3 ms, 1 image +
caption 73.9 ms, 10 s audio + text 42.0 ms, 8-frame video 230.1 ms. Details are in
`doc/full_model/README.md`.

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
