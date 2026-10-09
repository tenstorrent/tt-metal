# Vision on one p150a (stages 12A and 12B)

The pplx-decider vision tower is HF `Qwen3_5VisionModel`: patch embed, learned position embedding,
27 ViT blocks and a 2x2 patch merger. It turns one image into 5120-wide features, the rows that
`Qwen3_5Model` places at the `<|image_pad|>` tokens of the text. Stage 12A runs the tower on one
Blackhole p150a in BF16 and checks it against the HF BF16 golden. Stage 12B splices the features
into the text embeddings on device, adds the 3D mRoPE positions to the 16 full-attention layers
and runs image decisions end to end through `TTDecider.predict(state, question, images=[...])`.

## Image decisions end to end (stage 12B)

```text
host input prep (per request)                         device (no host calls)
 open_image -> chat template -> processor            vision tower (12A, BF16) -> features [n/4, 5120]
 get_rope_index -> 3D ids [3, S] -> cos/sin  ---+    embedding(tokens) ROW_MAJOR [bucket, 5120]
 splice index (s or bucket + k), vision tables  |    concat [text ; features] -> embedding(splice_index)
 upload: tokens, splice index, cos/sin, pixels -+--> 64 layers (16 full-attention use the request cos/sin)
                                                     -> final norm -> readout -> mask -> /T -> softmax
                                                     -> readback of the count probabilities
```

Results against the HF BF16 image-decision golden (8 rows, bucket 1024, 174-370 tokens, 64-256
image tokens; `tests/e2e/test_image_model.py`; measured):

| row | type | tokens | HF choice (prob, top-2 gap) | TT choice (prob) | logit PCC | final-hidden PCC | max prob diff |
|---|---|---:|---|---|---:|---:|---:|
| v01_dominant_color | choice | 174 | blue (0.9914, 0.989) | blue (0.9915) | 0.99996 | 0.99975 | 0.0001 |
| v02_count_circles | choice | 357 | 4 (0.5100, 0.080) | 4 (0.4705, **tie**) | 0.99940 | 0.99964 | 0.0402 |
| v03_receipt_total | choice | 315 | $17.50 (0.9942, 0.992) | $17.50 (0.9941) | 0.99999 | 0.99979 | 0.0001 |
| v04_tallest_bar | choice | 370 | Thu (0.9886, 0.985) | Thu (0.9886) | 0.99997 | 0.99973 | 0.0001 |
| v05_red_circle_yes | noul | 183 | true (0.9949, 0.990) | true (0.9949) | 1.00000 | 0.99973 | 0.0000 |
| v06_red_circle_no | noul | 243 | false (0.9847, 0.969) | false (0.9847) | 1.00000 | 0.99978 | 0.0000 |
| v07_brightness_dark | score | 271 | 0 (0.7825, 0.593) | 0 (0.7820) | 0.99990 | 0.99982 | 0.0005 |
| v08_progress_fill | score | 364 | 4 (0.9315, 0.885) | 4 (0.9319) | 0.99998 | 0.99966 | 0.0004 |

- Decisions: 8/8 agree (gate >= 7/8). Logit PCC min 0.99940 (gate 0.99).
- **v02 is an exact tie on TT.** HF logits for "4" and "5" are 23.0 and 22.625. TT gives 22.875
  for both. The readout output is BF16 on both sides (the HF readout is BF16 too), and the BF16
  step at this magnitude is 0.125. So the two TT probabilities are equal (0.4705). The app's
  `answer` takes the first maximum (`max(range(n), key=values.__getitem__)`), and so does
  `torch.argmax`, so TT answers "4" as HF does. The margin is zero. If the row flipped it would be
  a non-near-tie miss (HF gap 0.080 > 0.05) and the gate would fail.
- Splice gate (measured, all 8 rows): the image-token rows of the spliced `inputs_embeds` equal the
  TT vision-tower output bit for bit, and the text rows equal the TT text embedding bit for bit.
  The spliced-embedding PCC vs the HF golden is reported per row and held to the 12A tower bar
  (0.99): min 0.998655 (v01), which is the tower's own feature PCC on that one-color image
  (0.998654, see the 12A table below).
- Gate correction (orchestrator decision, 2026-10-09): the first 12B gate asked for spliced PCC
  >= 0.999 vs the golden. The splice is a row copy and is measured bit-exact, so that PCC only
  measures the tower, and 0.999 was a second, stricter tower bar than the 12A gate (0.99). v01
  failed it at 0.998655 while passing 12A. The test now checks the splice by exact equality
  (TT vs TT) and the golden PCC against the 12A bar.
- Layer trace (last-token residual PCC after each of the 64 layers): v02 min 0.99936, last 0.99984;
  v07 min 0.99972, last 0.99993.
- Text-only requests are unchanged: the 25-row stage-6 e2e test passes 25/25 and all 100 output
  tensors (probs, logits, final hidden, 64-layer trace per row) are bit identical to a run of the
  code before 12B.

### 3D mRoPE (`tt/rope.py`)

- `get_rope_index` is a host port of HF `Qwen3_5Model.get_rope_index` (text runs: `current +
  arange` on all 3 streams; an image: T = current, H = current + row, W = current + col over the
  merged grid, then `current += max(h, w) / 2`). It runs during input prep, like tokenization. It
  equals the golden `position_ids` bit for bit on all 8 rows.
- `PplxRequestRotary` builds the request's cos/sin from those ids: frequency j takes stream H
  when `j % 3 == 1, j < 33`, W when `j % 3 == 2, j < 30`, else T (sections [11, 11, 10]); same pair
  layout and 192 pass-through dims as `PplxRotary`. The BF16 tables equal HF
  `Qwen3_5TextRotaryEmbedding` on the golden ids bit for bit (8/8 rows).
- `start_pos` stays a cache / chunk index. The table is indexed by token index and holds that
  token's rope angle, so `rotary(start, length)`, `decoder.py` and `attention.py` are unchanged.
  After the v02 image, the last token (index 356) has rope position 140.
- Layer 3 (full attention) fed the HF layer-2 output of v02 with the request tables: PCC 0.999997
  vs the HF layer-3 output (bar 0.995). The layer's own update (output minus input) has PCC
  0.99983. With the 1D text tables instead (control) it drops to 0.99936.
- Padding rows of the bucket get text-continuation positions. They come after every real token,
  so they cannot reach a real row (causal).

### Splice (`PplxDeciderModel.splice`)

HF: `inputs_embeds.masked_scatter(input_ids == image_token_id, image_features)`. TT: the host
writes the splice index during input prep (`s` for a text or padding token, `bucket + k` for the
k-th image token). On device, `ttnn.embedding(tokens)` in ROW_MAJOR gives the `[bucket, 5120]`
text rows; the tower features are untilized and appended with `ttnn.concat` (dim 0); one
`ttnn.embedding(splice_index, table)` gathers the spliced `[1, bucket, 5120]` TILE tensor. Both
steps copy rows, so no value changes. The table is transient: (1024 + 256) x 5120 x 2 B = 12.5 MiB
at the 1024 bucket, 85 MiB at 8192.

### Run

```bash
pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_mrope.py -q -s          # mRoPE: CPU bit-exact checks + layer 3 on v02
pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_image_model.py -q -s    # splice, 8-row gate, predict, padding, audit
pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_model.py -q -s          # text regression (25 rows)
pytest models/demos/pplx_decider_v1_27b/tests/perf/test_image_perf.py -q -s    # latency, ~16 min (idle gaps)
python models/demos/pplx_decider_v1_27b/demo/demo.py --image photo.png
```

The image golden is `$PPLX_DECIDER_IMAGE_GOLDEN` (default `artifacts/pplx_decider/goldens/vision/e2e`);
results go to `$PPLX_DECIDER_STAGE12B_DIR` (default `artifacts/pplx_decider/stage12b/image_e2e`).

### API

```python
decider = TTDecider.from_pretrained(device)                      # text model + vision tower
decider.predict(state, question, images=["photo.png"])           # same call as the app's Decider.predict

model = PplxDeciderModel.from_snapshot(device, vision=True)
tokens, last_index, images = model.prepare_images(processor_output)   # host input prep + uploads
probs, logits, _ = model(tokens, last_index, count, images=images)    # vision -> splice -> layers -> head
```

`images` accepts what the app's `open_image` accepts: a path, a `data:image/...` URL or a PIL image.
The processor is the app's: `size = {"shortest_edge": 65536, "longest_edge": 262144}`, chat template
with `enable_thinking=False`. A request without images runs the text-only path unchanged.

### Runtime integrity

- Fallback audit (`count_host_calls`) around a warmed image forward (vision tower, splice, 64
  layers, head) plus sync: 0 host calls for v02 (936 patches, padded) and v04 (1024 patches).
- The same image request twice: probs, logits and final hidden bit identical.
- Watcher (`TT_METAL_WATCHER=10`) on an image request (v02): clean, no assert or sanitize message.
- The same image request (v04) padded to the 8192 bucket instead of 1024 gives bit-identical
  probabilities. With the text model, the vision tower and that 8192 forward's outputs resident,
  3.55 GiB of DRAM is free.

### Perf

Warmed image requests at the 1024 bucket take 387-410 ms (median of 5, each after 10 s idle),
against 367 ms for a text-only request of the same bucket. The extra 20-44 ms is the vision tower
(15.6-29.1 ms), the processor on host (3.6-10.9 ms) and input prep with uploads (3.4-6.2 ms); the
splice is 0.45 ms and the 64-layer text forward is unchanged (364 ms). Per-row table:
`work_log.md` (stage 12B, perf).

## Vision tower (stage 12A)

Status:

- Every module matches HF with real weights on all 8 golden images (256 to 1024 patches), teacher
  forced: patch embed PCC >= 0.999997, each of the 27 blocks >= 0.999976 and the merger >= 0.999996
  (bar 0.995).
- The whole tower, fed only the pixels, gives image-feature PCC 0.998654 to 0.999851 (bar 0.99).
- v02 has 936 patches in the 1024 bucket. It passes at 0.999758, which shows that the padded keys
  are masked. Without the mask, the same image drops to 0.994969.
- The forward makes no host calls, two runs of the same image are bit identical, and a watcher run
  is clean.
- Warmed latency is 15.5 ms at the 256 bucket and 28.6 ms at the 1024 bucket. The vision weights
  take 0.964 GiB of DRAM. With the full 64-layer text model (C0) also loaded, 3.57 GiB of DRAM is
  still free.

Numbers, commands and the full PCC table: [`work_log.md`](work_log.md).

## What runs where

| step | HF (`modeling_qwen3_5.py`) | TT (`tt/vision/`) |
|---|---|---|
| patch embed | `Conv3d(3, 1152, k = s = (2, 16, 16))` on `pixel_values.view(-1, 3, 2, 16, 16)` | one `ttnn.linear` `[S, 1536] x [1536, 1152] + bias` (`patch_embed.py`); kernel == stride, so the conv is exactly this matmul |
| pos embed | bilinear 4-corner interpolation of the 48 x 48 table, fp32, `.to(bf16)`, add | interpolation on host when the image is prepared, using HF's helper and op order (`inputs.py`, bit exact vs the golden); BF16 `ttnn.add` on device |
| 2D rotary | `(row, col) * inv_freq` -> `[n, 36]`, `cat(x, x)`, neox `rotate_half` on all 72 dims, q/k math in fp32 | cos/sin computed on host in fp32, then BF16 on device; q/k head dims rope-permuted in the weights (pair `(j, j+36)` -> `(2j, 2j+1)`), one `rotary_embedding_llama` per tensor |
| attention | `qkv` Linear, 16 heads x 72, bidirectional SDPA per image, scale 72^-0.5, `proj` | heads padded 72 -> 96 (zero weight columns and bias, cos 1 / sin 0, zero `proj` rows); `nlp_create_qkv_heads`; `scaled_dot_product_attention(is_causal=False, cu_window_seqlens=[0, n, S], output_concat_heads=True)` (`attention.py`) |
| MLP | `fc1 1152 -> 4304`, `gelu_pytorch_tanh`, `fc2` | intermediate padded 4304 -> 4320 with zeros; fused `activation="gelu_tanh"` (`mlp.py`) |
| norms | `LayerNorm(1152, eps 1e-6)` | `ttnn.layer_norm`, HiFi4, fp32 accumulation (`layernorm.py`) |
| merger | `LayerNorm` per patch, `view(-1, 4608)`, `fc1` + `nn.GELU()` (exact erf), `fc2 -> 5120` | `ttnn.reshape` `[S, 1152] -> [S/4, 4608]`, fused `activation="gelu"` (erf), output sliced to `n/4` rows (`merger.py`, `tower.py`) |

GELU variants were checked on device (`test_gelu_variant`, fp32 output on an identity matmul over
[-5, 5]). `"gelu_tanh"` differs from torch's tanh GELU by at most 4.8e-7 and from erf GELU by
4.7e-4. `"gelu"` differs from erf GELU by at most 9.5e-7.

## Precision

The vision tower is all BF16 (person decision): BF16 weights and activations, HiFi4 matmuls with
fp32 accumulation, and no BFP8 or BFP4. The policy is `VisionPrecisionPolicy` in
`tt/optimizations.py`, and the constructor rejects any non-BF16 weight dtype. The code adds no fp32
casts of its own. The host tables are fp32 only where HF computes them in fp32: the pos-embed
interpolation and the cos/sin values. Two differences from HF remain:

- HF applies the rotary in fp32. TT applies it in BF16 with fp32 accumulation.
- Bias and GELU run in the matmul epilogue on the fp32 accumulator. HF rounds the `fc1` output to
  BF16 before the GELU.

## Patch buckets

The app's pixel budget (65536 to 262144 px) gives 256 to 1024 patches per image. The tower pads
the patch rows to the next bucket of 256, 512, 768 or 1024 (`config.VISION_BUCKETS`). The buckets
are multiples of the 128-row SDPA chunk and of 4 x 32, so the 2x2 merger output stays tile aligned.
`cu_window_seqlens = [0, n, S]` makes SDPA block diagonal. As a result, the n real patches never
see the padded keys, and the padded rows only see each other. The output is sliced to `n/4` rows on
device.

## API

```python
from models.demos.pplx_decider_v1_27b.tt.vision.tower import PplxVisionTower

tower = PplxVisionTower.from_snapshot(device)          # reads only the 333 visual.* tensors, strict key check
inputs = tower.prepare_inputs(pixel_values, grid_thw)  # host: pad to bucket, pos embed, cos/sin, mask -> device
features = tower(inputs)                               # device [1, 1, n/4, 5120] BF16; no host calls
```

`prepare_inputs` plays the role that `PplxDeciderModel.upload_tokens` plays for text. It takes one
image (`grid_thw = (1, h, w)`) and the processor's `pixel_values` `[n, 1536]`. Each module follows
the TTv2 pattern: a `*Config` dataclass, `from_config`, `LazyWeight` bundles from `tt/vision/weights.py`,
and opts from `VisionOptimizations` (roles `vision_patch`, `vision_qkv`, `vision_proj`, `vision_fc1`,
`vision_fc2`, `merger_fc1`, `merger_fc2`).

## Run

```bash
pytest models/demos/pplx_decider_v1_27b/tests/vision/test_vision_inputs.py -q   # CPU: host tables, adapter, strict keys
pytest models/demos/pplx_decider_v1_27b/tests/vision/test_vision_pcc.py -q      # PCC, about 20 s
pytest models/demos/pplx_decider_v1_27b/tests/vision/test_vision_runtime.py -q  # fallback audit, determinism
pytest models/demos/pplx_decider_v1_27b/tests/vision/test_vision_perf.py -q -s  # latency, DRAM (+ full text model)
python models/demos/pplx_decider_v1_27b/tests/vision/vision_pcc_report.py       # PCC table from the log
```

Goldens come from `reference/hf_vision_reference.py`: `$PPLX_DECIDER_VISION_GOLDEN`, by default
`/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision`. The PCC log is
`$PPLX_DECIDER_VISION_PCC_LOG`, by default `.../stage12a/pcc_vision.jsonl`.

## Limits and follow-ups

- The tower takes one still image per call (`t = 1`). It does not handle video frames or several
  images in one sequence. The app sends one image per request.
- The forward is eager. At the 256 bucket, latency is probably dominated by host dispatch (inferred:
  about 300 ops). Trace capture per bucket is the obvious next step and is not done here.
- Stage 12B: several images in one prompt are handled by the input prep and the splice (features
  appended in prompt order), but only one image per prompt is tested. Video is rejected.
