# Vision tower on one p150a (stage 12A)

The pplx-decider vision tower is HF `Qwen3_5VisionModel`: patch embed, learned position embedding,
27 ViT blocks and a 2x2 patch merger. It turns one image into 5120-wide features, the rows that
`Qwen3_5Model` places at the `<|image_pad|>` tokens of the text. Stage 12A runs the tower on one
Blackhole p150a in BF16 and checks it against the HF BF16 golden. The text model is unchanged.
Splicing the features into the text and the 3D mRoPE belong to stage 12B.

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
- Stage 12B owns the splice into the text embeddings and the 3D mRoPE positions.
