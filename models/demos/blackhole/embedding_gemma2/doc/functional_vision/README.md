# E2: functional vision tower (stage report)

Scope: the EmbeddingGemma 2 vision path (HF `Gemma4VisionModel` + `embed_vision`) on one
Blackhole P150a: patch embedder with the learned 2D position table, 16 encoder layers, 3x3
position pooler and `embed_vision`, producing the soft tokens the text stream receives in place of
`<|image|>` / `<|video|>` placeholders. bf16 weights (the checkpoint is bf16, so weights are exact),
bf16 activations, fp32 pixel input and fp32 pooled features, HiFi4 + fp32 accumulation, DRAM
interleaved. Reference: the unmodified HF modules in fp32 on CPU (transformers 5.20.2 overlay),
computed live from real images and a real video. All numbers below are measured on sjc-snva-tp100
on 2026-10-27 unless marked otherwise.

## Architecture as implemented (checked against `gemma4/modeling_gemma4.py`, 5.20.2 overlay)

| item | HF (file:line) | TT |
|---|---|---|
| processor (host) | `Gemma4ImageProcessor`: aspect-preserving resize to <= 2520 patches (both sides multiples of 3 patches), rescale 1/255, no mean/std; `pixel_values [B,2520,768]` pre-patchified, `image_position_ids [B,2520,2]` (x, y), -1 = padding. Video: 1260 patches / frame | used as is (host); `VisionEncoder.prepare_inputs` consumes its outputs |
| patch embedder | `2*(px-0.5)` -> `Linear 768->768` (no bias) + `table[0][x] + table[1][y]`, zero on padding (`:585-627`) | `2*px-1` in fp32 on device, bf16-weight linear; two `ttnn.embedding` lookups into tables with an appended zero row (host maps padding to it) |
| mask | `create_bidirectional_mask` from `~padding` (`:1037`): key padding only | `key_padding_mask(keep)`: `(keep-1)*1e9` repeated to `[B,1,S,S]` |
| attention | q/k/v/o bias-free; q_norm, k_norm (weight), v_norm (no scale); 2D axial RoPE after the norm; SDPA scaling 1.0; no softcap (`:903-969`) | fused QKV linear, `nlp_create_qkv_heads` (12 x 64), `rms_norm`, `AxialRope`, SDPA `is_causal=False, scale=1.0, attn_mask` |
| axial RoPE | inv_freq 16 entries (`theta 100`, spatial_dim 32); cos/sin `[c(f_x), c(f_x), c(f_y), c(f_y)]`; channels 0-31 rotated by x, 32-63 by y, `rotate_half` inside each 32 block (`:713-769, 847-900`) | host builds per-patch cos/sin `[B,1,S,64]` from the positions (padding keeps position -1, as HF); device `x*cos + (x@R)*sin`, R a constant signed block permutation (exact) |
| layer | sandwich norms (input / post_attn / pre_ff / post_ff); no PLE, no layer_scalar (`:972-1013`) | same order |
| MLP | `down(gelu_tanh(gate x) * up x)`, 768->3072->768 (`:697-710`) | the E1 `MLP` module with vision widths |
| pooler | zero padding, average 3x3 blocks by position (`kernel_idx = x//3 + (max_x//3)*(y//3)`), x sqrt(768) in fp32, strip empty rows (`:630-694, 2039`); standardize off | one batched matmul with a host-built one-hot `[B,1,S_out,S]` membership matrix (fp32 output), then x sqrt(768)/9 in fp32; host strips rows >= n_valid/9 |
| embed_vision | RMSNorm(768, no scale, eps 1e-6) -> `Linear 768->512`, no bias (`modeling_embedding_gemma2.py:565-591`) | same; output bf16 |
| clipped linears | `use_clipped_linears: false`: plain Linear | loader folds `.linear.weight`; `VisionConfig.from_hf` rejects true |

## Module contracts

`S` is the physical patch bucket: 2560 for images (processor N = 2520) and 1280 for video frames
(N = 1260). Rows N..S-1 are padding like processor padding. `S_out = round_up(S/9, 32)` = 288 / 160.

| module (`tt/vision/`) | forward | in | out |
|---|---|---|---|
| `VisionPatchEmbedder` | `(pixel_values, pos_x, pos_y)` | `[B,1,S,768]` fp32 TILE; `[B,S]` uint32 ROW_MAJOR x2 | `[B,1,S,768]` bf16 |
| `AxialRope` | `(x, cos, sin)` | `[B,H,S,64]`; cos/sin `[B,1,S,64]` bf16 | `[B,H,S,64]` |
| `VisionAttention` | `(x, mask, cos, sin)` | `[B,1,S,768]`, mask `[B,1,S,S]` | `[B,1,S,768]` |
| `MLP` (E1 `tt/mlp.py`) | `(x)` | `[B,1,S,768]` | `[B,1,S,768]` |
| `VisionEncoderLayer` | `(x, mask, cos, sin)` | as above | `[B,1,S,768]` |
| `VisionPooler` | `(h, pool)` | `[B,1,S,768]`, pool `[B,1,S_out,S]` bf16 one-hot | `[B,1,S_out,768]` fp32 |
| `VisionEmbedder` | `(pooled)` | `[B,1,S_out,768]` fp32 | `[B,1,S_out,512]` bf16 |
| `VisionModel` | `(VisionDeviceInputs)`; `forward_hidden(...)` | 7 device tensors (below) | `(soft [B,1,S_out,512] bf16, pooled [B,1,S_out,768] fp32)`; `forward_hidden` -> `[B,1,S,768]` |
| `VisionEncoder` (host) | `encode(pixel_values, position_ids, traced=)`; `capture_all(shapes)` | torch processor outputs | list of `[n_b, 512]` per image / frame |

`VisionDeviceInputs` (all built on the host by `VisionEncoder.prepare_inputs`, uploaded before the
forward, refreshed with `copy_host_to_device_tensor` before a trace replay): `pixel_values`
`[B,1,S,768]` fp32, `pos_x`/`pos_y` `[B,S]` uint32 (padding -> row 10240, zero), `cos`/`sin`
`[B,1,S,64]` bf16, `keep` `[B,1,1,S]` bf16 (1 valid patch), `pool` `[B,1,S_out,S]` bf16.
Soft tokens of image b are rows `0..n_b-1` with `n_b = valid patches / 9` (`VisionHostInputs.num_soft_tokens`):
the processor's valid blocks form a prefix (`pooling_matrix` asserts it).
Every `from_state_dict` checks tensor shapes and writes the ttnn weight cache under
`~/.cache/ttnn/models/embedding_gemma2/<snapshot>/vision.*`.

## Inputs used (real data, all from the tt-metal tree)

| case | file | size | valid patches / 2520 (or 1260) | soft tokens |
|---|---|---|---|---|
| landscape | `models/sample_data/house_in_field_1080p.jpg` | 1600x900 | 2376 | 264 |
| portrait | `models/demos/vision/classification/resnet50/ttnn_resnet/demo/images/ILSVRC2012_val_00040007_n03877472.JPEG` | 332x500 | 2340 | 260 |
| square | `models/experimental/retinanet/resources/dog_800x800.jpg` | 800x800 | 2304 | 256 |
| small | `models/sample_data/iam_ocr_image.jpg` (text strip) | 248x76 | 2430 | 270 |
| tiny | `models/sample_data/torchvision_mnist_digit_7.jpg` (grayscale) | 28x28 | 2304 | 256 |
| b2 | square + small in one batch | | 2304, 2430 | 256, 270 |
| video8 | `tech_reports/ttnn/images/ttnn-dashboard.mp4`, 8 frames at 1 fps (cv2), HF video processor | 3456x1954 | 1080 / 1260 per frame (45 x 24) | 120 per frame |

Every case has padding patches (position -1), so the padding paths (zero position embedding, key
mask, pooler zeroing, output stripping) run in every test. The processor upsamples small images
to the patch budget. The soft-token counts are below the 280 / 140 maxima because of aspect-ratio
padding. The overlay's video processor accepts decoded frames (numpy), so no video decoder
dependency is needed beyond cv2.

## PCC vs HF (bar 0.995 per layer kind; soft tokens >= 0.99 per image)

Source: `pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision` (run 1: 66 PCC tests +
14 trace/audit/perf tests; metrics in `tt-ops/runs/embeddinggemma-2/e2/pcc_run1.jsonl`). Each
module gets the HF input of that module (hooks on the real HF modules); PCC over valid patches /
kept soft tokens only.

| layer kind | tests | min PCC | worst case |
|---|---|---|---|
| patch embedder (+ position table), valid patches | 7 | 0.999998 | small |
| patch embedder, padding rows (zero position embedding) | 7 | 0.999999 | landscape |
| attention (layers 0, 7, 15) | 9 | 0.999973 | video8, layer 15 |
| MLP (layers 0, 7, 15) | 9 | 0.999994 | video8, layer 15 |
| encoder layer 0, every case | 7 | 0.999993 | |
| all 16 layers, teacher forced (landscape, b2, video8) | 48 | 0.999958 | video8, layer 15 |
| pooler (incl. HF kept-row count match) | 15 images | 1.000000 | |
| embed_vision | 15 images | 0.999998 | video8 frame 5 |
| standalone reference vs `EmbeddingGemma2Model.get_image_features` / `get_video_features` | 3 | max abs diff 0.0 | |

End to end, free running (`VisionEncoder.encode`, eager; processor output -> soft tokens):

| case | encoder-out PCC | soft-token PCC (TT vs HF fp32) | per-token cosine min / mean | HF bf16 vs HF fp32: soft PCC / cosine min |
|---|---|---|---|---|
| landscape | 0.997357 | **0.998782** | 0.9270 / 0.99858 | 0.998823 / 0.9413 |
| portrait | 0.998206 | **0.999081** | 0.9429 / 0.99880 | 0.998427 / 0.8662 |
| square | 0.999249 | **0.999484** | 0.9794 / 0.99936 | 0.999217 / 0.9844 |
| small | 0.999269 | **0.999725** | 0.9959 / 0.99975 | 0.999838 / 0.9988 |
| tiny | 0.997529 | **0.999137** | 0.9889 / 0.99895 | 0.999604 / 0.9945 |
| b2 (square, small) | 0.999260 | **0.999484 / 0.999725** | same as single | same as single |
| video8 (8 frames) | 0.997938 | **0.997739 - 0.999035** | min 0.9272 (frame 4) | 0.997011 - 0.999265 / min 0.8148 (frame 4) |

Analysis of the gap between per-layer (>= 0.99996) and end-to-end (0.9977-0.9997) PCC, measured
with a free-running per-layer probe on the landscape image (scratch script, see `work_log.md`):
the free-running PCC stays >= 0.99999 through layer 10, then drops to 0.99942 (layer 14) and
0.99736 (layer 15). A few patches have hidden states that grow sharply in the last layers. The
per-patch figures from that probe are not repeated here because no committed log or test
reproduces them.

Comparison with HF itself run in bf16 (the dtype the checkpoint ships in), both against HF fp32,
from the committed `e2/final_results.jsonl` (`test_vision_model.py::test_soft_tokens`):

- Soft-token PCC: TT 0.9977-0.9997, HF-bf16 0.9970-0.9998. TT is higher on portrait, square and
  3 of 8 video frames, and lower on the others.
- Worst per-token cosine: TT is **lower** than HF-bf16 on landscape (0.92698 vs 0.94131), square
  (0.97943 vs 0.98440), small (0.99593 vs 0.99878), tiny (0.98892 vs 0.99448) and 5 of 8 video
  frames (0, 1, 3, 6, 7). TT is higher on portrait (0.94289 vs 0.86620) and 3 of 8 frames (2, 4,
  5; frame 4: 0.92725 vs 0.81484). So TT's worst token is not uniformly as good as HF-bf16's; it
  is worse on most cases, by up to 0.0146 (landscape) in cosine.

A fp32 residual stream was tried (scratch, not kept): landscape soft PCC 0.99878 ->
0.99963, but portrait 0.99908 -> 0.99910 and video frames mixed (0.99833-0.99923). So the residual
adds are not the main error source, and the bf16 path was kept. Precision levers belong to E5/E6.

## Trace, determinism, fallback audit, watcher

| check | command | result |
|---|---|---|
| trace replay == eager | `test_vision_trace.py::test_trace_matches_eager`: traces (1,2560), (2,2560), (8,1280) captured once; 5 single images replay the same (1,2560) trace, b2 and video8 their own | bit-identical soft tokens (max abs diff 0.0) for all 7 cases; replay soft PCC vs HF >= 0.9977 |
| trace allocation safety | trace + audit + perf files with `TT_METAL_TRACE_ALLOC_TRACKING=1` | 14 passed, no "still alive" report (`e2/logs/trace_run1.log`) |
| determinism | `test_vision_trace.py::test_vision_determinism` (b2, video8) | 5 replays + 2 eager runs bit-identical |
| fallback audit | `test_vision_fallback_audit.py` (b2) | 0 torch calls in `VisionModel.forward` (TorchFunctionMode), no ttnn host-transfer entry point called, `throw_exception_on_fallback=True` did not raise; Tracy report of one forward: **448 device ops, 0 host ops** |
| watcher | `TT_METAL_WATCHER=10 pytest tests/pcc/vision -k "not perf and not reference_matches"` (includes the trace tests) | 73 passed (146 s); `watcher.log` (986 lines) has no line matching `error|assert|exception|overflow|sanitiz|fault` |

Static audit: in `tt/vision/`, `torch` is used only in setup (`from_state_dict`, `AxialRope.build`)
and in the host side of `VisionEncoder` (`prepare_inputs`, `read_rows`, `_dummy_host`) plus the
host helpers `axial_cos_sin` and `pooling_matrix` that `prepare_inputs` calls.

## Performance (functional, unoptimized; traced, warmed, 20 replays)

`pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision/test_vision_perf.py -k traced -s`
(`device` = back-to-back `execute_trace` + one sync; `e2e` = input copy + replay + soft-token
readback; host prep = `prepare_inputs`, not part of e2e).

| batch x bucket | inputs | device ms | e2e ms | images/s (device) | host prep ms |
|---|---|---|---|---|---|
| 1 x 2560 | landscape | **55.60** | 56.38 | 18.0 | 3.8 |
| 4 x 2560 | landscape, portrait, square, small | **294.47** | 297.05 | 13.6 | 16.3 |
| 8 x 1280 | video8 frames | 199.61 | 201.92 | 40.1 frames/s | 11.3 |

Batch 4 costs 73.6 ms per image, more than batch 1 (55.6 ms). [guess] The per-image work is the
same, so the batched SDPA / matmul program configs (written for B = 1) scale badly. E5 should
profile b4 before choosing a batching strategy.

The b1 x 2560 tt-perf-report below is the representative op-level report for this stage. Batch 4
(and batch 8 x 1280) has a traced device-time number only (b4: 294.41 ms in the final run,
`e2/final_results.jsonl`); no tt-perf-report was taken for it.

tt-perf-report for one warmed eager forward, b1 x 2560 (landscape), signposts
`PERF_VISION`/`PERF_VISION_END` (`tracy/b1_s2560/forward_perf_report.{txt,csv}`, stacked
`forward_perf_report_stacked.csv`; `Device Time` in microseconds): 448 device ops, 0 host ops,
55,493 us device time + 361 us op-to-op gaps (matches the 55.6 ms replay).

| op | device us | share | count |
|---|---|---|---|
| SDPA (2.0 ms / layer, 64 cores) | 32,035 | 57.7 % | 16 |
| Matmul (QKV, RoPE R x2, O, gate, up, down per layer; patch, pool, embed) | 11,967 | 21.6 % | 115 |
| eltwise binary (residual adds, RoPE mul/add, input affine) | 5,371 | 9.7 % | 151 |
| RMSNorm (LayerNorm op) | 2,999 | 5.4 % | 113 |
| gelu (unary) | 1,443 | 2.6 % | 16 |
| create / concat heads | 1,241 | 2.2 % | 32 |
| mask repeat, tilize, embedding lookups, untilize | 435 | 0.8 % | 5 |

E5 levers [guess]: SDPA dominates. Candidates are the dense `[B,1,S,S]` mask read (the test
inputs have 90-216 processor padding patches plus 40 (images) or 20 (frames) bucket rows, all
at the end of the sequence (checked: valid patches are a prefix in every case), so a block-map mask or a mask-free path for unpadded chunks could skip it), larger grids than 8x8, and lower fidelity. Others: fuse the per-head
q/k norm and RoPE (5 ops per tensor today), bfp8 weights, and sharded layouts.

## Context contract (vision)

`doc/context_contract.json` -> `vision`. The model's input contract is fixed by the processor:
at most 280 soft tokens per image (2520 patches) and 140 per video frame (1260 patches), at most
32 frames per video at 1 fps. Both buckets are supported and tested at their maximum patch count.
There is no reduction. Position table size 10240 bounds patch coordinates. The processor's
largest side is about 90 patches (small: 90 x 27), well inside it; `prepare_inputs` rejects
larger coordinates.

## Out of scope (decoder-only items of the bring-up skill)

The vision tower is a bidirectional encoder that runs once per image. It has no KV cache,
decode step, page table, current-position tensor, sampling or chunked prefill. None was built.
The "context length" axis is the patch count, fixed by the processor (see above).

## Limitations

- Dense additive key-padding mask `[B,1,S,S]` (13 MB per image at S = 2560), rebuilt per forward
  on device. It is the main SDPA cost.
- Images always run at the 2560 bucket: an image with fewer soft tokens via a smaller
  `max_soft_tokens` (70 / 140) would still run 2560 rows unless it fits the 1280 bucket. Only the
  processor defaults (280 image / 140 video) were tested.
- Batch > 1 scales worse than linearly (see perf).
- `capture_all` must list every (batch, bucket) before the first capture (same rule as the text tower).
- Video frames are separate batch rows. A 32-frame video runs as batch 32 x 1280. It was not
  tested beyond 8 frames; host memory and device DRAM were not probed at 32.
- The multimodal merge (`masked_scatter` of soft tokens into the text embeddings) is E4.

## Artifacts

- `tt-ops/runs/embeddinggemma-2/e2/logs/`: `pcc_run1.log`, `trace_run1.log`, `tracy_run1.log`,
  `watcher_run1.log`, `final_run.log`
- `tt-ops/runs/embeddinggemma-2/e2/`: `pcc_run1.jsonl`, `trace_run1.jsonl`,
  `watcher_results.jsonl`, `final_results.jsonl`, `watcher/generated/watcher/watcher.log`
- `doc/functional_vision/tracy/b1_s2560/`: tt-perf-report table, CSV and stacked CSV. The raw ops
  CSV stays at `generated/profiler/reports/2026_10_27_13_05_21/` (not committed).
