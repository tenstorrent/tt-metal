# Molmo2 CPU Fallbacks

CPU ops in the forward pass measured on T3K with a 30-frame video input
(S=2701, padded→4096, n_crops=30, N_pooled=2430).

## Baseline (before any fixes)

```
========================================================================
  Molmo2 Forward Pass Profile  — baseline
  S=2701 (padded→4096), n_crops=30, N_pooled=2430, N_valid=2430
========================================================================
  Stage                                        ms       %  Type
  Embedding                                   1.4    0.0%  TTNN
  ViT encode (25 blks, all crops)          1319.3   47.0%  TTNN
  Image pooling (cross-attn)                414.4   14.8%  CPU  ◄
  Image projector (SwiGLU)                   14.3    0.5%  TTNN
  Scatter inject (D2H+add+H2D)               57.9    2.1%  CPU  ◄
  Prefill mask build (D2H incl.)            280.7   10.0%  CPU  ◄
  RoPE setup (cached=near-zero)               0.3    0.0%  CPU  ◄
  36 decoder blocks                         644.6   23.0%  TTNN
  ln_f                                        0.6    0.0%  TTNN
  Last-token slice (D2H+H2D)                 62.0    2.2%  CPU  ◄
  lm_head                                     8.6    0.3%  TTNN
  TOTAL prefill                            2804.1  100.0%
    TTNN subtotal                          1988.8   70.9%
    CPU subtotal (incl. D2H/H2D)            815.3   29.1%  ◄
  Decode step (1 step, traced):       55.5 ms  (18.0 tok/s)
========================================================================
```

## After fixes (steady state, pass 2)

```
========================================================================
  Molmo2 Forward Pass Profile  — after prefill mask + image pooling fixes
  S=2701 (padded→4096), n_crops=30, N_pooled=2430, N_valid=2430
========================================================================
  Stage                                        ms       %  Type
  Embedding                                   1.4    0.1%  TTNN
  ViT encode (25 blks, all crops)          1217.0   43.2%  TTNN
  Image pooling (cross-attn)                708.0   25.1%  TTNN  ← was CPU
  Image projector (SwiGLU)                   14.3    0.5%  TTNN
  Scatter inject (D2H+add+H2D)               35.0    1.2%  CPU  ◄
  Prefill mask build (D2H incl.)            137.0    4.9%  CPU  ◄
  RoPE setup (cached=near-zero)               0.3    0.0%  CPU  ◄
  36 decoder blocks                         643.0   22.8%  TTNN
  ln_f                                        0.6    0.0%  TTNN
  Last-token slice (D2H+H2D)                 54.0    1.9%  CPU  ◄
  lm_head                                     8.3    0.3%  TTNN
  TOTAL prefill                            2819.0  100.0%
    TTNN subtotal                          2593.0   92.0%
    CPU subtotal (incl. D2H/H2D)            226.0    8.0%  ◄
  Decode step (1 step, traced):       56.0 ms  (17.8 tok/s)
========================================================================
```

CPU ops reduced from **815 ms (29%)** → **~226 ms (8%)**.

## Fix Status

### ✅ 1. Prefill mask — DONE (commit 6cb44c97)

**Was**: 281 ms CPU (`build_molmo2_prefill_mask` — full [S,S] mask built on CPU, uploaded)
**Now**: 137 ms — mask built on device with `ttnn.tril` + `ttnn.mul` + `ttnn.where`. Only
the tiny [B,S] `is_mm` vector is uploaded. Remaining 137 ms is the H2D upload of the
[1,1,S,S] ones tensor needed for `ttnn.tril`.
**PCC**: unchanged (all 8 tests pass at same values)

---

### ✅ 2. Image pooling — DONE (this PR)

**Was**: 414 ms CPU (`image_pooling_2d` reference — full PyTorch cross-attention)
**Now**: ~708 ms TTNN chunked (`_run_chunked_ttnn_pooling` + new `TtMolmo2ImagePooling2D`)

**Why TTNN is slower for 30-frame video**: the matmul shapes are small
(`[2430, 9→tile32, 192]` per device after TP=8) and don't saturate hardware.
CPU PyTorch is faster at this size. The TTNN path is the only viable option
for 384-frame video where the CPU path OOMs at the gather step
(`[1, 31104, 9, 2304]` = 1.2 GB).

**Design**:
- `TtMolmo2ImagePooling2D`: separate column-parallel wq/wk/wv (TP=8, each device 2 of 16
  heads), row-parallel wo + `ttnn.all_reduce`, manual matmul attention
- `_run_chunked_ttnn_pooling`: full ViT feature table uploaded as 2D `ttnn.embedding` lookup;
  pooling windows processed in chunks of `_POOL_CHUNK_WINDOWS=4096`
  (1 chunk for 30-frame video, 8 chunks for 384-frame)
- `uint32` indices — bfloat16 only accurate to ~256, max video index is 280k+
- `ROW_MAJOR` reshape before attn-mask build — avoids tile-padding artefacts
  when k_pool (9 or 4) is not a multiple of tile size 32

**PCC**: 0.999813 (was 0.999807 with CPU path — within noise)

---

### Remaining CPU ops

| # | Op | ms | Status |
|---|----|----|--------|
| 3 | Last-token slice (D2H+H2D) | ~54 | Open — replace with `ttnn.slice` on device |
| 4 | Scatter injection (D2H+add+H2D) | ~35 | Open — dense zero-pad + `ttnn.add` |
| 5 | RoPE decode slice | <1 | Skip — non-traced path only |

Fixing #3 (trivial) and #4 (medium) removes the remaining ~89 ms CPU overhead
and is a prerequisite for DP>1 batching.
