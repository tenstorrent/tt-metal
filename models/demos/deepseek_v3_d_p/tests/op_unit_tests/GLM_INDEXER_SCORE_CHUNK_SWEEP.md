# GLM-5.2 indexer score (`ring_indexer_score_dsa`): chunk and KV-prefix sweep on 8x4

Baseline: sweep branch on `main` @ `db596ae144d`, Blackhole Galaxy, FABRIC_2D_TORUS_XY (STRICT_INIT). The test
is `test_glm_indexer_score_chunk_sweep.py`.

## Setup (matches `TtIndexer.score`, fused full-mesh route)

- `cluster_axis=None`: one 32-device snake ring gathers the deduped index-key cache while the score runs.
- Per device: q `[1, 32, chunk/32, 128]` bfp8, gate weights `[1, 1, chunk/32, 32]` bf16, and the index-key
  cache `[1, 1, T/32, 128]` bfp8, striped over sp*tp. The gathered-K scratch is replicated `[1, 1, T, 128]`.
- T = GLM-5.2's 1M context, rounded to whole chunks. The op sizes its K-band schedule from T.
- Scalar bounds: `chunk_start_idx = prefix`, `kv_len = prefix + chunk`. Starts are chunk-aligned.
- Program config is the production rule: `q_chunk = 64 if rows % 64 == 0 else 32`, `k_chunk = 320`,
  `head_group_size = 0`. A `q_chunk = 32` rerun for 2k and 4k is below.
- Core grid 12x10: the 4 fused all-gather workers take one column, leaving 110 score cores. "score cores" is the
  op's banded core count from its own perf model; it drops to 99 at 3k and 88 at 4k, because the q-tile count
  sets the group rows.
- Util = the op's LoFi ideal cycles (a mirror of the nightly `_ring_indexer_ideal_compute_cycles`) for the
  critical device / measured time.
- Compute-only: `INDEXER_SCORE_COMPUTE_ONLY=1` skips the Q/W/K reads, the Q/K mcasts on both sides, the fused
  all-gather gate and the logits writes, and stubs the fused all-gather kernels.
- Timing is the realtime profiler over 10 trace replays, median; max over the 32 chips.

## Chunk sweep at ~50k prefix

| chunk | rows / device | q_chunk | core grid | score cores | full us | compute-only us | DM cost | util full % | util compute-only % | us per 1k tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1k | 32 | 32 | 12x10 | 110 | 326 | 104 | +213% | 6.9 | 21.6 | 326 |
| 2k | 64 | 64 | 12x10 | 110 | 389 | 201 | +94% | 11.8 | 22.9 | 194 |
| 2k (q_chunk 32) | 64 | 32 | 12x10 | 110 | 319 | 181 | +76% | 14.4 | 25.4 | 160 |
| 3k | 96 | 32 | 12x10 | 99 | 315 | 184 | +71% | 25.2 | 43.0 | 105 |
| 4k | 128 | 64 | 12x10 | 88 | 434 | 357 | +21% | 26.4 | 32.1 | 108 |
| 4k (q_chunk 32) | 128 | 32 | 12x10 | 88 | 323 | 271 | +19% | 35.5 | 42.3 | 81 |
| 5k | 160 | 32 | 12x10 | 110 | 326 | 272 | +20% | 37.2 | 44.5 | 65 |

## q_chunk 64 vs 32 at 2k and 4k (the production rule picks 64)

| chunk | prefix | full us, q64 | full us, q32 | compute-only us, q64 | compute-only us, q32 |
|---|---|---|---|---|---|
| 2k | 0k | 320 | 197 | 201 | 105 |
| 2k | 16k | 348 | 228 | 199 | 104 |
| 2k | 50k | 389 | 319 | 201 | 181 |
| 2k | 100k | 542 | 396 | 356 | 267 |
| 2k | 256k | 806 | 688 | 526 | 438 |
| 4k | 0k | 280 | 182 | 202 | 109 |
| 4k | 16k | 302 | 212 | 200 | 108 |
| 4k | 48k | 434 | 323 | 357 | 271 |
| 4k | 100k | 562 | 516 | 530 | 443 |
| 4k | 256k | 981 | 923 | 873 | 871 |

## KV-prefix sweep (production program config)

### 1k chunk (32 rows / device, q_chunk 32): 12x10 grid, 110 score cores + 4 AG workers

| KV prefix | kv_len | core grid | score cores | ideal us | full us | compute-only us | DM cost | util full % | util compute-only % |
|---|---|---|---|---|---|---|---|---|---|
| 0k | 1024 | 12x10 | 110 | 0.4 | 239 | 104 | +129% | 0.2 | 0.4 |
| 2k | 3072 | 12x10 | 110 | 1.3 | 258 | 104 | +148% | 0.5 | 1.3 |
| 4k | 5120 | 12x10 | 110 | 2.2 | 254 | 104 | +145% | 0.9 | 2.1 |
| 8k | 9216 | 12x10 | 110 | 4.0 | 266 | 103 | +157% | 1.5 | 3.9 |
| 16k | 17408 | 12x10 | 110 | 7.5 | 278 | 103 | +168% | 2.7 | 7.2 |
| 32k | 33792 | 12x10 | 110 | 14.6 | 294 | 104 | +183% | 5.0 | 14.0 |
| 50k | 52224 | 12x10 | 110 | 22.5 | 326 | 104 | +213% | 6.9 | 21.6 |
| 64k | 66560 | 12x10 | 110 | 28.7 | 349 | 104 | +236% | 8.2 | 27.7 |
| 100k | 103424 | 12x10 | 110 | 44.6 | 432 | 180 | +140% | 10.3 | 24.8 |
| 128k | 132096 | 12x10 | 110 | 56.9 | 475 | 178 | +167% | 12.0 | 31.9 |
| 192k | 197632 | 12x10 | 110 | 85.2 | 584 | 179 | +226% | 14.6 | 47.6 |
| 256k | 263168 | 12x10 | 110 | 113.4 | 685 | 264 | +159% | 16.6 | 42.9 |

### 2k chunk (64 rows / device, q_chunk 64): 12x10 grid, 110 score cores + 4 AG workers

| KV prefix | kv_len | core grid | score cores | ideal us | full us | compute-only us | DM cost | util full % | util compute-only % |
|---|---|---|---|---|---|---|---|---|---|
| 0k | 2048 | 12x10 | 110 | 1.8 | 320 | 201 | +60% | 0.6 | 0.9 |
| 2k | 4096 | 12x10 | 110 | 3.5 | 327 | 200 | +63% | 1.1 | 1.8 |
| 4k | 6144 | 12x10 | 110 | 5.3 | 324 | 199 | +62% | 1.6 | 2.6 |
| 8k | 10240 | 12x10 | 110 | 8.8 | 337 | 198 | +70% | 2.6 | 4.5 |
| 16k | 18432 | 12x10 | 110 | 15.9 | 348 | 199 | +75% | 4.6 | 8.0 |
| 32k | 34816 | 12x10 | 110 | 30.0 | 365 | 200 | +83% | 8.2 | 15.0 |
| 50k | 53248 | 12x10 | 110 | 45.9 | 389 | 201 | +94% | 11.8 | 22.9 |
| 64k | 67584 | 12x10 | 110 | 58.2 | 407 | 199 | +104% | 14.3 | 29.2 |
| 100k | 104448 | 12x10 | 110 | 90.0 | 542 | 356 | +52% | 16.6 | 25.3 |
| 128k | 133120 | 12x10 | 110 | 114.7 | 579 | 353 | +64% | 19.8 | 32.5 |
| 192k | 198656 | 12x10 | 110 | 171.2 | 688 | 355 | +94% | 24.9 | 48.2 |
| 256k | 264192 | 12x10 | 110 | 227.7 | 806 | 526 | +53% | 28.3 | 43.2 |

### 3k chunk (96 rows / device, q_chunk 32): 12x10 grid, 99 score cores + 4 AG workers

| KV prefix | kv_len | core grid | score cores | ideal us | full us | compute-only us | DM cost | util full % | util compute-only % |
|---|---|---|---|---|---|---|---|---|---|
| 0k | 3072 | 12x10 | 99 | 4.4 | 190 | 108 | +76% | 2.3 | 4.0 |
| 3k | 6144 | 12x10 | 99 | 8.8 | 191 | 107 | +78% | 4.6 | 8.2 |
| 9k | 12288 | 12x10 | 99 | 17.6 | 198 | 108 | +84% | 8.9 | 16.4 |
| 15k | 18432 | 12x10 | 99 | 26.4 | 207 | 107 | +94% | 12.7 | 24.7 |
| 33k | 36864 | 12x10 | 99 | 52.9 | 285 | 184 | +55% | 18.6 | 28.7 |
| 51k | 55296 | 12x10 | 99 | 79.4 | 315 | 184 | +71% | 25.2 | 43.0 |
| 63k | 67584 | 12x10 | 99 | 97.0 | 344 | 270 | +27% | 28.2 | 35.9 |
| 99k | 104448 | 12x10 | 99 | 150.0 | 426 | 314 | +36% | 35.2 | 47.8 |
| 129k | 135168 | 12x10 | 99 | 194.1 | 518 | 357 | +45% | 37.5 | 54.4 |
| 192k | 199680 | 12x10 | 99 | 286.8 | 631 | 486 | +30% | 45.4 | 59.1 |
| 255k | 264192 | 12x10 | 99 | 379.5 | 730 | 615 | +19% | 52.0 | 61.8 |

### 4k chunk (128 rows / device, q_chunk 64): 12x10 grid, 88 score cores + 4 AG workers

| KV prefix | kv_len | core grid | score cores | ideal us | full us | compute-only us | DM cost | util full % | util compute-only % |
|---|---|---|---|---|---|---|---|---|---|
| 0k | 4096 | 12x10 | 88 | 8.7 | 280 | 202 | +38% | 3.1 | 4.3 |
| 4k | 8192 | 12x10 | 88 | 17.6 | 287 | 201 | +43% | 6.1 | 8.7 |
| 8k | 12288 | 12x10 | 88 | 26.4 | 278 | 202 | +38% | 9.5 | 13.0 |
| 16k | 20480 | 12x10 | 88 | 44.0 | 302 | 200 | +51% | 14.6 | 22.0 |
| 32k | 36864 | 12x10 | 88 | 79.3 | 332 | 201 | +65% | 23.9 | 39.4 |
| 48k | 53248 | 12x10 | 88 | 114.6 | 434 | 357 | +21% | 26.4 | 32.1 |
| 64k | 69632 | 12x10 | 88 | 149.9 | 463 | 356 | +30% | 32.4 | 42.1 |
| 100k | 106496 | 12x10 | 88 | 229.4 | 562 | 530 | +6% | 40.8 | 43.3 |
| 128k | 135168 | 12x10 | 88 | 291.2 | 584 | 530 | +10% | 49.9 | 55.0 |
| 192k | 200704 | 12x10 | 88 | 432.4 | 733 | 615 | +19% | 59.0 | 70.3 |
| 256k | 266240 | 12x10 | 88 | 573.6 | 981 | 873 | +12% | 58.5 | 65.7 |

### 5k chunk (160 rows / device, q_chunk 32): 12x10 grid, 110 score cores + 4 AG workers

| KV prefix | kv_len | core grid | score cores | ideal us | full us | compute-only us | DM cost | util full % | util compute-only % |
|---|---|---|---|---|---|---|---|---|---|
| 0k | 5120 | 12x10 | 110 | 10.9 | 184 | 110 | +68% | 5.9 | 9.9 |
| 5k | 10240 | 12x10 | 110 | 21.9 | 187 | 109 | +71% | 11.7 | 20.1 |
| 10k | 15360 | 12x10 | 110 | 33.0 | 189 | 110 | +72% | 17.4 | 29.9 |
| 15k | 20480 | 12x10 | 110 | 44.0 | 203 | 109 | +85% | 21.7 | 40.2 |
| 30k | 35840 | 12x10 | 110 | 77.1 | 268 | 186 | +44% | 28.7 | 41.4 |
| 50k | 56320 | 12x10 | 110 | 121.2 | 326 | 272 | +20% | 37.2 | 44.5 |
| 65k | 71680 | 12x10 | 110 | 154.3 | 384 | 315 | +22% | 40.2 | 49.0 |
| 100k | 107520 | 12x10 | 110 | 231.6 | 513 | 445 | +15% | 45.2 | 52.1 |
| 130k | 138240 | 12x10 | 110 | 297.8 | 584 | 488 | +20% | 51.0 | 61.0 |
| 190k | 199680 | 12x10 | 110 | 430.1 | 738 | 660 | +12% | 58.3 | 65.2 |
| 255k | 266240 | 12x10 | 110 | 573.6 | 936 | 875 | +7% | 61.3 | 65.6 |

## Findings

- **The indexer's time does not shrink with the chunk.** At a ~50k prefix it takes 315-326 us at 1k, 3k and 5k,
  and about 320 us at 2k and 4k with q_chunk 32. The cost is set by walking the K prefix (bands of 320 keys)
  across the score cores, and every chip does that walk whatever its query-row count. So per token it scales as
  1/chunk: at 50k, 318 us per 1k tokens at 1k, about 156 at 2k (q32), 105 at 3k, about 79 at 4k (q32) and 64 at 5k.
  **Of the four ops so far, this is the one that makes small chunks expensive.**
- **Compute-only time steps with the KV length, not with rows.** It stays flat at about 104-110 us up to about
  15k-65k (it depends on the chunk), then jumps about 80-90 us per step (e.g. 5k: 110, 186, 272, 315, 445 ...). Each
  step is one more K band per core in the banded schedule.
- **Data movement costs a lot at short prefixes.** DM is +60-250% below about 30k and +7-20% at 5k from about 50k up.
  1k is worst (+130-240%): there are too few query rows to hide the gather and the K reads behind.
- **The production q_chunk rule is a perf bug at 2k and 4k.** `qc = 64 if rows % 64 == 0 else 32` in `indexer.py`
  picks 64 there and costs 18-38% (2k) and 6-35% (4k) against q_chunk 32. Compute-only doubles (about 200 vs 104 us),
  because a 64-row q group does twice the work per K band at the same band parallelism.
- **Util at 50k:** 37% at 5k, 25% at 3k, 7% at 1k. Even the 5k production point reaches only 61% at a 256k prefix.
