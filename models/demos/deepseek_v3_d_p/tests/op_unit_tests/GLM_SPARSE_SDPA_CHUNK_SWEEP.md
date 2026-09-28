# GLM-5.2 sparse attention (`ttnn.transformer.sparse_sdpa`): chunk and KV-prefix sweep on 8x4

Baseline: `main` @ `db596ae144d`, Blackhole Galaxy. The test is `test_glm_sparse_sdpa_chunk_sweep.py`.

## Setup (matches `ttMLA._sparse_mla` after the head->sequence all-to-all)

- Per device: q `[1, 64, chunk/32, 576]` bf16 RM; indices `[1, 1, chunk/32, 2048]` uint32 RM (logical positions
  with a 0xFFFFFFFF tail); replicated KVPE buffer `[1, 1, prefix + chunk, 576]` bf16 RM (GLM-5.2's BF16_RM format)
  in block-cyclic order, tp-sharded (32 stripes x chunk/32).
- v_dim = 512, scale = 256^-0.5, `k_chunk_size` = 128 (production; 64 and 256 also swept), default kernel
  config (HiFi2, approx), full 12x10 grid = 120 cores. No sub-device.
- Work split: query rows (tokens) are spread over the 120 cores, and each core runs all 64 heads of its tokens.
  Each selected key row (1152 B) is gathered from DRAM by index, split across the reader and writer NoCs.
  There is no cross-core traffic.
- Rows model the slowest device (last stripe): row r has nv = min(pos + 1, 2048) keys. Indices are random
  distinct causal positions.
- Util = attention matmul FLOPs (2 x 64 heads x nv x (576 + 512)) / (120 cores x 2048 FLOP/cycle x 1.35 GHz x
  time). `KV GB/s` = selected rows x 1152 B / time, per chip.
- Compute-only: `SPARSE_SDPA_COMPUTE_ONLY=1` skips the Q reads, both halves of the indexed KV gather and the
  output writes. The 8 KB index-row read (it sets nv) and the CB handshakes stay.
- Timing is the realtime profiler over 10 trace replays, median; max over the 32 chips.

## Chunk sweep at ~50k prefix

| k_chunk | chunk | rows / device | core grid | cores | active cores | tokens / core | full us | compute-only us | DM cost | util full % | util compute-only % | KV GB/s | us per 1k tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 1k | 32 | 12x10 | 120 | 32 | 1 | 283 | 263 | +7% | 9.7 | 10.4 | 267 | 283 |
| 64 | 2k | 64 | 12x10 | 120 | 64 | 1 | 442 | 264 | +68% | 12.4 | 20.9 | 341 | 221 |
| 64 | 3k | 96 | 12x10 | 120 | 96 | 1 | 607 | 264 | +130% | 13.6 | 31.2 | 373 | 202 |
| 64 | 4k | 128 | 12x10 | 120 | 120 | 2 | 1032 | 527 | +96% | 10.7 | 20.9 | 293 | 258 |
| 64 | 5k | 160 | 12x10 | 120 | 120 | 2 | 1076 | 528 | +104% | 12.8 | 26.1 | 351 | 215 |
| 128 | 1k | 32 | 12x10 | 120 | 32 | 1 | 249 | 210 | +18% | 11.1 | 13.1 | 304 | 249 |
| 128 | 2k | 64 | 12x10 | 120 | 64 | 1 | 436 | 210 | +108% | 12.6 | 26.2 | 346 | 218 |
| 128 | 3k | 96 | 12x10 | 120 | 96 | 1 | 605 | 211 | +187% | 13.6 | 39.2 | 374 | 202 |
| 128 | 4k | 128 | 12x10 | 120 | 120 | 2 | 978 | 420 | +133% | 11.2 | 26.2 | 309 | 245 |
| 128 | 5k | 160 | 12x10 | 120 | 120 | 2 | 1058 | 421 | +152% | 13.0 | 32.7 | 357 | 212 |
| 256 | 1k | 32 | 12x10 | 120 | 32 | 1 | 257 | 187 | +37% | 10.7 | 14.7 | 294 | 257 |
| 256 | 2k | 64 | 12x10 | 120 | 64 | 1 | 440 | 188 | +135% | 12.5 | 29.3 | 343 | 220 |
| 256 | 3k | 96 | 12x10 | 120 | 96 | 1 | 606 | 188 | +222% | 13.6 | 43.9 | 374 | 202 |
| 256 | 4k | 128 | 12x10 | 120 | 120 | 2 | 962 | 375 | +157% | 11.4 | 29.4 | 314 | 240 |
| 256 | 5k | 160 | 12x10 | 120 | 120 | 2 | 1061 | 376 | +182% | 13.0 | 36.6 | 356 | 212 |

## KV-prefix sweep (k_chunk = 128)

### 1k chunk (32 rows / device): 12x10 = 120 cores, 32 active, 1 token(s) / core

| KV prefix | mean nv | core grid | cores | active cores | tokens / core | full us | compute-only us | DM cost | util full % | util compute-only % | KV GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 1008 | 12x10 | 120 | 32 | 1 | 139 | 107 | +30% | 9.8 | 12.7 | 268 |
| 2k | 2048 | 12x10 | 120 | 32 | 1 | 246 | 210 | +17% | 11.2 | 13.1 | 307 |
| 4k | 2048 | 12x10 | 120 | 32 | 1 | 246 | 210 | +17% | 11.2 | 13.1 | 306 |
| 8k | 2048 | 12x10 | 120 | 32 | 1 | 245 | 210 | +17% | 11.2 | 13.1 | 308 |
| 16k | 2048 | 12x10 | 120 | 32 | 1 | 246 | 210 | +17% | 11.2 | 13.1 | 307 |
| 32k | 2048 | 12x10 | 120 | 32 | 1 | 245 | 210 | +17% | 11.2 | 13.1 | 308 |
| 50k | 2048 | 12x10 | 120 | 32 | 1 | 249 | 210 | +18% | 11.1 | 13.1 | 304 |
| 64k | 2048 | 12x10 | 120 | 32 | 1 | 245 | 210 | +17% | 11.2 | 13.1 | 309 |
| 100k | 2048 | 12x10 | 120 | 32 | 1 | 247 | 210 | +18% | 11.1 | 13.1 | 305 |
| 128k | 2048 | 12x10 | 120 | 32 | 1 | 247 | 210 | +18% | 11.1 | 13.1 | 306 |
| 192k | 2048 | 12x10 | 120 | 32 | 1 | 247 | 210 | +17% | 11.2 | 13.1 | 306 |
| 256k | 2048 | 12x10 | 120 | 32 | 1 | 247 | 210 | +18% | 11.2 | 13.1 | 306 |

### 2k chunk (64 rows / device): 12x10 = 120 cores, 64 active, 1 token(s) / core

| KV prefix | mean nv | core grid | cores | active cores | tokens / core | full us | compute-only us | DM cost | util full % | util compute-only % | KV GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 2016 | 12x10 | 120 | 64 | 1 | 433 | 210 | +106% | 12.5 | 25.8 | 343 |
| 2k | 2048 | 12x10 | 120 | 64 | 1 | 438 | 210 | +108% | 12.6 | 26.2 | 345 |
| 4k | 2048 | 12x10 | 120 | 64 | 1 | 441 | 210 | +110% | 12.5 | 26.2 | 343 |
| 8k | 2048 | 12x10 | 120 | 64 | 1 | 442 | 210 | +110% | 12.5 | 26.2 | 342 |
| 16k | 2048 | 12x10 | 120 | 64 | 1 | 439 | 210 | +109% | 12.5 | 26.2 | 344 |
| 32k | 2048 | 12x10 | 120 | 64 | 1 | 438 | 210 | +108% | 12.6 | 26.2 | 345 |
| 50k | 2048 | 12x10 | 120 | 64 | 1 | 436 | 210 | +108% | 12.6 | 26.2 | 346 |
| 64k | 2048 | 12x10 | 120 | 64 | 1 | 440 | 210 | +109% | 12.5 | 26.2 | 343 |
| 100k | 2048 | 12x10 | 120 | 64 | 1 | 436 | 210 | +108% | 12.6 | 26.2 | 346 |
| 128k | 2048 | 12x10 | 120 | 64 | 1 | 442 | 210 | +111% | 12.4 | 26.2 | 341 |
| 192k | 2048 | 12x10 | 120 | 64 | 1 | 440 | 210 | +110% | 12.5 | 26.2 | 343 |
| 256k | 2048 | 12x10 | 120 | 64 | 1 | 442 | 210 | +110% | 12.4 | 26.2 | 342 |

### 3k chunk (96 rows / device): 12x10 = 120 cores, 96 active, 1 token(s) / core

| KV prefix | mean nv | core grid | cores | active cores | tokens / core | full us | compute-only us | DM cost | util full % | util compute-only % | KV GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 2048 | 12x10 | 120 | 96 | 1 | 609 | 211 | +189% | 13.6 | 39.2 | 372 |
| 3k | 2048 | 12x10 | 120 | 96 | 1 | 606 | 211 | +188% | 13.6 | 39.2 | 374 |
| 9k | 2048 | 12x10 | 120 | 96 | 1 | 612 | 211 | +191% | 13.5 | 39.2 | 370 |
| 15k | 2048 | 12x10 | 120 | 96 | 1 | 611 | 211 | +190% | 13.5 | 39.2 | 371 |
| 33k | 2048 | 12x10 | 120 | 96 | 1 | 609 | 211 | +189% | 13.6 | 39.2 | 372 |
| 51k | 2048 | 12x10 | 120 | 96 | 1 | 605 | 211 | +187% | 13.6 | 39.2 | 374 |
| 63k | 2048 | 12x10 | 120 | 96 | 1 | 608 | 211 | +189% | 13.6 | 39.2 | 372 |
| 99k | 2048 | 12x10 | 120 | 96 | 1 | 610 | 211 | +190% | 13.5 | 39.2 | 371 |
| 129k | 2048 | 12x10 | 120 | 96 | 1 | 618 | 211 | +193% | 13.4 | 39.2 | 367 |
| 192k | 2048 | 12x10 | 120 | 96 | 1 | 608 | 211 | +189% | 13.6 | 39.2 | 373 |
| 255k | 2048 | 12x10 | 120 | 96 | 1 | 618 | 211 | +193% | 13.4 | 39.2 | 367 |

### 4k chunk (128 rows / device): 12x10 = 120 cores, 120 active, 2 token(s) / core

| KV prefix | mean nv | core grid | cores | active cores | tokens / core | full us | compute-only us | DM cost | util full % | util compute-only % | KV GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 2048 | 12x10 | 120 | 120 | 2 | 976 | 420 | +133% | 11.3 | 26.2 | 309 |
| 4k | 2048 | 12x10 | 120 | 120 | 2 | 973 | 420 | +132% | 11.3 | 26.2 | 310 |
| 8k | 2048 | 12x10 | 120 | 120 | 2 | 973 | 420 | +132% | 11.3 | 26.2 | 310 |
| 16k | 2048 | 12x10 | 120 | 120 | 2 | 974 | 420 | +132% | 11.3 | 26.2 | 310 |
| 32k | 2048 | 12x10 | 120 | 120 | 2 | 973 | 420 | +132% | 11.3 | 26.2 | 310 |
| 48k | 2048 | 12x10 | 120 | 120 | 2 | 978 | 420 | +133% | 11.2 | 26.2 | 309 |
| 64k | 2048 | 12x10 | 120 | 120 | 2 | 975 | 420 | +132% | 11.3 | 26.2 | 310 |
| 100k | 2048 | 12x10 | 120 | 120 | 2 | 976 | 420 | +132% | 11.3 | 26.2 | 309 |
| 128k | 2048 | 12x10 | 120 | 120 | 2 | 975 | 420 | +132% | 11.3 | 26.2 | 310 |
| 192k | 2048 | 12x10 | 120 | 120 | 2 | 973 | 420 | +132% | 11.3 | 26.2 | 310 |
| 256k | 2048 | 12x10 | 120 | 120 | 2 | 971 | 420 | +131% | 11.3 | 26.2 | 311 |

### 5k chunk (160 rows / device): 12x10 = 120 cores, 120 active, 2 token(s) / core

| KV prefix | mean nv | core grid | cores | active cores | tokens / core | full us | compute-only us | DM cost | util full % | util compute-only % | KV GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 2048 | 12x10 | 120 | 120 | 2 | 1061 | 421 | +152% | 13.0 | 32.7 | 356 |
| 5k | 2048 | 12x10 | 120 | 120 | 2 | 1058 | 421 | +152% | 13.0 | 32.7 | 357 |
| 10k | 2048 | 12x10 | 120 | 120 | 2 | 1061 | 420 | +152% | 13.0 | 32.7 | 356 |
| 15k | 2048 | 12x10 | 120 | 120 | 2 | 1059 | 421 | +152% | 13.0 | 32.7 | 356 |
| 30k | 2048 | 12x10 | 120 | 120 | 2 | 1062 | 420 | +152% | 13.0 | 32.7 | 356 |
| 50k | 2048 | 12x10 | 120 | 120 | 2 | 1058 | 421 | +152% | 13.0 | 32.7 | 357 |
| 65k | 2048 | 12x10 | 120 | 120 | 2 | 1058 | 421 | +152% | 13.0 | 32.7 | 357 |
| 100k | 2048 | 12x10 | 120 | 120 | 2 | 1066 | 420 | +153% | 12.9 | 32.7 | 354 |
| 130k | 2048 | 12x10 | 120 | 120 | 2 | 1054 | 421 | +151% | 13.1 | 32.7 | 358 |
| 190k | 2048 | 12x10 | 120 | 120 | 2 | 1054 | 421 | +151% | 13.1 | 32.7 | 358 |
| 255k | 2048 | 12x10 | 120 | 120 | 2 | 1069 | 421 | +154% | 12.9 | 32.7 | 353 |

## Findings

- **sparse_sdpa is bound by the indexed KV gather, not by compute.** At 2k-5k the full time is 2-3x the
  compute-only time (+108-187% DM cost at k_chunk 128). The chip sustains ~300-375 GB/s of scattered 1152 B
  row reads, and full-mode math util is only 11-14%.
- **Because it is gather bound, time scales with tokens per chip, not tokens per core.** Full time at 50k
  (k_chunk 128): 1k 249, 2k 436, 3k 605, 4k 978, 5k 1058 us. us per 1k tokens: 243 / 213 / 197 / 239 / 207.
  The per-token cost is roughly flat from 2k to 5k, so a smaller chunk costs little extra per token.
- **Compute-only follows tokens per core.** It is 210 us per token per core (k_chunk 128): 1k-3k (1 token /
  core) all take 210 us, and 4k-5k (2 tokens / core) take 420 us. 4k is the odd one: 8 cores carry 2 tokens,
  so it pays the 2-token compute floor for only 128 rows (239 us per 1k tokens, the worst of 2k-5k).
- **1k is compute bound.** With only 32 of 120 cores busy, the gather has little in flight: full is 249 us vs
  210 us compute-only (+18%). That is the only chunk that sits near its compute floor.
- **Prefix does not matter** once every row has 2048 keys (prefix ≥ ~2k): time is flat within ±1% from 2k to
  256k. At a 0 prefix only the 1k chunk is cheaper (138 us), because its rows select fewer than 2048 keys.
- **k_chunk:** 256 cuts compute-only by ~11% (188 vs 210 us per token) and 64 adds ~25%. In full mode all three
  land within ~5%, because the gather dominates. 128 is fine; at 1k, 128 is the best (249 us vs 257 for 256 and
  283 for 64).
- **For the chunk-size question:** sparse_sdpa does not penalise 2k. Its per-token cost is within ~3% of 5k (213 vs 207 us per 1k tokens),
  it has no core-count cliff until 1k, and the lever for it is gather bandwidth (KV format / fp8 rows), not
  chunk size.
