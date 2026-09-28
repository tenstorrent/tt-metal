# GLM-5.2 indexer top-k (`topk_large_indices`): chunk and KV-prefix sweep on 8x4

Baseline: `main` @ `db596ae144d`, Blackhole Galaxy. The test is `test_glm_topk_chunk_sweep.py`.

## Setup (matches `TtIndexer.select_local`, production trace path)

- Per-device logits are `[1, 1, chunk/32, T]` bf16 ROW_MAJOR DRAM. Query rows are striped 32 ways over the full mesh.
- k = 2048. The bounds come from metadata: `valid_length_tensor` = prefix, `valid_length_offset` = chunk,
  `valid_end_tensor` = prefix + chunk.
- `prod80`: the production overlap grid, 8x10 = 80 cores at (0,0)-(7,9), on sub-device 0 of an 80/40 split
  (the sparse-KV gather owns the other 40). `full120`: the whole 12x10 grid, shown for comparison.
- Work is split by **rows only**: whole rows per core, with no cross-core merge. Each row costs
  ceil(valid / 2048) bitonic sort+merge steps. The critical path is `rows_per_core = ceil(rows / cores)`.
- `ns/step` = median time / (rows_per_core x sort steps). This is the critical core's cost per 2048-element
  step, and it is flat when the op scales cleanly. The op has no ideal-perf model.
- Compute-only: `TOPK_COMPUTE_ONLY=1` skips the row DRAM reads, the L1 index reorder and the DRAM write.
  The CB handshakes and the metadata read are kept.
- Timing is the realtime profiler over 10 trace replays, median; max over the 32 chips.

## Chunk sweep at ~50k prefix

| grid | chunk | rows / device | core grid | cores | active cores | rows / core | core util | sort steps / row | full us | compute-only us | DM cost | ns/step full | ns/step compute-only | us per 1k tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prod80 | 1k | 32 | 8x10 | 80 | 32 | 1 | 0.40 | 26 | 228 | 225 | +1.5% | 8783 | 8652 | 228 |
| prod80 | 2k | 64 | 8x10 | 80 | 64 | 1 | 0.80 | 26 | 229 | 225 | +1.9% | 8812 | 8652 | 115 |
| prod80 | 3k | 96 | 8x10 | 80 | 80 | 2 | 0.60 | 27 | 468 | 464 | +0.9% | 8659 | 8585 | 156 |
| prod80 | 4k | 128 | 8x10 | 80 | 80 | 2 | 0.80 | 26 | 452 | 447 | +1.2% | 8691 | 8590 | 113 |
| prod80 | 5k | 160 | 8x10 | 80 | 80 | 2 | 1.00 | 28 | 487 | 481 | +1.3% | 8693 | 8582 | 97 |
| full120 | 1k | 32 | 12x10 | 120 | 32 | 1 | 0.27 | 26 | 229 | 226 | +1.5% | 8824 | 8690 | 229 |
| full120 | 2k | 64 | 12x10 | 120 | 64 | 1 | 0.53 | 26 | 230 | 226 | +1.9% | 8856 | 8691 | 115 |
| full120 | 3k | 96 | 12x10 | 120 | 96 | 1 | 0.80 | 27 | 239 | 234 | +2.0% | 8857 | 8684 | 80 |
| full120 | 4k | 128 | 12x10 | 120 | 120 | 2 | 0.53 | 26 | 452 | 448 | +1.0% | 8697 | 8608 | 113 |
| full120 | 5k | 160 | 12x10 | 120 | 120 | 2 | 0.67 | 28 | 487 | 482 | +1.1% | 8695 | 8601 | 97 |

## KV-prefix sweep (prod80 grid)

### 1k chunk (32 rows / device): 8x10 = 80 cores, 32 active, 1 row(s) / core

| KV prefix | valid length | core grid | cores | active cores | rows / core | sort steps / row | full us | compute-only us | DM cost | ns/step full | ns/step compute-only | logits read GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 1024 | 8x10 | 80 | 32 | 1 | 1 | 17.4 | 14.0 | +23.9% | 17363 | 14013 | 3.8 |
| 2k | 3072 | 8x10 | 80 | 32 | 1 | 2 | 24.6 | 21.2 | +15.9% | 12298 | 10609 | 8.0 |
| 4k | 5120 | 8x10 | 80 | 32 | 1 | 3 | 33.1 | 29.7 | +11.5% | 11037 | 9897 | 9.9 |
| 8k | 9216 | 8x10 | 80 | 32 | 1 | 5 | 50.1 | 46.7 | +7.4% | 10023 | 9336 | 11.8 |
| 16k | 17408 | 8x10 | 80 | 32 | 1 | 9 | 84.0 | 80.6 | +4.3% | 9337 | 8956 | 13.3 |
| 32k | 33792 | 8x10 | 80 | 32 | 1 | 17 | 152.0 | 148.5 | +2.3% | 8939 | 8737 | 14.2 |
| 50k | 52224 | 8x10 | 80 | 32 | 1 | 26 | 228.4 | 224.9 | +1.5% | 8783 | 8652 | 14.6 |
| 64k | 66560 | 8x10 | 80 | 32 | 1 | 33 | 291.9 | 288.4 | +1.2% | 8846 | 8740 | 14.6 |
| 100k | 103424 | 8x10 | 80 | 32 | 1 | 51 | 443.4 | 440.0 | +0.8% | 8695 | 8627 | 14.9 |
| 128k | 132096 | 8x10 | 80 | 32 | 1 | 65 | 566.4 | 563.0 | +0.6% | 8714 | 8661 | 14.9 |
| 192k | 197632 | 8x10 | 80 | 32 | 1 | 97 | 840.9 | 837.4 | +0.4% | 8669 | 8633 | 15.0 |
| 256k | 263168 | 8x10 | 80 | 32 | 1 | 129 | 1115.3 | 1111.9 | +0.3% | 8646 | 8619 | 15.1 |

### 2k chunk (64 rows / device): 8x10 = 80 cores, 64 active, 1 row(s) / core

| KV prefix | valid length | core grid | cores | active cores | rows / core | sort steps / row | full us | compute-only us | DM cost | ns/step full | ns/step compute-only | logits read GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 2048 | 8x10 | 80 | 64 | 1 | 1 | 18.1 | 14.0 | +29.0% | 18086 | 14023 | 14.5 |
| 2k | 4096 | 8x10 | 80 | 64 | 1 | 2 | 25.3 | 21.2 | +19.5% | 12652 | 10591 | 20.7 |
| 4k | 6144 | 8x10 | 80 | 64 | 1 | 3 | 33.8 | 29.7 | +13.7% | 11261 | 9904 | 23.3 |
| 8k | 10240 | 8x10 | 80 | 64 | 1 | 5 | 50.8 | 46.7 | +8.9% | 10165 | 9334 | 25.8 |
| 16k | 18432 | 8x10 | 80 | 64 | 1 | 9 | 84.7 | 80.6 | +5.1% | 9413 | 8956 | 27.9 |
| 32k | 34816 | 8x10 | 80 | 64 | 1 | 17 | 152.6 | 148.5 | +2.8% | 8978 | 8738 | 29.2 |
| 50k | 53248 | 8x10 | 80 | 64 | 1 | 26 | 229.1 | 224.9 | +1.9% | 8812 | 8652 | 29.8 |
| 64k | 67584 | 8x10 | 80 | 64 | 1 | 33 | 292.6 | 288.4 | +1.4% | 8867 | 8741 | 29.6 |
| 100k | 104448 | 8x10 | 80 | 64 | 1 | 51 | 444.1 | 440.0 | +0.9% | 8709 | 8628 | 30.1 |
| 128k | 133120 | 8x10 | 80 | 64 | 1 | 65 | 567.1 | 563.0 | +0.7% | 8724 | 8661 | 30.1 |
| 192k | 198656 | 8x10 | 80 | 64 | 1 | 97 | 841.5 | 837.4 | +0.5% | 8676 | 8633 | 30.2 |
| 256k | 264192 | 8x10 | 80 | 64 | 1 | 129 | 1116.0 | 1111.9 | +0.4% | 8651 | 8619 | 30.3 |

### 3k chunk (96 rows / device): 8x10 = 80 cores, 80 active, 2 row(s) / core

| KV prefix | valid length | core grid | cores | active cores | rows / core | sort steps / row | full us | compute-only us | DM cost | ns/step full | ns/step compute-only | logits read GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 3072 | 8x10 | 80 | 80 | 2 | 2 | 42.7 | 39.1 | +9.3% | 10679 | 9766 | 13.8 |
| 3k | 6144 | 8x10 | 80 | 80 | 2 | 3 | 60.1 | 56.1 | +7.2% | 10020 | 9344 | 19.6 |
| 9k | 12288 | 8x10 | 80 | 80 | 2 | 6 | 111.0 | 107.0 | +3.7% | 9252 | 8917 | 21.2 |
| 15k | 18432 | 8x10 | 80 | 80 | 2 | 9 | 162.0 | 157.9 | +2.5% | 8998 | 8775 | 21.9 |
| 33k | 36864 | 8x10 | 80 | 80 | 2 | 18 | 314.8 | 310.7 | +1.3% | 8746 | 8632 | 22.5 |
| 51k | 55296 | 8x10 | 80 | 80 | 2 | 27 | 467.6 | 463.6 | +0.9% | 8659 | 8585 | 22.7 |
| 63k | 67584 | 8x10 | 80 | 80 | 2 | 33 | 577.8 | 573.6 | +0.7% | 8755 | 8690 | 22.5 |
| 99k | 104448 | 8x10 | 80 | 80 | 2 | 51 | 880.7 | 876.7 | +0.5% | 8634 | 8595 | 22.8 |
| 129k | 135168 | 8x10 | 80 | 80 | 2 | 66 | 1141.3 | 1137.0 | +0.4% | 8646 | 8613 | 22.7 |
| 192k | 199680 | 8x10 | 80 | 80 | 2 | 98 | 1690.0 | 1685.9 | +0.2% | 8623 | 8601 | 22.7 |
| 255k | 264192 | 8x10 | 80 | 80 | 2 | 129 | 2224.7 | 2220.4 | +0.2% | 8623 | 8606 | 22.8 |

### 4k chunk (128 rows / device): 8x10 = 80 cores, 80 active, 2 row(s) / core

| KV prefix | valid length | core grid | cores | active cores | rows / core | sort steps / row | full us | compute-only us | DM cost | ns/step full | ns/step compute-only | logits read GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 4096 | 8x10 | 80 | 80 | 2 | 2 | 44.3 | 39.2 | +13.1% | 11080 | 9794 | 23.7 |
| 4k | 8192 | 8x10 | 80 | 80 | 2 | 4 | 78.3 | 73.1 | +7.1% | 9788 | 9141 | 26.8 |
| 8k | 12288 | 8x10 | 80 | 80 | 2 | 6 | 112.3 | 107.1 | +4.9% | 9360 | 8926 | 28.0 |
| 16k | 20480 | 8x10 | 80 | 80 | 2 | 10 | 180.2 | 175.0 | +3.0% | 9012 | 8752 | 29.1 |
| 32k | 36864 | 8x10 | 80 | 80 | 2 | 18 | 316.1 | 310.9 | +1.7% | 8780 | 8635 | 29.9 |
| 48k | 53248 | 8x10 | 80 | 80 | 2 | 26 | 451.9 | 446.7 | +1.2% | 8691 | 8590 | 30.2 |
| 64k | 69632 | 8x10 | 80 | 80 | 2 | 34 | 593.5 | 588.1 | +0.9% | 8728 | 8649 | 30.0 |
| 100k | 106496 | 8x10 | 80 | 80 | 2 | 52 | 898.9 | 893.8 | +0.6% | 8644 | 8594 | 30.3 |
| 128k | 135168 | 8x10 | 80 | 80 | 2 | 66 | 1142.5 | 1137.1 | +0.5% | 8655 | 8614 | 30.3 |
| 192k | 200704 | 8x10 | 80 | 80 | 2 | 98 | 1691.5 | 1686.0 | +0.3% | 8630 | 8602 | 30.4 |
| 256k | 266240 | 8x10 | 80 | 80 | 2 | 130 | 2240.4 | 2234.9 | +0.2% | 8617 | 8596 | 30.4 |

### 5k chunk (160 rows / device): 8x10 = 80 cores, 80 active, 2 row(s) / core

| KV prefix | valid length | core grid | cores | active cores | rows / core | sort steps / row | full us | compute-only us | DM cost | ns/step full | ns/step compute-only | logits read GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0k | 5120 | 8x10 | 80 | 80 | 2 | 3 | 62.4 | 56.1 | +11.2% | 10406 | 9359 | 26.2 |
| 5k | 10240 | 8x10 | 80 | 80 | 2 | 5 | 96.3 | 90.1 | +6.9% | 9633 | 9014 | 34.0 |
| 10k | 15360 | 8x10 | 80 | 80 | 2 | 8 | 147.3 | 141.0 | +4.4% | 9206 | 8814 | 33.4 |
| 15k | 20480 | 8x10 | 80 | 80 | 2 | 10 | 181.3 | 175.0 | +3.6% | 9064 | 8751 | 36.1 |
| 30k | 35840 | 8x10 | 80 | 80 | 2 | 18 | 317.2 | 310.8 | +2.1% | 8812 | 8634 | 36.1 |
| 50k | 56320 | 8x10 | 80 | 80 | 2 | 28 | 486.8 | 480.6 | +1.3% | 8693 | 8582 | 37.0 |
| 65k | 71680 | 8x10 | 80 | 80 | 2 | 35 | 611.5 | 605.1 | +1.0% | 8735 | 8645 | 37.5 |
| 100k | 107520 | 8x10 | 80 | 80 | 2 | 53 | 917.0 | 910.7 | +0.7% | 8650 | 8592 | 37.5 |
| 130k | 138240 | 8x10 | 80 | 80 | 2 | 68 | 1177.4 | 1171.0 | +0.5% | 8657 | 8611 | 37.6 |
| 190k | 199680 | 8x10 | 80 | 80 | 2 | 98 | 1692.5 | 1686.0 | +0.4% | 8635 | 8602 | 37.8 |
| 255k | 266240 | 8x10 | 80 | 80 | 2 | 130 | 2241.5 | 2234.9 | +0.3% | 8621 | 8596 | 38.0 |

## Findings

- **Top-k is compute bound at every chunk size.** DM costs +1-2% at a ~50k prefix and under 1% from
  ~100k. Even at a 0 prefix it is only +10-30%. The critical core spends a flat ~8.6-8.7 us per
  2048-element sort step, full or compute-only, at every chunk size and grid. Time is
  `rows_per_core x ceil(valid / 2048) x ~8.65 us`, plus ~5-10 us of fixed cost.
- **Latency scales in whole rows per core, not in tokens.** On the production 80-core grid, 1k and 2k
  cost the same (1 row / core, ~229 us at 50k). 3k, 4k and 5k all cost about the same (2 rows / core,
  452-487 us). Shrinking the chunk only helps when it drops `ceil(rows / 80)`.
- **Per-token efficiency is core util.** us per 1k tokens at 50k: 5k = 97, 4k = 113, 2k = 115,
  3k = 156, 1k = 228. 2k is only ~19% worse than 5k per token, 3k is ~61% worse (96 rows = 80 + 16
  stragglers), and 1k is 2.35x worse (32 of 80 cores busy).
- **The full 12x10 grid only changes the step function.** With 120 cores, 3k fits in 1 row / core
  (239 us, 2x faster than on 80). 4k and 5k still need 2 rows / core. In production the extra 40
  cores run the sparse-KV gather concurrently.
- The reader peaks at ~38 GB/s of logits per chip (5k, 256k prefix), and the op is still compute
  bound there.
