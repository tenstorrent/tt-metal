# GLM-5.2 sparse-KV prefix gather (`high_bw_all_gather`): chunk and KV-prefix sweep on 8x4

Baseline: sweep branch on `main` @ `db596ae144d`, Blackhole Galaxy, FABRIC_2D_TORUS_XY. The test is
`test_glm_kvpe_gather_chunk_sweep.py`.

## Setup (matches `ttMLA._gather_kvpe_prefix`)

- One full-mesh snake gather (`cluster_axis=None`, `num_links=2`) of the TP-deduped BF16_RM KVPE cache: per
  device `[1, 1, T/32, 576]` row-major, one slot, into the replicated `[1, 1, T, 576]` scratch.
- Extent `gathered_dim_size` = prefix + chunk, rounded up to whole block-cyclic slabs (slab = chunk).
- The cache is sized to extent + one chunk. The model's is 1M tokens, but the gather only moves the extent.
- Runs standalone on its default workers. In production it runs on the 40-core overlap sub-device, concurrently
  with top-k on the other 80 cores.
- No compute-only mode, since this is a pure CCL. Each device must receive `extent x 31/32` rows of 1152 B.
  The roofline is 200 Gb/s per link per direction x 2 links x 2 ring directions = 100 GB/s received per device
  (the constants from `test_sparse_mla_ccl_perf.py`).
- Timing is the realtime profiler over 10 trace replays, median; max over the 32 chips.

## Chunk sweep at ~50k prefix

| chunk | prefix | extent | received per device | full us | ideal us | GB/s | % of roofline | us per 1k tokens |
|---|---|---|---|---|---|---|---|---|
| 1k | 51200 | 52224 | 58.3 MB | 790 | 583 | 73.7 | 73.7 | 790 |
| 2k | 51200 | 53248 | 59.4 MB | 809 | 594 | 73.4 | 73.4 | 405 |
| 3k | 52224 | 55296 | 61.7 MB | 739 | 617 | 83.5 | 83.5 | 246 |
| 4k | 49152 | 53248 | 59.4 MB | 732 | 594 | 81.2 | 81.2 | 183 |
| 5k | 51200 | 56320 | 62.9 MB | 786 | 629 | 80.0 | 80.0 | 157 |

## KV-prefix sweep

### 1k chunk

| KV prefix | extent | received per device | full us (min-max) | ideal us | GB/s | % of roofline | us per 1k tokens |
|---|---|---|---|---|---|---|---|
| 0k | 1024 | 1.1 MB | 116 (97-131) | 11 | 9.8 | 9.8 | 116 |
| 2k | 3072 | 3.4 MB | 132 (122-139) | 34 | 26.0 | 26.0 | 132 |
| 4k | 5120 | 5.7 MB | 135 (132-161) | 57 | 42.3 | 42.3 | 135 |
| 8k | 9216 | 10.3 MB | 200 (197-222) | 103 | 51.5 | 51.5 | 200 |
| 16k | 17408 | 19.4 MB | 317 (310-344) | 194 | 61.2 | 61.2 | 317 |
| 32k | 33792 | 37.7 MB | 539 (531-542) | 377 | 70.0 | 70.0 | 539 |
| 50k | 52224 | 58.3 MB | 790 (786-796) | 583 | 73.7 | 73.7 | 790 |
| 64k | 66560 | 74.3 MB | 879 (864-889) | 743 | 84.5 | 84.5 | 879 |
| 100k | 103424 | 115.4 MB | 1312 (1304-1328) | 1154 | 87.9 | 87.9 | 1312 |
| 128k | 132096 | 147.4 MB | 1664 (1650-1687) | 1474 | 88.6 | 88.6 | 1664 |
| 192k | 197632 | 220.6 MB | 2460 (2456-2465) | 2206 | 89.7 | 89.7 | 2460 |
| 256k | 263168 | 293.7 MB | 3199 (3186-3212) | 2937 | 91.8 | 91.8 | 3199 |

### 2k chunk

| KV prefix | extent | received per device | full us (min-max) | ideal us | GB/s | % of roofline | us per 1k tokens |
|---|---|---|---|---|---|---|---|
| 0k | 2048 | 2.3 MB | 124 (103-128) | 23 | 18.5 | 18.5 | 62 |
| 2k | 4096 | 4.6 MB | 149 (129-152) | 46 | 30.8 | 30.8 | 74 |
| 4k | 6144 | 6.9 MB | 167 (160-173) | 69 | 41.1 | 41.1 | 83 |
| 8k | 10240 | 11.4 MB | 224 (219-238) | 114 | 50.9 | 50.9 | 112 |
| 16k | 18432 | 20.6 MB | 327 (325-343) | 206 | 62.9 | 62.9 | 164 |
| 32k | 34816 | 38.9 MB | 553 (543-558) | 389 | 70.3 | 70.3 | 276 |
| 50k | 53248 | 59.4 MB | 809 (803-816) | 594 | 73.4 | 73.4 | 405 |
| 64k | 67584 | 75.4 MB | 888 (869-893) | 754 | 84.9 | 84.9 | 444 |
| 100k | 104448 | 116.6 MB | 1322 (1317-1326) | 1166 | 88.2 | 88.2 | 661 |
| 128k | 133120 | 148.6 MB | 1663 (1643-1674) | 1486 | 89.3 | 89.3 | 831 |
| 192k | 198656 | 221.7 MB | 2450 (2446-2466) | 2217 | 90.5 | 90.5 | 1225 |
| 256k | 264192 | 294.8 MB | 3205 (3192-3218) | 2948 | 92.0 | 92.0 | 1602 |

### 3k chunk

| KV prefix | extent | received per device | full us (min-max) | ideal us | GB/s | % of roofline | us per 1k tokens |
|---|---|---|---|---|---|---|---|
| 0k | 3072 | 3.4 MB | 128 (117-154) | 34 | 26.7 | 26.7 | 43 |
| 3k | 6144 | 6.9 MB | 164 (157-185) | 69 | 41.9 | 41.9 | 55 |
| 9k | 12288 | 13.7 MB | 245 (226-262) | 137 | 55.9 | 55.9 | 82 |
| 15k | 18432 | 20.6 MB | 324 (323-330) | 206 | 63.6 | 63.6 | 108 |
| 33k | 36864 | 41.1 MB | 579 (569-592) | 411 | 71.0 | 71.0 | 193 |
| 51k | 55296 | 61.7 MB | 739 (727-751) | 617 | 83.5 | 83.5 | 246 |
| 63k | 67584 | 75.4 MB | 892 (886-903) | 754 | 84.6 | 84.6 | 297 |
| 99k | 104448 | 116.6 MB | 1324 (1311-1340) | 1166 | 88.0 | 88.0 | 441 |
| 129k | 135168 | 150.8 MB | 1669 (1663-1703) | 1508 | 90.4 | 90.4 | 556 |
| 192k | 199680 | 222.8 MB | 2462 (2458-2467) | 2228 | 90.5 | 90.5 | 820 |
| 255k | 264192 | 294.8 MB | 3200 (3183-3220) | 2948 | 92.1 | 92.1 | 1067 |

### 4k chunk

| KV prefix | extent | received per device | full us (min-max) | ideal us | GB/s | % of roofline | us per 1k tokens |
|---|---|---|---|---|---|---|---|
| 0k | 4096 | 4.6 MB | 146 (128-150) | 46 | 31.4 | 31.4 | 36 |
| 4k | 8192 | 9.1 MB | 187 (172-190) | 91 | 48.9 | 48.9 | 47 |
| 8k | 12288 | 13.7 MB | 244 (227-259) | 137 | 56.2 | 56.2 | 61 |
| 16k | 20480 | 22.9 MB | 360 (341-371) | 229 | 63.5 | 63.5 | 90 |
| 32k | 36864 | 41.1 MB | 576 (564-583) | 411 | 71.4 | 71.4 | 144 |
| 48k | 53248 | 59.4 MB | 732 (718-745) | 594 | 81.2 | 81.2 | 183 |
| 64k | 69632 | 77.7 MB | 935 (925-954) | 777 | 83.1 | 83.1 | 234 |
| 100k | 106496 | 118.8 MB | 1370 (1361-1389) | 1188 | 86.7 | 86.7 | 343 |
| 128k | 135168 | 150.8 MB | 1673 (1668-1682) | 1508 | 90.2 | 90.2 | 418 |
| 192k | 200704 | 224.0 MB | 2472 (2463-2484) | 2240 | 90.6 | 90.6 | 618 |
| 256k | 266240 | 297.1 MB | 3246 (3232-3269) | 2971 | 91.5 | 91.5 | 812 |

### 5k chunk

| KV prefix | extent | received per device | full us (min-max) | ideal us | GB/s | % of roofline | us per 1k tokens |
|---|---|---|---|---|---|---|---|
| 0k | 5120 | 5.7 MB | 154 (150-156) | 57 | 37.2 | 37.2 | 31 |
| 5k | 10240 | 11.4 MB | 219 (209-229) | 114 | 52.2 | 52.2 | 44 |
| 10k | 15360 | 17.1 MB | 287 (267-328) | 171 | 59.7 | 59.7 | 57 |
| 15k | 20480 | 22.9 MB | 355 (347-374) | 229 | 64.5 | 64.5 | 71 |
| 30k | 35840 | 40.0 MB | 562 (559-571) | 400 | 71.1 | 71.1 | 112 |
| 50k | 56320 | 62.9 MB | 786 (783-803) | 629 | 80.0 | 80.0 | 157 |
| 65k | 71680 | 80.0 MB | 942 (937-957) | 800 | 84.9 | 84.9 | 188 |
| 100k | 107520 | 120.0 MB | 1375 (1369-1379) | 1200 | 87.3 | 87.3 | 275 |
| 130k | 138240 | 154.3 MB | 1746 (1743-1754) | 1543 | 88.4 | 88.4 | 349 |
| 190k | 199680 | 222.8 MB | 2466 (2446-2472) | 2228 | 90.4 | 90.4 | 493 |
| 255k | 266240 | 297.1 MB | 3257 (3243-3269) | 2971 | 91.2 | 91.2 | 651 |

## Findings

- **Gather time depends only on the extent, not on the chunk.** At about 50k it is 739-809 us for every chunk
  size, and about 3.2 ms at about 256k. Every chunk re-gathers the whole populated prefix, so per token it scales
  as 1/chunk: at 50k, 790 us per 1k tokens at 1k, 405 at 2k, 246 at 3k, 183 at 4k and 157 at 5k. Over a whole
  prompt, total gather work grows like N^2 / chunk.
- **Bandwidth is good at long extents:** 80-92% of the 100 GB/s roofline from about 50k up, 61-71% at 16-35k.
  Below about 10k the gather is latency bound, with a floor of about 100-150 us.
- **It already sets the critical path of the top-k overlap.** At 5k/50k the gather (786 us, standalone on its
  default workers) is longer than top-k on the 80-core grid (487 us). At 2k it is 809 vs 229 us. Shrinking the
  chunk makes top-k shorter but not the gather, so the overlap hides less and less.
- **For the chunk-size question,** this and the indexer score are the two ops that do not get cheaper per call
  with a smaller chunk. Both are O(prefix) per call, so their per-token cost goes up roughly 2.5x going from
  5k to 2k.
