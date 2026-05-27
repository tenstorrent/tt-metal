# Receiver-contiguous DRAM-core prefetcher BW results

Compares three prefetcher variants on Llama production shapes:

1. **K-row-major** (`test_bw_dram_core_prefetcher`): legacy DRAM-core layout. One shard per DRAM bank `(K, n_per_bank)`. Single DMA per K-block per sender pulls full bank width; fans out to receivers via row-strided NoC writes.
2. **Receiver-contiguous** (`test_bw_dram_core_prefetcher_recv_contig`): new DRAM-core layout. `num_shards = ring_size > num_dram_banks` via `NdShardSpec(Shape([K, n_per_recv]), ROUND_ROBIN_1D)`. Per `(receiver, block)` the source bytes are a single contiguous DRAM region; the kernel batches up to ~6 stage halves' worth of consecutive blocks per receiver per visit, amortizing one NoC `set_state` call across many `with_state` writes (commit on driscprefetcherllama branch).
3. **Worker-core** (`test_bw_workercore_prefetcher`): `ttnn.dram_prefetcher` with sender kernels on Tensix workers feeding the GCB.

All three use trace-replay (1 warmup layer + `trace_repeats` traced layers), discard receiver (`test_dram_prefetcher_consumer`), and per-receiver page = `k_block_w_tiles * n_per_recv_tiles * tile_bytes`. Single Blackhole (P150), 8 DRAM banks × 8 receivers per bank = ring 64. Numbers in GB/s aggregate across all receivers.

## Production shapes

| Op             | K     | N     | K-row  | recv-contig | worker | rc/K-row | rc/worker |
|----------------|------:|------:|-------:|------------:|-------:|---------:|----------:|
| 8B_FF1_1d      | 4096  | 14336 | 277.37 |      306.25 | 389.90 | 1.10×    | 0.79×     |
| 8B_QKV_1d      | 4096  | 12288 | 283.47 |      314.61 | 390.23 | 1.11×    | 0.81×     |
| 8B_WO_1d       | 4096  |  4096 | 213.88 |      296.25 | 226.49 | 1.39×    | **1.31×** |
| 1B_QKV         | 2048  |  3072 | 160.64 |      275.08 | 143.60 | 1.71×    | **1.92×** |
| 1B_WO          | 2048  |  2048 |  84.23 |      211.85 |  71.84 | 2.52×    | **2.95×** |
| 1B_FF1         | 2048  |  8192 | 214.57 |      289.40 | 283.00 | 1.35×    | **1.02×** |
| 1B_FF2         | 8192  |  2048 | 156.15 |      281.51 | 169.43 | 1.80×    | **1.66×** |
| 3B_QKV         | 3072  |  5120 | 247.36 |      305.72 | 331.35 | 1.24×    | 0.92×     |
| 3B_WO          | 3072  |  3072 | 212.07 |      296.42 | 226.75 | 1.40×    | **1.31×** |
| 3B_FF1         | 3072  |  8192 | 254.95 |      298.46 | 375.85 | 1.17×    | 0.79×     |
| 3B_FF2         | 8192  |  3072 | 249.01 |      302.77 | 331.81 | 1.22×    | 0.91×     |
| 8B_QKV_2d      | 4096  |  3072 | 212.09 |      295.97 | 226.73 | 1.40×    | **1.31×** |
| 8B_WO_2d       | 2048  |  4096 | 160.37 |      275.07 | 143.53 | 1.72×    | **1.92×** |
| 8B_FF1_2d      | 4096  |  7168 | 254.51 |      298.57 | 376.01 | 1.17×    | 0.79×     |
| 8B_FF2_2d      | 7168  |  4096 | 248.45 |      302.66 | 332.23 | 1.22×    | 0.91×     |
| 70B_QKV_8d     | 8192  |  1280 | 156.07 |      281.60 | 169.35 | 1.80×    | **1.66×** |
| 70B_WO_8d      | 1024  |  8192 | 214.39 |      289.42 | 282.98 | 1.35×    | **1.02×** |
| 70B_FF1_8d     | 8192  |  3584 | 248.92 |      302.83 | 332.37 | 1.22×    | 0.91×     |
| 70B_FF2_8d     | 3584  |  8192 | 255.28 |      298.51 | 375.22 | 1.17×    | 0.79×     |

(**bold** = recv-contig beats worker-core on that shape.)

## Takeaways

- **Receiver-contiguous now beats K-row-major on every shape**, by 1.10×–2.52×. The biggest wins are on shapes where K-row-major was M=1 with small `n_per_bank` (the smaller LLama variants and WO/QKV decomposition): 1B_WO 2.52×, 1B_QKV 1.71×, 70B_QKV_8d 1.80×.
- **Receiver-contiguous beats worker-core on 8 of 19 shapes**, primarily the bandwidth-starved small-block shapes (WO/QKV ops where `n_per_recv * tile_bytes` is small). On these the DRAM-core kernels' per-receiver batching amortizes setup costs over a longer streaming write, while the worker-core path bottlenecks elsewhere (likely worker→GCB NoC ingress).
- **Worker-core still wins on the large-N shapes** (FF1/FF2/_1d/_2d variants with `n_per_recv * tile_bytes >= 14 KB`): 0.79×–0.92×. There each worker's ~1.5 MB local L1 lets it triple-buffer enough payload to outrun the DRISC's 70 KB ping-pong stage. Closing this remaining gap likely needs a bigger DRISC stage or moving some receiver-prep work off the DRISC critical path.
- Where the new recv-contig wins vs. old recv-contig (commit prior to this change): the old layout pinned `blocks_per_dma = 1`, so each `(block, receiver)` issued its own DMA + 1 NoC write, repeatedly reprogramming the NoC destination (`set_state`) and finalizing pages_sent. Dynamic batching collapses 6+ blocks of one receiver into one set_state + many with_state writes plus a single per-round finalize across all receivers.

## Comparison to old recv-contig (pre-batching)

Same `test_bw_dram_core_prefetcher_recv_contig`, before the dynamic-batching change. Old numbers were 0.50×–0.88× of K-row-major (see git history of this file). The change is a 1.4×–5×+ improvement on the same shapes, e.g.:

| Op       | old rc | new rc | new/old |
|----------|-------:|-------:|--------:|
| 1B_WO    |   42.2 | 211.85 |   5.02× |
| 1B_QKV   |   75.8 | 275.08 |   3.63× |
| 8B_WO_2d |   75.8 | 275.07 |   3.63× |
| 8B_WO_1d |  132.6 | 296.25 |   2.23× |
| 1B_FF1   |  140.4 | 289.40 |   2.06× |

## Open questions / future work

- The `target_per_visit_pages = 6 * stage_half_pages` factor is a guess; could sweep across {2, 4, 6, 8, 12} for a tighter optimum.
- On large-N shapes where worker still wins (FF1/FF2 above ring=64), look into whether DMA setup or NoC-ingress on the receiver side dominates — DRISC profile counters can disambiguate.
- The packet-stride flat write loop assumes block_stride == page_bytes_per_recv (receiver-contiguous case) — if a future layout breaks that assumption, the inner write logic needs to gain a stride argument.
