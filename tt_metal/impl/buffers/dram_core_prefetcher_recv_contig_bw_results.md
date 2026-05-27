# Receiver-contiguous DRAM-core prefetcher BW results

Compares three prefetcher variants on Llama production shapes:

1. **K-row-major** (`test_bw_dram_core_prefetcher`): legacy DRAM-core layout. One shard per DRAM bank `(K, n_per_bank)`. Single DMA per K-block per sender pulls full bank width; fans out to receivers via row-strided NoC writes.
2. **Receiver-contiguous** (`test_bw_dram_core_prefetcher_recv_contig`): new DRAM-core layout. `num_shards = ring_size > num_dram_banks` via `NdShardSpec(Shape([K, n_per_recv]), ROUND_ROBIN_1D)`. Per `(receiver, block)` the source bytes are a single contiguous DRAM region; one DMA + one NoC write per receiver per block.
3. **Worker-core** (`test_bw_workercore_prefetcher`): `ttnn.dram_prefetcher` with sender kernels on Tensix workers feeding the GCB.

All three use trace-replay (1 warmup layer + `trace_repeats` traced layers), discard receiver (`test_dram_prefetcher_consumer`), and per-receiver page = `k_block_w_tiles * n_per_recv_tiles * tile_bytes`. Single Blackhole (P150), 8 DRAM banks × 8 receivers per bank = ring 64. Numbers in GB/s aggregate across all receivers.

## Production shapes

| Op             | K     | N     | K-row | recv-contig | worker | rc/K  | rc/wkr | Notes                              |
|----------------|------:|------:|------:|------------:|-------:|------:|-------:|------------------------------------|
| 8B_FF1_1d      | 4096  | 14336 | 274.6 |       273.7 |  389.8 | 1.00× |  0.70× | K-row lands on M=2 (Case 3)        |
| 8B_QKV_1d      | 4096  | 12288 | 280.9 |       260.9 |  389.6 | 0.93× |  0.67× | K-row lands on M=2 (Case 3)        |
| 8B_WO_1d       | 4096  |  4096 | 204.1 |       132.6 |  226.7 | 0.65× |  0.59× | K-row M=1 fast path                |
| 1B_QKV         | 2048  |  3072 | 152.6 |        75.8 |  142.6 | 0.50× |  0.53× | K-row M=1 fast path                |
| 1B_WO          | 2048  |  2048 |  78.7 |        42.2 |   71.9 | 0.54× |  0.59× | K-row M=1 fast path                |
| 1B_FF1         | 2048  |  8192 | 207.0 |       140.4 |  283.1 | 0.68× |  0.50× | K-row M=1 fast path                |
| 1B_FF2         | 8192  |  2048 | 154.0 |       130.9 |  169.4 | 0.85× |  0.77× | K-row M=1 fast path                |
| 3B_QKV         | 3072  |  5120 | 240.1 |       178.1 |  330.2 | 0.74× |  0.54× | K-row M=1 fast path                |
| 3B_WO          | 3072  |  3072 | 203.0 |       132.3 |  226.7 | 0.65× |  0.58× | K-row M=1 fast path                |
| 3B_FF1         | 3072  |  8192 | 249.5 |       217.5 |  374.4 | 0.87× |  0.58× | K-row M=1 fast path                |
| 3B_FF2         | 8192  |  3072 | 245.7 |       215.7 |  332.3 | 0.88× |  0.65× | K-row M=1 fast path                |
| 8B_QKV_2d      | 4096  |  3072 | 202.8 |       132.4 |  226.7 | 0.65× |  0.58× | K-row M=1 fast path                |
| 8B_WO_2d       | 2048  |  4096 | 150.4 |        75.8 |  143.5 | 0.50× |  0.53× | K-row M=1 fast path                |
| 8B_FF1_2d      | 4096  |  7168 | 249.6 |       217.7 |  375.9 | 0.87× |  0.58× | K-row M=1 fast path                |
| 8B_FF2_2d      | 7168  |  4096 | 245.2 |       215.5 |  331.1 | 0.88× |  0.65× | K-row M=1 fast path                |
| 70B_QKV_8d     | 8192  |  1280 | 153.9 |       130.9 |  169.4 | 0.85× |  0.77× | K-row M=1 fast path                |
| 70B_WO_8d      | 1024  |  8192 | 206.8 |       140.4 |  282.8 | 0.68× |  0.50× | K-row M=1 fast path                |
| 70B_FF1_8d     | 8192  |  3584 | 245.5 |       215.5 |  332.0 | 0.88× |  0.65× | K-row M=1 fast path                |
| 70B_FF2_8d     | 3584  |  8192 | 249.5 |       217.8 |  375.7 | 0.87× |  0.58× | K-row M=1 fast path                |

## Takeaways

- **Receiver-contiguous wins only when K-row-major is forced into M>1.** On 8B_FF1_1d / 8B_QKV_1d (where one K-row of `n_per_bank * tile_bytes` exceeds the DRISC L1 ring half), K-row-major has to issue `M` sub-row DMAs per block; recv-contig issues one full-block DMA per receiver instead. The two cancel out: ~1:1 (8B_FF1_1d) and 0.93× (8B_QKV_1d).
- **On the K-row-major M=1 fast path, recv-contig loses 0.50–0.88×.** When K-row-major can already issue one DMA covering all `num_receivers` per block, splitting that into `num_receivers` smaller DMAs (recv-contig's design) trades a single ~13 KB read for 8 ~1.5 KB reads. The smaller DMAs don't amortize their setup, and the NoC write count is identical in both layouts (no fewer writes).
- **Worker-core is still the bandwidth winner** by 1.4–2× over the better DRAM-core variant for these shapes, mostly because each worker has ~1.5 MB local L1 to triple-buffer through vs the DRISC's ~70 KB ping-pong.

## Open questions / future work

- The `blocks_per_dma > 1` batching (planned but not landed) would amortize DMA setup across consecutive blocks of the same receiver. Could close the gap on the M=1 fast-path shapes.
- The receiver-contiguous main loop currently issues `num_receivers × num_blocks × num_sub` DMAs per layer per sender — the same total as K-row-major's M=1 fast path × M but with smaller per-DMA bytes. Need to measure whether DMA setup or NoC ingress is the bottleneck.
