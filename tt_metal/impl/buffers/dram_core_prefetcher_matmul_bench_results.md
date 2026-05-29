# DRAM-core prefetcher — matmul benchmark comparison

Compares the three prefetcher variants feeding a **real `gather_in0` matmul**
(`test_prefetcher_BH_bench.py`), unlike the discard-receiver BW bench
(`dram_core_prefetcher_recv_contig_bw_results.md`). Each test traces 100
single-matmul launches fed by the prefetcher and replays the trace once;
headline metric is TFLOP/s on the unpadded `(M, K, N)`.

- **K-row-major** (`test_bench_dram_core_repeats`): legacy DRAM-core layout,
  one wide `(K, N/num_banks)` width shard per bank.
- **Receiver-contiguous** (`test_bench_dram_core_repeats_recv_contig`): new
  layout, `NdShardSpec(Shape([K, n_per_recv]), ROUND_ROBIN_1D)`,
  `num_shards = ring_size`. Strided `bank_to_receivers` (column `b` of the
  receiver matrix) so ring position `r` receives shard `r`. Triple-buffer +
  deferred-flush DRISC kernel.
- **Worker-core** (`test_bench_workercore_repeats`): `ttnn.dram_prefetcher`
  with BRISC/NCRISC senders on Tensix workers.

Single Blackhole P150, 8 DRAM banks × 8 receivers = ring 64, bf8_b. All three
pass matmul PCC ≥ 0.99 (recv-contig measured 0.99997).

## Unblocking recv-contig → matmul (this change)

The recv-contig weight is an `NdShardSpec` (reported `BLOCK_SHARDED`, no legacy
`ShardSpec`). Three small host-side relaxations were needed; none weaken the
K-row-major production path:

1. `matmul_..._1d_program_factory.cpp` — the `use_global_cb` branch read the
   weight's legacy `shard_spec()` to size a *dead* local in1 CB (the real CB
   comes from `global_cb->size()`). Now derives from `in0_block_w`/`per_core_N`.
2. `matmul_device_operation.cpp` validate() — accept a DRAM weight fed via a
   global CB (not just `WIDTH_SHARDED`/`DRAM-interleaved`) on the gather_in0
   path.
3. `matmul_device_operation.cpp` validate() — the DRAM-sender cross-check that
   asserts the K-row "bank `b` → contiguous ring `[b*rpb,(b+1)*rpb)`" convention
   is gated to `WIDTH_SHARDED` in1 only; recv-contig uses the strided
   (round-robin) convention, verified by `test_validator_dram_sender_recv_contig`
   + this bench's PCC check.

The GCB is built with `create_global_circular_buffer_with_dram_senders`
directly (the `_for_matmul_1d` wrapper only validates the K-row layout; the GCB
object it returns is identical).

## Results (TFLOP/s; best of 2 runs — see noise caveat)

| Shape       | K-row | recv-contig | worker | rc/K-row | rc/worker |
|-------------|------:|------------:|-------:|---------:|----------:|
| 1B_QKV      |  2.64 |        2.85 |   3.02 |   1.08×  |   0.95×   |
| 1B_WO       |  2.10 |        1.88 |   1.23 |   0.90×  | **1.53×** |
| 1B_FF1      |  5.73 |        6.34 |   8.18 |   1.10×  |   0.77×   |
| 1B_FF2      |  5.28 |        7.15 |   6.09 |   1.35×  | **1.17×** |
| 3B_QKV      |  5.39 |        4.31 |   4.33 |   0.80×  |   1.00×   |
| 3B_WO       |  3.58 |        3.15 |   2.97 |   0.88×  | **1.06×** |
| 3B_FF1      |  7.13 |        5.87 |   8.83 |   0.82×  |   0.66×   |
| 3B_FF2      |  6.16 |        6.88 |   7.96 |   1.12×  |   0.86×   |
| 8B_QKV_2d   |  3.64 |        6.17 |   4.88 |   1.69×  | **1.26×** |
| 8B_WO_2d    |  3.48 |        4.48 |   2.89 |   1.29×  | **1.55×** |
| 8B_FF1_2d   |  7.41 |        7.25 |  10.87 |   0.98×  |   0.67×   |
| 8B_FF2_2d   |  7.35 |        7.77 |   7.87 |   1.06×  |   0.99×   |
| 70B_QKV_8d  |  2.91 |        3.27 |   3.61 |   1.13×  |   0.91×   |
| 70B_WO_8d   |  2.82 |        3.28 |   2.38 |   1.16×  | **1.38×** |
| 70B_FF1_8d  |  8.31 |        8.27 |   9.50 |   1.00×  |   0.87×   |
| 70B_FF2_8d  |  7.02 |        9.20 |   8.18 |   1.31×  | **1.12×** |
| **geomean** |       |             |        | **1.08×**|  **1.01×**|

recv-contig beats K-row on 10/16 (geomean 1.08×) and beats worker on 7/16
(geomean 1.01×) — winning the memory-bound WO/QKV/FF2 shapes, losing the
large-N FF1 shapes where worker's ~1.5 MB L1 dominates (same pattern as the BW
bench).

## ⚠️ Noise caveat — the matmul bench is not a reliable prefetcher discriminator

The 100-matmul trace replays in only **~10–30 ms**, so the measurement is
dominated by trace-dispatch/scheduling overhead and async-prefetcher-daemon
timing, **not** steady-state prefetcher throughput. Run-to-run variance is
**±30–40%** on many shapes for *both* dram-core variants:

| Shape      | K-row run1/run2 | recv-contig run1/run2 |
|------------|-----------------|-----------------------|
| 1B_QKV     | 1.89 / 2.64     | 2.85 / 2.15           |
| 1B_FF1     | 5.73 / 4.65     | 5.29 / 6.34           |
| 8B_QKV_2d  | 3.34 / 3.64     | 6.17 / 3.77           |
| 70B_FF1_8d | 6.44 / 8.31     | 8.27 / 6.62           |
| 70B_FF2_8d | 7.02 / 5.92     | 8.93 / 9.20           |

The per-shape ranking flips between runs. Treat the table above as
"recv-contig is roughly on par with or modestly ahead of K-row, and competitive
with worker on memory-bound shapes" — not as precise per-shape ratios.

## Key conclusion

In the real-matmul workload the **prefetcher is not the bottleneck** for most
of these shapes — the matmul compute + dispatch dominate, so the large
sender-side throughput differences the BW bench shows (recv-contig 1.1–2.5× over
K-row; triple-buffer cutting sender cycles 3×) **do not translate** into matmul
TFLOP/s gains. This matches the earlier observation that triple-buffering didn't
move the BW bench's `trace_elapsed` either: once the sender keeps up with the
consumer, making it faster is invisible end-to-end.

The headline outcome of this work is therefore correctness/capability, not a
matmul speedup: **recv-contig can now feed a production gather_in0 matmul**
(previously blocked by the NdShardSpec-vs-legacy-ShardSpec mismatch), at parity
with or modestly ahead of K-row-major.
