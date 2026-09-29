# P0-C: MSA ops at M3 shapes, (2,4) sub-mesh

**Source.** `bench/msa.csv`, produced by `bench/bench_msa.py`. The inputs, timing and op structure are described
in `bench/README_bench.md`.

**Setup.** Per chip there are 16 q heads, 1 KV group and 1 index head, since TP=4 splits the 64 q heads, 4 KV
heads and 4 index heads. Selection is top-16 blocks of 128. K, V and index_k are bf8.

* **rows** is the query rows per chip (W/2 at SP=2).
* **kv_len** is the gathered context. The bench sets cached_len = kv_len − 2·rows. The point rows=4096 at
  kv=4096 is skipped.
* **Time** is device-kernel time: the median of 5 repeats per chip, then worst and mean over the 8 chips.

**Roofline.** This is Pavlo's `roofSeg` via `tools/roofline_ops.js --segments 2·rows:(kv_len − 2·rows)`; for
idx_branch it is `roofTok` at T = 2·rows.

* **sparse** = `max(4·rows·16·2048·128 / 304 TF, rows·2048·272 B / 32 / 512 GB/s)`. The /32 assumes each K/V row
  is reused by 32 queries.
* **indexer** = `2·rows·4·128·(2048 + kv_len/128) / 304 TF`. This is block-pooled scoring, which is not what the
  kernel does.
* **idx_branch** = `max(2·rows·6144·160 / 304 TF, 6144·640·1.0625 B / 512 GB/s)`.

**Achieved rates.** TFLOP/s and GB/s use the bench's own FLOP and byte counts (README "Inputs"):

* **sparse.** FLOPs are `4·rows·16·16·128·128`. Bytes are dominated by the K/V refetch, `rows·16 blocks·128·128·2·1.0625`.
* **indexer.** FLOPs are the full rectangle, `2·rows·kv_len·128`.
* **idx_branch.** FLOPs are `2·rows·6144·256`: index q (128 columns) plus index k (128 columns, replicated).
* **topk.** It has no FLOP count, and its bytes are the score read.

## Bench results

### sparse_sdpa_msa

| rows/chip | kv_len | worst ms | mean ms | roofline ms | eff | TFLOP/s | GB/s | cores |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4,096 | 1.540 | 1.537 | 0.0565 | 3.7% | 11.2 | 376 | 120 |
| 2048 | 4,096 | 2.935 | 2.332 | 0.1130 | 3.9% | 11.7 | 394 | 120 |
| 1024 | 141,312 | 1.500 | 1.495 | 0.0565 | 3.8% | 11.5 | 386 | 120 |
| 2048 | 141,312 | 2.879 | 2.870 | 0.1130 | 3.9% | 11.9 | 402 | 120 |
| 4096 | 141,312 | 6.116 | 6.102 | 0.2261 | 3.7% | 11.2 | 379 | 120 |
| 1024 | 548,864 | 1.503 | 1.497 | 0.0565 | 3.8% | 11.4 | 385 | 120 |
| 2048 | 548,864 | 2.878 | 2.871 | 0.1130 | 3.9% | 11.9 | 402 | 120 |
| 4096 | 548,864 | 6.123 | 6.101 | 0.2261 | 3.7% | 11.2 | 378 | 120 |

### indexer_score_msa

| rows/chip | kv_len | worst ms | mean ms | roofline ms | eff | TFLOP/s | GB/s | cores |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4,096 | 0.053 | 0.052 | 0.0072 | 13.7% | 20.4 | 17 | 32 |
| 2048 | 4,096 | 0.094 | 0.093 | 0.0143 | 15.2% | 22.8 | 13 | 32 |
| 1024 | 141,312 | 0.241 | 0.241 | 0.0109 | 4.5% | 153.6 | 90 | 96 |
| 2048 | 141,312 | 0.472 | 0.471 | 0.0217 | 4.6% | 157.1 | 52 | 96 |
| 4096 | 141,312 | 0.943 | 0.938 | 0.0435 | 4.6% | 157.1 | 32 | 96 |
| 1024 | 548,864 | 0.810 | 0.808 | 0.0219 | 2.7% | 177.6 | 103 | 96 |
| 2048 | 548,864 | 1.605 | 1.601 | 0.0437 | 2.7% | 179.3 | 58 | 96 |
| 4096 | 548,864 | 3.244 | 3.218 | 0.0874 | 2.7% | 177.4 | 34 | 96 |

### topk_large_indices (no roofline of its own; the sim folds it into `indexer`)

| rows/chip | kv_len | worst ms | mean ms | roofline ms | eff | TFLOP/s | GB/s | cores |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4,096 | 0.021 | 0.021 | - | - | - | 6 | 120 |
| 2048 | 4,096 | 0.041 | 0.041 | - | - | - | 6 | 120 |
| 1024 | 141,312 | 0.050 | 0.050 | - | - | - | 47 | 120 |
| 2048 | 141,312 | 0.095 | 0.095 | - | - | - | 50 | 120 |
| 4096 | 141,312 | 0.184 | 0.184 | - | - | - | 53 | 120 |
| 1024 | 548,864 | 0.135 | 0.135 | - | - | - | 66 | 120 |
| 2048 | 548,864 | 0.264 | 0.264 | - | - | - | 67 | 120 |
| 4096 | 548,864 | 0.513 | 0.513 | - | - | - | 69 | 120 |

### idx_branch (`index_branch_forward`, 18 programs)

| rows/chip | kv_len | worst ms | mean ms | roofline ms | eff | TFLOP/s | GB/s | cores |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4,096 | 0.223 | 0.222 | 0.0082 | 3.7% | 14.4 | 73 | 120 |
| 2048 | 4,096 | 0.402 | 0.400 | 0.0132 | 3.3% | 16.0 | 77 | 120 |
| 1024 | 141,312 | 0.223 | 0.222 | 0.0082 | 3.7% | 14.5 | 73 | 120 |
| 2048 | 141,312 | 0.402 | 0.401 | 0.0132 | 3.3% | 16.0 | 77 | 120 |
| 4096 | 141,312 | 0.751 | 0.747 | 0.0265 | 3.5% | 17.2 | 80 | 120 |
| 1024 | 548,864 | 0.223 | 0.223 | 0.0082 | 3.7% | 14.4 | 73 | 120 |
| 2048 | 548,864 | 0.402 | 0.400 | 0.0132 | 3.3% | 16.0 | 77 | 120 |
| 4096 | 548,864 | 0.750 | 0.746 | 0.0265 | 3.5% | 17.2 | 81 | 120 |

### Full `msa_indexer_sparse` chain (indexer + topk + sparse + 2 layout ops); roofline = indexer + sparse

| rows/chip | kv_len | worst ms | mean ms | roofline ms | eff | TFLOP/s | GB/s | cores |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4,096 | 1.687 | 1.679 | 0.0637 | 3.8% | 10.8 | 344 | 120 |
| 2048 | 4,096 | 3.204 | 2.597 | 0.1274 | 4.0% | 11.4 | 362 | 120 |
| 1024 | 141,312 | 1.854 | 1.845 | 0.0674 | 3.6% | 29.3 | 325 | 120 |
| 2048 | 141,312 | 3.563 | 3.544 | 0.1348 | 3.8% | 30.4 | 333 | 120 |
| 4096 | 141,312 | 7.465 | 7.450 | 0.2695 | 3.6% | 29.1 | 315 | 120 |
| 1024 | 548,864 | 2.487 | 2.451 | 0.0784 | 3.2% | 64.8 | 270 | 120 |
| 2048 | 548,864 | 4.842 | 4.819 | 0.1567 | 3.2% | 66.5 | 262 | 120 |
| 4096 | 548,864 | 10.067 | 10.033 | 0.3135 | 3.1% | 64.0 | 245 | 120 |

## Per-op findings

### sparse_sdpa_msa: gather-bound on per-token K/V refetch

* **Scaling.** Time is proportional to rows and flat in kv_len: 1.50 / 2.88 / 6.12 ms at 1024 / 2048 / 4096 rows,
  identical at 141k and 549k. It is not a latency floor, and it does not scale with kv_len.
* **Parallelism.** The kernel always uses the full grid, 120 of 120 cores, with no program config. The work items
  are the `rows × n_kv` (token, KV group) pairs, rows × 1 here, split evenly with the remainder going to the first
  cores: 8.5 / 17 / 34 tokens per core. So it is not under-parallelised.
* **Heads.** It is not serial over heads either. The 16 q heads of a group ride in one 32-row q tile, so half the
  tile is padding.
* **Per item.**
  1. Load the token's 16 block ids and find the −1 tail.
  2. For each of the 16 blocks, fetch its K/V tiles. The reader takes the upper half and the writer the lower
     half, with an in-kernel block-cyclic remap.
  3. Run QK → causal mask on the diagonal → online softmax → PV.
* **Bound.** Nothing is shared between tokens, so each query token re-reads 16·128 keys × (K+V) ≈ 557 KB. That
  gives **375-402 GB/s, 73-79% of the 512 GB/s DRAM peak**, while compute runs at 11-12 TFLOP/s (about 4% of
  HiFi2). The kernel is DRAM-bandwidth-bound on a gather that Pavlo's roofline assumes is reused 32x. That
  assumption is the whole 3.9% "efficiency".
* **Confirmation from dtype.** The in-model h=0 path feeds bf16 K/V (see Anomaly B), which is 1.88x the bytes, and
  runs 1.73-1.75x slower at the same shape.
* **Fix direction (fork for M3).** Amortise K/V across query tokens:
  * Process a tile of consecutive tokens per group against the union of their selected blocks. Neighbours share
    most blocks: the forced local block plus sinks.
  * Or order the work so a core's tokens reuse L1-resident blocks.
  * Pack 2 tokens × 16 heads per 32-row tile.
  With 4-8x K/V reuse, the K/V traffic alone at 2048 rows falls from 2.9 ms to 0.36-0.72 ms at the same
  400 GB/s. After that the bound is compute: the 34 GFLOP is 0.11 ms at HiFi2 peak.

### indexer_score_msa + topk: good against its real FLOPs, poor against the sim roofline

* **Scaling.** `indexer_score_msa` scales as rows × kv_len: 0.24 / 0.47 / 0.94 ms at 141k, and 3.4x from 141k to
  549k.
* **Throughput.** At kv ≥ 141k it runs at **154-179 TFLOP/s on its real rectangle (2·rows·kv·128)**. That is
  25-29% of the 608 TF LoFi peak (the op honours LoFi only for bf8), on 96 cores.
* **Grid.** The grid is a banded rectangle: rows/64 q-groups on grid rows, with Q multicast along a row, and
  ceil(kv_tiles/32) k-bands on up to grid_x columns, with K multicast down a column. At kv=4096 there are only 4
  bands, so just 32 cores are used and it runs at about 21 TF/s. That only matters for the first chunk (0.05-0.09 ms).
* **Causal masking.** Each group computes the full `[0, kv_len)` rectangle and masks in-band, so there is no causal
  saving.
* **Roofline mismatch.** The sim's roofline `2·rows·4·128·(2048 + kv/128)` models pooled scoring. It counts
  11x (141k) to 22x (549k) fewer FLOPs than the kernel does, which is where "7.8% → 3.9% → 2.7%" comes from.
* **Headroom.** Against the true FLOPs at a 70%-of-LoFi target the headroom is about 2.4-2.8x (0.47 ms → about
  0.17 ms at 2048 rows / 141k).
* **Beyond that.** Going further needs block-pooled index keys, an algorithm change that the M3 reference does not
  use.
* **topk_large_indices.** It scales with rows × row length (kv/128 blocks), sub-linearly in kv: 0.095 ms at 141k
  and 0.264 ms at 549k for 2048 rows. It uses 120 cores, with rows split by `split_work_to_cores`, and does a
  sort/merge per 512-2048-element window. It is 14-17% of the indexer zone.

### idx_branch: two thin matmuls plus 16 small ops

* **Scaling and cost.** Time is proportional to rows and flat in kv: 0.22 / 0.40 / 0.75 ms. In the model at
  W=4096 the zone has 18 ops:
  * 2 × `MatmulDeviceOperation`, about 0.157 ms each, [2048×6144]·[6144×128] for the index q and k projections.
    That is 20 TFLOP/s, about 7% of HiFi2: N = 4 tiles leaves the matmul badly parallelised.
  * 16 small ops add about 0.085 ms: untilize, permute, tilize, per-head RMSNorm, slices, indexed RoPE and concat.
* **Roofline.** Pavlo's roofline counts 640/tp = 160 output columns. The chip really computes 256 (index q 128 +
  replicated index k 128), so the true roofline is about 0.021 ms, not 0.013 ms.
* **Fix.** Fold the two projections into `qkv_proj` (N 2304 → 2560 per chip). qkv runs at 80-87%, so the index
  projections become about 0.03 ms. Then fuse norm + RoPE. This is a program-config and model-level change, not a
  new kernel.

## In-model zones against the bench (`per_op.csv`, prose, layers 3-6, worst chip)

The in-model `sparse` zone also holds `to_layout(q, ROW_MAJOR)` and `to_layout(out, TILE)`: 0.045 + 0.08 ms at
2048 rows, and about 2x that at 4096. The `indexer` zone holds indexer_score + topk.

| model point | rows, kv_len | sparse: zone / bench op | indexer: zone / bench indexer+topk | idx_branch: zone / bench |
|---|---|---|---|---|
| W=4096, h=139,264 | 2048, 143,360 | 2.994 / 2.879 (+0.125 layout = 3.00) | 0.570 / 0.567 (bench kv 141,312) | 0.402 / 0.402 |
| W=4096, h=548,864 | 2048, 552,960 | 2.942 / 2.878 | 1.902 / 1.869 (bench kv 548,864) | 0.402 / 0.402 |
| W=8192, h=139,264 | 4096, 147,456 | 6.369 / 6.116 | 1.232 / 1.127 (bench kv 141,312; ×1.04 for kv ≈ 1.17) | 0.753 / 0.751 |
| W=8192, h=548,864 | 4096, 557,056 | 6.377 / 6.123 | 3.834 / 3.757 | 0.750 / 0.750 |
| W=4096, h=0 | 2048, 4,096 | **5.253 / 2.935** (rank 1); rank 0 3.15 / 1.73 | 0.146 / 0.135 | 0.405 / 0.402 |
| W=8192, h=0 | 4096, 8,192 | **11.04** (rank 1) / 8.60 (rank 0); no bench point | 0.307 / - | 0.752 / - |

At depth, every MSA op in the model matches the bench within 1-4%, plus the layout ops. The bench therefore
isolates the kernels, and the model adds no hidden cost there. The exception is h=0.

## Anomaly B: at h=0, sparse is 5.3 ms against 3.0 ms at 139k, and misc 3.0 ms against 0.9 ms

These are W=4096 figures; W=8192 shows the same thing at 11.0 against 6.4 ms, and misc 4.1 against 1.8 ms.

### Cause 1: a different MSA path with bf16 K/V

* `tt/attention/prefill.py:330` branches on `cached_len > 0`. At h=0 it calls `msa_sp_attention_nocache`, not
  `msa_sp_attention_cache_read`.
* The no-cache path all-gathers the chunk's **fresh bf16 activations**, `tt_k` and `tt_v` as `[1,1,2048,128]`,
  into exact-size `[1,1,4096,128]` BFLOAT16 buffers. It then calls `sparse_sdpa_msa` with no block-cyclic remap and
  no kv_len bound.
* At depth the op reads the **bf8 KV cache**, `[1,1,143360,128]` BFLOAT8_B. The ops CSV shows exactly this:
  INPUT_1 and INPUT_2 are BFLOAT16 at h=0 and BFLOAT8_B at 139k.
* sparse is gather-bound, so 1.88x the bytes gives:
  * SP rank 1 (tokens 2048-4095): 5.07-5.12 ms, against 2.93 ms in the bench at the same shape with bf8. That is
    1.74x.
  * SP rank 0 (tokens 0-2047): 3.02-3.04 ms, against 1.73 ms in the bench. That is 1.75x. Rank 0 is cheaper at h=0
    because early tokens have fewer than 16 valid causal blocks.
* The sparse zone's worst chip is therefore always SP rank 1 at h=0.

### Cause 2: misc is an arrival wait, not a kernel

* The +2.05 ms is one op, the axis-0 `AllGatherDeviceOperation` in the MoE routing setup. It is untracked at
  LEVEL=2, so it lands in misc.
* It takes **2.05-2.08 ms on the four SP-rank-0 chips and 0.005 ms on the rank-1 chips**. Rank 0 finishes attention
  2 ms early and waits there for rank 1.
* Per chip, the layer total is still set by rank 1: sparse 5.2 plus misc 0.9 ms. The worst-chip sparse and the
  worst-chip misc are the same 2 ms counted on different chips.
* The same pattern holds at W=8192: rank 0's MoE all-gather waits 2.5 ms.
* It also holds in the packed forwards. Their h=0 segment costs 2.04 ms (W=4096, 1024 rows, rank 1) against
  0.81 ms on rank 0, and the MoE all-gather on rank 0 waits 1.25-1.34 ms.

### Ruled out

* **The calls count.** It is the same at h=0 and 139k: 74 ops per sparse layer, and 3 ops in `sparse_sdpa`
  (untilize, `SparseSDPAMsaOperation`, tilize).
* **Warm-up or compile effects.**
  * With `PROFILE_SKIP_COMPILE=1` and no prefix, `profile_prefill.py` runs `PROFILE_WARM_ITERS=2` warm forwards of
    the same chunk. The warm point repeats forward #2 three more times; the log says "warm point: forward #2
    repeated 3x".
  * Then comes one recorded warm forward. The profiled forward is the 7th execution of the same programs.
  * Device-kernel time does not include host compile in any case. `parse.log` reports no non-device ops.
* **Other ops.** The h=0 indexer (0.15 ms) and idx_branch (0.40 ms) are normal.

### Fix

Feed the h=0 MSA from bf8 K/V: typecast before the no-cache gather, or read the just-written cache through the
cache-read path. The expected result is about 2.93 + 0.13 ms on rank 1, and the misc wait falls to the residual
causal skew of about 1.2 ms. That saves about 2.2 ms per sparse layer (about 14%) on the first chunk of every
request at W=4096. At W=8192 the saving is about 4.6 ms (about 15%): rank 1 goes from 11.0 to about 6.4 ms,
which is the depth value at 4096 rows, since there is no bench point for this shape.

The remaining rank-0/rank-1 skew at h=0 is causal. Only a load-balanced SP layout for the first chunk would remove
it.
