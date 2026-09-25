# MiniMax-H3 AGMM on the Blackhole galaxy: fused vs unfused, per-step blocking sweeps

Status: complete 2026-09-23. Plan and run notes in [`README.md`](README.md). 45 device sessions (broker jobs),
3,521 combos measured, 3,509 OK (12 rejected by the op at program creation, all L1), every one above PCC 0.99998
against the fp32 reference.

## Question

Same as the Wormhole study ([`../minimax_h3_wormhole/agmm_fused_vs_unfused.md`](../minimax_h3_wormhole/agmm_fused_vs_unfused.md)):
for the three transformer-block projections that gather the K-sharded activation over the 4-device TP ring
(`ttnn.experimental.all_gather_minimal_matmul_async`, `models/tt_dit/layers/linear.py:436-462`), what is the fastest
fused blocking at the 15 s / 768P / 16:9 per-device shape (M = 13664 rows per device on the 4x8 mesh), and how does
the best unfused pair (all-gather swept on its own hyperparameters, standalone matmul swept on its blocking) compare?
On Blackhole the question has a new edge: the fabric bound in microseconds is the same as on Wormhole (2 links x
25 GB/s = 4 x 12.5 GB/s) while the compute bound shrinks 2.3x, so the gather is much harder to hide.

Shapes (per device, `models/tt_dit/tests/models/minimax_h3/tools/minimax_h3_ops.py`):

| op | K (gathered) | K per device | N per device | epilogue |
|---|---|---|---|---|
| to_qkv | 5376 | 1344 | 5376 | 3 output chunks |
| to_out | 7168 | 1792 | 1344 | addcmul, per-token gate |
| ff1 | 5376 | 1344 | 7168 (gate and up packed), 3584 out | SwiGLU |

## Harness

Host `g15blx02`: 32 x Blackhole (`1e52:b140`) as a 4x8 galaxy, tt-metal branch `minimax_h3_wh_optimizations` at
`60a16f57aba`, clean Release build with the Tracy profiler (2026-09-22). The device is shared through the
`tt-device-mcp` broker: one job at a time, 1500 s hard cap per job, so every device step below ran as a broker job
and the sweep cases were split into chunks of explicit combos (see "How the sweep was run").

**Device row.** `bh4x8links2_ring`
(`models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py:2162`): (4, 8) mesh,
`FABRIC_1D_RING` with an 8192 B payload, Ring, 2 links, 6 workers per link, `l1_small_size` 65536, TP on mesh axis 0.
The test runs on a 4x1 submesh of the parent mesh, so M is both the per-device and the global row count. Model
sources for the same values: `models/tt_dit/tests/models/minimax_h3/common.py:128,165-167`,
`models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py:177-209`, `models/tt_dit/utils/matmul.py:744-750`.

**What the tools resolved from the device** (sweep sidecars, `generated/agmm_h3_sweep/*.json`): `arch` blackhole;
fused / `mm_ring` worker grid **12x9** (`agmm_worker_grid`, the 12x10 compute grid minus the mux row,
`matmul.py:407-417`); `mm_full` grid **11x10** (`get_matmul_core_grid`, the Blackhole galaxy clamp keyed on the
parent mesh's 32 devices, `matmul.py:392-404`); `full_grid` 12x10; `num_buffers_per_channel` 24; shipped blocking
for all three ops `AGMM_BLOCK_SIZES` + subblock (2, 2), i.e. to_qkv (8, 7, 12, 2, 2), to_out (8, 8, 6, 2, 2),
ff1 (8, 3, 14, 2, 2) (`agmm_config.py:49-53`; the 12x9 table at `matmul.py:385-388` holds M = 3424 rows only, so
`get_agmm_config` -> `get_matmul_config` misses it at M = 13664).

**Roofline constants** (`models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py:135-150`): 1.35 GHz,
2048 FLOP/cycle/core at HiFi2, 108 cores for the fused op and `mm_ring` (110 for `mm_full`), DRAM 512 GB/s, fabric
2 links x 25 GB/s, ring of 4. Predicted bounds at M = 13664 (`transformer_roofline.py --arch bh --no-block --dump`):

| op | t_compute us (108 cores) | t_dram us | t_fabric us | ideal | limiter | Wormhole ideal |
|---|---|---|---|---|---|---|
| to_qkv | 2645 | 400 | 1102 | 2645 | compute | 6026 (compute) |
| to_out | 882 | 420 | 1469 | 1469 | **fabric** | 2009 (compute) |
| ff1 | 3527 | 438 | 1102 | 3527 | compute | 8034 (compute) |

The compute / fabric crossover `n_star` is 2239 columns at 2 links (983 on Wormhole); to_out's N = 1344 is below it.

**Candidate lists.** The same rules as on Wormhole (`h3_sweep_combos`, `test_all_gather_minimal_matmul_async.py:1745`;
`models/tt_dit/utils/sweep_mm_block_sizes.py:629-804`), evaluated for the Blackhole grids: per-core M is 36 tiles
on 12 columns (427 M tiles) and 39 on 11 columns; N per core 19 / 5 / 25 on 9 rows and 17 / 5 / 23 on 10 rows.

| case | fused (12x9) | mm_ring (12x9) | mm_full (11x10) | ag |
|---|---|---|---|---|
| to_qkv | 320 | 450 | 435 | 48 |
| to_out | 314 | 332 | 321 | 48 |
| ff1 | 320 | 450 | 435 | 48 |

**How the sweep was run.** `generated/agmm_h3_sweep/run_all_bh.py` (gitignored, beside the CSV) splits each case's
candidate list into chunks of 35 combos (48 for the all-gather), skips combos already in
`agmm_h3_sweep_results.csv`, and runs each chunk as one blocking broker job:
`agmm_unit_sweep.py run --ops <op> --M 13664 --modes <mode> --combos '<json>' --quiet --keep-going`. Each job is
one device session: every combo is warmed up and PCC-checked against the torch reference, then the valid ones are
trace-captured and executed one at a time between Tracy signposts; `report` and `roofline` dedupe to the best OK
row per combo, so the chunking does not change the statistics. Durations are device kernel time, mean over the 4
ring devices, one traced execution per combo, as on Wormhole.

## Block profile (step 1)

`scripts/run_safe_pytest.sh --profile` on
`test_minimax_h3_transformer_block_perf[blackhole-sp_sim1-test_prompt_text_tokens-15s_768p-4x8sp1tp0nl2_ring_is_fsdp0]`, second iteration
between the `start` / `stop` signposts, then `transformer_roofline.py --arch bh --profile-csv ... --dump`. Full
dump: `generated/agmm_h3_sweep/roofline_block_bh_M13664.txt`; figures in `transformer_roofline_out_bh/`.

| op | calls | measured ms | ideal ms | limiter | util | headroom |
|---|---|---|---|---|---|---|
| RingJointSDPA | 1 | 51.79 | 35.20 | compute | 68% | 1.47x |
| AGMM ff1 | 1 | 5.49 | 3.53 | compute | 64% | 1.56x |
| Embeddings (adaLN tables) | 6 | 5.13 | 0.43 | dram | 8% | 11.90x |
| AGMM to_qkv | 1 | 3.65 | 2.65 | compute | 73% | 1.38x |
| AGMM to_out | 1 | 2.92 | 1.47 | **fabric** | 50% | 1.98x |
| MM+RS ff2 (fused) | 1 | 2.50 | 1.98 | compute | 79% | 1.26x |
| DistributedRMSNorm | 4 | 1.50 | 1.02 | dram | 68% | 1.47x |
| other (small ops) | 37 | 1.84 | — | — | — | — |

Block 74.8 ms measured against 46.3 ms of summed ideals (1.62x). The three AGMM ops are 12.1 ms of the block
(16%); to_out is the only one whose ideal is its fabric bound, and it sits at exactly 2x that bound inside the
block, while to_qkv and ff1 run at 64 to 73% of their compute bound, better than the 53 to 59% the same ops reached
on Wormhole. ff2 is not an AGMM: on Blackhole it runs as the fused matmul + strided reduce-scatter
(`MinimalMatmulStridedReduceScatterAsync`, `models/tt_dit/layers/linear.py:715`, blocking from
`models/tt_dit/models/transformers/minimax_h3/mmrs_config.py`) on a 12x8 matmul grid with the reduce-scatter
workers in the remaining rows, at 79% of its compute bound; on Wormhole it was a plain matmul plus a separate
reduce-scatter. `transformer_roofline.py` gained a bound model for that op code during this run (it previously
fell into "other", which also hid its size).

## PCC gate for the shipped blockings (step 2)

```bash
python -m pytest models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py \
  -k "bh4x8links2 and h3_15s and (check or (ff1 and no_bias))" -p no:cacheprovider --timeout 1400
```

6 passed (to_qkv and to_out through `test_linear` fused and separate, ff1 through `test_linear_swiglu` fused and
separate): PCC 0.99998 to 0.99999, relative RMSE 0.005 to 0.009 against the fp32 reference, identical between the
fused and separate paths of each op. These unit rows use the 4 KB fabric payload; the sweep row uses the production
8 KB.

## Results at M = 13664

Device kernel time in us, mean over the 4 ring devices, one traced execution per combo; every row below passed
PCC > 0.99998 against the fp32 torch reference (minimum over all OK rows 0.999983). Full tables:
`agmm_unit_sweep.py report` (`generated/agmm_h3_sweep/report_bh_M13664.md`).

| op | best fused (blocking) | best all-gather (workers, chunks/sync, buffers) | best matmul 11x10 | best matmul 12x9 | AG + MM 11x10 | fused vs unfused |
|---|---|---|---|---|---|---|
| to_qkv | 3562 (9, 6, 10, 3, 1) | 1192 (3, 32, 8) | 3214 (13, 6, 6, 1, 3) | 3180 (9, 7, 10, 3, 1) | 4405 | fused 843 us faster (-19.1%) |
| to_out | 2668 (9, 14, 5, 3, 1) | 1590 (3, 32, 8) | 1355 (8, 8, 5, 4, 1) | 1248 (6, 7, 5, 3, 1) | 2945 | fused 276 us faster (-9.4%) |
| ff1 | 5045 (6, 7, 14, 2, 2) | 1193 (3, 32, 8) | 4673 (13, 6, 8, 1, 4) | 4787 (6, 6, 14, 2, 2) | 5866 | fused 821 us faster (-14.0%) |

Against the best pair on the 12x9 grid (AG + `mm_ring`) the fused margins are -18.5%, -6.0% and -15.6%.

### Fused blockings (12x9 worker grid)

to_qkv 320 combos, to_out 314, ff1 320, all OK.

| op | rank 1 | rank 2 | rank 3 | shipped |
|---|---|---|---|---|
| to_qkv | (9, 6, 10, 3, 1) 3562 | (9, 7, 10, 3, 1) 3586 | (9, 6, 12, 1, 4) 3609 | (8, 7, 12) 3673, rank 6, +3.1% |
| to_out | (9, 14, 5, 3, 1) 2668 | (6, 14, 5, 3, 1) 2672 | (8, 14, 5, 4, 1) 2673 | (8, 8, 6) 2709, rank 8, +1.5% |
| ff1 | (6, 7, 14, 2, 2) 5045 | (6, 6, 14, 2, 2) 5068 | (6, 6, 16, 2, 2) 5129 | (8, 3, 14) 5520, rank 16, +9.4% |

### All-gather hyperparameters (2 links, 48 points per tensor)

| tensor | rank 1 | rank 2 | rank 3 | production (3, 16, 2) | fabric bound |
|---|---|---|---|---|---|
| K = 5376 (to_qkv) | (3, 32, 8) 1192 | (3, 32, 4) 1193 | (3, 32, 2) 1195 | 1214, rank 5, +1.9% | 1102 (1.08x) |
| K = 7168 (to_out) | (3, 32, 8) 1590 | (3, 32, 4) 1592 | (3, 32, 2) 1594 | 1619, rank 6, +1.8% | 1469 (1.08x) |
| K = 5376 (ff1) | (3, 32, 8) 1193 | (3, 32, 4) 1194 | (3, 32, 2) 1195 | 1214, rank 5, +1.7% | 1102 (1.08x) |

3 workers per link wins every table (the six leading rows are all workers = 3); `chunks_per_sync` 32 beats 16 by
1.7 to 1.9% and 8 by 4.1 to 4.7%; the buffer count is worth 0.3%. The standalone gather runs at 1.08x its fabric
bound on Blackhole against 1.73x on Wormhole, i.e. 93% of the 25 GB/s per link-direction with the 8 KB payload.

### Standalone matmul on the gathered activation, 12x9 (`mm_ring`, the like-for-like split of the fused op)

| op | rank 1 | rank 2 | fused op's shipped blocking on this grid | OK / candidates |
|---|---|---|---|---|
| to_qkv | (9, 7, 10, 3, 1) 3180 | (9, 7, 4, 1, 4) 3216 | (8, 7, 12) 3559, rank 91, +11.9% | 446 / 450 (4 L1 rejections) |
| to_out | (6, 7, 5, 3, 1) 1248 | (6, 8, 5, 3, 1) 1249 | (8, 8, 6) 1465, rank 63, +17.4% | 330 / 332 (2 L1 rejections) |
| ff1 | (6, 6, 14, 2, 2) 4787 | (6, 7, 14, 2, 2) 4791 | (8, 3, 14) 4969, rank 49, +3.8% | 450 / 450 |

### Standalone matmul on the gathered activation, 11x10 (`mm_full`, what the model's unfused branch would run)

| op | rank 1 | rank 2 | model's blocking on this grid | OK / candidates | vs 12x9 rank 1 |
|---|---|---|---|---|---|
| to_qkv | (13, 6, 6, 1, 3) 3214 | (13, 7, 6, 1, 3) 3262 | (8, 7, 12) 3522, rank 47, +9.6% | 431 / 435 (4 L1 rejections) | +1.0% |
| to_out | (8, 8, 5, 4, 1) 1355 | (8, 7, 5, 4, 1) 1355 | (8, 8, 6) 1592, rank 46, +17.5% | 319 / 321 (2 L1 rejections) | +8.5% |
| ff1 | (13, 6, 8, 1, 4) 4673 | (13, 8, 4, 1, 4) 4689 | (8, 3, 14) 4978, rank 91, +6.5% | 435 / 435 | -2.4% |

On 11 columns each core holds 39 M tiles, and M_block 13 (three exact blocks) wins to_qkv and ff1; on 10 rows the
N per core is 17 / 5 / 23 tiles, so to_qkv pads 168 columns to 170 and ff1 224 to 230 while 9 rows pad them to
171 and 225: the 11x10 grid buys ff1 2.4% and costs to_qkv 1% and to_out 8.5%.

The rejections in both standalone sweeps are `Statically allocated circular buffers ... clash with L1 buffers`
(`tt_metal/impl/program/program.cpp:2154`) at program creation, recorded as `skipped`; the L1 pre-filter
(`estimate_l1_kb`) let 12 of 3,521 combos through.

### What the numbers say

1. **Keep the fused op.** It beats the best possible unfused pair by 19% (to_qkv), 9% (to_out) and 14% (ff1)
   before the dispatch gap between the two unfused ops is counted; on Wormhole the margins were 11 / 23 / 6%.
   to_out's margin is now the smallest of the three and ff1's the second largest, the reverse of Wormhole, for the
   reason in the next point.
2. **The gather is no longer hidden.** On the same 108 cores the fused op costs more than the standalone matmul
   by 382 us for to_qkv (3562 vs 3180, +12%), 258 us for ff1 (5045 vs 4787, +5.4%) and 1420 us for to_out
   (2668 vs 1248, +114%). On Wormhole the same comparison was 0 to 3% at this M. For to_out the difference equals
   its fabric bound (1469 us): the fused to_out on Blackhole behaves like a serial gather followed by the matmul
   (1248 + 1469 = 2717, measured 2668). For to_qkv and ff1 the ring traffic is still two-thirds to three-quarters
   hidden (the exposed 382 / 258 us against a 1102 us fabric bound).
3. **Fused to_out is fabric-bound, as predicted (prediction 2).** Its best fused time is 1.82x the fabric bound and
   3.0x the compute bound; compute utilisation is 33% against 71% for the standalone matmul on the same grid.
4. **M_block follows the per-core M count (prediction 3).** With 12 worker columns each core holds 36 M tiles;
   every winner uses an M_block that divides 36 (9 for to_qkv and to_out, 6 for ff1 and to_out's rank 2) and the
   shipped M_block 8, which pads 36 to 40 (five blocks, +11% rows delivered), trails by 1.5 to 9.4% fused and by
   3.8 to 17.4% standalone. The ff1 shipped entry loses most: its (8, 3, 14) also carries K_block 3, and every
   ff1 winner uses K_block 6 or 7.
5. **N_block follows the per-core N count (prediction 4).** to_qkv has 19 N tiles per core on 9 rows and wins with
   N_block 10 (two blocks, one padded column) over the shipped 12 (two blocks, five padded columns); to_out has 5
   and wins with N_block 5 exactly (shipped 6 pads one column, 20% of the op's columns); ff1 has 25 gate/up
   columns and wins with N_block 14 (two blocks, 28) over the shipped 14 only because of its M_block and K_block.
6. **All-gather (prediction 7).** `chunks_per_sync` 32 leads as on Wormhole, workers 3 as well; the 8 KB payload
   did not move the optimum but lifted link efficiency from 58% to 93%. The production (3, 16, 2) is within 2%,
   which is what the standalone gathers in the block (`models/tt_dit/parallel/manager.py:933-945`) have on the table.
7. **11x10 vs 12x9 (prediction 5).** The extra row of cores is worth -2.4% (ff1), +1.0% (to_qkv) and +8.5%
   (to_out) to the standalone matmul, against the 4 to 10% gain the 9th row gave on Wormhole: the N padding on 10
   rows eats the 1.9% of extra cores for two of the three shapes. This only matters for the unfused branch, whose
   `AGMM_BLOCK_SIZES` entries sit 6.5 to 17.5% off the 11x10 optimum (ranks 46 to 91).
8. **What to change (blocking only, no kernel change).** Adding M = 13664 rows to the 12x9 table
   (`models/tt_dit/utils/matmul.py:385-388`) with (9, 6, 10, (3, 1)) for to_qkv, (9, 14, 5, (3, 1)) for to_out and
   (6, 7, 14, (2, 2)) for ff1 saves 111 + 41 + 475 = 627 us per block in the unit test (0.8% of the 74.8 ms block,
   about 31 ms per 50-block forward). Everything larger is on the fused op's ring path: fused to_out spends
   1420 us above its standalone matmul, and to_qkv / ff1 382 / 258 us.

## Best measured time vs roofline

`agmm_unit_sweep.py roofline` (`generated/agmm_h3_sweep/roofline_bh_M13664.md`), constants from
`transformer_roofline.py:135-150`: 2048 FLOP/cycle/core at HiFi2 and 1.35 GHz, DRAM 512 GB/s, fabric 2 links x
25 GB/s, ring of 4. Compute bound = 2 M K N / (cores x 2.765 TFLOP/s); DRAM bound = 2 B (M K + K N) / 512 GB/s;
fabric bound = 3/4 of the device's in0 shard over 4 link-directions at 25 GB/s. Cores: 108 for the fused op and
the 12x9 matmul, 110 for 11x10. "compute util" = compute bound / measured; "measured / ideal" is against the largest
bound.

Compute utilisation at the best blocking (Wormhole at the same M in parentheses):

| op | fused 108 cores | matmul 12x9 | matmul 11x10 |
|---|---|---|---|
| to_qkv | 74% (59%) | 83% (59%) | 81% (56%) |
| to_out | 33% (40%) | 71% (49%) | 64% (44%) |
| ff1 | 70% (53%) | 74% (54%) | 74% (50%) |

Standalone all-gather: 1.08x its fabric bound on all three tensors (93% of 25 GB/s per link-direction).

| op | mode | cores | best blocking | measured us | compute us | DRAM us | fabric us | limiter | attained TFLOP/s | compute util | measured / ideal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| to_qkv | fused | 108 | 9, 6, 10, 3, 1 | 3562.0 | 2645.1 | 399.8 | 1101.9 | compute | 221.7 | 74% | 1.35x |
| to_qkv | mm_ring | 108 | 9, 7, 10, 3, 1 | 3180.5 | 2645.1 | 399.8 | - | compute | 248.3 | 83% | 1.20x |
| to_qkv | mm_full | 110 | 13, 6, 6, 1, 3 | 3213.5 | 2597.0 | 399.8 | - | compute | 245.8 | 81% | 1.24x |
| to_qkv | ag | - | 3, 32, 8 | 1191.7 | - | - | 1101.9 | fabric | - | - | 1.08x |
| to_out | fused | 108 | 9, 14, 5, 3, 1 | 2668.3 | 881.7 | 420.2 | 1469.2 | fabric | 98.7 | 33% | 1.82x |
| to_out | mm_ring | 108 | 6, 7, 5, 3, 1 | 1247.9 | 881.7 | 420.2 | - | compute | 211.0 | 71% | 1.42x |
| to_out | mm_full | 110 | 8, 8, 5, 4, 1 | 1354.6 | 865.7 | 420.2 | - | compute | 194.4 | 64% | 1.56x |
| to_out | ag | - | 3, 32, 8 | 1590.2 | - | - | 1469.2 | fabric | - | - | 1.08x |
| ff1 | fused | 108 | 6, 7, 14, 2, 2 | 5044.8 | 3526.8 | 437.5 | 1101.9 | compute | 208.7 | 70% | 1.43x |
| ff1 | mm_ring | 108 | 6, 6, 14, 2, 2 | 4787.2 | 3526.8 | 437.5 | - | compute | 220.0 | 74% | 1.36x |
| ff1 | mm_full | 110 | 13, 6, 8, 1, 4 | 4672.8 | 3462.7 | 437.5 | - | compute | 225.4 | 74% | 1.35x |
| ff1 | ag | - | 3, 32, 8 | 1193.3 | - | - | 1101.9 | fabric | - | - | 1.08x |

What the roofline says:

1. **The standalone matmuls run at 1.20 to 1.56x their compute bound** (64 to 83% utilisation), against 1.7 to
   2.3x (44 to 59%) for the same shapes on Wormhole. DRAM is never closer than 3x (to_out) and mostly 8 to 11x away.
   The remaining gap is per-core pipeline pace plus the padding of the per-core output block, as in the Wormhole
   per-op studies; a zone breakdown on this part has not been measured.
2. **The fused op loses 7 to 38 points of utilisation to the ring traffic**: to_qkv 83 -> 74%, ff1 74 -> 70%,
   to_out 71 -> 33%. On Wormhole the fused and standalone utilisations were equal for to_qkv and ff1 (the gather was
   free) and 9 points apart for to_out. With the compute bound 2.3x shorter and the fabric bound unchanged, the
   fused op's ring path is now the limiter for to_out and a visible cost for the other two.
3. **to_out is fabric-bound, and the fused op does not reach the fabric bound the standalone gather reaches.**
   The standalone gather of the same tensor runs at 1.08x the bound (1590 us); the fused op's excess over its
   standalone matmul is 1420 us, so its gather is at best as fast as the standalone one and completely exposed.
4. **The standalone all-gather is at 93% of link bandwidth** on every tensor, up from 58% on Wormhole; there is
   nothing left in its hyperparameters beyond the 2% of `chunks_per_sync` 32.

## Predictions from the README, checked

| # | prediction | outcome |
|---|---|---|
| 1 | fabric bounds equal to Wormhole in us, compute bounds 2.3x smaller | holds by construction: 1102 / 1469 us fabric, 2645 / 882 / 3527 us compute |
| 2 | fused to_out fabric-bound, time set by the gather, smallest fused-vs-unfused margin, possibly negative; to_qkv / ff1 fused overhead over `mm_ring` grows to several percent | holds except "negative": 2668 us = 1.82x fabric bound, margin -9.4% (smallest); overhead over `mm_ring` 12% / 5.4% |
| 3 | M_block 12 or 9 wins fused, shipped 8 trails 2 to 5% | 9 wins to_qkv and to_out, 6 wins ff1 (all divide 36); shipped trails 3.1 / 1.5 / 9.4% (ff1 also carries K_block 3) |
| 4 | N per core 19 / 5 / 25; ff1 wants 26; shipped ff1 further from rank 1 than Wormhole's 0.1% | N_block 10 / 5 / 14 win (ff1 takes two blocks of 14, not one of 26); shipped ff1 rank 16, +9.4% |
| 5 | 11x10 roughly equal or slower than 12x9 for to_out, Wormhole's 9th-row gain mostly gone | to_out +8.5%, to_qkv +1.0%, ff1 -2.4% |
| 6 | model resolves `AGMM_BLOCK_SIZES` + (2, 2) for all three ops at M = 13664 | confirmed by the sidecars (`shipped_blocks`) |
| 7 | 8 KB payload may move the `chunks_per_sync` optimum | it did not (32 still leads); link efficiency 58% -> 93% |

## Caveats

- Same caveats as the Wormhole study (per-token to_out gate; the unfused pair adds two kernel durations and no
  dispatch gap; the model's unfused to_out would be a plain matmul plus a separate `ttnn.addcmul`; no bias in the
  fused rows; the unit test exercises the TP ring alone).
- Broker cap: each chunk is its own device session, so the JIT cache and the torch reference are rebuilt per chunk;
  this affects wall time only, not the device kernel durations.
- The fused shipped to_out measures 2709 us in the unit test and 2920 us inside the block profile (to_qkv 3673 vs
  3650, ff1 5520 vs 5490): in the model the SP axis carries traffic at the same time, which the unit test does not
  exercise, and it shows on the fabric-bound op.
- The 4x32 mesh (M = 3424, where the 12x9 table already has entries) was not measured; the M_block rule above
  (divide the per-core M count) predicts M_block 9 there too (107 M tiles over 12 columns = 9 per core).
