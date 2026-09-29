# Pavlo's table at our conditions: [4,2] (SP=4, TP=2, EP=8)

One (4,2) stage (rows 0-3, cols 0-1 of the galaxy), layers 0-6 contiguous, real prefix. Single prose request at
h = 139,264, sparse layers 3-6. ND-sharded experts + hybrid 128, v1 dispatch/combine, 1d fabric. KV correct (PCC
>= 0.9985 vs (2,4), `kv_4x2_status.md`). Per-head KV copies are a harness workaround: "native" removes them from ag_kv.

eff = roof / (chip-mean ms - floor), Pavlo's form. Target = Pavlo's target. Sources: `goodput_results.md` section 1,
`compare_4x2_vs_2x4.txt`, `per_op_4x2.csv`.

| op | [4,2] eff W4096 | [4,2] eff W8192 | Pavlo [4,2] | [2,4] eff W4096 / W8192 | target | [4,2] share W4096 / W8192 | worst-ms ratio [4,2]/[2,4] W4096 / W8192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| norm_ag | 54.2% | 47.3% | 51.1% | 40.9 / 40.1% | 80% | 2 / 3% | 0.28 / 0.29 |
| qkv | 72.0% | 80.3% | 87.2% | 79.6 / 87.3% | 70% | 2 / 2% | 1.10 / 1.09 |
| idx_branch | 5.7% | 6.6% | 6.2% | 3.4 / 3.6% | 70% | | 0.61 / 0.55 |
| o_proj | 56.5% | 66.2% | 66.7% | 56.5 / 58.8% | 70% | | 1.0 / 0.89 |
| attn_rs | 45.4% | 43.0% | 44.6% | 42.7 / 43.0% | 80% | 2 / 2% | 0.43 / 0.38 |
| shared | 49.3% | 53.7% | 52.7% | 41.9 / 43.5% | 70% | 4 / 4% | 0.58 / 0.55 |
| router | 5.5% | 6.0% | 5.0% | 6.0 / 6.2% | 70% | | 0.57 / 0.54 |
| dispatch | 27.3% | 27.1% | 29.1% | 16.4 / 16.1% | 80% | 8 / 9% | 0.98 / 0.97 |
| experts | 78.1% | 74.8% | 44.3% | 76.9 / 74.5% | 70% | 19 / 17% | 0.99 / 1.00 |
| combine | 33.2% | 34.1% | 22.7% | 38.0 / 39.3% | 80% | 18 / 20% | 1.26 / 1.28 |
| moe_reduce | 16.1% | 16.1% | 17.6% | 17.0 / 17.4% | 80% | 12 / 14% | 0.65 / 0.67 |
| ag_kv (with copies / native) | 30.1 / 41.1% | 28.0 / 37.1% | 26.7% | 45.4 / 33.7% | 80% | 15 / 10% | 4.29 / 2.98 (native ~3.2 / 2.3) |
| ag_idx | 46.8% | 46.8% | 100% (clamped) | 54.1 / 53.9% | 80% | 3 / 2% | 1.6 / 1.6 |
| indexer | 2.1% | 1.9% | 4.2% | 4.1 / 3.7% | 70% | | ~1.0 |
| sparse | 4.1% | 3.9% | 3.9% | 4.0 / 3.6% | 70% | 21 / 24% | 0.99 / 0.92 |
| misc | 11.1% | 14.3% | 12.0% | 11.2 / 11.4% | 80% | | |
| dense ring_c | 31.2% | 35.0% | 35.5% | 37.4 / 35.0% | 70% | | 1.20 / 1.00 |
| **sparse layer, worst chip** | 14.22 ms | 24.39 ms | | 14.40 / 27.00 ms | | | 0.99 / 0.90 |

- The TP collectives are 2-3.5x cheaper on 2 chips. qkv is ~10% slower: each chip reads twice the weights.
- ag_kv is ~3x: each chip gathers 3/4 of the context for 2 heads instead of 1/2 for 1 head. The native gather removes
  only the copies (0.53 ms/layer at W4096 h=139k, 6.2 ms at W8192 packed).
- Dispatch/combine run Linear over 4 hops: dispatch costs the same ms at a 2x roofline, combine is 1.26x.
- The dense ring has a row floor near 2048 rows per chip: 1.20x at W4096 (1024 rows), 1.00x at W8192, 1.32x in packed
  2048-token segments (512 rows).
- Depth: at h=548,864 [4,2] sparse chip efficiency is 27% (W4096) / 21% (W8192) worse, driven by ag_kv.
