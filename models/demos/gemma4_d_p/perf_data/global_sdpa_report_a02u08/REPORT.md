# Gemma4-31B prefill: global attention improvements at 2K / 4K / 8K chunks

Machine: bh-glx-110-a02u08 (BH Galaxy 8x4, CP8 x TP4), Slurm job 140396, 2026-10-21.
Test: `text_demo_prefill.py::test_prefill_long_context_traced[blackhole-ctx_256k-chunk{N}-text-8x4]` (one replayed
trace per chunk size). Every number is the mean of two runs on this machine; run-to-run spread is in parentheses.
Contexts are rounded up to whole chunks (at 8K, "100k" is 13 chunks = 106,496 tokens). "First chunk" is the
first chunk's device time.

## Variants (all branches on main `909748641c1`, pushed to origin, no PRs)

| Variant | Branch | What it changes |
|---|---|---|
| main | `909748641c1` | Baseline |
| v-from-k | `mmanzoor/gemma4-gsdpa-v-from-k` | ring_joint_sdpa reader builds 12 of V's 16 tile columns from the K tiles already in L1; reads only V's tail from DRAM. All chunk sizes, bit-exact |
| v-from-k-overlap | `mmanzoor/gemma4-gsdpa-v-from-k-overlap` | v-from-k + V reads issued before the K multicast. Slower, and **fails at 8K** (program 928 B over the kernel config buffer). Record only |
| ksplit8-q128 | `mmanzoor/gemma4-gsdpa-ksplit8-q128` | 8-band K split (Kyle's c4142e621c8), Gemma4 L1_SMALL, chunk-2048 global attention at q128 x 5 bands. Changes only the 2K layout |
| ksplit8-q128 + v-from-k | `mmanzoor/gemma4-gsdpa-ksplit8-q128-v-from-k` | Both stacked |

## Device time (ms)

| Chunk | Variant | First chunk | 100k | 256k | 256k vs main |
|---:|---|---:|---:|---:|---:|
| 2K | main | 53.4 | 3179.6 (±0.9) | 10271.8 (±1.8) | — |
| 2K | v-from-k | 53.4 | **3136.4** (±0.5) | 9989.9 (±0.4) | -2.7% |
| 2K | v-from-k-overlap | 53.4 | 3145.8 | 10025.7 (±0.6) | -2.4% |
| 2K | ksplit8-q128 | 54.0 | 3153.7 (±0.7) | 9882.7 (±0.1) | -3.8% |
| 2K | **ksplit8-q128 + v-from-k** | 54.0 | 3152.0 (±0.6) | **9875.7** (±1.1) | **-3.9%** |
| 4K | main | 74.2 | 2205.2 (±0.8) | 7140.7 | — |
| 4K | v-from-k | 74.2 | 2204.3 (±0.7) | 7136.6 (±0.2) | -0.1% |
| 4K | v-from-k-overlap | 74.2 | 2209.7 | 7155.1 (±0.7) | +0.2% |
| 4K | ksplit8-q128 | 74.2 | 2206.8 (±0.4) | 7151.1 (±7.0) | +0.1% |
| 4K | ksplit8-q128 + v-from-k | 74.3 | 2209.0 (±1.7) | 7150.1 (±6.0) | +0.1% |
| 8K | main | 112.2 | 1876.5 (±2.0) | 6371.6 (±0.3) | — |
| 8K | v-from-k | 112.2 | 1869.1 (±0.6) | 6315.9 (±6.8) | -0.9% |
| 8K | v-from-k-overlap | fails | fails | fails | — |
| 8K | ksplit8-q128 | 112.3 | 1878.7 (±0.1) | 6377.6 (±0.8) | +0.1% |
| 8K | **ksplit8-q128 + v-from-k** | 112.3 | **1868.8** (±0.5) | **6315.2** (±2.9) | **-0.9%** |

## Wall time (ms; device + host staging, noisier)

| Chunk | Variant | First chunk | 100k | 256k | 256k vs main |
|---:|---|---:|---:|---:|---:|
| 2K | main | 55.7 | 3343.3 | 10684.8 | — |
| 2K | v-from-k | 55.8 | 3277.1 | 10384.8 | -2.8% |
| 2K | ksplit8-q128 | 56.5 | 3288.6 | 10221.2 | -4.3% |
| 2K | ksplit8-q128 + v-from-k | 57.0 | 3306.6 | 10268.1 | -3.9% |
| 4K | main | 77.0 | 2289.7 | 7355.1 | — |
| 4K | ksplit8-q128 + v-from-k | 76.7 | 2277.4 | 7330.6 | -0.3% |
| 8K | main | 113.7 | 1915.2 | 6472.9 | — |
| 8K | v-from-k | 113.7 | 1913.4 | 6420.0 | -0.8% |
| 8K | ksplit8-q128 + v-from-k | 113.8 | 1905.5 | 6438.1 | -0.5% |

## Accuracy

KV-cache PCC, `test_prefill_migration[mock-256k]` (thresholds: overall >= 0.978, min per head >= 0.928):

| Chunk | main | ksplit8-q128 | v-from-k | ksplit8-q128 + v-from-k |
|---:|---|---|---|---|
| 2K | 0.982643 / 0.941301 | 0.983214 / 0.943989 (b08u08) | 0.982643 / 0.941301 (b08u08) | 0.983214 / 0.943989 |
| 4K | 0.980254 / 0.935430 | identical to main | identical to main | identical to main |
| 8K | 0.980850 / 0.938811 | identical to main | identical to main | identical to main |

All pass. q128 at 2K is slightly more accurate than main; everything else is identical (v-from-k is bit-exact).
SDPA Gemma4 global accuracy tests (packed-K/V K-split, LoFi incl. the q128 x 5-band cases, segmented): 9/9 on the
combined branch.

## Reading the results

- **Best overall: ksplit8-q128 + v-from-k.** -3.9% at 2K, -0.9% at 8K, neutral at 4K, never slower beyond noise.
- **2K:** the q128 x 5-band layout carries the gain at depth (-3.8% at 256k). v-from-k alone is better up to ~100k
  (3136 vs 3154 ms) because q128 costs ~65 us more per global call while the history is short (first chunk 53.4 ->
  54.0 ms); past ~25k tokens q128 wins. Stacking v-from-k on q128 adds little at 2K (q128 is compute-bound).
- **4K:** nothing moves. The 4K layout (q128, 3 bands) is already compute-bound, and neither branch changes it.
- **8K:** v-from-k gives -0.9% (the q96 unsplit layout is partly waiting on K/V reads); ksplit8-q128 is neutral.
- **First chunk:** unchanged except +0.6 ms at 2K with q128.

## Not tested

- Partial readiness (hides the ~150 us first-shard stall at 2K with q128, ~6% of global attention): the port
  deadlocks on main and is not fixed, so it was not run.
- Lazy rescale (up to ~10% of global attention): not implemented.

## Files

`results.txt` (raw log), `<variant>_c<chunk>_r<rep>.log` (per run), `*_kvpcc_c*.log`, `combo_acc.log`,
`report.py` (regenerates the tables), `runner.sh` / `queue.txt`.
