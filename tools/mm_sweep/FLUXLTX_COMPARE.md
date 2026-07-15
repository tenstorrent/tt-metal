# FLUX / LTX real-model shapes — regime_a_matmul (latest picker) vs main & branch

Op = `ttnn.experimental.regime_a_matmul` **auto-picker (config=None)**, latest table (commits up to ab1acb7c871). `main` and `branch` are the historical minimal_matmul numbers from `bh_skinny_results.json`: **main** = the *main optimized baseline* (plain unicast, best-swept blocks, all branch levers gated + TT_MM_NO_LARGE_LEVERS=1 — verified == main bit-for-bit on the dataflow path); **branch** = the minimal_matmul production auto path.

> BW conventions are NOT mixed: **op % is of 512 GB/s**, **main/branch % is of 500 GB/s**. Compare across sources by **kernel µs only** (the speedup columns). Model labels + the regime-A set are from SMALL_MT_IMPL_PLAN.md.

| model | shape | Mt | op cfg (Ns,Pk,Sm,kb,nsb) | op us | op %512 | branch us | branch %500 | main us | main %500 | op vs branch | op vs main |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FLUX | 32x256x6144 | 1 | [3, 1, 1, 1, 8] | 8.6 | 80.4 | 11.2 | 63 | 23.9 | 30 | +30% | +177% |
| FLUX | 32x6144x1536 | 1 | [1, 6, 1, 4, 2] | 42.4 | 89.1 | 51.9 | 75 | 105.9 | 37 | +22% | +150% |
| FLUX | 32x6144x2304 | 1 | [1, 4, 1, 2, 9] | 61.5 | 91.6 | 78.3 | 74 | 136.3 | 42 | +27% | +122% |
| FLUX | 32x6144x4608 | 1 | [1, 12, 1, 2, 1] | 119.3 | 93.9 | - | - | - | - | - | - |
| FLUX | 32x6144x6144 | 1 | [1, 6, 1, 4, 2] | 154.3 | 96.5 | 199.6 | 76 | 308.3 | 50 | +29% | +100% |
| FLUX | 512x6144x768 | 16 | [1, 12, 1, 2, 1] | 98.7 | 32.7 | - | - | - | - | - | - |
| FLUX | 512x15360x768 | 16 | None | - | - | - | - | - | - | - | - |
| FLUX | 512x6144x2304 | 16 | [1, 12, 1, 2, 1] | 149.1 | 48.4 | - | - | - | - | - | - |
| FLUX | 512x2304x6144 | 16 | [4, 3, 1, 1, 1] | 196.9 | 36.7 | - | - | - | - | - | - |
| FLUX | 512x3072x6144 | 16 | [2, 6, 1, 2, 1] | 187.7 | 49.1 | - | - | - | - | - | - |
| FLUX | 512x6144x4608 | 16 | [1, 12, 1, 2, 1] | 225.3 | 58.6 | - | - | - | - | - | - |
| LTX | 32x2048x512 | 1 | [2, 4, 1, 2, 1] | 8.6 | 51.1 | 9.4 | 48 | 22.7 | 20 | +9% | +163% |
| LTX | 32x2048x1536 | 1 | [2, 2, 1, 4, 3] | 15.2 | 83.8 | 18.8 | 69 | 40.6 | 32 | +24% | +167% |
| LTX | 32x2048x2048 | 1 | [2, 2, 1, 4, 4] | 19.6 | 86.1 | 23.1 | 75 | 46.2 | 37 | +18% | +136% |
| LTX | 256x2048x1024 | 8 | [1, 4, 2, 2, 2] | 30.3 | 37.2 | - | - | - | - | - | - |

**Aggregate (shapes with historical numbers, by µs):** op is **1.23x** vs branch, **2.43x** vs main, over the 7 Mt=1 FLUX/LTX shapes.

**No historical main/branch exist for the large-Mt set** (FLUX M=512 Mt16 ×6, LTX 256x2048x1024 Mt8): these were the OOM/GAP shapes not in the minimal_matmul bh_skinny sweep, so main/branch are '-'. The op RUNS all of them correctly; 256x2048x1024 is at 37% (structural ceiling, see the main report), and the Mt16 FLUX set is diagnostic-only (out of the Mt<=8 acceptance scope) at 33-59%.
