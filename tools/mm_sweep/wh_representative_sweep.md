# Wormhole Representative Matmul Sweep — Analysis & Results

Sweep of `ttnn.experimental.minimal_matmul` across a representative shape suite on Wormhole (WH Galaxy, 8×8 grid per chip), to characterize performance coverage and (next phase) derive slicing / K-par / blocking heuristics.

**Generated:** 2026-06-26 · **Op build:** branch `cglagovich/minimal-matmul-wh-sweep` (K-par cache-replay fix applied; mcast default-off)

## Methodology

- **Shape suite:** 57 canonical shapes spanning 9 regimes (square, rectangular, few-token, output-starved/GEMV, wide-N/vocab, deep-K, shallow-K, LLM anchors, diffusion anchors). All tile-aligned, `K % 256 == 0` (clean for the `S,Pk ∈ {1,2,4,8}` joint sweep).
- **M↔N dedup:** the op canonicalizes `M > N` via `transpose_core_grid`, making `(M,K,N)` and `(N,K,M)` performance-equivalent. Shapes are deduplicated to canonical `M ≤ N`; two explicit transpose pairs were run in both orientations to validate (below).
- **Baseline (this doc):** our branch with all levers forced off (`BSWEEP_BASELINE=1`: S1/Pk1, no large-N levers, no auto-prefetch, no auto-K-par; mcast off) **+ a full block sweep** per shape. Because every blocking is searched, the per-shape optimum is found directly — the branch's block-sizer heuristic confers no advantage, making this a fair, strong "optimized main" reference on the same build.
- **Profiling:** tt-metal device profiler (on-device `-FW` zone timestamps from `profile_log_device.csv` via `ReadDeviceProfiler`), median of timed reps. Not Tracy host-capture.
- **Parallelism:** multiprocess (1 proc/chip, `TT_VISIBLE_DEVICES`) across 13 freshly-reset idle galaxies × 32 chips, dispatched by `orchestrate_dyn.py`.

**Run:** 2683 configs swept, 2679 PCC-pass, 57 canonical shapes (+ 4 symmetry orientations), 0 dead nodes, 446s wall-clock.

## M↔N symmetry validation

| pair | µs | transpose µs | ratio |
|---|---|---|---|
| 128×4096×8192 ↔ 8192×4096×128 | 804.7 | 801.3 | **0.996** |
| 512×8192×2048 ↔ 2048×8192×512 | 455.2 | 460.7 | **1.012** |

Both within ~1% → transpose-grid symmetry confirmed; canonical-M≤N dedup is justified.

## Baseline performance by category

Geomean utilization per regime (HiFi2, levers off, best blocking). Util = achieved / theoretical-peak MAC throughput.

| regime | shapes | geomean util | read |
|---|---|---|---|
| A. Square (peak MAC) | 5 | 29.5% | util climbs 9%→59% with size; small squares starve the 8×8 grid |
| B. Rectangular | 6 | 48.1% | already healthy on baseline |
| C. Few-token GEMM | 8 | 7.4% | small-M starvation — branch S/Pk target |
| D. Output-starved (GEMV) | 7 | 0.5% | catastrophic on baseline — prime K-par target |
| E. Wide-N / vocab | 6 | 19.9% | low at small M, →59% at M≥1024 |
| F. Deep-K | 6 | 28.8% | scales with output size |
| G. Shallow-K | 5 | 21.9% | output-write bound |
| H. LLM anchors | 8 | 15.2% | decode (M=32) ~2%, prefill (M=2048) 52–57% |
| I. Diffusion anchors | 6 | 21.2% | matches prior FLUX/LTX data (sanity ✓) |
| **Overall** | **57** | **12.7%** | dragged down by starved/decode regimes — branch S/Pk targets |

## Full per-shape baseline optima

Best blocking found per shape: `mb`=M_block, `kb`=K_block, `nb`=N_block tiles; `sb`=subblock h×w.

| regime | shape (M×K×N) | µs | util % | mb | kb | nb | sb |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 512×512×512 | 22.0 | 9.3 | 2 | 4 | 2 | 2×2 |
| A | 1024×1024×1024 | 81.2 | 20.2 | 4 | 4 | 4 | 1×4 |
| A | 2048×2048×2048 | 350.0 | 37.4 | 8 | 4 | 8 | 1×4 |
| A | 4096×4096×4096 | 1967.2 | 53.3 | 6 | 8 | 8 | 1×4 |
| A | 8192×8192×8192 | 14128.4 | 59.4 | 8 | 8 | 8 | 1×4 |
| B | 1024×2048×4096 | 354.6 | 37.0 | 4 | 4 | 8 | 1×4 |
| B | 1024×4096×2048 | 310.5 | 42.2 | 4 | 4 | 8 | 1×4 |
| B | 1024×8192×8192 | 1887.9 | 55.5 | 4 | 8 | 18 | 2×2 |
| B | 2048×2048×8192 | 1080.7 | 48.5 | 4 | 8 | 8 | 1×4 |
| B | 2048×4096×8192 | 1946.8 | 53.9 | 4 | 4 | 18 | 2×2 |
| B | 2048×8192×4096 | 1911.8 | 54.8 | 8 | 8 | 8 | 1×4 |
| C | 32×4096×4096 | 419.4 | 2.0 | 1 | 8 | 3 | 1×1 |
| C | 32×4096×16384 | 1543.3 | 2.1 | 1 | 16 | 3 | 1×1 |
| C | 64×8192×8192 | 1549.4 | 4.2 | 1 | 16 | 3 | 1×1 |
| C | 128×4096×4096 | 421.3 | 7.8 | 1 | 8 | 4 | 1×4 |
| C | 128×8192×8192 | 1556.4 | 8.4 | 1 | 16 | 3 | 1×1 |
| C | 256×4096×4096 | 428.3 | 15.3 | 1 | 8 | 4 | 1×4 |
| C | 256×8192×16384 | 3089.5 | 17.0 | 1 | 16 | 4 | 1×4 |
| C | 512×4096×4096 | 454.1 | 28.9 | 2 | 8 | 4 | 1×4 |
| D | 32×2048×32 | 26.1 | 0.1 | 1 | 4 | 1 | 1×1 |
| D | 32×4096×32 | 41.2 | 0.2 | 1 | 8 | 1 | 1×1 |
| D | 32×8192×32 | 78.0 | 0.2 | 1 | 8 | 1 | 1×1 |
| D | 32×16384×32 | 140.2 | 0.2 | 1 | 8 | 1 | 1×1 |
| D | 64×8192×64 | 81.3 | 0.6 | 1 | 8 | 1 | 1×1 |
| D | 128×16384×128 | 152.2 | 2.7 | 1 | 8 | 1 | 1×1 |
| D | 256×8192×256 | 93.3 | 8.8 | 1 | 8 | 1 | 1×1 |
| E | 64×4096×16384 | 1543.6 | 4.2 | 1 | 16 | 3 | 1×1 |
| E | 128×2048×16384 | 801.8 | 8.2 | 1 | 16 | 3 | 1×1 |
| E | 256×4096×16384 | 1582.5 | 16.6 | 1 | 16 | 3 | 1×1 |
| E | 512×4096×16384 | 1628.0 | 32.2 | 2 | 8 | 12 | 1×4 |
| E | 1024×8192×16384 | 3527.7 | 59.4 | 4 | 8 | 18 | 2×2 |
| E | 2048×4096×16384 | 3717.2 | 56.4 | 8 | 8 | 8 | 1×4 |
| F | 256×16384×1024 | 428.1 | 15.3 | 1 | 8 | 4 | 1×4 |
| F | 512×8192×1024 | 250.7 | 26.1 | 2 | 4 | 4 | 1×4 |
| F | 512×16384×512 | 328.6 | 19.9 | 2 | 4 | 2 | 2×2 |
| F | 512×16384×4096 | 1634.5 | 32.1 | 2 | 8 | 8 | 1×4 |
| F | 1024×16384×1024 | 658.3 | 39.8 | 4 | 4 | 4 | 1×4 |
| F | 2048×16384×2048 | 1891.2 | 55.4 | 8 | 8 | 8 | 1×4 |
| G | 2048×256×8192 | 350.7 | 18.7 | 2 | 8 | 2 | 2×2 |
| G | 2048×512×4096 | 273.8 | 23.9 | 2 | 4 | 8 | 1×4 |
| G | 4096×256×4096 | 393.1 | 16.7 | 2 | 8 | 2 | 2×2 |
| G | 4096×512×8192 | 847.7 | 30.9 | 12 | 16 | 2 | 4×1 |
| G | 8192×256×8192 | 1188.6 | 22.1 | 2 | 8 | 2 | 2×2 |
| H | 32×4096×12288 | 1213.3 | 2.0 | 1 | 8 | 8 | 1×4 |
| H | 32×8192×28672 | 5267.9 | 2.2 | 1 | 16 | 3 | 1×1 |
| H | 32×11008×4096 | 1072.4 | 2.1 | 1 | 8 | 4 | 1×4 |
| H | 512×8192×28672 | 5439.5 | 33.7 | 2 | 16 | 8 | 1×4 |
| H | 2048×4096×11008 | 2701.4 | 52.2 | 4 | 8 | 12 | 1×4 |
| H | 2048×4096×12288 | 2869.7 | 54.8 | 8 | 8 | 8 | 1×4 |
| H | 2048×4096×32000 | 7164.0 | 57.2 | 4 | 8 | 18 | 2×2 |
| H | 2048×11008×4096 | 2499.8 | 56.4 | 8 | 8 | 8 | 1×4 |
| I | 32×6144×4608 | 709.3 | 1.9 | 1 | 4 | 8 | 1×4 |
| I | 128×6144×16384 | 2307.0 | 8.5 | 1 | 16 | 3 | 1×1 |
| I | 512×6144×9216 | 1404.1 | 31.5 | 2 | 8 | 8 | 1×4 |
| I | 2048×6144×4608 | 1645.4 | 53.8 | 4 | 8 | 6 | 2×2 |
| I | 4096×6144×4608 | 3123.2 | 56.7 | 6 | 4 | 18 | 2×2 |
| I | 8256×6144×9216 | 12615.1 | 56.5 | 6 | 8 | 12 | 1×4 |

### Symmetry-validation shapes (both orientations)

| shape (M×K×N) | µs | util % | mb | kb | nb | sb |
|---|---:|---:|---:|---:|---:|---:|
| 128×4096×8192 | 804.7 | 8.1 | 1 | 16 | 3 | 1×1 |
| 8192×4096×128 | 801.3 | 8.2 | 3 | 16 | 1 | 1×1 |
| 512×8192×2048 | 455.2 | 28.8 | 2 | 8 | 4 | 1×4 |
| 2048×8192×512 | 460.7 | 28.4 | 4 | 8 | 2 | 4×1 |

## Next phase (pending)

Branch **joint (S, Pk, blocking) sweep** over the same 57 shapes → per-shape speedup vs this baseline, and heuristic fitting for:
- **N-slicing (S):** expected to help wide-N / few-token (regimes C, E).
- **K-parallelism (Pk):** expected to help output-starved / GEMV / decode (regime D, LLM decode).
- **Blocking:** generalize the per-core-sized block pattern observed above.

_Data: `baseline_v2_out.json` (this doc), shape suite `sweep_v2.json`._

---

# Branch results — joint (S, Pk, blocking) sweep

**Generated:** 2026-06-26 · Run 1 (levers ON): 29,829 cfgs / 29,134 PCC-pass / 61 shapes / 1829s. Run 2 (lever ablation, N≥4096, levers OFF): 26,038 cfgs / 25,380 PCC-pass / 42 shapes / 787s. 0 dead nodes.

**Speedup = baseline best µs ÷ branch best µs** (both at their own optimal blocking; branch additionally free over S∈{1,2,4,8}×Pk∈{1,2,4,8}, S·Pk≤8, large-N levers ON, mcast OFF).

## Headline

- **Overall geomean speedup 1.41× over 57 shapes** (36 wins / 19 ties / 2 losses).
- Branch geomean util 18.0% (baseline 12.7%).
- Wins concentrate exactly where predicted: output-starved/GEMV, few-token, decode, wide-N. Compute-saturated shapes correctly stay at ~1.0× (S1Pk1 already optimal).

## Speedup by category

| regime | shapes | geomean speedup | branch geomean util |
|---|---|---|---|
| A. Square | 5 | **1.01×** | 29.8% |
| B. Rectangular | 6 | **1.02×** | 49.2% |
| C. Few-token GEMM | 8 | **1.90×** | 14.0% |
| D. Output-starved (GEMV) | 7 | **2.36×** | 1.2% |
| E. Wide-N / vocab | 6 | **1.47×** | 29.3% |
| F. Deep-K | 6 | **1.24×** | 35.5% |
| G. Shallow-K | 5 | **1.10×** | 24.1% |
| H. LLM anchors | 8 | **1.37×** | 20.8% |
| I. Diffusion anchors | 6 | **1.35×** | 28.5% |
| **Overall** | **57** | **1.41×** | **18.0%** |

## Full per-shape branch results

Best (S,Pk) and resulting speedup vs baseline. `lever` = levers-ON ÷ levers-OFF for N≥4096 (>1 ⇒ large-N levers help).

| regime | shape (M×K×N) | baseline µs | branch µs | speedup | best (S,Pk) | branch util % | lever |
|---|---|---:|---:|---:|:--:|---:|---:|
| A | 1024×1024×1024 | 81.2 | 72.2 | **1.13×** | S1Pk1 | 22.7 | — |
| A | 4096×4096×4096 | 1967.2 | 1935.4 | **1.02×** | S1Pk1 | 54.2 | 1.02× |
| A | 2048×2048×2048 | 350.0 | 350.3 | **1.00×** | S1Pk1 | 37.4 | — |
| A | 8192×8192×8192 | 14128.4 | 14177.1 | **1.00×** | S1Pk1 | 59.2 | 1.00× |
| A | 512×512×512 | 22.0 | 23.5 | **0.94×** | S1Pk1 | 8.7 | — |
| B | 1024×2048×4096 | 354.6 | 324.8 | **1.09×** | S1Pk1 | 40.4 | 1.07× |
| B | 2048×2048×8192 | 1080.7 | 1048.8 | **1.03×** | S2Pk1 | 50.0 | 1.03× |
| B | 2048×4096×8192 | 1946.8 | 1916.5 | **1.02×** | S1Pk1 | 54.7 | 1.02× |
| B | 2048×8192×4096 | 1911.8 | 1899.1 | **1.01×** | S2Pk1 | 55.2 | 1.01× |
| B | 1024×4096×2048 | 310.5 | 311.5 | **1.00×** | S1Pk1 | 42.1 | — |
| B | 1024×8192×8192 | 1887.9 | 1895.8 | **1.00×** | S1Pk1 | 55.3 | 1.00× |
| C | 32×4096×16384 | 1543.3 | 724.7 | **2.13×** | S2Pk4 | 4.5 | 0.99× |
| C | 128×4096×4096 | 421.3 | 199.8 | **2.11×** | S8Pk1 | 16.4 | 1.05× |
| C | 32×4096×4096 | 419.4 | 202.1 | **2.07×** | S4Pk2 | 4.1 | 0.98× |
| C | 64×8192×8192 | 1549.4 | 774.1 | **2.00×** | S2Pk2 | 8.5 | 0.99× |
| C | 128×8192×8192 | 1556.4 | 787.3 | **1.98×** | S2Pk2 | 16.6 | 0.99× |
| C | 256×8192×16384 | 3089.5 | 1651.2 | **1.87×** | S2Pk2 | 31.8 | 0.99× |
| C | 256×4096×4096 | 428.3 | 237.3 | **1.80×** | S2Pk2 | 27.6 | 1.01× |
| C | 512×4096×4096 | 454.1 | 337.1 | **1.35×** | S2Pk2 | 38.9 | 0.97× |
| D | 32×16384×32 | 140.2 | 32.3 | **4.34×** | S1Pk8 | 0.8 | — |
| D | 32×8192×32 | 78.0 | 21.1 | **3.70×** | S1Pk8 | 0.6 | — |
| D | 32×4096×32 | 41.2 | 14.8 | **2.79×** | S1Pk8 | 0.4 | — |
| D | 64×8192×64 | 81.3 | 34.7 | **2.34×** | S1Pk4 | 1.5 | — |
| D | 32×2048×32 | 26.1 | 12.0 | **2.18×** | S1Pk8 | 0.3 | — |
| D | 128×16384×128 | 152.2 | 87.7 | **1.74×** | S1Pk2 | 4.7 | — |
| D | 256×8192×256 | 93.3 | 90.8 | **1.03×** | S1Pk2 | 9.0 | — |
| E | 64×4096×16384 | 1543.6 | 760.7 | **2.03×** | S2Pk4 | 8.6 | 0.99× |
| E | 128×2048×16384 | 801.8 | 412.4 | **1.94×** | S1Pk8 | 15.9 | 0.95× |
| E | 256×4096×16384 | 1582.5 | 862.5 | **1.83×** | S2Pk2 | 30.4 | 0.97× |
| E | 512×4096×16384 | 1628.0 | 1160.5 | **1.40×** | S2Pk2 | 45.2 | 1.01× |
| E | 2048×4096×16384 | 3717.2 | 3673.4 | **1.01×** | S2Pk1 | 57.1 | 1.01× |
| E | 1024×8192×16384 | 3527.7 | 3573.9 | **0.99×** | S1Pk1 | 58.7 | 0.99× |
| F | 256×16384×1024 | 428.1 | 251.2 | **1.70×** | S1Pk4 | 26.1 | — |
| F | 512×16384×4096 | 1634.5 | 1019.6 | **1.60×** | S2Pk2 | 51.4 | 1.00× |
| F | 512×8192×1024 | 250.7 | 194.2 | **1.29×** | S1Pk2 | 33.7 | — |
| F | 512×16384×512 | 328.6 | 326.0 | **1.01×** | S1Pk1 | 20.1 | — |
| F | 2048×16384×2048 | 1891.2 | 1889.0 | **1.00×** | S1Pk1 | 55.5 | — |
| F | 1024×16384×1024 | 658.3 | 659.4 | **1.00×** | S1Pk1 | 39.8 | — |
| G | 4096×256×4096 | 393.1 | 336.7 | **1.17×** | S1Pk1 | 19.5 | 1.15× |
| G | 2048×256×8192 | 350.7 | 310.8 | **1.13×** | S1Pk1 | 21.1 | 1.15× |
| G | 4096×512×8192 | 847.7 | 780.8 | **1.09×** | S1Pk1 | 33.6 | 1.04× |
| G | 2048×512×4096 | 273.8 | 258.4 | **1.06×** | S1Pk1 | 25.4 | 1.03× |
| G | 8192×256×8192 | 1188.6 | 1127.7 | **1.05×** | S1Pk1 | 23.2 | 1.06× |
| H | 32×8192×28672 | 5267.9 | 2384.3 | **2.21×** | S2Pk4 | 4.8 | 0.99× |
| H | 32×11008×4096 | 1072.4 | 507.2 | **2.11×** | S4Pk2 | 4.3 | 0.98× |
| H | 32×4096×12288 | 1213.3 | 698.1 | **1.74×** | S2Pk2 | 3.5 | 0.98× |
| H | 512×8192×28672 | 5439.5 | 3635.4 | **1.50×** | S2Pk2 | 50.5 | 1.02× |
| H | 2048×4096×11008 | 2701.4 | 2655.7 | **1.02×** | S1Pk1 | 53.1 | 1.02× |
| H | 2048×4096×12288 | 2869.7 | 2843.7 | **1.01×** | S1Pk1 | 55.3 | 1.01× |
| H | 2048×4096×32000 | 7164.0 | 7171.1 | **1.00×** | S1Pk1 | 57.1 | 1.00× |
| H | 2048×11008×4096 | 2499.8 | 2506.1 | **1.00×** | S1Pk1 | 56.2 | 1.00× |
| I | 32×6144×4608 | 709.3 | 318.8 | **2.22×** | S4Pk2 | 4.3 | 1.02× |
| I | 128×6144×16384 | 2307.0 | 1137.3 | **2.03×** | S2Pk4 | 17.3 | 1.01× |
| I | 512×6144×9216 | 1404.1 | 999.0 | **1.41×** | S2Pk1 | 44.3 | 0.97× |
| I | 8256×6144×9216 | 12615.1 | 12673.1 | **1.00×** | S1Pk1 | 56.3 | 1.00× |
| I | 4096×6144×4608 | 3123.2 | 3182.6 | **0.98×** | S1Pk1 | 55.6 | 0.98× |
| I | 2048×6144×4608 | 1645.4 | 1709.2 | **0.96×** | S1Pk1 | 51.8 | 0.96× |

## (S, Pk) selection — heuristic signal

Winning (S,Pk) across the 57 shapes:

| (S,Pk) | count | where it wins |
|---|---|---|
| S1Pk1 | 25 | compute-saturated (large M·N): square≥2048, rect/deep-K/diffusion/LLM-prefill |
| S1Pk2 | 3 | moderate output, deep-ish K |
| S1Pk4 | 2 | small output, deep K |
| S1Pk8 | 5 | tiny output + very deep K (GEMV: 32×{8192,16384}×32, decode) |
| S2Pk1 | 4 | large wide output (N-slice only) |
| S2Pk2 | 10 | few-token & moderate wide-N — the workhorse |
| S2Pk4 | 4 | few-token + wide-N + deep K |
| S4Pk2 | 3 | very skewed small-M (32×K×N decode / 32×6144×4608) |
| S8Pk1 | 1 | moderate square-ish small-M (128×4096×4096) |

**Derived rules:**
- **No slicing (S1Pk1)** once output saturates the grid (≈ M·N tiles ≥ 4×cores *and* both ≥ grid dim): all large square/rect/deep-K/diffusion/LLM-prefill. Slicing/K-par can't beat a saturated grid.
- **K-parallel (Pk↑)** as output shrinks and K deepens: tiny output (out ≤ cores) → Pk 4–8; the 32×K×32 ladder goes S1Pk8 by K=8192. Decode (M=32) → Pk 2–4.
- **N-slice (S↑)** for wide-N with few rows; usually paired as **S2Pk2** for few-token GEMM (the single most common winner, 10 shapes).
- **Crossover:** as M grows 32→2048 at fixed wide N (regime E), the winner walks S2Pk4 → S2Pk2 → S2Pk1 → S1Pk1 — i.e. less parallelism as rows fill the grid.

## Large-N lever contribution (ablation)

Over 41 wide shapes (N≥4096): **geomean 1.008×** — essentially neutral on aggregate (8 helped >3%, 28 neutral, 5 hurt >3%).

- **Where they help (keep ON):** shallow-K wide shapes — 4096×256×4096 and 2048×256×8192 both **+15%**, 8192×256×8192 +6% (low arithmetic intensity → output-write/DRAM-contention bound, exactly what the levers target).
- **Where they mildly hurt:** a few wide bf16-heavy shapes (128×2048×16384 −5%, 2048×6144×4608 −4%).
- **Verdict:** net slightly positive and concentrated on the shapes that need it; **leave the large-N levers ON by default** (matches current branch behavior). Ablation confirmed effective: 0/42 wide shapes had identical best with vs without levers.

_Data: `joint_v2_out.json` (branch, levers on), `joint_v2_nolev_out.json` (levers off), `baseline_v2_out.json` (baseline). Shapes: `sweep_v2.json` / `sweep_v2_wide.json`._
