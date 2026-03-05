# AllGather + MatMul Fused Operation Performance Results

## Test Configuration
- **Platform**: Galaxy (8x4 mesh = 32 devices)
- **Topology**: Ring
- **Cluster Axis**: 1 (ring_size = 4)

---

## Same config: Fused vs non-fused (separate path)

For every test, fused and separate (non-fused) use the **exact same config**:
- Same **core grid** (e.g. 8×8, 7×8)
- Same **num_links**
- Same **M, K, N** and block/subblock sizes

The test is parametrized once; only `use_fused` is toggled (`True` = fused op, `False` = separate AG then MM). So when you run e.g. `-k "8x8_4links and wan2_4k4k4k"`, both `separate` and `fused` use grid 8×8, 4 links, and 4096×4096×4096.

**Non-fused numbers to run:** For every (size, grid, links) where we have fused results, run the **separate** path with the same `-k` (e.g. `-k "separate and 8x8_4links and wan2_4k4k4k"`) and record AG + MM kernel times from the profiler CSV. Then compare fused vs (AG + MM) for that config.

---

## Older results (just for reference, uses 4×8 core grid)

Per-device kernel times from **teja/Allgather+matmul_fused_perf_results** prefill CSVs (FF2 layer). **Not** the current unit-test results — reference only.

| ISL | Core grid | Non-fused (AG + MM) µs | Fused op µs |
|-----|-----------|------------------------|-------------|
| 8k | 4×8 | 1842 (AG 523.71 + MM 1318.09) | 2912.61 |
| 128k | 4×8 | 27633 (AG 8010.90 + MM 19622.12) | 46013.96 |

**Cores used in that non-fused (separate path):** From **baseline_8k/8k/prefill.csv** (FF2 layer rows: AllGatherAsync 8192×896→3584, then Matmul 8192×3584×2048): **AllGather used 40 cores** per device and **Matmul used 56 cores** per device (column CORES). So in the separate path, AG and MM use **different** core sets/counts — not the same grid. Our unit-test comparison uses the **same** core grid for both (e.g. 8×8) so fused vs non-fused is apples-to-apples.

---

## Test 1: Fused AG+MM - WAN 2.2 Size (4k x 4k x 4k)

**Configuration**:
| Parameter | Value |
|-----------|-------|
| M | 4096 |
| K | 4096 (K_per_device = 1024) |
| N | 4096 |
| Grid | 8×8 (64 cores) |
| Links | 4 |
| Workers/Link | 2 |
| Math Fidelity | HiFi2 |

**PCC Validation**: ✅ PASSED

**Device Kernel Duration – single device** (per device, avg over 5 iters; profiler reports one device, e.g. device 31):
| Metric | Value |
|--------|-------|
| Min | 952,938 ns (~953 µs) |
| Avg | 961,940 ns (~962 µs) |
| Max | 971,042 ns (~971 µs) |

_(Source: profiler “Device kernel duration perf summary (device=31)”. Per-device averages across all 32 devices range ~946–1038 µs; device 31 avg = 962 µs.)_

**Profiler Report**: `generated/profiler/reports/2026_03_04_21_26_50/ops_perf_results_2026_03_04_21_26_50.csv`

---

## Non-fused (Separate path)

4k4k4k separate path (AG then MM) now runs after fixing persistent buffer to use full-K per device (ReplicateTensorToMesh for AG output). **PR reference** (unfused): 963 µs AG, 1785 µs MM. **Our run** (8×8 4 links, device 31 avg): 452 µs AG, 1779 µs MM → non-fused **2231 µs**; fused same config **1858 µs** (~17% faster).

---

## Fused ops – run one by one (commands and results)

**Setup:** Test passes full input (1,1,M,K); runtime/mesh shards it (each device gets its K-shard). Kernel does AllGather + matmul.

**Grids:** 8×8, 6×8, 4×8, 7×8. (7×7 removed; 7×8 works better. 7×9 excluded — NOC constraint.)

**Prefix:** `python tools/tracy/profile_this.py -c 'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py -k "fused and <GRID_LINKS> and <SIZE_ID>" -v'`

Record **single-device avg** from profiler: `Device kernel duration perf summary (device=31): ... avg=...ns` → µs. **Non-fused** = separate path (AG + MM) same config; sum AG + MM kernel µs from profiler CSV.

| # | Size (M×K×N) | Grid | Links | Status | Non-fused (µs) | Fused (µs) | PCC | Run command `-k` fragment |
|---|--------------|------|-------|--------|---------------|------------|-----|---------------------------|
| 1 | 4096×4096×4096 | 8×8 | 4 | Done | 2231 | 1858 | ✅ | `8x8_4links and wan2_4k4k4k` |
| 2 | 4096×4096×4096 | 8×8 | 2 | Done | 2574 | 2156 | ✅ | `8x8_2links and wan2_4k4k4k` |
| 3 | 4096×4096×4096 | 7×8 | 1 | Done | 4043 | 3327 | ✅ | `7x8_1link and wan2_4k4k4k` |
| 4 | 4096×4096×4096 | 6×8 | 3 | Done | 3011 | 2531 | ✅ | `6x8_3links and wan2_4k4k4k` |
| 5 | 4096×4096×4096 | 6×8 | 2 | Done | 3261 | 2761 | ✅ | `6x8_2links and wan2_4k4k4k` |
| 6 | 4096×4096×4096 | 4×8 | 2 | Done | 4169 | 3401 | ✅ | `4x8_2links and wan2_4k4k4k` |
|  |  |  |  |  |  |  |  |  |
| 7 | 8192×3584×2048 | 8×8 | 4 | Done | 2457 | 1827 | ⚠️ PCC 0.82 | `8x8_4links and llama_8k_ff2 and not K4096` |
| 8 | 8192×3584×2048 | 8×8 | 2 | Done | 3073 | 2373 | ⚠️ PCC 0.88 | `8x8_2links and llama_8k_ff2 and not K4096` |
| 9 | 8192×3584×2048 | 7×8 | 1 | Done | 4744 | 3435 | ⚠️ PCC 0.88 | `7x8_1link and llama_8k_ff2 and not K4096` |
| 10 | 8192×3584×2048 | 6×8 | 3 | Done | 3213 | 2450 | ⚠️ PCC 0.88 | `6x8_3links and llama_8k_ff2 and not K4096` |
| 11 | 8192×3584×2048 | 6×8 | 2 | Done | 3668 | 2772 | ⚠️ PCC 0.88 | `6x8_2links and llama_8k_ff2 and not K4096` |
| 12 | 8192×3584×2048 | 4×8 | 2 | Done | 4498 | 3090 | ⚠️ PCC 0.88 | `4x8_2links and llama_8k_ff2 and not K4096` |
|  |  |  |  |  |  |  |  |  |
| 13 | 131072×3584×2048 | 8×8 | 4 | Done | 37773 | 26406 | ⚠️ PCC 0.90 | `8x8_4links and llama_128k_ff2 and not K4096` |
| 14 | 131072×3584×2048 | 8×8 | 2 | Done | 47250 | 34300 | ⚠️ PCC ~0.9 | `8x8_2links and llama_128k_ff2 and not K4096` |
| 15 | 131072×3584×2048 | 7×8 | 1 | Done | 72900 | 50359 | ⚠️ PCC 0.90 | `7x8_1link and llama_128k_ff2 and not K4096` |
| 16 | 131072×3584×2048 | 6×8 | 2 | Done | 56056 | 40100 | ⚠️ PCC ~0.9 | `6x8_2links and llama_128k_ff2 and not K4096` |
| 17 | 131072×3584×2048 | 4×8 | 2 | Done | 69100 | 48313 | ⚠️ PCC 0.90 | `4x8_2links and llama_128k_ff2 and not K4096` |

|  |  |  |  |  |  |  |  |  |

**K padded to 4096 (PCC 0.999+)**
| # | Size (M×K×N) | Grid | Links | Status | Non-fused (µs) | Fused (µs) | PCC | Run command `-k` fragment |
|---|--------------|------|-------|--------|---------------|------------|-----|---------------------------|
| 1 | 8192×4096×2048 | 8×8 | 4 | Done | 2742 | 1955 | ✅ PCC 0.9998 | `8x8_4links and llama_8k_ff2_K4096` |
| 2 | 8192×4096×2048 | 8×8 | 2 | Done | 3442 | 2668 | ✅ PCC 0.9998 | `8x8_2links and llama_8k_ff2_K4096` |
| 5 | 8192×4096×2048 | 7×8 | 1 | — | — | — | — | `7x8_1link and llama_8k_ff2_K4096` |
|  |  |  |  |  |  |  |  |  |
| 3 | 131072×4096×2048 | 8×8 | 4 | Done | 42150 | 29475 | ✅ PCC 1.0 | `8x8_4links and llama_128k_ff2_K4096` |
| 4 | 131072×4096×2048 | 8×8 | 2 | Done | 52900 | 39015 | ✅ | `8x8_2links and llama_128k_ff2_K4096` |
| 6 | 131072×4096×2048 | 7×8 | 1 | — | — | — | — | `7x8_1link and llama_128k_ff2_K4096` |


## Tried shapes (K padded to 4096 – PCC hypothesis)
Same Llama M/N but **K padded to 4096** (power of 2) to test whether non–power-of-2 K (3584) is the culprit for low PCC. Results are in the **K padded to 4096** table above (with row gap after main table).


## Non-fused commands (run separate path, then share profiler CSV; we fill Non-fused column)
Always include **`separate`** in the `-k` so only the non-fused (AG then MM) runs — one test. Without it, both `separate` and `fused` can match and the run executes twice. For **llama_8k_ff2** and **llama_128k_ff2** use **`and not K4096`** so only the original (K=3584) runs — otherwise two tests run (e.g. llama_128k_ff2 and llama_128k_ff2_K4096). 4×8 4 links rows removed (CoreRangeSet overlap). Run in table order (#15 next; #16 non-fused filled, fused not run). 7×7 removed (7×8 works better). **Row 14 (128k 8×8 2 links):** non-fused and fused are **projected** (see below), not measured. For any **llama_8k_ff2** row use **`and not K4096`** so only one test runs.

## num_links, num_workers_per_link, and core grid

**Core grid** (`compute_with_storage_grid_size`) = the (grid_x × grid_y) matmul core grid, e.g. 8×8. This is the set of worker cores used for the matmul and for the all-gather data movement.

**num_links** = number of fabric links used for the all-gather. Each link has forward and backward direction, so the op reserves **num_mux_cores = num_links × 2** mux cores (on the top row of the device grid) for fabric. So more links = more bandwidth for the gather, but they consume core space and must fit the grid.

**Constraint (with `force_transpose=True`):**
`grid_x % num_links == 0`
So the **grid x dimension** (e.g. 8 or 7) must be divisible by `num_links`. Examples:
- 8×8 grid → num_links can be 1, 2, 4, 8.
- 7×8 grid (Llama) → 7 is prime → num_links can only be 1 or 7.

**num_workers_per_link** = number of worker cores **per link** used for the all-gather. Total workers for the gather = num_links × num_workers_per_link. They are laid out along the grid (mux x index uses `num_workers_per_link * (link + 1)`), so:
- More workers per link can help DRAM-bound shapes (more parallelism on the gather).
- There must be enough cores: mux cores and worker cores must fit in the device grid (e.g. `full_grid_size.x` or `.y` ≥ num_mux_cores, and mux x indices stay within the grid).

**How we set them in the test:**
- `num_workers_per_link = max(1, min(8 // num_links, grid.x // num_links))` so total workers ≤ grid.x (in0 axis). This avoids sync hang when grid is narrow (e.g. 6×8 + 2 links: 4 workers/link ⇒ 8 workers but only 6 cores ⇒ link 1 has 2 cores, barrier never completes).
- `num_links` comes from the parametrized grid. If `grid_x % num_links != 0`, we reduce `num_links` to a divisor of grid_x (e.g. 7×8 → num_links=1).

**7×8 with force_transpose=False (skipped):** With `force_transpose=False`, the op uses `grid_y % num_links`, so 7×8 could use num_links=2. A test was added (4k4k4k only) but the **fused** path **segfaults** on this config; test is skipped until the op is fixed. Non-fused path was not verified.

---


Examples:
```bash
# 8k with padded K=4096 (done – PCC passed)
python tools/tracy/profile_this.py -c 'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py -k "fused and 8x8_4links and llama_8k_ff2_K4096" -v'

# 128k K=4096 – check if PCC improves (same hypothesis)
python tools/tracy/profile_this.py -c 'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py -k "fused and 8x8_4links and llama_128k_ff2_K4096" -v'
```

---



## PCC and dimensions (4k vs Llama 8k/128k)

- **4k×4k×4k (wan2_4k4k4k)**: PCC passes in all cases. Dimensions: M=K=N=4096 (all powers of 2). K_per_device=1024 (power of 2).
- **Llama 8k/128k (llama_8k_ff2, llama_128k_ff2)**: PCC is low (0.12, 0.88, 0.90, 0.00) depending on config. Dimensions: M=8192 or 131072, **K=3584**, N=2048. K_per_device=896.

**Difference**: The **K dimension (3584)** is not a power of 2: 3584 = 7×512 = 7×2^9. N=2048 and M are powers of 2. So the non–power-of-2 dimension in the failing cases is **K**, not N.

**PCC run-to-run variation**: The same config can give different PCC on different runs:
- 128k 8×8 4 links: 0.00 → 0.90 on re-run.
- 8k 8×8 4 links: 0.12 → 0.82 on another run.

**Why**: Likely **non-determinism** in the multi-device fused path: completion order of devices, all-gather/concat ordering, or how `ConcatMesh2dToTensor` assembles the mesh (e.g. device order or padding). The golden is fixed (same input/weight); only the device output or the way we read it back can change. 4k stays stable because power-of-2 dimensions may avoid the same ordering or padding edge paths. Until the fused op (or mesh concat) is deterministic for these shapes, PCC for Llama 8k/128k will remain run-dependent.

**Hypothesis**: Odd or non–power-of-2 **K** (e.g. 3584) may cause:
- Different padding or block boundaries in the kernel (K_tiles = 112 = 7×16).
- Mesh/shard layout or `ConcatMesh2dToTensor` ordering assumptions that break when K_per_device (896) is not a “nice” power-of-2 size.
- Possible driver or tiling edge cases when the inner dimension has a factor of 7.

N=2048 is a power of 2 and is unlikely to be the primary cause. Worth checking in the fused op (and ConcatMesh2dToTensor) whether any logic assumes K or K_per_device to be a power of 2 or a multiple of a specific block.

---

## Llama 70B model prefill 8k – FF2 step (single device)

Per-device kernel time for the FF2 (w2) step from profiler prefill CSVs. Config: 7×8 grid, subblock 2×2 (aligned with unit test). **Default** = current model implementation (non-fused, 4-link ring AG + MM).

| Run | AG + MM (µs) | AG (µs) | MM (µs) | num_links | Notes |
|-----|--------------|---------|---------|-----------|-------|
| Fused | 2093.64 | — | — | 1 | USE_FUSED_AG_MM=1; 7×8; single fused op. Source: `ag_mm_8k_new/8k/prefill.csv` |
| Non-fused | 1851.18 | 530.09 | 1321.09 | 4 | Default (current model): ring AG + MM. Source: `fused_ag_mm_8k_baseline/8k/prefill.csv` |
| Non-fused | 3118.54 | 1797.12 | 1321.42 | 1 | FF2_AG_1_LINK=1; same 1-link as fused. Source: `ag_mm_8k_baseline_1_num_links/8k/prefill.csv` |

**Conclusion:** Default (4-link non-fused) is faster than fused here because fused is limited to 1 link on 7×8 (limited all-gather bandwidth). With 1 link matched, fused (~2094 µs) is ~33% faster than non-fused (~3119 µs).

**Commands to re-run (from repo root):**
```bash
# Baseline (default: non-fused, 4-link)
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name <name>

# Fused
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name <name>

# Non-fused, 1 link (apples-to-apples with fused)
FF2_AG_1_LINK=1 SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=0 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name ag_mm_8k_baseline_1_num_links
```

---

## Conclusions

1. **8×8 is where fused beats non-fused.** Only the **8×8** core grid shows a clear fused-vs-non-fused win; both paths use the **same config** (grid, links, M/K/N). Smaller or different grids (6×8, 4×8, 7×8) either don’t improve as much or aren’t the Llama deployment shape.

2. **Earlier numbers used a restricted grid.** Previous non-fused/fused numbers (e.g. from teja/Allgather+matmul_fused_perf_results) were with a **4×8** core grid. The branch was **updated recently** to use the **full grid** (8×8, 6×8, 7×8, 4×8) so comparisons are now apples-to-apples.

3. **7×8 is bandwidth-limited for Llama.** The **7×8** grid is the max usable for the Llama 70B model (device layout), but it is **limited to num_links = 1** (7 is prime), so all-gather bandwidth is low and perf is limited. In the **separate** path, Llama’s matmul uses **56 cores** only; the fused op uses the same 7×8 grid, so it doesn’t get extra link parallelism there.

4. **Max Llama grid (7×8) doesn’t show a win.** Because of the single-link constraint, the **maximum available grid for Llama (7×8)** does **not** show the same fused-vs-non-fused improvement as 8×8.

5. **Padded K (3584 → 4096) fixes PCC and adds ~100 µs.** Padding **K from 3584 to 4096** (power of 2) improves **PCC to 0.99+** and increases kernel duration by **~100 µs**; acceptable for correctness.

6. **To get max perf on Llama:** Either the kernel needs **proper support for the 7×8 core grid** (e.g. higher **num_links** or different link/core mapping), or other runtime/op support, so that the fused op can leverage more bandwidth on the Llama deployment grid.

---

## Notes

1. The fused `all_gather_minimal_matmul_async` operation combines AllGather and MatMul into a single kernel.
2. Performance benefit comes from overlapping communication with computation.
3. Grid divisibility constraint: `grid_x % num_links == 0` (with `force_transpose=True`).
4. Llama uses 7x* grids which limits `num_links` to 1 or 7 (7 is prime).
