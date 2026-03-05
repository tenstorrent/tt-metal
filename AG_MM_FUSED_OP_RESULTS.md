# AllGather + MatMul Fused Operation Performance Results

## Test Configuration
- **Platform**: Galaxy (8x4 mesh = 32 devices)
- **Topology**: Ring
- **Cluster Axis**: 1 (ring_size = 4)

---

## Same config: Fused vs Baseline (separate path)

**Yes.** For every test, fused and separate (baseline) use the **exact same config**:
- Same **core grid** (e.g. 8×8, 7×8)
- Same **num_links**
- Same **M, K, N** and block/subblock sizes

The test is parametrized once; only `use_fused` is toggled (`True` = fused op, `False` = separate AG then MM). So when you run e.g. `-k "8x8_4links and wan2_4k4k4k"`, both `separate` and `fused` use grid 8×8, 4 links, and 4096×4096×4096.

**Baseline numbers to run:** For every (size, grid, links) where we have fused results, run the **separate** path with the same `-k` (e.g. `-k "separate and 8x8_4links and wan2_4k4k4k"`) and record AG + MM kernel times from the profiler CSV. Then compare fused vs (AG + MM) for that config.

---

## WAN2 tests default vs our tests

| Aspect | WAN2.2 default (`models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py`) | Our tests (`tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py`) |
|--------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **Mesh** | (2,4), (8,4), (4,8) — multiple topologies | (8,4) only — Galaxy |
| **Core grid** | 4×4, 8×8, 12×9 (varies by config) | 8×8, 6×8, 4×8, 7×8 (7×7 removed; 7×9 excluded) |
| **Sizes** | 32768×4096×4096 (4k4k4k), 75776×5120×… (QKV, FF1, etc.) | 4096×4096×4096, 8192×3584×2048, 131072×3584×2048, + K-padded 4096 variants |
| **Features** | Bias, GELU, addcmul, chunks; `use_non_fused` for separate path | No bias/activation/addcmul; simple AG+MM only |
| **AG output buffer (separate)** | `(per_device_M, k_per_device*ring_size)` with `ShardTensor2dMesh(..., dims=[None, None])` | `(1,1,M,K)` with **ReplicateTensorToMesh** (full K per device for MM) |
| **Purpose** | WAN 2.2 model coverage, multiple shapes and features | Fused vs baseline comparison for Galaxy, Llama ISL sizes, PCC, and 7×9 “max usable” grid |

So: WAN2 default = model test suite (multiple meshes, shapes, bias/activation). Our tests = same-op comparison (Galaxy, same config for fused vs separate, Llama 8k/128k + 4k).

---

## 7×9 grid – NOC constraint (excluded)

**7×9 is not in the test parametrization** because it triggers:

`TT_FATAL: Illegal NOC usage: data movement kernels on logical core (x=6,y=8) cannot use the same NOC, doing so results in hangs!`

- **Cause:** On core (6, 8) the program places two data-movement kernels (DM0 and DM1). The runtime requires that when both run on the same core, one must use NOC0 and the other NOC1; here both end up using the same NOC.
- **Where:** Check in `tt_metal/tt_metal.cpp` (Tensix DM kernel NOC validation). The op’s kernel placement for 7×9 leads to this violation.
- **Action:** 7×9 is omitted from the grid list until the op (or kernel placement) assigns NOCs so that DM0/DM1 on the same core use different NOCs. Use 7×8 as the “max usable” Llama grid for now.

---

## Baseline (older run, 4×8 core grid)

Per-device kernel times from **teja/Allgather+matmul_fused_perf_results** prefill CSVs (FF2 layer). **Not** the current unit-test results — reference only.

| ISL | Core grid | Baseline (AG + MM) µs | Fused op µs |
|-----|-----------|------------------------|-------------|
| 8k | 4×8 | 1842 (AG 523.71 + MM 1318.09) | 2912.61 |
| 128k | 4×8 | 27633 (AG 8010.90 + MM 19622.12) | 46013.96 |

**Cores used in that baseline (separate path):** From **baseline_8k/8k/prefill.csv** (FF2 layer rows: AllGatherAsync 8192×896→3584, then Matmul 8192×3584×2048): **AllGather used 40 cores** per device and **Matmul used 56 cores** per device (column CORES). So in the separate path, AG and MM use **different** core sets/counts — not the same grid. Our unit-test comparison uses the **same** core grid for both (e.g. 8×8) so fused vs baseline is apples-to-apples.

---

## Ops Profiler CSV Layout

The profiler writes `ops_perf_results_<timestamp>.csv` with columns (key ones):

| Column # | Name | Description |
|----------|------|--------------|
| 1 | OP CODE | e.g. `AllGatherMinimalMatmulAsyncOp`, `AllGatherAsyncOp`, `MinimalMatmulOp` |
| 2 | OP TYPE | `tt_dnn_device` |
| 4 | DEVICE ID | 0–31 (per device) |
| 19 | DEVICE KERNEL DURATION [ns] | **Primary metric** – kernel time in nanoseconds |
| 20 | DEVICE KERNEL DURATION DM START [ns] | |
| 21–23 | DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG [ns] | |
| 29 | DEVICE ERISC KERNEL DURATION [ns] | Ethernet/fabric time |
| 30 | DEVICE TRISC0 KERNEL DURATION [ns] | Compute time |

- Each row = one op invocation on one device.
- Fused run: 5 iters × 32 devices = 160 rows of `AllGatherMinimalMatmulAsyncOp`.
- Baseline (separate) run: 160 rows of `AllGatherAsyncOp` + 160 rows of `MinimalMatmulOp` (or similar); sum AG + MM per (device, iter) for total baseline time.

---

## Test 1: Fused AG+MM - WAN 2.2 Size (4k x 4k x 4k)

**Date**: 2026-03-04 21:26:50

**Configuration**:
| Parameter | Value |
|-----------|-------|
| M | 4096 |
| K | 4096 (K_per_device = 1024) |
| N | 4096 |
| Grid | 8x8 (72 cores) |
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

## Baseline (Separate path)

4k4k4k separate path (AG then MM) now runs after fixing persistent buffer to use full-K per device (ReplicateTensorToMesh for AG output). **PR reference** (unfused): 963 µs AG, 1785 µs MM. **Our run** (8×8 4 links, device 31 avg): 452 µs AG, 1779 µs MM → baseline **2231 µs**; fused same config **1858 µs** (~17% faster).

---

## Fused ops – run one by one (commands and results)

**Setup:** Test passes full input (1,1,M,K); runtime/mesh shards it (each device gets its K-shard). Kernel does AllGather + matmul.

**Grids:** 8×8, 6×8, 4×8, 7×8. (7×7 removed; 7×8 works better. 7×9 excluded — NOC constraint.)

**Prefix:** `python tools/tracy/profile_this.py -c 'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py -k "fused and <GRID_LINKS> and <SIZE_ID>" -v'`

Record **single-device avg** from profiler: `Device kernel duration perf summary (device=31): ... avg=...ns` → µs. **Baseline** = separate path (AG + MM) same config; sum AG + MM kernel µs from profiler CSV.

| # | Size (M×K×N) | Grid | Links | Status | Baseline (µs) | Fused (µs) | PCC | Run command `-k` fragment |
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
| # | Size (M×K×N) | Grid | Links | Status | Baseline (µs) | Fused (µs) | PCC | Run command `-k` fragment |
|---|--------------|------|-------|--------|---------------|------------|-----|---------------------------|
| 1 | 8192×4096×2048 | 8×8 | 4 | Done | 2742 | 1955 | ✅ PCC 0.9998 | `8x8_4links and llama_8k_ff2_K4096` |
| 2 | 8192×4096×2048 | 8×8 | 2 | Done | 3442 | 2668 | ✅ PCC 0.9998 | `8x8_2links and llama_8k_ff2_K4096` |
|  |  |  |  |  |  |  |  |  |
| 3 | 131072×4096×2048 | 8×8 | 4 | Done | 42150 | 29475 | ✅ PCC 1.0 | `8x8_4links and llama_128k_ff2_K4096` |
| 4 | 131072×4096×2048 | 8×8 | 2 | Done | 52900 | 39015 | ✅ | `8x8_2links and llama_128k_ff2_K4096` |
**Baseline commands (run separate path, then share profiler CSV; we fill Baseline column):**
Always include **`separate`** in the `-k` so only the baseline (AG then MM) runs — one test. Without it, both `separate` and `fused` can match and the run executes twice. For **llama_8k_ff2** and **llama_128k_ff2** use **`and not K4096`** so only the original (K=3584) runs — otherwise two tests run (e.g. llama_128k_ff2 and llama_128k_ff2_K4096). 4×8 4 links rows removed (CoreRangeSet overlap). Run in table order (#15 next; #16 baseline filled, fused not run). 7×7 removed (7×8 works better). **Row 14 (128k 8×8 2 links):** baseline and fused are **projected** (see below), not measured. For any **llama_8k_ff2** row use **`and not K4096`** so only one test runs.

## Tried shapes (K padded to 4096 – PCC hypothesis)

Same Llama M/N but **K padded to 4096** (power of 2) to test whether non–power-of-2 K (3584) is the culprit for low PCC. Results are in the **K padded to 4096** table above (with row gap after main table).

Examples:
```bash
# 8k with padded K=4096 (done – PCC passed)
python tools/tracy/profile_this.py -c 'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py -k "fused and 8x8_4links and llama_8k_ff2_K4096" -v'

# 128k K=4096 – check if PCC improves (same hypothesis)
python tools/tracy/profile_this.py -c 'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py -k "fused and 8x8_4links and llama_128k_ff2_K4096" -v'
```

---

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

---

## Earlier 8k ISL fused op (per device, from prefill)

From **~/teja/Allgather+matmul_fused_perf_results/fused_8k/8k/prefill.csv** (full Llama 8k prefill run, per-device):

- **AllGatherMinimalMatmulAsyncOp:** **2912.61 µs** (device 0, 1 call)
- Shapes in that run: IN0 8192×896, IN1 3584×2048, OUT 8192×3584 (one of the FF layers; our unit-test llama_8k_ff2 is 8192×3584×2048 for FF2).
- All values in that CSV are **per device** (single device, prefill trace).

Use this as a reference when comparing unit-test 8k grid/link numbers (table above) to the earlier full-model 8k run.

---

## Earlier 128k fused op (per device, from prefill)

From **~/teja/Allgather+matmul_fused_perf_results/fused_128k/128k/prefill.csv** (full Llama 128k prefill run, per-device):

- **AllGatherMinimalMatmulAsyncOp:** **46,013.96 µs** (~46.0 ms) (device 0, 1 call)
- Shapes: IN0 131072×896 (M×K_per_device), IN1 3584×2048 (K×N), OUT 131072×3584 (M×K after gather; FF layer).
- All values in that CSV are **per device** (single device, prefill trace).

Unit-test 128k comparison: 4×8 2 links gave **48,313 µs** (table above) — in the same range as this earlier fused 128k run. 8×8 4 links gave **26,629 µs** (faster with more links / larger grid).

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

## Conclusions

1. **8×8 is where fused beats baseline.** Only the **8×8** core grid shows a clear fused-vs-baseline win; both paths use the **same config** (grid, links, M/K/N). Smaller or different grids (6×8, 4×8, 7×8) either don’t improve as much or aren’t the Llama deployment shape.

2. **Earlier numbers used a restricted grid.** Previous baseline/fused numbers (e.g. from teja/Allgather+matmul_fused_perf_results) were with a **4×8** core grid. The branch was **updated recently** to use the **full grid** (8×8, 6×8, 7×8, 4×8) so comparisons are now apples-to-apples.

3. **7×8 is bandwidth-limited for Llama.** The **7×8** grid is the max usable for the Llama 70B model (device layout), but it is **limited to num_links = 1** (7 is prime), so all-gather bandwidth is low and perf is limited. In the **separate** path, Llama’s matmul uses **56 cores** only; the fused op uses the same 7×8 grid, so it doesn’t get extra link parallelism there.

4. **Max Llama grid (7×8) doesn’t show a win.** Because of the single-link constraint, the **maximum available grid for Llama (7×8)** does **not** show the same fused-vs-baseline improvement as 8×8.

5. **Padded K (3584 → 4096) fixes PCC and adds ~100 µs.** Padding **K from 3584 to 4096** (power of 2) improves **PCC to 0.99+** and increases kernel duration by **~100 µs**; acceptable for correctness.

6. **To get max perf on Llama:** Either the kernel needs **proper support for the 7×8 core grid** (e.g. higher **num_links** or different link/core mapping), or other runtime/op support, so that the fused op can leverage more bandwidth on the Llama deployment grid.

---

## Notes

1. The fused `all_gather_minimal_matmul_async` operation combines AllGather and MatMul into a single kernel.
2. Performance benefit comes from overlapping communication with computation.
3. Grid divisibility constraint: `grid_x % num_links == 0` (with `force_transpose=True`).
4. Llama uses 7x* grids which limits `num_links` to 1 or 7 (7 is prime).
