# AllGather + MatMul Fused Operation Performance Results

## Test Configuration
- **Platform**: Galaxy (8x4 mesh = 32 devices)
- **Topology**: Ring
- **Cluster Axis**: 1 (ring_size = 4)

---

## Baseline (older run, 4×8 core grid)

Per-device kernel times from **teja/Allgather+matmul_fused_perf_results** prefill CSVs (FF2 layer). **Not** the current unit-test results — reference only.

| ISL | Core grid | Baseline (AG + MM) µs | Fused op µs |
|-----|-----------|------------------------|-------------|
| 8k | 4×8 | 1842 (AG 523.71 + MM 1318.09) | 2912.61 |
| 128k | 4×8 | 27633 (AG 8010.90 + MM 19622.12) | 46013.96 |

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

Baseline test (separate AllGather + MatMul) currently **fails** with Fabric Router Sync timeout. Fused ops are being run and recorded first.

---

## Fused ops – run one by one (commands and results)

**Setup:** Test passes full input (1,1,M,K); runtime/mesh shards it (each device gets its K-shard). Kernel does AllGather + matmul.

**Prefix:** `python tools/tracy/profile_this.py -c 'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py -k "fused and <GRID_LINKS> and <SIZE_ID>" -v'`

Record **single-device avg** from profiler: `Device kernel duration perf summary (device=31): ... avg=...ns` → µs.

| # | Size (M×K×N) | Size ID | Grid | Links | Status | Kernel (µs) | PCC | Run command `-k` fragment |
|---|--------------|---------|------|-------|--------|------------|-----|---------------------------|
| 1 | 4096×4096×4096 | wan2_4k4k4k | 8×8 | 4 | Done | 1858 | ✅ | `fused and 8x8_4links and wan2_4k4k4k` |
| 2 | 4096×4096×4096 | wan2_4k4k4k | 8×8 | 2 | Done | 2156 | ✅ | `fused and 8x8_2links and wan2_4k4k4k` |
| 3 | 4096×4096×4096 | wan2_4k4k4k | 6×8 | 3 | Done | 2531 | ✅ | `fused and 6x8_3links and wan2_4k4k4k` |
| 4 | 4096×4096×4096 | wan2_4k4k4k | 6×8 | 2 | Done | 2761 | ✅ | `fused and 6x8_2links and wan2_4k4k4k` |
| 5 | 4096×4096×4096 | wan2_4k4k4k | 4×8 | 4 | Skip (CoreRangeSet overlap) | — | — | `fused and 4x8_4links and wan2_4k4k4k` |
| 6 | 4096×4096×4096 | wan2_4k4k4k | 4×8 | 2 | Done | 3401 | ✅ | `fused and 4x8_2links and wan2_4k4k4k` |
| 7 | 4096×4096×4096 | wan2_4k4k4k | 7×7 | 1 | Done | 3764 | ✅ | `fused and 7x7_1link and wan2_4k4k4k` |
| 8 | 4096×4096×4096 | wan2_4k4k4k | 7×8 | 1 | Done | 3327 | ✅ | `fused and 7x8_1link and wan2_4k4k4k` |
| 9 | 8192×3584×2048 | llama_8k_ff2 | 8×8 | 4 | Done | 1827 | ⚠️ PCC 0.82 | `fused and 8x8_4links and llama_8k_ff2` |
| 10 | 8192×3584×2048 | llama_8k_ff2 | 8×8 | 2 | Done | 2373 | ⚠️ PCC 0.88 | `fused and 8x8_2links and llama_8k_ff2` |
| 11 | 8192×3584×2048 | llama_8k_ff2 | 6×8 | 3 | Done | 2450 | ⚠️ PCC 0.88 | `fused and 6x8_3links and llama_8k_ff2` |
| 12 | 8192×3584×2048 | llama_8k_ff2 | 6×8 | 2 | Done | 2772 | ⚠️ PCC 0.88 | `fused and 6x8_2links and llama_8k_ff2` |
| 13 | 8192×3584×2048 | llama_8k_ff2 | 4×8 | 4 | Skip (CoreRangeSet overlap) | — | — | `fused and 4x8_4links and llama_8k_ff2` |
| 14 | 8192×3584×2048 | llama_8k_ff2 | 4×8 | 2 | Done | 3090 | ⚠️ PCC 0.88 | `fused and 4x8_2links and llama_8k_ff2` |
| 15 | 8192×3584×2048 | llama_8k_ff2 | 7×7 | 1 | Done | 4318 | ⚠️ PCC 0.88 | `fused and 7x7_1link and llama_8k_ff2` |
| 16 | 8192×3584×2048 | llama_8k_ff2 | 7×8 | 1 | Done | 3435 | ⚠️ PCC 0.88 | `fused and 7x8_1link and llama_8k_ff2` |
| 17 | 131072×3584×2048 | llama_128k_ff2 | 8×8 | 4 | Done | 26406 | ⚠️ PCC 0.90 | `fused and 8x8_4links and llama_128k_ff2` |
| 18 | 131072×3584×2048 | llama_128k_ff2 | 8×8 | 2 | Pending | | | `fused and 8x8_2links and llama_128k_ff2` |
| 19 | 131072×3584×2048 | llama_128k_ff2 | 6×8 | 3 | Pending | | | `fused and 6x8_3links and llama_128k_ff2` |
| 20 | 131072×3584×2048 | llama_128k_ff2 | 6×8 | 2 | Pending | | | `fused and 6x8_2links and llama_128k_ff2` |
| 21 | 131072×3584×2048 | llama_128k_ff2 | 4×8 | 4 | Pending | | | `fused and 4x8_4links and llama_128k_ff2` |
| 22 | 131072×3584×2048 | llama_128k_ff2 | 4×8 | 2 | Done | 48313 | ⚠️ PCC 0.90 | `fused and 4x8_2links and llama_128k_ff2` |
| 23 | 131072×3584×2048 | llama_128k_ff2 | 7×7 | 1 | Pending | | | `fused and 7x7_1link and llama_128k_ff2` |
| 24 | 131072×3584×2048 | llama_128k_ff2 | 7×8 | 1 | Done | 50359 | ⚠️ PCC 0.90 | `fused and 7x8_1link and llama_128k_ff2` |

**Full command for #2 (next):**
```bash
python tools/tracy/profile_this.py -c 'pytest tests/ttnn/unit_tests/operations/ccl/test_llama_ag_mm_comparison.py -k "fused and 8x8_2links and wan2_4k4k4k" -v'
```

---

## Tried shapes (K padded to 4096 – PCC hypothesis)

Same Llama M/N but **K padded to 4096** (power of 2) to test whether non–power-of-2 K (3584) is the culprit for low PCC. Run fused tests and record PCC / kernel time below.

| Shape (M×K×N) | Size ID | Grid | Links | Status | Kernel (µs) | PCC | Run command `-k` fragment |
|---------------|---------|------|-------|--------|------------|-----|---------------------------|
| 8192×4096×2048 | llama_8k_ff2_K4096 | 8×8 | 4 | Done | 1955 | ✅ PCC 0.9998 | `fused and 8x8_4links and llama_8k_ff2_K4096` |
| 8192×4096×2048 | llama_8k_ff2_K4096 | 8×8 | 2 | Done | 2668 | ✅ PCC 0.9998 | `fused and 8x8_2links and llama_8k_ff2_K4096` |
| 131072×4096×2048 | llama_128k_ff2_K4096 | 8×8 | 4 | Done | 29475 | ✅ PCC 1.0 | `fused and 8x8_4links and llama_128k_ff2_K4096` |
| 131072×4096×2048 | llama_128k_ff2_K4096 | 8×8 | 2 | Pending | | | `fused and 8x8_2links and llama_128k_ff2_K4096` |

Examples:
```bash
# 8k K=4096 (done – PCC passed)
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

## Notes

1. The fused `all_gather_minimal_matmul_async` operation combines AllGather and MatMul into a single kernel.
2. Performance benefit comes from overlapping communication with computation.
3. Grid divisibility constraint: `grid_x % num_links == 0` (with `force_transpose=True`).
4. Llama uses 7x* grids which limits `num_links` to 1 or 7 (7 is prime).
