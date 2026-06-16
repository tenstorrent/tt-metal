Hi @Nikola Vukobrat
Task update from my side,
BEV YUV Conv2d DRAM Bottleneck — Method 1 Investigation

Following up on the issue comment (https://github.com/tenstorrent/tt-metal/issues/46831#issuecomment-4691303095), the community member proposed Method 1: add an `in_channels < TILE_WIDTH` guard to `use_matmul_for_1x1_conv()` so that C=3 routes to the regular conv path instead of the matmul path. The regular conv path reads activations in ROW_MAJOR format with 8-element channel alignment, reducing per-pixel DRAM reads from 64 B to 16 B. The C++ change was implemented and Tracy-profiled on both Block A and Block C using the `test_conv2d_only` test case (isolated conv2d, ROW_MAJOR input, no slice config).

 End-to-end result
| Config             | Input Shape          | Path                         | Total Ops | Total Kernel | vs Baseline  |
| ------------------ | -------------------- | ---------------------------- | --------- | ------------ | ------------ |
| Conv2d 1 — Block A | `(1, 3, 1536, 1536)` | Baseline (Matmul, TILE)      | 1         | 6.430 ms     | 1.00×        |
| Conv2d 1 — Block A | `(1, 3, 1536, 1536)` | Method 1 (RegConv, ROW_MAJOR)| 30        | 17.190 ms    | **2.67× slower** |
| Conv2d 2 — Block C | `(1, 3, 1280, 2304)` | Baseline (Matmul, TILE)      | 1         | 7.923 ms     | 1.00×        |
| Conv2d 2 — Block C | `(1, 3, 1280, 2304)` | Method 1 (RegConv, ROW_MAJOR)| 40        | 21.413 ms    | **2.70× slower** |
Method 1 correctly routes to the regular conv path — the op name changes from MatmulDeviceOperation to Conv2dDeviceOperation and the per-slice conv kernels are 12.9–65.6× faster individually. However, the regular conv path triggers DRAM slicing (6–8 slices × 5 ops = 30–40 ops total) whose infrastructure overhead dominates the runtime.

 Op breakdown — Block A (6 slices, 17.190 ms total)
| Op                         | Count | Total Kernel | % of Total |
| -------------------------- | ----- | ------------ | ---------- |
| PaddedSliceDeviceOperation | 6     | 5.582 ms     | 32.5%      |
| HaloDeviceOperation        | 6     | 5.202 ms     | 30.3%      |
| SliceWriteDeviceOperation  | 6     | 3.006 ms     | 17.5%      |
| Conv2dDeviceOperation      | 6     | 2.996 ms     | 17.4%      |
| MoveDeviceOperation        | 6     | 0.404 ms     | 2.3%       |
The actual conv compute accounts for only 17.4% of total time. The surrounding slice infrastructure — PaddedSlice + Halo + SliceWrite + Move — accounts for 82.6%. HaloDeviceOperation alone takes 5.2 ms (30.3%) despite being a mathematical no-op for a 1×1 / stride=1 / pad=0 kernel; it dispatches a full kernel per slice with no useful work.

 Why DRAM slicing is unavoidable
The output shard per core in TILE format is `36,864 px × OC_padded=32 × 2 B = 2,304 KB`, which is 1.6× the L1 bank capacity (1,400 KB). Even with ROW_MAJOR input (input shard = 576 KB, fits in L1), the output tile size alone prevents a single-pass L1 HEIGHT_SHARDED conv. All shard configurations — HEIGHT_SHARDED, BLOCK_SHARDED, and `act_block_h_override` tuning (abh=64, 128, 256) — were profiled. Reducing from 6–8 slices to 3 slices gave only a 2–3% improvement, confirming that the per-slice overhead scales with slice data volume and cannot be tuned away.

The detailed analysis is in `docs/method1_conv2d_only_analysis.md` and `docs/method1_l1_fitting_analysis.md`.
