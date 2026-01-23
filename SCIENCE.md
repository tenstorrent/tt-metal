# SCIENCE

## Baseline (from 2026-01-23 run)
- B=1, K=512, layers=4: DEVICE FW avg 10.63 us; DEVICE KERNEL avg 9.59 us.
- B=1, K=1024, layers=4: DEVICE FW avg 13.94 us; DEVICE KERNEL avg 12.20 us.
- Tracy warned that resblock compute kernel still uses deprecated `namespace NAMESPACE { void MAIN { } }` syntax.

## Experiments
- 2026-01-23: Removed redundant pre-loop `custom_mm_block_init` in `models/demos/resblock/kernels/compute.cpp` and migrated compute kernel entry point to `kernel_main`.
  - B=1, K=512, layers=4: DEVICE FW avg 10.57 us; DEVICE KERNEL avg 9.52 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 13.90 us; DEVICE KERNEL avg 12.15 us.
- 2026-01-23: Removed redundant `cb_wait_front` on the bias CB inside `matmul_with_bias_block`.
  - B=1, K=512, layers=4: DEVICE FW avg 10.55 us; DEVICE KERNEL avg 9.51 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 13.86 us; DEVICE KERNEL avg 12.10 us.
- 2026-01-23: Moved weight CB waits to a single upfront wait (removed per-layer weight waits).
  - Outcome: Regressed slightly vs prior best.
  - B=1, K=512, layers=4: DEVICE FW avg 10.58 us; DEVICE KERNEL avg 9.53 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 13.85 us; DEVICE KERNEL avg 12.11 us.
- 2026-01-23: Replaced bias `copy_tile` + SFPU add with `binary_dest_reuse_tiles` (reuse DST + add bias directly from CB).
  - B=1, K=512, layers=4: DEVICE FW avg 9.64 us; DEVICE KERNEL avg 8.60 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 12.95 us; DEVICE KERNEL avg 11.20 us.
- 2026-01-23: Removed `DeviceZoneScopedN` profiler scopes from resblock compute + dataflow kernels to cut per-layer instrumentation overhead.
  - B=1, K=512, layers=4: DEVICE FW avg 8.91 us; DEVICE KERNEL avg 7.87 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 12.22 us; DEVICE KERNEL avg 10.48 us.
- 2026-01-23: Hoisted `relu_tile_init()` out of the per-layer matmul+relu block (single init per kernel).
  - B=1, K=512, layers=4: DEVICE FW avg 8.88 us; DEVICE KERNEL avg 7.85 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 12.20 us; DEVICE KERNEL avg 10.46 us.
- 2026-01-23: Replaced SFPU relu with packer relu (toggle `llk_pack_relu_config` around MM1 pack).
  - B=1, K=512, layers=4: DEVICE FW avg 8.19 us; DEVICE KERNEL avg 7.16 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 11.52 us; DEVICE KERNEL avg 9.77 us.
- 2026-01-23: Hoisted weight CB waits to a single pre-loop wait and skipped per-layer weight waits in compute.
  - Outcome: Regressed slightly vs prior best.
  - B=1, K=512, layers=4: DEVICE FW avg 8.22 us; DEVICE KERNEL avg 7.19 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 11.53 us; DEVICE KERNEL avg 9.79 us.
- 2026-01-23: Replaced per-layer full custom_mm init with a single full init + short re-init (unpack+math) between layers.
  - Outcome: Essentially flat; tiny regression for K=512, unchanged for K=1024.
  - B=1, K=512, layers=4: DEVICE FW avg 8.21 us; DEVICE KERNEL avg 7.17 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 11.51 us; DEVICE KERNEL avg 9.77 us.
- 2026-01-23: Kept weight CB read pointer fixed (no per-layer pop) and indexed weights by layer tile offset.
  - Outcome: Regressed vs prior best.
  - B=1, K=512, layers=4: DEVICE FW avg 8.24 us; DEVICE KERNEL avg 7.20 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 11.54 us; DEVICE KERNEL avg 9.80 us.
- 2026-01-23: Disabled `dst_full_sync_en` while keeping `fp32_dest_acc_en=True` in compute config.
  - Outcome: Regressed vs prior best (slower kernel duration).
  - B=1, K=512, layers=4: DEVICE FW avg 8.23 us; DEVICE KERNEL avg 7.20 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 11.54 us; DEVICE KERNEL avg 9.79 us.
- 2026-01-23: Forced loop unrolling for the per-layer compute loop in `models/demos/resblock/kernels/compute.cpp`.
  - Outcome: Essentially flat; tiny regression for K=512, unchanged for K=1024.
  - B=1, K=512, layers=4: DEVICE FW avg 8.21 us; DEVICE KERNEL avg 7.17 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 11.51 us; DEVICE KERNEL avg 9.77 us.
- 2026-01-23: Enabled `math_approx_mode=True` in compute config.
  - Outcome: Tiny improvement for K=1024, unchanged for K=512.
  - B=1, K=512, layers=4: DEVICE FW avg 8.20 us; DEVICE KERNEL avg 7.16 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 11.49 us; DEVICE KERNEL avg 9.76 us.

## Knowledge
- Resblock uses five kernels in `models/demos/resblock/kernels/` with compute in `compute.cpp` and data movement in `reader.cpp`/`writer.cpp` + mcast.
- `compute.cpp` currently initializes custom MM before the layer loop and again inside the loop; the pre-loop init appears redundant.
- Migrating resblock compute kernel to `kernel_main()` removes the deprecated namespace/MAIN warning and slightly reduces kernel duration for K=512/1024.
- In this kernel, `MM1_FULL_CB` is already waited on for matmul1 and not popped until after matmul2, so waiting again for bias tiles is unnecessary overhead.
- `binary_dest_reuse_tiles` (eltwise_binary) can add bias directly from CB to the matmul DST without a separate copy tile; this substantially reduced kernel duration.
- `DeviceZoneScopedN` profiler scopes in the hot-path resblock kernels add ~0.7–0.8 us of device kernel overhead for K=512/1024; removing them improved kernel duration without changing math.
- `relu_tile_init()` can be called once per kernel without correctness issues; per-layer init was redundant and cost a small but measurable amount.
- Packer relu (`llk_pack_relu_config(ReluType::ZERO_RELU)`) is a drop-in replacement for SFPU relu in this kernel and cuts another ~0.7 us for K=512/1024.
- Hoisting weight CB waits out of the per-layer matmul path regressed slightly, suggesting the per-layer waits are not a meaningful overhead and may help pacing with other kernels.
- Short re-init (unpack+math only) for custom_mm between layers did not improve performance; full custom_mm init overhead does not appear to dominate.
- Treating weight CBs as static with explicit tile offsets regressed, so the standard per-layer pop/wait path appears to be better for this workload.
- Running with `dst_full_sync_en=False` (while keeping FP32 destination accumulation enabled) regressed slightly for both K=512 and K=1024, so the full sync path appears to be the better choice here.
- Forcing a `#pragma unroll` on the per-layer loop did not improve performance in the current configuration.
- Enabling `math_approx_mode=True` yielded a very small improvement at K=1024 (no change at K=512); this seems safe for the current matmul+relu+bias path but should be kept in mind if future SFPU ops are added.
