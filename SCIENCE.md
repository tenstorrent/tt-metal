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

## Knowledge
- Resblock uses five kernels in `models/demos/resblock/kernels/` with compute in `compute.cpp` and data movement in `reader.cpp`/`writer.cpp` + mcast.
- `compute.cpp` currently initializes custom MM before the layer loop and again inside the loop; the pre-loop init appears redundant.
- Migrating resblock compute kernel to `kernel_main()` removes the deprecated namespace/MAIN warning and slightly reduces kernel duration for K=512/1024.
- In this kernel, `MM1_FULL_CB` is already waited on for matmul1 and not popped until after matmul2, so waiting again for bias tiles is unnecessary overhead.
- `binary_dest_reuse_tiles` (eltwise_binary) can add bias directly from CB to the matmul DST without a separate copy tile; this substantially reduced kernel duration.
- `DeviceZoneScopedN` profiler scopes in the hot-path resblock kernels add ~0.7–0.8 us of device kernel overhead for K=512/1024; removing them improved kernel duration without changing math.
- `relu_tile_init()` can be called once per kernel without correctness issues; per-layer init was redundant and cost a small but measurable amount.
