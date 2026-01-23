# SCIENCE

## Baseline (from 2026-01-23 run)
- B=1, K=512, layers=4: DEVICE FW avg 10.63 us; DEVICE KERNEL avg 9.59 us.
- B=1, K=1024, layers=4: DEVICE FW avg 13.94 us; DEVICE KERNEL avg 12.20 us.
- Tracy warned that resblock compute kernel still uses deprecated `namespace NAMESPACE { void MAIN { } }` syntax.

## Experiments
- 2026-01-23: Removed redundant pre-loop `custom_mm_block_init` in `models/demos/resblock/kernels/compute.cpp` and migrated compute kernel entry point to `kernel_main`.
  - B=1, K=512, layers=4: DEVICE FW avg 10.57 us; DEVICE KERNEL avg 9.52 us.
  - B=1, K=1024, layers=4: DEVICE FW avg 13.90 us; DEVICE KERNEL avg 12.15 us.

## Knowledge
- Resblock uses five kernels in `models/demos/resblock/kernels/` with compute in `compute.cpp` and data movement in `reader.cpp`/`writer.cpp` + mcast.
- `compute.cpp` currently initializes custom MM before the layer loop and again inside the loop; the pre-loop init appears redundant.
- Migrating resblock compute kernel to `kernel_main()` removes the deprecated namespace/MAIN warning and slightly reduces kernel duration for K=512/1024.
