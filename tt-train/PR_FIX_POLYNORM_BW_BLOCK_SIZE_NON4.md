### Description
Fix a PolyNorm backward deadlock when `block_size != 4` by decoupling packed partial-gradient handling from the compute block size.
This non-4 block-size bug was flagged by `@lgalasTT`.

The intended invariant is: we keep `block_size = 4U` hardcoded for PolyNorm BW in normal operation, but the kernel reserve/push/pop/read/write/compute contracts are now consistent so the path also works if `block_size` is set to `1`, `2`, or `3`. Tails are handled via `current_block_size <= block_size`. Packed partial outputs remain a fixed 4-tile payload (`dw0`, `dw1`, `dw2`, `db`) per row, so that path keeps a fixed 4-tile reserve/push/pop/write contract.

### Changes Made
- [x] **Program factory:** keeps fixed `block_size = 4U` in `tt-train/sources/ttml/metal/ops/polynorm_bw/device/polynorm_bw_program_factory.cpp`.
- [x] **CB sizing:** kept packed partial-output CB capacity fixed at 4 tiles (independent of compute `block_size`).
- [x] **Compute kernel:** changed packed-partial emission to always push 4 tiles in `tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/compute/polynorm_bw_kernel.cpp`.
- [x] **Writer kernel:** changed packed-partial writer path to always consume/write a 4-tile block in `tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/dataflow/writer_polynorm_bw_interleaved_start_id.cpp`.

### Testing
- [x] Reproduced pre-fix behavior: `block_size=1` path hangs even on `PolyNormOpTest.PolyNorm_Compare_BasicSmall`.
- [x] `timeout 30s ./build/tt-train/tests/ttml_tests --gtest_filter=PolyNormOpTest.PolyNorm_Compare_BasicSmall`
- [x] `timeout 30s ./build/tt-train/tests/ttml_tests --gtest_filter=PolyNormOpTest.PolyNorm_Compare_BlockSizeRemainders` (covers `Wt=1,2,3,4`)
- [x] `timeout 30s ./build/tt-train/tests/ttml_tests --gtest_filter=PolyNormOpTest.PolyNorm_*` (non-nightly PolyNorm suite)

### Additional Context
- This PR intentionally **does not** include tolerance changes; tolerance tuning is handled separately.
- Scope is limited to the non-4 block-size backward deadlock fix.
- The backward tolerance issue on `main` (failing `PolyNormOpTest.PolyNorm_Compare_EpsilonVariants`) is acknowledged and will be addressed in [#44419](https://github.com/tenstorrent/tt-metal/pull/44419).
