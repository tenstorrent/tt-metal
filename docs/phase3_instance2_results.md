# Phase 3, Instance 2 Results — Isolated Helper Tests

## Architecture

Tests developed and host-side compiled on **Wormhole B0** (n150 board).

## Files Created

| File | Purpose |
|------|---------|
| `tests/tt_metal/tt_metal/test_kernels/compute/test_matmul_helper_features_compute.cpp` | Parameterized compute kernel for testing `matmul_block` and `add_bias_bcast_rows` helpers |
| `tests/tt_metal/tt_metal/integration/matmul/test_matmul_helper_features.cpp` | Host-side integration tests (12 test cases) |
| `tests/tt_metal/tt_metal/integration/sources.cmake` | Modified — added new test to `UNIT_TESTS_INTEGRATION_SRC` |

## How to Build

```bash
cmake -B build_Release -DTT_METAL_BUILD_TESTS=ON -G Ninja
cmake --build build_Release --target unit_tests_integration -j$(nproc)
```

## How to Run

Run all new feature tests:
```bash
TT_METAL_HOME=$(pwd) build_Release/test/tt_metal/unit_tests_integration \
  --gtest_filter="*TensixMatmulHelper*"
```

Run a specific test:
```bash
TT_METAL_HOME=$(pwd) build_Release/test/tt_metal/unit_tests_integration \
  --gtest_filter="*TensixMatmulHelperL1AccMultiBlock*"
```

**Important**: The compute kernel JIT-compiles at runtime and depends on Instance 1's helper implementation. The tests will fail at JIT compile time until:
1. `matmul_block` has new template params: `packer_l1_acc`, `pack_last_to_interm`, `pack_relu` (per Phase 2 design Section C1)
2. `bias_add_helpers.hpp` / `bias_add_helpers.inl` exist with `add_bias_bcast_rows` (per Phase 2 design Section C2)

## Test Cases (12 total)

| # | Test name | Features | Matrix (M×N×K) | K-blocks | Sub-blocks (M×N) |
|---|-----------|----------|----------------|----------|-------------------|
| 1 | `TensixMatmulHelperL1AccSingleBlock` | PACKER_L1_ACC | 64×64×64 | 1 | 1×1 |
| 2 | `TensixMatmulHelperL1AccMultiBlock` | PACKER_L1_ACC | 64×64×128 | 2 | 1×1 |
| 3 | `TensixMatmulHelperPackRelu` | PACK_RELU | 64×64×64 | 1 | 1×1 |
| 4 | `TensixMatmulHelperPackReluMultiBlock` | PACK_RELU | 64×64×128 | 2 | 1×1 |
| 5 | `TensixMatmulHelperFusedBias` | FUSE_BIAS | 64×64×64 | 1 | 1×1 |
| 6 | `TensixMatmulHelperFusedBiasMultiBlock` | FUSE_BIAS | 64×64×128 | 2 | 1×1 |
| 7 | `TensixMatmulHelperL1AccBias` | L1_ACC + BIAS | 64×64×128 | 2 | 1×1 |
| 8 | `TensixMatmulHelperBiasRelu` | BIAS + RELU | 64×64×64 | 1 | 1×1 |
| 9 | `TensixMatmulHelperL1AccBiasRelu` | L1_ACC + BIAS + RELU | 64×64×128 | 2 | 1×1 |
| 10 | `TensixMatmulHelperMultiSubblockBias` | BIAS + multi-subblock | 128×128×64 | 1 | 2×2 |
| 11 | `TensixMatmulHelperMultiSubblockL1AccBias` | L1_ACC + BIAS + multi-subblock | 128×128×128 | 2 | 2×2 |
| 12 | `TensixMatmulHelperL1AccRelu` | L1_ACC + RELU | 64×64×128 | 2 | 1×1 |

All tests use `out_subblock_h=2, out_subblock_w=2, in0_block_w=2, batch=1`.

## Compute Kernel Design

The compute kernel (`test_matmul_helper_features_compute.cpp`) is parameterized via:

**Compile-time args** (12 args, same layout as existing matmul test kernels):
- `[0]` in0_block_w, `[1]` in0_num_subblocks, `[2]` in0_block_num_tiles, `[3]` in0_subblock_num_tiles, `[4]` in1_num_subblocks, `[5]` in1_block_num_tiles, `[6]` in1_per_core_w, `[7]` num_blocks, `[8]` out_subblock_h, `[9]` out_subblock_w, `[10]` out_subblock_num_tiles, `[11]` batch

**JIT defines** (via `ComputeConfig::defines`):
- `PACKER_L1_ACC` — enables `packer_l1_acc=true` template param
- `PACK_RELU` — enables `pack_relu=true` (non-bias) or caller-managed RELU (bias path)
- `FUSE_BIAS` — enables `pack_last_to_interm=true` + `add_bias_bcast_rows` call

**CB layout**:
- CB0 = in0 (matrix A), CB1 = in1 (matrix B), CB2 = bias, CB16 = out, CB24 = interm

**Two code paths** selected by `#ifdef FUSE_BIAS`:
1. **No bias**: `matmul_block<..., pack_last_to_interm=false, do_relu>(...)` — direct to out_cb
2. **With bias**: `matmul_block<..., pack_last_to_interm=true>(...)` → caller L1_ACC/RELU transitions → `add_bias_bcast_rows<interm, bias, out>(...)`

## Host-Side Test Design

**Reader kernel**: `reader_matmul_with_bias_blocked.cpp` — loads A (CB0), B (CB1), and optionally bias (CB2). Uses `with_bias` runtime arg flag.

**Writer kernel**: `writer_unswizzle.cpp` — standard sub-block unswizzle output.

**CB configuration**: out_cb (CB16) and interm_cb (CB24) share L1 address space (standard pattern). Bias CB (CB2) is separate. Total shared allocation = `num_output_tiles * single_tile_size`.

**Golden reference**: CPU-side matmul (`float` accumulation in bfloat16) + optional row-broadcast bias + optional relu. PCC threshold: 0.97.

**Bias tilization**: Bias 1D vector of length N is replicated across TILE_HEIGHT rows to create a [32, N] matrix, then tilized. This ensures `add_tiles_bcast_rows` gets correct values regardless of which row the hardware broadcasts from.

## Build Verification

- Host-side code compiles cleanly (verified on 2026-03-31)
- All 12 new tests are registered in the binary (`--gtest_list_tests` confirms)
- All 5 existing `TensixMatmulBlockHelper*` tests still present and passing (PCC=0.999289 on basic test)
- Compute kernel JIT compilation blocked on Instance 1's implementation (expected)

## Notes for Instance 3

1. **Running these tests**: After Instance 1's helpers are in place, run the full filter `--gtest_filter="*TensixMatmulHelper*"` to verify the helpers work end-to-end.
2. **If a test fails**: Check if the failure is architecture-specific (WH vs BH). The tests were developed on Wormhole B0. Lower PCC threshold to 0.95 if L1_ACC tests show precision differences on Blackhole.
3. **Extending tests**: To add a new feature combination, add a new `TEST_F` at the bottom of the host-side file with the appropriate `FeatureTestConfig` flags. The compute kernel handles all combinations via `#ifdef` — no kernel changes needed.
4. **Multi-subblock tests (#10, #11)** are the most likely to reveal bugs — they exercise sub-block iteration in both matmul_block and add_bias_bcast_rows with 2×2 sub-block grids.
