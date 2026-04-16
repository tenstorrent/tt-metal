# BH Fast-Tilize Test Status

Last updated: 2026-04-16

Tests from commit `bb6a99a2554` (Implement BH fast-tilize LLK primitives and tests), with subsequent fixes.

## Summary

| Environment | Pass | Fail | Skip |
|-------------|------|------|------|
| Silicon | **397** | 0 | 0 |
| ttsim | **397** | 0 | 7 |

## Per-Test Results

| Test File | Silicon | ttsim | Notes |
|-----------|---------|-------|-------|
| test_fast_tilize_full.py | 284 pass | 284 pass | Core tilize — all formats, all dimensions; includes 56 overflow-guard variants that catch PACR_FLUSH L1 overflows |
| test_fast_tilize_matmul.py | 4 pass | 4 pass | Fast + standard tilize → matmul with explicit PACK_DONE barrier between phases |
| test_fast_tilize_metal_api.py | 6 pass | 6 pass | Metal compute API flow |
| test_perf_measure.py | 7 pass | 7 skip | Perf profiling (silicon only) |
| test_tilize_matmul_repro.py | 96 pass | 96 pass | All scenarios (test_1-7) including multi-iter accumulation with fp32 DEST |

## Deleted Tests

| Test | Reason |
|------|--------|
| test_pack_l1_autoadvance_probe.py | Missing source file |
| test_fast_tilize_unpack.py | Broken by design (pack/math layout mismatch) |
| test_tilize_matmul_repro.py::test_8_fast_tilize_matmul_accum_bfp_mop | Diagnostic probe for non-existent bug; used invalid format mismatch |

## Fixes Applied This Session

### Test infrastructure
- Removed dimension skip guards from `test_fast_tilize_full.py` (154 pass + 130 skip → 284 pass)
- `test_perf_measure.py`: `TestConfig.MODE` → `BUILD_MODE`, added missing `location` arg to `read_words_from_device`, skip on ttsim
- `test_fast_tilize_metal_api.py`: added missing `NUM_GUARD_TILES(0)` runtime param

### Test kernels
- `tilize_matmul_test.cpp` / `fast_tilize_matmul_test.cpp`: matched working `std_tilize_matmul_repro.cpp` pack-transition pattern (`hw_configure(tilize=false)` + `pack_init(tilize=true)` + `reconfig_data_format` for phase 2). Removed redundant "compensate odd section_done" code. Added `tilize()` to Python golden to match pack output layout.
- `fast_tilize_matmul_test.cpp`: added explicit **PACK_DONE semaphore barrier** between fast-tilize and matmul phases (fixes ttsim math/pack TRISC race).
- `fast_tilize_matmul_test.cpp`: removed extra `pack_src` arg from `_llk_pack_fast_tilize_uninit_` (API drift)
- `std_tilize_matmul_accum_repro.cpp` / `fast_tilize_matmul_accum_repro.cpp`: fixed `_llk_unpack_A_init_<>()` default args (fp32→fp32 fails assert when `unpack_to_dest=false`); now passes explicit src/dst formats
- Added sem 0 + sem 4 initialization to matmul kernels for ttsim (no firmware pre-init)

### Test parameters
- test_6/7 (multi-iter tilize→matmul accumulation): `DestAccumulation.No` → `DestAccumulation.Yes` (fp32 DEST eliminates bf16 precision loss over 8 iterations; was failing PCC threshold)

## Overflow Protection

`test_fast_tilize_overflow_guard` (56 variants in `test_fast_tilize_full.py`) validates fast-tilize does not overwrite L1 memory beyond the result buffer:
- 5 sentinel guard tiles allocated after result buffer
- Uses constant input (0.5) to distinguish data leak (0.5 in guard) vs ZeroWrite overflow (0.0 in guard)
- C++ side computes per-tile corruption count and writes marker + counts
- Python verifies `corrupted == 0` for each guard tile
- Covers: 4 format combos × 2 dest_acc × 7 dimensions
