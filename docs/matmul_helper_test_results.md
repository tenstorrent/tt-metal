# Matmul Helper Library Test Results

**Date**: 2026-03-25
**Branch**: wransom/llk3
**Hardware**: Blackhole p100a (single device)
**Note**: Previous testing was on Wormhole. This is the first validation on Blackhole.

---

## Summary

### Post-migration results (bmm_large_block_zm.cpp now uses matmul_block helper)

| Test Suite | Passed | Failed | Skipped | Notes |
|------------|--------|--------|---------|-------|
| **C++ Integration (all matmul)** | 11 | 0 | 0 | Includes 4 matmul_tile helper tests (PCC > 0.997) |
| **Programming Example: multicore_reuse** | 1 | 0 | 0 | PCC = 0.999, uses matmul_block helper |
| **Python Unit: test_matmul.py (targeted)** | 26 | 0 | 0 | Key matmul patterns |
| **Python Unit: test_matmul.py (full)** | 32 | 1 | 16 | Failure: pre-existing BH tiny_tiles issue |
| **Nightly: test_matmul.py** | 45 | 0 | 0 | SD/SDXL matmul configs |
| **Nightly: test_bert_matmuls.py** | 144 | 0 | 320 | BERT matmul patterns |
| **Model: SentenceBERT (BH)** | 1 | 0 | 0 | Full BH model end-to-end, 73.75s |

### Totals: 260 passed, 1 failed (pre-existing), 336 skipped

The single failure (`test_optional_output_argument_with_tiny_tiles`) is a pre-existing Blackhole issue in `system_memory_manager.cpp`, unrelated to our changes. All results match the pre-change baseline exactly.

---

## Changes Tested

The primary change tested in this round is the migration of `bmm_large_block_zm.cpp` — the core TTNN production matmul compute kernel — to use the `matmul_block` helper. This kernel is exercised by:
- All TTNN matmul Python tests (unit + nightly)
- All model tests that use matmul (SentenceBERT, BERT configs)
- The multicore_reuse programming example

The helper was also fixed with:
1. **`_with_dt` data format reconfiguration** in the spill/reload path
2. **`mm_init` with `interm_cb`** for correct pack configuration during K-blocking
3. **`transpose` template parameter** for both `matmul_block` and `matmul_tile`

---

## Detailed Results

### C++ Integration Tests (GTest)

```
MeshDispatchFixture.TensixMatmulTileHelperSmall      — PCC=0.999289 — PASSED
MeshDispatchFixture.TensixMatmulTileHelperRectangular — PCC=0.998758 — PASSED
MeshDispatchFixture.TensixMatmulTileHelperLarger      — PCC=0.997801 — PASSED
MeshDispatchFixture.TensixMatmulTileHelperSingleTile  — PCC=0.999742 — PASSED
MeshDispatchFixture.TensixMatmulSingleTile            — PASSED
MeshDispatchFixture.TensixMatmulMultiTile             — PASSED
MeshDispatchFixture.TensixMatmulBlock                 — PASSED
MeshDispatchFixture.TensixMatmulBlockInitShort        — PASSED
MeshDispatchFixture.TensixMatmulBlockInitShortWithDt  — PASSED
MeshDispatchFixture.TensixMatmulLargeBlock            — PASSED
MeshDispatchFixture.TensixMatmulSingleCoreSmall       — PASSED
[PASSED] 11 tests.
```

### Model Test: SentenceBERT on Blackhole

Full end-to-end model inference test on Blackhole p100a. Exercises matmul through multiple transformer layers including multi-head attention and feed-forward networks.
- **Result**: PASSED (73.75s)
- **Note**: Requires --timeout=300 in dev mode due to watcher/assert overhead

---

## Full Test Landscape (80+ files discovered)

Beyond what we ran, the repository contains 80+ test files that exercise matmul:
- **8 Python unit test files** in `tests/ttnn/unit_tests/operations/matmul/`
- **14 nightly test files** in `tests/ttnn/nightly/unit_tests/operations/matmul/` + moreh
- **14 C++ GTest files** in `tests/tt_metal/tt_metal/` (legacy root-level)
- **7 C++ integration tests** in `tests/tt_metal/tt_metal/integration/matmul/`
- **15 sweep test YAML configs** for GS/WH/BH parameter sweeps
- **5 DIDT determinism tests** (FF1, LM head, SDXL, DeepSeek, minimal)
- **20+ model-specific tests** (Falcon7B, Falcon40B, DeepSeek V3, Llama3 70B, SDXL, T5)
- **1 stress test** file with 10x repeat variants
- **CI workflows**: `tt-metal-l2-nightly-impl.yaml` (nightly matmul job), `ttnn-run-sweeps.yaml` (16 matmul sweep configs), `didt-tests.yaml`

---

## BH vs WH Differences Observed

1. **PCC values**: Slightly different from WH (expected — different FPU implementations). All within threshold.
2. **Core grid**: BH has different core grid geometry than WH. Tests using `CoreGrid(x=5, y=8)` work correctly.
3. **Dynamic throttling**: BH has firmware-controlled dynamic throttling for matmul_block. Invisible to helpers but affects performance.
4. **Skipped tests**: Many nightly tests are skipped on BH due to WH-specific markers.
