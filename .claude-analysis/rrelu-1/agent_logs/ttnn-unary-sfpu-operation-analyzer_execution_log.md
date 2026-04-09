# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Session Info (prelu_sfpu - prior run)
- **Operation**: prelu_sfpu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-09
- **Status**: SUCCESS

## Summary (prelu_sfpu)
Analyzed the SFPU kernel implementation for the `PRELU_SFPU` unary operation. The operation was deep-nuked from this repository, so the analysis was reconstructed from pre-nuke documentation, surviving reference implementations (threshold, hardtanh, clamp), and the shared dispatch infrastructure.

## Key Findings (prelu_sfpu)
- **Formula**: `max(0, x) + weight * min(0, x)` = `x if x >= 0, weight * x if x < 0`
- **Macro group**: `SFPU_OP_PRELU_INCLUDE` (standalone)
- **API signature**: `prelu_tile(uint32_t idst, uint32_t param0)` + `prelu_tile_init()`
- **Kernel style**: SFPI-based (Style A) using `v_if`/`v_endif` for sign-conditional branching
- **Parameter**: Single `uint32_t param0` (bit-cast weight float)
- **Approximation mode**: `false` (no approximation-dependent branches in the algorithm)
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPMAD (multiply), SFPSETCC/SFPENCC/SFPPUSHC/SFPPOPC (CC management)
- **Address mode**: ADDR_MOD_7 with dest.incr=0 (standard for most unary SFPU ops)

## Deep-Nuke Impact (prelu_sfpu)
The following files were confirmed removed:
- `unary_op_types.hpp`: PRELU_SFPU enum entry removed
- `unary_op_utils.cpp`: get_op_init_and_func / get_macro_definition cases removed
- `prelu.h`: API header removed from `tt_metal/hw/inc/api/compute/eltwise_unary/`
- `ckernel_sfpu_prelu.h`: Core SFPU kernel removed from both WH and BH tt_llk directories
- `ckernel_sfpu_relu.h`: Emptied (just `#pragma once`)
- Per-operation LLK dispatch files: Removed from both tt_llk and hw/ckernels

## Reconstruction Confidence: HIGH
The reconstruction is based on well-documented formula, confirmed API signature from Doxygen docs, and structurally identical surviving operations.

## Output (prelu_sfpu)
- **Analysis file**: `.claude-analysis/rrelu-1/prelu_sfpu_analysis.md`

---

## Session Info (threshold)
- **Operation**: threshold
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-09
- **Status**: SUCCESS

## Summary (threshold)
Analyzed the SFPU kernel implementation for the `THRESHOLD` unary operation. While the API header and LLK dispatch files have been nuked, the core SFPU kernel (`ckernel_sfpu_threshold.h`) survives intact in both Wormhole B0 and Blackhole tt_llk directories. The analysis covers the complete SFPU implementation.

## Key Findings (threshold)
- **Formula**: `if (in <= threshold) { out = value } else { out = in }`
- **API signature**: `threshold_tile(uint32_t idst, uint32_t param0, uint32_t param1)` + `threshold_tile_init()`
- **Kernel style**: SFPI-based (Style A) using `v_if`/`v_endif` for LTE conditional
- **Parameters**: Two `uint32_t` params (threshold value and replacement value, both bitcast from float)
- **Approximation mode**: `false` (APPROXIMATION_MODE template param is unused in the function body)
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPMAD, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPLOADI
- **Address mode**: ADDR_MOD_7 with dest.incr=0 (same on WH and BH)
- **WH/BH parity**: Core kernel is byte-for-byte identical on both architectures

## Execution Timeline (threshold)
1. Read reference files (sfpu-hardware-model.md, diagram-templates.md, logging docs)
2. Traced dispatch in unary_op_utils.cpp -- THRESHOLD in enum but dispatch cases nuked
3. Found core SFPU kernel in tt_llk (WH + BH), confirmed identical
4. Read LLK dispatch infrastructure (params, init, addr_mod)
5. Read sfpi.h for v_if/v_endif, vFloat comparison instruction mapping
6. Verified all identifiers and file paths via grep
7. Wrote threshold_analysis.md

## Output (threshold)
- **Analysis file**: `.claude-analysis/rrelu-1/threshold_analysis.md`
