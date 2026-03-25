# Matmul Helper Call Site Audit

**Date**: 2026-03-25
**Branch**: wransom/llk3
**Hardware**: Blackhole p100a

## Overview

This audit examines all files under `ttnn/cpp/ttnn/operations/` that call `matmul_block` (the LLK function) or `matmul_tiles` (the LLK function) to determine which can use our helper wrappers and which cannot.

### Helpers Available
- **`matmul_tile`** (`matmul_tile_helpers.hpp`) — wraps `mm_init` + `matmul_tiles` for tile-at-a-time matmul with WaitMode and InitUninitMode
- **`matmul_block`** (`matmul_block_helpers.hpp`) — wraps sub-blocked matmul with spill/reload, using struct parameters

### Migrated
| File | Helper Used | When |
|------|-------------|------|
| `matmul/device/kernels/compute/bmm.cpp` | `matmul_tile` | Previous commit |
| `matmul/device/kernels/compute/bmm_large_block_zm.cpp` | `matmul_block` | This commit |
| `programming_examples/matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp` | `matmul_block` | Previous commit |

---

## Summary Table: matmul_block (LLK) Call Sites

| File | Pattern | Helper Applicable? | Blocker |
|------|---------|-------------------|---------|
| `matmul/.../bmm_large_block_zm.cpp` | Sub-blocked matmul with spill/reload | **MIGRATED** | Now uses matmul_block helper |
| `matmul/.../bmm_large_block_zm_fused_bias_activation.cpp` | Sub-blocked + fused bias + activation + tilize/untilize + transpose | **NO** | Fused bias/activation, conditional untilize, transpose block, ~500 lines of interleaved logic |
| `matmul/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp` | Same as above + gathered/all-reduce variant | **NO** | Same as above + multi-device concerns |
| `conv2d/.../conv_bmm_tilize.cpp` | matmul_block with tilize pre-processing and bias | **NO** | Fused tilize + matmul_block + bias, custom CB switching with `mm_block_init_short_with_both_dt` |
| `transformer/sdpa/.../compute_streaming.hpp` | Multiple matmul_block calls interleaved with softmax, reduce, scaling | **NO** | Complex fused SDPA pipeline — matmul is just one stage |
| `transformer/sdpa/.../compute_common.hpp` | Similar to streaming — fused QK and QKtV matmuls | **NO** | Same as above |
| `transformer/sdpa_decode/.../sdpa_flash_decode.cpp` | matmul_block in flash attention decode loop | **NO** | Interleaved with softmax, online correction, masking |
| `experimental/matmul/group_attn_matmul/.../transformer_group_attn_matmul.cpp` | Sub-blocked with transpose support and group repetition | **NO** | Custom group-repeat CB management, conditional transpose, matmul_tiles + matmul_block mixed |
| `experimental/conv3d/.../compute.cpp` | matmul_block for 3D convolution | **NO** | Custom volume-to-column processing interleaved with matmul |
| `experimental/minimal_matmul/.../compute.cpp` | matmul_block with custom init/short patterns | **Partial** | Uses `mm_init` + `mm_block_init_short` mid-loop; would need helper reinit support |
| `experimental/deepseek/mla/matmul_wo/.../compute.cpp` | matmul_block for ring-distributed matmul | **NO** | Multi-device ring communication interleaved with matmul |
| `experimental/ccl/llama_all_gather_matmul_async/.../compute.cpp` | Fused all-gather + matmul + bias + activation | **NO** | Same fused pattern as bmm_fused_bias_activation + CCL |
| `experimental/ccl/moe_compute/.../compute.cpp` | matmul_block for MoE gating | **NO** | Custom MoE dispatch logic interleaved with matmul |
| `experimental/deepseek/moe/moe_gate_mm/.../compute.cpp` | matmul_block for MoE gate matmul | **NO** | Fixed small-block matmul with custom MoE routing |

## Summary Table: matmul_tiles (LLK) Call Sites

| File | Pattern | Helper Applicable? | Blocker |
|------|---------|-------------------|---------|
| `matmul/.../bmm_large_block_zm.cpp` | Sub-blocked inner loop | **YES — matmul_block** | Same as above (full file is matmul_block pattern) |
| `moreh/moreh_matmul/.../moreh_matmul.cpp` | Tile-at-a-time with transpose and accumulation | **Partial** | Has transpose support, custom CB management per-dimension, conditional mm_init_short mid-loop |
| `reduction/generic/.../reduce_w.cpp` | matmul_tiles for reduce-by-matmul pattern | **NO** | Not actually matmul — uses matmul_tiles as reduce mechanism with scaler |
| `experimental/transformer/rotary_embedding_llama/.../rotary_embedding_llama.cpp` | matmul_tiles in rotation matrix application | **NO** | Embedded in complex rotary embedding pipeline with sin/cos CB switching |
| `experimental/transformer/fused_distributed_rmsnorm/.../rmsnorm_post_allgather.cpp` | matmul_tiles for weight multiplication in RMSNorm | **NO** | Part of fused RMSNorm pipeline |
| `experimental/matmul/attn_matmul/.../transformer_attn_matmul.cpp` | Tile-at-a-time with transpose and spill/reload | **Partial** | Similar to matmul_tile pattern but has transpose, spill/reload, and conditional `mm_init_short_with_dt` |
| `moreh/moreh_sum/.../moreh_sum_w.cpp` | matmul_tiles for sum reduction via matmul | **NO** | Reduce pattern, not matmul |
| `moreh/moreh_mean/.../moreh_mean_w.cpp` | matmul_tiles for mean reduction via matmul | **NO** | Reduce pattern, not matmul |
| `experimental/transformer/rotary_embedding_llama_fused_qk/.../*.cpp` (3 files) | matmul_tiles in fused QK rotary embedding | **NO** | Deep embedding pipeline |

---

## Gap Analysis: What Prevents Wider Helper Adoption

### Gap #1: Data Format Reconfiguration in Reload Path — FIXED
The `matmul_block` helper now uses `copy_tile_to_dst_init_short_with_dt(in1_cb, interm_cb)` and
`mm_init_short_with_dt(in0_cb, in1_cb, interm_cb, transpose)` in the reload path. It also inits
with `mm_init(in0_cb, in1_cb, interm_cb, transpose)` to match the production pattern.

### Gap #2: Transpose Support — FIXED
Both `matmul_tile` and `matmul_block` now accept a `transpose` template parameter (default: false).
It is threaded through to `mm_init`, `mm_init_short_with_dt`, etc.

### Gap #3: No Fused Bias/Activation (EXPECTED — OUT OF SCOPE)
The `bmm_large_block_zm_fused_bias_activation.cpp` pattern (matmul + bias add + SFPU activation) is the most common production variant but is explicitly out of scope. This is a ~500 line kernel that interleaves matmul, bcast_add, and various SFPU operations with complex init/reconfig sequences.

**Recommendation**: This should be a separate helper (e.g., `matmul_block_fused`) or addressed by the LLK API v2 composition model.

### Gap #4: No mm_block_init Support (LOW-MEDIUM)
Our `matmul_block` helper calls `mm_init` but the actual `matmul_block` LLK function requires `mm_block_init` which takes additional `ct_dim`, `rt_dim`, `kt_dim` parameters for the hardware unroll. We use `matmul_tiles` (the tile-level function) in a loop rather than calling the `matmul_block` LLK function directly.

This means our helper doesn't benefit from the hardware-level block optimizations that `mm_block_init` enables.

**Fix**: Switch the inner implementation to use `mm_block_init` + `matmul_block` (LLK) instead of `mm_init` + `matmul_tiles`.

---

## Coverage Summary

| Category | Count | Details |
|----------|-------|---------|
| **Migrated** | 3 | `bmm.cpp` (matmul_tile), `bmm_large_block_zm.cpp` (matmul_block), programming example |
| **Not migratable — fused pipelines** | 16+ | Fused bias/activation, SDPA, MoE, conv2d, CCL |
| **Not migratable — interleaved with other ops** | 5+ | attn_matmul (untilize/tilize), rmsnorm (reduce/bcast), rotary embedding |
| **Not actually matmul** | 5 | Reduce/sum/mean via matmul_tiles |

### Key Insight
The helper covers the **clean matmul pattern** well (2 files directly, 3 more with minor extensions). However, most production matmul call sites are deeply embedded in **fused computation pipelines** (SDPA, fused bias+activation, MoE, conv2d) where matmul is just one step among many. These will never use a standalone matmul helper — they need either:
1. A composition model (see LLK API v2 report)
2. A fused matmul+bias+activation helper
3. To remain hand-written (most likely for complex cases like SDPA)

### Recommended Next Steps
1. **Consider mm_block_init** — would improve performance for the block helper by using hardware-level block optimizations instead of tile-level `matmul_tiles` in a loop
2. **Don't try to absorb fused patterns** — they're fundamentally different and should stay hand-written or be addressed by API v2
3. **Future fused helper** — a `matmul_block_fused` helper for the matmul+bias+activation pattern could cover `bmm_large_block_zm_fused_bias_activation.cpp` and its variants, but this is a significant effort
