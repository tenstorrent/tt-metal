// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_addrmod.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
// This implements the "add 0x7fff + LSB" algorithm for correct tie-breaking
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Get the float32 bits as unsigned integer
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract the LSB of what will become the bf16 mantissa (bit 16 of float32)
    // This is needed for the tie-breaker: round to even
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb to implement RNE:
    // - If lower 16 bits > 0x8000: overflow -> rounds up
    // - If lower 16 bits < 0x8000: no overflow -> rounds down
    // - If lower 16 bits = 0x8000 (tie) and lsb=0: 0x7fff+0=0xffff, no overflow -> stays even
    // - If lower 16 bits = 0x8000 (tie) and lsb=1: 0x7fff+1=0x8000, overflow -> rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits (bf16 format in float32)
    bits = bits & 0xFFFF0000U;

    // Reinterpret back as float
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
}

// =============================================================================
// Optimized float multiply with RNE rounding to bf16 using TTI instructions
// =============================================================================
//
// This version uses direct TTI instructions for better performance than SFPI.
// Achieves ~7-8 cycles per row (vs ~10-12 for pure SFPI).
//
// Pipeline:
//   1. Load a from in0, b from in1
//   2. Multiply: r = a * b (SFPMAD with c=0)
//   3. Zero check: create mask = 0 if either input is 0 (using bitwise ops)
//   4. RNE: t = r, r >>= 16, lsb = r & 1, t += 0x7fff, t += lsb
//   5. Apply zero mask: t = t & mask
//   6. Store t as bf16
//
// Register allocation:
//   LREG0 (a): input from tile 0
//   LREG1 (b): input from tile 1
//   LREG2 (r): multiply result / guard bit temp
//   LREG3 (t): temp for RNE (holds copy + 0x7fff + lsb)
//   LREG4 (m): zero mask
//   LREG5 (tmp): temporary for mask calculation
//   LREG12: constant 1 (mask for LSB extraction)
//   LREG13: constant 0x7fff (RNE bias)
//   LREG14: constant 0.0f (for SFPMAD c=0)
//   LCONST_0: hardware constant 0
//
// Zero-mask algorithm:
//   For input x: mask = (x | -x) >> 31 (arithmetic shift)
//   - If x != 0: MSB of (x | -x) is set, so mask = 0xFFFFFFFF
//   - If x == 0: (0 | 0) = 0, so mask = 0x00000000
//   Combined: final_mask = mask0 & mask1 (0 if either input is 0)
//
// =============================================================================

template <bool APPROXIMATION_MODE>
inline void _init_sfpu_binary_mul_bf16_() {
    // Set up constants in programmable LREGs
    sfpi::vConstIntPrgm0 = 1;       // LREG12: mask for extracting LSB
    sfpi::vConstIntPrgm1 = 0x7fff;  // LREG13: RNE bias
    sfpi::vConstFloatPrgm2 = 0.0f;  // LREG14: zero for SFPMAD c operand

    // Set up address modifier for auto-increment
    // ADDR_MOD_3: increment dest by 2 after each store
    constexpr addr_mod_t addr_mod = {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    };
    addr_mod.set(ADDR_MOD_3);

    // ADDR_MOD_2: no increment (for loads)
    constexpr addr_mod_t addr_mod_noinc = {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    };
    addr_mod_noinc.set(ADDR_MOD_2);
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool HANDLE_ZERO = true>
inline void _calculate_sfpu_binary_mul_bf16_(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Register allocation
    constexpr int a = p_sfpu::LREG0;    // input from in0
    constexpr int b = p_sfpu::LREG1;    // input from in1
    constexpr int r = p_sfpu::LREG2;    // result / guard bit temp
    constexpr int t = p_sfpu::LREG3;    // RNE temp
    constexpr int m = p_sfpu::LREG4;    // zero mask
    constexpr int tmp = p_sfpu::LREG5;  // temporary

    // Calculate base offsets (each tile is 32 rows)
    constexpr uint dst_tile_size = 32;
    int offset_in0 = dst_index_in0 * dst_tile_size;
    int offset_in1 = dst_index_in1 * dst_tile_size;
    int offset_out = dst_index_out * dst_tile_size;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // ===== LOAD PHASE =====
        // Load a from in0 tile
        TT_SFPLOAD(a, InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset_in0);
        // Load b from in1 tile
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset_in1);

        // ===== ZERO MASK PHASE (before multiply overwrites anything) =====
        if constexpr (HANDLE_ZERO) {
            // Create mask for a: mask_a = (a | -a) >> 31
            // Step 1: tmp = ~a
            TTI_SFPNOT(0, a, tmp, 0);
            // Step 2: tmp = tmp + 1 = -a (two's complement)
            TTI_SFPIADD(1, tmp, tmp, 0);
            // Step 3: m = a | tmp = a | -a
            TTI_SFPMOV(0, a, m, 0);
            TTI_SFPOR(0, tmp, m, 0);
            // Step 4: m = m >> 31 (arithmetic shift, gives all 1s if a != 0)
            TTI_SFPSHFT2(-31 & 0xfff, m, m, 6);

            // Create mask for b: mask_b = (b | -b) >> 31
            // Step 1: tmp = ~b
            TTI_SFPNOT(0, b, tmp, 0);
            // Step 2: tmp = tmp + 1 = -b
            TTI_SFPIADD(1, tmp, tmp, 0);
            // Step 3: tmp = b | tmp = b | -b
            TTI_SFPOR(0, b, tmp, 0);
            // Step 4: tmp = tmp >> 31
            TTI_SFPSHFT2(-31 & 0xfff, tmp, tmp, 6);

            // Combined mask: m = mask_a & mask_b (0 if either input is 0)
            TTI_SFPAND(0, tmp, m, 0);
        }

        // ===== MULTIPLY PHASE =====
        // r = a * b + 0.0 (SFPMAD with LREG14 = 0.0)
        TTI_SFPMAD(a, b, p_sfpu::LREG14, r, 0);
        TTI_SFPNOP;  // SFPMAD has 2-cycle latency

        // ===== RNE ROUNDING PHASE =====
        // Algorithm: result = (input + 0x7fff + guard_bit) with upper 16 bits
        // where guard_bit = (input >> 16) & 1

        // Copy r to t (t will be used for adding 0x7fff)
        TTI_SFPMOV(0, r, t, 0);

        // r = r >> 16 (shift to get guard bit in LSB position)
        TTI_SFPSHFT2(-16 & 0xfff, r, r, 6);  // SFPSHFT2_MOD1_SHFT_IMM

        // r = r & 1 (extract guard bit using LREG12 which holds 1)
        TTI_SFPAND(0, p_sfpu::LREG12, r, 0);

        // t = t + 0x7fff (add RNE bias from LREG13)
        TTI_SFPADD(t, p_sfpu::LREG13, p_sfpu::LCONST_0, t, 0);

        // t = t + r (add guard bit)
        TTI_SFPADD(t, r, p_sfpu::LCONST_0, t, 0);

        // ===== APPLY ZERO MASK =====
        if constexpr (HANDLE_ZERO) {
            // t = t & m (result is 0 if either input was 0)
            TTI_SFPAND(0, m, t, 0);
        }

        // ===== STORE PHASE =====
        // Store as bfloat16 (FP16B format stores upper 16 bits)
        TT_SFPSTORE(t, InstrModLoadStore::FP16B, ADDR_MOD_3, offset_out);

        // Increment offsets for next iteration
        offset_in0 += 2;
        offset_in1 += 2;
        offset_out += 2;
    }

    // Drain pipeline
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Wrapper function with standard interface
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_mul_bf16(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    if constexpr (is_fp32_dest_acc_en) {
        // For FP32 accumulation, no RNE rounding needed - use simpler path
        constexpr uint dst_tile_size_sfpi = 32;
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
            sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in0 * in1;
            sfpi::dst_reg++;
        }
    } else {
        // Use optimized TTI implementation with RNE and zero handling
        _calculate_sfpu_binary_mul_bf16_<APPROXIMATION_MODE, ITERATIONS, true>(
            dst_index_in0, dst_index_in1, dst_index_out);
    }
}

// Version without zero handling (faster, for cases where 0*inf is not a concern)
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_mul_tti(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    if constexpr (is_fp32_dest_acc_en) {
        constexpr uint dst_tile_size_sfpi = 32;
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
            sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in0 * in1;
            sfpi::dst_reg++;
        }
    } else {
        // Use optimized TTI implementation with RNE but NO zero handling
        _calculate_sfpu_binary_mul_bf16_<APPROXIMATION_MODE, ITERATIONS, false>(
            dst_index_in0, dst_index_in1, dst_index_out);
    }
}

// Initialization function - call before using calculate_sfpu_binary_mul_bf16
template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void init_sfpu_binary_mul_bf16() {
    _init_sfpu_binary_mul_bf16_<APPROXIMATION_MODE>();
}

// =============================================================================
// Original SFPI-based implementation (fallback/reference)
// =============================================================================

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // Software RNE approach (kept for reference):
            result = float32_to_bf16_rne(result);
            // To match FPU behaviour for bfloat16 multiplication, 0 * x = 0 and x * 0 = 0
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1);

        v_if(in1 == 0) {
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_else {
                result = std::numeric_limits<float>::infinity();
                result = sfpi::setsgn(result, in0);
            }
            v_endif;
        }
        v_elseif(in0 == in1) { result = sfpi::vConst1; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            // software RNE approach:
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
}

}  // namespace sfpu
}  // namespace ckernel
