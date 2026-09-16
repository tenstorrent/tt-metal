// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifndef TT_POLY_SELECTED_CONFIG_HEADER
#error "one typed compiler configuration is required"
#endif

// The included canonical source provides its own helpers and shared math.
// No alternate evaluator is emitted, and no other selected config may coexist.
#define TTPOLY_LLK_COMPILATION 1
#include TT_POLY_SELECTED_CONFIG_HEADER
#undef TTPOLY_LLK_COMPILATION

#if defined(FUSE_GRAD_MUL) != defined(TT_POLY_BACKWARD_INPUTS)
#error "gradient-bearing math requires the explicit two-input lifecycle"
#endif

#ifdef TT_POLY_SHARED_RATIONAL
#ifndef TT_ACT_RATIONAL_LUT
#error "rational lifecycle requires the compiler's rational configuration"
#endif
#ifdef DST_COEFF_PRELOAD_ONCE
#error "caller tile-index preload is not selected by the canonical compiler"
#endif
#endif

namespace ckernel::sfpu {
inline void ttpoly_compiled_program_init() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_2;
    constexpr uint32_t lut_size = LUT_SIZE;
#ifdef TT_POLY_SHARED_RATIONAL
    constexpr uint32_t num_degree = NUM_DEGREE;
    constexpr uint32_t den_degree = DEN_DEGREE;
#else
    constexpr uint32_t poly_degree = POLY_DEGREE;
#endif
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto* p_lut = &LUT_DATA;
#ifdef TT_POLY_SHARED_RATIONAL
#include "deployment/generic_lut_activation/kernels/compute/shared_rational_init.inc"
#else
#include "deployment/generic_lut_activation/kernels/compute/shared_generic_init.inc"
#endif
}

inline void ttpoly_compiled_tile() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_2;
    constexpr uint32_t lut_size = LUT_SIZE;
#ifdef TT_POLY_SHARED_RATIONAL
    constexpr uint32_t num_degree = NUM_DEGREE;
    constexpr uint32_t den_degree = DEN_DEGREE;
#else
    constexpr uint32_t poly_degree = POLY_DEGREE;
#endif
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto* p_lut = &LUT_DATA;
#ifdef TT_POLY_SHARED_RATIONAL
#include "deployment/generic_lut_activation/kernels/compute/shared_rational_tile.inc"
#else
#include "deployment/generic_lut_activation/kernels/compute/shared_generic_tile.inc"
#endif
}
}  // namespace ckernel::sfpu

namespace ckernel {
#ifdef TT_POLY_BACKWARD_INPUTS
// Exactly the canonical binary transport order. The last copy establishes the
// DST0 base consumed by both shared evaluator families; gradients remain DST1.
ALWI void ttpoly_compiled_backward_copy(uint32_t cb_input, uint32_t cb_grad) {
    cb_wait_front(cb_grad, 1);
    copy_tile(cb_grad, 0, 1);
    copy_tile(cb_input, 0, 0);
}
#endif
#if defined(TT_POLY_SHARED_RATIONAL) && TT_POLY_INPUT_DST_TILE == 1
// Identical transport to the canonical rational class-shadow branch: one
// input copy supplies DST1, then math rebases to DST0 before the shared body.
ALWI void ttpoly_compiled_copy(uint32_t cb_input) {
    copy_tile(cb_input, 0, 1);
#ifdef TRISC_MATH
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::get_dest_buffer_base());
#endif
}
#endif
// Caller has performed compute_kernel_hw_startup and copy_init exactly once.
ALWI void ttpoly_compiled_program_init() {
    MATH(sfpu::ttpoly_compiled_program_init());
#ifdef PACK_RELU_MODE
    // Identical pack-stage policy to the canonical whole-program shell.
    // pack_relu_config itself is PACK-gated by the compute API.
#if PACK_RELU_MODE == 1
    pack_relu_config(ckernel::ReluConfig::zero());
#elif PACK_RELU_MODE == 2
    pack_relu_config(ckernel::ReluConfig::min_threshold(PACK_RELU_THRESHOLD));
#elif PACK_RELU_MODE == 3
    pack_relu_config(ckernel::ReluConfig::max_threshold(PACK_RELU_THRESHOLD));
#else
#error "PACK_RELU_MODE must be 1, 2 or 3"
#endif
#endif
}
ALWI void ttpoly_compiled_program_finish() {
#ifdef PACK_RELU_MODE
    // State persists across kernels: disarm after the final pack/release.
    pack_relu_config(ckernel::ReluConfig::none());
#endif
}
// The preceding copy hook obeys the compiler's DST0/DST1 transport contract.
// The shared fragment owns traversal and any nested native SFPU calls:
// adding an outer SFPU start/done would add work absent from deployment.
ALWI void ttpoly_compiled_tile() { MATH(sfpu::ttpoly_compiled_tile()); }
}  // namespace ckernel
