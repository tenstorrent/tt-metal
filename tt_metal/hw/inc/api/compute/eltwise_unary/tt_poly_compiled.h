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

namespace ckernel::sfpu {
inline void ttpoly_compiled_program_init() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_2;
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t poly_degree = POLY_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto* p_lut = &LUT_DATA;
#include "deployment/generic_lut_activation/kernels/compute/shared_generic_init.inc"
}

inline void ttpoly_compiled_tile() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_2;
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t poly_degree = POLY_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto* p_lut = &LUT_DATA;
#include "deployment/generic_lut_activation/kernels/compute/shared_generic_tile.inc"
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
