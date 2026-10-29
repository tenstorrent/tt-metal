// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>
#include "llk_math_common_api.h"
#include "experimental/llk_math_sdpa_custom_mm_reuse_dest_srcb.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK SDPA_CUSTOM_MM_REUSE_DEST_SRCB
 *
 * Custom matmul that uses MOP to loop both srcA and srcB along inner dim. Output height
 * and width should be single tile with tile shape [1, 32]. Further work will uplift the
 * custom mm to support for tiles along the width.
 *
 * Uses llk_math_sdpa_custom_mm_reuse_dest_srcb.h as the low-level implementation.
 *************************************************************************/

template <MathFidelity math_fidelity>
inline void llk_math_sdpa_custom_mm_reuse_dest_srcb_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t kt_dim) {
    SAN_HOOK(unsupported());
    // An integer at the former transpose position binds here as kt_dim = 0 or 1; both are below the
    // smallest K the unpack MOP can feed (2), so both are caught here.
    LLK_ASSERT(kt_dim >= 2, "llk_math_sdpa_custom_mm_reuse_dest_srcb_init: kt_dim must be at least 2");
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    // Issue #31387: this flag is only for computing 8x32 tile shape, although current impl assumes the in0 tile is
    // still 16x32. We should remove this flag in the future and add impl for 8x32 input tile shape
    const bool partial_face = 0;

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    // The LLK's transpose argument is unused: operand B is always used as stored.
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_init_<math_fidelity>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, 0 /*transpose*/, kt_dim);
}

// Legacy (operandA, operandB, transpose) argument lists have the same arity as the current
// (operandA, operandB, kt_dim) list. This overload is viable only for an exact bool at the old
// transpose position and is deleted, so a bool transpose does not re-bind as kt_dim; integers
// resolve to the live overload and its kt_dim assert. A transpose of operand B is not
// implemented: see the transpose_v static_assert in compute_sdpa_chunk.
template <MathFidelity math_fidelity, typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
inline void llk_math_sdpa_custom_mm_reuse_dest_srcb_init(
    const std::uint32_t operandA, const std::uint32_t operandB, T transpose) = delete;

// Runtime parameter signal_output:
//   false: Full sdpa_custom_mm_reuse_dest_srcb - MVMULs + finalization
//   true: Partial K accumulation - MVMULs only, no finalization (for intermediate K subblocks)
//
// Compile-time parameter output_granularity:
//   Number of output tiles produced between successive FPU->SFPU semaphore posts when
//   signal_output is true. nt_dim must be divisible by output_granularity.
template <MathFidelity math_fidelity, std::uint32_t output_granularity, std::uint32_t input_granularity = 1>
inline void llk_math_sdpa_custom_mm_reuse_dest_srcb(
    const std::uint32_t src_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t nt_dim,
    bool signal_output) {
    SAN_HOOK(unsupported());
    // An integer at the former transpose position binds here as kt_dim = 0 or 1; both are below the
    // smallest K the unpack MOP can feed (2), so both are caught here.
    LLK_ASSERT(
        kt_dim >= 2 && nt_dim >= 1,
        "llk_math_sdpa_custom_mm_reuse_dest_srcb: kt_dim must be at least 2 and nt_dim at least 1");
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_<output_granularity, input_granularity>(
        src_index, dst_index, false /*transpose*/, kt_dim, nt_dim, signal_output);
}

// Legacy (src_index, dst_index, transpose, kt_dim, nt_dim) argument lists have the same arity
// as the current (src_index, dst_index, kt_dim, nt_dim, signal_output) list. This overload is
// viable only for an exact bool at the old transpose position and is deleted, so a bool
// transpose does not re-bind as kt_dim; integers resolve to the live overload and its asserts.
// A transpose of operand B is not implemented: see the transpose_v static_assert in
// compute_sdpa_chunk.
template <
    MathFidelity math_fidelity,
    std::uint32_t output_granularity,
    std::uint32_t input_granularity = 1,
    typename T,
    std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
inline void llk_math_sdpa_custom_mm_reuse_dest_srcb(
    const std::uint32_t src_index,
    const std::uint32_t dst_index,
    T transpose,
    const std::uint32_t kt_dim,
    const std::uint32_t nt_dim) = delete;
