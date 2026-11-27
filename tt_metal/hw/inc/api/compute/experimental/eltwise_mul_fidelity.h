// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"

namespace ckernel {

// Explicit-fidelity forms of mul_reuse_dest_{init,tiles} and mul_{bcast_scalar_init,tiles_bcast_scalar}.
// Standard compute APIs bind MATH_FIDELITY (a MATH-thread-only macro); fused kernels that mix fidelities
// within one compute program use these wrappers to override fidelity per op.

/** Explicit-fidelity form of mul_reuse_dest_init. Pair with mul_reuse_dest_tiles_fidelity. */
template <EltwiseBinaryReuseDestType reuse_dest, MathFidelity math_fidelity>
ALWI void mul_reuse_dest_init_fidelity(std::uint32_t icb, std::uint32_t call_line = __builtin_LINE()) {
    static_assert(
        reuse_dest != EltwiseBinaryReuseDestType::NONE,
        "reuse_dest must be DEST_TO_SRCA or DEST_TO_SRCB; for the two-operand op call mul_init(icb0, icb1).");
    state_configure(icb, call_line);
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, acc_to_dest, reuse_dest>(false, false, icb)));
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, math_fidelity, reuse_dest>(
        icb, icb, false /* acc_to_dest */)));
}

/** Explicit-fidelity form of mul_reuse_dest_tiles. Pair with mul_reuse_dest_init_fidelity. */
template <EltwiseBinaryReuseDestType reuse_dest, MathFidelity math_fidelity, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_reuse_dest_tiles_fidelity(
    std::uint32_t in_cb_id, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
#ifndef ARCH_QUASAR
    UNPACK(constexpr bool acc_to_dest = true);
#else
    UNPACK(constexpr bool acc_to_dest = false);
#endif
    UNPACK((llk_unpack_A<BroadcastType::NONE, acc_to_dest, reuse_dest>(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          is_fp32_dest_acc_en,
          math_fidelity,
          reuse_dest>(in_cb_id, in_cb_id, dst_tile_index, true /* clear_fp32_dst_acc */)));
}

/** Explicit-fidelity form of mul_bcast_scalar_init. Pair with mul_tiles_bcast_scalar_fidelity. */
template <MathFidelity math_fidelity>
ALWI void mul_bcast_scalar_init_fidelity(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR, math_fidelity>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

/** Explicit-fidelity form of mul_tiles_bcast_scalar. Pair with mul_bcast_scalar_init_fidelity. */
template <MathFidelity math_fidelity, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_tiles_bcast_scalar_fidelity(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t itile0, std::uint32_t itile1, std::uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::SCALAR,
          is_fp32_dest_acc_en,
          math_fidelity,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

}  // namespace ckernel
