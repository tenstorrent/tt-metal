// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_mem_descriptor.h"

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_binary.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_AB.h"
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// Id-free (2.0) two-operand eltwise binary (add/sub/mul). Each op takes one LLKOperand per input
// (A -> SrcA, B -> SrcB), format+geometry as NTTPs, L1 addresses the only runtime state. Binary is
// FORMAT-FREE at the op level: the src/dst register formats are programmed at compute_kernel_hw_startup;
// the op forwards only geometry (from operand A, mirroring legacy) + the two per-tile L1 addresses. Packing
// is done separately via the id-free experimental::pack_tile. The dest-reuse and broadcast variants are not
// part of this Phase-1 surface (BroadcastType::NONE only).

// clang-format off
/**
 * Paired init for a two-operand eltwise binary op. Configures MATH (always) and, when full_init, the AB
 * unpacker. Geometry is taken from operand A. compute_kernel_hw_startup(a_cb, b_cb, out_cb) must already
 * have programmed the formats.
 *
 * | Template | full_init           | Do the UNPACK init too (not just MATH) | bool             |  | True |
 * | Template | eltwise_binary_type | ELWADD / ELWSUB / ELWMUL               | EltwiseBinaryType |  | True |
 * | Function | a / b               | Input operands (A -> SrcA, B -> SrcB)  | LLKOperand       |  | True |
 * | Function | acc_to_dest         | Accumulate the result into DST         | bool             |  | False |
 */
// clang-format on
template <
    bool full_init,
    EltwiseBinaryType eltwise_binary_type,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void binary_tiles_init(
    LLKOperand<AFormat, AShape> /*a*/, LLKOperand<BFormat, BShape> /*b*/, bool acc_to_dest = false) {
    MATH((llk_math_eltwise_binary_init<
          LLKOperand<AFormat, AShape>::descriptor,
          eltwise_binary_type,
          BroadcastType::NONE,
          MATH_FIDELITY>(acc_to_dest)));
    if constexpr (full_init) {
        UNPACK((llk_unpack_AB_init<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(
            ckernel::Transpose::None)));
    }
}

template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void add_init(LLKOperand<AFormat, AShape> a, LLKOperand<BFormat, BShape> b, bool acc_to_dest = false) {
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(a, b, acc_to_dest);
}

template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void sub_init(LLKOperand<AFormat, AShape> a, LLKOperand<BFormat, BShape> b, bool acc_to_dest = false) {
    binary_tiles_init<true, EltwiseBinaryType::ELWSUB>(a, b, acc_to_dest);
}

template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void mul_init(LLKOperand<AFormat, AShape> a, LLKOperand<BFormat, BShape> b, bool acc_to_dest = true) {
    binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(a, b, acc_to_dest);
}

namespace detail {
// Per-tile L1 base for operand X at tile index (16B words); stride folds to a constant from X's geometry.
// Assumes fifo_page_size == a single tile's size (exact for linear formats), consistent with tilize/untilize.
template <DataFormat Format, TensorShape Shape>
ALWI std::uint32_t tile_address(LLKOperand<Format, Shape> op, std::uint32_t tile_index) {
    constexpr std::uint32_t stride =
        SCALE_DATUM_SIZE(static_cast<std::uint32_t>(Format), Shape.total_tensor_size()) >> 4;
    return op.l1_address + tile_index * stride;
}
}  // namespace detail

// clang-format off
/**
 * Element-wise C = A [op] B for one tile pair, writing DST[idst]. Pair with the matching *_init. The DST
 * register must be acquired. itile0/itile1 index within operand A/B; idst indexes DST. Geometry (for MATH)
 * comes from operand A, matching the legacy op.
 *
 * | Function | a / b           | Input operands                      | LLKOperand | | True |
 * | Function | itile0 / itile1 | Tile indices within A / B           | uint32_t   | | True |
 * | Function | idst            | DST register index for the result   | uint32_t   | | True |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void add_tiles(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst) {
    UNPACK((llk_unpack_AB<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(
        detail::tile_address(a, itile0), detail::tile_address(b, itile1))));
    MATH((llk_math_eltwise_binary<
          LLKOperand<AFormat, AShape>::descriptor,
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
}

template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void sub_tiles(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst) {
    UNPACK((llk_unpack_AB<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(
        detail::tile_address(a, itile0), detail::tile_address(b, itile1))));
    MATH((llk_math_eltwise_binary<
          LLKOperand<AFormat, AShape>::descriptor,
          EltwiseBinaryType::ELWSUB,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
}

template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void mul_tiles(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst) {
    UNPACK((llk_unpack_AB<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(
        detail::tile_address(a, itile0), detail::tile_address(b, itile1))));
    MATH((llk_math_eltwise_binary<
          LLKOperand<AFormat, AShape>::descriptor,
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
