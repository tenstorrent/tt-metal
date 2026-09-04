// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_op.h"
#include "sfpu/ckernel_sfpu_binary_comp.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// BinaryComp<APPROX, RELATIONAL_OP, FORMAT, DST_SYNC, DST_ACCUM, SIGN_MAGNITUDE_FORMAT, ITERATIONS>. Same
// interface as WH/BH. Quasar only has the signed Int32 relational kernel (lt/gt/le/ge); SIGN_MAGNITUDE_FORMAT
// selects the sign-magnitude Int32 dest encoding produced by Int8 copy_tile + fp32_dest_acc FPU.
//   calculate(in0, in1, out, vector_mode) -> calculate_binary_comp_int32
//   init()                                -> _llk_math_eltwise_sfpu_init_()
// Backs gt_int_tile / gt_int_tile_init and llk_math_eltwise_binary_sfpu_gt_int.
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    BinaryCompMode RELATIONAL_OP,
    DataFormat FORMAT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    bool SIGN_MAGNITUDE_FORMAT = false,
    int ITERATIONS = SFPU_ITERATIONS>
struct BinaryComp
    : SfpuBinaryOp<
          BinaryComp<APPROXIMATION_MODE, RELATIONAL_OP, FORMAT, DST_SYNC, DST_ACCUM, SIGN_MAGNITUDE_FORMAT, ITERATIONS>,
          DST_SYNC,
          DST_ACCUM> {
    static_assert(FORMAT == DataFormat::Int32, "Quasar SFPU binary comparison currently supports Int32 only");
    static_assert(
        RELATIONAL_OP == BinaryCompMode::Lt || RELATIONAL_OP == BinaryCompMode::Gt ||
            RELATIONAL_OP == BinaryCompMode::Le || RELATIONAL_OP == BinaryCompMode::Ge,
        "Quasar SFPU binary comparison supports BinaryCompMode::Lt, Gt, Le, Ge");

    static void kernel(std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out) {
        calculate_binary_comp_int32<APPROXIMATION_MODE, ITERATIONS, RELATIONAL_OP, SIGN_MAGNITUDE_FORMAT>(
            dst_index_in0, dst_index_in1, dst_index_out);
    }
};

}  // namespace sfpu
}  // namespace ckernel
