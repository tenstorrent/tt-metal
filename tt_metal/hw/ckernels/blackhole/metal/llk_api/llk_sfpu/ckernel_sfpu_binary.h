// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "llk_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROX_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
}

template <ApproximationMode APPROX_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROX_MODE, BINOP>();
}

}  // namespace sfpu
}  // namespace ckernel
