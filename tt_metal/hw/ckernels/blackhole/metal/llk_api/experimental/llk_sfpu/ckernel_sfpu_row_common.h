// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared SFPU lane/subvector primitives for per-row Blackhole LLKs (row_norm and softmax_k).
#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu::row_common {

constexpr int kFaceVecs = 8;
constexpr int kTileVecs = 32;

template <std::uint32_t NUM_ROWS>
constexpr bool valid_num_rows = (NUM_ROWS == 1 || NUM_ROWS == 2 || NUM_ROWS == 4 || NUM_ROWS == 8);

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat round_for_dest(sfpi::vFloat v) {
    if constexpr (!is_fp32_dest_acc_en) {
        v = sfpi::convert<sfpi::vFloat16b>(v, sfpi::RoundMode::Nearest);
    }
    return v;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline void store(int vec, sfpi::vFloat v) {
    sfpi::dst_reg[vec] = round_for_dest<is_fp32_dest_acc_en>(v);
}

// Sum across the 8 columns of each 4-row subvector, broadcast to all 8 lanes.
sfpi_inline sfpi::vFloat subvec_sum(sfpi::vFloat v) {
    v += sfpi::subvec_shflror1(v);
    v += sfpi::subvec_shflror1(sfpi::subvec_shflror1(v));
    v += sfpi::subvec_shflror1(sfpi::subvec_shflror1(sfpi::subvec_shflror1(sfpi::subvec_shflror1(v))));
    return v;
}

// Max across the 8 columns of each 4-row subvector, broadcast to all 8 lanes.
sfpi_inline sfpi::vFloat subvec_max(sfpi::vFloat v) {
    sfpi::vFloat r = sfpi::subvec_shflror1(v);
    v = sfpi::max(v, r);
    r = sfpi::subvec_shflror1(sfpi::subvec_shflror1(v));
    v = sfpi::max(v, r);
    r = sfpi::subvec_shflror1(sfpi::subvec_shflror1(sfpi::subvec_shflror1(sfpi::subvec_shflror1(v))));
    return sfpi::max(v, r);
}

}  // namespace ckernel::sfpu::row_common
