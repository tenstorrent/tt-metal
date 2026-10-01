// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// Unlike the general SDPA helper, FW always refines the reciprocal, independent of APPROX.
template <bool is_fp32_dest_acc_en>
inline void calculate_sdpa_fw_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat out;
        if constexpr (is_fp32_dest_acc_en) {
            out = ckernel::sfpu::sfpu_reciprocal_iter<2>(in);
        } else {
            out = ckernel::sfpu::sfpu_reciprocal_iter<1>(in);
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg += 2;
    }
}

template <uint16_t scale_bf16, bool is_fp32_dest_acc_en>
inline void calculate_exponential_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    // bf16 arm: exp_21f's four fp32 constants, loaded once and kept in LRegs across the loop (sfpi 7.83.0
    // never hoists a literal out of a loop by itself, so each costs an SFPLOADI pair per row otherwise).
    // The fp32 arm keeps _ckernel_sfpu_exp_accurate_ as is, so these fold to unused floats there.
    HoistedIf<!is_fp32_dest_acc_en> one_ln2 = EXP_21F_ONE_LN2, c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            result = ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true, is_fp32_dest_acc_en>(val, scale_bf16);
        } else {
            // _ckernel_sfpu_exp_accurate_<true, false> with the constants from the caller: same operations.
            result = _sfpu_exp_21f_bf16_<false>(val * sfpi::sFloat16b(scale_bf16), one_ln2, c0, c1, c2);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
