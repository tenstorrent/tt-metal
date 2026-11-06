// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_cosh_bf16_replay_window.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_cosh_bf16_hyperbolic_exp_core.h"
namespace ckernel::sfpu::bf16::cosh {
template <typename Config, int Iterations = 32>
inline void calculate_hyperbolic_exp() {
    static_assert(Iterations == 32 && Config::kDegree == 4);
    static_assert(Config::kBare);
    // The scale is reloaded per row into LREG4: LREG11 is the -1 constant other ops read.
    ttpoly::cosh::hyperbolic_exp_pins<Config>();
    constexpr unsigned body_slots = Config::kOdd ? 29u : 32u;
    // Rows are addressed through ADDR_MOD_7 alone; the DEST counter steps one row past the replay.
    auto row_head = [&] {
        ttpoly::cosh::hyperbolic_exp_core_head<
            ADDR_MOD_7,
            ttpoly::cosh::HyperbolicC0<Config>,
            ttpoly::cosh::HyperbolicScale<Config>>();
    };
    auto row_tail = [&] {
        ttpoly::cosh::hyperbolic_exp_core_tail<
            ADDR_MOD_7,
            ttpoly::cosh::HyperbolicC0<Config>,
            ttpoly::cosh::HyperbolicScale<Config>>();
        ttpoly::cosh::hyperbolic_even_store<ADDR_MOD_7>();
    };
    record_split<0, body_slots, 16>(row_head, row_tail);
    TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 32
    for (int row = 1; row < Iterations; ++row) {
        replay_split<0, body_slots, 16>(row_head, row_tail);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16::cosh
