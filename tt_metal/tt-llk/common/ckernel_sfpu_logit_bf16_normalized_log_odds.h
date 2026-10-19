// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_logit_bf16_horner.h"
#include "ckernel_sfpu_logit_bf16_normalized_log_odds_core.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::logit
{
// No init: the core reads every constant from Config::kLut, none from the programmable registers.
template <typename Config, int Iterations = 32>
inline void calculate_normalized_log_odds()
{
    static_assert(Iterations == 32, "normalized log odds requires a complete tile");
    ttpoly::logit::normalized_log_odds_tile<Config>(Config::kLut, [](sfpi::vFloat x) { return x; });
}
} // namespace ckernel::sfpu::bf16::logit
