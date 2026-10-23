// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_softplus_bf16_replay_window.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_softplus_bf16_min_max.h"
#include "sfpu/ckernel_sfpu_softplus_bf16_finite_reciprocal.h"
#include "sfpu/ckernel_sfpu_softplus_bf16_abs_exp_correction_core.h"
#include "sfpu/ckernel_sfpu_softplus_bf16_abs_exp_correction.h"
namespace ckernel::sfpu::bf16::softplus {
// The call loads every constant it reads, as stock's reads none its init sets. Wormhole's
// reciprocal still reads its Newton constant from Prgm0.
template <typename Config>
inline void init_abs_exp_correction() {}
}  // namespace ckernel::sfpu::bf16::softplus
