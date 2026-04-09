// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_mul_int.h"

namespace ckernel {
namespace sfpu {

// Aliases: local LLK wrapper uses mul_int32/mul_int32_init names
// but the tt_llk submodule defines _mul_int_ / _init_mul_int_
template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    _init_mul_int_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void mul_int32(uint32_t a, uint32_t b, uint32_t c) {
    _mul_int_<APPROXIMATION_MODE, 8>(a, b, c);
}

}  // namespace sfpu
}  // namespace ckernel
