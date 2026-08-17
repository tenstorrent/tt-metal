// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "llk_math_common.h"

inline std::uint32_t llk_math_get_compute_special_value_flags() { return _llk_math_get_compute_special_value_flags_(); }

template <bool isFpu>
inline std::uint32_t llk_math_extract_compute_special_value_flags(std::uint32_t special_value_flags_reg) {
    constexpr std::uint32_t special_value_flags_mask = isFpu ? 0x7 : 0xf;
    constexpr std::uint32_t special_value_flags_shift = isFpu ? 4 : 0;
    return (special_value_flags_reg >> special_value_flags_shift) & special_value_flags_mask;
}

inline void llk_math_clear_compute_special_value_flags() { _llk_math_clear_compute_special_value_flags_(); }
