// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub: mul_int32 SFPU kernel removed during nuke
// _mul_int_ and _init_mul_int_ remain in tt_llk (ckernel_sfpu_mul_int.h)
namespace ckernel {
namespace sfpu {

template <bool APPROXIMATE>
inline void mul_int32_init() {}

template <bool APPROXIMATE>
inline void mul_int32() {}

}  // namespace sfpu
}  // namespace ckernel
