// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::sfpu
{

template <typename T>
constexpr T POLYVAL5(T coef4, T coef3, T coef2, T coef1, T coef0, T val)
{
    return (((((coef4 * val) + coef3) * val + coef2) * val + coef1) * val + coef0);
}
} // namespace ckernel::sfpu
