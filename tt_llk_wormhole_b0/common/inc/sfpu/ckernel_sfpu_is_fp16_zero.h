// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

sfpi_inline sfpi::vInt _sfpu_is_fp16_zero_(const sfpi::vFloat& v, uint exponent_size_8)
{
    if (exponent_size_8)
    {
        // fp16b
        return v == 0.0F;
    }
    else
    {
        // fp16a
        // if math data format is fp16, SFPU will convert 5 bit exp to 8 bit exp
        // in grayskull, this unconditionally adds bias value to exp (even for zero)
        sfpi::vInt tmp = 0x3800; // loads {0, 8'd112, 10'b0}
        tmp += sfpi::reinterpret<sfpi::vInt>(v);

        return tmp == 0;
    }
}

} // namespace sfpu
} // namespace ckernel
