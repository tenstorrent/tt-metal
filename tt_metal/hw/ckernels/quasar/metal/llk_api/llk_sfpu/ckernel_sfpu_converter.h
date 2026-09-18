// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu
{

class Converter
{
public:
    static float as_float(std::uint32_t value)
    {
        return __builtin_bit_cast(float, value);
    }
};

} // namespace ckernel::sfpu
