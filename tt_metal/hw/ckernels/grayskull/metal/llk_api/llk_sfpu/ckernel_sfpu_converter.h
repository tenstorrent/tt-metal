// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


namespace ckernel
{
namespace sfpu
{

union Converter {
    float f;
    uint32_t u;
    static float to_float(uint32_t _v) {
        Converter c{};
        c.u = _v;
        return c.f;
    }
    static uint32_t to_uint(float _f) {
      Converter c{};
      c.f = _f;
      return c.u;
    }
};

} // namespace sfpu
} // namespace ckernel
