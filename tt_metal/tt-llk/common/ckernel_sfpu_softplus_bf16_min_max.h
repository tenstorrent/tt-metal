// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SFPI deleted vec_min_max; retain its former in-place implementation exactly.
// Both results come from one SFPSWAP, preserving operand and assignment order.
namespace ckernel::sfpu::ttpoly::softplus
{
using namespace ::sfpi;
using ::sfpi::DataLayout;

__attribute__((always_inline)) inline void ordered_min_max(vFloat& a, vFloat& b)
{
    auto r = min_max(a, b);
    a      = r.first;
    b      = r.second;
}
} // namespace ckernel::sfpu::ttpoly::softplus
