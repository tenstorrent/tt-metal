// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu::ttpoly::hardmish {
using namespace ::sfpi;
using ::sfpi::DataLayout;
sfpi_inline vFloat simple_gated_product(vFloat x, float q0, float q1) {
    vFloat gate = q1 * x + q0;
    vFloat zero = 0.0f;
    vFloat one = 1.0f;
    ordered_min_max(zero, gate);  // gate = max(0, gate)
    ordered_min_max(gate, one);   // gate = min(gate, 1)
    vFloat y = x * gate;
    return y;
}
}  // namespace ckernel::sfpu::ttpoly::hardmish
