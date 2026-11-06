// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::sfpu::ttpoly::lgamma {
using namespace ::sfpi;
using ::sfpi::DataLayout;
__attribute__((always_inline)) inline void ordered_min_max(vFloat& a, vFloat& b) {
    auto r = min_max(a, b);
    a = r.first;
    b = r.second;
}
}  // namespace ckernel::sfpu::ttpoly::lgamma
