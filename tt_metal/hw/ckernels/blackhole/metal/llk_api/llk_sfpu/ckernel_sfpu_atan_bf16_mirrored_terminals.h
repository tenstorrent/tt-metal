// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu::ttpoly::atan {
using namespace ::sfpi;
using ::sfpi::DataLayout;
__attribute__((always_inline)) inline vFloat target_raw_terminal_value(
    [[maybe_unused]] vFloat computed, [[maybe_unused]] float constant = 0.0f) {
    return std::numeric_limits<float>::infinity();
}
}  // namespace ckernel::sfpu::ttpoly::atan
