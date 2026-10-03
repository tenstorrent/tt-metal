// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct AbsBf16Config {
    static constexpr uint32_t kLoadFormat = 0u;
    static constexpr uint32_t kAbsMode = 1u;
    static constexpr uint32_t kStoreFormat = 0u;
    static constexpr uint32_t kBodySlots = 6u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_abs_value.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_abs_bf16() {
    ckernel::sfpu::bf16::calculate_abs_value<ckernel::sfpu::AbsBf16Config, ITERATIONS>();
}
inline void init_abs_bf16() { ckernel::sfpu::bf16::init_abs_value<ckernel::sfpu::AbsBf16Config>(); }

}  // namespace ckernel::sfpu
