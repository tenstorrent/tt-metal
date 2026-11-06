// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
namespace ckernel::sfpu::ttpoly::logsigmoid {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <typename Reciprocal, bool Held = false>
__attribute__((always_inline)) inline vFloat correction_finite_reciprocal(
    vFloat x, [[maybe_unused]] Reciprocal reciprocal) {
    return reciprocal(x);
}
}  // namespace ckernel::sfpu::ttpoly::logsigmoid
