// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace sfpi {
template <typename Reciprocal>
inline vFloat correction_finite_reciprocal(vFloat x, [[maybe_unused]] Reciprocal reciprocal) {
    return reciprocal(x);
}
}  // namespace sfpi
