// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
namespace ckernel::sfpu::bf16 {
template <typename Config, int Iterations = 32>
inline void calculate_config_tile() {
    static_assert(Iterations == 32, "selected Config owns one complete tile");
    Config::tile();
}
}  // namespace ckernel::sfpu::bf16
