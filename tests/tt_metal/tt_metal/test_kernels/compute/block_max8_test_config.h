// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace block_max8_test {
constexpr std::uint32_t slots = 8;
constexpr std::uint32_t block_size = 8;
constexpr std::uint32_t result_rows = 8;
constexpr std::uint32_t valid_counts[] = {0, 1, 7, 8, 9, 15, 16, 17, 511, 512, 513, 1023, 1024};
constexpr std::uint32_t dst_indices[] = {0, 3, 7};
constexpr std::uint32_t batches = sizeof(valid_counts) / sizeof(valid_counts[0]);
constexpr std::uint32_t dst_index_count = sizeof(dst_indices) / sizeof(dst_indices[0]);
}  // namespace block_max8_test
