// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

namespace tensor_accessor {
constexpr std::size_t MAX_RANK = 8;                            // Maximum rank supported by the accessor
constexpr std::size_t UNKNOWN = std::numeric_limits<std::size_t>::max();  // Used to indicate unknown values
}  // namespace tensor_accessor
