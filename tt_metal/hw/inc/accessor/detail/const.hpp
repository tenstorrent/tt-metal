// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace nd_sharding::detail {
constexpr size_t MAX_RANK = 8;                       // Maximum rank supported by the accessor
constexpr size_t UNKNOWN = static_cast<size_t>(-1);  // Used to indicate unknown values
}  // namespace nd_sharding::detail
