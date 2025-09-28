// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt {

/**
 * @brief Specifies the target devices on which the graph can be run.
 */
enum class TargetDevice : std::uint8_t {
    Silicon = 0,
    Simulator = 1,
    Mock = 2,
    Invalid = 0xFF,
};

}  // namespace tt
