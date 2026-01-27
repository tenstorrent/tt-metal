// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::tt_fabric {
// TODO: move to *_msgs.h definition
enum RouterStateCommon : std::uint32_t {
    INITIALIZING = 0,
    RUNNING = 1,
    PAUSED = 2,
    DRAINING = 3,
    RETRAINING = 4
};

}  // namespace tt::tt_fabric
