// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
enum class FabricConfig : uint32_t {
    DISABLED = 0,
    FABRIC_1D = 1,       // Instatiates fabric with 1D routing and no deadlock avoidance
    FABRIC_1D_RING = 2,  // Instatiates fabric with 1D routing and with deadlock avoidance using datelines
    FABRIC_2D = 3,       // Instatiates fabric with 2D routing in pull mode
    FABRIC_2D_PUSH = 4,  // Instatiates fabric with 2D routing in push mode
    CUSTOM = 5
};

}  // namespace tt::tt_metal
