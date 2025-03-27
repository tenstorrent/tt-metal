// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
enum class FabricConfig : uint32_t { DISABLED = 0, FABRIC_1D = 1, FABRIC_2D = 2, FABRIC_2D_PUSH = 3, CUSTOM = 4 };

}  // namespace tt::tt_metal
