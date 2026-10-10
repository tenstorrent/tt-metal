// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

using DeviceAddr = std::uint64_t;

enum class HalProcessorClassType : uint8_t { DM = 0, COMPUTE = 1 };

enum class HalProgrammableCoreType { TENSIX = 0, ACTIVE_ETH = 1, IDLE_ETH = 2, DRAM = 3, DISPATCH = 4, COUNT = 5 };

enum class HalMemType : uint8_t { L1 = 0, DRAM = 1, HOST = 2, COUNT = 3 };

}  // namespace tt::tt_metal
