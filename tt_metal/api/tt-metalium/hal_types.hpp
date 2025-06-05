// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

using DeviceAddr = std::uint64_t;

enum class HalProgrammableCoreType { TENSIX = 0, ACTIVE_ETH = 1, IDLE_ETH = 2, COUNT = 3 };

static constexpr uint32_t NumHalProgrammableCoreTypes = static_cast<uint32_t>(HalProgrammableCoreType::COUNT);

enum class HalProcessorClassType : uint8_t {
    DM = 0,
    // Setting this to 2 because we currently treat brisc and ncrisc as two unique processor classes on Tensix
    // TODO: Uplift view of Tensix processor classes to be 1 DM class with 2 processor types
    COMPUTE = 2
};

enum class HalL1MemAddrType : uint8_t {
    BASE,
    BARRIER,
    MAILBOX,
    LAUNCH,
    WATCHER,
    DPRINT,
    PROFILER,
    KERNEL_CONFIG,  // End is start of unreserved memory
    UNRESERVED,     // Only for ethernet cores
    DEFAULT_UNRESERVED,
    CORE_INFO,
    GO_MSG,
    LAUNCH_MSG_BUFFER_RD_PTR,
    LOCAL,
    BANK_TO_NOC_SCRATCH,
    APP_SYNC_INFO,
    APP_ROUTING_INFO,
    RETRAIN_COUNT,
    RETRAIN_FORCE,
    FABRIC_ROUTER_CONFIG,
    ETH_FW_MAILBOX,
    COUNT  // Keep this last so it always indicates number of enum options
};

enum class HalDramMemAddrType : uint8_t { BARRIER = 0, PROFILER = 1, UNRESERVED = 2, COUNT = 3 };

enum class HalMemType : uint8_t { L1 = 0, DRAM = 1, HOST = 2, COUNT = 3 };

enum class HalTensixHarvestAxis : uint8_t { ROW = 0x1, COL = 0x2 };

}  // namespace tt::tt_metal
