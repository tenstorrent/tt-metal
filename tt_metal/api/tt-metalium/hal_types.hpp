// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

using DeviceAddr = std::uint64_t;

enum class HalProcessorClassType : uint8_t { DM = 0, COMPUTE = 1 };

enum class HalProgrammableCoreType { TENSIX = 0, ACTIVE_ETH = 1, IDLE_ETH = 2, COUNT = 3 };

static constexpr uint32_t NumHalProgrammableCoreTypes = static_cast<uint32_t>(HalProgrammableCoreType::COUNT);

// TODO: Move to llrt/hal.hpp after device function cleanup
enum class HalL1MemAddrType : uint8_t {
    BASE,
    BARRIER,
    MAILBOX,
    LAUNCH,
    WATCHER,
    DPRINT_BUFFERS,
    PROFILER,
    KERNEL_CONFIG,  // End is start of unreserved memory
    UNRESERVED,     // Only for ethernet cores
    DEFAULT_UNRESERVED,
    CORE_INFO,
    GO_MSG,
    LAUNCH_MSG_BUFFER_RD_PTR,
    GO_MSG_INDEX,
    LOCAL,
    BANK_TO_NOC_SCRATCH,
    LOGICAL_TO_VIRTUAL_SCRATCH,
    APP_SYNC_INFO,
    APP_ROUTING_INFO,
    RETRAIN_COUNT,
    RETRAIN_FORCE,
    CRC_ERR,    // Link status - CRC error count
    CORR_CW,    // Link status - Corrected Codewords count
    UNCORR_CW,  // Link status - Uncorrected Codewords count
    LINK_UP,    // Link status - Link up status
    FABRIC_TELEMETRY,
    ROUTING_TABLE,
    ETH_FW_MAILBOX,
    TENSIX_FABRIC_CONNECTIONS,
    FABRIC_CONNECTION_LOCK,
    COUNT  // Keep this last so it always indicates number of enum options
};

enum class HalMemType : uint8_t { L1 = 0, DRAM = 1, HOST = 2, COUNT = 3 };

}  // namespace tt::tt_metal
