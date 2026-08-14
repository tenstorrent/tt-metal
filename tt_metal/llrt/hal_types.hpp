// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

static constexpr uint32_t NumHalProgrammableCoreTypes = static_cast<uint32_t>(HalProgrammableCoreType::COUNT);

enum class HalL1MemAddrType : uint8_t {
    BASE,
    BARRIER,
    MAILBOX,
    LAUNCH,
    WATCHER,
    DPRINT_BUFFERS,
    PROFILER,
    KERNEL_CONFIG,  // End is start of unreserved memory
    UNRESERVED,     // For ethernet and DRAM cores
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
    CRC_ERR,          // Link status - CRC error count
    CORR_CW,          // Link status - Corrected Codewords count
    UNCORR_CW,        // Link status - Uncorrected Codewords count
    TXQ0_RESEND_CNT,  // Link status - TX queue 0 packet resend count (Blackhole only)
    TXQ1_RESEND_CNT,  // Link status - TX queue 1 packet resend count (Blackhole only)
    TXQ2_RESEND_CNT,  // Link status - TX queue 2 packet resend count (Blackhole only)
    RXQ0_PKT_DROP,    // Link status - RX queue 0 packet drop count (Blackhole only)
    RXQ1_PKT_DROP,    // Link status - RX queue 1 packet drop count (Blackhole only)
    RXQ2_PKT_DROP,    // Link status - RX queue 2 packet drop count (Blackhole only)
    LINK_UP,          // Link status - Link up status
    FABRIC_TELEMETRY,
    ROUTING_TABLE,
    ROUTER_STATE,
    ROUTER_COMMAND,
    ETH_FW_MAILBOX,
    TENSIX_FABRIC_CONNECTIONS,
    FABRIC_CONNECTION_LOCK,
    COUNT  // Keep this last so it always indicates number of enum options
};

}  // namespace tt::tt_metal
