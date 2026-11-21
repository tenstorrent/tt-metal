// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "dev_mem_map.h"

namespace eth_l1_mem {

// Backward compatibility
struct address_map {
    static constexpr std::uint32_t MAX_NUM_CONCURRENT_TRANSACTIONS = MEM_MAX_NUM_CONCURRENT_TRANSACTIONS;
    static constexpr std::uint32_t ERISC_APP_SYNC_INFO_BASE = MEM_ERISC_APP_SYNC_INFO_BASE;
    static constexpr std::uint32_t ERISC_APP_ROUTING_INFO_BASE = MEM_ERISC_APP_ROUTING_INFO_BASE;
    static constexpr std::uint32_t AERISC_FABRIC_TELEMETRY_ADDR = MEM_AERISC_FABRIC_TELEMETRY_BASE;
    static constexpr std::uint32_t LAUNCH_ERISC_APP_FLAG = 0;  // don't need this - just to get things to compile

    static constexpr std::uint32_t ERISC_MEM_MAILBOX_BASE = MEM_AERISC_MAILBOX_BASE;
    static constexpr std::uint32_t ERISC_MEM_MAILBOX_END = MEM_AERISC_MAILBOX_END;
    static constexpr std::uint32_t ERISC_L1_UNRESERVED_BASE =
        (MEM_AERISC_MAP_END + MEM_ERISC_KERNEL_CONFIG_SIZE + 63) & ~63;

    static_assert(ERISC_MEM_MAILBOX_BASE == MEM_AERISC_MAILBOX_BASE);
};
}  // namespace eth_l1_mem
