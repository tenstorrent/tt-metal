// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace eth_l1_mem {

struct address_map {
    // From top of L1:
    // - Syseng reserves [Max L1 Eth - SYSENG_RESERVED_SIZE, Max L1 Eth)
    // - L1 barrier [Max L1 Eth - SYSENG_RESERVED_SIZE - ERISC_BARRIER_SIZE, Max L1 Eth - SYSENG_RESERVED_SIZE)
    // - Tunneling [Max L1 Eth - SYSENG_RESERVED_SIZE - ERISC_BARRIER_SIZE - ERISC_APP_SYNC_INFO_SIZE, Max L1 Eth -
    // SYSENG_RESERVED_SIZE - ERISC_BARRIER_SIZE)
    static constexpr std::int32_t SYSENG_RESERVED_SIZE = 64 * 1024;
    static constexpr std::int32_t ERISC_BARRIER_SIZE = 64;
    static constexpr std::int32_t ERISC_APP_ROUTING_INFO_SIZE = 48;
    static constexpr std::int32_t MAX_NUM_CONCURRENT_TRANSACTIONS = 8;
    static constexpr std::int32_t ERISC_APP_SYNC_INFO_SIZE = 160 + 16 * MAX_NUM_CONCURRENT_TRANSACTIONS;
    static constexpr std::int32_t FABRIC_ROUTER_CONFIG_SIZE = 2064;  // aligning this to L1_ALIGNMENT

    static constexpr std::int32_t MAX_SIZE = 512 * 1024 - SYSENG_RESERVED_SIZE - ERISC_BARRIER_SIZE -
                                             ERISC_APP_ROUTING_INFO_SIZE - ERISC_APP_SYNC_INFO_SIZE -
                                             FABRIC_ROUTER_CONFIG_SIZE;
    static constexpr std::int32_t MAX_L1_LOADING_SIZE = MAX_SIZE;

    static constexpr std::int32_t FABRIC_ROUTER_CONFIG_BASE = MAX_SIZE;
    static constexpr std::int32_t ERISC_APP_SYNC_INFO_BASE = FABRIC_ROUTER_CONFIG_BASE + FABRIC_ROUTER_CONFIG_SIZE;
    static constexpr std::int32_t ERISC_APP_ROUTING_INFO_BASE = ERISC_APP_SYNC_INFO_BASE + ERISC_APP_SYNC_INFO_SIZE;
    static constexpr std::uint32_t ERISC_BARRIER_BASE = ERISC_APP_ROUTING_INFO_BASE + ERISC_APP_ROUTING_INFO_SIZE;

    static constexpr std::int32_t ERISC_FIRMWARE_SIZE = 24 * 1024;
    static constexpr std::uint32_t MEM_ERISC_LOCAL_SIZE = (8 * 1024);
    static constexpr std::int32_t RISC_LOCAL_MEM_BASE =
        0xFFB00000;  // Actual local memory address as seen from risc firmware
                     // As part of the init risc firmware will copy local memory data from
                     // l1 locations listed above into internal local memory that starts
                     // at RISC_LOCAL_MEM_BASE address

    static constexpr uint32_t MEM_ERISC_RESERVED1 = 0;
    static constexpr uint32_t MEM_ERISC_RESERVED1_SIZE = 1024;

    static constexpr std::int32_t ERISC_MEM_MAILBOX_BASE = MEM_ERISC_RESERVED1 + MEM_ERISC_RESERVED1_SIZE;
    static constexpr std::uint32_t ERISC_MEM_MAILBOX_SIZE = 3728;
    static constexpr std::uint32_t ERISC_MEM_MAILBOX_END = ERISC_MEM_MAILBOX_BASE + ERISC_MEM_MAILBOX_SIZE;

    static constexpr std::int32_t FIRMWARE_BASE = ERISC_MEM_MAILBOX_END;
    static constexpr std::int32_t MEM_ERISC_MAP_END = FIRMWARE_BASE + ERISC_FIRMWARE_SIZE;

    static constexpr std::uint32_t MEM_ERISC_KERNEL_SIZE = (24 * 1024);
    static constexpr std::int32_t MEM_ERISC_INIT_LOCAL_L1_BASE_SCRATCH = MEM_ERISC_MAP_END;
    static constexpr std::int32_t MEM_ERISC_STACK_SIZE = 1024;
    static constexpr std::int32_t MEM_SLAVE_ERISC_STACK_SIZE = 1024;
    static constexpr std::int32_t MEM_ERISC_STACK_BASE =
        RISC_LOCAL_MEM_BASE + MEM_ERISC_LOCAL_SIZE - MEM_ERISC_STACK_SIZE;

    static constexpr std::int32_t LAUNCH_ERISC_APP_FLAG = 0;  // don't need this - just to get things to compile
    static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE = (MEM_ERISC_MAP_END + (69 * 1024) + 63) & ~63;
    static constexpr std::int32_t ERISC_L1_UNRESERVED_SIZE = MAX_SIZE - ERISC_L1_UNRESERVED_BASE;

    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_SCRATCH = ERISC_L1_UNRESERVED_BASE;
    // Memory for (dram/l1)_bank_to_noc_xy arrays, size needs to be atleast 2 * NUM_NOCS * (NUM_DRAM_BANKS +
    // NUM_L1_BANKS)
    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_XY_SIZE = 1024;
    // Memory for bank_to_dram_offset and bank_to_l1_offset arrays, size needs to be atleast 4 * (NUM_DRAM_BANKS +
    // NUM_L1_BANKS)
    static constexpr std::int32_t ERISC_MEM_BANK_OFFSET_SIZE = 1024;
    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_SIZE = ERISC_MEM_BANK_TO_NOC_XY_SIZE + ERISC_MEM_BANK_OFFSET_SIZE;

    static_assert((ERISC_L1_UNRESERVED_BASE % 64) == 0);

    template <std::size_t A, std::size_t B>
    struct TAssertEquality {
        static_assert(A == B, "Not equal");
        static constexpr bool _cResult = (A == B);
    };

    static constexpr std::uint32_t RETRAIN_COUNT_ADDR = 0x1EDC;  // UPDATE ADDR FOR BH!
    static constexpr std::uint32_t RETRAIN_FORCE_ADDR = 0x1EFC;
};
}  // namespace eth_l1_mem
