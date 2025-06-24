// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "noc/noc_parameters.h"

namespace eth_iram_mem {
struct address_map {
    static constexpr std::int32_t ERISC_IRAM_BASE = 0xFFC00000;
    static constexpr std::int32_t ERISC_IRAM_SIZE = 16 * 1024;
    static constexpr std::int32_t ERISC_KERNEL_BASE = ERISC_IRAM_BASE;
};
};  // namespace eth_iram_mem

namespace eth_l1_mem {

struct address_map {
    // UMD doesn't distinguish between active/idle eth cores
    // UMD needs space for l1_barrier
    // active/idle eth cores have very different mem maps
    // Reserve some space at the end of l1 for l1_barrier
    static constexpr std::int32_t ERISC_BARRIER_SIZE = 32;
    static constexpr std::int32_t MAX_SIZE = 256 * 1024 - ERISC_BARRIER_SIZE;

    // Sizes
    static constexpr std::int32_t APP_FIRMWARE_SIZE = 32 * 1024;
    static constexpr std::int32_t ROUTING_FW_RESERVED_SIZE = 28 * 1024;

    //  Memory for (dram/l1)_bank_to_noc_xy arrays, size needs to be atleast 2 * NUM_NOCS * (NUM_DRAM_BANKS +
    //  NUM_L1_BANKS)
    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_XY_SIZE = 1024;
    // Memory for bank_to_dram_offset and bank_to_l1_offset arrays, size needs to be atleast 4 * (NUM_DRAM_BANKS + NUM_L1_BANKS)
    static constexpr std::int32_t ERISC_MEM_BANK_OFFSET_SIZE = 1024;

    // Kernel config buffer is WIP
    // Size is presently based on the old sizes of the RTAs + CB config + Sems
    static constexpr std::int32_t ERISC_L1_KERNEL_CONFIG_SIZE = 96 * 4 + 16 * 16;

    // Base addresses
    static constexpr std::int32_t FIRMWARE_BASE = 0x9040;
    static constexpr std::int32_t L1_EPOCH_Q_BASE = 0x9000;  // Epoch Q start in L1.
    static constexpr std::int32_t KERNEL_BASE = 0xA840;
    static constexpr std::int32_t ROUTING_FW_RESERVED_BASE = L1_EPOCH_Q_BASE + APP_FIRMWARE_SIZE;

    static constexpr std::int32_t MAX_L1_LOADING_ADDR = 0x3E000;

    // TT Metal Specific
    //     Optional FW reserved space for routing FW
    //     ERISC unreserved space
    //     ERISC config space

    // ERISC UNRESERVED space starting at ROUTING_FW_RESERVED_BASE on systems without routing FW
    // MAX is always MAX_L1_LOADING_ADDR
    // NOTE: ERISC_L1_UNRESERVED_BASE resolves to ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE when compiled with
    // ROUTING_FW_ENABLED
    // On host ERISC_L1_UNRESERVED_BASE will always resolve to ROUTING_FW_RESERVED_BASE
    static constexpr std::int32_t ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE =
        ROUTING_FW_RESERVED_BASE + ROUTING_FW_RESERVED_SIZE;
    static constexpr std::int32_t ROUTING_ENABLED_ERISC_L1_UNRESERVED_SIZE =
        MAX_L1_LOADING_ADDR - ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE;
#ifdef ROUTING_FW_ENABLED
    static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE = ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE;
#else
    static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE = ROUTING_FW_RESERVED_BASE;
#endif
    static constexpr std::int32_t ERISC_L1_UNRESERVED_SIZE = MAX_L1_LOADING_ADDR - ERISC_L1_UNRESERVED_BASE;

    static_assert((ERISC_L1_UNRESERVED_BASE % 32) == 0);

    // CONFIG SPACE starting from MAX_L1_LOADING_ADDR to MAX_SIZE
    static constexpr std::int32_t MAX_NUM_CONCURRENT_TRANSACTIONS = 8;
    static constexpr std::int32_t ERISC_APP_ROUTING_INFO_SIZE = 48;
    static constexpr std::int32_t ERISC_APP_SYNC_INFO_SIZE = 160 + 16 * MAX_NUM_CONCURRENT_TRANSACTIONS;

    static constexpr std::int32_t ERISC_APP_ROUTING_INFO_BASE = MAX_L1_LOADING_ADDR;
    static constexpr std::int32_t ERISC_APP_SYNC_INFO_BASE = ERISC_APP_ROUTING_INFO_BASE + ERISC_APP_ROUTING_INFO_SIZE;

    static constexpr std::int32_t ERISC_MEM_MAILBOX_BASE = ERISC_APP_SYNC_INFO_BASE + ERISC_APP_SYNC_INFO_SIZE;

    static constexpr std::uint32_t ERISC_MEM_MAILBOX_SIZE = 5072;
    static constexpr std::uint32_t ERISC_MEM_MAILBOX_END = ERISC_MEM_MAILBOX_BASE + ERISC_MEM_MAILBOX_SIZE;
    static constexpr std::int32_t ERISC_L1_KERNEL_CONFIG_BASE = ERISC_MEM_MAILBOX_END;
    static constexpr std::int32_t FABRIC_ROUTER_CONFIG_BASE =
        (ERISC_L1_KERNEL_CONFIG_BASE + ERISC_L1_KERNEL_CONFIG_SIZE + 31) & ~31;
    static constexpr std::int32_t FABRIC_ROUTER_CONFIG_SIZE = 2064;

    static constexpr std::int32_t ERISC_BARRIER_BASE =
        (FABRIC_ROUTER_CONFIG_BASE + FABRIC_ROUTER_CONFIG_SIZE + 31) & ~31;
    static_assert(ERISC_BARRIER_BASE < MAX_SIZE, "Erisc config region is greater than MAX_SIZE");

    // This scratch address is same as ERISC_L1_UNRESERVED_BASE, as the scratch space is used to copy data during
    // runtime build, and is unused once FW copies the data to local memory during FW initialization.
    // We use ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE as this is a safe address both with/without routing FW
    // enabled.
    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_SCRATCH = ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE;
    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_SIZE = ERISC_MEM_BANK_TO_NOC_XY_SIZE + ERISC_MEM_BANK_OFFSET_SIZE;
    static_assert(ERISC_MEM_BANK_TO_NOC_SCRATCH + ERISC_MEM_BANK_TO_NOC_SIZE <= MAX_L1_LOADING_ADDR);

    static constexpr std::int32_t LAUNCH_ERISC_APP_FLAG = L1_EPOCH_Q_BASE + 4;

    template <std::size_t A, std::size_t B>
    struct TAssertEquality {
        static_assert(A == B, "Not equal");
        static constexpr bool _cResult = (A == B);
    };

    static constexpr std::int32_t RISC_LOCAL_MEM_BASE =
        0xffb00000;  // Actual local memory address as seen from risc firmware
                     // As part of the init risc firmware will copy local memory data from
                     // l1 locations listed above into internal local memory that starts
                     // at RISC_LOCAL_MEM_BASE address

    static constexpr std::uint32_t RETRAIN_COUNT_ADDR = 0x1EDC;
    static constexpr std::uint32_t RETRAIN_FORCE_ADDR = 0x1EFC;

    static constexpr uint32_t ETH_LINK_REMOTE_INFO_ADDR = 0x1EC0;
    static constexpr std::uint32_t INTERMESH_ETH_LINK_CONFIG_ADDR = 0x104C;
    static constexpr std::uint32_t INTERMESH_ETH_LINK_STATUS_ADDR = 0x1104;
};
}  // namespace eth_l1_mem
