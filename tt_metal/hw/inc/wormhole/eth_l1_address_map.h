// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "noc/noc_parameters.h"
#include "tt_fabric/hw/inc/routing_table.h"

namespace eth_l1_mem {

struct address_map {
    // UMD doesn't distinguish between active/idle eth cores
    // UMD needs space for l1_barrier
    // active/idle eth cores have very different mem maps
    // Reserve some space at the end of l1 for l1_barrier
    static constexpr std::int32_t ERISC_BARRIER_SIZE = 32;
    static constexpr std::int32_t MAX_SIZE = 256 * 1024 - ERISC_BARRIER_SIZE;
    static constexpr std::int32_t MAX_L1_LOADING_SIZE = 1 * 256 * 1024 - ERISC_BARRIER_SIZE;

    // Sizes
    static constexpr std::int32_t FIRMWARE_SIZE = 32 * 1024;
    static constexpr std::int32_t COMMAND_Q_SIZE = 4 * 1024;
    static constexpr std::int32_t DATA_BUFFER_SIZE_HOST = 4 * 1024;
    static constexpr std::int32_t DATA_BUFFER_SIZE_ETH = 4 * 1024;
    static constexpr std::int32_t DATA_BUFFER_SIZE_NOC = 16 * 1024;
    static constexpr std::int32_t DATA_BUFFER_SIZE = 24 * 1024;
    // Memory for (dram/l1)_bank_to_noc_xy arrays, size needs to be atleast 2 * NUM_NOCS * (NUM_DRAM_BANKS + NUM_L1_BANKS)
    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_XY_SIZE = 1024;
    // Memory for bank_to_dram_offset and bank_to_l1_offset arrays, size needs to be atleast 4 * (NUM_DRAM_BANKS + NUM_L1_BANKS)
    static constexpr std::int32_t ERISC_MEM_BANK_OFFSET_SIZE = 1024;

    // Kernel config buffer is WIP
    // Size is presently based on the old sizes of the RTAs + CB config + Sems
    static constexpr std::int32_t ERISC_L1_KERNEL_CONFIG_SIZE = 96 * 4 + 8 * 16;

    // Base addresses
    static constexpr std::int32_t FIRMWARE_BASE = 0x9040;
    static constexpr std::int32_t L1_EPOCH_Q_BASE = 0x9000;  // Epoch Q start in L1.
    static constexpr std::int32_t COMMAND_Q_BASE = L1_EPOCH_Q_BASE + FIRMWARE_SIZE;
    static constexpr std::int32_t DATA_BUFFER_BASE = COMMAND_Q_BASE + COMMAND_Q_SIZE;
    static constexpr std::int32_t TILE_HEADER_BUFFER_BASE = DATA_BUFFER_BASE + DATA_BUFFER_SIZE;

    // TT Metal Specific
    static constexpr std::int32_t ERISC_FIRMWARE_SIZE = 2 * 1024;
    // Total 160 * 1024 L1 starting from TILE_HEADER_BUFFER_BASE
    //    -   1 * 1024 misc args
    //    -  53 * 1024 eth app reserved buffer space
    //    - 106 * 1024 L1 unreserved buffer space
    static constexpr std::int32_t MAX_NUM_CONCURRENT_TRANSACTIONS = 8;
    static constexpr std::int32_t ERISC_APP_ROUTING_INFO_SIZE = 48;
    static constexpr std::int32_t ERISC_APP_SYNC_INFO_SIZE = 160 + 16 * MAX_NUM_CONCURRENT_TRANSACTIONS;

    static constexpr std::int32_t ERISC_BARRIER_BASE = MAX_SIZE;
    static constexpr std::int32_t ERISC_APP_ROUTING_INFO_BASE = TILE_HEADER_BUFFER_BASE;
    static constexpr std::int32_t ERISC_APP_SYNC_INFO_BASE = ERISC_APP_ROUTING_INFO_BASE + ERISC_APP_ROUTING_INFO_SIZE;

    static constexpr std::int32_t ERISC_MEM_MAILBOX_BASE = ERISC_APP_SYNC_INFO_BASE + ERISC_APP_SYNC_INFO_SIZE;

    static constexpr std::uint32_t ERISC_MEM_MAILBOX_SIZE = 3232;
    static constexpr std::uint32_t ERISC_MEM_MAILBOX_END = ERISC_MEM_MAILBOX_BASE + ERISC_MEM_MAILBOX_SIZE;
    static constexpr std::int32_t ERISC_L1_KERNEL_CONFIG_BASE = ERISC_MEM_MAILBOX_END;
    static constexpr std::int32_t FABRIC_ROUTER_CONFIG_BASE =
        (ERISC_L1_KERNEL_CONFIG_BASE + ERISC_L1_KERNEL_CONFIG_SIZE + 31) & ~31;
    static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE =
        (FABRIC_ROUTER_CONFIG_BASE + sizeof(tt::tt_fabric::fabric_router_l1_config_t) + 31) & ~31;
    static constexpr std::int32_t ERISC_L1_UNRESERVED_SIZE = MAX_L1_LOADING_SIZE - ERISC_L1_UNRESERVED_BASE;

    static_assert((ERISC_L1_UNRESERVED_BASE % 32) == 0);

    // This scratch address is same as ERISC_L1_UNRESERVED_BASE, as the scratch space is used to copy data during
    // runtime build, and is unused once FW copies the data to local memory during FW initialization.
    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_SCRATCH =
        (ERISC_L1_KERNEL_CONFIG_BASE + ERISC_L1_KERNEL_CONFIG_SIZE + 31) & ~31;
    static constexpr std::int32_t ERISC_MEM_BANK_TO_NOC_SIZE = ERISC_MEM_BANK_TO_NOC_XY_SIZE + ERISC_MEM_BANK_OFFSET_SIZE;

    static constexpr std::int32_t LAUNCH_ERISC_APP_FLAG = L1_EPOCH_Q_BASE + 4;

    template <std::size_t A, std::size_t B>
    struct TAssertEquality {
        static_assert(A == B, "Not equal");
        static constexpr bool _cResult = (A == B);
    };

    static constexpr std::int32_t RISC_LOCAL_MEM_BASE =
        0xffb00000;  // Actaul local memory address as seen from risc firmware
                     // As part of the init risc firmware will copy local memory data from
                     // l1 locations listed above into internal local memory that starts
                     // at RISC_LOCAL_MEM_BASE address

    static constexpr std::uint32_t FW_VERSION_ADDR = 0x210;
    static constexpr std::uint32_t RETRAIN_COUNT_ADDR = 0x1EDC;
    static constexpr std::uint32_t RETRAIN_FORCE_ADDR = 0x1EFC;
};
}  // namespace eth_l1_mem
