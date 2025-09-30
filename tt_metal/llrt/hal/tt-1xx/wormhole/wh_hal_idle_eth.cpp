// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define HAL_BUILD tt::tt_metal::wormhole::idle_eth
#define COMPILE_FOR_IDLE_ERISC

#include "dev_msgs.h"
using namespace tt::tt_metal::wormhole::idle_eth;

#include <cstdint>

#include "dev_mem_map.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "noc/noc_parameters.h"
#include <umd/device/types/core_coordinates.hpp>
#include "wormhole/wh_hal.hpp"
#include "wormhole/wh_hal_eth_asserts.hpp"

#define GET_IERISC_MAILBOX_ADDRESS_HOST(x) ((std::uint64_t)&(((mailboxes_t*)MEM_IERISC_MAILBOX_BASE)->x))

namespace tt::tt_metal::wormhole {

// This file is intended to be wrapped inside arch/core-specific namespace.
namespace idle_eth_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

HalCoreInfoType create_idle_eth_mem_map() {
    constexpr std::uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    static_assert(MEM_IERISC_MAP_END % L1_ALIGNMENT == 0);

    std::vector<DeviceAddr> mem_map_bases;
    constexpr std::uint32_t L1_KERNEL_CONFIG_SIZE = 25 * 1024;

    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ETH_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_IERISC_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_IERISC_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] =
        GET_IERISC_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_IERISC_MAP_END;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        ((MEM_IERISC_MAP_END + L1_KERNEL_CONFIG_SIZE - 1) | (max_alignment - 1)) + 1;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_IERISC_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_IERISC_MAILBOX_ADDRESS_HOST(go_messages);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] =
        GET_IERISC_MAILBOX_ADDRESS_HOST(go_message_index);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_IERISC_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_IERISC_BANK_TO_NOC_SCRATCH;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTING_TABLE)] = MEM_IERISC_ROUTING_TABLE_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_EXIT_NODE_TABLE)] = MEM_IERISC_EXIT_NODE_TABLE_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTING_PATH_1D)] =
        MEM_IERISC_FABRIC_ROUTING_PATH_BASE_1D;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTING_PATH_2D)] =
        MEM_IERISC_FABRIC_ROUTING_PATH_BASE_2D;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ETH_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_IERISC_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] =
        L1_KERNEL_CONFIG_SIZE;  // TODO: this is wrong, need idle eth specific value
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        MEM_ETH_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)];
    ;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t) * go_message_num_entries;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_IERISC_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTING_TABLE)] = MEM_ROUTING_TABLE_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_EXIT_NODE_TABLE)] = MEM_EXIT_NODE_TABLE_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTING_PATH_1D)] =
        MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_1D;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTING_PATH_2D)] =
        MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_2D;

    // Base FW api not supported on WH
    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes = {
        // DM
        {
            // ERISC
            {.fw_base_addr = MEM_IERISC_FIRMWARE_BASE,
             .local_init_addr = MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH,
             .fw_launch_addr = 0x0,
             .fw_launch_addr_value = generate_risc_startup_addr(MEM_IERISC_FIRMWARE_BASE),
             .memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP},
        },
    };
    static_assert(sizeof(mailboxes_t) <= MEM_IERISC_MAILBOX_SIZE);
    return {
        HalProgrammableCoreType::IDLE_ETH,
        CoreType::ETH,
        std::move(processor_classes),
        std::move(mem_map_bases),
        std::move(mem_map_sizes),
        std::move(fw_mailbox_addr),
        false /*supports_cbs*/,
        false /*supports_receiving_multicast_cmds*/,
        idle_eth_dev_msgs::create_factory()};
}

}  // namespace tt::tt_metal::wormhole
