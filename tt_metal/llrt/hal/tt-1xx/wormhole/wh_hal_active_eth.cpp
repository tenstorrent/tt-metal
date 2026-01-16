// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define HAL_BUILD tt::tt_metal::wormhole::active_eth
#define COMPILE_FOR_ERISC

#include "hostdev/dev_msgs.h"
#include "hostdev/fabric_telemetry_msgs.h"
using namespace tt::tt_metal::wormhole::active_eth;

#include "eth_l1_address_map.h"
#include "eth_fw_api.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include "wormhole/wh_hal.hpp"
#include "wormhole/wh_hal_eth_asserts.hpp"

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) \
    ((uint64_t)&(((mailboxes_t*)eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE)->x))

namespace tt::tt_metal::wormhole {

// This file is intended to be wrapped inside arch-specific namespace.
namespace active_eth_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

namespace active_eth_fabric_telemetry {
#include "hal/generated/fabric_telemetry_impl.hpp"
}

HalCoreInfoType create_active_eth_mem_map(bool is_base_routing_fw_enabled) {
    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = 0x0;  // Anything better to use?
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = eth_l1_mem::address_map::ERISC_BARRIER_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] =
        eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_ETH_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_ETH_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] =
        eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        is_base_routing_fw_enabled ? eth_l1_mem::address_map::ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE
                                   : eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_ETH_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_ETH_MAILBOX_ADDRESS_HOST(go_messages);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(go_message_index);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] =
        eth_l1_mem::address_map::ERISC_MEM_BANK_TO_NOC_SCRATCH;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::APP_SYNC_INFO)] =
        eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::APP_ROUTING_INFO)] =
        eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_COUNT)] =
        eth_l1_mem::address_map::RETRAIN_COUNT_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_FORCE)] =
        eth_l1_mem::address_map::RETRAIN_FORCE_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CRC_ERR)] = eth_l1_mem::address_map::CRC_ERR_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORR_CW)] = eth_l1_mem::address_map::CORR_CW_HI_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNCORR_CW)] = eth_l1_mem::address_map::UNCORR_CW_HI_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LINK_UP)] =
        MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, link_status);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_TELEMETRY)] = MEM_AERISC_FABRIC_TELEMETRY_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ROUTING_TABLE)] = MEM_AERISC_ROUTING_TABLE_BASE;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] =
        eth_l1_mem::address_map::MAX_SIZE;  // Anything better to use?
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = eth_l1_mem::address_map::ERISC_BARRIER_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] =
        eth_l1_mem::address_map::ERISC_MEM_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] =
        eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        is_base_routing_fw_enabled ? eth_l1_mem::address_map::ROUTING_ENABLED_ERISC_L1_UNRESERVED_SIZE
                                   : eth_l1_mem::address_map::ERISC_L1_UNRESERVED_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t) * go_message_num_entries;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] =
        eth_l1_mem::address_map::ERISC_MEM_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::APP_SYNC_INFO)] =
        eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::APP_ROUTING_INFO)] =
        eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_COUNT)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_FORCE)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_TELEMETRY)] = MEM_AERISC_FABRIC_TELEMETRY_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ROUTING_TABLE)] = MEM_AERISC_ROUTING_TABLE_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LINK_UP)] = sizeof(uint32_t);
    // Base FW api not supported on WH
    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes = {
        // DM
        {
            // ERISC
            {.fw_base_addr = eth_l1_mem::address_map::FIRMWARE_BASE,
             .local_init_addr = eth_l1_mem::address_map::FIRMWARE_BASE,
             .fw_launch_addr = eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG,
             .fw_launch_addr_value = 0x1,
             .memory_load = ll_api::memory::Loading::DISCRETE},
        },
    };
    std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names = {
        // DM
        {
            {"ER", "ERISC"},
        },
    };
    std::vector<uint8_t> processor_classes_num_fw_binaries = {/*DM*/ 1};

    static_assert(sizeof(mailboxes_t) <= eth_l1_mem::address_map::ERISC_MEM_MAILBOX_SIZE);
    return {
        HalProgrammableCoreType::ACTIVE_ETH,
        CoreType::ETH,
        std::move(processor_classes),
        std::move(processor_classes_num_fw_binaries),
        std::move(mem_map_bases),
        std::move(mem_map_sizes),
        std::move(fw_mailbox_addr),
        std::move(processor_classes_names),
        false /*supports_cbs*/,
        false /*supports_receiving_multicast_cmds*/,
        active_eth_dev_msgs::create_factory(),
        active_eth_fabric_telemetry::create_factory()};
}

}  // namespace tt::tt_metal::wormhole
