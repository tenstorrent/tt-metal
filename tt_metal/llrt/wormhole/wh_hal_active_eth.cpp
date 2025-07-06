// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define COMPILE_FOR_ERISC

#include "dev_msgs.h"
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core_config.h"
#include "eth_l1_address_map.h"
#include "llrt_common/mailbox.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <umd/device/tt_core_coordinates.h>
#include "wormhole/wh_hal.hpp"
#include "wormhole/wh_hal_eth_asserts.hpp"

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) \
    ((uint64_t)&(((mailboxes_t*)eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE)->x))

namespace tt::tt_metal::wormhole {

HalCoreInfoType create_active_eth_mem_map(bool is_base_routing_fw_enabled) {
    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = 0x0;  // Anything better to use?
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = eth_l1_mem::address_map::ERISC_BARRIER_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] =
        eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_ETH_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = GET_ETH_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_ETH_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] =
        eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        is_base_routing_fw_enabled ? eth_l1_mem::address_map::ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE
                                   : eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_ETH_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_ETH_MAILBOX_ADDRESS_HOST(go_message);
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
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTER_CONFIG)] =
        eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] =
        eth_l1_mem::address_map::MAX_SIZE;  // Anything better to use?
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = eth_l1_mem::address_map::ERISC_BARRIER_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] =
        eth_l1_mem::address_map::ERISC_MEM_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] =
        eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        is_base_routing_fw_enabled ? eth_l1_mem::address_map::ROUTING_ENABLED_ERISC_L1_UNRESERVED_SIZE
                                   : eth_l1_mem::address_map::ERISC_L1_UNRESERVED_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = eth_l1_mem::address_map::ERISC_MEM_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::APP_SYNC_INFO)] =
        eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::APP_ROUTING_INFO)] =
        eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_COUNT)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_FORCE)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTER_CONFIG)] =
        eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_SIZE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_LINK_REMOTE_INFO)] =
        eth_l1_mem::address_map::ETH_LINK_REMOTE_INFO_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::INTERMESH_ETH_LINK_CONFIG)] =
        eth_l1_mem::address_map::INTERMESH_ETH_LINK_CONFIG_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::INTERMESH_ETH_LINK_STATUS)] =
        eth_l1_mem::address_map::INTERMESH_ETH_LINK_STATUS_ADDR;
    // Base FW api not supported on WH
    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumEthDispatchClasses);
    std::vector<HalJitBuildConfig> processor_types(1);
    for (uint8_t processor_class_idx = 0; processor_class_idx < NumEthDispatchClasses; processor_class_idx++) {
        processor_types[0] = HalJitBuildConfig{
            .fw_base_addr = eth_l1_mem::address_map::FIRMWARE_BASE,
            .local_init_addr = eth_l1_mem::address_map::FIRMWARE_BASE,
            .fw_launch_addr = eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG,
            .fw_launch_addr_value = 0x1,
            .memory_load = ll_api::memory::Loading::DISCRETE,
        };
        processor_classes[processor_class_idx] = processor_types;
    }
    static_assert(
        llrt_common::k_SingleProcessorMailboxSize<EthProcessorTypes> <=
        eth_l1_mem::address_map::ERISC_MEM_MAILBOX_SIZE);
    return {
        HalProgrammableCoreType::ACTIVE_ETH,
        CoreType::ETH,
        processor_classes,
        mem_map_bases,
        mem_map_sizes,
        fw_mailbox_addr,
        false /*supports_cbs*/,
        false /*supports_receiving_multicast_cmds*/};
}

}  // namespace tt::tt_metal::wormhole
