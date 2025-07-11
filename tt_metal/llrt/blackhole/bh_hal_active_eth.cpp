// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt_common/mailbox.hpp"
#define COMPILE_FOR_ERISC

#include "tt_align.hpp"
#include "dev_msgs.h"
#include <cstddef>
#include <cstdint>
#include <vector>

#include "blackhole/bh_hal.hpp"
#include "blackhole/bh_hal_eth_asserts.hpp"
#include "core_config.h"
#include "dev_mem_map.h"
#include "eth_l1_address_map.h"
#include "eth_fw_api.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <umd/device/tt_core_coordinates.h>
#include "noc/noc_parameters.h"

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) ((std::uint64_t)&(((mailboxes_t*)MEM_AERISC_MAILBOX_BASE)->x))

namespace tt::tt_metal::blackhole {

HalCoreInfoType create_active_eth_mem_map() {
    std::uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    static_assert(MEM_IERISC_MAP_END % L1_ALIGNMENT == 0);

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ERISC_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_AERISC_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_ETH_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = GET_ETH_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_ETH_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_AERISC_MAP_END;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        tt::align(MEM_AERISC_MAP_END + MEM_ERISC_KERNEL_CONFIG_SIZE, max_alignment);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_ETH_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_ETH_MAILBOX_ADDRESS_HOST(go_messages);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(go_message_index);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_AERISC_BANK_TO_NOC_SCRATCH;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::APP_SYNC_INFO)] = MEM_ERISC_APP_SYNC_INFO_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::APP_ROUTING_INFO)] = MEM_ERISC_APP_ROUTING_INFO_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_COUNT)] = MEM_RETRAIN_COUNT_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_FORCE)] = MEM_RETRAIN_FORCE_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_LINK_REMOTE_INFO)] = MEM_ETH_LINK_REMOTE_INFO_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::INTERMESH_ETH_LINK_CONFIG)] =
        MEM_INTERMESH_ETH_LINK_CONFIG_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::INTERMESH_ETH_LINK_STATUS)] =
        MEM_INTERMESH_ETH_LINK_STATUS_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTER_CONFIG)] =
        MEM_ERISC_FABRIC_ROUTER_CONFIG_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX)] = MEM_SYSENG_ETH_MAILBOX_ADDR;

    std::vector<std::uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ETH_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_ERISC_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    // TODO: this is wrong, need eth specific value. For now use same value as idle
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_ERISC_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        MEM_ERISC_MAX_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t) * go_message_num_entries;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_AERISC_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::APP_SYNC_INFO)] = MEM_ERISC_SYNC_INFO_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::APP_ROUTING_INFO)] = MEM_ERISC_APP_ROUTING_INFO_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_COUNT)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::RETRAIN_FORCE)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_ROUTER_CONFIG)] =
        MEM_ERISC_FABRIC_ROUTER_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX)] =
        sizeof(uint32_t) + (sizeof(uint32_t) * MEM_SYSENG_ETH_MAILBOX_NUM_ARGS);

    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_STATUS_MASK)] =
        MEM_SYSENG_ETH_MSG_STATUS_MASK;
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_CALL)] = MEM_SYSENG_ETH_MSG_CALL;
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_DONE)] = MEM_SYSENG_ETH_MSG_DONE;
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_LINK_STATUS_CHECK)] =
        MEM_SYSENG_ETH_MSG_LINK_STATUS_CHECK;
    fw_mailbox_addr[utils::underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_RELEASE_CORE)] =
        MEM_SYSENG_ETH_MSG_RELEASE_CORE;

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumEthDispatchClasses - 1);
    std::vector<HalJitBuildConfig> processor_types(1);
    for (std::size_t processor_class_idx = 0; processor_class_idx < processor_classes.size(); processor_class_idx++) {
        // BH active ethernet runs idle erisc FW on the second ethernet
        processor_types[0] = HalJitBuildConfig{
            .fw_base_addr = MEM_AERISC_FIRMWARE_BASE,
            .local_init_addr = MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH,
            .fw_launch_addr = SUBORDINATE_IERISC_RESET_PC,
            .fw_launch_addr_value = MEM_AERISC_FIRMWARE_BASE,
            .memory_load = ll_api::memory::Loading::CONTIGUOUS,
        };
        processor_classes[processor_class_idx] = processor_types;
    }

    static_assert(llrt_common::k_SingleProcessorMailboxSize<EthProcessorTypes> <= MEM_AERISC_MAILBOX_SIZE);
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

}  // namespace tt::tt_metal::blackhole
