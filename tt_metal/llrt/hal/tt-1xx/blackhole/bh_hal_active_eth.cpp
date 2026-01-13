// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define HAL_BUILD tt::tt_metal::blackhole::active_eth
#define COMPILE_FOR_ERISC

#include "tt_align.hpp"
#include "hostdev/dev_msgs.h"
#include "hostdev/fabric_telemetry_msgs.h"
using namespace tt::tt_metal::blackhole::active_eth;

#include <cstdint>

#include "blackhole/bh_hal.hpp"
#include "blackhole/bh_hal_eth_asserts.hpp"
#include "dev_mem_map.h"
#include "eth_l1_address_map.h"
#include "eth_fw_api.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include "noc/noc_parameters.h"

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) ((std::uint64_t)&(((mailboxes_t*)MEM_AERISC_MAILBOX_BASE)->x))

namespace tt::tt_metal::blackhole {

// This file is intended to be wrapped inside arch/core-specific namespace.
namespace active_eth_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

namespace active_eth_fabric_telemetry {
#include "hal/generated/fabric_telemetry_impl.hpp"
}

std::vector<std::vector<HalJitBuildConfig>> configure_for_1erisc() {
    return {
        // DM
        {
            // ERISC0
            {.fw_base_addr = MEM_AERISC_FIRMWARE_BASE,
             .local_init_addr = MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH,
             .fw_launch_addr = SUBORDINATE_AERISC_RESET_PC,
             .fw_launch_addr_value = MEM_AERISC_FIRMWARE_BASE,
             .memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP},
        },
    };
}

std::vector<std::vector<HalJitBuildConfig>> configure_for_2erisc() {
    return {
        // DM
        {
            // ERISC0
            {.fw_base_addr = MEM_AERISC_FIRMWARE_BASE,
             .local_init_addr = MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH,
             .fw_launch_addr = MEM_AERISC_VOID_LAUNCH_FLAG,
             .fw_launch_addr_value = MEM_AERISC_FIRMWARE_BASE,
             .memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP},
            // ERISC1
            {.fw_base_addr = MEM_SUBORDINATE_AERISC_FIRMWARE_BASE,
             .local_init_addr = MEM_SUBORDINATE_AERISC_INIT_LOCAL_L1_BASE_SCRATCH,
             .fw_launch_addr = SUBORDINATE_AERISC_RESET_PC,
             .fw_launch_addr_value = MEM_SUBORDINATE_AERISC_FIRMWARE_BASE,
             .memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP},
        },
    };
}

HalCoreInfoType create_active_eth_mem_map(bool enable_2_erisc_mode) {
    std::uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    static_assert(MEM_IERISC_MAP_END % L1_ALIGNMENT == 0);

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ERISC_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_AERISC_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_ETH_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(dprint_buf);
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
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORR_CW)] = MEM_CORR_CW_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNCORR_CW)] = MEM_UNCORR_CW_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_TELEMETRY)] = MEM_AERISC_FABRIC_TELEMETRY_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ROUTING_TABLE)] = MEM_AERISC_ROUTING_TABLE_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX)] = MEM_SYSENG_ETH_MAILBOX_ADDR;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LINK_UP)] = MEM_SYSENG_BOOT_RESULTS_BASE +
                                                                         offsetof(boot_results_t, eth_live_status) +
                                                                         offsetof(eth_live_status_t, rx_link_up);

    std::vector<std::uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ETH_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_ERISC_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = sizeof(dprint_buf_msg_t);
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
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_TELEMETRY)] = MEM_AERISC_FABRIC_TELEMETRY_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ROUTING_TABLE)] = MEM_AERISC_ROUTING_TABLE_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ETH_FW_MAILBOX)] =
        sizeof(uint32_t) + (sizeof(uint32_t) * MEM_SYSENG_ETH_MAILBOX_NUM_ARGS);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LINK_UP)] = sizeof(uint32_t);

    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_STATUS_MASK)] =
        MEM_SYSENG_ETH_MSG_STATUS_MASK;
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_CALL)] = MEM_SYSENG_ETH_MSG_CALL;
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_DONE)] = MEM_SYSENG_ETH_MSG_DONE;
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_LINK_STATUS_CHECK)] =
        MEM_SYSENG_ETH_MSG_LINK_STATUS_CHECK;
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::ETH_MSG_RELEASE_CORE)] =
        MEM_SYSENG_ETH_MSG_RELEASE_CORE;
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::HEARTBEAT)] = MEM_SYSENG_ETH_HEARTBEAT;
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::RETRAIN_COUNT)] =
        (uint64_t)&((eth_live_status_t*)MEM_SYSENG_ETH_LIVE_STATUS)->retrain_count;
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::RX_LINK_UP)] =
        (uint64_t)&((eth_live_status_t*)MEM_SYSENG_ETH_LIVE_STATUS)->rx_link_up;
    fw_mailbox_addr[ttsl::as_underlying_type<FWMailboxMsg>(FWMailboxMsg::PORT_STATUS)] =
        (uint64_t)&((eth_status_t*)MEM_SYSENG_ETH_STATUS)->port_status;

    std::vector<std::vector<HalJitBuildConfig>> processor_classes;
    std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names;
    if (enable_2_erisc_mode) {
        processor_classes = configure_for_2erisc();
        processor_classes_names = {{{"ER0", "ERISC0"}, {"ER1", "ERISC1"}}};
    } else {
        processor_classes = configure_for_1erisc();
        processor_classes_names = {{{"ER", "ERISC"}}};
    }

    static_assert(sizeof(mailboxes_t) <= MEM_AERISC_MAILBOX_SIZE);
    return {
        HalProgrammableCoreType::ACTIVE_ETH,
        CoreType::ETH,
        std::move(processor_classes),
        std::move(mem_map_bases),
        std::move(mem_map_sizes),
        std::move(fw_mailbox_addr),
        std::move(processor_classes_names),
        false /*supports_cbs*/,
        false /*supports_receiving_multicast_cmds*/,
        active_eth_dev_msgs::create_factory(),
        active_eth_fabric_telemetry::create_factory()};
}

}  // namespace tt::tt_metal::blackhole
