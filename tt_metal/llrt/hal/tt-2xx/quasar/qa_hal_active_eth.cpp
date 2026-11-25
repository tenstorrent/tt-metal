// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define HAL_BUILD tt::tt_metal::quasar::active_eth
#define COMPILE_FOR_ERISC

#include <tt-metalium/math.hpp>
#include "hostdev/keraunos_noc_channels.h"
#include "hostdev/dev_msgs.h"
#include "hostdevcommon/fabric_telemetry_msgs.h"
#include "hostdev/realtime_profiler_msgs.h"
using namespace tt::tt_metal::quasar::active_eth;

#include <algorithm>
#include <cstdint>
#include <fmt/format.h>

#include "quasar/qa_hal.hpp"
#include "quasar/qa_hal_eth_asserts.hpp"
#include "dev_mem_map.h"
#include "llrt/hal_types.hpp"
#include "llrt/hal.hpp"

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) ((std::uint64_t)&(((mailboxes_t*)MEM_AERISC_MAILBOX_BASE)->x))

namespace tt::tt_metal::quasar {

// This file is intended to be wrapped inside arch/core-specific namespace.
namespace active_eth_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

namespace active_eth_fabric_telemetry {
#include "hal/generated/fabric_telemetry_impl.hpp"
}

namespace active_eth_realtime_profiler_msgs {
#include "hal/generated/realtime_profiler_msgs_impl.hpp"
}

// A linked CCE runs the idle-Ethernet firmware, linked on the MEM_IERISC_* firmware and kernel slots, and its device
// code names these regions MEM_IERISC_*.
static_assert(MEM_AERISC_LOCAL_HARTS == MEM_IERISC_LOCAL_HARTS);
static_assert(MEM_AERISC_MAILBOX_BASE == MEM_IERISC_MAILBOX_BASE && MEM_AERISC_MAILBOX_SIZE == MEM_IERISC_MAILBOX_SIZE);
static_assert(
    MEM_AERISC_FIRMWARE_BASE == MEM_IERISC_FIRMWARE_BASE && MEM_AERISC_FIRMWARE_SIZE == MEM_IERISC_FIRMWARE_SIZE);
static_assert(MEM_AERISC_GLOBAL_BASE == MEM_IERISC_GLOBAL_BASE && MEM_AERISC_GLOBAL_SIZE == MEM_IERISC_GLOBAL_SIZE);
static_assert(MEM_AERISC_LOCAL_BASE == MEM_IERISC_LOCAL_BASE && MEM_AERISC_LOCAL_SIZE == MEM_IERISC_LOCAL_SIZE);
static_assert(MEM_AERISC_KERNEL_BASE == MEM_IERISC_KERNEL_BASE && MEM_AERISC_KERNEL_SIZE == MEM_IERISC_KERNEL_SIZE);
static_assert(MEM_AERISC_SPA_ROUTES_BASE == MEM_IERISC_SPA_ROUTES_BASE);
static_assert(MEM_AERISC_NOC_CHANNELS_BASE == MEM_IERISC_NOC_CHANNELS_BASE);
static_assert(MEM_AERISC_FABRIC_CONNECTIONS_BASE == MEM_IERISC_FABRIC_CONNECTIONS_BASE);
static_assert(
    MEM_AERISC_PACKET_HEADER_POOL_BASE == MEM_IERISC_PACKET_HEADER_POOL_BASE &&
    MEM_AERISC_PACKET_HEADER_POOL_SIZE == MEM_IERISC_PACKET_HEADER_POOL_SIZE);
static_assert(
    MEM_AERISC_FABRIC_ETH_LINK_BASE == MEM_IERISC_FABRIC_ETH_LINK_BASE &&
    MEM_AERISC_FABRIC_ETH_LINK_SIZE == MEM_IERISC_FABRIC_ETH_LINK_SIZE);
static_assert(
    MEM_AERISC_FABRIC_NOC_DWORD_STAGE_BASE == MEM_IERISC_FABRIC_NOC_DWORD_STAGE_BASE &&
    MEM_AERISC_FABRIC_NOC_DWORD_STAGE_SIZE == MEM_IERISC_FABRIC_NOC_DWORD_STAGE_SIZE);

HalCoreInfoType create_active_eth_mem_map(bool programmable) {
    static_assert(PROCESSOR_COUNT == MEM_AERISC_LOCAL_HARTS);
    static_assert(sizeof(mailboxes_t) <= MEM_AERISC_MAILBOX_SIZE);
    static_assert(sizeof(KeraunosNocChannels) <= MEM_AERISC_NOC_CHANNELS_SIZE);
    static_assert(sizeof(::tt::tt_fabric::fabric_telemetry::FabricTelemetry) <= MEM_AERISC_FABRIC_TELEMETRY_SIZE);

    std::uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_ETH_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = 64;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_AERISC_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_ETH_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_ETH_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_AERISC_KERNEL_CONFIG_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        tt::align(MEM_AERISC_UNRESERVED_BASE, max_alignment);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_ETH_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_ETH_MAILBOX_ADDRESS_HOST(go_messages);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(go_message_index);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_AERISC_BANK_TO_NOC_SCRATCH;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERAUNOS_NOC_CHANNELS)] = MEM_AERISC_NOC_CHANNELS_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ROUTING_TABLE)] = MEM_AERISC_ROUTING_TABLE_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS)] =
        MEM_AERISC_FABRIC_CONNECTIONS_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_TELEMETRY)] = MEM_AERISC_FABRIC_TELEMETRY_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ROUTER_STATE)] = MEM_AERISC_FABRIC_ROUTER_STATE_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ROUTER_COMMAND)] = MEM_AERISC_FABRIC_ROUTER_COMMAND_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ETH_LINK)] = MEM_AERISC_FABRIC_ETH_LINK_BASE;

    std::vector<std::uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_IERISC_L1_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_AERISC_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = sizeof(DevicePrintMemoryLayout);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_AERISC_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        MEM_IERISC_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_ring_t) * go_message_num_entries;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_AERISC_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERAUNOS_NOC_CHANNELS)] = MEM_AERISC_NOC_CHANNELS_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ROUTING_TABLE)] = MEM_AERISC_ROUTING_TABLE_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS)] =
        MEM_AERISC_FABRIC_CONNECTIONS_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_TELEMETRY)] = MEM_AERISC_FABRIC_TELEMETRY_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ROUTER_STATE)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ROUTER_COMMAND)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ETH_LINK)] = MEM_AERISC_FABRIC_ETH_LINK_SIZE;

    assert_kernel_config_no_overlap(mem_map_bases, mem_map_sizes, HalL1MemAddrType::UNRESERVED, "Active-eth");

    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);
    std::vector<std::vector<HalJitBuildConfig>> processor_classes;
    std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names;
    std::vector<uint8_t> processor_classes_num_fw_binaries;
    if (programmable) {
        std::vector<HalJitBuildConfig> processors;
        std::vector<std::pair<std::string, std::string>> names;
        processors.reserve(MEM_AERISC_LOCAL_HARTS);
        names.reserve(MEM_AERISC_LOCAL_HARTS);
        for (uint32_t hart = 0; hart < MEM_AERISC_LOCAL_HARTS; hart++) {
            processors.push_back(
                {.fw_base_addr = MEM_AERISC_FIRMWARE_BASE,
                 .local_init_addr = MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH,
                 .fw_launch_addr = ETH_BOOT_HART_RESET_VECTOR,
                 .fw_launch_addr_value = MEM_AERISC_FIRMWARE_BASE,
                 .memory_load = ll_api::memory::Loading::CONTIGUOUS,
                 .l1_noc_offset = 0});
            names.emplace_back(fmt::format("DM{}", hart), fmt::format("DM{}", hart));
        }
        processor_classes.push_back(std::move(processors));
        processor_classes_names.push_back(std::move(names));
        // A linked core has the CCE0 image only.
        processor_classes_num_fw_binaries.push_back(1);
    }

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
        false /*supports_dfbs*/,
        false /*supports_receiving_multicast_cmds*/,
        active_eth_dev_msgs::create_factory(),
        active_eth_fabric_telemetry::create_factory(),
        active_eth_realtime_profiler_msgs::create_factory()};
}

}  // namespace tt::tt_metal::quasar
