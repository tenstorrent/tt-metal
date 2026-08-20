// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define COMPILE_FOR_DISPATCH_ENGINE 1
#define HAL_BUILD tt::tt_metal::quasar::dispatch
#include "hostdev/dev_msgs.h"
#include "hostdev/fabric_telemetry_msgs.h"
#include "hostdev/realtime_profiler_msgs.h"
using namespace tt::tt_metal::quasar::dispatch;

#include <cstdint>

#include "quasar/qa_hal.hpp"
#include "quasar/qa_hal_dispatch_asserts.hpp"
#include "dev_mem_map.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "noc/noc_parameters.h"
#include "tensix.h"
#include "tt_align.hpp"
#include <umd/device/types/core_coordinates.hpp>

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t)&(((mailboxes_t*)MEM_MAILBOX_BASE)->x))

namespace tt::tt_metal::quasar {

namespace dispatch_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

namespace dispatch_fabric_telemetry {
#include "hal/generated/fabric_telemetry_impl.hpp"
}

namespace dispatch_realtime_profiler_msgs {
#include "hal/generated/realtime_profiler_msgs_impl.hpp"
}

HalCoreInfoType create_dispatch_mem_map() {
    uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    static constexpr DeviceAddr dispatch_dm_kernel_bases[] = {
        MEM_DISPATCH_DM0_KERNEL_BASE,
        MEM_DISPATCH_DM1_KERNEL_BASE,
        MEM_DISPATCH_DM2_KERNEL_BASE,
        MEM_DISPATCH_DM3_KERNEL_BASE,
        MEM_DISPATCH_DM4_KERNEL_BASE,
        MEM_DISPATCH_DM5_KERNEL_BASE,
        MEM_DISPATCH_DM6_KERNEL_BASE,
        MEM_DISPATCH_DM7_KERNEL_BASE,
    };
    static_assert(std::size(dispatch_dm_kernel_bases) == NUM_DM_CORES);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(1);
    processor_classes[0].reserve(NUM_DM_CORES);
    for (unsigned long dispatch_dm_kernel_base : dispatch_dm_kernel_bases) {
        processor_classes[0].push_back({
            .fw_base_addr = dispatch_dm_kernel_base,
            .local_init_addr = UINT32_MAX,
            .fw_launch_addr = 0x0,
            // DM firmware is linked and loaded at MEM_DISPATCH_DM_FIRMWARE_BASE (qa_hal.cpp passes it as
            // --defsym=__fw_text); per-DM fw_base_addr is the cq-kernel link/load slot only. Reset still
            // boots via JAL from L1[0] into firmware, so this must match where the firmware was linked.
            .fw_launch_addr_value = generate_risc_startup_addr(MEM_DISPATCH_DM_FIRMWARE_BASE),
            .memory_load = ll_api::memory::Loading::CONTIGUOUS,
        });
    }

    // Prefix with DE- so dispatch-engine DMs are distinguishable from Tensix DMs in DPRINT/watcher
    // output (e.g. "DE-DM4" vs "DM4"). Abbreviated and full names match so both GetRiscName paths agree.
    std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names = {
        {{"DE-DM0", "DE-DM0"},
         {"DE-DM1", "DE-DM1"},
         {"DE-DM2", "DE-DM2"},
         {"DE-DM3", "DE-DM3"},
         {"DE-DM4", "DE-DM4"},
         {"DE-DM5", "DE-DM5"},
         {"DE-DM6", "DE-DM6"},
         {"DE-DM7", "DE-DM7"}},
    };

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_DISPATCH_MAP_END;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_MAILBOX_ADDRESS_HOST(go_messages);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] =
        GET_MAILBOX_ADDRESS_HOST(go_message_index);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] = MEM_LOCAL_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_DISPATCH_BANK_TO_NOC_SCRATCH;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::ROUTING_TABLE)] = MEM_DISPATCH_TENSIX_ROUTING_TABLE_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS)] =
        MEM_DISPATCH_TENSIX_FABRIC_CONNECTIONS_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_CONNECTION_LOCK)] =
        MEM_DISPATCH_FABRIC_CONNECTION_LOCK_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)] =
        tt::align(DISPATCH_MEM_MAP_END, max_alignment);

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_DISPATCH_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = sizeof(DevicePrintMemoryLayout);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_DISPATCH_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t) * go_message_num_entries;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] = MEM_DM_LOCAL_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::ROUTING_TABLE)] = MEM_ROUTING_TABLE_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS)] =
        MEM_TENSIX_FABRIC_CONNECTIONS_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::FABRIC_CONNECTION_LOCK)] = MEM_FABRIC_CONNECTION_LOCK_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)] =
        MEM_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)];

    // Base FW mailbox not used on dispatch engines
    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);

    return HalCoreInfoType(
        HalProgrammableCoreType::DISPATCH,
        CoreType::DISPATCH,
        std::move(processor_classes),
        std::vector<uint8_t>{1},
        std::move(mem_map_bases),
        std::move(mem_map_sizes),
        std::move(fw_mailbox_addr),
        std::move(processor_classes_names),
        true,
        true,
        false,
        dispatch_dev_msgs::create_factory(),
        dispatch_fabric_telemetry::create_factory(),
        dispatch_realtime_profiler_msgs::create_factory());
}

}  // namespace tt::tt_metal::quasar
