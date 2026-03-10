// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define HAL_BUILD tt::tt_metal::blackhole::dram
#define COMPILE_FOR_DRISC

#include "tt_align.hpp"
#include "hostdev/dev_msgs.h"
#include "hostdev/fabric_telemetry_msgs.h"
using namespace tt::tt_metal::blackhole::dram;

#include <cstdint>

#include "blackhole/bh_hal.hpp"
#include "dev_mem_map.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <umd/device/types/core_coordinates.hpp>

#define GET_DRISC_MAILBOX_ADDRESS_HOST(x) \
    (reinterpret_cast<std::uint64_t>(&(reinterpret_cast<mailboxes_t*>(MEM_DRISC_MAILBOX_BASE)->x)))

namespace tt::tt_metal::blackhole {

namespace dram_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

namespace dram_fabric_telemetry {
#include "hal/generated/fabric_telemetry_impl.hpp"
}

HalCoreInfoType create_dram_mem_map() {
    static_assert(sizeof(mailboxes_t) <= MEM_DRISC_MAILBOX_SIZE);
    static_assert(MEM_DRISC_FIRMWARE_BASE % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);

    std::uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = 0;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_DRISC_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_DRISC_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_DRISC_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] =
        GET_DRISC_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_DRISC_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_DRISC_KERNEL_CONFIG_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        tt::align(MEM_DRISC_KERNEL_CONFIG_BASE + MEM_DRISC_KERNEL_CONFIG_SIZE, max_alignment);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_DRISC_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_DRISC_MAILBOX_ADDRESS_HOST(go_messages);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] =
        GET_DRISC_MAILBOX_ADDRESS_HOST(go_message_index);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_DRISC_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_DRISC_BANK_TO_NOC_SCRATCH;

    std::vector<std::uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_DRISC_L1_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_DRISC_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_DRISC_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        MEM_DRISC_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t) * go_message_num_entries;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_DRISC_BANK_TO_NOC_SIZE;

    // No FW mailbox on DRAM cores
    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);

    constexpr DeviceAddr DRAM_L1_NOC_OFFSET = 0x2000000000ULL;

    std::vector<std::vector<HalJitBuildConfig>> processor_classes = {
        // DM
        {
            // DRISC0
            {.fw_base_addr = MEM_DRISC_FIRMWARE_BASE,
             .local_init_addr = MEM_DRISC_INIT_LOCAL_L1_BASE_SCRATCH,
             .fw_launch_addr = DRISC_RESET_PC,
             .fw_launch_addr_value = MEM_DRISC_FIRMWARE_BASE,
             .memory_load = ll_api::memory::Loading::CONTIGUOUS,
             .l1_noc_offset = DRAM_L1_NOC_OFFSET},
        },
    };
    std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names = {
        // DM
        {
            {"DR0", "DRISC0"},
        },
    };
    std::vector<uint8_t> processor_classes_num_fw_binaries = {/*DM*/ 1};

    return {
        HalProgrammableCoreType::DRAM,
        CoreType::DRAM_WORKER,
        std::move(processor_classes),
        std::move(processor_classes_num_fw_binaries),
        std::move(mem_map_bases),
        std::move(mem_map_sizes),
        std::move(fw_mailbox_addr),
        std::move(processor_classes_names),
        false /*supports_cbs*/,
        false /*supports_dfbs*/,
        false /*supports_receiving_multicast_cmds*/,
        dram_dev_msgs::create_factory(),
        dram_fabric_telemetry::create_factory()};
}

}  // namespace tt::tt_metal::blackhole
