// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define HAL_BUILD tt::tt_metal::quasar::dram
#define COMPILE_FOR_DRISC

#include "tt_align.hpp"
#include "hostdev/dev_msgs.h"
#include "hostdev/fabric_telemetry_msgs.h"
#include "hostdev/realtime_profiler_msgs.h"
using namespace tt::tt_metal::quasar::dram;

#include <algorithm>
#include <cstdint>
#include <fmt/format.h>

#include "quasar/qa_hal.hpp"
#include "dev_mem_map.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include <umd/device/types/core_coordinates.hpp>

#define GET_CCE_MAILBOX_ADDRESS_HOST(x) \
    (reinterpret_cast<std::uint64_t>(&(reinterpret_cast<mailboxes_t*>(MEM_CCE_MAILBOX_BASE)->x)))

namespace tt::tt_metal::quasar {

namespace dram_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

namespace dram_fabric_telemetry {
#include "hal/generated/fabric_telemetry_impl.hpp"
}

namespace dram_realtime_profiler_msgs {
#include "hal/generated/realtime_profiler_msgs_impl.hpp"
}

HalCoreInfoType create_dram_mem_map() {
    static_assert(decltype(DevicePrintMemoryLayout::buffer)::processor_count == PROCESSOR_COUNT);
    static_assert(sizeof(mailboxes_t) <= MEM_CCE_MAILBOX_SIZE);
    static_assert(MEM_CCE_WATCHER_RING_BUFFER_LOCK % 64 == 0);
    static_assert(MEM_CCE_WATCHER_RING_BUFFER_LOCK + 64 <= MEM_CCE_MAILBOX_BASE);
    static_assert(MEM_CCE_FIRMWARE_BASE % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
    static_assert((MEM_CCE_MAILBOX_BASE + offsetof(mailboxes_t, launch)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
    static_assert((MEM_CCE_MAILBOX_BASE + offsetof(mailboxes_t, profiler)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);
    static_assert(
        (MEM_CCE_MAILBOX_BASE + offsetof(mailboxes_t, go_message_index)) % TT_ARCH_MAX_NOC_WRITE_ALIGNMENT == 0);

    std::uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_CCE_L1_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_CCE_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_CCE_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_CCE_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] =
        GET_CCE_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_CCE_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_CCE_KERNEL_CONFIG_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        tt::align(MEM_CCE_KERNEL_BASE + MEM_CCE_KERNEL_SIZE * MEM_CCE_LOCAL_HARTS, max_alignment);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_CCE_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_CCE_MAILBOX_ADDRESS_HOST(go_messages);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] =
        GET_CCE_MAILBOX_ADDRESS_HOST(go_message_index);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_CCE_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_CCE_BANK_TO_NOC_SCRATCH;

    std::vector<std::uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_CCE_L1_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_CCE_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = sizeof(DevicePrintMemoryLayout);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_CCE_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)] =
        MEM_CCE_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::UNRESERVED)];
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t) * go_message_num_entries;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_CCE_BANK_TO_NOC_SIZE;

    assert_kernel_config_no_overlap(mem_map_bases, mem_map_sizes, HalL1MemAddrType::UNRESERVED, "CCE");

    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);

    std::vector<HalJitBuildConfig> dram_dm_processors;
    std::vector<std::pair<std::string, std::string>> dram_dm_names;
    dram_dm_processors.reserve(MEM_CCE_LOCAL_HARTS);
    dram_dm_names.reserve(MEM_CCE_LOCAL_HARTS);
    for (uint32_t hart = 0; hart < MEM_CCE_LOCAL_HARTS; hart++) {
        // One shared firmware binary. All CCE harts enter the same image; crt0/TLS are per mhartid.
        dram_dm_processors.push_back(
            {.fw_base_addr = MEM_CCE_FIRMWARE_BASE,
             .local_init_addr = MEM_CCE_INIT_LOCAL_L1_BASE_SCRATCH,
             .fw_launch_addr = CCE_BOOT_HART_RESET_VECTOR,
             .fw_launch_addr_value = MEM_CCE_SRAM_LOCAL_BASE + MEM_CCE_FIRMWARE_BASE,
             .memory_load = ll_api::memory::Loading::CONTIGUOUS,
             .l1_noc_offset = MEM_CCE_L1_NOC_OFFSET});
        dram_dm_names.emplace_back(fmt::format("CCE{}", hart), fmt::format("CCE{}", hart));
    }
    std::vector<std::vector<HalJitBuildConfig>> processor_classes = {std::move(dram_dm_processors)};
    std::vector<std::vector<std::pair<std::string, std::string>>> processor_classes_names = {std::move(dram_dm_names)};
    std::vector<uint8_t> processor_classes_num_fw_binaries = {/*DM*/ 1};

    return {
        HalProgrammableCoreType::DRAM,
        CoreType::DRAM,
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
        dram_fabric_telemetry::create_factory(),
        dram_realtime_profiler_msgs::create_factory()};
}

}  // namespace tt::tt_metal::quasar
