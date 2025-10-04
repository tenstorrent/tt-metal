// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define HAL_BUILD tt::tt_metal::quasar::tensix
#include "dev_msgs.h"
using namespace tt::tt_metal::quasar::tensix;

#include <cstdint>

#include <tt_stl/assert.hpp>
#include "quasar/qa_hal.hpp"
#include "quasar/qa_hal_tensix_asserts.hpp"
#include "dev_mem_map.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "noc/noc_parameters.h"
#include "tensix.h"
#include <umd/device/types/core_coordinates.hpp>

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t)&(((mailboxes_t*)MEM_MAILBOX_BASE)->x))

namespace tt::tt_metal::quasar {

// This file is intended to be wrapped inside arch/core-specific namespace.
namespace tensix_dev_msgs {
#include "hal/generated/dev_msgs_impl.hpp"
}

HalCoreInfoType create_tensix_mem_map() {
    uint32_t max_alignment = std::max(DRAM_ALIGNMENT, L1_ALIGNMENT);

    std::vector<DeviceAddr> mem_map_bases;
    const uint32_t default_l1_kernel_config_size = 69 * 1024;

    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_MAILBOX_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_MAP_END;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_MAILBOX_ADDRESS_HOST(go_messages);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] =
        GET_MAILBOX_ADDRESS_HOST(go_message_index);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] = MEM_LOCAL_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_BANK_TO_NOC_SCRATCH;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_ROUTING_TABLE)] = MEM_TENSIX_ROUTING_TABLE_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS)] =
        MEM_TENSIX_FABRIC_CONNECTIONS_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_ROUTING_PATH_1D)] = MEM_TENSIX_ROUTING_PATH_BASE_1D;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_ROUTING_PATH_2D)] = MEM_TENSIX_ROUTING_PATH_BASE_2D;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)] =
        ((MEM_MAP_END + default_l1_kernel_config_size - 1) | (max_alignment - 1)) + 1;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT_BUFFERS)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t) * go_message_num_entries;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG_INDEX)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] =
        MEM_TRISC_LOCAL_SIZE;  // TRISC, BRISC, or NCRISC?
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_ROUTING_TABLE)] = MEM_ROUTING_TABLE_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS)] =
        MEM_TENSIX_FABRIC_CONNECTIONS_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_ROUTING_PATH_1D)] = ROUTING_PATH_SIZE_1D;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::TENSIX_ROUTING_PATH_2D)] = COMPRESSED_ROUTING_PATH_SIZE_2D;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)] =
        MEM_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)];

    // Base FW api not supported on WH
    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes = {
        // DM
        {
            {.fw_base_addr = MEM_DM_FIRMWARE_BASE,
             .local_init_addr = MEM_DM0_INIT_LOCAL_L1_BASE_SCRATCH,
             .fw_launch_addr = 0x0,
             .fw_launch_addr_value = generate_risc_startup_addr(MEM_DM_FIRMWARE_BASE),
             .memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP},
        },
        // COMPUTE
        // {
        //     // TRISC0
        //     {.fw_base_addr = MEM_TRISC0_FIRMWARE_BASE,
        //      .local_init_addr = MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH,
        //      .fw_launch_addr = RISCV_DEBUG_REG_TRISC0_RESET_PC,
        //      .fw_launch_addr_value = MEM_TRISC0_FIRMWARE_BASE,
        //      .memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP},
        //     // TRISC1
        //     {.fw_base_addr = MEM_TRISC1_FIRMWARE_BASE,
        //      .local_init_addr = MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH,
        //      .fw_launch_addr = RISCV_DEBUG_REG_TRISC1_RESET_PC,
        //      .fw_launch_addr_value = MEM_TRISC1_FIRMWARE_BASE,
        //      .memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP},
        //     // TRISC2
        //     {.fw_base_addr = MEM_TRISC2_FIRMWARE_BASE,
        //      .local_init_addr = MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH,
        //      .fw_launch_addr = RISCV_DEBUG_REG_TRISC2_RESET_PC,
        //      .fw_launch_addr_value = MEM_TRISC2_FIRMWARE_BASE,
        //      .memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP},
        // },
    };
    static_assert(sizeof(mailboxes_t) <= MEM_MAILBOX_SIZE);
    return {
        HalProgrammableCoreType::TENSIX,
        CoreType::WORKER,
        std::move(processor_classes),
        std::move(mem_map_bases),
        std::move(mem_map_sizes),
        std::move(fw_mailbox_addr),
        true /*supports_cbs*/,
        true /*supports_receiving_multicast_cmds*/,
        tensix_dev_msgs::create_factory()};
}

}  // namespace tt::tt_metal::quasar
