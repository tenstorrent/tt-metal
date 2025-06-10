// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dev_msgs.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "assert.hpp"
#include "blackhole/bh_hal.hpp"
#include "blackhole/bh_hal_tensix_asserts.hpp"
#include "core_config.h"
#include "dev_mem_map.h"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "llrt_common/mailbox.hpp"
#include "noc/noc_parameters.h"
#include "tensix.h"
#include <umd/device/tt_core_coordinates.h>

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t)&(((mailboxes_t*)MEM_MAILBOX_BASE)->x))

namespace tt::tt_metal::blackhole {

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
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = GET_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::KERNEL_CONFIG)] = MEM_MAP_END;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] = MEM_LOCAL_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_BANK_TO_NOC_SCRATCH;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)] =
        ((MEM_MAP_END + default_l1_kernel_config_size - 1) | (max_alignment - 1)) + 1;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT), 0);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BASE)] = MEM_L1_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::MAILBOX)] = MEM_MAILBOX_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LOCAL)] = MEM_TRISC_LOCAL_SIZE; // TRISC, BRISC, or NCRISC?
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] = MEM_BANK_TO_NOC_SIZE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)] =
        MEM_L1_SIZE - mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::DEFAULT_UNRESERVED)];

    // Base FW api not supported on WH
    std::vector<uint32_t> fw_mailbox_addr(static_cast<std::size_t>(FWMailboxMsg::COUNT), 0);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumTensixDispatchClasses);
    std::vector<HalJitBuildConfig> processor_types;
    for (uint8_t processor_class_idx = 0; processor_class_idx < NumTensixDispatchClasses; processor_class_idx++) {
        uint32_t num_processors = processor_class_idx == (NumTensixDispatchClasses - 1) ? 3 : 1;
        processor_types.resize(num_processors);
        for (size_t processor_type_idx = 0; processor_type_idx < processor_types.size(); processor_type_idx++) {
            DeviceAddr fw_base{}, local_init{}, fw_launch{};
            uint32_t fw_launch_value{};
            ll_api::memory::Loading memory_load = ll_api::memory::Loading::CONTIGUOUS_XIP;
            switch (processor_class_idx) {
                case 0: {
                    fw_base = MEM_BRISC_FIRMWARE_BASE;
                    local_init = MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH;
                    fw_launch = 0x0; // BRISC is hardcoded to have reset PC of 0
                    fw_launch_value = generate_risc_startup_addr(fw_base);
                }
                break;
                case 1: {
                    fw_base = MEM_NCRISC_FIRMWARE_BASE;
                    local_init = MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH;
                    fw_launch = RISCV_DEBUG_REG_NCRISC_RESET_PC;
                    fw_launch_value = fw_base;
                }
                break;
                case 2: {
                    switch (processor_type_idx) {
                        case 0: {
                            fw_base = MEM_TRISC0_FIRMWARE_BASE;
                            local_init = MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH;
                            fw_launch = RISCV_DEBUG_REG_TRISC0_RESET_PC;
                            fw_launch_value = fw_base;
                        }
                        break;
                        case 1: {
                            fw_base = MEM_TRISC1_FIRMWARE_BASE;
                            local_init = MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH;
                            fw_launch = RISCV_DEBUG_REG_TRISC1_RESET_PC;
                            fw_launch_value = fw_base;
                        }
                        break;
                        case 2: {
                            fw_base = MEM_TRISC2_FIRMWARE_BASE;
                            local_init = MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH;
                            fw_launch = RISCV_DEBUG_REG_TRISC2_RESET_PC;
                            fw_launch_value = fw_base;
                        }
                        break;
                    }
                } break;
                default: TT_THROW("Unexpected processor class {} for Blackhole Tensix", processor_class_idx);
            }

            processor_types[processor_type_idx] = HalJitBuildConfig{
                .fw_base_addr = fw_base,
                .local_init_addr = local_init,
                .fw_launch_addr = fw_launch,
                .fw_launch_addr_value = fw_launch_value,
                .memory_load = memory_load,
            };
        }
        processor_classes[processor_class_idx] = processor_types;
    }
    static_assert(llrt_common::k_SingleProcessorMailboxSize<TensixProcessorTypes> <= MEM_MAILBOX_SIZE);
    return {
        HalProgrammableCoreType::TENSIX,
        CoreType::WORKER,
        processor_classes,
        mem_map_bases,
        mem_map_sizes,
        fw_mailbox_addr,
        true /*supports_cbs*/,
        true /*supports_receiving_multicast_cmds*/};
}

}  // namespace tt::tt_metal::blackhole
