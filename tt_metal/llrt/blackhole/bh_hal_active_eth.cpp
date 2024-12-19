// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define COMPILE_FOR_ERISC

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core_config.h"
#include "dev_msgs.h"
#include "eth_l1_address_map.h"

#include "hal.hpp"
#include "hal_asserts.hpp"
#include "blackhole/bh_hal.hpp"

#include "umd/device/tt_soc_descriptor.h"  // CoreType

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) \
    ((std::uint64_t)&(((mailboxes_t*)eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE)->x))

namespace tt::tt_metal::blackhole {

HalCoreInfoType create_active_eth_mem_map() {
    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT));
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
        eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::CORE_INFO)] = GET_ETH_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = GET_ETH_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] =
        GET_ETH_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] =
        eth_l1_mem::address_map::ERISC_MEM_BANK_TO_NOC_SCRATCH;

    std::vector<std::uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::size_t>(HalL1MemAddrType::COUNT));
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
        eth_l1_mem::address_map::MAX_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(std::uint32_t);
    mem_map_sizes[static_cast<std::size_t>(HalL1MemAddrType::BANK_TO_NOC_SCRATCH)] =
        eth_l1_mem::address_map::ERISC_MEM_BANK_TO_NOC_SIZE;

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumEthDispatchClasses - 1);
    std::vector<HalJitBuildConfig> processor_types(1);
    for (std::size_t processor_class_idx = 0; processor_class_idx < processor_classes.size(); processor_class_idx++) {
        // BH active ethernet runs idle erisc FW on the second ethernet
        processor_types[0] = HalJitBuildConfig{
            .fw_base_addr = eth_l1_mem::address_map::FIRMWARE_BASE,
            .local_init_addr = eth_l1_mem::address_map::FIRMWARE_BASE,  // this will be uplifted in subsequent commits
                                                                        // enabling active erisc
        };
        processor_classes[processor_class_idx] = processor_types;
    }

    return {HalProgrammableCoreType::ACTIVE_ETH, CoreType::ETH, processor_classes, mem_map_bases, mem_map_sizes, false};
}

}  // namespace tt::tt_metal::blackhole
