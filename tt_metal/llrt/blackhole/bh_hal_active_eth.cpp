// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_BLACKHOLE)

#define COMPILE_FOR_IDLE_ERISC

#include <cstdint>

#include "llrt/hal.hpp"
#include "llrt/blackhole/bh_hal.hpp"
#include "hw/inc/blackhole/core_config.h"
#include "hw/inc/blackhole/dev_mem_map.h"
#include "hw/inc/blackhole/eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "hw/inc/dev_msgs.h"

#include <magic_enum.hpp>

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) \
    ((uint64_t) & (((mailboxes_t *)eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

HalCoreInfoType create_active_eth_mem_map() {

    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::COUNT));
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::BARRIER)] = eth_l1_mem::address_map::ERISC_BARRIER_BASE;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::MAILBOX)] = eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::WATCHER)] = GET_ETH_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::DPRINT)] = GET_ETH_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::PROFILER)] = GET_ETH_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::KERNEL_CONFIG)] = eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_BASE;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::UNRESERVED)] = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::CORE_INFO)] = GET_ETH_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::GO_MSG)] = GET_ETH_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);
    mem_map_bases[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::FW_VERSION_ADDR)] = eth_l1_mem::address_map::FW_VERSION_ADDR;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::COUNT));
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::BARRIER)] = eth_l1_mem::address_map::ERISC_BARRIER_SIZE;
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::MAILBOX)] = eth_l1_mem::address_map::ERISC_MEM_MAILBOX_SIZE;
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::KERNEL_CONFIG)] = eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::UNRESERVED)] = eth_l1_mem::address_map::MAX_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);
    mem_map_sizes[utils::underlying_type<HalL1MemAddrType>(HalL1MemAddrType::FW_VERSION_ADDR)] = sizeof(std::uint32_t);

    std::vector<std::vector<HalJitBuildConfig>> processor_classes(NumEthDispatchClasses - 1);
    std::vector<HalJitBuildConfig> processor_types(1);
    for (uint8_t processor_class_idx = 0; processor_class_idx < processor_classes.size(); processor_class_idx++) {
        // BH active ethernet runs idle erisc FW on the second ethernet
        processor_types[0] = HalJitBuildConfig{
            .fw_base_addr = eth_l1_mem::address_map::FIRMWARE_BASE,
            .local_init_addr = MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH
        };
        processor_classes[processor_class_idx] = processor_types;
    }

    return {HalProgrammableCoreType::ACTIVE_ETH, CoreType::ETH, processor_classes, mem_map_bases, mem_map_sizes, false};
}

}  // namespace tt_metal
}  // namespace tt
#endif
