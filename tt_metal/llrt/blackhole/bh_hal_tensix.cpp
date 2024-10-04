// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_BLACKHOLE)

#include "llrt/hal.hpp"
#include "llrt/blackhole/bh_hal.hpp"
#include "hw/inc/blackhole/dev_mem_map.h"
#include "hw/inc/blackhole/eth_l1_address_map.h"  // XXXX FIXME
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "hw/inc/dev_msgs.h"

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

HalCoreInfoType create_tensix_mem_map() {

    constexpr uint32_t num_proc_per_tensix_core = 5;

    std::vector<DeviceAddr> mem_map_bases;

    mem_map_bases.resize(utils::underlying_type<HalMemAddrType>(HalMemAddrType::COUNT));
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::DPRINT)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::PROFILER)] = GET_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_BASE;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::UNRESERVED)] = ((L1_KERNEL_CONFIG_BASE + L1_KERNEL_CONFIG_SIZE - 1) | (ALLOCATOR_ALIGNMENT - 1)) + 1;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::CORE_INFO)] = GET_MAILBOX_ADDRESS_HOST(core_info);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::GO_MSG)] = GET_MAILBOX_ADDRESS_HOST(go_message);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = GET_MAILBOX_ADDRESS_HOST(launch_msg_rd_ptr);


    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(utils::underlying_type<HalMemAddrType>(HalMemAddrType::COUNT));
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::UNRESERVED)] = MEM_L1_SIZE - mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::UNRESERVED)];
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::GO_MSG)] = sizeof(go_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)] = sizeof(uint32_t);

    return {HalProgrammableCoreType::TENSIX, CoreType::WORKER, num_proc_per_tensix_core, mem_map_bases, mem_map_sizes, true};
}

}  // namespace tt_metal
}  // namespace tt

#endif
