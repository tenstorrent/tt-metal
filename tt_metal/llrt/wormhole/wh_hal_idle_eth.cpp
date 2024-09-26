// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_WORMHOLE_B0)

#define COMPILE_FOR_IDLE_ERISC

#include "llrt/hal.hpp"
#include "llrt/wormhole/wh_hal.hpp"
#include "hw/inc/wormhole/dev_mem_map.h"
#include "hw/inc/wormhole/eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "hw/inc/dev_msgs.h"

#define GET_IERISC_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_IERISC_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

HalCoreInfoType create_idle_eth_mem_map() {

    constexpr uint32_t num_proc_per_idle_eth_core = 1;

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(utils::underlying_type<HalMemAddrType>(HalMemAddrType::COUNT));
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH)] = GET_IERISC_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::WATCHER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::DPRINT)] = GET_IERISC_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::PROFILER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::KERNEL_CONFIG)] = IDLE_ERISC_L1_KERNEL_CONFIG_BASE;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::UNRESERVED)] = ERISC_L1_UNRESERVED_BASE;
    mem_map_bases[utils::underlying_type<HalMemAddrType>(HalMemAddrType::CORE_INFO)] = GET_IERISC_MAILBOX_ADDRESS_HOST(core_info);

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(utils::underlying_type<HalMemAddrType>(HalMemAddrType::COUNT));
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_SIZE; // TODO: this is wrong, need idle eth specific value
    mem_map_sizes[utils::underlying_type<HalMemAddrType>(HalMemAddrType::UNRESERVED)] = MEM_ETH_SIZE - ERISC_L1_UNRESERVED_BASE;

    return {HalProgrammableCoreType::IDLE_ETH, CoreType::ETH, num_proc_per_idle_eth_core, mem_map_bases, mem_map_sizes, false};
}

}  // namespace tt_metal
}  // namespace tt

#endif
