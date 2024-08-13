// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_BLACKHOLE)

#define COMPILE_FOR_IDLE_ERISC

#include "llrt/hal.hpp"
#include "llrt/wormhole/wh_hal.hpp"
#include "hw/inc/blackhole/dev_mem_map.h"
#include "hw/inc/blackhole/eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hw/inc/dev_msgs.h"

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) \
    ((uint64_t) & (((mailboxes_t *)eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

static inline int hv (enum HalMemAddrType v) {
    return static_cast<int>(v);
}

std::vector<DeviceAddr> create_active_eth_mem_map() {

    std::vector<DeviceAddr> active_eth_mem_map;
    active_eth_mem_map.resize(hv(HalMemAddrType::COUNT));
    active_eth_mem_map[hv(HalMemAddrType::BARRIER)] = MEM_L1_BARRIER;
    active_eth_mem_map[hv(HalMemAddrType::LAUNCH)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch);
    active_eth_mem_map[hv(HalMemAddrType::WATCHER)] = GET_ETH_MAILBOX_ADDRESS_HOST(watcher);
    active_eth_mem_map[hv(HalMemAddrType::DPRINT)] = GET_ETH_MAILBOX_ADDRESS_HOST(dprint_buf);
    active_eth_mem_map[hv(HalMemAddrType::KERNEL_CONFIG_BASE)] = eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_BASE;
    active_eth_mem_map[hv(HalMemAddrType::UNRESERVED_BASE)] = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    return active_eth_mem_map;
}

}  // namespace tt_metal
}  // namespace tt
#endif
