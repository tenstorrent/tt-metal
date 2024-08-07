// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_BLACKHOLE)

#define COMPILE_FOR_ERISC

#include "llrt/hal.hpp"
#include "llrt/wormhole/wh_hal.hpp"
#include "hw/inc/blackhole/dev_mem_map.h"
#include "hw/inc/blackhole/eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hw/inc/dev_msgs.h"

#define GET_IERISC_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_IERISC_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

static inline int hv (enum HalMemAddrType v) {
    return static_cast<int>(v);
}

std::vector<DeviceAddr> create_idle_eth_mem_map() {

    std::vector<DeviceAddr> idle_eth_mem_map;
    idle_eth_mem_map.resize(hv(HalMemAddrType::COUNT));
    idle_eth_mem_map[hv(HalMemAddrType::BARRIER)] = MEM_L1_BARRIER;
    idle_eth_mem_map[hv(HalMemAddrType::LAUNCH)] = GET_IERISC_MAILBOX_ADDRESS_HOST(launch);
    idle_eth_mem_map[hv(HalMemAddrType::WATCHER)] = GET_IERISC_MAILBOX_ADDRESS_HOST(watcher);
    idle_eth_mem_map[hv(HalMemAddrType::DPRINT)] = GET_IERISC_MAILBOX_ADDRESS_HOST(dprint_buf);
    idle_eth_mem_map[hv(HalMemAddrType::KERNEL_CONFIG_BASE)] = IDLE_ERISC_L1_KERNEL_CONFIG_BASE;
    idle_eth_mem_map[hv(HalMemAddrType::UNRESERVED_BASE)] = ERISC_L1_UNRESERVED_BASE;

    return idle_eth_mem_map;
}

}  // namespace tt_metal
}  // namespace tt

#endif
