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
#include "hw/inc/dev_msgs.h"

#define GET_IERISC_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_IERISC_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

std::vector<DeviceAddr> create_idle_eth_mem_map() {

    std::vector<DeviceAddr> idle_eth_mem_map;
    idle_eth_mem_map.resize(HalMemAddrType::HAL_L1_MEM_ADDR_COUNT);
    idle_eth_mem_map[HAL_L1_MEM_ADDR_BARRIER] = MEM_L1_BARRIER;
    idle_eth_mem_map[HAL_L1_MEM_ADDR_LAUNCH] = GET_IERISC_MAILBOX_ADDRESS_HOST(launch);
    idle_eth_mem_map[HAL_L1_MEM_ADDR_WATCHER] = GET_IERISC_MAILBOX_ADDRESS_HOST(watcher);
    idle_eth_mem_map[HAL_L1_MEM_ADDR_DPRINT] = GET_IERISC_MAILBOX_ADDRESS_HOST(dprint_buf);
    //idle_eth_mem_map[HAL_L1_MEM_ADDR_PROFILER] = 4;
    idle_eth_mem_map[HAL_L1_MEM_ADDR_KERNEL_CONFIG_BASE] = IDLE_ERISC_L1_KERNEL_CONFIG_BASE;
    idle_eth_mem_map[HAL_L1_MEM_ADDR_UNRESERVED_BASE] = ERISC_L1_UNRESERVED_BASE;

    return idle_eth_mem_map;
}

}  // namespace tt_metal
}  // namespace tt

#endif
