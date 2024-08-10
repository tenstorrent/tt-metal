// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_BLACKHOLE)

#include "llrt/hal.hpp"
#include "llrt/wormhole/wh_hal.hpp"
#include "hw/inc/blackhole/dev_mem_map.h"
#include "hw/inc/blackhole/eth_l1_address_map.h"  // XXXX FIXME
#include "hostdevcommon/common_runtime_address_map.h"
#include "hw/inc/dev_msgs.h"

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

static inline int hv (enum HalMemAddrType v) {
    return static_cast<int>(v);
}

std::vector<DeviceAddr> create_tensix_mem_map() {

    std::vector<DeviceAddr> tensix_mem_map;
    tensix_mem_map.resize(hv(HalMemAddrType::COUNT));
    tensix_mem_map[hv(HalMemAddrType::BARRIER)] = MEM_L1_BARRIER;
    tensix_mem_map[hv(HalMemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    tensix_mem_map[hv(HalMemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    tensix_mem_map[hv(HalMemAddrType::DPRINT)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    tensix_mem_map[hv(HalMemAddrType::KERNEL_CONFIG_BASE)] = L1_KERNEL_CONFIG_BASE;
    tensix_mem_map[hv(HalMemAddrType::UNRESERVED_BASE)] = L1_UNRESERVED_BASE;

    return tensix_mem_map;
}

}  // namespace tt_metal
}  // namespace tt

#endif
