// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_WORMHOLE_B0)

#include "llrt/hal.hpp"
#include "llrt/wormhole/wh_hal.hpp"
#include "hw/inc/wormhole/dev_mem_map.h"
#include "hw/inc/wormhole/eth_l1_address_map.h"  // XXXX FIXME
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "hw/inc/dev_msgs.h"

#define GET_MAILBOX_ADDRESS_HOST(x) ((uint64_t) & (((mailboxes_t *)MEM_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

static inline int hv (enum HalMemAddrType v) {
    return static_cast<int>(v);
}

HalCoreInfoType create_tensix_mem_map() {

    constexpr uint32_t num_proc_per_tensix_core = 5;

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(hv(HalMemAddrType::COUNT));
    mem_map_bases[hv(HalMemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[hv(HalMemAddrType::LAUNCH)] = GET_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[hv(HalMemAddrType::WATCHER)] = GET_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[hv(HalMemAddrType::DPRINT)] = GET_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[hv(HalMemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_BASE;
    mem_map_bases[hv(HalMemAddrType::UNRESERVED)] = L1_UNRESERVED_BASE;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(hv(HalMemAddrType::COUNT));
    mem_map_sizes[hv(HalMemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[hv(HalMemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[hv(HalMemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[hv(HalMemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[hv(HalMemAddrType::KERNEL_CONFIG)] = L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[hv(HalMemAddrType::UNRESERVED)] = MEM_L1_SIZE - L1_UNRESERVED_BASE;

    return {HalProgrammableCoreType::TENSIX, CoreType::WORKER, num_proc_per_tensix_core, mem_map_bases, mem_map_sizes};
}

}  // namespace tt_metal
}  // namespace tt

#endif
