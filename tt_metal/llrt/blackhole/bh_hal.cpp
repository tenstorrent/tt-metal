// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt/hal.hpp"
#include "llrt/blackhole/bh_hal.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"

namespace tt {

namespace tt_metal {

static inline int hv (enum HalMemAddrType v) {
    return static_cast<int>(v);
}

void Hal::initialize_bh() {
#if defined(ARCH_BLACKHOLE)
    HalCoreInfoType tensix_mem_map = create_tensix_mem_map();
    this->core_info_.push_back(tensix_mem_map);

    HalCoreInfoType active_eth_mem_map = create_active_eth_mem_map();
    this->core_info_.push_back(active_eth_mem_map);

    HalCoreInfoType idle_eth_mem_map = create_idle_eth_mem_map();
    this->core_info_.push_back(idle_eth_mem_map);
#endif
}

}  // namespace tt_metal
}  // namespace tt
