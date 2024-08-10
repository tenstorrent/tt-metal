// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt/hal.hpp"
#include "llrt/wormhole/wh_hal.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"

namespace tt {

namespace tt_metal {

void Hal::initialize_wh() {
#if defined(ARCH_WORMHOLE_B0)
    constexpr uint32_t num_proc_per_tensix_core = 5;
    std::vector<DeviceAddr> tensix_mem_map = create_tensix_mem_map();
    this->core_info_.push_back(
        {HalProgrammableCoreType::TENSIX, CoreType::WORKER, num_proc_per_tensix_core, tensix_mem_map}
    );

    constexpr uint32_t num_proc_per_active_eth_core = 1;
    std::vector<DeviceAddr> active_eth_mem_map = create_active_eth_mem_map();
    this->core_info_.push_back(
        {HalProgrammableCoreType::ACTIVE_ETH, CoreType::ETH, num_proc_per_active_eth_core, active_eth_mem_map}
    );

    constexpr uint32_t num_proc_per_idle_eth_core = 1;
    std::vector<DeviceAddr> idle_eth_mem_map = create_idle_eth_mem_map();
    this->core_info_.push_back(
        {HalProgrammableCoreType::IDLE_ETH, CoreType::ETH, num_proc_per_idle_eth_core, idle_eth_mem_map}
    );
#endif
}

}  // namespace tt_metal
}  // namespace tt
