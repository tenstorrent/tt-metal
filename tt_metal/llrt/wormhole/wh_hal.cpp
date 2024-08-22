// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_config.h"
#include "llrt/hal.hpp"
#include "llrt/wormhole/wh_hal.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"

namespace tt {

namespace tt_metal {

void Hal::initialize_wh() {
#if defined(ARCH_WORMHOLE_B0)
    static_assert(static_cast<int>(HalProgrammableCoreType::TENSIX) == static_cast<int>(ProgrammableCoreType::TENSIX));
    static_assert(static_cast<int>(HalProgrammableCoreType::ACTIVE_ETH) == static_cast<int>(ProgrammableCoreType::ACTIVE_ETH));
    static_assert(static_cast<int>(HalProgrammableCoreType::IDLE_ETH) == static_cast<int>(ProgrammableCoreType::IDLE_ETH));

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
