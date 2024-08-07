// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hal.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "tt_metal/third_party/umd/device/tt_arch_types.h"

namespace tt {

namespace tt_metal {

Hal hal;

// This back poitner is a little clunky but necessary at least for now
Hal::Hal() : initialized_(false) {
}

void Hal::initialize(tt::ARCH arch) {

    const std::lock_guard<std::mutex> lock(this->lock);

    if (!this->initialized_) {
        switch (arch) {
        case tt::ARCH::GRAYSKULL:
            initialize_gs();
            break;

        case tt::ARCH::WORMHOLE_B0:
            initialize_wh();
            break;

        case tt::ARCH::BLACKHOLE:
            initialize_bh();
            break;

        default:
            TT_ASSERT("Unsupported arch for HAL");
        }

        this->initialized_ = true;
    }
}

HalCoreInfoType::HalCoreInfoType(HalProgrammableCoreType programmable_core_type,
                                 CoreType core_type,
                                 uint32_t core_proc_count,
                                 const std::vector<DeviceAddr>& core_mem_map) :
    programmable_core_type_(programmable_core_type),
    core_type_(core_type),
    proc_count_(core_proc_count),
    mem_map_(core_mem_map) {
}

}  // namespace tt_metal
}  // namespace tt
