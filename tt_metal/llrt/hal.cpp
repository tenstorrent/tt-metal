// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hal.hpp"

#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/common/assert.hpp"

#include "get_platform_architecture.hpp"
namespace tt {

namespace tt_metal {

// Hal Constructor determines the platform architecture by using UMD
// Once it knows the architecture it can self initialize architecture specific memory maps
Hal::Hal() : arch_(get_platform_architecture()) {
    switch (this->arch_) {
        case tt::ARCH::GRAYSKULL: initialize_gs(); break;

        case tt::ARCH::WORMHOLE_B0: initialize_wh(); break;

        case tt::ARCH::BLACKHOLE: initialize_bh(); break;

        case tt::ARCH::Invalid: /*TT_THROW("Unsupported arch for HAL")*/; break;
    }
}

uint32_t Hal::get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const {
    uint32_t index = static_cast<uint32_t>(programmable_core_type_index);

    // TODO: this assumes unused entries occur at the end
    // Assumes unused indices go at the end
    if (index >= core_info_.size()) {
        return -1;
    } else {
        return index;
    }
}

uint32_t Hal::get_num_risc_processors() const {
    uint32_t num_riscs = 0;
    for (uint32_t core_idx = 0; core_idx < core_info_.size(); core_idx++) {
        uint32_t num_processor_classes = core_info_[core_idx].get_processor_classes_count();
        for (uint32_t processor_class_idx = 0; processor_class_idx < num_processor_classes; processor_class_idx++) {
            num_riscs += core_info_[core_idx].get_processor_types_count(processor_class_idx);
        }
    }
    return num_riscs;
}

HalCoreInfoType::HalCoreInfoType(
    HalProgrammableCoreType programmable_core_type,
    CoreType core_type,
    const std::vector<std::vector<HalJitBuildConfig>>& processor_classes,
    const std::vector<DeviceAddr>& mem_map_bases,
    const std::vector<uint32_t>& mem_map_sizes,
    bool supports_cbs) :
    programmable_core_type_(programmable_core_type),
    core_type_(core_type),
    processor_classes_(processor_classes),
    mem_map_bases_(mem_map_bases),
    mem_map_sizes_(mem_map_sizes),
    supports_cbs_(supports_cbs) {}

}  // namespace tt_metal
}  // namespace tt
