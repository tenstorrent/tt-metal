// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>

#include "hal.hpp"

#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/third_party/umd/device/cluster.h"

namespace tt {

namespace tt_metal {

// Hal Constructor determines the platform architecture by using UMD
// Once it knows the architecture it can self initialize architecture specific memory maps
Hal::Hal() {
    this->arch_ = tt::ARCH::Invalid;
    if(std::getenv("TT_METAL_SIMULATOR_EN")) {
        auto arch_env = std::getenv("ARCH_NAME");
        TT_FATAL(arch_env, "ARCH_NAME env var needed for VCS");
        this->arch_ = tt::get_arch_from_string(arch_env);
    }else {
        std::vector<chip_id_t> physical_mmio_device_ids = tt::umd::Cluster::detect_available_device_ids();
        //TT_FATAL(physical_mmio_device_ids.size() > 0, "Could not detect any devices");
        if (physical_mmio_device_ids.size() > 0) {
            this->arch_ = detect_arch(physical_mmio_device_ids.at(0));
            for (int i = 1; i < physical_mmio_device_ids.size(); ++i) {
                chip_id_t device_id = physical_mmio_device_ids.at(i);
                tt::ARCH detected_arch = detect_arch(device_id);
                TT_FATAL(
                    this->arch_ == detected_arch,
                    "Expected all devices to be {} but device {} is {}",
                    get_arch_str(this->arch_),
                    device_id,
                    get_arch_str(detected_arch));
            }
        }
    }
    switch (this->arch_) {
        case tt::ARCH::GRAYSKULL: initialize_gs();
        break;

        case tt::ARCH::WORMHOLE_B0: initialize_wh();
        break;

        case tt::ARCH::BLACKHOLE: initialize_bh();
        break;

        case tt::ARCH::Invalid: /*TT_THROW("Unsupported arch for HAL")*/;
        break;
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

HalCoreInfoType::HalCoreInfoType(HalProgrammableCoreType programmable_core_type,
                                 CoreType core_type,
                                 const std::vector<std::vector<uint8_t>> &processor_classes,
                                 const std::vector<DeviceAddr>& mem_map_bases,
                                 const std::vector<uint32_t>& mem_map_sizes,
                                 bool supports_cbs) :
    programmable_core_type_(programmable_core_type),
    core_type_(core_type),
    processor_classes_(processor_classes),
    mem_map_bases_(mem_map_bases),
    mem_map_sizes_(mem_map_sizes),
    supports_cbs_(supports_cbs) {
}

}  // namespace tt_metal
}  // namespace tt
