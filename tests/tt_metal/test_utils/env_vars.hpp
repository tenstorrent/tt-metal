// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "common/utils.hpp"

#include "third_party/umd/device/device_api_metal.h"

#include <string>

namespace {

std::string get_string_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE: return "wormhole"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole_b0"; break;
        case tt::ARCH::BLACKHOLE: return "blackhole"; break;
        case tt::ARCH::Invalid: return "invalid"; break;
        default: return "invalid"; break;
    }
}

}

namespace tt {
namespace test_utils {
inline std::string get_env_arch_name() {
    constexpr std::string_view ARCH_NAME_ENV_VAR = "ARCH_NAME";
    std::string arch_name;

    if (const char* arch_name_ptr = std::getenv(ARCH_NAME_ENV_VAR.data())) {
        arch_name = arch_name_ptr;
    } else {
        TT_THROW("Env var {} is not set.", ARCH_NAME_ENV_VAR);
    }
    return arch_name;
}

inline std::string get_umd_arch_name() {

    if(std::getenv("TT_METAL_SIMULATOR_EN")) {
        return get_env_arch_name();
    }

    std::vector<chip_id_t> physical_mmio_device_ids = tt_SiliconDevice::detect_available_device_ids();
    tt::ARCH arch = detect_arch(physical_mmio_device_ids.at(0));
    for (int dev_index = 1; dev_index < physical_mmio_device_ids.size(); dev_index++) {
        chip_id_t device_id = physical_mmio_device_ids.at(dev_index);
        tt::ARCH detected_arch = detect_arch(device_id);
        TT_FATAL(
            arch == detected_arch,
            "Expected all devices to be {} but device {} is {}",
            get_string_lowercase(arch),
            device_id,
            get_string_lowercase(detected_arch));
    }

    return get_string_lowercase(arch); 

}


}  // namespace test_utils
}  // namespace tt
