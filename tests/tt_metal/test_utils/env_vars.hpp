// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/utils.hpp>

#include "umd/device/device_api_metal.h"
#include "umd/device/tt_cluster_descriptor.h"
#include "umd/device/tt_simulation_device.h"
#include "impl/context/metal_context.hpp"

#include <string>

inline std::string get_string_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole_b0"; break;
        case tt::ARCH::BLACKHOLE: return "blackhole"; break;
        case tt::ARCH::Invalid: return "invalid"; break;
        default: return "invalid"; break;
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

    if(tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled()) {
        tt_SimulationDeviceInit init(tt_metal::MetalContext::instance().rtoptions().get_simulator_path());
        return tt::arch_to_str(init.get_arch_name());
    }

    auto cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
    const std::unordered_set<chip_id_t> &device_ids = cluster_desc->get_all_chips();
    tt::ARCH arch = cluster_desc->get_arch(*device_ids.begin());
    for (auto device_id : device_ids) {
        tt::ARCH detected_arch = cluster_desc->get_arch(device_id);
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
