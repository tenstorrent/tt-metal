// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/driver_atomics.hpp>
#include <umd/device/cluster_descriptor.hpp>
#include <umd/device/simulation/simulation_chip.hpp>
#include "impl/context/metal_context.hpp"

#include <string>

inline std::string get_string_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull";
        case tt::ARCH::WORMHOLE_B0: return "wormhole_b0";
        case tt::ARCH::BLACKHOLE: return "blackhole";
        case tt::ARCH::Invalid:
        default: return "invalid";
    }
}

namespace tt::test_utils {
inline std::string get_env_arch_name() {
    constexpr auto ARCH_NAME_ENV_VAR = "ARCH_NAME";

    auto* arch_name_ptr = std::getenv(ARCH_NAME_ENV_VAR);
    if (!arch_name_ptr) {
        TT_THROW("Env var {} is not set.", ARCH_NAME_ENV_VAR);
    }

    return std::string(arch_name_ptr);
}

inline std::string get_umd_arch_name() {

    if(tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled()) {
        auto soc_desc = tt::umd::SimulationChip::get_soc_descriptor_path_from_simulator_path(tt_metal::MetalContext::instance().rtoptions().get_simulator_path());
        return tt::arch_to_str(tt::umd::SocDescriptor::get_arch_from_soc_descriptor_path(soc_desc));
    }

    auto cluster_desc = tt::umd::Cluster::create_cluster_descriptor();
    const std::unordered_set<ChipId> &device_ids = cluster_desc->get_all_chips();
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

}  // namespace tt::test_utils
