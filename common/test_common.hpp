
#pragma once

#include <functional>
#include <string>
#include <algorithm>
#include <filesystem>
#include "common/tt_soc_descriptor.h"

// Needed for TargetDevice enum
#include "common/base.hpp"

inline std::string get_soc_description_file(const tt::ARCH &arch, tt::TargetDevice target_device, string output_dir = "") {

    // Ability to skip this runtime opt, since trimmed SOC desc limits which DRAM channels are available.
    bool use_full_soc_desc = getenv("FORCE_FULL_SOC_DESC");
    string buda_home;
    if (getenv("TT_METAL_HOME")) {
        buda_home = getenv("TT_METAL_HOME");
    } else {
        buda_home = "./";
    }
    if (buda_home.back() != '/') {
        buda_home += "/";
    }
    if (target_device == tt::TargetDevice::Versim && !use_full_soc_desc) {
        TT_ASSERT(output_dir != "", "Output directory path is not set. In versim, soc-descriptor must get generated and copied to output-dir.");
        return output_dir + "/device_desc.yaml";
    } else {
        switch (arch) {
            case tt::ARCH::Invalid: return buda_home + "device/none.yaml"; // will be overwritten in tt_global_state constructor
            case tt::ARCH::JAWBRIDGE: throw std::runtime_error("JAWBRIDGE arch not supported");
            case tt::ARCH::GRAYSKULL: return buda_home + "device/grayskull_120_arch.yaml";
            case tt::ARCH::WORMHOLE: return buda_home + "device/wormhole_80_arch.yaml";
            case tt::ARCH::WORMHOLE_B0: return buda_home + "device/wormhole_b0_80_arch.yaml";
            default: throw std::runtime_error("Unsupported device arch");
        };
    }
    return "";
}
