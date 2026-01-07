// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/mock_device.hpp>

#include <cstdlib>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "llrt/get_platform_architecture.hpp"

namespace tt::tt_metal::experimental {

// Static storage for registered mock config
static std::optional<MockDeviceConfig> g_registered_mock_config = std::nullopt;

void configure_mock_mode(tt::ARCH arch, uint32_t num_chips) {
    g_registered_mock_config = MockDeviceConfig{arch, num_chips};
    log_info(tt::LogMetal, "Mock mode configured: arch={}, num_chips={}", static_cast<int>(arch), num_chips);
}

void configure_mock_mode_from_hw() {
    tt::ARCH arch = get_physical_architecture();
    // Default to 1 chip - could be enhanced to detect actual chip count
    uint32_t num_chips = 1;
    configure_mock_mode(arch, num_chips);
}

void disable_mock_mode() {
    g_registered_mock_config = std::nullopt;
    log_info(tt::LogMetal, "Mock mode disabled");
}

bool is_mock_mode_registered() {
    return g_registered_mock_config.has_value();
}

std::optional<MockDeviceConfig> get_registered_mock_config() {
    return g_registered_mock_config;
}

std::string get_mock_cluster_desc_path(const MockDeviceConfig& config) {
    // Get root dir from environment variable (set before MetalContext initialization)
    const char* root_dir = std::getenv("TT_METAL_HOME");
    if (root_dir == nullptr) {
        root_dir = std::getenv("TT_METAL_RUNTIME_ROOT");
    }
    TT_FATAL(root_dir != nullptr,
        "TT_METAL_HOME or TT_METAL_RUNTIME_ROOT environment variable must be set for mock device mode");

    std::string base_path = std::string(root_dir) +
                            "/tt_metal/third_party/umd/tests/cluster_descriptor_examples/";

    if (config.arch == tt::ARCH::WORMHOLE_B0) {
        switch (config.num_chips) {
            case 1: return base_path + "wormhole_N150.yaml";
            case 2: return base_path + "wormhole_N300.yaml";
            case 4: return base_path + "wormhole_2xN300_unconnected.yaml";
            case 8: return base_path + "t3k_cluster_desc.yaml";
            case 32: return base_path + "tg_cluster_desc.yaml";
            default: break;
        }
    } else if (config.arch == tt::ARCH::BLACKHOLE) {
        switch (config.num_chips) {
            case 1: return base_path + "blackhole_P100.yaml";
            case 2: return base_path + "blackhole_P150.yaml";
            case 4: return base_path + "blackhole_P300_first_mmio.yaml";
            default: break;
        }
    }

    TT_THROW(
        "Unsupported mock device configuration: arch={}, num_chips={}",
        static_cast<int>(config.arch),
        config.num_chips);
}

}  // namespace tt::tt_metal::experimental
