// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/mock_device.hpp>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include "llrt/get_platform_architecture.hpp"

namespace tt::tt_metal::experimental {

// Internal configuration struct
struct MockDeviceConfig {
    tt::ARCH arch;
    uint32_t num_chips;
};

// TODO: Remove this global once MetalContext can be initialized with a config object
// that includes mock device configuration. See issue #XXXXX for tracking.
static std::optional<MockDeviceConfig> g_registered_mock_config = std::nullopt;

void configure_mock_mode(tt::ARCH arch, uint32_t num_chips) {
    g_registered_mock_config = MockDeviceConfig{arch, num_chips};
    log_info(tt::LogMetal, "Mock mode configured: arch={}, num_chips={}", static_cast<int>(arch), num_chips);
}

void configure_mock_mode_from_hw() {
    tt::ARCH arch = get_physical_architecture();
    TT_FATAL(arch != tt::ARCH::Invalid, "No TT hardware detected - cannot auto-detect architecture for mock mode");
    configure_mock_mode(arch, 1);
}

void disable_mock_mode() {
    g_registered_mock_config = std::nullopt;
    log_info(tt::LogMetal, "Mock mode disabled");
}

bool is_mock_mode_registered() {
    return g_registered_mock_config.has_value();
}

std::optional<std::string> get_mock_cluster_desc() {
    if (!g_registered_mock_config.has_value()) {
        return std::nullopt;
    }

    const auto& config = *g_registered_mock_config;

    if (config.arch == tt::ARCH::WORMHOLE_B0) {
        switch (config.num_chips) {
            case 1: return "wormhole_N150.yaml";
            case 2: return "wormhole_N300.yaml";
            case 4: return "2x2_n300_cluster_desc.yaml";
            case 8: return "t3k_cluster_desc.yaml";
            case 32: return "tg_cluster_desc.yaml";
            default: break;
        }
    } else if (config.arch == tt::ARCH::BLACKHOLE) {
        switch (config.num_chips) {
            case 1: return "blackhole_P150.yaml";
            case 2: return "blackhole_P300_both_mmio.yaml";
            default: break;
        }
    }

    TT_THROW(
        "Unsupported mock device configuration: arch={}, num_chips={}",
        static_cast<int>(config.arch),
        config.num_chips);
}

}  // namespace tt::tt_metal::experimental
