// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/mock_device.hpp>

#include "mock_device_common.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <unordered_map>
#include "llrt/get_platform_architecture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal.hpp"

namespace tt::tt_metal::experimental {

struct MockDeviceConfig {
    tt::ARCH arch;
    uint32_t num_chips;
};

// TODO: Remove this global once MetalContext can be initialized with a config object
// that includes mock device configuration.
static std::optional<MockDeviceConfig> g_registered_mock_config = std::nullopt;

void configure_mock_mode(tt::ARCH arch, uint32_t num_chips) {
    g_registered_mock_config = MockDeviceConfig{arch, num_chips};
    log_info(tt::LogMetal, "Mock mode configured: arch={}, num_chips={}", static_cast<int>(arch), num_chips);
    tt::tt_metal::detail::ReleaseOwnership();
}

void configure_mock_mode_from_hw() {
    tt::ARCH arch = get_physical_architecture();
    TT_FATAL(arch != tt::ARCH::Invalid, "No TT hardware detected - cannot auto-detect architecture for mock mode");
    configure_mock_mode(arch, 1);
}

void disable_mock_mode() {
    if (!g_registered_mock_config.has_value()) {
        return;
    }

    g_registered_mock_config = std::nullopt;
    tt::tt_metal::detail::ReleaseOwnership();
}

bool is_mock_mode_registered() {
    return g_registered_mock_config.has_value();
}

const std::unordered_map<tt::ARCH, std::unordered_map<uint32_t, std::string>>& get_mock_cluster_config_map() {
    static const std::unordered_map<tt::ARCH, std::unordered_map<uint32_t, std::string>> cluster_configs = {
        {tt::ARCH::WORMHOLE_B0,
         {{1, "wormhole_N150.yaml"},
          {2, "wormhole_N300.yaml"},
          {4, "2x2_n300_cluster_desc.yaml"},
          {8, "t3k_cluster_desc.yaml"},
          {32, "tg_cluster_desc.yaml"}}},
        {tt::ARCH::BLACKHOLE, {{1, "blackhole_P150.yaml"}, {2, "blackhole_P300_both_mmio.yaml"}}}};
    return cluster_configs;
}

std::optional<std::string> get_mock_cluster_desc_for_config(tt::ARCH arch, uint32_t num_chips) {
    const auto& cluster_configs = get_mock_cluster_config_map();
    auto arch_it = cluster_configs.find(arch);
    if (arch_it != cluster_configs.end()) {
        auto chip_it = arch_it->second.find(num_chips);
        if (chip_it != arch_it->second.end()) {
            return std::string(chip_it->second);
        }
    }
    return std::nullopt;
}

std::optional<std::string> get_mock_cluster_desc() {
    if (!g_registered_mock_config.has_value()) {
        return std::nullopt;
    }

    const auto& config = *g_registered_mock_config;
    auto path = get_mock_cluster_desc_for_config(config.arch, config.num_chips);
    if (!path.has_value()) {
        TT_THROW(
            "Unsupported mock device configuration: arch={}, num_chips={}",
            static_cast<int>(config.arch),
            config.num_chips);
    }
    return path;
}

}  // namespace tt::tt_metal::experimental
