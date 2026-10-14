// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Offline checks of harness/targets against the Blackhole SoC descriptor.

#include "harness/targets/drisc_endpoint.hpp"

#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/soc_arch_descriptor.hpp>
#include <umd/device/soc_descriptor.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ttm = tt::tt_metal;

namespace {

// The Blackhole P150 descriptor tt-metal itself loads, built from the YAML
// under TT_METAL_DIR so the test needs no device.
metal_SocDescriptor load_blackhole_soc_descriptor() {
    const char* env_metal = std::getenv("TT_METAL_DIR");
    if (env_metal == nullptr || env_metal[0] == '\0') {
        throw std::runtime_error("load_blackhole_soc_descriptor requires non-empty TT_METAL_DIR environment variable");
    }
    const std::filesystem::path yaml_path =
        std::filesystem::path(env_metal) / "tt_metal/soc_descriptors/blackhole_140_arch.yaml";
    if (!std::filesystem::exists(yaml_path)) {
        throw std::runtime_error(
            "load_blackhole_soc_descriptor: SoC descriptor YAML not found at: " + yaml_path.string());
    }

    auto arch_desc = std::make_shared<tt::umd::SocArchDescriptor>(tt::ARCH::BLACKHOLE);
    tt::ChipInfo chip_info{};
    chip_info.board_type = tt::BoardType::P150;
    chip_info.harvesting_masks.pcie_harvesting_mask = 0x2;
    chip_info.harvesting_masks.eth_harvesting_mask = 0x3;
    tt::umd::SocDescriptor umd_soc(arch_desc, chip_info);
    umd_soc.device_descriptor_file_path = yaml_path.string();
    return metal_SocDescriptor(umd_soc, tt::BoardType::P150);
}

TEST(MetalTargetOffline, SelectDriscEndpointExcludesFirmwareNoc0Endpoint) {
    const metal_SocDescriptor soc_desc = load_blackhole_soc_descriptor();
    const auto& endpoints = soc_desc.dram_bank_endpoint_coords.at(0);
    const std::vector<ttm::CoreCoord> launchable = soc_desc.get_metal_dram_cores(tt::CoordSystem::TRANSLATED);
    const auto launchable_has = [&](const ttm::CoreCoord& c) {
        return std::find(launchable.begin(), launchable.end(), c) != launchable.end();
    };
    ASSERT_EQ(endpoints.at(0), soc_desc.get_preferred_worker_core_for_dram_view(0, /*noc=*/0))
        << "index 0 of dram_bank_endpoint_coords is the NOC0 worker endpoint";
    ASSERT_FALSE(launchable_has(endpoints.at(0)))
        << "tt-metal reserves the NOC0 worker endpoint for the syseng firmware";

    const auto logical = x280::harness::targets::select_drisc_endpoint(soc_desc);
    ASSERT_TRUE(logical.has_value());
    EXPECT_EQ(logical->x, 0u);
    EXPECT_NE(logical->y, 0u);
    ASSERT_LT(logical->y, endpoints.size());
    EXPECT_TRUE(launchable_has(endpoints[logical->y]));
    for (uint32_t i = 0; i < logical->y; ++i) {
        EXPECT_FALSE(launchable_has(endpoints[i])) << "endpoint " << i << " is launchable but was skipped";
    }
}

}  // namespace
