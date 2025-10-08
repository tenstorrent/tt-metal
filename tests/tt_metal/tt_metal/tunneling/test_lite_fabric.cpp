// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>

#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/fabric.hpp>

#include "context/metal_context.hpp"
#include "distributed.hpp"
#include "fabric_types.hpp"
#include "llrt/hal.hpp"
#include "tt_cluster.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

#include "tt_metal/lite_fabric/host_util.hpp"
#include "tt_metal/lite_fabric/build.hpp"

#define CHECK_TEST_REQS()                                                                       \
    if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) { \
        GTEST_SKIP() << "Blackhole only";                                                       \
    }                                                                                           \
    if (tt::tt_metal::GetNumAvailableDevices() < 2) {                                           \
        GTEST_SKIP() << "At least 2 Devices are required";                                      \
    }                                                                                           \

struct FabricLiteTestConfig {
    bool standalone{false};
    tt::tt_fabric::FabricConfig fabric_config{tt::tt_fabric::FabricConfig::DISABLED};
};

// Lite Fabric Test Fixture
class FabricLite : public testing::TestWithParam<FabricLiteTestConfig> {
protected:
    inline static lite_fabric::SystemDescriptor desc;

    // Instance variables instead of static ones for parameter-dependent resources
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    bool fabric_configured_{false};

    static void SetUpTestSuite() {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto& hal = tt::tt_metal::MetalContext::instance().hal();
        CHECK_TEST_REQS();

        desc = lite_fabric::GetSystemDescriptorFromMmio(cluster, 0);

        lite_fabric::LaunchLiteFabric(cluster, hal, desc);
    }

    static void TearDownTestSuite() {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

        lite_fabric::TerminateLiteFabric(cluster, desc);
    }

    void SetUp() override {
        CHECK_TEST_REQS();

        if (desc.tunnels_from_mmio.empty()) {
            GTEST_SKIP() << "No tunnels found";
        }

        // Configure fabric if needed
        if (GetParam().fabric_config != tt::tt_fabric::FabricConfig::DISABLED) {
            tt::tt_fabric::SetFabricConfig(GetParam().fabric_config);
            fabric_configured_ = true;
        }

        // Create mesh device if not standalone
        if (!GetParam().standalone) {
            if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
                GTEST_SKIP() << "Fast dispatch is required for this test (remove TT_METAL_SLOW_DISPATCH_MODE)";
            }

            auto number_of_devices = tt::tt_metal::GetNumAvailableDevices();
            mesh_device_ = tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(
                tt::tt_metal::distributed::MeshShape{number_of_devices, 1}));
        }
    }

    void TearDown() override {
        // Clean up mesh device
        if (mesh_device_) {
            mesh_device_.reset();
        }

        // Reset fabric configuration
        if (fabric_configured_) {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            fabric_configured_ = false;
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    FabricLiteFixture,
    FabricLite,
    ::testing::Values(
        // Standalone tests (no mesh device, no fabric)
        FabricLiteTestConfig{.standalone = true},
        // Standard tests with mesh device but no fabric
        FabricLiteTestConfig{.standalone = false},
        // Test with 1D fabric active (full fabric)
        FabricLiteTestConfig{.standalone = false, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D},
        // Test with 2D fabric active (full fabric)
        FabricLiteTestConfig{.standalone = false, .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D}),
    [](const testing::TestParamInfo<FabricLiteTestConfig>& info) {
        std::string name;
        if (info.param.standalone) {
            name = "Standalone";
        } else if (info.param.fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
            name = "MeshDevice";
        } else {
            name = "MeshDevice_";
            name += enchantum::to_string(info.param.fabric_config);
        }
        return name;
    });

TEST(FabricLiteBuild, BuildOnly) {
    auto home_dir_string = std::getenv("TT_METAL_HOME");
    if (home_dir_string == nullptr) {
        GTEST_FAIL() << "TT_METAL_HOME not set";
    }
    auto home_directory = std::filesystem::path(std::getenv("TT_METAL_HOME"));
    auto output_directory = home_directory / "lite_fabric";
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (lite_fabric::CompileFabricLite(cluster, home_directory, output_directory)) {
        throw std::runtime_error("Failed to compile");
    }
    if (lite_fabric::LinkFabricLite(home_directory, output_directory, output_directory / "lite_fabric.elf")) {
        throw std::runtime_error("Failed to link");
    }
}

TEST_P(FabricLite, Init) { EXPECT_GT(desc.tunnels_from_mmio.size(), 0) << "No tunnels found"; }
