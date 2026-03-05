// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

// Prefer to use API rather than internals in this
// test as we are testing end to end functionality
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/context/metalium_env.hpp>
#include <tt-metalium/experimental/hal.hpp>
#include <tt-metalium/experimental/tt_metal.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include <umd/device/types/arch.hpp>
#include "device/mock_device_util.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

TEST(MetalContextIntegrationTest, HelloWorld) {
    auto mesh_shape = tt_metal::distributed::SystemMesh::instance().shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create(mesh_device_config);
    EXPECT_EQ(mesh_device->num_devices(), mesh_shape.mesh_size());
    // Required for this unit test because legacy behaviour of MetalContext is not to close the cluster until atexit
    // Close it right now so remaining tests can proceed
    mesh_device->close();
    tt::tt_metal::experimental::DestroyAllContexts();

    // It was found that during ~MeshDevice, some calls to MetalContext::instance() were made which caused
    // MetalContext to implicitly reinitialize thus undoing the effects of DestroyAllContexts().
    // Subsequent tests will hang if that happens.
}

TEST(MetalContextIntegrationTest, HalFunctions) {
    // Create a MetaliumEnv and query hardware state
    MetaliumEnv env;
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_arch(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_arch_name(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_dram_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_pcie_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_max_worker_l1_unreserved_size(env));
}

TEST(MetalContextIntegrationTest, HalFunctionsWithMock) {
    auto env_settings = MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2));
    MetaliumEnv env(env_settings);
    EXPECT_EQ(tt::tt_metal::experimental::hal::get_arch(env), tt::ARCH::BLACKHOLE);
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_dram_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_l1_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_pcie_alignment(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_size(env));
    EXPECT_NO_THROW(tt::tt_metal::experimental::hal::get_max_worker_l1_unreserved_size(env));
}

TEST(MetalContextIntegrationTest, MockDevice) {
    {
        auto mock_env_bh_1 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)));
        auto mock_context_id = tt::tt_metal::experimental::CreateContext(mock_env_bh_1);
        EXPECT_NE(mock_context_id, SILICON_CONTEXT_ID);

        auto mesh_config_mock = distributed::MeshDeviceConfig(distributed::MeshShape(2));
        auto mock_device = distributed::MeshDevice::create(mock_context_id, mesh_config_mock);

        auto env = std::make_shared<MetaliumEnv>();
        auto context_id = tt::tt_metal::experimental::CreateContext(env);
        auto mesh_shape = tt::tt_metal::MetalContext::instance(context_id).get_system_mesh().shape();
        auto mesh_config_silicon = distributed::MeshDeviceConfig(mesh_shape);
        auto silicon_device = distributed::MeshDevice::create(context_id, mesh_config_silicon);
    }

    tt::tt_metal::experimental::DestroyAllContexts();
}

TEST(MetalContextIntegrationTest, CoexistingSiliconAndMockDevice) {
    {
        // Create mock mesh device with 1 blackhole chip
        auto mock_env_bh_1 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1)));
        auto mock_context_id_bh_1 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_1);
        log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_1);
        EXPECT_NE(mock_context_id_bh_1, SILICON_CONTEXT_ID);

        auto mock_mesh_shape_bh_1 =
            tt::tt_metal::MetalContext::instance(mock_context_id_bh_1).get_system_mesh().shape();
        auto mock_mesh_device_config_bh_1 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_1);
        std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_1 =
            distributed::MeshDevice::create(mock_context_id_bh_1, mock_mesh_device_config_bh_1);
        log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_1->shape().dims());

        // Create mock mesh device with 2 blackhole chips
        auto mock_env_bh_2 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)));
        auto mock_context_id_bh_2 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_2);
        log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_2);
        EXPECT_NE(mock_context_id_bh_2, SILICON_CONTEXT_ID);

        auto mock_mesh_shape_bh_2 =
            tt::tt_metal::MetalContext::instance(mock_context_id_bh_2).get_system_mesh().shape();
        auto mock_mesh_device_config_bh_2 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_2);
        std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_2 =
            distributed::MeshDevice::create(mock_context_id_bh_2, mock_mesh_device_config_bh_2);
        log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_2->shape().dims());

        // Create silicon mesh
        auto silicon_env = std::make_shared<MetaliumEnv>();
        auto silicon_context_id = tt::tt_metal::MetalContext::create_instance(silicon_env);
        log_info(tt::LogTest, "MetaliumEnv (silicon) created with context id {}", silicon_context_id);
        EXPECT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

        auto mesh_shape = tt::tt_metal::MetalContext::instance(silicon_context_id).get_system_mesh().shape();
        auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
        std::shared_ptr<distributed::MeshDevice> mesh_device =
            distributed::MeshDevice::create(silicon_context_id, mesh_device_config);
        log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

        EXPECT_NE(mock_context_id_bh_1, mock_context_id_bh_2);
        ASSERT_EQ(mock_mesh_device_bh_1->get_devices().size(), 1);
        ASSERT_EQ(mock_mesh_device_bh_2->get_devices().size(), 2);
    }

    tt::tt_metal::experimental::DestroyAllContexts();
}

// Same test as above but reverse the order to ensure no hangs due to unexpected internal objects created for the
// incorrect context id
TEST(MetalContextIntegrationTest, CoexistingMockAndSiliconDevice) {
    {
        // Create silicon mesh
        auto silicon_env = std::make_shared<MetaliumEnv>();
        auto silicon_context_id = tt::tt_metal::MetalContext::create_instance(silicon_env);
        log_info(tt::LogTest, "MetaliumEnv (silicon) created with context id {}", silicon_context_id);
        EXPECT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

        auto mesh_shape = tt::tt_metal::MetalContext::instance(silicon_context_id).get_system_mesh().shape();
        auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
        std::shared_ptr<distributed::MeshDevice> mesh_device =
            distributed::MeshDevice::create(silicon_context_id, mesh_device_config);
        log_info(tt::LogTest, "Created silicon mesh device with shape {}", mesh_device->shape().dims());

        // Create mock mesh device with 1 blackhole chip
        auto mock_env_bh_1 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1)));
        auto mock_context_id_bh_1 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_1);
        log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_1);
        EXPECT_NE(mock_context_id_bh_1, SILICON_CONTEXT_ID);

        auto mock_mesh_shape_bh_1 =
            tt::tt_metal::MetalContext::instance(mock_context_id_bh_1).get_system_mesh().shape();
        auto mock_mesh_device_config_bh_1 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_1);
        std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_1 =
            distributed::MeshDevice::create(mock_context_id_bh_1, mock_mesh_device_config_bh_1);
        log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_1->shape().dims());

        // Create mock mesh device with 2 blackhole chips
        auto mock_env_bh_2 = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)));
        auto mock_context_id_bh_2 = tt::tt_metal::MetalContext::create_instance(mock_env_bh_2);
        log_info(tt::LogTest, "MetaliumEnv (mock) created with context id {}", mock_context_id_bh_2);
        EXPECT_NE(mock_context_id_bh_2, SILICON_CONTEXT_ID);

        auto mock_mesh_shape_bh_2 =
            tt::tt_metal::MetalContext::instance(mock_context_id_bh_2).get_system_mesh().shape();
        auto mock_mesh_device_config_bh_2 = distributed::MeshDeviceConfig(mock_mesh_shape_bh_2);
        std::shared_ptr<distributed::MeshDevice> mock_mesh_device_bh_2 =
            distributed::MeshDevice::create(mock_context_id_bh_2, mock_mesh_device_config_bh_2);
        log_info(tt::LogTest, "Created mock mesh device with shape {}", mock_mesh_device_bh_2->shape().dims());

        EXPECT_NE(mock_context_id_bh_1, mock_context_id_bh_2);
        ASSERT_EQ(mock_mesh_device_bh_1->get_devices().size(), 1);
        ASSERT_EQ(mock_mesh_device_bh_2->get_devices().size(), 2);
    }

    tt::tt_metal::experimental::DestroyAllContexts();
}

}  // namespace tt::tt_metal
