// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

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

TEST(MetalContextIntegrationTest, HelloWorldExplicit) {
    auto env = std::make_shared<MetaliumEnv>();
    int context_id = tt::tt_metal::experimental::CreateContext(env);

    auto mesh_shape = tt_metal::distributed::SystemMesh::instance(context_id).shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    auto mesh_device = distributed::MeshDevice::create(context_id, mesh_device_config);

    mesh_device->close();
    tt::tt_metal::experimental::DestroyAllContexts();
}

TEST(MetalContextIntegrationTest, HelloWorldQueryThenCreate) {
    auto env = std::make_shared<MetaliumEnv>();
    int context_id = tt::tt_metal::experimental::CreateContext(env);

    size_t l1_size = tt::tt_metal::experimental::hal::get_l1_size(*env);
    size_t trace_region_size = l1_size * 0.3;
    size_t l1_small_region_size = l1_size * 0.05;

    auto mesh_shape = tt_metal::distributed::SystemMesh::instance(context_id).shape();
    auto mesh_device_config = distributed::MeshDeviceConfig(mesh_shape);
    auto mesh_device =
        distributed::MeshDevice::create(context_id, mesh_device_config, trace_region_size, l1_small_region_size);

    mesh_device->close();
    tt::tt_metal::experimental::DestroyAllContexts();
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

TEST(MetalContextIntegrationTest, ForkMockAndRealDevice) {
    // Query hardware state before forking
    {
        auto env = std::make_shared<MetaliumEnv>();
        auto silicon_context_id = tt::tt_metal::experimental::CreateContext(env);
        EXPECT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

        auto arch = tt::tt_metal::experimental::hal::get_arch(*env);
        auto l1_size = tt::tt_metal::experimental::hal::get_l1_size(*env);
        auto num_devices = tt::tt_metal::experimental::GetNumAvailableDevices(env);
        log_info(tt::LogTest, "Pre-fork: arch={}, L1 size={}, num_devices={}", arch, l1_size, num_devices);
        EXPECT_GT(l1_size, 0u);
        EXPECT_GT(num_devices, 0u);
    }

    // Tear down all state so we can safely fork
    tt::tt_metal::experimental::DestroyAllContexts();

    int pipe_fd[2];
    ASSERT_EQ(pipe(pipe_fd), 0) << "pipe() failed";

    pid_t pid = fork();
    if (pid == -1) {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        FAIL() << "Failed to fork";
    }

    if (pid == 0) {
        // Child: create and verify a mock mesh device
        close(pipe_fd[1]);

        auto mock_env = std::make_shared<MetaliumEnv>(
            MetaliumEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value()));
        int mock_context_id = MetalContext::create_instance(mock_env);

        if (mock_context_id < 1 || !MetalContext::instance(mock_context_id).rtoptions().get_mock_enabled()) {
            _exit(1);
        }

        if (tt::tt_metal::experimental::hal::get_arch(*mock_env) != tt::ARCH::BLACKHOLE) {
            _exit(2);
        }

        auto mock_mesh_shape = MetalContext::instance(mock_context_id).get_system_mesh().shape();
        auto mock_mesh_device =
            distributed::MeshDevice::create(mock_context_id, distributed::MeshDeviceConfig(mock_mesh_shape));
        if (mock_mesh_device->get_devices().size() != 2) {
            _exit(3);
        }

        char byte = 0;
        if (read(pipe_fd[0], &byte, 1) != 1) {
            _exit(4);
        }
        close(pipe_fd[0]);

        _exit(0);
    }

    // Parent: real device work
    close(pipe_fd[0]);

    auto silicon_env = std::make_shared<MetaliumEnv>();
    auto silicon_context_id = tt::tt_metal::experimental::CreateContext(silicon_env);
    ASSERT_EQ(silicon_context_id, SILICON_CONTEXT_ID);

    auto real_arch = tt::tt_metal::experimental::hal::get_arch(*silicon_env);
    auto real_l1_size = tt::tt_metal::experimental::hal::get_l1_size(*silicon_env);
    log_info(tt::LogTest, "Parent (real): arch={}, L1 size={}", real_arch, real_l1_size);
    EXPECT_GT(real_l1_size, 0u);

    auto mesh_shape = MetalContext::instance(silicon_context_id).get_system_mesh().shape();
    auto silicon_mesh_device =
        distributed::MeshDevice::create(silicon_context_id, distributed::MeshDeviceConfig(mesh_shape));
    EXPECT_GT(silicon_mesh_device->num_devices(), 0u);
    log_info(tt::LogTest, "Parent: created silicon mesh device with shape {}", silicon_mesh_device->shape().dims());

    // Signal child to exit
    char byte = 1;
    ASSERT_EQ(write(pipe_fd[1], &byte, 1), 1) << "write(pipe) failed";
    close(pipe_fd[1]);

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    silicon_mesh_device.reset();
    tt::tt_metal::experimental::DestroyAllContexts();
}

}  // namespace tt::tt_metal
