// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/device/mock_device_util.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

TEST(MetalEnv, Init) {
    // If anywhere along the initialization path calls MetalContext::instance() then
    // this will hang because we will try to create 2 umd clusters
    // All child objects of the MetalEnv must take in dependencies (Hal, RuntimeOptions, Cluster, etc.)
    // explicitly rather than calling MetalContext::instance() (being deprecated)
    auto env = MetalEnv();
}

TEST(MetalEnv, Mock) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    MetalEnvDescriptor settings(mock_path);
    MetalEnv env_1(settings);
    MetalEnv env_2(settings);
    EXPECT_EQ(env_1.get_arch(), tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(env_2.get_arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetalEnv, OnePhysicalMultipleMock) {
    auto env_physical = MetalEnv();

    auto mock_path_1 = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value();
    MetalEnv env_mock_1{MetalEnvDescriptor(mock_path_1)};

    auto mock_path_2 = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2).value();
    MetalEnv env_mock_2{MetalEnvDescriptor(mock_path_2)};

    EXPECT_EQ(MetalEnvAccessor(env_mock_1).impl().get_cluster().arch(), tt::ARCH::BLACKHOLE);
    EXPECT_EQ(MetalEnvAccessor(env_mock_2).impl().get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetalEnv, EnvQueries) {
    MetalEnv env;
    EXPECT_NO_THROW(env.get_arch());
    EXPECT_NO_THROW(env.get_arch_name());
    EXPECT_NO_THROW(env.get_num_pcie_devices());
    EXPECT_NO_THROW(env.get_l1_size());
    EXPECT_NO_THROW(env.get_dram_alignment());
    EXPECT_NO_THROW(env.get_l1_alignment());
    EXPECT_NO_THROW(env.get_num_dataflow_buffers());
    EXPECT_NO_THROW(env.get_max_worker_l1_unreserved_size());
    EXPECT_NO_THROW(env.get_eps());
    EXPECT_NO_THROW(env.get_nan());
    EXPECT_NO_THROW(env.get_inf());
}

TEST(MetalEnv, EnvQueriesMock) {
    auto env_settings = MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2));
    MetalEnv env(env_settings);
    EXPECT_EQ(env.get_arch(), tt::ARCH::BLACKHOLE);
    EXPECT_NO_THROW(env.get_l1_size());
    EXPECT_EQ(env.get_num_pcie_devices(), 2);
    EXPECT_NO_THROW(env.get_dram_alignment());
    EXPECT_NO_THROW(env.get_l1_alignment());
    EXPECT_NO_THROW(env.get_num_dataflow_buffers());
    EXPECT_NO_THROW(env.get_max_worker_l1_unreserved_size());
    EXPECT_NO_THROW(env.get_eps());
    EXPECT_NO_THROW(env.get_nan());
    EXPECT_NO_THROW(env.get_inf());
}

TEST(MetalEnv, ForkSafety) {
    MetalEnv env;

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";

    if (pid == 0) {
        // Child: the prefork prepare handler already ran in the parent before
        // the fork.  If it didn't crash/abort, the child just exits cleanly.
        _exit(0);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(MetalEnv, ForkSafetyMultipleEnvs) {
    MetalEnv env_physical;

    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value();
    MetalEnv env_mock{MetalEnvDescriptor(mock_path)};

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";

    if (pid == 0) {
        _exit(0);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(MetalEnv, ForkSafetyActiveEnv) {
    MetalEnv env;
    MetalEnvImpl& accessor = MetalEnvAccessor(env).impl();

    accessor.acquire();

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";

    if (pid == 0) {
        // Child inherits the nonzero use count from the parent.
        // check_use_count_zero() should return false.
        _exit(accessor.check_use_count_zero() ? 0 : 1);
    }

    accessor.release();

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 1);
}

// --- FabricConfigDescriptor tests ---

TEST(MetalEnv, FabricConfigDescriptorDefaults) {
    FabricConfigDescriptor desc;
    EXPECT_EQ(desc.fabric_config, tt_fabric::FabricConfig::DISABLED);
    EXPECT_EQ(desc.reliability_mode, tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    EXPECT_FALSE(desc.num_routing_planes.has_value());
    EXPECT_EQ(desc.fabric_tensix_config, tt_fabric::FabricTensixConfig::DISABLED);
    EXPECT_EQ(desc.fabric_udm_mode, tt_fabric::FabricUDMMode::DISABLED);
    EXPECT_EQ(desc.fabric_manager, tt_fabric::FabricManagerMode::DEFAULT);
}

TEST(MetalEnv, DescriptorWithFabricConfig) {
    FabricConfigDescriptor fc;
    fc.fabric_config = tt_fabric::FabricConfig::FABRIC_1D;
    fc.reliability_mode = tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
    fc.num_routing_planes = 4;
    fc.fabric_tensix_config = tt_fabric::FabricTensixConfig::MUX;

    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1);
    MetalEnvDescriptor settings(mock_path, fc);

    EXPECT_TRUE(settings.is_mock_device());
    const auto& stored = settings.fabric_config_descriptor();
    EXPECT_EQ(stored.fabric_config, tt_fabric::FabricConfig::FABRIC_1D);
    EXPECT_EQ(stored.reliability_mode, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    EXPECT_TRUE(stored.num_routing_planes.has_value());
    EXPECT_EQ(stored.num_routing_planes.value(), 4);
    EXPECT_EQ(stored.fabric_tensix_config, tt_fabric::FabricTensixConfig::MUX);
}

TEST(MetalEnv, DefaultDescriptorFabricConfigIsDisabled) {
    MetalEnvDescriptor settings;
    EXPECT_EQ(settings.fabric_config_descriptor().fabric_config, tt_fabric::FabricConfig::DISABLED);
}

TEST(MetalEnv, FabricConfigPreservedThroughEnv) {
    FabricConfigDescriptor fc;
    fc.fabric_config = tt_fabric::FabricConfig::FABRIC_2D;

    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2);
    MetalEnvDescriptor settings(mock_path, fc);
    MetalEnv env(settings);

    const auto& stored = env.get_descriptor().fabric_config_descriptor();
    EXPECT_EQ(stored.fabric_config, tt_fabric::FabricConfig::FABRIC_2D);
}

TEST(MetalEnv, AccessorFabricConfigMatchesDescriptor) {
    FabricConfigDescriptor fc;
    fc.fabric_config = tt_fabric::FabricConfig::FABRIC_1D_RING;
    fc.fabric_udm_mode = tt_fabric::FabricUDMMode::ENABLED;

    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1);
    MetalEnvDescriptor settings(mock_path, fc);
    MetalEnv env(settings);

    MetalEnvImpl& accessor = MetalEnvAccessor(env).impl();

    EXPECT_EQ(accessor.get_fabric_config(), tt_fabric::FabricConfig::FABRIC_1D_RING);
    EXPECT_EQ(accessor.get_fabric_udm_mode(), tt_fabric::FabricUDMMode::ENABLED);
}

// --- System mesh tests ---

TEST(MetalEnv, SystemMeshAccessibleOnMock) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1);
    MetalEnvDescriptor settings(mock_path);
    MetalEnv env(settings);

    auto& mesh = env.get_system_mesh();
    EXPECT_GT(mesh.shape().mesh_size(), 0);
}

TEST(MetalEnv, SystemMeshSameEnv) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2);
    MetalEnvDescriptor settings(mock_path);
    MetalEnv env(settings);

    auto& mesh1 = env.get_system_mesh();
    auto& mesh2 = env.get_system_mesh();
    EXPECT_EQ(&mesh1, &mesh2);
}

// The MetalEnv constructor are user settings.
// If fabric not enabled by the user, but it turns out we need dispatch on fabric, then the
// env needs to be reconfigured to enable fabric.
TEST(MetalEnv, ReconfigureFabricForDispatch) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2);
    MetalEnvDescriptor settings(mock_path);
    MetalEnv env(settings);

    // MetalEnv cannot be reconfigured after init so use the internal accessor
    MetalEnvImpl& accessor = MetalEnvAccessor(env).impl();

    EXPECT_EQ(accessor.get_fabric_config(), tt_fabric::FabricConfig::DISABLED);

    auto& cp_before = accessor.get_control_plane();
    EXPECT_EQ(cp_before.get_fabric_config(), tt_fabric::FabricConfig::DISABLED);

    auto& mesh_before = env.get_system_mesh();
    EXPECT_GT(mesh_before.shape().mesh_size(), 0);
    const auto* cp_ptr_before = &cp_before;

    accessor.set_fabric_config(
        tt_fabric::FabricConfig::FABRIC_1D, tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);

    EXPECT_EQ(accessor.get_fabric_config(), tt_fabric::FabricConfig::FABRIC_1D);

    auto& cp_after = accessor.get_control_plane();
    EXPECT_EQ(cp_after.get_fabric_config(), tt_fabric::FabricConfig::FABRIC_1D);
    EXPECT_NE(cp_ptr_before, &cp_after);

    auto& mesh_after = env.get_system_mesh();
    EXPECT_GT(mesh_after.shape().mesh_size(), 0);
}

// --- Context leaks on failed mesh device creation (#57286) ---

namespace {

// The lowest free slot is handed out first, so a leaked context shifts this upward.
ContextId next_free_context_id(const std::string& mock_path) {
    MetalEnv probe{MetalEnvDescriptor(mock_path)};
    ContextId id = MetalContext::create_instance(probe);
    MetalContext::destroy_instance(/*check_device_count=*/false, id);
    return id;
}

}  // namespace

TEST(MetalEnv, FailedCreateMeshDeviceDoesNotLeakContext) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    const ContextId baseline = next_free_context_id(mock_path);

    for (int i = 0; i < 3; ++i) {
        MetalEnv env{MetalEnvDescriptor(mock_path)};
        // Fatals inside MeshDeviceImpl::create, after the context exists and before ownership transfers.
        EXPECT_THROW(
            env.create_mesh_device(distributed::MeshDeviceConfig(
                /*mesh_shape=*/std::nullopt, /*offset=*/std::nullopt, /*physical_device_ids=*/{0})),
            std::runtime_error);
    }

    EXPECT_EQ(next_free_context_id(mock_path).get(), baseline.get());
}

TEST(MetalEnv, FailedCreateUnitMeshDeviceDoesNotLeakContext) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    const ContextId baseline = next_free_context_id(mock_path);

    for (int i = 0; i < 3; ++i) {
        MetalEnv env{MetalEnvDescriptor(mock_path)};
        // Device 99 is not in the 1-chip mock cluster, so opening it fails.
        EXPECT_THROW(env.create_unit_mesh_device(99), std::runtime_error);
    }

    EXPECT_EQ(next_free_context_id(mock_path).get(), baseline.get());
}

TEST(MetalEnv, FailedCreateUnitMeshesDoesNotLeakContext) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    const ContextId baseline = next_free_context_id(mock_path);

    for (int i = 0; i < 3; ++i) {
        MetalEnv env{MetalEnvDescriptor(mock_path)};
        EXPECT_THROW(env.create_unit_meshes({}), std::runtime_error);
    }

    EXPECT_EQ(next_free_context_id(mock_path).get(), baseline.get());
}

}  // namespace tt::tt_metal
