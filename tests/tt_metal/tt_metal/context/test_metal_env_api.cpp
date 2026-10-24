// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

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
    const MetalEnvDescriptor settings{
        .mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)};
    MetalEnv env_1(settings);
    MetalEnv env_2(settings);
    EXPECT_EQ(env_1.get_device_properties().arch(), tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(env_2.get_device_properties().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetalEnv, OnePhysicalMultipleMock) {
    auto env_physical = MetalEnv();

    MetalEnv env_mock_1({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)});
    MetalEnv env_mock_2({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});

    EXPECT_EQ(MetalEnvAccessor(env_mock_1).impl().get_cluster().arch(), tt::ARCH::BLACKHOLE);
    EXPECT_EQ(MetalEnvAccessor(env_mock_2).impl().get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetalEnv, EnvQueries) {
    MetalEnv env;
    EXPECT_NO_THROW(env.get_num_pcie_devices());
    const auto props = env.get_device_properties();
    EXPECT_NO_THROW(props.arch());
    EXPECT_NO_THROW(props.arch_name());
    EXPECT_NO_THROW(props.node_local_mem_size());
    EXPECT_NO_THROW(props.dram_alignment());
    EXPECT_NO_THROW(props.node_local_mem_alignment());
    EXPECT_NO_THROW(props.num_dataflow_buffers());
    EXPECT_NO_THROW(props.max_worker_node_local_mem_unreserved_size());
    EXPECT_NO_THROW(props.sfpu_eps());
    EXPECT_NO_THROW(props.sfpu_nan());
    EXPECT_NO_THROW(props.sfpu_inf());
}

TEST(MetalEnv, EnvQueriesMock) {
    MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)});
    EXPECT_EQ(env.get_num_pcie_devices(), 2);
    const auto props = env.get_device_properties();
    EXPECT_EQ(props.arch(), tt::ARCH::BLACKHOLE);
    EXPECT_NO_THROW(props.node_local_mem_size());
    EXPECT_NO_THROW(props.dram_alignment());
    EXPECT_NO_THROW(props.node_local_mem_alignment());
    EXPECT_NO_THROW(props.num_dataflow_buffers());
    EXPECT_NO_THROW(props.max_worker_node_local_mem_unreserved_size());
    EXPECT_NO_THROW(props.sfpu_eps());
    EXPECT_NO_THROW(props.sfpu_nan());
    EXPECT_NO_THROW(props.sfpu_inf());
}

TEST(MetalEnv, DevicePropertiesCopyAndAssign) {
    MetalEnv env_wh({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)});
    MetalEnv env_bh({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)});

    // The handle is a regular value type: copyable and reassignable to a view of another env.
    MetalDeviceProperties props = env_wh.get_device_properties();
    const MetalDeviceProperties props_copy = props;
    EXPECT_EQ(props_copy.arch(), tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(props_copy.node_local_mem_alignment(), env_wh.get_device_properties().node_local_mem_alignment());

    props = env_bh.get_device_properties();
    EXPECT_EQ(props.arch(), tt::ARCH::BLACKHOLE);
    EXPECT_EQ(props.arch_name(), "blackhole");
    EXPECT_EQ(props.node_local_mem_size(), env_bh.get_device_properties().node_local_mem_size());

    // The earlier copy still refers to the first env.
    EXPECT_EQ(props_copy.arch(), tt::ARCH::WORMHOLE_B0);
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

    MetalEnv env_mock({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)});

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

// --- MetalEnvDescriptor tests ---

TEST(MetalEnv, DefaultDescriptorTargetsSilicon) {
    const MetalEnvDescriptor settings;
    EXPECT_FALSE(settings.is_mock_device());
    EXPECT_FALSE(settings.mock_cluster_desc_path.has_value());
}

TEST(MetalEnv, DescriptorMockByPath) {
    const MetalEnvDescriptor settings{.mock_cluster_desc_path = "wormhole_N150.yaml"};
    EXPECT_TRUE(settings.is_mock_device());
    EXPECT_EQ(settings.mock_cluster_desc_path, "wormhole_N150.yaml");
}

TEST(MetalEnv, EmptyPathIsNotMock) {
    const MetalEnvDescriptor settings{.mock_cluster_desc_path = ""};
    EXPECT_FALSE(settings.is_mock_device());
}

// --- Fabric configuration tests ---

TEST(MetalEnv, FabricConfigDescriptorDefaults) {
    FabricConfigDescriptor desc;
    EXPECT_EQ(desc.fabric_config, tt_fabric::FabricConfig::DISABLED);
    EXPECT_EQ(desc.reliability_mode, tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    EXPECT_FALSE(desc.num_routing_planes.has_value());
    EXPECT_EQ(desc.fabric_tensix_config, tt_fabric::FabricTensixConfig::DISABLED);
    EXPECT_EQ(desc.fabric_udm_mode, tt_fabric::FabricUDMMode::DISABLED);
    EXPECT_EQ(desc.fabric_manager, tt_fabric::FabricManagerMode::DEFAULT);
}

MetalEnvDescriptor wormhole_mock_descriptor() {
    return {.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)};
}

TEST(MetalEnv, ConfigureFabricStoresDescriptor) {
    FabricConfigDescriptor fc;
    fc.fabric_config = tt_fabric::FabricConfig::FABRIC_1D;
    fc.reliability_mode = tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
    fc.num_routing_planes = 4;
    fc.fabric_tensix_config = tt_fabric::FabricTensixConfig::MUX;

    MetalEnv env(wormhole_mock_descriptor());
    env.configure_fabric(fc);

    const auto& stored = env.get_fabric_config_descriptor();
    EXPECT_EQ(stored.fabric_config, tt_fabric::FabricConfig::FABRIC_1D);
    EXPECT_EQ(stored.reliability_mode, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    EXPECT_TRUE(stored.num_routing_planes.has_value());
    EXPECT_EQ(stored.num_routing_planes.value(), 4);
    EXPECT_EQ(stored.fabric_tensix_config, tt_fabric::FabricTensixConfig::MUX);
}

TEST(MetalEnv, DefaultFabricConfigIsDisabled) {
    MetalEnv env(wormhole_mock_descriptor());
    EXPECT_EQ(env.get_fabric_config_descriptor().fabric_config, tt_fabric::FabricConfig::DISABLED);
}

TEST(MetalEnv, ConfigureFabricLastCallWins) {
    MetalEnv env(wormhole_mock_descriptor());
    env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D});
    env.configure_fabric(
        {.fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
         .fabric_tensix_config = tt_fabric::FabricTensixConfig::MUX});

    EXPECT_EQ(env.get_fabric_config_descriptor().fabric_config, tt_fabric::FabricConfig::FABRIC_2D);
    EXPECT_EQ(MetalEnvAccessor(env).impl().get_fabric_config(), tt_fabric::FabricConfig::FABRIC_2D);
    EXPECT_EQ(MetalEnvAccessor(env).impl().get_fabric_tensix_config(), tt_fabric::FabricTensixConfig::MUX);
}

TEST(MetalEnv, ConfigureFabricNormalizesRoutingPlanes) {
    MetalEnv env(wormhole_mock_descriptor());
    env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D});

    EXPECT_FALSE(env.get_fabric_config_descriptor().num_routing_planes.has_value());
    EXPECT_EQ(MetalEnvAccessor(env).impl().get_num_fabric_active_routing_planes(), std::numeric_limits<uint8_t>::max());

    env.configure_fabric({});
    EXPECT_EQ(MetalEnvAccessor(env).impl().get_num_fabric_active_routing_planes(), 0);
}

TEST(MetalEnv, ConfigureFabricRejectsZeroRoutingPlanes) {
    MetalEnv env(wormhole_mock_descriptor());
    EXPECT_THROW(
        env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D, .num_routing_planes = 0}),
        std::runtime_error);
}

TEST(MetalEnv, ConfigureFabricThrowsAfterSystemMesh) {
    MetalEnv env(wormhole_mock_descriptor());
    (void)env.get_system_mesh();
    EXPECT_THROW(env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D}), std::runtime_error);
}

TEST(MetalEnv, ConfigureFabricThrowsAfterCreateMeshDevice) {
    MetalEnv env(wormhole_mock_descriptor());
    auto mesh =
        distributed::MeshDevice::open_mesh_device(env, distributed::MeshDeviceConfig(distributed::MeshShape{1u, 1u}));
    EXPECT_THROW(env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D}), std::runtime_error);
    mesh->close();
    EXPECT_THROW(env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D}), std::runtime_error);
}

TEST(MetalEnv, QueryThenConfigureFabric) {
    MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
    EXPECT_GT(env.get_num_available_devices(), 0u);

    env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D});
    EXPECT_EQ(env.get_fabric_config_descriptor().fabric_config, tt_fabric::FabricConfig::FABRIC_1D);
}

TEST(MetalEnv, AccessorFabricConfigMatchesRequest) {
    MetalEnv env(wormhole_mock_descriptor());
    env.configure_fabric(
        {.fabric_config = tt_fabric::FabricConfig::FABRIC_1D_RING,
         .fabric_udm_mode = tt_fabric::FabricUDMMode::ENABLED});

    MetalEnvImpl& accessor = MetalEnvAccessor(env).impl();
    EXPECT_EQ(accessor.get_fabric_config(), tt_fabric::FabricConfig::FABRIC_1D_RING);
    EXPECT_EQ(accessor.get_fabric_udm_mode(), tt_fabric::FabricUDMMode::ENABLED);
}

// --- System mesh tests ---

TEST(MetalEnv, SystemMeshAccessibleOnMock) {
    MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)});

    auto& mesh = env.get_system_mesh();
    EXPECT_GT(mesh.shape().mesh_size(), 0);
}

TEST(MetalEnv, SystemMeshSameEnv) {
    MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2)});

    auto& mesh1 = env.get_system_mesh();
    auto& mesh2 = env.get_system_mesh();
    EXPECT_EQ(&mesh1, &mesh2);
}

// If the user left fabric disabled but dispatch to a remote device needs it, the runtime enables FABRIC_1D
// without replacing a system mesh that was already handed out.
TEST(MetalEnv, DispatchFabricKeepsSystemMesh) {
    MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)});
    MetalEnvImpl& accessor = MetalEnvAccessor(env).impl();

    EXPECT_EQ(accessor.get_fabric_config(), tt_fabric::FabricConfig::DISABLED);

    // Register this env's context before the control plane is built. Building it first leaves
    // MetalContext::instance() with nothing to find, and it opens the physical cluster.
    auto& mesh_before = env.get_system_mesh();
    auto& cp_before = accessor.get_control_plane();
    EXPECT_EQ(cp_before.get_fabric_config(), tt_fabric::FabricConfig::DISABLED);
    EXPECT_GT(mesh_before.shape().mesh_size(), 0);

    accessor.enable_fabric_for_dispatch();

    EXPECT_EQ(accessor.get_fabric_config(), tt_fabric::FabricConfig::FABRIC_1D);
    // The request the user made (nothing) is unchanged; only the runtime fabric config moved.
    EXPECT_EQ(env.get_fabric_config_descriptor().fabric_config, tt_fabric::FabricConfig::DISABLED);

    auto& cp_after = accessor.get_control_plane();
    EXPECT_EQ(cp_after.get_fabric_config(), tt_fabric::FabricConfig::FABRIC_1D);
    EXPECT_NE(&cp_before, &cp_after);

    auto& mesh_after = env.get_system_mesh();
    EXPECT_EQ(&mesh_before, &mesh_after);
    EXPECT_GT(mesh_after.shape().mesh_size(), 0);
}

// --- Context leaks on failed mesh device creation (#57286) ---

namespace {

// The lowest free slot is handed out first, so a leaked context shifts this upward.
ContextId next_free_context_id(const std::string& mock_path) {
    MetalEnv probe({.mock_cluster_desc_path = mock_path});
    ContextId id = MetalContext::create_instance(probe);
    MetalContext::destroy_instance(/*check_device_count=*/false, id);
    return id;
}

}  // namespace

TEST(MetalEnv, FailedCreateMeshDeviceDoesNotLeakContext) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1);
    const ContextId baseline = next_free_context_id(mock_path);

    for (int i = 0; i < 3; ++i) {
        MetalEnv env({.mock_cluster_desc_path = mock_path});
        // Fatals inside MeshDeviceImpl::create, after the context exists and before ownership transfers.
        EXPECT_THROW(
            distributed::MeshDevice::open_mesh_device(
                env,
                distributed::MeshDeviceConfig(
                    /*mesh_shape=*/std::nullopt, /*offset=*/std::nullopt, /*physical_device_ids=*/{0})),
            std::runtime_error);
    }

    EXPECT_EQ(next_free_context_id(mock_path).get(), baseline.get());
}

TEST(MetalEnv, FailedCreateUnitMeshDeviceDoesNotLeakContext) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1);
    const ContextId baseline = next_free_context_id(mock_path);

    for (int i = 0; i < 3; ++i) {
        MetalEnv env({.mock_cluster_desc_path = mock_path});
        // Device 99 is not in the 1-chip mock cluster, so opening it fails.
        EXPECT_THROW(distributed::MeshDevice::open_unit_mesh(env, 99), std::runtime_error);
    }

    EXPECT_EQ(next_free_context_id(mock_path).get(), baseline.get());
}

TEST(MetalEnv, FailedCreateUnitMeshesDoesNotLeakContext) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1);
    const ContextId baseline = next_free_context_id(mock_path);

    for (int i = 0; i < 3; ++i) {
        MetalEnv env({.mock_cluster_desc_path = mock_path});
        EXPECT_THROW(distributed::MeshDevice::open_unit_meshes(env, {}), std::runtime_error);
    }

    EXPECT_EQ(next_free_context_id(mock_path).get(), baseline.get());
}

// --- MeshDevice device properties ---

namespace {

void expect_same_properties(const MetalDeviceProperties& expected, const MetalDeviceProperties& actual) {
    EXPECT_EQ(actual.arch(), expected.arch());
    EXPECT_EQ(actual.arch_name(), expected.arch_name());
    EXPECT_EQ(actual.node_local_mem_size(), expected.node_local_mem_size());
    EXPECT_EQ(actual.dram_alignment(), expected.dram_alignment());
    EXPECT_EQ(actual.node_local_mem_alignment(), expected.node_local_mem_alignment());
    EXPECT_EQ(actual.num_dataflow_buffers(), expected.num_dataflow_buffers());
    EXPECT_EQ(actual.max_worker_node_local_mem_unreserved_size(), expected.max_worker_node_local_mem_unreserved_size());
    EXPECT_EQ(actual.sfpu_eps(), expected.sfpu_eps());
    EXPECT_EQ(std::bit_cast<uint32_t>(actual.sfpu_nan()), std::bit_cast<uint32_t>(expected.sfpu_nan()));
    EXPECT_EQ(actual.sfpu_inf(), expected.sfpu_inf());
}

void expect_mesh_matches(const distributed::MeshDevice& mesh, const MetalDeviceProperties& expected) {
    expect_same_properties(expected, experimental::get_device_properties(mesh));
    for (const auto& coord : distributed::MeshCoordinateRange(mesh.shape())) {
        SCOPED_TRACE(::testing::Message() << "coord " << coord);
        expect_same_properties(expected, experimental::get_device_properties(mesh, coord));
    }
}

MetalEnvDescriptor wormhole_n300_mock_descriptor() {
    return {.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2)};
}

// disable_mock_mode() only tears down the default context while its device manager is initialized, which is no
// longer the case once the mesh is closed. Drop it explicitly so later tests in this binary never inherit a mock
// default context.
struct LegacyMockCleanup {
    bool active = false;
    void release() {
        if (!active) {
            return;
        }
        active = false;
        experimental::disable_mock_mode();
        if (MetalContext::instance_exists()) {
            tt::tt_metal::detail::ReleaseOwnership();
        }
    }
    ~LegacyMockCleanup() { release(); }
};

}  // namespace

TEST(MeshDeviceProperties, ExplicitEnvMatchesEnv) {
    MetalEnv env(wormhole_n300_mock_descriptor());
    auto mesh =
        distributed::MeshDevice::open_mesh_device(env, distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(mesh, nullptr);

    expect_mesh_matches(*mesh, env.get_device_properties());

    mesh->close();
}

TEST(MeshDeviceProperties, EachMeshUsesItsOwnEnv) {
    MetalEnv wh_env(wormhole_n300_mock_descriptor());
    MetalEnv bh_env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1)});
    ASSERT_EQ(wh_env.get_device_properties().arch(), tt::ARCH::WORMHOLE_B0);
    ASSERT_EQ(bh_env.get_device_properties().arch(), tt::ARCH::BLACKHOLE);

    auto wh_mesh = distributed::MeshDevice::open_mesh_device(
        wh_env, distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    auto bh_mesh = distributed::MeshDevice::open_unit_mesh(bh_env, 0);
    ASSERT_NE(wh_mesh, nullptr);
    ASSERT_NE(bh_mesh, nullptr);

    {
        SCOPED_TRACE("blackhole mesh");
        EXPECT_EQ(experimental::get_device_properties(*bh_mesh).arch(), tt::ARCH::BLACKHOLE);
        expect_mesh_matches(*bh_mesh, bh_env.get_device_properties());
    }
    {
        SCOPED_TRACE("wormhole mesh");
        EXPECT_EQ(experimental::get_device_properties(*wh_mesh).arch(), tt::ARCH::WORMHOLE_B0);
        expect_mesh_matches(*wh_mesh, wh_env.get_device_properties());
    }

    bh_mesh->close();
    wh_mesh->close();
}

TEST(MeshDeviceProperties, LegacyOpenMatchesEnv) {
    LegacyMockCleanup cleanup;
    experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, /*num_chips=*/2);
    cleanup.active = true;

    auto mesh = distributed::MeshDevice::create(distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(mesh, nullptr);

    expect_mesh_matches(*mesh, MetalContext::instance().get_env().get_device_properties());

    mesh->close();
    mesh.reset();
    cleanup.release();
    EXPECT_FALSE(experimental::is_mock_mode_registered());
    EXPECT_FALSE(MetalContext::instance_exists());
}

TEST(MeshDeviceProperties, OutOfRangeCoordinateThrows) {
    MetalEnv env(wormhole_n300_mock_descriptor());
    auto mesh =
        distributed::MeshDevice::open_mesh_device(env, distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
    ASSERT_NE(mesh, nullptr);

    EXPECT_THROW(experimental::get_device_properties(*mesh, distributed::MeshCoordinate(0, 2)), std::runtime_error);
    EXPECT_THROW(experimental::get_device_properties(*mesh, distributed::MeshCoordinate(1, 0)), std::runtime_error);

    mesh->close();
}

TEST(MeshDeviceProperties, SubmeshAndUnitMesh) {
    {
        MetalEnv env(wormhole_n300_mock_descriptor());
        auto mesh = distributed::MeshDevice::open_mesh_device(
            env, distributed::MeshDeviceConfig{distributed::MeshShape{1u, 2u}});
        ASSERT_NE(mesh, nullptr);

        // Submesh coordinates are local: {0, 1} exists in the parent but not in a 1x1 submesh.
        auto submesh = mesh->create_submesh(distributed::MeshShape(1, 1));
        ASSERT_NE(submesh, nullptr);
        expect_mesh_matches(*submesh, env.get_device_properties());
        EXPECT_THROW(
            experimental::get_device_properties(*submesh, distributed::MeshCoordinate(0, 1)), std::runtime_error);

        submesh.reset();
        mesh->close();
    }
    {
        MetalEnv env(wormhole_n300_mock_descriptor());
        auto unit_mesh = distributed::MeshDevice::open_unit_mesh(env, 0);
        ASSERT_NE(unit_mesh, nullptr);
        expect_same_properties(env.get_device_properties(), experimental::get_device_properties(*unit_mesh));
        expect_same_properties(
            env.get_device_properties(),
            experimental::get_device_properties(*unit_mesh, distributed::MeshCoordinate(0, 0)));
        unit_mesh->close();
    }
}

}  // namespace tt::tt_metal
