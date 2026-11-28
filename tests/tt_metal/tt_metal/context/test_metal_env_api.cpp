// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <tt_stl/cleanup.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/discovery/system_mesh.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/device/mock_device_util.hpp"
#include "impl/drisc_service/seat_lock.hpp"
#include "tt_metal/llrt/hal.hpp"
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

TEST(MetalEnv, FabricSettingsDefaults) {
    FabricSettings desc;
    EXPECT_EQ(desc.fabric_config, tt_fabric::FabricConfig::DISABLED);
    EXPECT_EQ(desc.reliability_mode, tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    EXPECT_FALSE(desc.num_routing_planes.has_value());
    EXPECT_EQ(desc.fabric_tensix_config, tt_fabric::FabricTensixConfig::DISABLED);
    EXPECT_EQ(desc.fabric_udm_mode, tt_fabric::FabricUDMMode::DISABLED);
    EXPECT_EQ(desc.fabric_manager, tt_fabric::FabricManagerMode::DEFAULT);
    EXPECT_EQ(desc.mesh_id, tt_fabric::MeshId{0});
    EXPECT_EQ(desc.host_rank, tt_fabric::MESH_HOST_RANK_UNSET);
}

MetalEnvDescriptor wormhole_mock_descriptor() {
    return {.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)};
}

TEST(MetalEnv, ConfigureFabricStoresDescriptor) {
    FabricSettings fc;
    fc.fabric_config = tt_fabric::FabricConfig::FABRIC_1D;
    fc.reliability_mode = tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
    fc.num_routing_planes = 4;
    fc.fabric_tensix_config = tt_fabric::FabricTensixConfig::MUX;
    fc.mesh_id = tt_fabric::MeshId{1};
    fc.host_rank = tt_fabric::MeshHostRankId{2};

    MetalEnv env(wormhole_mock_descriptor());
    env.configure_fabric(fc);

    const auto& stored = env.get_fabric_settings();
    EXPECT_EQ(stored.fabric_config, tt_fabric::FabricConfig::FABRIC_1D);
    EXPECT_EQ(stored.reliability_mode, tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    EXPECT_TRUE(stored.num_routing_planes.has_value());
    EXPECT_EQ(stored.num_routing_planes.value(), 4);
    EXPECT_EQ(stored.fabric_tensix_config, tt_fabric::FabricTensixConfig::MUX);
    EXPECT_EQ(stored.mesh_id, tt_fabric::MeshId{1});
    EXPECT_EQ(stored.host_rank, tt_fabric::MeshHostRankId{2});
}

TEST(MetalEnv, DefaultFabricConfigIsDisabled) {
    MetalEnv env(wormhole_mock_descriptor());
    EXPECT_EQ(env.get_fabric_settings().fabric_config, tt_fabric::FabricConfig::DISABLED);
}

// Without configure_fabric, the mesh binding comes from TT_MESH_ID / TT_MESH_HOST_RANK (mesh 0 when unset; host rank
// MESH_HOST_RANK_UNSET when unset, deferred to the topology mapper).
TEST(MetalEnv, DefaultMeshBindingComesFromRtOptions) {
    MetalEnv env(wormhole_mock_descriptor());
    const auto& rtoptions = MetalEnvAccessor(env).impl().get_rtoptions();
    const auto& desc = env.get_fabric_settings();
    EXPECT_EQ(desc.mesh_id, rtoptions.get_fabric_mesh_id());
    EXPECT_EQ(desc.host_rank, rtoptions.get_fabric_mesh_host_rank());
}

// configure_fabric takes the mesh binding from the descriptor it is given, not from the environment.
TEST(MetalEnv, ConfigureFabricMeshBindingFromDescriptor) {
    MetalEnv env(wormhole_mock_descriptor());
    env.configure_fabric(
        {.fabric_config = tt_fabric::FabricConfig::FABRIC_1D,
         .mesh_id = tt_fabric::MeshId{3},
         .host_rank = tt_fabric::MeshHostRankId{1}});
    EXPECT_EQ(env.get_fabric_settings().mesh_id, tt_fabric::MeshId{3});
    EXPECT_EQ(env.get_fabric_settings().host_rank, tt_fabric::MeshHostRankId{1});

    env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D});
    EXPECT_EQ(env.get_fabric_settings().mesh_id, tt_fabric::MeshId{0});
    EXPECT_EQ(env.get_fabric_settings().host_rank, tt_fabric::MESH_HOST_RANK_UNSET);
}

TEST(MetalEnv, ConfigureFabricLastCallWins) {
    MetalEnv env(wormhole_mock_descriptor());
    env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D});
    env.configure_fabric(
        {.fabric_config = tt_fabric::FabricConfig::FABRIC_2D,
         .fabric_tensix_config = tt_fabric::FabricTensixConfig::MUX});

    EXPECT_EQ(env.get_fabric_settings().fabric_config, tt_fabric::FabricConfig::FABRIC_2D);
    EXPECT_EQ(MetalEnvAccessor(env).impl().get_fabric_config(), tt_fabric::FabricConfig::FABRIC_2D);
    EXPECT_EQ(MetalEnvAccessor(env).impl().get_fabric_tensix_config(), tt_fabric::FabricTensixConfig::MUX);
}

TEST(MetalEnv, ConfigureFabricNormalizesRoutingPlanes) {
    MetalEnv env(wormhole_mock_descriptor());
    env.configure_fabric({.fabric_config = tt_fabric::FabricConfig::FABRIC_1D});

    EXPECT_FALSE(env.get_fabric_settings().num_routing_planes.has_value());
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
    EXPECT_EQ(env.get_fabric_settings().fabric_config, tt_fabric::FabricConfig::FABRIC_1D);
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

// MGD_DEFAULT is resolved by the control plane from the mesh's dim_types and stored back as soon as the control plane
// is built. After that the usual override rule applies to the concrete config: re-setting it rebuilds (and
// re-validates) the control plane, anything else must go through DISABLED.
TEST(MetalEnv, MgdDefaultFabricConfigResolvesAndReset) {
    MetalEnv env({.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 8)});
    MetalEnvImpl& impl = MetalEnvAccessor(env).impl();
    const auto mgd_path = std::filesystem::path(impl.get_rtoptions().get_root_dir()) /
                          "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    impl.set_custom_fabric_topology(mgd_path.string(), {});
    const auto relaxed = tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
    impl.set_fabric_config(tt_fabric::FabricConfig::MGD_DEFAULT, relaxed);

    const auto resolved = impl.get_fabric_config();
    EXPECT_NE(resolved, tt_fabric::FabricConfig::MGD_DEFAULT);
    EXPECT_EQ(
        resolved,
        impl.get_control_plane().get_mesh_graph().get_mesh_graph_descriptor().default_fabric_config_from_dim_types(
            tt_fabric::MeshId{0}));

    EXPECT_NO_THROW(impl.set_fabric_config(resolved, relaxed));
    EXPECT_EQ(impl.get_fabric_config(), resolved);
    EXPECT_EQ(impl.get_control_plane().get_fabric_config(), resolved);

    EXPECT_THROW(impl.set_fabric_config(tt_fabric::FabricConfig::MGD_DEFAULT, relaxed), std::runtime_error);
    EXPECT_THROW(impl.set_fabric_config(tt_fabric::FabricConfig::FABRIC_1D, relaxed), std::runtime_error);
    EXPECT_EQ(impl.get_fabric_config(), resolved);

    impl.set_fabric_config(tt_fabric::FabricConfig::DISABLED);
    impl.set_fabric_config(tt_fabric::FabricConfig::MGD_DEFAULT, relaxed);
    EXPECT_EQ(impl.get_fabric_config(), resolved);

    impl.set_fabric_config(tt_fabric::FabricConfig::DISABLED);
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
    EXPECT_EQ(env.get_fabric_settings().fabric_config, tt_fabric::FabricConfig::DISABLED);

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

// Seat locks against a real MetalEnv. Needs Blackhole silicon whose firmware turns the DRAM programmable cores on.
// Seat locks exclude each other within one process too, so this process stands in for another one (a DRISC service
// client, say). Opens no device, so no DRISC is touched whatever the override says.
class MetalEnvSeatLocks : public ::testing::Test {
protected:
    static constexpr const char* override_var = "TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES";

    void SetUp() override {
        if (const char* value = std::getenv(override_var)) {
            saved_override_ = value;
        }
        // Unset, metal decides from the firmware.
        unsetenv(override_var);
        MetalEnv env;
        const auto& cluster = MetalEnvAccessor(env).impl().get_cluster();
        if (cluster.arch() != tt::ARCH::BLACKHOLE || cluster.get_target_device_type() != tt::TargetDevice::Silicon) {
            GTEST_SKIP() << "Seat locks guard the DRAM programmable cores of Blackhole silicon only";
        }
        if (!dram_cores_on(env)) {
            GTEST_SKIP() << "This firmware leaves the DRAM programmable cores off";
        }
        for (const ChipId chip : cluster.all_chip_ids()) {
            const uint64_t asic_id = cluster.get_unique_chip_ids().at(chip);
            for (const CoreCoord& core : cluster.get_soc_desc(chip).get_metal_dram_cores(CoordSystem::NOC0)) {
                seats_.push_back({
                    .asic_id = asic_id,
                    .noc0_x = static_cast<uint32_t>(core.x),
                    .noc0_y = static_cast<uint32_t>(core.y),
                });
            }
        }
    }

    void TearDown() override {
        if (saved_override_.has_value()) {
            setenv(override_var, saved_override_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(override_var);
        }
    }

    static bool dram_cores_on(MetalEnv& env) {
        return MetalEnvAccessor(env).impl().get_hal().has_programmable_core_type(HalProgrammableCoreType::DRAM);
    }

    // The seat of every DRAM core a MetalEnv programs on this host.
    std::vector<internal::drisc_service::SeatId> seats_;

private:
    std::optional<std::string> saved_override_;
};

// While another process holds the seats, a MetalEnv turns its DRAM programmable cores off, fails when they are forced
// on, and turns them on again once the seats are free.
TEST_F(MetalEnvSeatLocks, LeavesDramCoresWhoseSeatsAreHeld) {
    std::vector<internal::drisc_service::SeatLock> held;
    for (const internal::drisc_service::SeatId& seat : seats_) {
        std::optional<internal::drisc_service::SeatLock> lock =
            internal::drisc_service::SeatLock::try_acquire_client(seat);
        ASSERT_TRUE(lock.has_value()) << "another process holds the seat of DRAM core (" << seat.noc0_x << ", "
                                      << seat.noc0_y << ")";
        held.push_back(std::move(*lock));
    }
    {
        MetalEnv env;
        EXPECT_FALSE(dram_cores_on(env)) << "metal turned its DRAM programmable cores on while their seats were held";
    }
    setenv(override_var, "1", /*overwrite=*/1);
    try {
        MetalEnv env;
        ADD_FAILURE() << "metal started with its DRAM programmable cores forced on while their seats were held";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string_view(e.what()).find("holds the seat of"), std::string_view::npos) << e.what();
    }
    unsetenv(override_var);
    held.clear();
    MetalEnv env;
    EXPECT_TRUE(dram_cores_on(env)) << "metal kept its DRAM programmable cores off after the seats were freed";
}

// One held seat turns every DRAM core off, so metal must give back the seats it took before reaching the held one.
TEST_F(MetalEnvSeatLocks, ReleasesEverySeatWhenOneIsHeld) {
    ASSERT_FALSE(seats_.empty());
    const std::optional<internal::drisc_service::SeatLock> last =
        internal::drisc_service::SeatLock::try_acquire_client(seats_.back());
    ASSERT_TRUE(last.has_value()) << "another process holds the seat of the last DRAM core";
    MetalEnv env;
    ASSERT_FALSE(dram_cores_on(env)) << "metal turned its DRAM programmable cores on while a seat was held";
    for (size_t i = 0; i + 1 < seats_.size(); i++) {
        EXPECT_TRUE(internal::drisc_service::SeatLock::try_acquire_exclusive(seats_[i]).has_value())
            << "metal kept the seat of DRAM core (" << seats_[i].noc0_x << ", " << seats_[i].noc0_y
            << ") with its DRAM programmable cores off";
    }
}

// TT_METAL_DRAM_KERNEL_SEATS against a real MetalEnv: metal takes the seats it names and leaves every other DRAM core
// to other owners.
class MetalEnvDramKernelSeats : public MetalEnvSeatLocks {
protected:
    static constexpr const char* seats_var = "TT_METAL_DRAM_KERNEL_SEATS";
    // Bank 0's free core: a seat every Blackhole product has.
    static constexpr const char* one_seat = "0:1";

    void SetUp() override {
        if (const char* value = std::getenv(seats_var)) {
            saved_seats_ = value;
        }
        unsetenv(seats_var);
        MetalEnvSeatLocks::SetUp();
    }

    void TearDown() override {
        if (saved_seats_.has_value()) {
            setenv(seats_var, saved_seats_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(seats_var);
        }
        MetalEnvSeatLocks::TearDown();
    }

    // The seat of every DRAM core `env` programs.
    static std::vector<internal::drisc_service::SeatId> seats_of(MetalEnv& env) {
        const auto& cluster = MetalEnvAccessor(env).impl().get_cluster();
        std::vector<internal::drisc_service::SeatId> seats;
        for (const ChipId chip : cluster.all_chip_ids()) {
            const uint64_t asic_id = cluster.get_unique_chip_ids().at(chip);
            for (const CoreCoord& core : cluster.get_soc_desc(chip).get_metal_dram_cores(CoordSystem::NOC0)) {
                seats.push_back(
                    {.asic_id = asic_id,
                     .noc0_x = static_cast<uint32_t>(core.x),
                     .noc0_y = static_cast<uint32_t>(core.y)});
            }
        }
        return seats;
    }

    static bool contains(
        const std::vector<internal::drisc_service::SeatId>& seats, const internal::drisc_service::SeatId& seat) {
        return std::any_of(seats.begin(), seats.end(), [&](const internal::drisc_service::SeatId& s) {
            return s.asic_id == seat.asic_id && s.noc0_x == seat.noc0_x && s.noc0_y == seat.noc0_y;
        });
    }

    static void expect_metal_env_fails_with(std::string_view message) {
        try {
            MetalEnv env;
            ADD_FAILURE() << "metal started; expected a failure mentioning '" << message << "'";
        } catch (const std::exception& e) {
            EXPECT_NE(std::string_view(e.what()).find(message), std::string_view::npos) << e.what();
        }
    }

private:
    std::optional<std::string> saved_seats_;
};

TEST_F(MetalEnvDramKernelSeats, TakesOnlyTheNamedSeats) {
    setenv(seats_var, one_seat, /*overwrite=*/1);
    MetalEnv env;
    ASSERT_TRUE(dram_cores_on(env));
    const std::vector<internal::drisc_service::SeatId> reserved = seats_of(env);
    EXPECT_EQ(reserved.size(), MetalEnvAccessor(env).impl().get_cluster().all_chip_ids().size())
        << "one seat named, so one DRAM core per chip";
    for (const internal::drisc_service::SeatId& seat : seats_) {
        const bool is_reserved = contains(reserved, seat);
        EXPECT_EQ(internal::drisc_service::SeatLock::try_acquire_client(seat).has_value(), !is_reserved)
            << "DRAM core (" << seat.noc0_x << ", " << seat.noc0_y << ") of ASIC " << seat.asic_id << " is "
            << (is_reserved ? "a named seat, so metal holds it" : "not a named seat, so a client can take it");
    }
}

TEST_F(MetalEnvDramKernelSeats, FailsWhenANamedSeatIsHeld) {
    setenv(seats_var, one_seat, /*overwrite=*/1);
    std::vector<internal::drisc_service::SeatId> reserved;
    {
        MetalEnv env;
        reserved = seats_of(env);
    }
    ASSERT_FALSE(reserved.empty());
    const std::optional<internal::drisc_service::SeatLock> client =
        internal::drisc_service::SeatLock::try_acquire_client(reserved.front());
    ASSERT_TRUE(client.has_value()) << "another process holds the named seat";
    expect_metal_env_fails_with("TT_METAL_DRAM_KERNEL_SEATS names the seat of");
}

TEST_F(MetalEnvDramKernelSeats, RejectsASeatTheChipLacks) {
    setenv(seats_var, "99:0", /*overwrite=*/1);
    expect_metal_env_fails_with("out of range");
}

TEST_F(MetalEnvDramKernelSeats, RejectsASeatThatIsNotBankColonNoc) {
    for (const char* value : {"3", "3:", ":1", "3:2", "a:0", "3:1,", "3:1,3:1"}) {
        setenv(seats_var, value, /*overwrite=*/1);
        expect_metal_env_fails_with("TT_METAL_DRAM_KERNEL_SEATS");
    }
}

// With TT_METAL_CLEAR_L1, opening a device clears the L1 of every DRAM core metal takes. A core metal did not reserve
// must come through untouched.
TEST_F(MetalEnvDramKernelSeats, DeviceOpenLeavesOtherDramCoresAlone) {
    constexpr uint32_t sentinel = 0x5EA7C0DE;
    constexpr const char* clear_l1_var = "TT_METAL_CLEAR_L1";
    const char* clear_l1_before = std::getenv(clear_l1_var);
    const std::optional<std::string> saved_clear_l1 =
        clear_l1_before != nullptr ? std::optional<std::string>(clear_l1_before) : std::nullopt;
    const auto restore_clear_l1 = ttsl::make_cleanup([&] {
        if (saved_clear_l1.has_value()) {
            setenv(clear_l1_var, saved_clear_l1->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(clear_l1_var);
        }
    });
    std::map<ChipId, std::vector<CoreCoord>> programmable;
    {
        MetalEnv env;
        const auto& cluster = MetalEnvAccessor(env).impl().get_cluster();
        for (const ChipId chip : cluster.all_chip_ids()) {
            programmable[chip] = cluster.get_soc_desc(chip).get_metal_dram_cores(CoordSystem::TRANSLATED);
        }
    }

    setenv(seats_var, one_seat, /*overwrite=*/1);
    setenv(clear_l1_var, "1", /*overwrite=*/1);
    MetalEnv env;
    const auto& cluster = MetalEnvAccessor(env).impl().get_cluster();
    const uint64_t l1_addr = MetalEnvAccessor(env).impl().get_hal().get_dev_noc_addr(
        HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const auto is_reserved = [&](ChipId chip, const CoreCoord& core) {
        const std::vector<CoreCoord> reserved =
            cluster.get_soc_desc(chip).get_metal_dram_cores(CoordSystem::TRANSLATED);
        return std::find(reserved.begin(), reserved.end(), core) != reserved.end();
    };

    std::vector<ChipId> chips;
    for (const auto& [chip, cores] : programmable) {
        chips.push_back(chip);
        for (const CoreCoord& core : cores) {
            cluster.write_core(&sentinel, sizeof(sentinel), tt_cxy_pair(chip, core), l1_addr);
        }
    }
    {
        auto devices = distributed::MeshDevice::open_unit_meshes(env, chips);
        for (auto& [chip, device] : devices) {
            device->close();
        }
    }
    size_t untouched = 0;
    for (const auto& [chip, cores] : programmable) {
        for (const CoreCoord& core : cores) {
            uint32_t value = 0;
            cluster.read_core(&value, sizeof(value), tt_cxy_pair(chip, core), l1_addr);
            if (is_reserved(chip, core)) {
                EXPECT_NE(value, sentinel)
                    << "chip " << chip << " DRAM core " << core.str() << " is a named seat, which a device open clears";
            } else {
                EXPECT_EQ(value, sentinel) << "chip " << chip << " DRAM core " << core.str()
                                           << " is not a named seat, yet a device open changed its L1";
                untouched++;
            }
        }
    }
    EXPECT_GT(untouched, 0u);
}

}  // namespace tt::tt_metal
