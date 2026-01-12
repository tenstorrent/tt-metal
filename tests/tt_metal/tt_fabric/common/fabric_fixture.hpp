// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <hostdevcommon/common_values.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "common/tt_backend_api_types.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_fabric::fabric_router_tests {

class ControlPlaneFixture : public ::testing::Test {
   protected:
       tt::ARCH arch_{tt::ARCH::Invalid};
       void SetUp() override {
           auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
           if (not slow_dispatch) {
               log_info(
                   tt::LogTest,
                   "Control plane test suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
               GTEST_SKIP();
           }
           // reserve max available planes
           uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
           tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
               tt::tt_fabric::FabricConfig::FABRIC_2D, num_routing_planes);
       }

       void TearDown() override {
           tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
               tt::tt_fabric::FabricConfig::DISABLED);
       }
};

class BaseFabricFixture : public ::testing::Test {
public:
    inline static tt::ARCH arch_;
    inline static std::map<ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_map_;
    inline static std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    inline static bool slow_dispatch_;

    const std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& get_devices() const { return devices_; }
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& get_device(ChipId id) const {
        return devices_map_.at(id);
    }

    void SetUp() override {
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 2) {
            log_info(tt::LogTest, "Skipping fabric tests as there are less than 2 devices available");
            GTEST_SKIP();
        }
    }

    static void DoSetUpTestSuite(
        tt_fabric::FabricConfig fabric_config,
        std::optional<uint8_t> num_routing_planes = std::nullopt,
        tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED,
        tt_fabric::FabricUDMMode fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED) {
        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch_) {
            log_info(tt::LogTest, "Running fabric api tests with slow dispatch");
        } else {
            log_info(tt::LogTest, "Running fabric api tests with fast dispatch");
        }

        // Fabric Reliability Mode
        // Default to STRICT_SYSTEM_HEALTH_SETUP_MODE
        // If runtime option RELIABILITY_MODE is set, use the value from the runtime option
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
        // Query runtime options for an env-parsed override
        auto reliability_mode_override = tt::tt_metal::MetalContext::instance().rtoptions().get_reliability_mode();
        if (reliability_mode_override.has_value()) {
            reliability_mode = reliability_mode_override.value();
        }
        log_info(tt::LogTest, "Fabric Reliability Mode: {}", enchantum::to_string(reliability_mode));

        // Set up all available devices
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<ChipId> ids;
        ids.reserve(num_devices);
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        tt::tt_fabric::SetFabricConfig(
            fabric_config, reliability_mode, num_routing_planes, fabric_tensix_config, fabric_udm_mode);
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        devices_map_ = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);
        for (auto& [id, device] : devices_map_) {
            devices_.push_back(device);
        }
    }

    static void DoTearDownTestSuite() {
        for (auto& [id, device] : devices_map_) {
            device->close();
        }
        devices_map_.clear();
        devices_.clear();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }

    static void SetUpTestSuite() { TT_THROW("SetUpTestSuite not implemented in BaseFabricFixture"); }

    static void TearDownTestSuite() { TT_THROW("TearDownTestSuite not implemented in BaseFabricFixture"); }

    // NOLINTNEXTLINE(readability-make-member-function-const)
    void RunProgramNonblocking(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::LaunchProgram(device->get_devices()[0], program, false);
        } else {
            tt::tt_metal::distributed::MeshCommandQueue& cq = device->mesh_command_queue();
            // Create a mesh workload from the program
            auto& program_copy = program;
            auto mesh_workload = tt::tt_metal::distributed::MeshWorkload();
            mesh_workload.add_program(
                tt::tt_metal::distributed::MeshCoordinateRange(
                    tt::tt_metal::distributed::MeshCoordinate(0, 0), tt::tt_metal::distributed::MeshCoordinate(0, 0)),
                std::move(program_copy));
            tt::tt_metal::distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
        }
    }

    // NOLINTNEXTLINE(readability-make-member-function-const)
    void WaitForSingleProgramDone(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            // Wait for the program to finish
            tt::tt_metal::detail::WaitProgramDone(device->get_devices()[0], program);
        } else {
            // Wait for all programs on cq to finish
            tt::tt_metal::distributed::MeshCommandQueue& cq = device->mesh_command_queue();
            tt::tt_metal::distributed::Finish(cq);
        }
    }
};

class Fabric1DFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() { BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_1D); }
    static void TearDownTestSuite() { BaseFabricFixture::DoTearDownTestSuite(); }
};

// Template base class for Tensix fixtures with Galaxy skip logic
template <
    tt::tt_fabric::FabricConfig FabricConfigValue,
    tt::tt_fabric::FabricTensixConfig TensixConfigValue,
    tt::tt_fabric::FabricUDMMode UDMModeValue = tt::tt_fabric::FabricUDMMode::DISABLED>
class FabricTensixFixtureTemplate : public BaseFabricFixture {
private:
    inline static bool should_skip_ = false;

protected:
    static void SetUpTestSuite() {
        if (tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy() ||
            tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
            should_skip_ = true;
            return;
        }
        BaseFabricFixture::DoSetUpTestSuite(FabricConfigValue, std::nullopt, TensixConfigValue, UDMModeValue);
    }

    static void TearDownTestSuite() {
        if (!should_skip_) {
            BaseFabricFixture::DoTearDownTestSuite();
        }
    }

    void SetUp() override {
        if (should_skip_) {
            GTEST_SKIP() << "Tensix fixture tests are not supported on Galaxy systems";
        }
        BaseFabricFixture::SetUp();
    }
};

// Concrete fixture types using the template
using Fabric1DTensixFixture =
    FabricTensixFixtureTemplate<tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricTensixConfig::MUX>;

using NightlyFabric1DTensixFixture =
    FabricTensixFixtureTemplate<tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricTensixConfig::MUX>;

using NightlyFabric2DTensixFixture =
    FabricTensixFixtureTemplate<tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricTensixConfig::MUX>;

class NightlyFabric1DFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() { BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_1D); }
    static void TearDownTestSuite() { BaseFabricFixture::DoTearDownTestSuite(); }
};

class Fabric2DFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() { BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D); }
    static void TearDownTestSuite() { BaseFabricFixture::DoTearDownTestSuite(); }
};

class Fabric2DUDMModeFixture : public BaseFabricFixture {
private:
    inline static bool should_skip_ = false;

protected:
    static void SetUpTestSuite() {
        // Check specifically for Wormhole Galaxy
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        bool is_wormhole_galaxy = (arch_ == tt::ARCH::WORMHOLE_B0) &&
                                  (tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy() ||
                                   tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster());

        if (is_wormhole_galaxy) {
            should_skip_ = true;
            return;
        }

        BaseFabricFixture::DoSetUpTestSuite(
            tt::tt_fabric::FabricConfig::FABRIC_2D,
            std::nullopt,
            tt_fabric::FabricTensixConfig::UDM,
            tt_fabric::FabricUDMMode::ENABLED);
    }

    static void TearDownTestSuite() {
        if (!should_skip_) {
            BaseFabricFixture::DoTearDownTestSuite();
        }
    }

    void SetUp() override {
        if (should_skip_) {
            GTEST_SKIP() << "Tensix fixture tests are not supported on Wormhole Galaxy systems";
        }
        BaseFabricFixture::SetUp();
    }
};

class NightlyFabric2DUDMModeFixture : public Fabric2DUDMModeFixture {
protected:
    void SetUp() override {
        if (devices_.size() < 8) {
            GTEST_SKIP() << "Test requires at least 8 devices (2x4 mesh), found " << devices_.size();
        }
        Fabric2DUDMModeFixture::SetUp();
    }
};

class NightlyFabric2DFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() { BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D); }
    static void TearDownTestSuite() { BaseFabricFixture::DoTearDownTestSuite(); }
};

class CustomMeshGraphFabric2DFixture : public BaseFabricFixture {
public:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp(
        const std::string& mesh_graph_desc_file,
        const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
        tt::tt_metal::MetalContext::instance().set_custom_fabric_topology(
            mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
        BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D);
    }

private:
    void SetUp() override {}

    void TearDown() override {
        BaseFabricFixture::DoTearDownTestSuite();
        tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    }
};

class Galaxy1x32Fabric1DFixture : public BaseFabricFixture {
public:
    static constexpr std::string_view kMeshGraphDescriptorRelativePath =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto";

protected:
    inline static bool should_skip_ = false;

    static void SetUpTestSuite() {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        if (cluster.get_cluster_type() != tt::tt_metal::ClusterType::GALAXY ||
            tt::tt_metal::GetNumAvailableDevices() < 32) {
            should_skip_ = true;
            return;
        }

        std::filesystem::path mesh_graph_desc_path =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
            kMeshGraphDescriptorRelativePath;
        TT_FATAL(
            std::filesystem::exists(mesh_graph_desc_path),
            "Galaxy1x32Fabric1DFixture requires mesh graph descriptor {} but it was not found",
            mesh_graph_desc_path.string());

        tt::tt_metal::MetalContext::instance().set_custom_fabric_topology(mesh_graph_desc_path.string(), {});
        BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_1D);
    }

    static void TearDownTestSuite() {
        if (should_skip_) {
            return;
        }
        BaseFabricFixture::DoTearDownTestSuite();
        tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    }

    void SetUp() override {
        if (should_skip_) {
            GTEST_SKIP() << "Galaxy1x32Fabric1DFixture requires a Galaxy system with at least 32 chips.";
        }
        BaseFabricFixture::SetUp();
    }
};

class T3kCustomMeshGraphFabric2DFixture
    : public CustomMeshGraphFabric2DFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::vector<std::vector<EthCoord>>>> {
    void SetUp() override {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
            GTEST_SKIP();
        }
    }
};

struct McastRoutingInfo {
    RoutingDirection mcast_dir;
    uint32_t num_mcast_hops;
};

void RunTestUnicastRaw(
    BaseFabricFixture* fixture, uint32_t num_hops = 1, RoutingDirection direction = RoutingDirection::E);

void RunTestUnicastConnAPI(
    BaseFabricFixture* fixture, uint32_t num_hops = 1, RoutingDirection direction = RoutingDirection::E, bool use_dram_dst = false);

void RunTestUnicastConnAPIRandom(BaseFabricFixture* fixture);

void RunTestUnicastRaw2D(
    BaseFabricFixture* fixture, uint32_t ns_hops, RoutingDirection ns_dir, uint32_t ew_hops, RoutingDirection ew_dir);

void RunTestMCastConnAPI(
    BaseFabricFixture* fixture,
    RoutingDirection fwd_dir = RoutingDirection::W,
    uint32_t fwd_hops = 1,
    RoutingDirection bwd_dir = RoutingDirection::E,
    uint32_t bwd_hops = 1);

void RunTest2DMCastConnAPI(
    BaseFabricFixture* fixture, uint32_t north_hops, uint32_t south_hops, uint32_t east_hops, uint32_t west_hops);

void RunTestChipMCast1D(BaseFabricFixture* fixture, RoutingDirection dir, uint32_t start_distance, uint32_t range);

void RunTestLineMcast(BaseFabricFixture* fixture, const std::vector<McastRoutingInfo>& mcast_routing_info);

enum NocSendType : uint8_t {
    NOC_UNICAST_WRITE = 0,
    NOC_UNICAST_INLINE_WRITE = 1,
    NOC_UNICAST_ATOMIC_INC = 2,
    NOC_FUSED_UNICAST_ATOMIC_INC = 3,
    NOC_UNICAST_SCATTER_WRITE = 4,
    NOC_MULTICAST_WRITE = 5,       // mcast has bug
    NOC_MULTICAST_ATOMIC_INC = 6,  // mcast has bug
    NOC_UNICAST_READ = 7,          // read wont be supported without UDM mode
    NOC_SEND_TYPE_LAST = NOC_UNICAST_SCATTER_WRITE
};

void FabricUnicastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::vector<std::tuple<RoutingDirection, uint32_t /*num_hops*/>>& pair_ordered_dirs,
    FabricApiType api_type = FabricApiType::Linear,
    bool with_state = false);

void UDMFabricUnicastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::variant<
        std::tuple<RoutingDirection, uint32_t /*num_hops*/>,
        std::tuple<uint32_t /*src_node*/, uint32_t /*dest_node*/>>& routing_info,
    std::optional<RoutingDirection> override_initial_direction = std::nullopt,
    std::optional<std::vector<std::pair<CoreCoord, CoreCoord>>> worker_coords_list = std::nullopt,
    bool dual_risc = false);

void UDMFabricUnicastAllToAllCommon(BaseFabricFixture* fixture, NocSendType noc_send_type, bool dual_risc = false);

void FabricMulticastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::vector<std::tuple<RoutingDirection, uint32_t /*start_distance*/, uint32_t /*range*/>>& dir_configs,
    bool with_state = false);

void Fabric2DMulticastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::vector<std::vector<std::tuple<RoutingDirection, uint32_t, uint32_t>>>& connection_configs,
    bool with_state = false);

void RunEDMConnectionStressTest(
    BaseFabricFixture* fixture,
    const std::vector<size_t>& stall_durations_cycles,
    const std::vector<size_t>& message_counts,
    const std::vector<size_t>& packet_sizes,
    size_t num_iterations,
    size_t num_times_to_connect,
    const std::vector<size_t>& workers_count,
    const std::vector<size_t>& test_rows);

void RunTestUnicastSmoke(BaseFabricFixture* fixture);

}  // namespace tt::tt_fabric::fabric_router_tests
