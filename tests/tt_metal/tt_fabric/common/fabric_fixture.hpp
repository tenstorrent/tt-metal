// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include <tt-metalium/allocator.hpp>
#include "test_host_kernel_common.hpp"

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

struct WorkerMemMap {
    uint32_t source_l1_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t test_results_address;
    uint32_t target_address;
    uint32_t notification_mailbox_address;
    uint32_t test_results_size_bytes;
};

class BaseFabricFixture : public ::testing::Test {
public:
    inline static tt::ARCH arch_;
    inline static std::map<ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_map_;
    inline static std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    inline static bool slow_dispatch_;
    // FIX TK (#42429): Set to true when the fabric cluster has too few chips to run tests.
    // Happens when the topology mapper downgrades to 1x1 on a severely degraded T3K cluster
    // (all ETH links dead after progressive SIGKILL teardowns).  SetUp() skips when this is true.
    inline static bool cluster_degraded_skip_ = false;

    const std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& get_devices() const { return devices_; }
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& get_device(ChipId id) const {
        return devices_map_.at(id);
    }

    void SetUp() override {
        // FIX TK (#42429): Skip per-test if the fabric cluster was downgraded (e.g. 1x1 on T3K
        // with all ETH links dead).  This check runs before the device-count guard so we don't
        // crash trying to use devices that aren't in the cluster.
        if (cluster_degraded_skip_) {
            GTEST_SKIP() << "FIX TK (#42429): fabric cluster has fewer chips than expected "
                            "(topology downgraded on severely degraded hardware — skipping)";
        }
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

        // FIX TK (#42429): After topology discovery, some chips may not be in the fabric cluster
        // (e.g. when all ETH links are dead and the mapper degrades to 1x1).  Filter ids to only
        // chips the control plane knows about; if fewer chips remain than requested, mark the
        // suite as degraded so per-test SetUp() can skip gracefully.
        cluster_degraded_skip_ = false;
        {
            const auto& control_plane =
                tt::tt_metal::MetalContext::instance().get_control_plane();
            std::vector<ChipId> cluster_ids;
            cluster_ids.reserve(ids.size());
            for (ChipId id : ids) {
                if (control_plane.is_physical_chip_in_fabric_cluster(id)) {
                    cluster_ids.push_back(id);
                } else {
                    log_warning(
                        tt::LogTest,
                        "FIX TK (#42429): Physical chip {} not in fabric cluster "
                        "(topology downgraded on degraded hardware). Excluding from unit meshes.",
                        id);
                }
            }
            if (cluster_ids.size() < ids.size()) {
                cluster_degraded_skip_ = true;
                // FIX TL (#42429): Do NOT proceed to create_unit_meshes with a partial chip set.
                // A degraded cluster (topology downgraded to 1x1 etc.) may crash fabric initializers
                // (e.g. UDM tensix builder) that require the full expected topology.
                // SetUp() will GTEST_SKIP each test via cluster_degraded_skip_.
                log_warning(
                    tt::LogTest,
                    "FIX TL (#42429): Fabric cluster has only {}/{} chips — skipping create_unit_meshes "
                    "to avoid crashing fabric initializers on a degenerate topology.",
                    cluster_ids.size(),
                    ids.size());
                return;
            }
            if (cluster_ids.empty()) {
                log_warning(tt::LogTest, "FIX TK (#42429): No chips in fabric cluster — skipping SetUpTestSuite.");
                return;
            }
            ids = std::move(cluster_ids);
        }

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
        cluster_degraded_skip_ = false;  // FIX TK (#42429): reset for next suite
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

    // Utility function reused across tests to get address params
    static WorkerMemMap generate_worker_mem_map(
        const std::shared_ptr<tt_metal::distributed::MeshDevice>& device, Topology /*topology*/) {
        constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
        constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
        constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;

        uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
        uint32_t source_l1_buffer_address = base_addr + PACKET_HEADER_RESERVED_BYTES;
        uint32_t test_results_address = source_l1_buffer_address + DATA_SPACE_RESERVED_BYTES;
        uint32_t target_address = source_l1_buffer_address;
        uint32_t notification_mailbox_address = test_results_address + TEST_RESULTS_SIZE_BYTES;

        uint32_t packet_payload_size_bytes = get_tt_fabric_max_payload_size_bytes();

        return {
            source_l1_buffer_address,
            packet_payload_size_bytes,
            test_results_address,
            target_address,
            notification_mailbox_address,
            TEST_RESULTS_SIZE_BYTES};
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

    void SetUp() override {
        BaseFabricFixture::SetUp();
        // FIX SA (#42429): skip fabric unicast tests on degraded cluster — non-MMIO devices
        // with broken relay paths will throw TT_THROW in read_completion_queue_event (FIX Z),
        // turning a data-path race condition test into an uninformative crash.
        for (const auto& mesh_dev : devices_) {
            for (auto* dev : mesh_dev->get_devices()) {
                if (dev->is_fabric_relay_path_broken() || dev->is_fabric_channels_not_ready_for_traffic() ||
                    dev->is_fabric_stale_base_umd_channels()) {
                    GTEST_SKIP() << "Fabric2DFixture: device " << dev->id()
                                 << " has broken relay path or channels not ready"
                                 << " — skipping unicast test on degraded cluster";
                }
            }
        }
    }
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
    // FIX TC (#42429): prevents double-init when T3kCustomMeshGraphFabric2DFixture::SetUp()
    // (GTest hook) calls SetUp(desc, mapping) early so GTEST_SKIP() fires in the right context,
    // and the test body then calls SetUp(desc, mapping) again.
    inline static bool custom_setup_initialized_ = false;

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp(
        const std::string& mesh_graph_desc_file,
        const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
        // FIX TC (#42429): If T3kCustomMeshGraphFabric2DFixture::SetUp() (the GTest hook) already
        // called us, the fabric mesh is set up and the degraded check already ran.  Calling again
        // from the test body would double-init; bail out early.
        if (custom_setup_initialized_) {
            return;
        }
        custom_setup_initialized_ = true;
        tt::tt_metal::MetalContext::instance().set_custom_fabric_topology(
            mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
        BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D);
        // FIX SA (#42429): skip fabric unicast tests on degraded cluster — non-MMIO devices
        // with broken relay paths will throw TT_THROW in read_completion_queue_event (FIX Z),
        // turning a data-path race condition test into an uninformative crash.
        // NOTE: GTEST_SKIP() only stops the test when called in a GTest hook (SetUp/TearDown).
        // When called from the test body (or helpers called from the test body) it just returns.
        // T3kCustomMeshGraphFabric2DFixture::SetUp() calls us from the hook, so the skip below
        // will be honoured by GTest.  Other callers (e.g. Galaxy1x32Fabric1DFixture) invoke us
        // from SetUpTestSuite which is also a hook, so they are fine too.
        for (const auto& mesh_dev : devices_) {
            for (auto* dev : mesh_dev->get_devices()) {
                if (dev->is_fabric_relay_path_broken() || dev->is_fabric_channels_not_ready_for_traffic() ||
                    dev->is_fabric_stale_base_umd_channels()) {
                    GTEST_SKIP() << "CustomMeshGraphFabric2DFixture: device " << dev->id()
                                 << " has broken relay path or channels not ready"
                                 << " — skipping unicast test on degraded cluster";
                }
            }
        }
    }

private:
    void SetUp() override {}

    void TearDown() override {
        custom_setup_initialized_ = false;  // FIX TC: reset for next test
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
        // FIX TB (#42429): skip non-T3K systems before any fabric init to avoid teardown
        // timeouts and topology re-discovery crashes on incompatible cluster types.
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
            GTEST_SKIP();
        }
        // FIX TB (#42429): run the base-class guard (e.g. < 2 devices) so the fixture
        // is correctly skipped on under-provisioned systems without entering fabric init.
        BaseFabricFixture::SetUp();

        // FIX TC (#42429): initialize the fabric mesh from within the GTest SetUp() hook so
        // that GTEST_SKIP() in the degraded-cluster check inside
        // CustomMeshGraphFabric2DFixture::SetUp(desc, mapping) actually stops test execution.
        //
        // Root cause: GTEST_SKIP() expands to "return GTEST_MESSAGE_(..., kSkip)".  When called
        // from a helper that is invoked from the *test body* (not from SetUp), the "return" only
        // exits the helper — the test body keeps running regardless of the kSkip result.  GTest
        // only checks IsSkipped() after SetUp() returns, not mid-test-body.
        //
        // By calling CustomMeshGraphFabric2DFixture::SetUp(desc, mapping) here (from the GTest
        // hook), any GTEST_SKIP() inside propagates "return" back to this function, which then
        // returns to GTest.  GTest sees IsSkipped()==true and skips the test body entirely.
        //
        // CustomMeshGraphFabric2DFixture::SetUp(desc, mapping) is idempotent: the
        // custom_setup_initialized_ flag prevents re-init when the test body calls it again.
        const auto& [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        std::map<FabricNodeId, ChipId> physical_chip_ids_mapping;
        for (std::uint32_t mesh_id = 0; mesh_id < mesh_graph_eth_coords.size(); mesh_id++) {
            for (std::uint32_t chip_id = 0; chip_id < mesh_graph_eth_coords[mesh_id].size(); chip_id++) {
                const auto& eth_coord = mesh_graph_eth_coords[mesh_id][chip_id];
                auto maybe_chip_id = cluster.try_get_physical_chip_id_from_eth_coord(eth_coord);
                if (!maybe_chip_id.has_value()) {
                    GTEST_SKIP()
                        << "T3kCustomMeshGraphFabric2DFixture: EthCoord ("
                        << eth_coord.rack << "," << eth_coord.shelf << "," << eth_coord.x << "," << eth_coord.y
                        << ") not found in cluster — skipping on incompatible topology";
                }
                physical_chip_ids_mapping.insert(
                    {FabricNodeId(MeshId{mesh_id}, chip_id), *maybe_chip_id});
            }
        }
        CustomMeshGraphFabric2DFixture::SetUp(mesh_graph_desc_path, physical_chip_ids_mapping);
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

void FabricUnicastCommon(
    BaseFabricFixture* fixture,
    NocPacketType noc_packet_type,
    const std::vector<std::tuple<RoutingDirection, uint32_t /*num_hops*/>>& pair_ordered_dirs,
    FabricApiType api_type = FabricApiType::Linear,
    bool with_state = false);

void UDMFabricUnicastCommon(
    BaseFabricFixture* fixture,
    NocPacketType noc_packet_type,
    const std::variant<
        std::tuple<RoutingDirection, uint32_t /*num_hops*/>,
        std::tuple<uint32_t /*src_node*/, uint32_t /*dest_node*/>>& routing_info,
    std::optional<RoutingDirection> override_initial_direction = std::nullopt,
    std::optional<std::vector<std::pair<CoreCoord, CoreCoord>>> worker_coords_list = std::nullopt,
    bool dual_risc = false);

void UDMFabricUnicastAllToAllCommon(BaseFabricFixture* fixture, NocPacketType noc_packet_type, bool dual_risc = false);

void FabricMulticastCommon(
    BaseFabricFixture* fixture,
    NocPacketType noc_packet_type,
    const std::vector<std::tuple<RoutingDirection, uint32_t /*start_distance*/, uint32_t /*range*/>>& dir_configs,
    bool with_state = false);

void Fabric2DMulticastCommon(
    BaseFabricFixture* fixture,
    NocPacketType noc_packet_type,
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
