// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <hostdevcommon/common_values.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/control_plane.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests {

class ControlPlaneFixture : public ::testing::Test {
   protected:
       tt::ARCH arch_{tt::ARCH::Invalid};
       void SetUp() override {
           auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
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
    inline static std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_map_;
    inline static std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    inline static bool slow_dispatch_;

    const std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& get_devices() const { return devices_; }
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& get_device(chip_id_t id) const {
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
        tt_fabric::FabricTensixConfig fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED) {
        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch_) {
            log_info(tt::LogTest, "Running fabric api tests with slow dispatch");
        } else {
            log_info(tt::LogTest, "Running fabric api tests with fast dispatch");
        }
        // Set up all available devices
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> ids;
        ids.reserve(num_devices);
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        tt::tt_fabric::SetFabricConfig(
            fabric_config,
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
            num_routing_planes,
            fabric_tensix_config);
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

    void RunProgramNonblocking(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::LaunchProgram(device->get_devices()[0], program, false);
        } else {
            tt::tt_metal::distributed::MeshCommandQueue& cq = device->mesh_command_queue();
            // Create a mesh workload from the program
            auto& program_copy = program;
            auto mesh_workload = tt::tt_metal::distributed::CreateMeshWorkload();
            tt::tt_metal::distributed::AddProgramToMeshWorkload(
                mesh_workload,
                std::move(program_copy),
                tt::tt_metal::distributed::MeshCoordinateRange(
                    tt::tt_metal::distributed::MeshCoordinate(0, 0), tt::tt_metal::distributed::MeshCoordinate(0, 0)));
            tt::tt_metal::distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
        }
    }

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

class Fabric1DTensixFixture : public BaseFabricFixture {
private:
    inline static bool should_skip_ = false;

protected:
    static void SetUpTestSuite() {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() ==
                tt::tt_metal::ClusterType::GALAXY ||
            tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::TG) {
            should_skip_ = true;
            return;
        }
        BaseFabricFixture::DoSetUpTestSuite(
            tt::tt_fabric::FabricConfig::FABRIC_1D, std::nullopt, tt::tt_fabric::FabricTensixConfig::MUX);
    }
    static void TearDownTestSuite() {
        if (!should_skip_) {
            BaseFabricFixture::DoTearDownTestSuite();
        }
    }
    void SetUp() override {
        if (should_skip_) {
            GTEST_SKIP() << "Fabric1DTensixFixture tests are not supported on Galaxy systems";
        }
        BaseFabricFixture::SetUp();
    }
};

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

class NightlyFabric2DFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() { BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D); }
    static void TearDownTestSuite() { BaseFabricFixture::DoTearDownTestSuite(); }
};

class Fabric2DDynamicFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() { BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC); }
    static void TearDownTestSuite() { BaseFabricFixture::DoTearDownTestSuite(); }
};

class NightlyFabric2DDynamicFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() { BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC); }
    static void TearDownTestSuite() { BaseFabricFixture::DoTearDownTestSuite(); }
};

class CustomMeshGraphFabric2DDynamicFixture : public BaseFabricFixture {
public:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    void SetUp(
        const std::string& mesh_graph_desc_file,
        const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
        tt::tt_metal::MetalContext::instance().set_custom_fabric_topology(
            mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
        BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC);
    }

private:
    void SetUp() override {}

    void TearDown() override {
        BaseFabricFixture::DoTearDownTestSuite();
        tt::tt_metal::MetalContext::instance().set_default_fabric_topology();
    }
};

class T3kCustomMeshGraphFabric2DDynamicFixture
    : public CustomMeshGraphFabric2DDynamicFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::vector<std::vector<eth_coord_t>>>> {
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
    NOC_SEND_TYPE_LAST = NOC_UNICAST_SCATTER_WRITE
};

void FabricUnicastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::vector<std::tuple<RoutingDirection, uint32_t /*num_hops*/>>& dir_configs,
    bool with_state = false);

void FabricMulticastCommon(
    BaseFabricFixture* fixture,
    NocSendType noc_send_type,
    const std::vector<std::tuple<RoutingDirection, uint32_t /*start_distance*/, uint32_t /*range*/>>& dir_configs,
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

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
