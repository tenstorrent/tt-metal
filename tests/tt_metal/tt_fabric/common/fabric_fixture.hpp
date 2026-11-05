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
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "test_host_kernel_common.hpp"
#include "impl/kernels/kernel.hpp"

#include <algorithm>
#include <tuple>
#include <unordered_map>

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
           // Reset the process-wide fabric config to DISABLED so the next test starts clean. set_fabric_config is
           // one-shot per process (metal_env.cpp: non-DISABLED -> non-DISABLED is a TT_FATAL), and tearing down to
           // DISABLED is always allowed. Without this, a test that set e.g. FABRIC_2D would make a subsequent test
           // that sets a different config (TORUS_Y / MGD_DEFAULT / per-mesh heterogeneous) abort with
           // "Tried to override previous value of fabric config".
           tt::tt_metal::MetalContext::instance().set_fabric_config(tt::tt_fabric::FabricConfig::DISABLED);
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
    // The addresses above are the kernel's view: the L1 offset plus this. The host reads and writes the offset.
    uint32_t kernel_view_offset = 0;
};

// Where a fabric data-movement test runs a worker kernel. See BaseFabricFixture::get_fabric_test_worker().
struct FabricTestWorker {
    tt::tt_metal::CoreCoord logical;
    CoreType core_type;                   // WORKER or ETH: for CreateSemaphore and L1 reads
    tt::tt_metal::CoreCoord noc_core;     // the coordinate other cores address it at
};

class BaseFabricFixture : public ::testing::Test {
public:
    inline static tt::ARCH arch_;
    inline static std::map<ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_map_;
    inline static std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    inline static bool slow_dispatch_;

    std::unordered_map<tt::tt_metal::distributed::MeshDevice*, std::vector<tt::tt_metal::distributed::MeshWorkload>>
        pending_async_workloads_;

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
        // Fabric cannot be run if we have < 2 devices
        if (num_devices < 2) {
            log_info(tt::LogTest, "Skipping fabric tests as fabric requires > 2 devices, but only saw {} devices", num_devices);
            GTEST_SKIP();
        }

        std::vector<ChipId> ids;
        ids.reserve(num_devices);
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        tt::tt_fabric::SetFabricConfig(
            fabric_config, reliability_mode, num_routing_planes, fabric_tensix_config, fabric_udm_mode);
        const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().resolve_dispatch_core_config();
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
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device, tt::tt_metal::Program&& program) {
        pending_async_workloads_[device.get()].push_back(tt_metal::LaunchProgramAsync(*device, std::move(program)));
    }

    // NOLINTNEXTLINE(readability-make-member-function-const)
    void WaitForSingleProgramDone(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device) {
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        pending_async_workloads_.erase(device.get());
    }

    // Whether `device` has no Tensix cores, so fabric test workers run on its idle Ethernet cores (Keraunos CCEs):
    // the Keraunos test chips in tests/tt_metal/tt_fabric/keraunos/.
    static bool workers_on_ethernet_cores(const std::shared_ptr<tt_metal::distributed::MeshDevice>& device) {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        return cluster.get_soc_desc(device->get_device_ids()[0]).get_cores(CoreType::TENSIX, tt::CoordSystem::NOC0).empty();
    }

    // Where a fabric data-movement test runs a worker kernel on `device`: the Tensix core at `tensix_logical`, or, on a
    // chip whose workers run on Ethernet cores, its first idle Ethernet core in logical order. Fatal when that chip has
    // no idle Ethernet core.
    static FabricTestWorker get_fabric_test_worker(
        const std::shared_ptr<tt_metal::distributed::MeshDevice>& device, tt_metal::CoreCoord tensix_logical) {
        if (!workers_on_ethernet_cores(device)) {
            return {tensix_logical, CoreType::WORKER, device->worker_core_from_logical_core(tensix_logical)};
        }
        const ChipId chip = device->get_device_ids()[0];
        const auto idle = tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(chip);
        TT_FATAL(!idle.empty(), "Chip {} has no idle Ethernet core for a fabric test worker", chip);
        const tt_metal::CoreCoord logical = *std::ranges::min_element(
            idle, [](const auto& a, const auto& b) { return std::tie(a.y, a.x) < std::tie(b.y, b.x); });
        return {logical, CoreType::ETH, device->virtual_core_from_logical_core(logical, CoreType::ETH)};
    }

    // A data-movement kernel on `worker`: DataMovementConfig on RISCV_0 for a Tensix core, EthernetConfig on DM0 of
    // an idle Ethernet core.
    static tt_metal::KernelHandle create_fabric_test_worker_kernel(
        tt_metal::Program& program,
        const std::string& file_name,
        const FabricTestWorker& worker,
        const std::vector<uint32_t>& compile_args,
        const std::map<std::string, std::string>& defines = {}) {
        if (worker.core_type == CoreType::ETH) {
            return tt_metal::CreateKernel(
                program,
                file_name,
                worker.logical,
                tt_metal::EthernetConfig{
                    .eth_mode = tt_metal::Eth::IDLE,
                    .noc = tt_metal::NOC::NOC_0,
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .compile_args = compile_args,
                    .defines = defines});
        }
        return tt_metal::CreateKernel(
            program,
            file_name,
            worker.logical,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = defines});
    }

    // Utility function reused across tests to get address params. On a chip whose workers run on Ethernet cores the
    // map starts at the idle Ethernet core's unreserved L1, and the kernel's view is the CCE's uncached alias
    // (MEM_IERISC_UNCACHED_BASE): the worker's stores must reach SRAM, where the iDMA and the host read them.
    static WorkerMemMap generate_worker_mem_map(
        const std::shared_ptr<tt_metal::distributed::MeshDevice>& device, Topology /*topology*/) {
        constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
        constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
        constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;
        constexpr uint32_t KERAUNOS_CCE_UNCACHED_ALIAS = 0x800000;

        uint32_t base_addr = 0;
        uint32_t kernel_view_offset = 0;
        if (workers_on_ethernet_cores(device)) {
            const auto& hal = tt::tt_metal::MetalContext::instance().hal();
            base_addr = hal.get_dev_addr(
                tt_metal::HalProgrammableCoreType::IDLE_ETH, tt_metal::HalL1MemAddrType::UNRESERVED);
            const uint32_t size =
                hal.get_dev_size(tt_metal::HalProgrammableCoreType::IDLE_ETH, tt_metal::HalL1MemAddrType::UNRESERVED);
            TT_FATAL(
                PACKET_HEADER_RESERVED_BYTES + DATA_SPACE_RESERVED_BYTES + 2 * TEST_RESULTS_SIZE_BYTES <= size,
                "Idle Ethernet unreserved L1 ({} B) cannot hold the fabric test worker map",
                size);
            kernel_view_offset = KERAUNOS_CCE_UNCACHED_ALIAS;
        } else {
            base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
        }
        uint32_t source_l1_buffer_address = base_addr + kernel_view_offset + PACKET_HEADER_RESERVED_BYTES;
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
            TEST_RESULTS_SIZE_BYTES,
            kernel_view_offset};
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
private:
    inline static bool insufficient_devices_ = false;

protected:
    static void SetUpTestSuite() {
        insufficient_devices_ = tt::tt_metal::GetNumAvailableDevices() < 8;
        if (insufficient_devices_) {
            return;
        }
        Fabric2DUDMModeFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (!insufficient_devices_) {
            Fabric2DUDMModeFixture::TearDownTestSuite();
        }
    }

    void SetUp() override {
        if (insufficient_devices_) {
            GTEST_SKIP() << "Test requires at least 8 devices (2x4 mesh), found "
                         << tt::tt_metal::GetNumAvailableDevices();
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
    std::optional<std::vector<std::pair<tt::tt_metal::CoreCoord, tt::tt_metal::CoreCoord>>> worker_coords_list = std::nullopt,
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
