// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Channel Trimming Capture Integration Tests
 *
 * Tests the host-side integration for channel trimming resource usage capture:
 * - Runtime option plumbing (rtoptions → L1 allocation → named CT args)
 * - L1 buffer allocation in FabricEriscDatamoverConfig
 * - FabricRouterDiagnosticBufferMap via FabricBuilderContext
 * - FabricDatapathUsageL1Results struct correctness
 *
 * Device-level tests that launch sender/receiver kernels, send traffic through
 * the fabric, and read back FabricDatapathUsageL1Results from eth core L1.
 * Tests use mesh coordinates (FabricNodeId) to select devices and the connection
 * API (get_forwarding_link_indices + append_fabric_connection_rt_args) for routing.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_trimming_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <llrt/tt_cluster.hpp>

#include "tt_metal/fabric/channel_trimming_export.hpp"
#include "tt_metal/fabric/channel_trimming_import.hpp"

#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// ============================================================================
// Type alias for the capture results struct used throughout these tests
// ============================================================================
using CaptureResults = FabricDatapathUsageL1Results<true, builder_config::num_max_receiver_channels, builder_config::num_max_sender_channels>;

// ============================================================================
// Helper: Result of reading capture data from a single ETH core
// ============================================================================
struct EthCoreCaptureResult {
    ChipId physical_chip_id;
    CoreCoord logical_eth_core;
    chan_id_t channel_id;
    CaptureResults capture;
};

// ============================================================================
// Helper: Read capture data from all active ETH cores on a device
// ============================================================================
std::vector<EthCoreCaptureResult> read_capture_from_all_eth_cores(
    BaseFabricFixture* fixture, ChipId physical_chip_id) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& builder_ctx = control_plane.get_fabric_context().get_builder_context();
    auto buffer_map = builder_ctx.get_telemetry_and_metadata_buffer_map();

    TT_FATAL(
        buffer_map.channel_trimming_capture.is_enabled(),
        "Channel trimming capture is not enabled — cannot read capture data");

    size_t capture_addr = buffer_map.channel_trimming_capture.l1_address;
    size_t capture_size = buffer_map.channel_trimming_capture.size_bytes;

    const auto logical_cores = control_plane.get_active_ethernet_cores(physical_chip_id);
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& chan_map = cluster.get_soc_desc(physical_chip_id).logical_eth_core_to_chan_map;

    const auto& device = fixture->get_device(physical_chip_id);

    std::vector<EthCoreCaptureResult> results;
    results.reserve(logical_cores.size());

    for (const auto& logical_core : logical_cores) {
        auto chan_it = chan_map.find(logical_core);
        if (chan_it == chan_map.end()) {
            continue;
        }
        chan_id_t channel_id = static_cast<chan_id_t>(chan_it->second);

        std::vector<uint32_t> raw_data;
        tt_metal::detail::ReadFromDeviceL1(
            device->get_devices()[0], logical_core, capture_addr, capture_size, raw_data, CoreType::ETH);

        CaptureResults capture{};
        std::memcpy(static_cast<void*>(&capture), raw_data.data(), std::min(capture_size, raw_data.size() * sizeof(uint32_t)));

        results.push_back(EthCoreCaptureResult{physical_chip_id, logical_core, channel_id, capture});
    }

    return results;
}

// ============================================================================
// Helper: Clear capture buffers on all active ETH cores for a device.
// Writes a zeroed CaptureResults struct (with min_packet_size sentinel 0xFFFF)
// to each core's L1 capture address, matching the device-side reset() behavior.
// ============================================================================
void clear_capture_on_device(BaseFabricFixture* fixture, ChipId physical_chip_id) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& builder_ctx = control_plane.get_fabric_context().get_builder_context();
    auto buffer_map = builder_ctx.get_telemetry_and_metadata_buffer_map();
    if (!buffer_map.channel_trimming_capture.is_enabled()) {
        return;
    }

    size_t capture_addr = buffer_map.channel_trimming_capture.l1_address;
    size_t capture_size = buffer_map.channel_trimming_capture.size_bytes;

    // Build a reset capture struct matching device-side reset()
    CaptureResults reset_data{};
    reset_data.sender_channel_min_packet_size_seen_bytes_by_vc.fill(0xFFFF);

    std::vector<uint32_t> raw(capture_size / sizeof(uint32_t), 0);
    std::memcpy(raw.data(), &reset_data, std::min(capture_size, sizeof(reset_data)));

    const auto logical_cores = control_plane.get_active_ethernet_cores(physical_chip_id);
    const auto& device = fixture->get_device(physical_chip_id);
    for (const auto& logical_core : logical_cores) {
        tt_metal::detail::WriteToDeviceL1(
            device->get_devices()[0], logical_core, capture_addr, raw, CoreType::ETH);
    }
}

// ============================================================================
// Helper: Clear capture buffers on ALL devices in the mesh.
// ============================================================================
void clear_all_capture_buffers(BaseFabricFixture* fixture) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_shape = control_plane.get_physical_mesh_shape(MeshId{0});
    for (uint32_t i = 0; i < mesh_shape.mesh_size(); i++) {
        ChipId phys = control_plane.get_physical_chip_id_from_fabric_node_id(FabricNodeId(MeshId{0}, i));
        clear_capture_on_device(fixture, phys);
    }
}

// ============================================================================
// Helper: Run the real exporter, import the YAML back, and verify that
// every capture read from L1 matches the imported data.
// ============================================================================
void verify_capture_roundtrip(const std::vector<EthCoreCaptureResult>& pre_export_captures) {
    if (pre_export_captures.empty()) {
        return;
    }

    // Run the real exporter (writes to {logs_dir}/generated/reports/channel_trimming_capture.yaml)
    export_channel_trimming_capture();

    // Determine the output path
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto yaml_path =
        std::filesystem::path(rtoptions.get_logs_dir()) / "generated" / "reports" / "channel_trimming_capture.yaml";
    ASSERT_TRUE(std::filesystem::exists(yaml_path)) << "Export YAML not found: " << yaml_path;

    // Import back
    auto imported = load_channel_trimming_overrides(yaml_path.string());

    // Compare: for each pre-export capture, the imported entry must match
    for (const auto& cap : pre_export_captures) {
        uint64_t key = make_override_key(cap.physical_chip_id, cap.channel_id);
        ASSERT_TRUE(imported.contains(key))
            << "Missing imported entry for chip=" << cap.physical_chip_id
            << " eth_chan=" << static_cast<int>(cap.channel_id);
        EXPECT_EQ(imported.at(key), cap.capture)
            << "Roundtrip mismatch for chip=" << cap.physical_chip_id
            << " eth_chan=" << static_cast<int>(cap.channel_id);
    }
}

// ============================================================================
// Helper: Result of running unicast traffic
// ============================================================================
struct UnicastTrafficResult {
    ChipId src_physical_device_id;
    ChipId dst_physical_device_id;
    FabricNodeId src_fabric_node_id;
    FabricNodeId dst_fabric_node_id;
    chan_id_t edm_port;
    bool skipped;
};

// ============================================================================
// Helper: Run unicast traffic between two fabric nodes using the connection API
// Follows the run_unicast_test_bw_chips pattern from test_basic_1d_fabric.cpp
// ============================================================================
UnicastTrafficResult run_unicast_traffic_bw_nodes(
    BaseFabricFixture* fixture,
    FabricNodeId src_fabric_node_id,
    FabricNodeId dst_fabric_node_id,
    uint32_t num_hops,
    uint32_t num_packets = 10) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    ChipId src_physical = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    ChipId dst_physical = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);

    // For multi-hop in 1D mode, append_fabric_connection_rt_args only works for direct
    // neighbors. Determine the first-hop neighbor and use it for the connection setup.
    // The sender kernel uses num_hops/dst_chip_id runtime args to forward further.
    auto forwarding_dir_opt = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    if (!forwarding_dir_opt.has_value()) {
        return UnicastTrafficResult{{}, {}, FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 0), 0, true};
    }

    // The connection destination is the first-hop neighbor (which equals dst for 1-hop)
    FabricNodeId connection_dst = dst_fabric_node_id;
    if (num_hops > 1) {
        auto first_hop_neighbors =
            control_plane.get_intra_chip_neighbors(src_fabric_node_id, *forwarding_dir_opt);
        if (first_hop_neighbors.empty()) {
            return UnicastTrafficResult{{}, {}, FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 0), 0, true};
        }
        connection_dst = FabricNodeId(src_fabric_node_id.mesh_id, first_hop_neighbors[0]);
    }

    const auto& available_links = get_forwarding_link_indices(src_fabric_node_id, connection_dst);
    if (available_links.empty()) {
        return UnicastTrafficResult{{}, {}, FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 0), 0, true};
    }
    uint32_t link_idx = available_links[0];

    auto dir_eth_chans =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, *forwarding_dir_opt);
    chan_id_t edm_port = dir_eth_chans[link_idx];

    auto sender_device = fixture->get_device(src_physical);
    auto receiver_device = fixture->get_device(dst_physical);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    auto worker_mem_map = BaseFabricFixture::generate_worker_mem_map(sender_device, topology);
    auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* use_dram_dst */,
        is_2d_fabric,
        0 /* is_chip_multicast */,
        0 /* additional_dir */};

    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    // Create sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        num_hops,
        1 /* fwd_range */,
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id};

    // Use connection API to append EDM connection args (to first-hop neighbor)
    append_fabric_connection_rt_args(
        src_fabric_node_id,
        connection_dst,
        link_idx,
        sender_program,
        sender_logical_core,
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create receiver program
    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch and wait
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate sender/receiver status
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device->get_devices()[0],
        receiver_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    return UnicastTrafficResult{
        src_physical, dst_physical, src_fabric_node_id, dst_fabric_node_id, edm_port, false};
}

// ============================================================================
// Helper: Walk the mesh to find a neighbor pair with the specified hop count.
// Scans all mesh positions and all directions (E/W/N/S) to find a source with
// at least `num_hops` neighbors. Returns the chain of nodes along the path.
// ============================================================================
struct MeshPathResult {
    FabricNodeId src_node;
    FabricNodeId dst_node;
    std::vector<FabricNodeId> path;  // src, hop1, hop2, ..., dst
    RoutingDirection direction;
    bool found;
};

MeshPathResult find_1d_path_with_hops(uint32_t num_hops) {
    // Walk the 1D chain from chip 0 using get_forwarding_direction to determine
    // the actual forward direction at each hop. In 1D mode, get_forwarding_link_indices
    // returns links for ANY destination (all traffic goes "forward"), so we can't use
    // it to distinguish neighbors. Instead, we ask for the forwarding direction towards
    // an unvisited chip, then get the neighbor in that direction.
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_shape = control_plane.get_physical_mesh_shape(MeshId{0});
    uint32_t num_chips = mesh_shape.mesh_size();

    FabricNodeId current(MeshId{0}, 0);
    std::vector<FabricNodeId> path = {current};

    for (uint32_t hop = 0; hop < num_hops; hop++) {
        // Find the furthest unvisited chip to use as a forwarding target.
        // Using the LAST unvisited chip avoids picking a direct neighbor, which would
        // cause get_forwarding_direction to return the 2D shortest-path direction
        // instead of the 1D chain forward direction.
        FabricNodeId target(MeshId{0}, 0);
        bool found_target = false;
        for (int i = num_chips - 1; i >= 0; i--) {
            bool visited = std::any_of(path.begin(), path.end(), [&](const FabricNodeId& n) {
                return n.chip_id == static_cast<uint32_t>(i);
            });
            if (!visited) {
                target = FabricNodeId(MeshId{0}, static_cast<uint32_t>(i));
                found_target = true;
                break;
            }
        }
        if (!found_target) {
            log_warning(tt::LogTest, "find_1d_path_with_hops: no unvisited chip at hop {}", hop);
            return MeshPathResult{current, current, {}, RoutingDirection::E, false};
        }

        // Get the forward direction from current towards the target
        auto fwd_dir = control_plane.get_forwarding_direction(current, target);
        if (!fwd_dir.has_value()) {
            log_warning(
                tt::LogTest,
                "find_1d_path_with_hops: no forwarding direction from chip_id={} to chip_id={} at hop {}",
                current.chip_id,
                target.chip_id,
                hop);
            return MeshPathResult{current, current, {}, RoutingDirection::E, false};
        }

        // Get the actual neighbor in the forward direction
        auto neighbors = control_plane.get_intra_chip_neighbors(current, *fwd_dir);
        if (neighbors.empty()) {
            log_warning(
                tt::LogTest,
                "find_1d_path_with_hops: no neighbor in direction {} from chip_id={} at hop {}",
                static_cast<int>(*fwd_dir),
                current.chip_id,
                hop);
            return MeshPathResult{current, current, {}, RoutingDirection::E, false};
        }

        FabricNodeId next(current.mesh_id, neighbors[0]);
        path.push_back(next);
        current = next;
    }

    return MeshPathResult{path.front(), path.back(), path, RoutingDirection::E, true};
}

// ============================================================================
// Helper: Find a src/dst pair where src has a forwarding channel in target_direction
// Uses mesh coordinates to scan all positions.
// ============================================================================
bool find_pair_for_direction(
    RoutingDirection target_direction,
    FabricNodeId& src_node,
    FabricNodeId& dst_node) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_shape = control_plane.get_physical_mesh_shape(MeshId{0});
    uint32_t num_chips = mesh_shape.mesh_size();

    for (uint32_t i = 0; i < num_chips; i++) {
        FabricNodeId candidate(MeshId{0}, i);
        auto neighbors = control_plane.get_intra_chip_neighbors(candidate, target_direction);
        if (neighbors.empty()) {
            continue;
        }

        FabricNodeId neighbor_node(MeshId{0}, neighbors[0]);
        auto links = get_forwarding_link_indices(candidate, neighbor_node);
        if (links.empty()) {
            continue;
        }

        src_node = candidate;
        dst_node = neighbor_node;
        return true;
    }

    return false;
}

// ============================================================================
// Fixture: Fabric1D with channel trimming capture enabled before fabric setup
// ============================================================================
class Fabric1DChannelTrimmingFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) {
            should_skip_ = true;
            return;
        }
        if (tt::tt_metal::GetNumAvailableDevices() < 4) {
            should_skip_ = true;
            return;
        }
        tt::tt_metal::MetalContext::instance().rtoptions().set_enable_channel_trimming_capture(true);
        BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_1D);
    }

    static void TearDownTestSuite() {
        if (!should_skip_) {
            BaseFabricFixture::DoTearDownTestSuite();
            tt::tt_metal::MetalContext::instance().rtoptions().set_enable_channel_trimming_capture(false);
        }
    }

    void SetUp() override {
        if (should_skip_) {
            GTEST_SKIP() << "Channel trimming tests require Blackhole architecture with at least 4 devices";
        }
        BaseFabricFixture::SetUp();
    }

    inline static bool should_skip_ = false;
};

// ============================================================================
// Tests using Fabric1DChannelTrimmingFixture (capture enabled at fabric setup)
// ============================================================================

// Test: Runtime option is enabled as set by the fixture
TEST_F(Fabric1DChannelTrimmingFixture, RuntimeOptionEnabled) {
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    EXPECT_TRUE(rtoptions.get_enable_channel_trimming_capture());
}

// Test: Config allocates L1 buffer when capture is enabled
TEST_F(Fabric1DChannelTrimmingFixture, ConfigAllocatesL1Buffer) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& config = control_plane.get_fabric_context().get_builder_context().get_fabric_router_config();

    EXPECT_NE(config.datapath_usage_l1_address, 0u);
    EXPECT_GT(config.datapath_usage_buffer_size, 0u);
    EXPECT_EQ(config.datapath_usage_buffer_size,
              sizeof(CaptureResults));
}

// Test: L1 allocation placed before handshake address
TEST_F(Fabric1DChannelTrimmingFixture, L1AllocationOrdering) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& config = control_plane.get_fabric_context().get_builder_context().get_fabric_router_config();

    // Capture buffer + its size should not exceed the handshake address
    size_t capture_end = config.datapath_usage_l1_address + config.datapath_usage_buffer_size;
    EXPECT_LE(capture_end, config.handshake_addr);
}

// Test: DiagnosticBufferMap shows channel_trimming_capture enabled via FabricBuilderContext
TEST_F(Fabric1DChannelTrimmingFixture, DiagnosticBufferMapViaBuilderContext) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& builder_ctx = control_plane.get_fabric_context().get_builder_context();
    auto buffer_map = builder_ctx.get_telemetry_and_metadata_buffer_map();

    EXPECT_TRUE(buffer_map.channel_trimming_capture.is_enabled());
    EXPECT_NE(buffer_map.channel_trimming_capture.l1_address, 0u);
    EXPECT_GT(buffer_map.channel_trimming_capture.size_bytes, 0u);
    EXPECT_EQ(buffer_map.channel_trimming_capture.size_bytes,
              sizeof(CaptureResults));
}

// Test: DiagnosticBufferMap regions are non-overlapping and ordered
TEST_F(Fabric1DChannelTrimmingFixture, DiagnosticBufferMapRegionsNonOverlapping) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& builder_ctx = control_plane.get_fabric_context().get_builder_context();
    auto buffer_map = builder_ctx.get_telemetry_and_metadata_buffer_map();

    // Collect all enabled regions
    std::vector<std::pair<size_t, size_t>> regions;  // (address, end)
    if (buffer_map.perf_telemetry.is_enabled()) {
        regions.emplace_back(
            buffer_map.perf_telemetry.l1_address,
            buffer_map.perf_telemetry.l1_address + buffer_map.perf_telemetry.size_bytes);
    }
    if (buffer_map.code_profiling.is_enabled()) {
        regions.emplace_back(
            buffer_map.code_profiling.l1_address,
            buffer_map.code_profiling.l1_address + buffer_map.code_profiling.size_bytes);
    }
    if (buffer_map.channel_trimming_capture.is_enabled()) {
        regions.emplace_back(
            buffer_map.channel_trimming_capture.l1_address,
            buffer_map.channel_trimming_capture.l1_address + buffer_map.channel_trimming_capture.size_bytes);
    }

    // Verify no overlaps: sort by address and check each end <= next start
    std::sort(regions.begin(), regions.end());
    for (size_t i = 1; i < regions.size(); i++) {
        EXPECT_LE(regions[i - 1].second, regions[i].first)
            << "Diagnostic buffer regions overlap at index " << i;
    }
}

// ============================================================================
// Tests using Fabric1DChannelTrimmingFixture (capture disabled — tests disabled path and
// pure struct logic that don't need the runtime option set at setup time)
// ============================================================================

// Test: Runtime option getter/setter interface
TEST_F(Fabric1DChannelTrimmingFixture, ChannelTrimmingCapture_RuntimeOptionGetterSetter) {
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    bool original = rtoptions.get_enable_channel_trimming_capture();

    rtoptions.set_enable_channel_trimming_capture(true);
    EXPECT_TRUE(rtoptions.get_enable_channel_trimming_capture());
    rtoptions.set_enable_channel_trimming_capture(false);
    EXPECT_FALSE(rtoptions.get_enable_channel_trimming_capture());

    rtoptions.set_enable_channel_trimming_capture(original);
}

// ============================================================================
// Fixture: Fabric2D with channel trimming capture enabled before fabric setup
// ============================================================================
class Fabric2DChannelTrimmingFixture : public BaseFabricFixture {
protected:
    static void SetUpTestSuite() {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) {
            should_skip_ = true;
            return;
        }
        if (tt::tt_metal::GetNumAvailableDevices() < 4) {
            should_skip_ = true;
            return;
        }
        tt::tt_metal::MetalContext::instance().rtoptions().set_enable_channel_trimming_capture(true);
        BaseFabricFixture::DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D);
    }

    static void TearDownTestSuite() {
        if (!should_skip_) {
            BaseFabricFixture::DoTearDownTestSuite();
            tt::tt_metal::MetalContext::instance().rtoptions().set_enable_channel_trimming_capture(false);
        }
    }

    void SetUp() override {
        if (should_skip_) {
            GTEST_SKIP() << "Channel trimming tests require Blackhole architecture with at least 4 devices";
        }
        BaseFabricFixture::SetUp();
    }

    inline static bool should_skip_ = false;
};

// ============================================================================
// Device-Level Tests: Fabric1DChannelTrimmingFixture
// ============================================================================

// Test 1: 1-hop unicast — verify both source and destination routers recorded traffic
TEST_F(Fabric1DChannelTrimmingFixture, UnicastSenderAndReceiverChannelUsed) {
    clear_all_capture_buffers(this);

    // Use mesh coordinates to find a 1-hop neighbor pair (any direction on the chain)
    auto path_result = find_1d_path_with_hops(1);
    if (!path_result.found) {
        GTEST_SKIP() << "No 1-hop neighbor pair found on the mesh";
    }

    log_info(
        tt::LogTest,
        "1-hop test: src mesh chip_id={} -> dst mesh chip_id={} in direction {}",
        path_result.src_node.chip_id,
        path_result.dst_node.chip_id,
        static_cast<int>(path_result.direction));

    auto result = run_unicast_traffic_bw_nodes(this, path_result.src_node, path_result.dst_node, 1);
    if (result.skipped) {
        GTEST_SKIP() << "No forwarding links between selected nodes";
    }

    // Source router: the eth core matching edm_port should show sender channel activity
    auto src_captures = read_capture_from_all_eth_cores(this, result.src_physical_device_id);
    int num_routers_with_sender_activity = 0;
    for (const auto& cap : src_captures) {
        if (cap.capture.sender_channel_used_bitfield_by_vc != 0) {
            num_routers_with_sender_activity++;
            // Sender channel validation on sender chip
            EXPECT_EQ(cap.capture.sender_channel_used_bitfield_by_vc, 1)
                << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should only show channel 0 (worker channel) activity";
            // Receiver channel validation on sender chip — no inbound traffic
            EXPECT_EQ(cap.capture.receiver_channel_data_forwarded_bitfield_by_vc, 0)
                << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should not show receiver channel forwarding activity since traffic is one-way";
            // NocSendType validation — source doesn't deliver locally
            for (size_t vc = 0; vc < 2; vc++) {
                EXPECT_EQ(cap.capture.used_noc_send_type_by_vc_bitfield[vc], 0)
                    << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                    << ") should not show NocSendType activity on VC " << vc << " since traffic is one-way outbound";
                EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[vc], 0)
                    << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                    << ") should not show forwarding-to on VC " << vc << " since traffic originates here (not forwarded from receiver)";
            }

        }
    }
    EXPECT_TRUE(num_routers_with_sender_activity == 1);

    // Destination router: the inbound eth core should show full receiver state
    auto dst_captures = read_capture_from_all_eth_cores(this, result.dst_physical_device_id);
    bool found_receiver_forwarding = false;
    int num_routers_with_receiver_activity = 0;
    for (const auto& cap : dst_captures) {
        if (cap.capture.receiver_channel_data_forwarded_bitfield_by_vc != 0) {
            num_routers_with_receiver_activity++;
            // Receiver channel validation on destination chip
            EXPECT_NE(cap.capture.receiver_channel_data_forwarded_bitfield_by_vc, 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should show receiver channel forwarding activity";
            // NocSendType validation — destination delivers packets locally via noc

            EXPECT_TRUE(cap.capture.used_noc_send_type_by_vc_bitfield[0] != 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") receiver channel 0 should show NocSendType activity since it delivers packets locally";
            EXPECT_TRUE(cap.capture.used_noc_send_type_by_vc_bitfield[1] == 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") receiver channel 1 should NOT show NocSendType activity since it delivers packets locally";
            // Sender channel validation — destination is 1-hop endpoint, should NOT forward
            EXPECT_EQ(cap.capture.sender_channel_used_bitfield_by_vc, 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should not show sender channel activity since it is the 1-hop endpoint";

            EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[0], 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") VC0 should not show forwarding-to on 1-hop endpoint";
            EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[1], 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") VC1 should not show forwarding-to (no VC1 traffic in 1D)";

            found_receiver_forwarding = true;
        }
    }
    EXPECT_EQ(num_routers_with_receiver_activity, 1);
    EXPECT_TRUE(found_receiver_forwarding) << "Expected at least one eth core on destination to show receiver "
                                              "channel data forwarding activity";

    // Roundtrip: export to YAML → import → compare
    auto all_captures = src_captures;
    all_captures.insert(all_captures.end(), dst_captures.begin(), dst_captures.end());
    verify_capture_roundtrip(all_captures);
}

// Test 2: 2-hop unicast — verify intermediate router forwarding
TEST_F(Fabric1DChannelTrimmingFixture, UnicastMultiHopForwarding) {
    clear_all_capture_buffers(this);

    // Use mesh coordinates to find a 2-hop path (scans all positions and directions)
    auto path_result = find_1d_path_with_hops(2);
    if (!path_result.found) {
        GTEST_SKIP() << "No 2-hop path found on the mesh (need >=3 devices in some direction)";
    }

    log_info(
        tt::LogTest,
        "2-hop test: src mesh chip_id={} -> intermediate chip_id={} -> dst mesh chip_id={} in direction {}",
        path_result.path[0].chip_id,
        path_result.path[1].chip_id,
        path_result.path[2].chip_id,
        static_cast<int>(path_result.direction));

    auto result = run_unicast_traffic_bw_nodes(this, path_result.src_node, path_result.dst_node, 2);
    if (result.skipped) {
        GTEST_SKIP() << "No forwarding links between selected nodes";
    }

    // Get the intermediate device's physical chip ID
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    FabricNodeId intermediate_node = path_result.path[1];
    ChipId intermediate_physical = control_plane.get_physical_chip_id_from_fabric_node_id(intermediate_node);

    ASSERT_NE(intermediate_physical, result.dst_physical_device_id)
        << "Intermediate should differ from destination in a 2-hop route";

    // --- Source router validation ---
    auto src_captures = read_capture_from_all_eth_cores(this, result.src_physical_device_id);
    int num_routers_with_sender_activity = 0;
    for (const auto& cap : src_captures) {
        if (cap.channel_id == result.edm_port) {
            num_routers_with_sender_activity++;
            // Sender channel validation — source originates traffic
            EXPECT_EQ(cap.capture.sender_channel_used_bitfield_by_vc, 1)
                << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should only show channel 0 (worker channel) activity";
            // Receiver channel validation — source has no inbound traffic
            EXPECT_EQ(cap.capture.receiver_channel_data_forwarded_bitfield_by_vc, 0)
                << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should not show receiver forwarding (traffic originates here)";
            for (size_t vc = 0; vc < 2; vc++) {
                EXPECT_EQ(cap.capture.used_noc_send_type_by_vc_bitfield[vc], 0)
                    << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                    << ") should not show NocSendType on VC " << vc << " (outbound only)";
                EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[vc], 0)
                    << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                    << ") should not show forwarding-to on VC " << vc << " (traffic originates here)";
            }
        }
    }
    EXPECT_TRUE(num_routers_with_sender_activity == 1) << "Expected to find exactly one router with sender activity on source device's edm_port";

    // --- Intermediate router validation ---
    auto intermediate_captures = read_capture_from_all_eth_cores(this, intermediate_physical);
    int num_routers_with_intermediate_sender_activity = 0;
    int num_routers_with_intermediate_receiver_activity = 0;
    bool sender_and_receiver_channels_active_on_different_routers = true;
    for (const auto& cap : intermediate_captures) {
        if (cap.capture.sender_channel_used_bitfield_by_vc != 0) {
            num_routers_with_intermediate_sender_activity++;
            // Intermediate forwards outbound — sender channel should be forwarding channel, not worker
            EXPECT_EQ(cap.capture.sender_channel_used_bitfield_by_vc & 1u, 0)
                << "Intermediate router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should NOT show worker channel (bit 0) activity — it only forwards";
            // NocSendType should be zero — intermediate doesn't deliver locally
            for (size_t vc = 0; vc < 2; vc++) {
                EXPECT_EQ(cap.capture.used_noc_send_type_by_vc_bitfield[vc], 0)
                    << "Intermediate router eth core (chan " << static_cast<int>(cap.channel_id)
                    << ") should not show NocSendType on VC " << vc << " (forwarding only, no local delivery)";
            }
        }
        if (cap.capture.receiver_channel_data_forwarded_bitfield_by_vc != 0) {
            num_routers_with_intermediate_receiver_activity++;
            // Intermediate receives and forwards — forwarding relationship should be recorded
            EXPECT_NE(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[0], 0)
                << "Intermediate router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") VC0 should show forwarding-to (receiver forwards to sender channel)";
        }
        sender_and_receiver_channels_active_on_different_routers = sender_and_receiver_channels_active_on_different_routers &&
            !((cap.capture.receiver_channel_data_forwarded_bitfield_by_vc != 0) && (cap.capture.sender_channel_used_bitfield_by_vc != 0));
    }
    EXPECT_EQ(num_routers_with_intermediate_sender_activity, 1) << "Expected to find exactly one router with sender activity on intermediate device";
    EXPECT_EQ(num_routers_with_intermediate_receiver_activity, 1) << "Expected to find exactly one router with receiver activity on intermediate device";
    EXPECT_TRUE(sender_and_receiver_channels_active_on_different_routers) << "Sender and receiver channels should be active on different routers";

    // --- Destination router validation ---
    auto dst_captures = read_capture_from_all_eth_cores(this, result.dst_physical_device_id);
    int num_routers_with_dst_receiver_activity = 0;
    for (const auto& cap : dst_captures) {
        if (cap.capture.receiver_channel_data_forwarded_bitfield_by_vc != 0) {
            num_routers_with_dst_receiver_activity++;
            // Receiver channel validation — destination receives and delivers locally
            EXPECT_NE(cap.capture.receiver_channel_data_forwarded_bitfield_by_vc, 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should show receiver channel forwarding activity";
            // NocSendType validation — delivers locally
            bool has_noc_send_type = false;
            for (size_t vc = 0; vc < 2; vc++) {
                if (cap.capture.used_noc_send_type_by_vc_bitfield[vc] != 0) {
                    has_noc_send_type = true;
                }
            }
            EXPECT_TRUE(has_noc_send_type)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should show NocSendType activity (local delivery)";
            // Sender channel validation — destination is endpoint, no sender transmission
            EXPECT_EQ(cap.capture.sender_channel_used_bitfield_by_vc, 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should not show sender channel activity (2-hop endpoint)";

            EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[0], 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") VC0 should not show forwarding-to on 2-hop endpoint";
            EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[1], 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") VC1 should not show forwarding-to (no VC1 traffic in 1D)";
        }
    }
    EXPECT_TRUE(num_routers_with_dst_receiver_activity == 1) << "Expected to find exactly one router with receiver activity on destination device";

    // Roundtrip: export to YAML → import → compare
    auto all_captures = src_captures;
    all_captures.insert(all_captures.end(), intermediate_captures.begin(), intermediate_captures.end());
    all_captures.insert(all_captures.end(), dst_captures.begin(), dst_captures.end());
    verify_capture_roundtrip(all_captures);
}

// Test 3: 1-hop unicast — verify min/max packet size tracking
TEST_F(Fabric1DChannelTrimmingFixture, UnicastPacketSizeTracking) {
    clear_all_capture_buffers(this);

    auto path_result = find_1d_path_with_hops(1);
    if (!path_result.found) {
        GTEST_SKIP() << "No 1-hop neighbor pair found on the mesh";
    }

    auto result = run_unicast_traffic_bw_nodes(this, path_result.src_node, path_result.dst_node, 1);
    if (result.skipped) {
        GTEST_SKIP() << "No forwarding links between selected nodes";
    }

    // --- Source router: packet size tracking ---
    auto src_captures = read_capture_from_all_eth_cores(this, result.src_physical_device_id);

    for (const auto& cap : src_captures) {
        if (cap.channel_id != result.edm_port) {
            continue;
        }

        uint16_t used_bitfield = cap.capture.sender_channel_used_bitfield_by_vc;
        ASSERT_NE(used_bitfield, 0) << "Expected sender channel activity on edm_port "
                                    << static_cast<int>(result.edm_port);
        EXPECT_EQ(used_bitfield, 1)
            << "Source should only show channel 0 (worker channel) activity";

        // Full state: source is outbound only
        EXPECT_EQ(cap.capture.receiver_channel_data_forwarded_bitfield_by_vc, 0)
            << "Source router should not show receiver forwarding (1-hop outbound)";
        for (size_t vc = 0; vc < 2; vc++) {
            EXPECT_EQ(cap.capture.used_noc_send_type_by_vc_bitfield[vc], 0)
                << "Source router should not show NocSendType on VC " << vc << " (outbound only)";
            EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[vc], 0)
                << "Source router should not show forwarding-to on VC " << vc << " (traffic originates here)";
        }

        // Check each sender channel that has its bit set — packet size invariants
        for (size_t ch = 0; ch < builder_config::num_max_sender_channels; ch++) {
            if (!(used_bitfield & (1u << ch))) {
                continue;
            }
            uint16_t min_pkt = cap.capture.sender_channel_min_packet_size_seen_bytes_by_vc[ch];
            uint16_t max_pkt = cap.capture.sender_channel_max_packet_size_seen_bytes_by_vc[ch];

            EXPECT_NE(min_pkt, 0xFFFF)
                << "Sender channel " << ch << ": min_packet_size should be updated from sentinel 0xFFFF";
            EXPECT_NE(max_pkt, 0u) << "Sender channel " << ch << ": max_packet_size should be updated from 0";
            EXPECT_LE(min_pkt, max_pkt)
                << "Sender channel " << ch << ": min_packet_size (" << min_pkt
                << ") should be <= max_packet_size (" << max_pkt << ")";
        }
        break;
    }

    // --- Destination router: full state validation ---
    auto dst_captures = read_capture_from_all_eth_cores(this, result.dst_physical_device_id);
    bool found_dst_receiver = false;
    for (const auto& cap : dst_captures) {
        if (cap.capture.receiver_channel_data_forwarded_bitfield_by_vc != 0) {
            // Receiver delivers locally
            bool has_noc_send_type = false;
            for (size_t vc = 0; vc < 2; vc++) {
                if (cap.capture.used_noc_send_type_by_vc_bitfield[vc] != 0) {
                    has_noc_send_type = true;
                }
            }
            EXPECT_TRUE(has_noc_send_type)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should show NocSendType activity (local delivery)";
            // Endpoint — no sender transmission (receiver delivers locally, doesn't originate traffic)
            EXPECT_EQ(cap.capture.sender_channel_used_bitfield_by_vc, 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should not show sender channel activity (endpoint)";
            EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[0], 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") VC0 should not show forwarding-to on 1-hop endpoint";
            EXPECT_EQ(cap.capture.sender_channel_forwarded_to_bitfield_by_vc[1], 0)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") VC1 should not show forwarding-to (no VC1 traffic in 1D)";
            found_dst_receiver = true;
        }
    }
    EXPECT_TRUE(found_dst_receiver)
        << "Destination router should show receiver channel forwarding activity";

    // Roundtrip: export to YAML → import → compare
    auto all_captures = src_captures;
    all_captures.insert(all_captures.end(), dst_captures.begin(), dst_captures.end());
    verify_capture_roundtrip(all_captures);
}

// ============================================================================
// Device-Level Tests: Fabric2DChannelTrimmingFixture
// ============================================================================

// Test 4: Exercise each directional channel (E/W/N/S)
TEST_F(Fabric2DChannelTrimmingFixture, DirectionalChannelLiveness) {
    clear_all_capture_buffers(this);

    std::vector<RoutingDirection> directions = {
        RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S};
    std::vector<RoutingDirection> exercised_directions;

    for (auto direction : directions) {
        FabricNodeId src_node(MeshId{0}, 0);
        FabricNodeId dst_node(MeshId{0}, 0);

        if (!find_pair_for_direction(direction, src_node, dst_node)) {
            log_info(
                tt::LogTest,
                "No device pair found for direction {} — skipping this direction",
                static_cast<int>(direction));
            continue;
        }

        auto result = run_unicast_traffic_bw_nodes(this, src_node, dst_node, 1, /*num_packets=*/10);
        if (result.skipped) {
            log_info(
                tt::LogTest,
                "Traffic run skipped for direction {} — no forwarding links",
                static_cast<int>(direction));
            continue;
        }

        // --- Source device: validate sender channel activity on the edm_port ---
        // NOTE: We only check positive conditions (sender was used). Negative conditions
        // (receiver/noc_send_type should be zero) are unreliable because the firmware only
        // resets capture data at kernel_main() startup, and across loop iterations the same
        // device may play different roles (source vs destination), accumulating stale data.
        auto src_captures = read_capture_from_all_eth_cores(this, result.src_physical_device_id);
        bool found_sender = false;
        for (const auto& cap : src_captures) {
            if (cap.channel_id == result.edm_port) {
                EXPECT_NE(cap.capture.sender_channel_used_bitfield_by_vc, 0)
                    << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                    << ", direction " << static_cast<int>(direction) << ") should show sender channel activity";
                // Worker channel (bit 0) should be set
                EXPECT_NE(cap.capture.sender_channel_used_bitfield_by_vc & 1, 0)
                    << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                    << ", direction " << static_cast<int>(direction) << ") should show worker channel (0) activity";
                found_sender = true;
            }
        }

        // --- Destination device: validate receiver channel activity ---
        // NOTE: In 2D, sender_channel_forwarded_to_bitfield_by_vc[0] is 0 at the terminal
        // destination because hop_cmd_to_sender_channel_mask masks out my_direction,
        // resulting in fwd_mask=0. Also, sender_channel_used may be non-zero if this
        // router also forwards in other directions. We only check positive conditions.
        auto dst_captures = read_capture_from_all_eth_cores(this, result.dst_physical_device_id);
        bool found_receiver = false;
        for (const auto& cap : dst_captures) {
            if (cap.capture.receiver_channel_data_forwarded_bitfield_by_vc != 0) {
                bool has_noc_send_type = false;
                for (size_t vc = 0; vc < 2; vc++) {
                    if (cap.capture.used_noc_send_type_by_vc_bitfield[vc] != 0) {
                        has_noc_send_type = true;
                    }
                }
                EXPECT_TRUE(has_noc_send_type)
                    << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                    << ", direction " << static_cast<int>(direction)
                    << ") should show NocSendType activity (local delivery)";
                found_receiver = true;
            }
        }

        if (found_sender && found_receiver) {
            exercised_directions.push_back(direction);
            log_info(
                tt::LogTest,
                "Direction {} successfully exercised (src mesh chip_id={}, dst mesh chip_id={})",
                static_cast<int>(direction),
                src_node.chip_id,
                dst_node.chip_id);

            // Roundtrip: export to YAML → import → compare
            auto all_captures = src_captures;
            all_captures.insert(all_captures.end(), dst_captures.begin(), dst_captures.end());
            verify_capture_roundtrip(all_captures);
        }
    }

    // On any multi-device system, at least 2 directions should be exercisable
    EXPECT_GE(exercised_directions.size(), 2u)
        << "Expected at least 2 directions to be exercised; only got " << exercised_directions.size();
}

// Test 5: Verify forwarding relationships in 2D (sender_channel_forwarded_to_bitfield)
TEST_F(Fabric2DChannelTrimmingFixture, DirectionalChannelForwardedTo) {
    clear_all_capture_buffers(this);

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_shape = control_plane.get_physical_mesh_shape(MeshId{0});
    uint32_t num_chips = mesh_shape.mesh_size();

    if (num_chips < 4) {
        GTEST_SKIP() << "Need at least 4 devices for multi-hop 2D routing test";
    }

    // Try to find two devices that are not direct neighbors (want multi-hop)
    FabricNodeId src_node(MeshId{0}, 0);
    FabricNodeId dst_node(MeshId{0}, 0);
    bool found_pair = false;

    for (uint32_t i = 0; i < num_chips && !found_pair; i++) {
        FabricNodeId candidate_src(MeshId{0}, i);

        for (uint32_t j = 0; j < num_chips; j++) {
            if (i == j) {
                continue;
            }
            FabricNodeId candidate_dst(MeshId{0}, j);

            auto links = get_forwarding_link_indices(candidate_src, candidate_dst);
            if (links.empty()) {
                continue;
            }

            // Check this isn't a direct neighbor (want multi-hop)
            // get_intra_chip_neighbors returns mesh chip_ids, so compare against j (mesh chip_id)
            bool is_direct_neighbor = false;
            for (auto dir :
                 {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S}) {
                auto neighbors = control_plane.get_intra_chip_neighbors(candidate_src, dir);
                for (auto n : neighbors) {
                    if (n == j) {
                        is_direct_neighbor = true;
                        break;
                    }
                }
                if (is_direct_neighbor) {
                    break;
                }
            }

            if (!is_direct_neighbor) {
                src_node = candidate_src;
                dst_node = candidate_dst;
                found_pair = true;
                break;
            }
        }
    }

    if (!found_pair) {
        // Fall back to any src/dst pair with forwarding links
        for (uint32_t i = 0; i < num_chips && !found_pair; i++) {
            for (uint32_t j = 0; j < num_chips; j++) {
                if (i == j) {
                    continue;
                }
                FabricNodeId candidate_src(MeshId{0}, i);
                FabricNodeId candidate_dst(MeshId{0}, j);
                auto links = get_forwarding_link_indices(candidate_src, candidate_dst);
                if (!links.empty()) {
                    src_node = candidate_src;
                    dst_node = candidate_dst;
                    found_pair = true;
                    break;
                }
            }
        }
    }

    if (!found_pair) {
        GTEST_SKIP() << "No valid src/dst pair with forwarding links found";
    }

    auto result = run_unicast_traffic_bw_nodes(this, src_node, dst_node, 1, /*num_packets=*/10);
    if (result.skipped) {
        GTEST_SKIP() << "No forwarding channels between selected src/dst pair";
    }

    // --- Source device: validate sender channel activity ---
    auto src_captures = read_capture_from_all_eth_cores(this, result.src_physical_device_id);
    bool found_src_sender = false;
    bool found_forwarded_to = false;
    for (const auto& cap : src_captures) {
        if (cap.channel_id == result.edm_port) {
            EXPECT_NE(cap.capture.sender_channel_used_bitfield_by_vc, 0)
                << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should show sender channel activity";
            EXPECT_NE(cap.capture.sender_channel_used_bitfield_by_vc & 1, 0)
                << "Source router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should show worker channel (0) activity";
            found_src_sender = true;
        }

        // Check forwarded_to on any eth core (for multi-hop intermediate routers on src device)
        for (size_t vc = 0; vc < 2; vc++) {
            if (cap.capture.sender_channel_forwarded_to_bitfield_by_vc[vc] != 0) {
                found_forwarded_to = true;
                log_info(
                    tt::LogTest,
                    "Source chip {} eth core chan {}: sender_channel_forwarded_to_bitfield_by_vc[{}] = 0x{:04x}",
                    cap.physical_chip_id,
                    static_cast<int>(cap.channel_id),
                    vc,
                    cap.capture.sender_channel_forwarded_to_bitfield_by_vc[vc]);
            }
        }
    }
    EXPECT_TRUE(found_src_sender)
        << "Expected to find sender activity on source device's edm_port " << static_cast<int>(result.edm_port);

    // --- Destination device: validate receiver channel activity ---
    // NOTE: In 2D, forwarded_to[0] is 0 at the terminal destination because
    // hop_cmd_to_sender_channel_mask masks out my_direction (local write).
    // sender_channel_used may also be non-zero if this router forwards in other directions.
    auto dst_captures = read_capture_from_all_eth_cores(this, result.dst_physical_device_id);
    bool found_receiver_forwarding = false;
    for (const auto& cap : dst_captures) {
        if (cap.capture.receiver_channel_data_forwarded_bitfield_by_vc != 0) {
            bool has_noc_send_type = false;
            for (size_t vc = 0; vc < 2; vc++) {
                if (cap.capture.used_noc_send_type_by_vc_bitfield[vc] != 0) {
                    has_noc_send_type = true;
                }
            }
            EXPECT_TRUE(has_noc_send_type)
                << "Destination router eth core (chan " << static_cast<int>(cap.channel_id)
                << ") should show NocSendType activity (local delivery)";
            found_receiver_forwarding = true;
        }
    }

    // At least one of forwarded_to or receiver forwarding should be set
    EXPECT_TRUE(found_forwarded_to || found_receiver_forwarding)
        << "Expected forwarding relationship to be recorded either as sender_channel_forwarded_to on source "
           "or receiver_channel_data_forwarded on destination";

    // Roundtrip: export to YAML → import → compare
    auto all_captures = src_captures;
    all_captures.insert(all_captures.end(), dst_captures.begin(), dst_captures.end());
    verify_capture_roundtrip(all_captures);
}

}  // namespace tt::tt_fabric::fabric_router_tests
