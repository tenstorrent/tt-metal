// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <optional>
#include <random>
#include <string>
#include <cstdlib>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/system_mesh.hpp>
#include "tt_align.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common_types.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/impl/dispatch/hardware_command_queue.hpp"

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using MeshShape = tt::tt_metal::distributed::MeshShape;
using MeshWorkload = tt::tt_metal::distributed::MeshWorkload;
using MeshCoordinateRange = tt::tt_metal::distributed::MeshCoordinateRange;
using DeviceLocalBufferConfig = tt::tt_metal::distributed::DeviceLocalBufferConfig;
using MeshBufferConfig = tt::tt_metal::distributed::MeshBufferConfig;
using ReplicatedBufferConfig = tt::tt_metal::distributed::ReplicatedBufferConfig;
using MeshBuffer = tt::tt_metal::distributed::MeshBuffer;
using BufferDistributionSpec = tt::tt_metal::BufferDistributionSpec;
using Shape = tt::tt_metal::Shape;
using MeshHostRankId = tt::tt_fabric::MeshHostRankId;
using SystemMesh = tt::tt_metal::distributed::SystemMesh;
using MeshDeviceConfig = tt::tt_metal::distributed::MeshDeviceConfig;
using HalProgrammableCoreType = tt::tt_metal::HalProgrammableCoreType;
using HalL1MemAddrType = tt::tt_metal::HalL1MemAddrType;
using ShardOrientation = tt::tt_metal::ShardOrientation;
using ShardSpecBuffer = tt::tt_metal::ShardSpecBuffer;
using BufferType = tt::tt_metal::BufferType;
using TensorMemoryLayout = tt::tt_metal::TensorMemoryLayout;
using BufferShardingArgs = tt::tt_metal::BufferShardingArgs;

using Topology = tt::tt_fabric::Topology;

namespace tt::tt_fabric::fabric_tests {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // Simple mixing
        return h1 ^ (h2 << 1);
    }
};

struct tuple_hash {
    template <class T1, class T2, class T3>
    std::size_t operator()(const std::tuple<T1, T2, T3>& t) const {
        auto h1 = std::hash<T1>{}(std::get<0>(t));
        auto h2 = std::hash<T2>{}(std::get<1>(t));
        auto h3 = std::hash<T3>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class TestFixture : public IDeviceInfoProvider, public IRouteManager, public IDistributedContextManager {
    static constexpr uint32_t ROW_DIM = 0;
    static constexpr uint32_t COL_DIM = 1;

    // mapping to convert coords to directions
    static constexpr uint32_t EW_DIM = 1;
    static constexpr uint32_t NS_DIM = 0;

    static const std::unordered_map<Topology, tt::tt_fabric::FabricConfig> topology_to_fabric_config_map;

    static const std::unordered_map<std::pair<Topology, std::string>, tt::tt_fabric::FabricConfig, pair_hash>
        torus_topology_to_fabric_config_map;

public:
    bool validate_device_frequencies_for_performance_tests() {
        // Skip if already validated - device frequencies are cached for the lifetime of the fixture
        if (frequency_validated_) {
            return true;
        }

        uint32_t expected_freq = get_expected_baseline_frequency_mhz();
        uint32_t tolerance = get_frequency_tolerance_mhz();

        std::vector<std::pair<FabricNodeId, uint32_t>> failed_devices;
        for (const auto& device_id : get_global_node_ids()) {
            uint32_t actual_freq = get_device_frequency_mhz(device_id);
            int32_t diff = static_cast<int32_t>(actual_freq) - static_cast<int32_t>(expected_freq);
            if (std::abs(diff) > static_cast<int32_t>(tolerance)) {
                failed_devices.emplace_back(device_id, actual_freq);
            }
        }

        if (!failed_devices.empty()) {
            log_error(tt::LogTest, "=== DEVICE FREQUENCY VALIDATION FAILED ===");
            log_error(tt::LogTest, "Cannot run performance benchmarks - frequency mismatch detected");
            log_error(tt::LogTest, "Expected: {}MHz ± {}MHz", expected_freq, tolerance);
            log_error(tt::LogTest, "Failed devices ({} total):", failed_devices.size());
            for (const auto& [dev_id, actual] : failed_devices) {
                int32_t diff = static_cast<int32_t>(actual) - static_cast<int32_t>(expected_freq);
                log_error(tt::LogTest, "  Device {}: {}MHz (diff: {:+}MHz)", dev_id, actual, diff);
            }
            return false;
        }

        log_info(
            tt::LogTest,
            "Device frequency validation passed: {} devices at {}MHz ± {}MHz",
            get_global_node_ids().size(),
            expected_freq,
            tolerance);
        frequency_validated_ = true;
        return true;
    }

    void init(std::optional<PhysicalMeshConfig> physical_mesh_config = std::nullopt) {
        if (physical_mesh_config.has_value()) {
            initialize_and_validate_custom_physical_config(physical_mesh_config.value());
        }

        // to ensure fabric config is set first, which affects mesh graph descriptor selection
        current_fabric_config_ = tt::tt_fabric::FabricConfig::DISABLED;
    }

    MeshCoordinateRange get_host_local_device_coordinates() const {
        return tt::tt_metal::MetalContext::instance().get_control_plane().get_coord_range(
            local_mesh_id_, MeshScope::LOCAL);
    }

    bool open_devices(const TestFabricSetup& fabric_setup) {
        const auto& topology = fabric_setup.topology;
        const auto& fabric_tensix_config = fabric_setup.fabric_tensix_config.value();

        // Fabric Reliability Mode
        // Default to STRICT; if runtime option (from rtoptions) is set, it takes precedence over
        // fabric_setup.fabric_reliability_mode
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
        auto reliability_mode_override = tt::tt_metal::MetalContext::instance().rtoptions().get_reliability_mode();
        if (reliability_mode_override.has_value()) {
            reliability_mode = reliability_mode_override.value();
        } else if (fabric_setup.fabric_reliability_mode.has_value()) {
            reliability_mode = fabric_setup.fabric_reliability_mode.value();
        }

        FabricConfig new_fabric_config;
        if (topology == Topology::Torus) {
            const auto& torus_config = fabric_setup.torus_config.value();
            auto it = torus_topology_to_fabric_config_map.find({topology, torus_config});
            TT_FATAL(
                it != torus_topology_to_fabric_config_map.end(),
                "Unsupported topology: {} with torus_config: {}",
                topology,
                torus_config);
            new_fabric_config = it->second;
        } else {
            auto it = topology_to_fabric_config_map.find(topology);
            TT_FATAL(it != topology_to_fabric_config_map.end(), "Unsupported topology: {}", topology);
            new_fabric_config = it->second;
        }

        // Use the new ControlPlane validation API - always skip on mismatch
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        try {
            if (cluster.get_cluster_type() == tt::tt_metal::ClusterType::GALAXY &&
                !control_plane.is_fabric_config_valid(new_fabric_config)) {
                log_warning(tt::LogTest, "Fabric configuration validation failed - can't open device");
                return false;
            }
        } catch (const std::exception& e) {
            log_warning(tt::LogTest, "Fabric configuration validation failed: {} - can't open device", e.what());
            return false;
        }

        if (new_fabric_config != current_fabric_config_ || fabric_tensix_config != current_fabric_tensix_config_ ||
            reliability_mode != current_fabric_reliability_mode_ ||
            fabric_setup.max_packet_size != current_max_packet_size_) {
            if (are_devices_open_) {
                log_info(tt::LogTest, "Closing devices and switching to new fabric config: {}", new_fabric_config);
                close_devices();
            }
            log_info(tt::LogTest, "Opening devices with fabric reliability mode: {}", reliability_mode);
            open_devices_internal(new_fabric_config, fabric_tensix_config, reliability_mode, fabric_setup);

            topology_ = topology;
        } else {
            log_info(tt::LogTest, "Reusing existing device setup with fabric config: {}", current_fabric_config_);
        }
        return true;
    }

    void setup_workload() {
        // create a new mesh workload for every run
        mesh_workload_ = std::make_unique<MeshWorkload>();
    }

    void enqueue_program(const MeshCoordinate& mesh_coord, tt::tt_metal::Program program) {
        MeshCoordinateRange device(mesh_coord, mesh_coord);
        mesh_workload_->add_program(device, std::move(program));
    }

    void run_programs() {
        if (mesh_workload_->get_programs().empty()) {
            return;
        }

        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *mesh_workload_, false);
    }

    void wait_for_programs() { tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue()); }

    void close_devices() {
        if (!are_devices_open_) {
            log_info(tt::LogTest, "Devices are already closed, skipping close_devices call");
            return;
        }
        mesh_device_->close();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);

        // Clear all class members
        local_available_node_ids_.clear();
        global_available_node_ids_.clear();
        available_mesh_ids_.clear();
        mesh_device_.reset();
        mesh_workload_.reset();
        current_fabric_config_ = tt::tt_fabric::FabricConfig::DISABLED;
        current_fabric_tensix_config_ = tt_fabric::FabricTensixConfig::DISABLED;
        current_fabric_reliability_mode_ = tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
        current_max_packet_size_ = std::nullopt;
        are_devices_open_ = false;
    }

    // ======================================================================================
    // IDeviceInfoProvider methods
    // ======================================================================================
    FabricNodeId get_fabric_node_id(const ChipId physical_chip_id) const override {
        return tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_node_id_from_physical_chip_id(
            physical_chip_id);
    }

    FabricNodeId get_fabric_node_id(const MeshCoordinate& device_coord) const override {
        const auto& mesh_graph = tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_graph();
        return FabricNodeId(local_mesh_id_, mesh_graph.coordinate_to_chip(local_mesh_id_, device_coord));
    }

    FabricNodeId get_fabric_node_id(MeshId mesh_id, const MeshCoordinate& device_coord) const override {
        const auto& mesh_graph = tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_graph();
        return FabricNodeId(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, device_coord));
    }

    MeshCoordinate get_device_coord(const FabricNodeId& node_id) const override {
        const auto& mesh_graph = tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_graph();
        return mesh_graph.chip_to_coordinate(node_id.mesh_id, node_id.chip_id);
    }

    uint32_t get_worker_noc_encoding(const CoreCoord logical_core) const override {
        const auto virtual_core = mesh_device_->worker_core_from_logical_core(logical_core);
        return tt_metal::MetalContext::instance().hal().noc_xy_encoding(virtual_core.x, virtual_core.y);
    }

    CoreCoord get_virtual_core_from_logical_core(CoreCoord logical_core) const override {
        return mesh_device_->worker_core_from_logical_core(logical_core);
    }

    CoreCoord get_worker_grid_size() const override { return mesh_device_->compute_with_storage_grid_size(); }

    uint32_t get_worker_id(const FabricNodeId& node_id, CoreCoord logical_core) const override {
        return (*node_id.mesh_id << 12) | (node_id.chip_id << 8) | (logical_core.x << 4) | (logical_core.y);
    }

    std::vector<FabricNodeId> get_local_node_ids() const override { return local_available_node_ids_; }

    std::vector<FabricNodeId> get_global_node_ids() const override { return global_available_node_ids_; }

    bool is_local_fabric_node_id(const FabricNodeId& id) const override {
        return std::find(local_available_node_ids_.begin(), local_available_node_ids_.end(), id) !=
               local_available_node_ids_.end();
    };

    uint32_t get_l1_unreserved_base() const override {
        return tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
    }

    uint32_t get_l1_unreserved_size() const override { return tt::tt_metal::hal::get_max_worker_l1_unreserved_size(); }

    uint32_t get_l1_alignment() const override { return tt::tt_metal::hal::get_l1_alignment(); }

    uint32_t get_max_payload_size_bytes() const override {
        return tt::tt_metal::MetalContext::instance()
            .get_control_plane()
            .get_fabric_context()
            .get_fabric_max_payload_size_bytes();
    }

    bool is_2D_routing_enabled() const override {
        return tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().is_2D_routing_enabled();
    }

    uint32_t get_device_frequency_mhz(const FabricNodeId& device_id) const override {
        auto cached = device_frequency_cache_.find(device_id);
        if (cached != device_frequency_cache_.end()) {
            return cached->second;
        }
        auto& metal_context = tt::tt_metal::MetalContext::instance();
        auto physical_chip_id = metal_context.get_control_plane().get_physical_chip_id_from_fabric_node_id(device_id);
        auto freq = metal_context.get_cluster().get_device_aiclk(physical_chip_id);
        device_frequency_cache_.emplace(device_id, freq);
        return freq;
    }

    bool is_multi_mesh() const override {
        const auto& mesh_graph = tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_graph();
        return mesh_graph.get_mesh_ids().size() > 1;
    }

    std::unordered_map<MeshId, std::unordered_set<MeshId>> get_mesh_adjacency_map() const override {
        std::unordered_map<MeshId, std::unordered_set<MeshId>> mesh_adjacency_map;
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto& global_nodes = get_global_node_ids();
        const std::vector<RoutingDirection> directions = {
            RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W};

        for (const auto& src_node : global_nodes) {
            MeshId src_mesh_id = src_node.mesh_id;
            for (const auto& direction : directions) {
                const auto& neighbors = control_plane.get_chip_neighbors(src_node, direction);
                for (const auto& [neighbor_mesh_id, neighbor_chips] : neighbors) {
                    if (neighbor_mesh_id != src_mesh_id && !neighbor_chips.empty()) {
                        mesh_adjacency_map[src_mesh_id].insert(neighbor_mesh_id);
                    }
                }
            }
        }
        return mesh_adjacency_map;
    }

    /**
     * This function takes hop information and computes the actual destination nodes that would be visited during a ring
     * traversal multicast.
     *
     * For ring topology, there are two cases:
     * 1. Wrap-around mesh: nodes are arranged in a serpentine pattern where traffic flows in a ring/circular pattern
     *    around the mesh. The direction of turn (at mesh boundaries) depends on the current row/position.
     * 2. Non wrap-around mesh: when hitting a boundary, traffic wraps around to the opposite edge of the mesh.
     */
    std::vector<FabricNodeId> get_ring_topology_dst_node_ids(
        const FabricNodeId& src_node_id,
        RoutingDirection initial_direction,
        uint32_t total_hops,
        ChipSendType chip_send_type) const {
        std::vector<std::pair<FabricNodeId, RoutingDirection>> ring_path;

        // Check if this is a wrap-around mesh
        bool is_wrap_around = wrap_around_mesh(src_node_id);

        if (is_wrap_around) {
            // Use the existing wrap-around mesh logic
            ring_path = trace_wrap_around_mesh_ring_path(src_node_id, initial_direction, total_hops);
        } else {
            // Use the new non wrap-around mesh logic
            ring_path = trace_ring_path(src_node_id, initial_direction, total_hops);
        }

        std::vector<FabricNodeId> ring_destinations;
        ring_destinations.reserve(total_hops);

        // Extract destination nodes (skip the source node, get the next nodes in path)
        for (uint32_t hop = 0; hop < ring_path.size(); ++hop) {
            const auto& [current_node, direction] = ring_path[hop];
            if (chip_send_type == ChipSendType::CHIP_UNICAST) {
                // only push the dest node for last hop
                if (hop == total_hops - 1) {
                    ring_destinations.push_back(current_node);
                }
            } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
                ring_destinations.push_back(current_node);
            }
        }

        log_debug(
            tt::LogTest,
            "src_node: {}, ring_destinations: {}, total_hops: {}, is_wrap_around: {}",
            src_node_id,
            ring_destinations,
            total_hops,
            is_wrap_around);

        return ring_destinations;
    }

    std::shared_ptr<MeshBuffer> create_mesh_buffer_helper(
        const std::vector<CoreCoord>& cores, uint32_t address, uint32_t size_bytes) const {
        std::set<CoreRange> all_cores_set;
        for (const auto& core : cores) {
            all_cores_set.insert(CoreRange(core));
        }

        auto all_cores = CoreRangeSet(all_cores_set);
        auto num_cores = all_cores_set.size();
        auto total_size = size_bytes * num_cores;
        auto shard_params = tt::tt_metal::ShardSpecBuffer(
            all_cores, {1, 1}, tt::tt_metal::ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

        auto buffer_distribution_spec = tt::tt_metal::BufferDistributionSpec(
            Shape{num_cores, 1}, Shape{1, 1}, all_cores, tt::tt_metal::ShardOrientation::ROW_MAJOR);

        auto buffer_page_mapping = buffer_distribution_spec.compute_page_mapping();

        DeviceLocalBufferConfig buffer_specs = {
            .page_size = size_bytes,
            .buffer_type = tt::tt_metal::BufferType::L1,
            .sharding_args = tt::tt_metal::BufferShardingArgs(
                buffer_distribution_spec, shard_params, tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };

        MeshBufferConfig mesh_buffer_specs = ReplicatedBufferConfig{
            .size = total_size,
        };
        auto mesh_buffer = MeshBuffer::create(mesh_buffer_specs, buffer_specs, mesh_device_.get(), address);

        return mesh_buffer;
    }

    // Structure to hold read operation state
    struct ReadBufferOperation {
        std::shared_ptr<MeshBuffer> mesh_buffer;
        tt::tt_metal::UncompressedBufferPageMapping buffer_page_mapping;
        std::vector<uint32_t> data;
        uint32_t size_bytes = 0;
    };

    // Start non-blocking read operation
    ReadBufferOperation initiate_read_buffer_from_cores(
        const MeshCoordinate& device_coord,
        const std::vector<CoreCoord>& cores,
        uint32_t address,
        uint32_t size_bytes) const {
        auto mesh_buffer = create_mesh_buffer_helper(cores, address, size_bytes);

        const auto& buffer_distribution_spec =
            mesh_buffer->device_local_config().sharding_args.buffer_distribution_spec();
        TT_FATAL(buffer_distribution_spec.has_value(), "Buffer distribution spec is not set");
        const auto buffer_page_mapping = buffer_distribution_spec->compute_page_mapping();

        // Preallocate data buffer
        auto total_size = size_bytes * buffer_page_mapping.all_cores.size();
        std::vector<uint32_t> data(total_size / sizeof(uint32_t));

        // Start non-blocking read
        tt::tt_metal::distributed::ReadShard(
            mesh_device_->mesh_command_queue(),
            data,
            mesh_buffer,
            device_coord,
            false  // blocking=false
        );

        return {mesh_buffer, buffer_page_mapping, std::move(data), size_bytes};
    }

    // Process results after barrier_reads() has been called
    std::unordered_map<CoreCoord, std::vector<uint32_t>> complete_read_buffer_from_cores(
        const ReadBufferOperation& op) const {
        // Process results (existing splice logic)
        std::unordered_map<CoreCoord, std::vector<uint32_t>> results;
        auto num_words_per_core = op.size_bytes / sizeof(uint32_t);

        for (auto i = 0; i < op.buffer_page_mapping.all_cores.size(); i++) {
            const auto& core = op.buffer_page_mapping.all_cores[i];
            const auto& page_indices = op.buffer_page_mapping.core_host_page_indices[i];
            std::vector<uint32_t> core_data;
            core_data.reserve(page_indices.size() * num_words_per_core);
            for (const auto& page_idx : page_indices) {
                if (page_idx == tt::tt_metal::UncompressedBufferPageMapping::PADDING) {
                    continue;
                }
                auto start_idx = page_idx * num_words_per_core;
                auto end_idx = start_idx + num_words_per_core;
                core_data.insert(core_data.end(), op.data.begin() + start_idx, op.data.begin() + end_idx);
            }
            results.emplace(core, core_data);
        }

        return results;
    }

    // Data reading helpers - preserve backward compatibility
    std::unordered_map<CoreCoord, std::vector<uint32_t>> read_buffer_from_cores(
        const MeshCoordinate& device_coord,
        const std::vector<CoreCoord>& cores,
        uint32_t address,
        uint32_t size_bytes) const override {
        auto op = initiate_read_buffer_from_cores(device_coord, cores, address, size_bytes);
        barrier_reads();  // Wait for read to complete
        return complete_read_buffer_from_cores(op);
    }

    // When blocking is enabled, results_out must be pre-allocated for each core
    void read_buffer_from_ethernet_cores(
        const MeshCoordinate& device_coord,
        const std::vector<CoreCoord>& cores,
        uint32_t address,
        uint32_t size_bytes,
        bool blocking,
        std::unordered_map<CoreCoord, std::vector<uint32_t>>& results_out) const {
        auto* device = mesh_device_->get_device(device_coord);
        auto num_elements = tt::align(size_bytes, sizeof(uint32_t));
        for (const auto& logical_core : cores) {
            auto virtual_core = device->ethernet_core_from_logical_core(logical_core);
            if (!blocking) {
                TT_FATAL(
                    results_out.contains(logical_core),
                    "read_buffer_from_ethernet_cores was called in non-blocking mode without pre-allocating the "
                    "results_out container. Non-blocking mode requires preallocating the results entries for each "
                    "core.");
                results_out.at(logical_core).resize(num_elements, 0);
            } else {
                results_out[logical_core] = std::vector<uint32_t>(num_elements, 0);
            }
            dynamic_cast<tt::tt_metal::distributed::FDMeshCommandQueue&>(mesh_device_->mesh_command_queue())
                .enqueue_read_shard_from_core(
                    tt::tt_metal::distributed::DeviceMemoryAddress{device_coord, virtual_core, address},
                    results_out.at(logical_core).data(),
                    size_bytes,
                    blocking);
        }
    }

    void barrier_reads() const { mesh_device_->mesh_command_queue().finish(); }

    void write_buffer_to_ethernet_cores(
        const MeshCoordinate& device_coord,
        const std::vector<CoreCoord>& cores,
        uint32_t address,
        const std::vector<uint8_t>& data) const {
        auto* device = mesh_device_->get_device(device_coord);
        for (const auto& logical_core : cores) {
            auto virtual_core = device->ethernet_core_from_logical_core(logical_core);

            dynamic_cast<tt::tt_metal::distributed::FDMeshCommandQueue&>(mesh_device_->mesh_command_queue())
                .enqueue_write_shard_to_core(
                    tt::tt_metal::distributed::DeviceMemoryAddress{device_coord, virtual_core, address},
                    data.data(),
                    data.size(),
                    false);
        }
        mesh_device_->mesh_command_queue().finish();
    }

    void zero_out_buffer_on_cores(
        const MeshCoordinate& device_coord,
        const std::vector<CoreCoord>& cores,
        uint32_t address,
        uint32_t size_bytes) const override {
        auto mesh_buffer = create_mesh_buffer_helper(cores, address, size_bytes);

        const auto total_size = size_bytes * cores.size();
        std::vector<uint32_t> zero_buffer(total_size / sizeof(uint32_t), 0);
        tt::tt_metal::distributed::WriteShard(
            mesh_device_->mesh_command_queue(), mesh_buffer, zero_buffer, device_coord, true);
    }

    // Local runtime args function - writes args to local args buffer instead of using SetRuntimeArgs
    void write_data_to_core(
        const MeshCoordinate& device_coord,
        const CoreCoord& core,
        uint32_t local_args_address,
        const std::vector<uint32_t>& args) const override {
        auto mesh_buffer = create_mesh_buffer_helper({core}, local_args_address, args.size() * sizeof(uint32_t));

        const auto total_size = args.size() * sizeof(uint32_t);
        std::vector<uint32_t> all_args_buffer;
        all_args_buffer.reserve(total_size / sizeof(uint32_t));
        all_args_buffer.insert(all_args_buffer.end(), args.begin(), args.end());

        tt::tt_metal::distributed::WriteShard(
            mesh_device_->mesh_command_queue(), mesh_buffer, all_args_buffer, device_coord, true);
    }

    // ======================================================================================
    // IRouteManager methods
    // ======================================================================================
    uint32_t get_num_mesh_dims() const override { return mesh_shape_.dims(); }

    // TODO: instead of parsing ChipSendType, this should only care about unicast/mcast
    // or capturing every device in the path or not
    std::vector<FabricNodeId> get_dst_node_ids_from_hops(
        FabricNodeId src_node,
        std::unordered_map<RoutingDirection, uint32_t>& hops,
        ChipSendType chip_send_type) const override {
        if (this->topology_ == Topology::Ring) {
            RoutingDirection initial_direction = RoutingDirection::N;
            uint32_t total_hops = 0;
            for (const auto& [direction, hop_count] : hops) {
                if (hop_count != 0) {
                    total_hops = hop_count;
                    initial_direction = direction;
                }
            }
            TT_FATAL(total_hops != 0, "all directions has 0 hops");
            return get_ring_topology_dst_node_ids(src_node, initial_direction, total_hops, chip_send_type);
        }

        const MeshCoordinate& src_coord = get_device_coord(src_node);
        return compute_destination_nodes_from_hops(src_node, src_coord, hops, chip_send_type);
    }

    bool are_devices_linear(const std::vector<FabricNodeId>& node_ids) const override {
        if (node_ids.size() <= 1) {
            return true;
        }

        auto first_coord = get_device_coord(node_ids[0]);
        bool all_same_row = true;
        bool all_same_col = true;

        for (size_t i = 1; i < node_ids.size(); ++i) {
            auto next_coord = get_device_coord(node_ids[i]);
            if (next_coord[COL_DIM] != first_coord[COL_DIM]) {
                all_same_col = false;
            }
            if (next_coord[ROW_DIM] != first_coord[ROW_DIM]) {
                all_same_row = false;
            }
        }
        return all_same_row || all_same_col;
    }

    std::unordered_map<RoutingDirection, uint32_t> get_hops_to_chip(
        FabricNodeId src_node_id, FabricNodeId dst_node_id) const override {
        const auto& src_coord = get_device_coord(src_node_id);
        const auto& dst_coord = get_device_coord(dst_node_id);

        const auto displacement = get_displacement(src_coord, dst_coord);
        return get_hops_from_displacement(displacement);
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_full_device_random_pairs(std::mt19937& gen) const override {
        auto unpaired = get_global_node_ids();

        if (unpaired.size() % 2 != 0) {
            log_warning(
                tt::LogTest,
                "full_device_random_pairing pattern requires an even number of devices, but found {}. One device will "
                "be left unpaired.",
                unpaired.size());
        }

        std::shuffle(unpaired.begin(), unpaired.end(), gen);
        std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;
        pairs.reserve(unpaired.size() / 2);

        while (unpaired.size() >= 2) {
            FabricNodeId src_node = unpaired.back();
            unpaired.pop_back();

            std::vector<size_t> valid_dst_indices;
            for (size_t i = 0; i < unpaired.size(); ++i) {
                const auto& dst_node = unpaired[i];
                bool is_valid_pair =
                    (this->topology_ != Topology::Linear) || this->are_devices_linear({src_node, dst_node});
                if (is_valid_pair) {
                    valid_dst_indices.push_back(i);
                }
            }

            if (valid_dst_indices.empty()) {
                log_warning(
                    tt::LogTest,
                    "Could not find a valid partner for device {}. Unable to create a full random pairing.",
                    src_node);
                log_warning(tt::LogTest, "Exiting early with {} pairs", pairs.size());
                break;
            }

            std::shuffle(valid_dst_indices.begin(), valid_dst_indices.end(), gen);
            size_t picked_idx = valid_dst_indices.front();
            FabricNodeId dst_node = unpaired[picked_idx];
            pairs.push_back({src_node, dst_node});

            std::swap(unpaired[picked_idx], unpaired.back());
            unpaired.pop_back();
        }

        return pairs;
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_to_all_unicast_pairs() const override {
        const auto device_ids = get_global_node_ids();
        std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;
        pairs.reserve(device_ids.size() * (device_ids.size() - 1));
        for (const auto& src_node : device_ids) {
            for (const auto& dst_node : device_ids) {
                if (src_node == dst_node) {
                    continue;
                }
                if (this->topology_ == Topology::Linear) {
                    if (!this->are_devices_linear({src_node, dst_node})) {
                        continue;
                    }
                }
                pairs.push_back({src_node, dst_node});
            }
        }
        return pairs;
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_to_one_unicast_pairs(
        const uint32_t device_idx) const override {
        // device_idx is used to deterministically select a destination node from all available global nodes
        const auto device_ids = get_global_node_ids();
        std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;
        auto dst_node_id = device_ids[device_idx % device_ids.size()];
        pairs.reserve(device_ids.size() - 1);
        for (const auto& src_node : device_ids) {
            if (src_node == dst_node_id) {
                continue;
            }
            if (this->topology_ == Topology::Linear) {
                if (!this->are_devices_linear({src_node, dst_node_id})) {
                    continue;
                }
            }
            pairs.push_back({src_node, dst_node_id});
        }
        return pairs;
    }

    std::unordered_map<RoutingDirection, uint32_t> get_full_mcast_hops(const FabricNodeId& src_node_id) const override {
        std::unordered_map<RoutingDirection, uint32_t> hops;
        for (const auto& direction : FabricContext::routing_directions) {
            hops[direction] = 0;
        }

        const auto src_coord = get_device_coord(src_node_id);
        hops[RoutingDirection::N] = src_coord[NS_DIM];
        hops[RoutingDirection::S] = mesh_shape_[NS_DIM] - src_coord[NS_DIM] - 1;
        hops[RoutingDirection::E] = mesh_shape_[EW_DIM] - src_coord[EW_DIM] - 1;
        hops[RoutingDirection::W] = src_coord[EW_DIM];

        return hops;
    }

    /**
     * Unlike traditional multicast that sends in all directions from a source, unidirectional
     * linear multicast divides the mesh into halves and only sends in ONE direction per source:
     *
     * **Horizontal Division:**
     * - Left half of mesh (x < mesh_width/2): multicasts EAST to right half
     * - Right half of mesh (x >= mesh_width/2): multicasts WEST to left half
     *
     * **Vertical Division:**
     * - Upper half of mesh (y < mesh_height/2): multicasts SOUTH to lower half
     * - Lower half of mesh (y >= mesh_height/2): multicasts NORTH to upper half
     */
    std::unordered_map<RoutingDirection, uint32_t> get_unidirectional_linear_mcast_hops(
        const FabricNodeId& src_node_id, uint32_t dim) const override {
        std::unordered_map<RoutingDirection, uint32_t> hops;
        for (const auto& direction : FabricContext::routing_directions) {
            hops[direction] = 0;
        }

        const auto src_coord = get_device_coord(src_node_id);

        if (dim == NS_DIM) {
            if (src_coord[NS_DIM] < mesh_shape_[NS_DIM] / 2) {
                hops[RoutingDirection::S] = mesh_shape_[NS_DIM] - src_coord[NS_DIM] - 1;
            } else {
                hops[RoutingDirection::N] = src_coord[NS_DIM];
            }
        } else if (dim == EW_DIM) {
            if (src_coord[EW_DIM] < mesh_shape_[EW_DIM] / 2) {
                hops[RoutingDirection::E] = mesh_shape_[EW_DIM] - src_coord[EW_DIM] - 1;
            } else {
                hops[RoutingDirection::W] = src_coord[EW_DIM];
            }
        } else {
            TT_THROW("input mesh dim is not supported: {}", dim);
        }

        return hops;
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_neighbor_exchange_pairs() const override {
        const auto device_ids = get_global_node_ids();
        std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;

        // To support device meshes that do not have wraparound connections, Ring topology is handled separately
        if (topology_ != Topology::Ring) {
            // Handle mesh, torus, neighbor exchange and linear topologies with directional neighbors
            const std::vector<RoutingDirection> directions = {
                RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W};

            for (const auto& src_node : device_ids) {
                for (const auto& direction : directions) {
                    // Check if neighbor exists in this direction
                    const auto& neighbors =
                        tt::tt_metal::MetalContext::instance().get_control_plane().get_chip_neighbors(
                            src_node, direction);

                    if (!neighbors.empty()) {
                        // Get the first (and should be only) neighbor mesh
                        auto neighbor_mesh_it = neighbors.begin();
                        const auto& neighbor_chips = neighbor_mesh_it->second;

                        if (!neighbor_chips.empty()) {
                            // Get the first (and should be only) neighbor chip
                            FabricNodeId neighbor(neighbor_mesh_it->first, neighbor_chips[0]);

                            // Only add if neighbor exists and is different from source
                            bool is_valid_neighbor = (neighbor != src_node);

                            // For linear topology, also check if devices are on the same line
                            if (topology_ == Topology::Linear) {
                                is_valid_neighbor &= are_devices_linear({src_node, neighbor});
                            }

                            if (is_valid_neighbor) {
                                pairs.emplace_back(src_node, neighbor);
                            }
                        }
                    }
                }
            }
        } else {
            // If a Ring topology is used, only the devices on the perimeter of a mesh participate in the test.
            // Instead of using physical wraparound connections, a large "ring" is formed with the perimeter devices, as
            // is enforced by the get_wrap_around_mesh_ring_neighbors function.
            for (const auto& src_node : device_ids) {
                auto ring_neighbors = get_wrap_around_mesh_ring_neighbors(src_node, device_ids);
                if (ring_neighbors.has_value()) {
                    auto [forward_neighbor, backward_neighbor] = ring_neighbors.value();
                    if (forward_neighbor != src_node) {
                        pairs.emplace_back(src_node, forward_neighbor);
                    }
                    if (backward_neighbor != src_node) {
                        pairs.emplace_back(src_node, backward_neighbor);
                    }
                }
            }
        }

        return pairs;
    }

    std::optional<std::pair<FabricNodeId, FabricNodeId>> get_wrap_around_mesh_ring_neighbors(
        const FabricNodeId& src_node, const std::vector<FabricNodeId>& /*devices*/) const override {
        // Get mesh dimensions
        uint32_t mesh_height = mesh_shape_[NS_DIM];
        uint32_t mesh_width = mesh_shape_[EW_DIM];

        // Convert chip_id to row/col coordinates (row-major order)
        uint32_t row = src_node.chip_id / mesh_width;
        uint32_t col = src_node.chip_id % mesh_width;

        // Check if the device is on the outer ring (perimeter)
        bool is_perimeter = (row == 0) || (row == mesh_height - 1) || (col == 0) || (col == mesh_width - 1);

        // If not on perimeter, return nullopt to indicate no valid ring neighbors
        if (!is_perimeter) {
            return std::nullopt;
        }

        // Calculate ring neighbors based on position on perimeter
        // forward always try to go right/up first, backward always try to go left/down first
        ChipId forward_chip_id, backward_chip_id;

        if (row == 0 && col == 0) {
            // Top-left corner (0): forward=1, backward=4 (4x4 mesh)
            forward_chip_id = 1;
            backward_chip_id = mesh_width;
        } else if (row == 0 && col == mesh_width - 1) {
            // Top-right corner (3): forward=7, backward=2
            forward_chip_id = src_node.chip_id + mesh_width;
            backward_chip_id = src_node.chip_id - 1;
        } else if (row == mesh_height - 1 && col == mesh_width - 1) {
            // Bottom-right corner (15): forward=11, backward=14
            forward_chip_id = src_node.chip_id - mesh_width;
            backward_chip_id = src_node.chip_id - 1;
        } else if (row == mesh_height - 1 && col == 0) {
            // Bottom-left corner (12): forward=13, backward=8
            forward_chip_id = src_node.chip_id + 1;
            backward_chip_id = src_node.chip_id - mesh_width;
        } else if (row == 0 || row == mesh_height - 1) {
            // Top or bottom row (not corners): forward=right, backward=left
            forward_chip_id = src_node.chip_id + 1;
            backward_chip_id = src_node.chip_id - 1;
        } else if (col == mesh_width - 1 || col == 0) {
            // Right or left column (not corners): forward=up, backward=down
            forward_chip_id = src_node.chip_id - mesh_width;
            backward_chip_id = src_node.chip_id + mesh_width;
        } else {
            TT_THROW("Device {} should be on perimeter but logic error occurred", src_node.chip_id);
        }

        log_debug(
            LogTest,
            "src_node: {}, forward_chip_id: {}, backward_chip_id: {}",
            src_node.chip_id,
            forward_chip_id,
            backward_chip_id);

        FabricNodeId dst_node_forward = FabricNodeId{src_node.mesh_id, forward_chip_id};
        FabricNodeId dst_node_backward = FabricNodeId{src_node.mesh_id, backward_chip_id};

        return std::make_pair(dst_node_forward, dst_node_backward);
    }

    uint32_t get_num_sync_devices() const {
        uint32_t num_devices;
        switch (topology_) {
            case tt::tt_fabric::Topology::Linear: {
                num_devices = mesh_shape_[NS_DIM] + mesh_shape_[EW_DIM] - 1;
                return num_devices;
            }
            case tt::tt_fabric::Topology::Ring: {
                if (wrap_around_mesh_) {
                    // sync using full ring mcast, ie, mcast on both forward/backward path.
                    num_devices = 2 * (mesh_shape_[NS_DIM] - 1 + mesh_shape_[EW_DIM] - 1);
                } else {
                    num_devices = mesh_shape_[NS_DIM] + mesh_shape_[EW_DIM] - 1;
                }
                return num_devices;
            }
            // for torus, the handling should be same as mesh since we need to sync with all the devices
            case tt::tt_fabric::Topology::Torus:
            case tt::tt_fabric::Topology::Mesh: {
                num_devices = mesh_shape_[NS_DIM] * mesh_shape_[EW_DIM];
                return num_devices;
            }
            default: TT_THROW("Unsupported topology for get_num_sync_devices: {}", static_cast<int>(topology_));
        }
    }

    std::unordered_map<RoutingDirection, uint32_t> get_wrap_around_mesh_full_or_half_ring_mcast_hops(
        const FabricNodeId& src_node_id,
        const FabricNodeId& dst_node_forward_id,
        const FabricNodeId& dst_node_backward_id,
        HighLevelTrafficPattern pattern_type) const override {
        std::unordered_map<RoutingDirection, uint32_t> hops;
        for (const auto& direction : FabricContext::routing_directions) {
            hops[direction] = 0;
        }

        auto direction_forward = get_forwarding_direction(src_node_id, dst_node_forward_id);
        auto direction_backward = get_forwarding_direction(src_node_id, dst_node_backward_id);

        auto num_forward_hops = 0;
        auto num_backward_hops = 0;
        uint32_t full_hop_count = (2 * (mesh_shape_[NS_DIM] - 1 + mesh_shape_[EW_DIM] - 1)) - 1;

        if (pattern_type == HighLevelTrafficPattern::FullRing) {
            num_forward_hops = full_hop_count;
            num_backward_hops = full_hop_count;
        } else if (pattern_type == HighLevelTrafficPattern::HalfRing) {
            num_forward_hops = tt::div_up(full_hop_count, 2);
            num_backward_hops = full_hop_count - num_forward_hops;
            if (src_node_id.chip_id % 2 == 0) {
                std::swap(num_forward_hops, num_backward_hops);
            }
        } else {
            TT_THROW(
                "Unsupported pattern type for ring: only FullRing and HalfRing are "
                "supported");
        }

        hops[direction_forward] = num_forward_hops;
        hops[direction_backward] = num_backward_hops;

        return hops;
    }

    std::unordered_map<RoutingDirection, uint32_t> get_full_or_half_ring_mcast_hops(
        const FabricNodeId& src_node_id, HighLevelTrafficPattern pattern_type, uint32_t dim) const override {
        std::unordered_map<RoutingDirection, uint32_t> hops;
        for (const auto& direction : FabricContext::routing_directions) {
            hops[direction] = 0;
        }

        auto num_forward_hops = 0;
        auto num_backward_hops = 0;
        uint32_t full_hop_count = 0;
        RoutingDirection direction_forward = RoutingDirection::N;
        RoutingDirection direction_backward = RoutingDirection::N;

        if (dim == NS_DIM) {
            full_hop_count = mesh_shape_[NS_DIM] - 1;
            direction_forward = RoutingDirection::N;
            direction_backward = RoutingDirection::S;
        } else if (dim == EW_DIM) {
            full_hop_count = mesh_shape_[EW_DIM] - 1;
            direction_forward = RoutingDirection::E;
            direction_backward = RoutingDirection::W;
        } else {
            TT_THROW("input mesh dim is not supported: {}", dim);
        }

        if (pattern_type == HighLevelTrafficPattern::FullRing) {
            num_forward_hops = full_hop_count;
            num_backward_hops = full_hop_count;
        } else if (pattern_type == HighLevelTrafficPattern::HalfRing) {
            num_forward_hops = tt::div_up(full_hop_count, 2);
            num_backward_hops = full_hop_count - num_forward_hops;
            if (src_node_id.chip_id % 2 == 0) {
                std::swap(num_forward_hops, num_backward_hops);
            }
        } else {
            TT_THROW(
                "Unsupported pattern type for ring: only FullRing and HalfRing are "
                "supported");
        }

        hops[direction_forward] = num_forward_hops;
        hops[direction_backward] = num_backward_hops;

        return hops;
    }

    std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_multicast_hops(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_hops;
        split_hops.reserve(4);

        if (this->topology_ == Topology::Linear || this->topology_ == Topology::Ring ||
            this->topology_ == Topology::NeighborExchange) {
            for (const auto& [dir, hop_count] : hops) {
                if (hop_count > 0) {
                    split_hops.push_back({{dir, hop_count}});
                }
            }
        } else if (this->topology_ == Topology::Mesh || this->topology_ == Topology::Torus) {
            // Generate up to 4 maps: pure E, pure W, N+spines (if N>0), S+spines (if S>0)
            auto north_hops = hops.contains(RoutingDirection::N) ? hops.at(RoutingDirection::N) : 0;
            auto south_hops = hops.contains(RoutingDirection::S) ? hops.at(RoutingDirection::S) : 0;
            auto east_hops = hops.contains(RoutingDirection::E) ? hops.at(RoutingDirection::E) : 0;
            auto west_hops = hops.contains(RoutingDirection::W) ? hops.at(RoutingDirection::W) : 0;

            // Pure E
            if (east_hops > 0) {
                split_hops.push_back({{RoutingDirection::E, east_hops}});
            }

            // Pure W
            if (west_hops > 0) {
                split_hops.push_back({{RoutingDirection::W, west_hops}});
            }

            // N trunk + spines (only if N >0)
            if (north_hops > 0) {
                std::unordered_map<RoutingDirection, uint32_t> n_map = {
                    {RoutingDirection::N, north_hops},
                    {RoutingDirection::E, east_hops},
                    {RoutingDirection::W, west_hops}};
                split_hops.push_back(n_map);
            }

            // S trunk + spines (only if S >0)
            if (south_hops > 0) {
                std::unordered_map<RoutingDirection, uint32_t> s_map = {
                    {RoutingDirection::S, south_hops},
                    {RoutingDirection::E, east_hops},
                    {RoutingDirection::W, west_hops}};
                split_hops.push_back(s_map);
            }
        } else {
            TT_THROW("Unsupported topology: {} for split_multicast_hops", this->topology_);
        }

        return split_hops;
    }

    FabricNodeId get_random_unicast_destination(FabricNodeId src_node_id, std::mt19937& gen) const override {
        auto all_devices = this->get_global_node_ids();
        std::vector<FabricNodeId> possible_dsts;
        possible_dsts.reserve(all_devices.size());
        for (const auto& dev : all_devices) {
            if (dev == src_node_id) {
                continue;
            }
            if (this->topology_ == Topology::Linear && !this->are_devices_linear({src_node_id, dev})) {
                continue;
            }
            possible_dsts.push_back(dev);
        }

        if (possible_dsts.empty()) {
            TT_THROW(
                "Cannot pick a random unicast destination for sender on device {}: no valid partner devices available.",
                src_node_id);
        }

        std::shuffle(possible_dsts.begin(), possible_dsts.end(), gen);
        return possible_dsts[0];
    }

    MeshShape get_mesh_shape() const override { return mesh_shape_; }

    Topology get_topology() const { return topology_; }

    bool wrap_around_mesh(FabricNodeId node) const override {
        return tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().is_wrap_around_mesh(
            node.mesh_id);
    }

    RoutingDirection get_forwarding_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const override {
        auto forwarding_direction = tt::tt_metal::MetalContext::instance().get_control_plane().get_forwarding_direction(
            src_node_id, dst_node_id);
        TT_FATAL(
            forwarding_direction.has_value(), "No forwarding direction found for {} -> {}", src_node_id, dst_node_id);
        return forwarding_direction.value();
    }

    RoutingDirection get_forwarding_direction(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        if (topology_ == Topology::Linear || topology_ == Topology::Ring || topology_ == Topology::NeighborExchange) {
            // for 1d, we expect hops only in one direction to be non-zero
            for (const auto& [direction, hop] : hops) {
                if (hop != 0) {
                    return direction;
                }
            }
        } else if (topology_ == Topology::Mesh || topology_ == Topology::Torus) {
            // TODO: update this logic once 2D mcast is supported
            // for now we return the first direction that is non-zero
            // for 2D, since we use dimension order routing, lookup the directions in the order of N, S, E, W
            for (const auto& direction : FabricContext::routing_directions) {
                if (hops.contains(direction) && hops.at(direction) != 0) {
                    return direction;
                }
            }
        } else {
            TT_THROW("Unsupported topology: {} for hop-based get_forwarding_direction", topology_);
        }

        TT_THROW("Failed to find a forwarding direction for hops: {}", hops);
    }

    FabricNodeId get_mcast_start_node_id(
        const FabricNodeId& src_node_id, const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        const auto src_coord = get_device_coord(src_node_id);
        const auto forwarding_direction = get_forwarding_direction(hops);

        // get the new coord
        const auto new_coord = src_coord.get_neighbor(
            mesh_shape_,
            get_step_for_direction(forwarding_direction),
            get_dim_for_direction(forwarding_direction),
            get_boundary_mode_for_dimension(get_dim_for_direction(forwarding_direction)));
        TT_FATAL(new_coord.has_value(), "Failed to get mcast start node id for src: {}, hops: {}", src_node_id, hops);

        return get_fabric_node_id(new_coord.value());
    }

    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id,
        const FabricNodeId& dst_node_id,
        const RoutingDirection& direction) const override {
        return tt::tt_fabric::get_forwarding_link_indices_in_direction(src_node_id, dst_node_id, direction);
    }

    std::optional<FabricNodeId> get_neighbor_node_id_or_nullopt(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
        // This function is used to get the neighbor node ID for a given source node ID and direction, returning nullopt
        // if no neighbor is found.
        std::optional<FabricNodeId> neighbor_node_id = std::nullopt;

        const auto& neighbors =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_chip_neighbors(src_node_id, direction);

        // If more than 1 mesh is returned, throw an error.
        TT_FATAL(
            neighbors.size() <= 1,
            "Expected at most one neighbor mesh for {} in direction: {}",
            src_node_id,
            direction);

        if (!neighbors.empty()) {
            neighbor_node_id = FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
        }

        return neighbor_node_id;
    }

    FabricNodeId get_neighbor_node_id(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
        // This function is used to get the neighbor node ID for a given source node ID and direction, throwing an error
        // if no neighbor is found.
        std::optional<FabricNodeId> neighbor_node_id = get_neighbor_node_id_or_nullopt(src_node_id, direction);
        TT_FATAL(
            neighbor_node_id.has_value(),
            "Expected at least 1 neighbor chip for {} in direction: {}",
            src_node_id,
            direction);
        return neighbor_node_id.value();
    }

    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
        const auto& neighbor_node_id = get_neighbor_node_id(src_node_id, direction);
        return this->get_forwarding_link_indices_in_direction(src_node_id, neighbor_node_id, direction);
    }

    /// Helper to compute per‐direction hops and the global sync value
    std::pair<std::unordered_map<RoutingDirection, uint32_t>, uint32_t> get_sync_hops_and_val(
        const FabricNodeId& src_device, const std::vector<FabricNodeId>& devices) const override {
        std::unordered_map<RoutingDirection, uint32_t> multi_directional_hops;
        uint32_t global_sync_val = 0;

        switch (topology_) {
            case tt::tt_fabric::Topology::Ring: {
                if (wrap_around_mesh_) {
                    // Get ring neighbors - returns nullopt for non-perimeter devices
                    auto ring_neighbors = this->get_wrap_around_mesh_ring_neighbors(src_device, devices);
                    // Check if the result is valid (has value)
                    if (!ring_neighbors.has_value()) {
                        // Skip this device as it's not on the perimeter and can't participate in ring multicast
                        log_info(LogTest, "Skipping device {} as it's not on the perimeter ring", src_device.chip_id);
                        return {{}, 0};
                    }

                    // Extract the valid ring neighbors
                    auto [dst_node_forward, dst_node_backward] = ring_neighbors.value();

                    multi_directional_hops = this->get_wrap_around_mesh_full_or_half_ring_mcast_hops(
                        src_device, dst_node_forward, dst_node_backward, HighLevelTrafficPattern::FullRing);

                } else {
                    // if not wrap around mesh, then need to get the neighbours on all directions.
                    auto ns_hops =
                        this->get_full_or_half_ring_mcast_hops(src_device, HighLevelTrafficPattern::FullRing, NS_DIM);
                    auto ew_hops =
                        this->get_full_or_half_ring_mcast_hops(src_device, HighLevelTrafficPattern::FullRing, EW_DIM);
                    for (const auto& [direction, hops] : ns_hops) {
                        if (hops != 0) {
                            multi_directional_hops[direction] = hops;
                        }
                    }
                    for (const auto& [direction, hops] : ew_hops) {
                        if (hops != 0) {
                            multi_directional_hops[direction] = hops;
                        }
                    }
                }
                // minus 2 because full ring pattern traverse each node twice.
                auto num_sync_devices = this->get_num_sync_devices();
                global_sync_val =
                    2 * num_sync_devices - 2;  // minus 2 because in a full ring pattern we dont mcast to self (twice).
                break;
            }
            case tt::tt_fabric::Topology::Linear: {
                multi_directional_hops = this->get_full_mcast_hops(src_device);
                global_sync_val = this->get_num_sync_devices() - 1;
                break;
            }
            case tt::tt_fabric::Topology::NeighborExchange: {
                // NeighborExchange topology only performs syncs between a device's nearest neighbors
                multi_directional_hops = this->get_hops_to_nearest_neighbors(src_device);
                global_sync_val = multi_directional_hops.size();
                break;
            }
            // for torus, the handling should be same as mesh since we need to sync with all the devices
            // it doesnt matter if we use torus links or internal links to get the sync packets across
            case tt::tt_fabric::Topology::Torus:
            case tt::tt_fabric::Topology::Mesh: {
                multi_directional_hops = this->get_full_mcast_hops(src_device);
                global_sync_val = this->get_num_sync_devices() - 1;
                break;
            }
            default: TT_THROW("Unsupported topology for line sync: {}", static_cast<int>(topology_));
        }

        return {std::move(multi_directional_hops), global_sync_val};
    }

    // Helper function to trace ring path with boundary turning logic
    std::vector<std::pair<FabricNodeId, RoutingDirection>> trace_wrap_around_mesh_ring_path(
        const FabricNodeId& src_node_id, RoutingDirection initial_direction, uint32_t total_hops) const {
        std::vector<std::pair<FabricNodeId, RoutingDirection>> path;
        path.reserve(total_hops);

        // Get starting coordinate
        MeshCoordinate current_coord = get_device_coord(src_node_id);
        RoutingDirection current_direction = initial_direction;
        FabricNodeId current_node = src_node_id;

        for (uint32_t hop = 0; hop < total_hops; ++hop) {
            // Try to move in current direction
            MeshCoordinate next_coord = current_coord;
            bool can_move = true;

            switch (current_direction) {
                case RoutingDirection::N:
                    if (current_coord[NS_DIM] == 0) {
                        can_move = false;
                    } else {
                        next_coord = MeshCoordinate(current_coord[NS_DIM] - 1, current_coord[EW_DIM]);
                    }
                    break;
                case RoutingDirection::S:
                    if (current_coord[NS_DIM] == mesh_shape_[NS_DIM] - 1) {
                        can_move = false;
                    } else {
                        next_coord = MeshCoordinate(current_coord[NS_DIM] + 1, current_coord[EW_DIM]);
                    }
                    break;
                case RoutingDirection::E:
                    if (current_coord[EW_DIM] == mesh_shape_[EW_DIM] - 1) {
                        can_move = false;
                    } else {
                        next_coord = MeshCoordinate(current_coord[NS_DIM], current_coord[EW_DIM] + 1);
                    }
                    break;
                case RoutingDirection::W:
                    if (current_coord[EW_DIM] == 0) {
                        can_move = false;
                    } else {
                        next_coord = MeshCoordinate(current_coord[NS_DIM], current_coord[EW_DIM] - 1);
                    }
                    break;
                default: TT_THROW("routing direction not supported: {}", current_direction);
            }

            // If we hit a boundary, determine next direction based on current position
            if (!can_move) {
                if (current_direction == RoutingDirection::E) {
                    // Hit east boundary - direction depends on current row
                    if (current_coord[NS_DIM] == 0) {
                        current_direction = RoutingDirection::S;
                    } else {
                        current_direction = RoutingDirection::N;
                    }
                } else if (current_direction == RoutingDirection::W) {
                    if (current_coord[NS_DIM] == 0) {
                        current_direction = RoutingDirection::S;
                    } else {
                        log_info(
                            tt::LogTest,
                            "current_coord {} current_direction {} next direction RoutingDirection::N",
                            get_fabric_node_id(current_coord),
                            current_direction);
                        current_direction = RoutingDirection::N;
                    }
                } else if (current_direction == RoutingDirection::S || current_direction == RoutingDirection::N) {
                    if (current_coord[EW_DIM] == 0) {
                        current_direction = RoutingDirection::E;
                    } else {
                        current_direction = RoutingDirection::W;
                    }
                }

                // Try again with new direction
                switch (current_direction) {
                    case RoutingDirection::N:
                        next_coord = MeshCoordinate(current_coord[NS_DIM] - 1, current_coord[EW_DIM]);
                        break;
                    case RoutingDirection::S:
                        next_coord = MeshCoordinate(current_coord[NS_DIM] + 1, current_coord[EW_DIM]);
                        break;
                    case RoutingDirection::E:
                        next_coord = MeshCoordinate(current_coord[NS_DIM], current_coord[EW_DIM] + 1);
                        break;
                    case RoutingDirection::W:
                        next_coord = MeshCoordinate(current_coord[NS_DIM], current_coord[EW_DIM] - 1);
                        break;
                    default: TT_THROW("routing direction not supported: {}", current_direction);
                }
            }

            // Move to next coordinate
            current_coord = next_coord;
            current_node = get_fabric_node_id(current_coord);
            // Record current node and outgoing direction
            path.emplace_back(current_node, current_direction);
        }

        return path;
    }

    // Helper function to trace ring path with boundary wraparound logic for non wrap-around meshes
    std::vector<std::pair<FabricNodeId, RoutingDirection>> trace_ring_path(
        const FabricNodeId& src_node_id, RoutingDirection initial_direction, uint32_t total_hops) const {
        std::vector<std::pair<FabricNodeId, RoutingDirection>> path;
        path.reserve(total_hops);

        // Get starting coordinate
        MeshCoordinate current_coord = get_device_coord(src_node_id);
        RoutingDirection current_direction = initial_direction;
        FabricNodeId current_node = src_node_id;

        for (uint32_t hop = 0; hop < total_hops; ++hop) {
            // Try to move in current direction
            MeshCoordinate next_coord = current_coord;
            [[maybe_unused]] bool need_wraparound = false;

            switch (current_direction) {
                case RoutingDirection::N:
                    if (current_coord[NS_DIM] == 0) {
                        // Wrap around to bottom edge
                        next_coord = MeshCoordinate(mesh_shape_[NS_DIM] - 1, current_coord[EW_DIM]);
                        need_wraparound = true;
                    } else {
                        next_coord = MeshCoordinate(current_coord[NS_DIM] - 1, current_coord[EW_DIM]);
                    }
                    break;
                case RoutingDirection::S:
                    if (current_coord[NS_DIM] == mesh_shape_[NS_DIM] - 1) {
                        // Wrap around to top edge
                        next_coord = MeshCoordinate(0, current_coord[EW_DIM]);
                        need_wraparound = true;
                    } else {
                        next_coord = MeshCoordinate(current_coord[NS_DIM] + 1, current_coord[EW_DIM]);
                    }
                    break;
                case RoutingDirection::E:
                    if (current_coord[EW_DIM] == mesh_shape_[EW_DIM] - 1) {
                        // Wrap around to left edge
                        next_coord = MeshCoordinate(current_coord[NS_DIM], 0);
                        need_wraparound = true;
                    } else {
                        next_coord = MeshCoordinate(current_coord[NS_DIM], current_coord[EW_DIM] + 1);
                    }
                    break;
                case RoutingDirection::W:
                    if (current_coord[EW_DIM] == 0) {
                        // Wrap around to right edge
                        next_coord = MeshCoordinate(current_coord[NS_DIM], mesh_shape_[EW_DIM] - 1);
                        need_wraparound = true;
                    } else {
                        next_coord = MeshCoordinate(current_coord[NS_DIM], current_coord[EW_DIM] - 1);
                    }
                    break;
                default: TT_THROW("routing direction not supported: {}", current_direction);
            }

            // Move to next coordinate
            current_coord = next_coord;
            current_node = get_fabric_node_id(current_coord);

            log_debug(
                tt::LogTest,
                "hop {}: moved from {} to {} in direction {}, wraparound: {}",
                hop,
                get_device_coord(src_node_id),
                current_coord,
                current_direction,
                need_wraparound);

            // Record current node and outgoing direction
            path.emplace_back(current_node, current_direction);
        }

        return path;
    }

    bool validate_num_links_supported(uint32_t num_links) const override {
        // Validate that num_links doesn't exceed available routing planes for any row/column
        const auto num_pci_devices = tt::tt_metal::MetalContext::instance().get_cluster().number_of_pci_devices();
        const auto num_devices = tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices();

        std::vector<FabricNodeId> devices = get_local_node_ids();
        for (const auto& device : devices) {
            uint32_t max_routing_planes = get_max_routing_planes_for_device(device);
            // TODO: remove this once we have correct
            if (num_pci_devices != num_devices) {
                max_routing_planes -= 1;
            }
            if (num_links > max_routing_planes) {
                log_warning(
                    LogTest,
                    "Skipping: Requested num_links ({}) exceeds maximum available routing planes ({}) for "
                    "device {}. "
                    "Please reduce num_links or check your fabric configuration.",
                    num_links,
                    max_routing_planes,
                    device.chip_id);
                return false;  // Indicate test should be skipped
            }
        }

        return true;
    }

    uint32_t get_max_routing_planes_for_device(const FabricNodeId& node_id) const {
        // Find the minimum number of routing planes across all directions for this device
        uint32_t min_routing_planes = std::numeric_limits<uint32_t>::max();

        // Check all possible directions
        for (const auto& direction : FabricContext::routing_directions) {
            size_t routing_planes = tt::tt_metal::MetalContext::instance()
                                        .get_control_plane()
                                        .get_num_available_routing_planes_in_direction(node_id, direction);
            if (routing_planes > 0) {  // Only consider directions that have routing planes
                min_routing_planes = std::min(min_routing_planes, static_cast<uint32_t>(routing_planes));
            }
        }

        // If no valid directions found, return 0
        return (min_routing_planes == std::numeric_limits<uint32_t>::max()) ? 0 : min_routing_planes;
    }

    // ======================================================================================
    // IDistributedContextManager methods
    // ======================================================================================
    uint32_t get_randomized_master_seed() const override {
        uint32_t master_seed = std::random_device()();
        log_info(tt::LogTest, "No master seed provided. Using randomly generated seed: {}", master_seed);

        const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
        // only need to handshake if we need to generate seed, since each host will have the same commandline arguments.
        if (*(distributed_context->size()) > 1) {
            if (*(distributed_context->rank()) == 0) {
                master_seed = std::random_device()();
                for (int recv_host_rank = 1; recv_host_rank < *(distributed_context->size()); ++recv_host_rank) {
                    distributed_context->send(
                        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&master_seed), sizeof(master_seed)),
                        tt::tt_metal::distributed::multihost::Rank{recv_host_rank},  // send to receiver host
                        tt::tt_metal::distributed::multihost::Tag{0}                 // exchange seed over tag 0
                    );
                }
                log_info(tt::LogTest, "Master seed sent: {}", master_seed);
            } else {
                distributed_context->recv(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&master_seed), sizeof(master_seed)),
                    tt::tt_metal::distributed::multihost::Rank{0},  // receive from sender host
                    tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
                );
                log_info(tt::LogTest, "Master seed received : {}", master_seed);
            }
        }
        return master_seed;
    }

    void barrier() const override {
        const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
        distributed_context->barrier();
    }

    // Returns a map of the hops to the nearest neighbors for a node in all directions. If there is no neighbor in a
    // direction, the key will not be present in the map.
    std::unordered_map<RoutingDirection, uint32_t> get_hops_to_nearest_neighbors(
        const FabricNodeId& src_device) const override {
        std::unordered_map<RoutingDirection, uint32_t> hops_to_nearest_neighbors;
        for (const auto& direction :
             {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W}) {
            const std::optional<FabricNodeId> neighbor_node_id = get_neighbor_node_id_or_nullopt(src_device, direction);
            // If there is no neighbor, function will return src_device
            if (neighbor_node_id.has_value()) {
                hops_to_nearest_neighbors[direction] = 1;
            }
        }
        return hops_to_nearest_neighbors;
    }

    void validate_single_hop(const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        // A valid hop map will have exactly 1 hop in 1 direction
        bool hop_encountered = false;
        for (const auto& [direction, hop_count] : hops) {
            TT_FATAL((hop_count <= 1), "NeighborExchange topology does not support multi-hop traffic patterns");
            if (hop_count == 1) {
                TT_FATAL(
                    !hop_encountered,
                    "NeighborExchange topology does not support hops in multiple directions inside the same traffic "
                    "pattern");
                hop_encountered = true;
            }
        }
    }

private:
    // Helper methods for device frequency validation (performance testing)
    static uint32_t get_expected_baseline_frequency_mhz() {
        auto arch = tt::tt_metal::hal::get_arch();
        switch (arch) {
            case tt::ARCH::WORMHOLE_B0: return 1000;
            case tt::ARCH::BLACKHOLE: return 1350;
            default: TT_THROW("Unsupported architecture for performance testing: {}", arch);
        }
    }

    static uint32_t get_frequency_tolerance_mhz() {
        // Note: these tolerances are initally set as placeholders and should be calibrated based on empirical data
        auto arch = tt::tt_metal::hal::get_arch();
        switch (arch) {
            case tt::ARCH::WORMHOLE_B0: return 10;
            case tt::ARCH::BLACKHOLE: return 20;
            default: TT_THROW("Unsupported architecture for performance testing: {}", arch);
        }
    }

    Topology topology_{0};
    MeshShape mesh_shape_;
    std::set<MeshId> available_mesh_ids_;
    std::vector<FabricNodeId> local_available_node_ids_;
    std::vector<FabricNodeId> global_available_node_ids_;
    tt::tt_fabric::FabricConfig current_fabric_config_{FabricConfig::DISABLED};
    tt_fabric::FabricTensixConfig current_fabric_tensix_config_{tt_fabric::FabricTensixConfig::DISABLED};
    tt_fabric::FabricReliabilityMode current_fabric_reliability_mode_{
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE};
    std::optional<uint32_t> current_max_packet_size_{std::nullopt};
    std::shared_ptr<MeshDevice> mesh_device_;
    std::shared_ptr<MeshWorkload> mesh_workload_;
    MeshId local_mesh_id_;
    std::optional<MeshHostRankId> local_host_rank_;

    bool are_devices_open_ = false;
    bool wrap_around_mesh_ = false;
    mutable std::map<FabricNodeId, uint32_t> device_frequency_cache_;
    mutable bool frequency_validated_ = false;

    void initialize_and_validate_custom_physical_config(const PhysicalMeshConfig& physical_mesh_config) {
        const auto local_mesh_id = MeshId{std::stoi(std::getenv("TT_MESH_ID"))};
        const auto& eth_coord_mapping = physical_mesh_config.eth_coord_mapping;
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

        // ethernet coordinate chip mapping, which should be migrated away from
        std::map<FabricNodeId, ChipId> chip_to_eth_coord_mapping;
        for (std::uint32_t mesh_id = 0; mesh_id < eth_coord_mapping.size(); mesh_id++) {
            if (mesh_id == *local_mesh_id) {
                for (std::uint32_t chip_id = 0; chip_id < eth_coord_mapping[mesh_id].size(); chip_id++) {
                    const auto& eth_coord = eth_coord_mapping[mesh_id][chip_id];
                    chip_to_eth_coord_mapping.insert(
                        {FabricNodeId(MeshId{mesh_id}, chip_id),
                         cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
                }
            }
        }
        tt::tt_metal::MetalContext::instance().set_custom_fabric_topology(
            physical_mesh_config.mesh_descriptor_path, chip_to_eth_coord_mapping);

        // ensure user specified matches what control plane sees
        const auto user_mesh_id =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_user_physical_mesh_ids()[0];
        TT_FATAL(
            *user_mesh_id == *local_mesh_id,
            "Local mesh id {} does not not match user mesh id {}",
            *user_mesh_id,
            *local_mesh_id);

        local_host_rank_ = tt::tt_metal::MetalContext::instance().get_control_plane().get_local_host_rank_id_binding();
    }

    void open_devices_internal(
        tt::tt_fabric::FabricConfig fabric_config,
        tt_fabric::FabricTensixConfig fabric_tensix_config,
        tt_fabric::FabricReliabilityMode reliability_mode,
        const TestFabricSetup& fabric_setup) {
        // Set fabric config FIRST, before any control plane access, this will reset control plane in metal context
        // Create FabricRouterConfig if max_packet_size is specified
        tt::tt_fabric::FabricRouterConfig router_config{};
        if (fabric_setup.max_packet_size.has_value()) {
            router_config.max_packet_payload_size_bytes = fabric_setup.max_packet_size.value();
        }

        tt::tt_fabric::SetFabricConfig(
            fabric_config,
            reliability_mode,
            std::nullopt,
            fabric_tensix_config,
            tt::tt_fabric::FabricUDMMode::DISABLED,
            tt::tt_fabric::FabricManagerMode::DEFAULT,
            router_config);

        // Now it's safe to initialize control plane (will use correct mesh graph descriptor)
        // first need to re-init contorl plane so that it checks out the latest fabric config.
        tt::tt_metal::MetalContext::instance().initialize_control_plane();
        local_host_rank_ = tt::tt_metal::MetalContext::instance().get_control_plane().get_local_host_rank_id_binding();

        // Initialize mesh and device info that was deferred from init()
        const auto user_meshes =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_user_physical_mesh_ids();
        TT_FATAL(
            user_meshes.size() == 1,
            "Only expected a single user mesh for a single host, but got: {}",
            user_meshes.size());

        local_mesh_id_ = user_meshes[0];

        available_mesh_ids_.insert(local_mesh_id_);
        mesh_shape_ = tt::tt_metal::MetalContext::instance().get_control_plane().get_physical_mesh_shape(
            local_mesh_id_, MeshScope::GLOBAL);

        const auto& mesh_graph = tt::tt_metal::MetalContext::instance().get_control_plane().get_mesh_graph();

        for (auto mesh_id : mesh_graph.get_mesh_ids()) {
            if (mesh_id == local_mesh_id_) {  // Populate all nodes available locally. Note the use of host rank to
                                              // ensure compatibility with big mesh
                for (auto chip : mesh_graph.get_chip_ids(mesh_id, local_host_rank_)) {
                    local_available_node_ids_.emplace_back(FabricNodeId(MeshId{mesh_id}, chip.value()));
                }
            }  // Populate Ids across all hosts and meshes
            for (auto chip : mesh_graph.get_chip_ids(mesh_id)) {
                global_available_node_ids_.emplace_back(FabricNodeId(MeshId{mesh_id}, chip.value()));
            }
        }

        mesh_device_ = MeshDevice::create(MeshDeviceConfig(mesh_shape_));
        // Now fabric context should be initialized, safe to query wrap_around_mesh
        wrap_around_mesh_ =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().is_wrap_around_mesh(
                user_meshes[0]);
        TT_FATAL(mesh_device_ != nullptr, "Failed to create MeshDevice with shape {}", mesh_shape_);

        current_fabric_config_ = fabric_config;
        current_fabric_tensix_config_ = fabric_tensix_config;
        current_fabric_reliability_mode_ = reliability_mode;
        current_max_packet_size_ = fabric_setup.max_packet_size;
        are_devices_open_ = true;
    }

    MeshCoordinate get_displacement(const MeshCoordinate& src_coords, const MeshCoordinate& dst_coords) const {
        TT_FATAL(
            src_coords.dims() == dst_coords.dims(),
            "Cannot find distance from coords with different dimensions: {} != {}",
            src_coords.dims(),
            dst_coords.dims());

        std::vector<uint32_t> coords(src_coords.dims(), 0);
        for (size_t i = 0; i < src_coords.dims(); ++i) {
            coords[i] = dst_coords[i] - src_coords[i];
        }
        return MeshCoordinate(coords);
    }

    MeshCoordinateRange get_coords_from_range(
        const MeshCoordinate& src_coords, const MeshCoordinate& dst_coords) const {
        const auto start = std::min(src_coords, dst_coords);
        const auto end = std::max(src_coords, dst_coords);
        return MeshCoordinateRange(start, end);
    }

    void adjust_hops_for_toroidal_links(std::unordered_map<RoutingDirection, uint32_t>& hops, uint32_t dim) const {
        RoutingDirection forward_direction = (dim == EW_DIM) ? RoutingDirection::E : RoutingDirection::S;
        RoutingDirection backward_direction = (dim == EW_DIM) ? RoutingDirection::W : RoutingDirection::N;

        if (hops[forward_direction] > mesh_shape_[dim] / 2) {
            hops[backward_direction] = mesh_shape_[dim] - hops[forward_direction];
            hops[forward_direction] = 0;
        } else if (hops[backward_direction] > mesh_shape_[dim] / 2) {
            hops[forward_direction] = mesh_shape_[dim] - hops[backward_direction];
            hops[backward_direction] = 0;
        }
    }

    std::unordered_map<RoutingDirection, uint32_t> get_hops_from_displacement(
        const MeshCoordinate& displacement) const {
        std::unordered_map<RoutingDirection, uint32_t> hops;
        for (const auto& direction : FabricContext::routing_directions) {
            hops[direction] = 0;
        }

        if (displacement[EW_DIM] >= mesh_shape_[EW_DIM]) {
            // wrapped around, negative distance
            hops[RoutingDirection::W] = std::numeric_limits<uint32_t>::max() - displacement[EW_DIM] + 1;
        } else {
            hops[RoutingDirection::E] = displacement[EW_DIM];
        }

        // positive y is south direction in ctrl plane
        if (displacement[NS_DIM] >= mesh_shape_[NS_DIM]) {
            // wrapped around, negative distance
            hops[RoutingDirection::N] = std::numeric_limits<uint32_t>::max() - displacement[NS_DIM] + 1;
        } else {
            hops[RoutingDirection::S] = displacement[NS_DIM];
        }

        // If toroidal wraparound connections have been enabled, we need to check whether going in an opposite direction
        // over the wraparound link is faster
        auto fabric_type = tt::tt_fabric::get_fabric_type(current_fabric_config_);
        // If the physical topology is a mesh, no wraparound links are present, so we can return the hops as is
        if (fabric_type == tt::tt_fabric::FabricType::MESH) {
            return hops;
        }

        // Wraparound links in the X dimension
        if (fabric_type == tt::tt_fabric::FabricType::TORUS_X || fabric_type == tt::tt_fabric::FabricType::TORUS_XY) {
            adjust_hops_for_toroidal_links(hops, EW_DIM);
        }

        // Wraparound links in the Y dimension
        if (fabric_type == tt::tt_fabric::FabricType::TORUS_Y || fabric_type == tt::tt_fabric::FabricType::TORUS_XY) {
            adjust_hops_for_toroidal_links(hops, NS_DIM);
        }

        return hops;
    }

    // In a multi-host setup, we currently dont store info about how the meshes are connected across hosts
    // So we only compute destinations within the local mesh from hops
    std::vector<FabricNodeId> compute_destination_nodes_from_hops(
        const FabricNodeId& src_node,
        const MeshCoordinate& src_coord,
        const std::unordered_map<RoutingDirection, uint32_t>& hops,
        ChipSendType send_type) const {
        // for now src_node is only passed for multicast, since we dont allow unicast hop expansion across hosts
        if (send_type == ChipSendType::CHIP_UNICAST) {
            return compute_unicast_destinations(src_coord, hops);
        }
        if (send_type == ChipSendType::CHIP_MULTICAST) {
            return compute_multicast_destinations(src_node, src_coord, hops);
        }
        TT_THROW("Unsupported send type: {}", send_type);
        return {};
    }

    std::vector<FabricNodeId> compute_unicast_destinations(
        const MeshCoordinate& src_coord, const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
        // Validation
        std::vector<RoutingDirection> non_zero_dirs;
        for (const auto& [dir, count] : hops) {
            if (count > 0) {
                non_zero_dirs.push_back(dir);
            }
        }

        if (is_2D_routing_enabled()) {
            TT_FATAL(non_zero_dirs.size() <= 2, "2D fabric supports at most 2 directions for unicast");
        } else {
            TT_FATAL(non_zero_dirs.size() <= 1, "1D fabric supports at most 1 direction for unicast");
        }

        bool has_ns = false, has_ew = false;
        bool opp_ns =
            (hops.contains(RoutingDirection::N) && hops.contains(RoutingDirection::S) &&
             hops.at(RoutingDirection::N) > 0 && hops.at(RoutingDirection::S) > 0);
        bool opp_ew =
            (hops.contains(RoutingDirection::E) && hops.contains(RoutingDirection::W) &&
             hops.at(RoutingDirection::E) > 0 && hops.at(RoutingDirection::W) > 0);

        if (opp_ns || opp_ew) {
            TT_THROW("Unicast cannot have opposing directions in the same dimension");
        }

        for (const auto& dir : non_zero_dirs) {
            if (dir == RoutingDirection::N || dir == RoutingDirection::S) {
                has_ns = true;
            } else {
                has_ew = true;
            }
        }
        if (non_zero_dirs.size() == 2 && !(has_ns && has_ew)) {
            TT_THROW("Unicast with 2 directions must be one from NS_DIM and one from EW_DIM");
        }

        // Determine major/minor
        RoutingDirection major = RoutingDirection::N;  // Default, will be set
        RoutingDirection minor = RoutingDirection::N;  // Default
        uint32_t major_hops = 0, minor_hops = 0;

        if (non_zero_dirs.size() == 1) {
            major = non_zero_dirs[0];
            major_hops = hops.at(major);
        } else if (non_zero_dirs.size() == 2) {
            // NS is major
            for (auto dir : non_zero_dirs) {
                if (dir == RoutingDirection::N || dir == RoutingDirection::S) {
                    major = dir;
                    major_hops = hops.at(dir);
                } else {
                    minor = dir;
                    minor_hops = hops.at(dir);
                }
            }
        }

        // Simulate linear path
        MeshCoordinate current = src_coord;
        auto major_path = simulate_linear_path(current, major, major_hops);
        if (!major_path.empty()) {
            current = major_path.back();
        }
        auto minor_path = simulate_linear_path(current, minor, minor_hops);
        if (!minor_path.empty()) {
            current = minor_path.back();
        }

        TT_FATAL(current != src_coord, "Unicast invalid: Destination is source after hops");

        return {get_fabric_node_id(current)};
    }

    std::vector<FabricNodeId> compute_multicast_destinations(
        const FabricNodeId& src_node,
        const MeshCoordinate& src_coord,
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
        // src_node is needed to grab the right mesh id to convert from coord to node id
        // Assume hops is pre-split single map from builder - simulate directly
        auto visited = simulate_multicast_split(src_coord, hops);

        std::unordered_set<FabricNodeId> unique_nodes;
        for (const auto& coord : visited) {
            if (coord != src_coord) {
                unique_nodes.insert(get_fabric_node_id(src_node.mesh_id, coord));
            }
        }

        return {unique_nodes.begin(), unique_nodes.end()};
    }

    std::vector<MeshCoordinate> simulate_multicast_split(
        const MeshCoordinate& start, const std::unordered_map<RoutingDirection, uint32_t>& split_hops) const {
        std::vector<MeshCoordinate> visited;
        std::unordered_set<MeshCoordinate> seen;  // For internal dedup

        // Check for actual non-zero hops to handle possible zero entries in map
        bool is_grid = ((split_hops.contains(RoutingDirection::N) && split_hops.at(RoutingDirection::N) > 0) ||
                        (split_hops.contains(RoutingDirection::S) && split_hops.at(RoutingDirection::S) > 0)) &&
                       ((split_hops.contains(RoutingDirection::E) && split_hops.at(RoutingDirection::E) > 0) ||
                        (split_hops.contains(RoutingDirection::W) && split_hops.at(RoutingDirection::W) > 0));

        if (!is_grid) {
            // Find the single non-zero direction for linear sim; throw if multiple or none
            std::optional<RoutingDirection> dir = std::nullopt;
            uint32_t count = 0;
            for (const auto& [d, c] : split_hops) {
                if (c > 0) {
                    if (dir.has_value()) {
                        TT_THROW("Linear multicast map has multiple non-zero directions: invalid");
                    }
                    dir = d;
                    count = c;
                }
            }
            TT_FATAL(dir.has_value(), "Linear multicast map has no non-zero directions: invalid");

            auto path = simulate_linear_path(start, dir.value(), count);
            for (const auto& coord : path) {
                if (seen.insert(coord).second) {
                    visited.push_back(coord);
                } else {
                    TT_THROW(
                        "Duplicate coordinate detected during multicast simulation at {} in direction {}. This may "
                        "indicate a cycle or invalid hop configuration.",
                        coord,
                        dir);
                }
            }
        } else {
            // Grid: trunk simulation, branch spines after each trunk hop (not from start)
            RoutingDirection trunk_dir = get_trunk_direction(split_hops);
            uint32_t trunk_hops = split_hops.at(trunk_dir);
            auto trunk_path = simulate_linear_path(start, trunk_dir, trunk_hops);
            for (const auto& coord : trunk_path) {
                if (seen.insert(coord).second) {
                    visited.push_back(coord);
                } else {
                    TT_THROW(
                        "Duplicate coordinate detected during multicast simulation at {} in direction {}. This may "
                        "indicate a cycle or invalid hop configuration.",
                        coord,
                        trunk_dir);
                }

                // Branch spines from this trunk position
                for (auto spine_dir : {RoutingDirection::E, RoutingDirection::W}) {
                    if (!split_hops.contains(spine_dir) || split_hops.at(spine_dir) == 0) {
                        continue;
                    }
                    uint32_t spine_count = split_hops.at(spine_dir);
                    auto spine_path = simulate_linear_path(coord, spine_dir, spine_count);
                    for (const auto& spine_coord : spine_path) {
                        if (seen.insert(spine_coord).second) {
                            visited.push_back(spine_coord);
                        } else {
                            TT_THROW(
                                "Duplicate coordinate detected during multicast simulation at {} in direction {}. This "
                                "may indicate a cycle or invalid hop configuration.",
                                spine_coord,
                                spine_dir);
                        }
                    }
                }
            }
        }

        return visited;
    }

    int32_t get_step_for_direction(RoutingDirection dir) const {
        switch (dir) {
            case RoutingDirection::N:
            case RoutingDirection::W: return -1;
            case RoutingDirection::S:
            case RoutingDirection::E: return 1;
            default: return 0;
        }
    }

    int32_t get_dim_for_direction(RoutingDirection dir) const {
        switch (dir) {
            case RoutingDirection::N:
            case RoutingDirection::S: return NS_DIM;
            case RoutingDirection::E:
            case RoutingDirection::W: return EW_DIM;
            default: return -1;
        }
    }

    MeshCoordinate::BoundaryMode get_boundary_mode_for_dimension(int32_t dim) const {
        if (topology_ == Topology::NeighborExchange || topology_ == Topology::Ring || topology_ == Topology::Torus) {
            auto fabric_type = tt::tt_fabric::get_fabric_type(current_fabric_config_);
            switch (fabric_type) {
                case tt::tt_fabric::FabricType::TORUS_X:
                    return (dim == EW_DIM) ? MeshCoordinate::BoundaryMode::WRAP : MeshCoordinate::BoundaryMode::NONE;
                case tt::tt_fabric::FabricType::TORUS_Y:
                    return (dim == NS_DIM) ? MeshCoordinate::BoundaryMode::WRAP : MeshCoordinate::BoundaryMode::NONE;
                case tt::tt_fabric::FabricType::TORUS_XY: return MeshCoordinate::BoundaryMode::WRAP;
                default: return MeshCoordinate::BoundaryMode::NONE;
            }
        }
        return MeshCoordinate::BoundaryMode::NONE;
    }

    RoutingDirection get_trunk_direction(const std::unordered_map<RoutingDirection, uint32_t>& split_hops) const {
        if (split_hops.contains(RoutingDirection::N) && split_hops.at(RoutingDirection::N) > 0) {
            return RoutingDirection::N;
        }
        if (split_hops.contains(RoutingDirection::S) && split_hops.at(RoutingDirection::S) > 0) {
            return RoutingDirection::S;
        }
        // If no NS, assume not a grid or handle error
        TT_THROW("No trunk direction found in split_hops");
        return RoutingDirection::N;  // Unreachable
    }

    // Add this before simulate_direction_hops or in private section
    std::vector<MeshCoordinate> simulate_linear_path(
        const MeshCoordinate& start, RoutingDirection dir, uint32_t count) const {
        std::vector<MeshCoordinate> path;
        if (count == 0) {
            return path;
        }
        path.reserve(count);

        int32_t step = get_step_for_direction(dir);
        int32_t dim = get_dim_for_direction(dir);
        auto mode = get_boundary_mode_for_dimension(dim);

        MeshCoordinate current = start;
        for (uint32_t i = 0; i < count; ++i) {
            auto next_opt = current.get_neighbor(mesh_shape_, step, dim, mode);
            if (!next_opt.has_value()) {
                TT_THROW("Linear path invalid: Boundary exceeded in direction {} at {}", dir, current);
            }
            current = next_opt.value();
            path.push_back(current);
        }
        return path;
    }
};

}  // namespace tt::tt_fabric::fabric_tests
