// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <optional>
#include <random>

#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/system_mesh.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common_types.hpp"

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
using SystemMesh = tt::tt_metal::distributed::SystemMesh;
using MeshDeviceConfig = tt::tt_metal::distributed::MeshDeviceConfig;

using Topology = tt::tt_fabric::Topology;

namespace tt::tt_fabric {
namespace fabric_tests {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // Simple mixing
        return h1 ^ (h2 << 1);
    }
};

class TestFixture : public IDeviceInfoProvider, public IRouteManager {
    static constexpr uint32_t ROW_DIM = 0;
    static constexpr uint32_t COL_DIM = 1;

    // mapping to convert coords to directions
    static constexpr uint32_t EW_DIM = 1;
    static constexpr uint32_t NS_DIM = 0;

    static const std::unordered_map<std::pair<Topology, RoutingType>, tt::tt_fabric::FabricConfig, pair_hash>
        topology_to_fabric_config_map;

public:
    void init() {
        // NOTE: We defer all control plane access until open_devices_internal
        // to ensure fabric config is set first, which affects mesh graph descriptor selection
        current_fabric_config_ = tt::tt_fabric::FabricConfig::DISABLED;
    }

    std::vector<MeshCoordinate> get_available_device_coordinates() const { return this->available_device_coordinates_; }

    void open_devices(Topology topology, RoutingType routing_type) {
        auto it = topology_to_fabric_config_map.find({topology, routing_type});
        TT_FATAL(
            it != topology_to_fabric_config_map.end(),
            "Unsupported topology: {} with routing type: {}",
            topology,
            routing_type);
        auto new_fabric_config = it->second;
        if (new_fabric_config != current_fabric_config_) {
            if (are_devices_open_) {
                log_info(tt::LogTest, "Closing devices and switching to new fabric config: {}", new_fabric_config);
                close_devices();
            }
            open_devices_internal(new_fabric_config);

            topology_ = topology;
            routing_type_ = routing_type;
        } else {
            log_info(tt::LogTest, "Reusing existing device setup with fabric config: {}", current_fabric_config_);
        }
    }

    void setup_workload() {
        // create a new mesh workload for every run
        mesh_workload_ = std::make_unique<MeshWorkload>();
    }

    void enqueue_program(const MeshCoordinate& mesh_coord, tt::tt_metal::Program program) {
        MeshCoordinateRange device(mesh_coord, mesh_coord);
        tt::tt_metal::distributed::AddProgramToMeshWorkload(*mesh_workload_, std::move(program), device);
    }

    void run_programs() {
        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *mesh_workload_, true);
    }

    void wait_for_programs() { tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue()); }

    void close_devices() {
        mesh_device_->close();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);

        // Clear all class members
        control_plane_ptr_ = nullptr;
        mesh_coordinate_to_node_id_.clear();
        node_id_to_mesh_coordinate_.clear();
        available_device_coordinates_.clear();
        available_node_ids_.clear();
        available_mesh_ids_.clear();
        mesh_device_.reset();
        mesh_workload_.reset();
        current_fabric_config_ = tt::tt_fabric::FabricConfig::DISABLED;
        are_devices_open_ = false;
    }

    // ======================================================================================
    // IDeviceInfoProvider methods
    // ======================================================================================
    FabricNodeId get_fabric_node_id(const chip_id_t physical_chip_id) const override {
        return control_plane_ptr_->get_fabric_node_id_from_physical_chip_id(physical_chip_id);
    }

    FabricNodeId get_fabric_node_id(const MeshCoordinate& device_coord) const override {
        return mesh_coordinate_to_node_id_.at(device_coord);
    }

    FabricNodeId get_fabric_node_id(MeshId mesh_id, const MeshCoordinate& device_coord) const override {
        TT_FATAL(
            available_mesh_ids_.count(mesh_id) > 0,
            "Mesh id: {} is not available for querying fabric node id",
            mesh_id);
        TT_FATAL(
            mesh_coordinate_to_node_id_.count(device_coord) > 0,
            "Mesh coordinate: {} is not available for querying fabric node id",
            device_coord);
        return mesh_coordinate_to_node_id_.at(device_coord);
    }

    MeshCoordinate get_device_coord(const FabricNodeId& node_id) const override {
        auto it = node_id_to_mesh_coordinate_.find(node_id);
        TT_FATAL(it != node_id_to_mesh_coordinate_.end(), "Unknown node id: {} for querying mesh coord", node_id);

        return it->second;
    }

    uint32_t get_worker_noc_encoding(const MeshCoordinate& device_coord, const CoreCoord logical_core) const override {
        auto* device = mesh_device_->get_device(device_coord);
        const auto virtual_core = device->worker_core_from_logical_core(logical_core);
        return tt_metal::MetalContext::instance().hal().noc_xy_encoding(virtual_core.x, virtual_core.y);
    }

    uint32_t get_worker_noc_encoding(const FabricNodeId& node_id, const CoreCoord logical_core) const override {
        const auto& device_coord = get_device_coord(node_id);
        return get_worker_noc_encoding(device_coord, logical_core);
    }

    CoreCoord get_worker_grid_size() const override { return mesh_device_->compute_with_storage_grid_size(); }

    uint32_t get_worker_id(const FabricNodeId& node_id, CoreCoord logical_core) const override {
        return (*node_id.mesh_id << 12) | (node_id.chip_id << 8) | (logical_core.x << 4) | (logical_core.y);
    }

    std::vector<FabricNodeId> get_all_node_ids() const override { return available_node_ids_; }

    uint32_t get_l1_unreserved_base() const override {
        return tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    }

    uint32_t get_l1_unreserved_size() const override { return tt::tt_metal::hal::get_max_worker_l1_unreserved_size(); }

    uint32_t get_l1_alignment() const override { return tt::tt_metal::hal::get_l1_alignment(); }

    uint32_t get_max_payload_size_bytes() const override {
        return control_plane_ptr_->get_fabric_context().get_fabric_max_payload_size_bytes();
    }

    bool is_2d_fabric() const override { return topology_ == Topology::Mesh; }

    bool use_dynamic_routing() const override { return routing_type_ == RoutingType::Dynamic; }

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
        const FabricNodeId& src_node_id, RoutingDirection initial_direction, uint32_t total_hops) const {
        std::vector<std::pair<FabricNodeId, RoutingDirection>> ring_path;

        // Check if this is a wrap-around mesh
        bool is_wrap_around = wrap_around_mesh(src_node_id);

        if (is_wrap_around) {
            // Use the existing wrap-around mesh logic
            ring_path = trace_wrap_around_mesh_ring_path(src_node_id, initial_direction, total_hops);
        } else {
            // Use the new non wrap-around mesh logic
            ring_path = trace_non_wrap_around_mesh_ring_path(src_node_id, initial_direction, total_hops);
        }

        std::vector<FabricNodeId> ring_destinations;
        ring_destinations.reserve(total_hops);

        // Extract destination nodes (skip the source node, get the next nodes in path)
        for (const auto& [current_node, direction] : ring_path) {
            ring_destinations.push_back(current_node);
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
        auto shard_params = ShardSpecBuffer(all_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

        auto buffer_distribution_spec =
            BufferDistributionSpec(Shape{num_cores, 1}, Shape{1, 1}, all_cores, ShardOrientation::ROW_MAJOR);

        auto buffer_page_mapping = buffer_distribution_spec.compute_page_mapping();

        DeviceLocalBufferConfig buffer_specs = {
            .page_size = size_bytes,
            .buffer_type = BufferType::L1,
            .sharding_args =
                BufferShardingArgs(buffer_distribution_spec, shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };

        MeshBufferConfig mesh_buffer_specs = ReplicatedBufferConfig{
            .size = total_size,
        };
        auto mesh_buffer = MeshBuffer::create(mesh_buffer_specs, buffer_specs, mesh_device_.get(), address);

        return mesh_buffer;
    }

    // Data reading helpers
    std::unordered_map<CoreCoord, std::vector<uint32_t>> read_buffer_from_cores(
        const MeshCoordinate& device_coord,
        const std::vector<CoreCoord>& cores,
        uint32_t address,
        uint32_t size_bytes) const override {
        auto mesh_buffer = create_mesh_buffer_helper(cores, address, size_bytes);

        const auto& buffer_distribution_spec =
            mesh_buffer->device_local_config().sharding_args.buffer_distribution_spec();
        TT_FATAL(buffer_distribution_spec.has_value(), "Buffer distribution spec is not set");
        const auto buffer_page_mapping = buffer_distribution_spec->compute_page_mapping();

        auto total_size = size_bytes * buffer_page_mapping.all_cores.size();
        std::vector<uint32_t> data;
        data.resize(total_size / sizeof(uint32_t));
        tt::tt_metal::distributed::ReadShard(mesh_device_->mesh_command_queue(), data, mesh_buffer, device_coord);

        // splice up data into map
        std::unordered_map<CoreCoord, std::vector<uint32_t>> results;
        auto num_words_per_core = size_bytes / sizeof(uint32_t);

        for (auto i = 0; i < buffer_page_mapping.all_cores.size(); i++) {
            const auto& core = buffer_page_mapping.all_cores[i];
            const auto& page_indices = buffer_page_mapping.core_host_page_indices[i];
            std::vector<uint32_t> core_data;
            core_data.reserve(page_indices.size() * num_words_per_core);
            for (const auto& page_idx : page_indices) {
                if (page_idx == tt::tt_metal::UncompressedBufferPageMapping::PADDING) {
                    continue;
                }
                auto start_idx = page_idx * num_words_per_core;
                auto end_idx = start_idx + num_words_per_core;
                core_data.insert(core_data.end(), data.begin() + start_idx, data.begin() + end_idx);
            }
            results.emplace(core, core_data);
        }

        return results;
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
            return get_ring_topology_dst_node_ids(src_node, initial_direction, total_hops);
        }

        std::vector<FabricNodeId> dst_nodes;
        bool use_displacement_for_dst_nodes =
            chip_send_type == ChipSendType::CHIP_UNICAST || this->topology_ == Topology::Linear;

        if (use_displacement_for_dst_nodes) {
            const MeshCoordinate& src_coord = get_device_coord(src_node);
            auto displacements = convert_hops_to_displacement(hops);
            for (const auto& displacement : displacements) {
                // Ignore zero-length displacements that can occur for some directions in the hops map
                if (displacement == MeshCoordinate::zero_coordinate(displacement.dims())) {
                    continue;
                }

                const auto dst_coord = get_coord_from_displacement(src_coord, displacement);

                if (chip_send_type == ChipSendType::CHIP_UNICAST) {
                    // For unicast, we only care about the final destination of each displacement vector.
                    dst_nodes.push_back(get_fabric_node_id(dst_coord));
                } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
                    // For multicast, we care about all nodes along the path.
                    const auto coords_in_path = get_coords_from_range(src_coord, dst_coord);
                    for (const auto& coord : coords_in_path) {
                        if (coord == src_coord) {
                            continue;  // Don't include the source itself
                        }
                        dst_nodes.push_back(get_fabric_node_id(coord));
                    }
                }
            }
        } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
            dst_nodes = get_mesh_topology_dst_node_ids(src_node, hops);
        }
        return dst_nodes;
    }

    bool are_devices_linear(const std::vector<FabricNodeId>& node_ids) const override {
        if (node_ids.size() <= 1) {
            return true;
        }

        auto first_coord = node_id_to_mesh_coordinate_.at(node_ids[0]);
        bool all_same_row = true;
        bool all_same_col = true;

        for (size_t i = 1; i < node_ids.size(); ++i) {
            auto next_coord = node_id_to_mesh_coordinate_.at(node_ids[i]);
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
        auto unpaired = get_all_node_ids();

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
        const auto device_ids = get_all_node_ids();
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

    std::optional<std::pair<FabricNodeId, FabricNodeId>> get_wrap_around_mesh_ring_neighbors(
        const FabricNodeId& src_node, const std::vector<FabricNodeId>& devices) const override {
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
        chip_id_t forward_chip_id, backward_chip_id;

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
        } else if (row == 0) {
            // Top row (not corners): forward=right, backward=left
            forward_chip_id = src_node.chip_id + 1;
            backward_chip_id = src_node.chip_id - 1;
        } else if (col == mesh_width - 1) {
            // Right column (not corners): forward=up, backward=down
            forward_chip_id = src_node.chip_id - mesh_width;
            backward_chip_id = src_node.chip_id + mesh_width;
        } else if (row == mesh_height - 1) {
            // Bottom row (not corners): forward=right, backward=left
            forward_chip_id = src_node.chip_id + 1;
            backward_chip_id = src_node.chip_id - 1;
        } else if (col == 0) {
            // Left column (not corners): forward=up, backward=down
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

    uint32_t get_num_sync_devices(bool wrap_around_mesh = true) const override {
        uint32_t num_devices;
        switch (topology_) {
            case tt::tt_fabric::Topology::Linear: {
                num_devices = mesh_shape_[NS_DIM] + mesh_shape_[EW_DIM] - 1;
                return num_devices;
            }
            case tt::tt_fabric::Topology::Ring: {
                if (wrap_around_mesh) {
                    // sync using full ring mcast, ie, mcast on both forward/backward path.
                    num_devices = 2 * (mesh_shape_[NS_DIM] - 1 + mesh_shape_[EW_DIM] - 1);
                } else {
                    num_devices = mesh_shape_[NS_DIM] + mesh_shape_[EW_DIM] - 1;
                }
                return num_devices;
            }
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
        uint32_t full_hop_count = 2 * (mesh_shape_[NS_DIM] - 1 + mesh_shape_[EW_DIM] - 1) - 1;

        if (pattern_type == HighLevelTrafficPattern::FullRingMulticast) {
            num_forward_hops = full_hop_count;
            num_backward_hops = full_hop_count;
        } else if (pattern_type == HighLevelTrafficPattern::HalfRingMulticast) {
            num_forward_hops = tt::div_up(full_hop_count, 2);
            num_backward_hops = full_hop_count - num_forward_hops;
            if (src_node_id.chip_id % 2 == 0) {
                std::swap(num_forward_hops, num_backward_hops);
            }
        } else {
            TT_THROW(
                "Unsupported pattern type for ring multicast: only FullRingMulticast and HalfRingMulticast are "
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

        if (pattern_type == HighLevelTrafficPattern::FullRingMulticast) {
            num_forward_hops = full_hop_count;
            num_backward_hops = full_hop_count;
        } else if (pattern_type == HighLevelTrafficPattern::HalfRingMulticast) {
            num_forward_hops = tt::div_up(full_hop_count, 2);
            num_backward_hops = full_hop_count - num_forward_hops;
            if (src_node_id.chip_id % 2 == 0) {
                std::swap(num_forward_hops, num_backward_hops);
            }
        } else {
            TT_THROW(
                "Unsupported pattern type for ring multicast: only FullRingMulticast and HalfRingMulticast are "
                "supported");
        }

        hops[direction_forward] = num_forward_hops;
        hops[direction_backward] = num_backward_hops;

        return hops;
    }

    std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_multicast_hops(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_hops;
        if (this->topology_ == Topology::Linear || this->topology_ == Topology::Ring) {
            split_hops.reserve(4);
            for (const auto& [dir, hop_count] : hops) {
                if (hop_count > 0) {
                    split_hops.push_back({{dir, hop_count}});
                }
            }
        } else if (this->topology_ == Topology::Mesh) {
            // For mesh topology, handle all cases including three-entry case
            split_hops.reserve(8);

            auto north_hops = hops.count(RoutingDirection::N) > 0 ? hops.at(RoutingDirection::N) : 0;
            auto south_hops = hops.count(RoutingDirection::S) > 0 ? hops.at(RoutingDirection::S) : 0;
            auto east_hops = hops.count(RoutingDirection::E) > 0 ? hops.at(RoutingDirection::E) : 0;
            auto west_hops = hops.count(RoutingDirection::W) > 0 ? hops.at(RoutingDirection::W) : 0;

            // East/West hops always get their own separate entries
            if (east_hops > 0) {
                split_hops.push_back({{RoutingDirection::E, east_hops}});
            }
            if (west_hops > 0) {
                split_hops.push_back({{RoutingDirection::W, west_hops}});
            }

            // Individual north/south directions (only if no east/west)
            if (north_hops > 0 && east_hops == 0 && west_hops == 0) {
                split_hops.push_back({{RoutingDirection::N, north_hops}});
            }
            if (south_hops > 0 && east_hops == 0 && west_hops == 0) {
                split_hops.push_back({{RoutingDirection::S, south_hops}});
            }

            // Two-direction combinations
            if (north_hops > 0 && east_hops > 0 && west_hops == 0) {
                split_hops.push_back({{RoutingDirection::N, north_hops}, {RoutingDirection::E, east_hops}});
            }
            if (north_hops > 0 && west_hops > 0 && east_hops == 0) {
                split_hops.push_back({{RoutingDirection::N, north_hops}, {RoutingDirection::W, west_hops}});
            }
            if (south_hops > 0 && east_hops > 0 && west_hops == 0) {
                split_hops.push_back({{RoutingDirection::S, south_hops}, {RoutingDirection::E, east_hops}});
            }
            if (south_hops > 0 && west_hops > 0 && east_hops == 0) {
                split_hops.push_back({{RoutingDirection::S, south_hops}, {RoutingDirection::W, west_hops}});
            }

            // Three-direction case (north/south + east + west)
            if (north_hops > 0 && east_hops > 0 && west_hops > 0) {
                split_hops.push_back(
                    {{RoutingDirection::N, north_hops},
                     {RoutingDirection::E, east_hops},
                     {RoutingDirection::W, west_hops}});
            }
            if (south_hops > 0 && east_hops > 0 && west_hops > 0) {
                split_hops.push_back(
                    {{RoutingDirection::S, south_hops},
                     {RoutingDirection::E, east_hops},
                     {RoutingDirection::W, west_hops}});
            }
        } else {
            TT_THROW("Unsupported topology: {} for split_multicast_hops", this->topology_);
        }
        return split_hops;
    }

    FabricNodeId get_random_unicast_destination(FabricNodeId src_node_id, std::mt19937& gen) const override {
        auto all_devices = this->get_all_node_ids();
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
        return control_plane_ptr_->get_fabric_context().is_wrap_around_mesh(node.mesh_id);
    }

    RoutingDirection get_forwarding_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const override {
        auto forwarding_direction = control_plane_ptr_->get_forwarding_direction(src_node_id, dst_node_id);
        TT_FATAL(
            forwarding_direction.has_value(), "No forwarding direction found for {} -> {}", src_node_id, dst_node_id);
        return forwarding_direction.value();
    }

    RoutingDirection get_forwarding_direction(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        if (topology_ == Topology::Linear || topology_ == Topology::Ring) {
            // for 1d, we expect hops only in one direction to be non-zero
            for (const auto& [direction, hop] : hops) {
                if (hop != 0) {
                    return direction;
                }
            }
        } else if (topology_ == Topology::Mesh) {
            // TODO: update this logic once 2D mcast is supported
            // for now we return the first direction that is non-zero
            // for 2D, since we use dimension order routing, lookup the directions in the order of N, S, E, W
            for (const auto& direction : FabricContext::routing_directions) {
                if (hops.count(direction) > 0 && hops.at(direction) != 0) {
                    return direction;
                }
            }
        } else {
            TT_THROW("Unsupported topology: {} for get_forwarding_direction", topology_);
        }

        TT_THROW("Failed to find a forwarding direction for hops: {}", hops);
    }

    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id,
        const FabricNodeId& dst_node_id,
        const RoutingDirection& direction) const override {
        return tt::tt_fabric::get_forwarding_link_indices_in_direction(src_node_id, dst_node_id, direction);
    }

    FabricNodeId get_neighbor_node_id(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
        const auto& neighbors = control_plane_ptr_->get_chip_neighbors(src_node_id, direction);
        TT_FATAL(neighbors.size() == 1, "Expected only neighbor mesh for {} in direction: {}", src_node_id, direction);
        TT_FATAL(
            neighbors.begin()->second.size() >= 1,
            "Expected at least 1 neighbor chip for {} in direction: {}",
            src_node_id,
            direction);
        return FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
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
        bool wrap_around_mesh = control_plane_ptr_->get_fabric_context().is_wrap_around_mesh(src_device.mesh_id);

        switch (topology_) {
            case tt::tt_fabric::Topology::Ring: {
                if (wrap_around_mesh) {
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
                        src_device, dst_node_forward, dst_node_backward, HighLevelTrafficPattern::FullRingMulticast);

                } else {
                    // if not wrap around mesh, then need to get the neighbours on all directions.
                    auto ns_hops = this->get_full_or_half_ring_mcast_hops(
                        src_device, HighLevelTrafficPattern::FullRingMulticast, NS_DIM);
                    auto ew_hops = this->get_full_or_half_ring_mcast_hops(
                        src_device, HighLevelTrafficPattern::FullRingMulticast, EW_DIM);
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
                auto num_sync_devices = this->get_num_sync_devices(wrap_around_mesh);
                global_sync_val =
                    2 * num_sync_devices - 2;  // minus 2 because in a full ring pattern we dont mcast to self (twice).
                break;
            }
            case tt::tt_fabric::Topology::Linear: {
                multi_directional_hops = this->get_full_mcast_hops(src_device);
                global_sync_val = this->get_num_sync_devices() - 1;
                break;
            }
            case tt::tt_fabric::Topology::Mesh: {
                multi_directional_hops = this->get_full_mcast_hops(src_device);
                global_sync_val = this->get_num_sync_devices() - 1;
                break;
            }
            default: TT_THROW("Unsupported topology for line sync: {}", static_cast<int>(topology_));
        }

        return {std::move(multi_directional_hops), global_sync_val};
    }

    std::vector<FabricNodeId> get_mesh_topology_dst_node_ids(
        const FabricNodeId& src_node_id, const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
        std::vector<FabricNodeId> dst_nodes;
        std::unordered_set<FabricNodeId> seen_nodes;

        // Use the specialized splitting function that avoids three-entry case
        auto split_hops_list = split_hops_for_tracing(hops);

        // Trace each split separately
        for (const auto& split_hop : split_hops_list) {
            auto path_nodes = trace_mesh_single_path(src_node_id, split_hop);
            for (const auto& node : path_nodes) {
                // try to preserve the ordering during its creation (path traversal order)
                if (seen_nodes.find(node) == seen_nodes.end()) {
                    seen_nodes.insert(node);
                    dst_nodes.push_back(node);
                }
            }
        }

        return dst_nodes;
    }

    std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_hops_for_tracing(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
        std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_hops;

        auto north_hops = hops.count(RoutingDirection::N) > 0 ? hops.at(RoutingDirection::N) : 0;
        auto south_hops = hops.count(RoutingDirection::S) > 0 ? hops.at(RoutingDirection::S) : 0;
        auto east_hops = hops.count(RoutingDirection::E) > 0 ? hops.at(RoutingDirection::E) : 0;
        auto west_hops = hops.count(RoutingDirection::W) > 0 ? hops.at(RoutingDirection::W) : 0;

        // Case 1: Only east/west
        if ((north_hops == 0 && south_hops == 0) && (east_hops > 0 || west_hops > 0)) {
            split_hops.push_back(hops);
        }
        // Case 2: north/south + only one of east/west
        else if ((north_hops > 0 || south_hops > 0) && (east_hops > 0) != (west_hops > 0)) {
            split_hops.push_back(hops);
        }
        // Case 3: north/south + both east and west - split into two paths
        else if ((north_hops > 0 || south_hops > 0) && east_hops > 0 && west_hops > 0) {
            // Split into north/south + east
            std::unordered_map<RoutingDirection, uint32_t> path1;
            if (north_hops > 0) {
                path1[RoutingDirection::N] = north_hops;
            }
            if (south_hops > 0) {
                path1[RoutingDirection::S] = south_hops;
            }
            path1[RoutingDirection::E] = east_hops;
            split_hops.push_back(path1);

            // Split into north/south + west
            std::unordered_map<RoutingDirection, uint32_t> path2;
            if (north_hops > 0) {
                path2[RoutingDirection::N] = north_hops;
            }
            if (south_hops > 0) {
                path2[RoutingDirection::S] = south_hops;
            }
            path2[RoutingDirection::W] = west_hops;
            split_hops.push_back(path2);
        } else {
            // Default case: just use the original hops
            split_hops.push_back(hops);
        }

        return split_hops;
    }

    std::vector<FabricNodeId> trace_mesh_single_path(
        const FabricNodeId& src_node_id, const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
        std::vector<FabricNodeId> dst_nodes;
        auto remaining_hops = hops;  // Make a copy to modify
        FabricNodeId current_node = src_node_id;

        // Trace the path using dimension order routing
        while (true) {
            // Check if all remaining hops are 0
            bool all_hops_zero = true;
            for (const auto& [direction, hop_count] : remaining_hops) {
                if (hop_count > 0) {
                    all_hops_zero = false;
                    break;
                }
            }
            if (all_hops_zero) {
                break;  // No more hops to process
            }
            // Find the next direction to route in
            RoutingDirection next_direction = get_forwarding_direction(remaining_hops);

            // Check if we have any remaining hops in this direction
            if (remaining_hops.count(next_direction) == 0 || remaining_hops[next_direction] == 0) {
                break;  // No more hops to process
            }

            uint32_t hops_in_direction = remaining_hops[next_direction];

            // Trace all hops in this direction sequentially
            for (uint32_t hop = 0; hop < hops_in_direction; hop++) {
                // Move to next node in this direction
                current_node = get_neighbor_node_id(current_node, next_direction);
                dst_nodes.push_back(current_node);
            }

            // Mark this direction as completed
            remaining_hops[next_direction] = 0;
        }

        return dst_nodes;
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
                } else if (current_direction == RoutingDirection::S) {
                    if (current_coord[EW_DIM] == 0) {
                        current_direction = RoutingDirection::E;
                    } else {
                        current_direction = RoutingDirection::W;
                    }
                } else if (current_direction == RoutingDirection::N) {
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
    std::vector<std::pair<FabricNodeId, RoutingDirection>> trace_non_wrap_around_mesh_ring_path(
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
            bool need_wraparound = false;

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

    uint32_t get_max_routing_planes_for_device(const FabricNodeId& node_id) const override {
        // Find the minimum number of routing planes across all directions for this device
        uint32_t min_routing_planes = std::numeric_limits<uint32_t>::max();

        // Check all possible directions
        for (const auto& direction : FabricContext::routing_directions) {
            size_t routing_planes =
                control_plane_ptr_->get_num_available_routing_planes_in_direction(node_id, direction);
            if (routing_planes > 0) {  // Only consider directions that have routing planes
                min_routing_planes = std::min(min_routing_planes, static_cast<uint32_t>(routing_planes));
            }
        }

        // If no valid directions found, return 0
        return (min_routing_planes == std::numeric_limits<uint32_t>::max()) ? 0 : min_routing_planes;
    }

private:
    ControlPlane* control_plane_ptr_;
    Topology topology_;
    RoutingType routing_type_;
    MeshShape mesh_shape_;
    std::set<MeshId> available_mesh_ids_;
    tt::tt_fabric::FabricConfig current_fabric_config_;
    std::vector<MeshCoordinate> available_device_coordinates_;
    std::vector<FabricNodeId> available_node_ids_;
    std::shared_ptr<MeshDevice> mesh_device_;
    std::unordered_map<MeshCoordinate, FabricNodeId> mesh_coordinate_to_node_id_;
    std::unordered_map<FabricNodeId, MeshCoordinate> node_id_to_mesh_coordinate_;
    std::shared_ptr<MeshWorkload> mesh_workload_;
    bool are_devices_open_ = false;

    void open_devices_internal(tt::tt_fabric::FabricConfig fabric_config) {
        // Set fabric config FIRST, before any control plane access, this will reset control plane in metal context
        tt::tt_fabric::SetFabricConfig(fabric_config);

        // Now it's safe to initialize control plane (will use correct mesh graph descriptor)
        control_plane_ptr_ = &tt::tt_metal::MetalContext::instance().get_control_plane();

        // Initialize mesh and device info that was deferred from init()
        const auto user_meshes = control_plane_ptr_->get_user_physical_mesh_ids();
        TT_FATAL(
            user_meshes.size() == 1,
            "Only expected a single user mesh for a single host, but got: {}",
            user_meshes.size());

        // TODO: for now we are just dealing with user mesh 0 here
        available_mesh_ids_.insert(user_meshes[0]);
        mesh_shape_ = control_plane_ptr_->get_physical_mesh_shape(user_meshes[0]);
        const auto coordinates = MeshCoordinateRange(mesh_shape_);
        for (const auto& coord : coordinates) {
            available_device_coordinates_.push_back(coord);
        }

        // TODO: available node ids should be able to capture the node ids for other meshes as well
        const auto mesh_id = user_meshes[0];
        for (auto i = 0; i < available_device_coordinates_.size(); i++) {
            available_node_ids_.emplace_back(FabricNodeId(mesh_id, i));
        }

        mesh_device_ = MeshDevice::create(MeshDeviceConfig(mesh_shape_));

        // Now fabric context should be initialized, safe to query wrap_around_mesh
        bool wrap_around_mesh = control_plane_ptr_->get_fabric_context().is_wrap_around_mesh(user_meshes[0]);
        log_info(LogTest, "wrap_around_mesh {}", wrap_around_mesh);

        TT_FATAL(mesh_device_ != nullptr, "Failed to create MeshDevice with shape {}", mesh_shape_);

        for (const auto& coord : available_device_coordinates_) {
            TT_FATAL(
                coord.dims() == mesh_shape_.dims(),
                "Device coordinate {} has different dimensions than mesh shape {}",
                coord,
                mesh_shape_);

            // Validate coordinate bounds
            for (size_t i = 0; i < coord.dims(); ++i) {
                TT_FATAL(
                    coord[i] < mesh_shape_[i],
                    "Device coordinate {} is out of bounds for mesh shape {} (dimension {} >= {})",
                    coord,
                    mesh_shape_,
                    i,
                    mesh_shape_[i]);
            }

            auto* device = mesh_device_->get_device(coord);
            TT_FATAL(device != nullptr, "Failed to get device at coordinate {}", coord);

            const auto fabric_node_id = control_plane_ptr_->get_fabric_node_id_from_physical_chip_id(device->id());
            mesh_coordinate_to_node_id_.emplace(coord, fabric_node_id);
            node_id_to_mesh_coordinate_.emplace(fabric_node_id, coord);
        }

        current_fabric_config_ = fabric_config;
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

    MeshCoordinate get_coord_from_displacement(
        const MeshCoordinate& src_coords, const MeshCoordinate& displacement) const {
        std::vector<uint32_t> coords(src_coords.dims(), 0);
        for (size_t i = 0; i < src_coords.dims(); ++i) {
            coords[i] = src_coords[i] + displacement[i];
        }
        return MeshCoordinate(coords);
    }

    MeshCoordinateRange get_coords_from_range(
        const MeshCoordinate& src_coords, const MeshCoordinate& dst_coords) const {
        const auto start = std::min(src_coords, dst_coords);
        const auto end = std::max(src_coords, dst_coords);
        return MeshCoordinateRange(start, end);
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

        return hops;
    }

    MeshCoordinate get_displacement_from_hops(const std::vector<std::pair<RoutingDirection, uint32_t>>& hops) const {
        std::vector<uint32_t> displacement(mesh_shape_.dims(), 0);
        for (const auto& [direction, hop] : hops) {
            switch (direction) {
                case RoutingDirection::N: displacement[NS_DIM] -= hop; break;
                case RoutingDirection::S: displacement[NS_DIM] += hop; break;
                case RoutingDirection::E: displacement[EW_DIM] += hop; break;
                case RoutingDirection::W: displacement[EW_DIM] -= hop; break;
                default: break;
            }
        }
        return MeshCoordinate(displacement);
    }

    // this depends on the fabric topology
    // for 1D, we handle each direction separately
    // for 2D, we look at the combination of directions
    std::vector<MeshCoordinate> convert_hops_to_displacement(
        std::unordered_map<RoutingDirection, uint32_t>& hops) const {
        std::vector<MeshCoordinate> displacements;

        if (topology_ == Topology::Linear) {
            displacements.reserve(4);
            displacements.push_back(get_displacement_from_hops({{RoutingDirection::N, hops[RoutingDirection::N]}}));
            displacements.push_back(get_displacement_from_hops({{RoutingDirection::S, hops[RoutingDirection::S]}}));
            displacements.push_back(get_displacement_from_hops({{RoutingDirection::E, hops[RoutingDirection::E]}}));
            displacements.push_back(get_displacement_from_hops({{RoutingDirection::W, hops[RoutingDirection::W]}}));
        } else if (topology_ == Topology::Mesh) {
            displacements.reserve(4);
            displacements.push_back(get_displacement_from_hops(
                {{RoutingDirection::N, hops[RoutingDirection::N]}, {RoutingDirection::E, hops[RoutingDirection::E]}}));
            displacements.push_back(get_displacement_from_hops(
                {{RoutingDirection::N, hops[RoutingDirection::N]}, {RoutingDirection::W, hops[RoutingDirection::W]}}));
            displacements.push_back(get_displacement_from_hops(
                {{RoutingDirection::S, hops[RoutingDirection::S]}, {RoutingDirection::E, hops[RoutingDirection::E]}}));
            displacements.push_back(get_displacement_from_hops(
                {{RoutingDirection::S, hops[RoutingDirection::S]}, {RoutingDirection::W, hops[RoutingDirection::W]}}));
        } else {
            TT_THROW("Unsupported topology: {}", topology_);
        }
        return displacements;
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
