// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_tensix_builder.hpp"
#include "fabric_tensix_builder_impl.hpp"

#include <tt_stl/assert.hpp>
#include <hal.hpp>
#include <tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/core_descriptor.hpp"
#include "impl/device/device_manager.hpp"
#include "fabric_context.hpp"
#include "fabric_builder_context.hpp"
#include "fabric_host_utils.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include <bit>
#include <algorithm>
#include <utility>

namespace tt::tt_fabric {

namespace {
bool device_has_dispatch_tunnel(ChipId device_id) {
    auto mmio_device_id = tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    auto tunnels_from_mmio =
        tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id);
    // results are inclusive of the mmio_device_id so they will never be zero
    TT_FATAL(!tunnels_from_mmio.empty(), "must have at least one mmio device");
    return (tunnels_from_mmio.size() - 1) > 0;
}

}  // namespace

// FabricTensixDatamoverConfig implementation

void FabricTensixDatamoverConfig::find_min_max_eth_channels(const std::vector<tt_metal::IDevice*>& all_active_devices) {
    min_eth_channels_ = SIZE_MAX;
    max_eth_channels_ = 0;
    num_non_dispatch_routing_planes_ = 0;

    auto device_id = all_active_devices.front()->id();
    const auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    has_dispatch_tunnel_ = device_has_dispatch_tunnel(device_id);

    for (const auto& device : all_active_devices) {
        std::unordered_map<RoutingDirection, std::vector<chan_id_t>> active_fabric_eth_channels;
        std::unordered_map<RoutingDirection, FabricNodeId> chip_neighbors;

        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());

        for (const auto& direction : tt::tt_fabric::FabricContext::routing_directions) {
            auto active_eth_chans =
                control_plane.get_active_fabric_eth_routing_planes_in_direction(fabric_node_id, direction);
            if (active_eth_chans.empty()) {
                continue;
            }
            auto neighbors = control_plane.get_chip_neighbors(fabric_node_id, direction);

            // assume same neighbor per direction
            TT_FATAL(neighbors.size() == 1, "Multiple neighbor meshes per direction is unsupported");
            TT_FATAL(
                std::set<ChipId>(neighbors.begin()->second.begin(), neighbors.begin()->second.end()).size() == 1,
                "Multiple neighbors per direction is currently unsupported");

            FabricNodeId neighbor_fabric_node_id = FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
            chip_neighbors.emplace(direction, neighbor_fabric_node_id);

            active_fabric_eth_channels.insert({direction, active_eth_chans});
        }

        std::vector<chan_id_t> non_dispatch_active_channels;
        std::set<routing_plane_id_t> non_dispatch_routing_planes;
        for (const auto& [direction, remote_fabric_node_id] : chip_neighbors) {
            dispatch_link_idx_ =
                tt_metal::RelayMux::get_dispatch_link_index(fabric_node_id, remote_fabric_node_id, device);

            for (const auto& eth_chan : active_fabric_eth_channels[direction]) {
                auto link_idx = control_plane.get_routing_plane_id(fabric_node_id, eth_chan);

                if (!(has_dispatch_tunnel_ && link_idx == dispatch_link_idx_)) {
                    non_dispatch_active_channels.push_back(eth_chan);
                    non_dispatch_routing_planes.insert(link_idx);
                }
            }
        }

        // Update both min and max in a single pass
        size_t channel_count = non_dispatch_active_channels.size();
        max_eth_channels_ = std::max(max_eth_channels_, channel_count);
        min_eth_channels_ = std::min(min_eth_channels_, channel_count);

        // Track number of unique non-dispatch routing planes
        // Should be the same across all devices, so just take the max
        num_non_dispatch_routing_planes_ =
            std::max(num_non_dispatch_routing_planes_, non_dispatch_routing_planes.size());
    }

    // If no channels found, set min to 0
    if (min_eth_channels_ == SIZE_MAX) {
        min_eth_channels_ = 0;
    }
}

void FabricTensixDatamoverConfig::build_per_device_channel_mappings(
    const std::vector<tt_metal::IDevice*>& all_active_devices) {
    const auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_tensix_config = tt_metal::MetalContext::instance().get_fabric_tensix_config();

    // Create per-device channel mappings using real ethernet channel IDs
    for (const auto& device : all_active_devices) {
        auto dev_id = device->id();
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dev_id);
        // Get all active ethernet channels for this device
        auto active_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);

        // Initialize per-device mappings
        eth_chan_to_core_index_[dev_id] = std::unordered_map<size_t, size_t>();
        eth_chan_to_core_id_[dev_id] = std::unordered_map<size_t, FabricTensixCoreType>();

        // Create round-robin mapping using the actual ethernet channel IDs from active_channels
        // Skip dispatch link channels - they don't get tensix builders
        size_t channel_index = 0;
        for (auto [eth_chan_id, eth_chan_dir] : active_channels) {
            routing_plane_id_t routing_plane_id = control_plane.get_routing_plane_id(fabric_node_id, eth_chan_id);
            eth_chan_to_core_index_[dev_id][eth_chan_id] = channel_index;

            // Also populate direction_to_core_index_ for active directions
            direction_to_core_index_[dev_id][routing_plane_id][eth_chan_dir] = channel_index;

            // Determine core type: In both MUX and UDM modes, all worker channels go to MUX (core type 0)
            // The RELAY (core type 1) in UDM mode handles fabric-to-fabric routing, not worker channels
            FabricTensixCoreType core_id = FabricTensixCoreType::MUX;  // Always assign to MUX
            eth_chan_to_core_id_[dev_id][eth_chan_id] = core_id;

            if (!(has_dispatch_tunnel_ && routing_plane_id == dispatch_link_idx_)) {
                channel_index++;
            }
        }

        // In UDM mode, track missing directions for inter-mux communication
        if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
            track_missing_directions_for_udm(device, fabric_node_id, active_channels, channel_index);
        }
    }
}

void FabricTensixDatamoverConfig::build_fabric_tensix_noc_coords_map(
    const std::vector<tt_metal::IDevice*>& all_active_devices) {
    const auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();

    // Build the fabric_router_noc_coords_map_ to track which routers/tensix exist in each direction
    // for each fabric node and routing plane (link index)
    for (const auto& device : all_active_devices) {
        auto dev_id = device->id();
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dev_id);

        // Get all active ethernet channels for this device
        auto active_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);

        // For each active ethernet channel, record its direction info in the map
        // By looping through all channels, we naturally cover all directions that have active routers
        for (auto [eth_chan_id, eth_chan_dir] : active_channels) {
            routing_plane_id_t routing_plane_id = control_plane.get_routing_plane_id(fabric_node_id, eth_chan_id);

            // Get the tensix NOC coordinates for this channel
            auto [noc_x, noc_y] = get_noc_xy(device, eth_chan_id);

            // Record in both maps - fabric_tensix_noc_coords_map_ (all) and fabric_active_tensix_noc_coords_map_
            // (active only)
            fabric_tensix_noc_coords_map_[fabric_node_id][routing_plane_id][eth_chan_dir] = {noc_x, noc_y};
            fabric_active_tensix_noc_coords_map_[fabric_node_id][routing_plane_id][eth_chan_dir] = {noc_x, noc_y};
        }
    }
}

FabricTensixDatamoverConfig::FabricTensixDatamoverConfig() {
    // Initialize channel mappings and configurations, skipping the rest initilization if there are no ethernet found
    if (!initialize_channel_mappings()) {
        return;
    }
    calculate_buffer_allocations();
    create_configs();  // Mode-aware config creation
}

void FabricTensixDatamoverConfig::track_missing_directions_for_udm(
    tt_metal::IDevice* device,
    const FabricNodeId& fabric_node_id,
    const std::set<std::pair<chan_id_t, eth_chan_directions>>& active_channels,
    size_t& channel_index) {
    const auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    ChipId dev_id = device->id();

    // Collect all active (routing_plane_id, direction) pairs from active_channels
    // Skip dispatch routing planes (using pre-calculated dispatch link info)
    std::set<std::pair<routing_plane_id_t, eth_chan_directions>> active_plane_directions;
    std::set<routing_plane_id_t> active_routing_planes;
    for (const auto& [eth_chan_id, eth_chan_dir] : active_channels) {
        routing_plane_id_t routing_plane_id = control_plane.get_routing_plane_id(fabric_node_id, eth_chan_id);

        // Skip dispatch routing plane
        if (has_dispatch_tunnel_ && routing_plane_id == dispatch_link_idx_) {
            continue;
        }

        active_plane_directions.insert({routing_plane_id, eth_chan_dir});
        active_routing_planes.insert(routing_plane_id);
    }

    // For each active routing plane, check which of the 4 directions (E, W, N, S) are missing
    // Skip Z direction - it's for 3D routing and not relevant for missing directions on a chip
    std::set<std::pair<routing_plane_id_t, eth_chan_directions>> missing_plane_dirs;
    for (auto routing_plane_id : active_routing_planes) {
        for (uint8_t dir_idx = 0; dir_idx < eth_chan_directions::Z; dir_idx++) {
            auto dir = static_cast<eth_chan_directions>(dir_idx);
            if (!active_plane_directions.contains({routing_plane_id, dir})) {
                missing_plane_dirs.insert({routing_plane_id, dir});
            }
        }
    }

    if (!missing_plane_dirs.empty()) {
        // Calculate how many cores are remaining
        // channel_index represents the number of active directions already assigned
        size_t total_cores = logical_fabric_mux_cores_.size();
        size_t cores_used = channel_index;
        size_t cores_remaining = (cores_used < total_cores) ? (total_cores - cores_used) : 0;

        // Only add as many missing directions as we have remaining cores
        size_t missing_dirs_to_add = std::min(missing_plane_dirs.size(), cores_remaining);

        if (missing_dirs_to_add == 0) {
            log_warning(
                tt::LogMetal,
                "Device {}: No remaining cores for missing directions. "
                "Total cores: {}, cores used by active channels: {}, missing directions: {}",
                dev_id,
                total_cores,
                cores_used,
                missing_plane_dirs.size());
            return;
        }

        if (missing_dirs_to_add < missing_plane_dirs.size()) {
            log_warning(
                tt::LogMetal,
                "Device {}: Not enough cores for all missing directions. "
                "Adding {} out of {} missing directions. Total cores: {}, cores used: {}",
                dev_id,
                missing_dirs_to_add,
                missing_plane_dirs.size(),
                total_cores,
                cores_used);
        }

        // Only store the missing directions we're actually adding
        std::set<std::pair<routing_plane_id_t, eth_chan_directions>> added_missing_dirs;
        size_t added_count = 0;

        for (const auto& [routing_plane_id, missing_dir] : missing_plane_dirs) {
            if (added_count >= missing_dirs_to_add) {
                break;
            }

            size_t core_index = channel_index;
            direction_to_core_index_[dev_id][routing_plane_id][missing_dir] = core_index;

            // Also populate fabric_tensix_noc_coords_map_ for missing directions
            CoreCoord logical_core = logical_fabric_mux_cores_[core_index];
            CoreCoord translated_core = device->worker_core_from_logical_core(logical_core);
            fabric_tensix_noc_coords_map_[fabric_node_id][routing_plane_id][missing_dir] = {
                translated_core.x, translated_core.y};

            added_missing_dirs.insert({routing_plane_id, missing_dir});
            channel_index++;
            added_count++;
        }

        // Store only the missing directions that were actually added
        missing_directions_per_device_[dev_id] = added_missing_dirs;
    }
}

bool FabricTensixDatamoverConfig::initialize_channel_mappings() {
    // Get logical fabric mux cores from the first available device (same for all devices), except for TG
    const bool is_TG =
        (tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt_metal::ClusterType::TG);
    TT_FATAL(!is_TG, "Fabric with tensix extension is not supported for TG");

    const auto& all_active_devices = tt_metal::MetalContext::instance().device_manager()->get_all_active_devices();
    TT_FATAL(!all_active_devices.empty(), "No active devices found in DeviceManager");

    // Calculate and cache min/max ethernet channels once for later use
    find_min_max_eth_channels(all_active_devices);

    auto device_id = all_active_devices.front()->id();

    auto num_hw_cqs = tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
    tt_metal::DispatchCoreConfig dispatch_core_config =
        tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    logical_fabric_mux_cores_ = tt::get_logical_fabric_mux_cores(device_id, num_hw_cqs, dispatch_core_config);
    // TODO: once we merge the mux cores from dispatch to fabric, we can remove this.
    logical_dispatch_mux_cores_ = tt::get_logical_dispatch_cores(device_id, num_hw_cqs, dispatch_core_config);

    TT_FATAL(!logical_fabric_mux_cores_.empty(), "No logical fabric mux cores found for device {}", device_id);

    // Initialize translated mux cores (coordinates should be same across devices)
    auto* device = tt_metal::MetalContext::instance().device_manager()->get_active_device(device_id);
    TT_FATAL(device != nullptr, "Device {} not found in DeviceManager", device_id);
    for (const auto& logical_core : logical_fabric_mux_cores_) {
        CoreCoord translated_core = device->worker_core_from_logical_core(logical_core);
        translated_fabric_or_dispatch_mux_cores_.insert(translated_core);
        translated_fabric_mux_cores_.insert(translated_core);
    }
    for (const auto& logical_core : logical_dispatch_mux_cores_) {
        CoreCoord translated_core = device->worker_core_from_logical_core(logical_core);
        translated_fabric_or_dispatch_mux_cores_.insert(translated_core);
        translated_dispatch_mux_cores_.insert(translated_core);
    }

    // Check if we found any active ethernet channels
    if (max_eth_channels_ == 0) {
        log_warning(tt::LogMetal, "No active ethernet channels found in the system");
        return false;
    }

    TT_FATAL(!logical_fabric_mux_cores_.empty(), "logical_fabric_mux_cores_ is empty before division");

    // Calculate number of configs per core (should always be 1: one eth channel per core)
    num_configs_per_core_ =
        (max_eth_channels_ + logical_fabric_mux_cores_.size() - 1) / logical_fabric_mux_cores_.size();
    TT_FATAL(
        num_configs_per_core_ == 1,
        "Expected 1 config per core (one eth channel per core), but got {} configs per core",
        num_configs_per_core_);

    // Set num_used_riscs_per_tensix based on mode
    // This determines how many core types we use on each tensix core
    auto fabric_tensix_config = tt_metal::MetalContext::instance().get_fabric_tensix_config();
    switch (fabric_tensix_config) {
        case tt::tt_fabric::FabricTensixConfig::MUX:
            // MUX mode: only 1 core type (MUX on BRISC) is used per tensix core
            num_used_riscs_per_tensix_ = 1;
            break;
        case tt::tt_fabric::FabricTensixConfig::UDM:
            // UDM mode: 2 core types (MUX on BRISC + RELAY on NCRISC) per tensix core
            num_used_riscs_per_tensix_ = 2;
            break;
        case tt::tt_fabric::FabricTensixConfig::DISABLED: num_used_riscs_per_tensix_ = 0; break;
        default: TT_THROW("Unsupported FabricTensixConfig mode: {}", static_cast<int>(fabric_tensix_config));
    }

    // Second pass: create per-device channel mappings
    build_per_device_channel_mappings(all_active_devices);

    // Third pass: Build the fabric_router_noc_coords_map_
    build_fabric_tensix_noc_coords_map(all_active_devices);

    return true;
}

// UDM mode helper: builds list of workers sorted by column (x first, then y within each column)
std::vector<CoreCoord> FabricTensixDatamoverConfig::build_workers_by_column(tt_metal::IDevice* device) const {
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t total_workers = compute_grid.x * compute_grid.y;

    std::vector<CoreCoord> workers_by_column(total_workers);

    // Sort by column: x first (outer loop), then y within each column (inner loop)
    // This ensures workers in the same column are contiguous and get assigned to the same tensix
    size_t idx = 0;
    for (uint32_t x = 0; x < compute_grid.x; x++) {
        for (uint32_t y = 0; y < compute_grid.y; y++) {
            CoreCoord logical_worker(x, y);
            CoreCoord translated_worker = device->worker_core_from_logical_core(logical_worker);
            workers_by_column[idx++] = translated_worker;
        }
    }

    return workers_by_column;
}

// UDM mode helper: gets unique tensix cores for worker assignment
std::vector<CoreCoord> FabricTensixDatamoverConfig::get_tensix_cores_for_workers(tt_metal::IDevice* device) const {
    const auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());

    const auto& node_map = fabric_tensix_noc_coords_map_.at(fabric_node_id);

    // Collect unique tensix cores across all routing planes and directions
    std::set<CoreCoord> unique_tensix_cores;
    for (const auto& [routing_plane_id, direction_map] : node_map) {
        for (const auto& [direction, noc_coords] : direction_map) {
            const auto& [noc_x, noc_y] = noc_coords;
            unique_tensix_cores.insert(CoreCoord(noc_x, noc_y));
        }
    }

    // Convert set to vector (maintains sorted order)
    return std::vector<CoreCoord>(unique_tensix_cores.begin(), unique_tensix_cores.end());
}

// UDM mode helper: assigns workers to tensix cores in contiguous chunks
void FabricTensixDatamoverConfig::assign_workers_to_tensix_cores(
    ChipId device_id,
    const std::vector<CoreCoord>& workers_by_column,
    const std::vector<CoreCoord>& tensix_cores_for_workers,
    uint32_t num_worker_channels) {
    auto& info_map = worker_to_tensix_info_map_[device_id];
    info_map.clear();

    for (size_t i = 0; i < workers_by_column.size(); i++) {
        size_t tensix_idx = i / num_worker_channels;
        TT_FATAL(
            tensix_idx < tensix_cores_for_workers.size(),
            "tensix_idx {} exceeds tensix_cores_for_workers size {}",
            tensix_idx,
            tensix_cores_for_workers.size());
        CoreCoord tensix_core = tensix_cores_for_workers[tensix_idx];
        uint32_t channel_index = i % num_worker_channels;
        // Store tensix core and channel index together
        info_map[workers_by_column[i]] = WorkerTensixInfo{tensix_core, channel_index};
    }
}

// Helper to calculate number of channels for mux (handles both UDM and Legacy modes)
// Also builds worker_to_tensix_info_map_ in UDM mode
std::map<ChannelTypes, uint32_t> FabricTensixDatamoverConfig::calculate_mux_channel_counts(
    const std::vector<tt_metal::IDevice*>& all_active_devices) {
    std::map<ChannelTypes, uint32_t> channel_counts;

    auto fabric_tensix_config = tt_metal::MetalContext::instance().get_fabric_tensix_config();

    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        // UDM mode: calculate channels based on compute grid
        // UDM has WORKER_CHANNEL, RELAY_TO_MUX_CHANNEL, MUX_TO_MUX_CHANNEL (NO ROUTER_CHANNEL)

        // Calculate num_worker_channels using first device (all devices have same configuration)
        auto* first_device = all_active_devices.front();
        auto workers_by_column = build_workers_by_column(first_device);
        auto tensix_cores_for_workers = get_tensix_cores_for_workers(first_device);

        uint32_t total_workers = workers_by_column.size();
        uint32_t num_worker_channels = static_cast<uint32_t>(
            (total_workers + tensix_cores_for_workers.size() - 1) / tensix_cores_for_workers.size());

        log_debug(
            tt::LogMetal,
            "UDM mode: total_workers={}, tensix_cores={}, num_worker_channels={}",
            total_workers,
            tensix_cores_for_workers.size(),
            num_worker_channels);

        // Build per-device worker-to-tensix maps
        // fabric_tensix_noc_coords_map_ is indexed by fabric_node_id, so process each device
        for (auto* device : all_active_devices) {
            auto device_workers = build_workers_by_column(device);
            auto device_tensix_cores = get_tensix_cores_for_workers(device);

            // Assign workers to tensix cores in contiguous chunks for this device
            assign_workers_to_tensix_cores(device->id(), device_workers, device_tensix_cores, num_worker_channels);
        }

        channel_counts[ChannelTypes::WORKER_CHANNEL] = num_worker_channels;

        // Relay channels: 3 channels (LOCAL_RELAY, EAST_OR_NORTH_RELAY, WEST_OR_SOUTH_RELAY)
        channel_counts[ChannelTypes::RELAY_TO_MUX_CHANNEL] =
            static_cast<uint32_t>(UdmMuxRelayToMuxChannelId::NUM_CHANNELS);

        // Inter-mux forwarding channels: 3 channels (EAST_OR_WEST, WEST_OR_NORTH, NORTH_OR_SOUTH)
        channel_counts[ChannelTypes::MUX_TO_MUX_CHANNEL] = static_cast<uint32_t>(UdmMuxInterMuxChannelId::NUM_CHANNELS);

        // UDM mode does NOT have ROUTER_CHANNEL
    } else {
        // Legacy MUX mode: use topology-based channel count
        // Legacy has ROUTER_CHANNEL and WORKER_CHANNEL
        const auto& fabric_context = tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
        bool is_2D_routing = fabric_context.is_2D_routing_enabled();

        // Router channel count: 1 for 1D topologies, 3 for 2D topologies
        channel_counts[ChannelTypes::ROUTER_CHANNEL] = builder_config::get_vc0_downstream_edm_count(is_2D_routing);
        channel_counts[ChannelTypes::WORKER_CHANNEL] = 1;  // Always 1 worker channel in legacy mode
    }

    return channel_counts;
}

void FabricTensixDatamoverConfig::calculate_buffer_allocations() {
    const auto& hal = tt_metal::MetalContext::instance().hal();
    const auto& fabric_context = tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& all_active_devices = tt_metal::MetalContext::instance().device_manager()->get_all_active_devices();

    // Get buffer size from fabric context
    buffer_size_bytes_full_size_channel_ =
        fabric_context.get_fabric_packet_header_size_bytes() + tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    // Calculate available L1 space for tensix cores
    uint32_t l1_base =
        hal.get_dev_addr(tt_metal::HalProgrammableCoreType::TENSIX, tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint32_t l1_size =
        hal.get_dev_size(tt_metal::HalProgrammableCoreType::TENSIX, tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);

    // Get L1 alignment requirement
    size_t l1_alignment = hal.get_alignment(tt_metal::HalMemType::L1);

    // Reserve space for both core types with proper alignment
    space_per_risc_ = l1_size / num_used_riscs_per_tensix_;             // Split between MUX and RELAY
    space_per_risc_ = (space_per_risc_ / l1_alignment) * l1_alignment;  // Align down to L1 alignment

    // Determine num_channels_for_mux and build channel type maps based on mode
    // The helper function handles both UDM and Legacy modes internally
    // Also builds worker_to_tensix_core_map_ in UDM mode
    mux_channel_counts_ = calculate_mux_channel_counts(all_active_devices);

    // Calculate total number of mux channels
    num_channels_for_mux_ = 0;
    for (const auto& [type, count] : mux_channel_counts_) {
        num_channels_for_mux_ += count;
    }

    // Calculate buffers per channel based on available space and max channels
    size_t space_needed_for_max_channels = num_channels_for_mux_ * buffer_size_bytes_full_size_channel_;

    size_t number_of_buffers_per_channel = std::bit_floor(space_per_risc_ / space_needed_for_max_channels);
    TT_FATAL(number_of_buffers_per_channel > 0, "number of buffers per channel must be non-zero");

    // To prevent overflow of num_buffers_per_channel_ (which is uint8_t), we max number of buffers per channel to 128
    if (number_of_buffers_per_channel >= std::numeric_limits<uint8_t>::max()) {
        log_warning(
            tt::LogMetal, "Number of buffers per channel overflows uint8_t, setting to 128 to prevent byte overflow");
        num_buffers_per_channel_ = 128;
    } else {
        num_buffers_per_channel_ = static_cast<uint8_t>(number_of_buffers_per_channel);
    }

    // Build buffer counts map for each mux channel type (all use same buffer count for now)
    for (const auto& [type, channel_count] : mux_channel_counts_) {
        mux_channel_buffer_counts_[type] = num_buffers_per_channel_;
    }

    // Set base addresses for each core type with proper L1 alignment
    for (size_t i = 0; i < num_used_riscs_per_tensix_; ++i) {
        FabricTensixCoreType core_id = static_cast<FabricTensixCoreType>(i);
        base_l1_addresses_[core_id] = l1_base + (i * space_per_risc_);
    }
}

std::shared_ptr<FabricTensixDatamoverMuxConfig> FabricTensixDatamoverConfig::create_mux_config(
    FabricTensixCoreType core_id) {
    // Calculate the end address for this core's allocated L1 space
    size_t l1_end_address = base_l1_addresses_[core_id] + space_per_risc_;

    // Both UDM and Legacy MUX modes use channel type maps now
    std::map<ChannelTypes, ChannelTypeConfig> channel_type_configs;
    for (const auto& [type, count] : mux_channel_counts_) {
        uint32_t num_buffers = mux_channel_buffer_counts_[type];
        channel_type_configs[type] = ChannelTypeConfig(count, num_buffers, buffer_size_bytes_full_size_channel_);
    }

    return std::make_shared<FabricTensixDatamoverMuxConfig>(
        channel_type_configs,         // channel_type_configs map (already sorted)
        base_l1_addresses_[core_id],  // base_l1_address
        l1_end_address                // l1_end_address
    );
}

std::shared_ptr<FabricTensixDatamoverRelayConfig> FabricTensixDatamoverConfig::create_relay_config(
    FabricTensixCoreType core_id) {
    // Calculate the end address for this core's allocated L1 space
    size_t l1_end_address = base_l1_addresses_[core_id] + space_per_risc_;

    // Relay has only one channel type: ROUTER_CHANNEL
    std::map<ChannelTypes, ChannelTypeConfig> channel_type_configs;
    channel_type_configs[ChannelTypes::ROUTER_CHANNEL] = ChannelTypeConfig(
        static_cast<size_t>(UdmRelayChannelId::NUM_CHANNELS),  // num_channels (relay has 1 channel)
        num_buffers_per_channel_,                              // num_buffers_per_channel
        buffer_size_bytes_full_size_channel_                   // buffer_size_bytes
    );

    return std::make_shared<FabricTensixDatamoverRelayConfig>(
        channel_type_configs,         // channel_type_configs map (already sorted)
        base_l1_addresses_[core_id],  // base_l1_address
        l1_end_address                // l1_end_address
    );
}

void FabricTensixDatamoverConfig::create_configs() {
    // Get the fabric tensix config mode
    auto fabric_tensix_config = tt_metal::MetalContext::instance().get_fabric_tensix_config();

    switch (fabric_tensix_config) {
        case tt::tt_fabric::FabricTensixConfig::MUX:
            // MUX mode: only create mux config for MUX core type
            configs_[FabricTensixCoreType::MUX] = create_mux_config(FabricTensixCoreType::MUX);
            break;
        case tt::tt_fabric::FabricTensixConfig::UDM:
            // UDM mode: create mux config for MUX core type and relay config for RELAY core type
            configs_[FabricTensixCoreType::MUX] = create_mux_config(FabricTensixCoreType::MUX);
            configs_[FabricTensixCoreType::RELAY] = create_relay_config(FabricTensixCoreType::RELAY);
            break;
        case tt::tt_fabric::FabricTensixConfig::DISABLED:
        default: break;
    }
}

// Getter implementations
size_t FabricTensixDatamoverConfig::get_base_l1_address(FabricTensixCoreType core_id) const {
    auto it = base_l1_addresses_.find(core_id);
    TT_FATAL(it != base_l1_addresses_.end(), "Base L1 address not found for core type {}", static_cast<int>(core_id));
    return it->second;
}

std::pair<uint32_t, uint32_t> FabricTensixDatamoverConfig::get_noc_xy(
    tt_metal::IDevice* device, uint32_t eth_chan_id) const {
    CoreCoord core = get_core_for_channel(device->id(), eth_chan_id);
    CoreCoord physical_core = device->worker_core_from_logical_core(core);
    return {physical_core.x, physical_core.y};
}

size_t FabricTensixDatamoverConfig::get_channels_base_address(
    FabricTensixCoreType core_id, size_t tensix_channel_id) const {
    TT_FATAL(
        core_id == FabricTensixCoreType::MUX,
        "caller must be for accessing mux config, but was accessing: {} config",
        core_id);
    auto config = get_config(core_id);
    return config->get_channel_base_address(tt::tt_fabric::ChannelTypes::WORKER_CHANNEL, tensix_channel_id);
}

FabricTensixCoreType FabricTensixDatamoverConfig::get_core_id_for_channel(
    ChipId device_id, uint32_t eth_chan_id) const {
    auto device_it = eth_chan_to_core_id_.find(device_id);
    TT_FATAL(device_it != eth_chan_to_core_id_.end(), "Device {} not found in core type mapping", device_id);

    auto it = device_it->second.find(eth_chan_id);
    TT_FATAL(
        it != device_it->second.end(),
        "Core type not found for ethernet channel {} on device {}",
        eth_chan_id,
        device_id);
    return it->second;
}

CoreCoord FabricTensixDatamoverConfig::get_core_for_channel(ChipId device_id, uint32_t eth_chan_id) const {
    auto device_it = eth_chan_to_core_index_.find(device_id);
    TT_FATAL(device_it != eth_chan_to_core_index_.end(), "Device {} not found in core mapping", device_id);

    auto it = device_it->second.find(eth_chan_id);
    TT_FATAL(
        it != device_it->second.end(),
        "Core index not found for ethernet channel {} on device {}",
        eth_chan_id,
        device_id);

    size_t core_index = it->second;
    TT_FATAL(core_index < logical_fabric_mux_cores_.size(), "Invalid core index {}", core_index);
    return logical_fabric_mux_cores_[core_index];
}

std::shared_ptr<FabricTensixDatamoverBaseConfig> FabricTensixDatamoverConfig::get_config(
    FabricTensixCoreType core_id) const {
    auto it = configs_.find(core_id);
    TT_FATAL(it != configs_.end(), "Config not found for core type {}", static_cast<int>(core_id));
    return it->second;
}

bool FabricTensixDatamoverConfig::is_core_id_active(FabricTensixCoreType core_id) const {
    return configs_.contains(core_id);
}

size_t FabricTensixDatamoverConfig::get_local_flow_control_semaphore_address(
    uint32_t channel_id, FabricTensixCoreType core_id) const {
    TT_FATAL(
        core_id == FabricTensixCoreType::MUX,
        "caller must be for accessing mux config, but was accessing: {} config",
        core_id);
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::ChannelTypes::WORKER_CHANNEL;
    return config->get_flow_control_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_connection_semaphore_address(
    uint32_t channel_id, FabricTensixCoreType core_id) const {
    TT_FATAL(
        core_id == FabricTensixCoreType::MUX,
        "caller must be for accessing mux config, but was accessing: {} config",
        core_id);
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::ChannelTypes::WORKER_CHANNEL;
    return config->get_connection_handshake_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_worker_conn_info_base_address(
    uint32_t channel_id, FabricTensixCoreType core_id) const {
    TT_FATAL(
        core_id == FabricTensixCoreType::MUX,
        "caller must be for accessing mux config, but was accessing: {} config",
        core_id);
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::ChannelTypes::WORKER_CHANNEL;
    return config->get_connection_info_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_buffer_index_semaphore_address(
    uint32_t channel_id, FabricTensixCoreType core_id) const {
    TT_FATAL(
        core_id == FabricTensixCoreType::MUX,
        "caller must be for accessing mux config, but was accessing: {} config",
        core_id);
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::ChannelTypes::WORKER_CHANNEL;
    return config->get_buffer_index_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_channel_credits_stream_id(
    uint32_t channel_id, FabricTensixCoreType core_id) const {
    auto config = get_config(core_id);
    tt::tt_fabric::ChannelTypes channel_type;
    if (core_id == FabricTensixCoreType::MUX) {
        channel_type = tt::tt_fabric::ChannelTypes::WORKER_CHANNEL;
    } else {
        channel_type = tt::tt_fabric::ChannelTypes::ROUTER_CHANNEL;
    }
    size_t stream_id = config->get_channel_credits_stream_id(channel_type, channel_id);
    return stream_id;
}

std::pair<uint32_t, uint32_t> FabricTensixDatamoverConfig::get_termination_address_and_signal(
    FabricTensixCoreType core_id) const {
    TT_FATAL(
        is_core_id_active(core_id), "Core type {} is not active in fabric tensix config", static_cast<int>(core_id));

    auto config = get_config(core_id);
    return std::make_pair(
        config->get_termination_signal_address(), tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
}

const std::pair<uint32_t, uint32_t>* FabricTensixDatamoverConfig::get_tensix_noc_coords(
    const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    auto node_it = fabric_tensix_noc_coords_map_.find(fabric_node_id);
    if (node_it == fabric_tensix_noc_coords_map_.end()) {
        return nullptr;
    }

    auto plane_it = node_it->second.find(routing_plane_id);
    if (plane_it == node_it->second.end()) {
        return nullptr;
    }

    auto dir_it = plane_it->second.find(direction);
    if (dir_it == plane_it->second.end()) {
        return nullptr;
    }

    return &(dir_it->second);
}

const std::pair<uint32_t, uint32_t>* FabricTensixDatamoverConfig::get_active_tensix_noc_coords(
    const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    auto node_it = fabric_active_tensix_noc_coords_map_.find(fabric_node_id);
    if (node_it == fabric_active_tensix_noc_coords_map_.end()) {
        return nullptr;
    }

    auto plane_it = node_it->second.find(routing_plane_id);
    if (plane_it == node_it->second.end()) {
        return nullptr;
    }

    auto dir_it = plane_it->second.find(direction);
    if (dir_it == plane_it->second.end()) {
        return nullptr;
    }

    return &(dir_it->second);
}

std::set<std::pair<routing_plane_id_t, eth_chan_directions>> FabricTensixDatamoverConfig::get_missing_directions(
    ChipId device_id) const {
    auto it = missing_directions_per_device_.find(device_id);
    if (it == missing_directions_per_device_.end()) {
        return {};
    }
    return it->second;
}

bool FabricTensixDatamoverConfig::has_missing_directions(ChipId device_id) const {
    auto it = missing_directions_per_device_.find(device_id);
    return it != missing_directions_per_device_.end() && !it->second.empty();
}

size_t FabricTensixDatamoverConfig::get_core_index_for_direction(
    ChipId device_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    auto dev_it = direction_to_core_index_.find(device_id);
    TT_FATAL(dev_it != direction_to_core_index_.end(), "Device {} not found in direction_to_core_index_", device_id);

    auto plane_it = dev_it->second.find(routing_plane_id);
    TT_FATAL(
        plane_it != dev_it->second.end(),
        "Routing plane {} not found for device {} in direction_to_core_index_",
        routing_plane_id,
        device_id);

    auto dir_it = plane_it->second.find(direction);
    TT_FATAL(
        dir_it != plane_it->second.end(),
        "Direction {} not found for device {}, routing plane {} in direction_to_core_index_",
        static_cast<uint32_t>(direction),
        device_id,
        routing_plane_id);

    return dir_it->second;
}

CoreCoord FabricTensixDatamoverConfig::get_core_for_direction(
    ChipId device_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    size_t core_index = get_core_index_for_direction(device_id, routing_plane_id, direction);
    TT_FATAL(
        core_index < logical_fabric_mux_cores_.size(),
        "Core index {} out of bounds for logical_fabric_mux_cores_ (size {})",
        core_index,
        logical_fabric_mux_cores_.size());
    return logical_fabric_mux_cores_[core_index];
}

std::pair<uint32_t, uint32_t> FabricTensixDatamoverConfig::get_noc_xy_for_direction(
    tt_metal::IDevice* device, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    CoreCoord logical_core = get_core_for_direction(device->id(), routing_plane_id, direction);
    CoreCoord translated_core = device->worker_core_from_logical_core(logical_core);
    return {translated_core.x, translated_core.y};
}

FabricTensixDatamoverConfig::WorkerTensixInfo FabricTensixDatamoverConfig::get_worker_tensix_info(
    ChipId device_id, const CoreCoord& worker_coord) const {
    auto device_it = worker_to_tensix_info_map_.find(device_id);
    TT_FATAL(device_it != worker_to_tensix_info_map_.end(), "Device {} not found in worker tensix info map", device_id);
    auto worker_it = device_it->second.find(worker_coord);
    TT_FATAL(
        worker_it != device_it->second.end(),
        "Worker {} not found in tensix info map for device {}",
        worker_coord,
        device_id);
    return worker_it->second;
}

// FabricTensixDatamoverBuilder implementation

template <typename BuilderType, typename ConfigType>
std::unique_ptr<BuilderType> FabricTensixDatamoverBuilder::create_builder(
    FabricTensixCoreType core_type,
    const FabricTensixDatamoverConfig& tensix_config,
    const CoreCoord& my_core_logical,
    tt::tt_fabric::FabricNodeId local_fabric_node_id,
    tt::tt_fabric::FabricNodeId remote_fabric_node_id,
    uint32_t ethernet_channel_id,
    uint32_t link_idx,
    uint32_t noc_x,
    uint32_t noc_y,
    eth_chan_directions direction,
    bool has_fabric_router) {
    // Check if the core type is active (for relay, may return nullptr)
    if (!tensix_config.is_core_id_active(core_type)) {
        return nullptr;
    }

    // Get and cast the config to the appropriate type
    auto config_base = tensix_config.get_config(core_type);
    auto config = std::dynamic_pointer_cast<ConfigType>(config_base);
    TT_FATAL(config != nullptr, "Expected config for core type {}", static_cast<int>(core_type));

    // Create and return the builder
    return std::make_unique<BuilderType>(
        my_core_logical,
        local_fabric_node_id,
        remote_fabric_node_id,
        ethernet_channel_id,
        link_idx,
        core_type,
        noc_x,
        noc_y,
        config,
        direction,
        has_fabric_router);
}

FabricTensixDatamoverBuilder::FabricTensixDatamoverBuilder(
    std::unique_ptr<FabricTensixDatamoverMuxBuilder> mux_builder,
    std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder,
    const CoreCoord& logical_core,
    tt::tt_fabric::FabricNodeId local_fabric_node_id,
    tt::tt_fabric::FabricNodeId remote_fabric_node_id,
    uint32_t noc_x,
    uint32_t noc_y,
    eth_chan_directions direction) :
    FabricDatamoverBuilderBase(noc_x, noc_y, direction),
    mux_builder_(std::move(mux_builder)),
    relay_builder_(std::move(relay_builder)),
    logical_core_(logical_core),
    local_fabric_node_id_(local_fabric_node_id),
    remote_fabric_node_id_(remote_fabric_node_id) {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder cannot be null");
}

FabricTensixDatamoverBuilder FabricTensixDatamoverBuilder::build(
    tt_metal::IDevice* device,
    tt_metal::Program& /*program*/,
    tt::tt_fabric::FabricNodeId local_fabric_node_id,
    tt::tt_fabric::FabricNodeId remote_fabric_node_id,
    uint32_t ethernet_channel_id,
    eth_chan_directions direction,
    std::vector<bool>&& sender_channel_injection_flags) {
    const auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();
    auto fabric_tensix_config = tt_metal::MetalContext::instance().get_fabric_tensix_config();

    // Get core for this ethernet channel
    CoreCoord my_core_logical = tensix_config.get_core_for_channel(device->id(), ethernet_channel_id);

    // Get NOC coordinates
    auto [noc_x, noc_y] = tensix_config.get_noc_xy(device, ethernet_channel_id);

    // Get link index (routing plane ID) from control plane using channel ID
    uint32_t link_idx = control_plane.get_routing_plane_id(local_fabric_node_id, ethernet_channel_id);

    // Create mux builder (always needed in both MUX and UDM modes)
    // has_fabric_router = true because this is a normal build with an actual eth channel/router
    auto mux_builder = create_builder<FabricTensixDatamoverMuxBuilder, FabricTensixDatamoverMuxConfig>(
        FabricTensixCoreType::MUX,
        tensix_config,
        my_core_logical,
        local_fabric_node_id,
        remote_fabric_node_id,
        ethernet_channel_id,
        link_idx,
        noc_x,
        noc_y,
        direction,
        true /* has_fabric_router */);

    // Create relay builder (only in UDM mode if relay core is active)
    std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder = nullptr;
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        relay_builder = create_builder<FabricTensixDatamoverRelayBuilder, FabricTensixDatamoverRelayConfig>(
            FabricTensixCoreType::RELAY,
            tensix_config,
            my_core_logical,
            local_fabric_node_id,
            remote_fabric_node_id,
            ethernet_channel_id,
            link_idx,
            noc_x,
            noc_y,
            direction,
            true /* has_fabric_router - not used for relay*/);
    }

    auto builder = FabricTensixDatamoverBuilder(
        std::move(mux_builder),
        std::move(relay_builder),
        my_core_logical,
        local_fabric_node_id,
        remote_fabric_node_id,
        noc_x,
        noc_y,
        direction);

    // Set injection flags on the builder's configs
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::MUX) {
        builder.set_sender_channel_injection_flags_from_vector(std::move(sender_channel_injection_flags));
    }

    return builder;
}

FabricTensixDatamoverBuilder FabricTensixDatamoverBuilder::build_for_missing_direction(
    tt_metal::IDevice* device,
    tt_metal::Program& /*program*/,
    tt::tt_fabric::FabricNodeId local_fabric_node_id,
    routing_plane_id_t routing_plane_id,
    eth_chan_directions direction) {
    const auto& control_plane = tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();
    auto fabric_tensix_config = tt_metal::MetalContext::instance().get_fabric_tensix_config();

    // This method is only valid for UDM mode
    TT_FATAL(
        fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM,
        "build_for_missing_direction is only valid in UDM mode");

    // Get core for this (routing_plane_id, direction) pair (assigned in initialize_channel_mappings)
    CoreCoord my_core_logical = tensix_config.get_core_for_direction(device->id(), routing_plane_id, direction);

    // Get NOC coordinates for this (routing_plane_id, direction) pair
    auto [noc_x, noc_y] = tensix_config.get_noc_xy_for_direction(device, routing_plane_id, direction);

    // Use the routing_plane_id as the link_idx
    uint32_t link_idx = routing_plane_id;

    // For missing directions, get remote_fabric_node_id from any valid neighbor direction
    // We need a valid remote_fabric_node_id even for missing directions
    FabricNodeId remote_fabric_node_id = local_fabric_node_id;
    for (const auto& routing_dir : tt::tt_fabric::FabricContext::routing_directions) {
        auto neighbors = control_plane.get_chip_neighbors(local_fabric_node_id, routing_dir);
        if (!neighbors.empty()) {
            remote_fabric_node_id = FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
            break;
        }
    }

    // For missing directions, we don't have a real eth channel
    // Use 0 as a placeholder - it's not used for anything meaningful in this case
    uint32_t ethernet_channel_id = 0;

    // Create mux builder only (no relay for missing directions since there's no router)
    // has_fabric_router = false because this is a missing direction with no actual router
    auto mux_builder = create_builder<FabricTensixDatamoverMuxBuilder, FabricTensixDatamoverMuxConfig>(
        FabricTensixCoreType::MUX,
        tensix_config,
        my_core_logical,
        local_fabric_node_id,
        remote_fabric_node_id,
        ethernet_channel_id,
        link_idx,
        noc_x,
        noc_y,
        direction,
        false /* has_fabric_router */);

    // No relay builder for missing directions - these are purely for inter-mux forwarding
    std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder = nullptr;

    return FabricTensixDatamoverBuilder(
        std::move(mux_builder),
        std::move(relay_builder),
        my_core_logical,
        local_fabric_node_id,
        remote_fabric_node_id,
        noc_x,
        noc_y,
        direction);
}

void FabricTensixDatamoverBuilder::create_and_compile(tt_metal::Program& program) {
    // Always create and compile mux builder
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    mux_builder_->create_and_compile(program);

    // In UDM mode, also create and compile relay builder
    if (relay_builder_ != nullptr) {
        relay_builder_->create_and_compile(program);
    }
}

tt::tt_fabric::SenderWorkerAdapterSpec FabricTensixDatamoverBuilder::build_connection_to_fabric_channel(
    uint32_t channel_id) const {
    // This method connects to mux channels (for worker traffic, inter-mux forwarding, etc.)
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->build_connection_to_fabric_channel(channel_id);
}

tt::tt_fabric::SenderWorkerAdapterSpec FabricTensixDatamoverBuilder::build_connection_to_relay_channel() const {
    // This method connects to relay's channel (for router-to-relay traffic in UDM mode)
    TT_FATAL(relay_builder_ != nullptr, "Relay builder must not be null in UDM mode");
    constexpr uint32_t relay_channel_id = static_cast<uint32_t>(UdmRelayChannelId::ROUTER_CHANNEL);
    return relay_builder_->build_connection_to_fabric_channel(relay_channel_id);
}

uint32_t FabricTensixDatamoverBuilder::get_noc_x() const { return FabricDatamoverBuilderBase::get_noc_x(); }

uint32_t FabricTensixDatamoverBuilder::get_noc_y() const { return FabricDatamoverBuilderBase::get_noc_y(); }

eth_chan_directions FabricTensixDatamoverBuilder::get_direction() const {
    return FabricDatamoverBuilderBase::get_direction();
}

void FabricTensixDatamoverBuilder::append_upstream_routers_noc_xy(uint32_t noc_x, uint32_t noc_y) {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    mux_builder_->append_upstream_routers_noc_xy(noc_x, noc_y);
}

void FabricTensixDatamoverBuilder::append_relay_router_noc_xy(uint32_t noc_x, uint32_t noc_y) {
    TT_FATAL(relay_builder_ != nullptr, "Relay builder must not be null in UDM mode");
    relay_builder_->append_router_noc_xy(noc_x, noc_y);
}

void FabricTensixDatamoverBuilder::set_sender_channel_injection_flags_from_vector(std::vector<bool>&& flags) {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    // Validate that input vector size matches the number of channels
    uint8_t total_num_channels = mux_builder_->config_->get_total_num_channels();

    TT_FATAL(
        flags.size() == total_num_channels,
        "Internal error: injection flags vector size {} does not match total number of mux channels {}",
        flags.size(),
        total_num_channels);

    // Move flags to mux config (transfers ownership via setter)
    mux_builder_->set_sender_channel_injection_flags(std::move(flags));
}

}  // namespace tt::tt_fabric
