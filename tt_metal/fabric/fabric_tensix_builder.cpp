// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_tensix_builder.hpp"
#include "fabric_tensix_builder_impl.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/core_descriptor.hpp"
#include <tt-metalium/device_pool.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_align.hpp"
#include <bit>
#include <algorithm>
#include <utility>

namespace tt::tt_fabric {

namespace {
bool device_has_dispatch_tunnel(ChipId device_id) {
    auto mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    auto tunnels_from_mmio =
        tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id);
    // results are inclusive of the mmio_device_id so they will never be zero
    TT_FATAL(!tunnels_from_mmio.empty(), "must have at least one mmio device");
    return (tunnels_from_mmio.size() - 1) > 0;
}

// Helper function to find the maximum number of ethernet channels across all devices
size_t find_max_eth_channels(const std::vector<tt_metal::IDevice*>& all_active_devices) {
    size_t max_eth_channels = 0;
    auto device_id = all_active_devices.front()->id();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    const bool has_dispatch_tunnel = device_has_dispatch_tunnel(device_id);

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
        for (const auto& [direction, remote_fabric_node_id] : chip_neighbors) {
            uint32_t dispatch_link_idx =
                tt_metal::RelayMux::get_dispatch_link_index(fabric_node_id, remote_fabric_node_id, device);

            for (const auto& eth_chan : active_fabric_eth_channels[direction]) {
                auto link_idx = control_plane.get_routing_plane_id(fabric_node_id, eth_chan);

                if (!(has_dispatch_tunnel && link_idx == dispatch_link_idx)) {
                    non_dispatch_active_channels.push_back(eth_chan);
                }
            }
        }

        max_eth_channels = std::max(max_eth_channels, non_dispatch_active_channels.size());
    }

    return max_eth_channels;
}

}  // namespace

// FabricTensixDatamoverConfig implementation

FabricTensixDatamoverConfig::FabricTensixDatamoverConfig() {
    // Initialize channel mappings and configurations, skipping the rest initilization if there are no ethernet found
    if (!initialize_channel_mappings()) {
        return;
    }
    calculate_buffer_allocations();
    create_configs();  // Mode-aware config creation
}

bool FabricTensixDatamoverConfig::initialize_channel_mappings() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Get logical fabric mux cores from the first available device (same for all devices), except for TG
    const bool is_TG =
        (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::TG);
    TT_FATAL(!is_TG, "Fabric with tensix extension is not supported for TG");

    const auto& all_active_devices = tt::DevicePool::instance().get_all_active_devices();
    TT_FATAL(!all_active_devices.empty(), "No active devices found in DevicePool");

    auto device_id = all_active_devices.front()->id();

    uint8_t num_hw_cqs = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
    tt::tt_metal::DispatchCoreConfig dispatch_core_config =
        tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    logical_fabric_mux_cores_ = tt::get_logical_fabric_mux_cores(device_id, num_hw_cqs, dispatch_core_config);
    // TODO: once we merge the mux cores from dispatch to fabric, we can remove this.
    logical_dispatch_mux_cores_ = tt::get_logical_dispatch_cores(device_id, num_hw_cqs, dispatch_core_config);

    TT_FATAL(!logical_fabric_mux_cores_.empty(), "No logical fabric mux cores found for device {}", device_id);

    // Initialize translated mux cores (coordinates should be same across devices)
    auto device = tt::DevicePool::instance().get_active_device(device_id);
    TT_FATAL(device != nullptr, "Device {} not found in DevicePool", device_id);
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

    // Get maximum number of active ethernet channels from control plane across all devices
    size_t max_eth_channels = find_max_eth_channels(all_active_devices);
    if (max_eth_channels == 0) {
        log_warning(tt::LogMetal, "No active ethernet channels found in the system");
        return false;
    }

    TT_FATAL(!logical_fabric_mux_cores_.empty(), "logical_fabric_mux_cores_ is empty before division");

    // Calculate number of configs per core (should always be 1: one eth channel per core)
    num_configs_per_core_ =
        (max_eth_channels + logical_fabric_mux_cores_.size() - 1) / logical_fabric_mux_cores_.size();
    TT_FATAL(
        num_configs_per_core_ == 1,
        "Expected 1 config per core (one eth channel per core), but got {} configs per core",
        num_configs_per_core_);

    // Set num_used_riscs_per_tensix based on mode
    // This determines how many core types we use on each tensix core
    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
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

    // Second pass: create per-device channel mappings using real ethernet channel IDs
    for (const auto& device : all_active_devices) {
        auto dev_id = device->id();
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dev_id);

        // Get all active ethernet channels for this device
        auto active_channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);

        // Initialize per-device mappings
        eth_chan_to_core_index_[dev_id] = std::unordered_map<size_t, size_t>();
        eth_chan_to_core_id_[dev_id] = std::unordered_map<size_t, FabricTensixCoreType>();

        // Create round-robin mapping using the actual ethernet channel IDs from active_channels
        size_t channel_index = 0;
        for (auto [eth_chan_id, eth_chan_dir] : active_channels) {
            size_t core_index = channel_index % logical_fabric_mux_cores_.size();
            eth_chan_to_core_index_[dev_id][eth_chan_id] = core_index;

            // Determine core type: In both MUX and UDM modes, all worker channels go to MUX (core type 0)
            // The RELAY (core type 1) in UDM mode handles fabric-to-fabric routing, not worker channels
            FabricTensixCoreType core_id = FabricTensixCoreType::MUX;  // Always assign to MUX
            eth_chan_to_core_id_[dev_id][eth_chan_id] = core_id;

            channel_index++;
        }
    }

    return true;
}

void FabricTensixDatamoverConfig::calculate_buffer_allocations() {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();

    // Get buffer size from fabric context
    buffer_size_bytes_full_size_channel_ =
        fabric_context.get_fabric_packet_header_size_bytes() + tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    // Calculate available L1 space for tensix cores
    uint32_t l1_base = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint32_t l1_size = hal.get_dev_size(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);

    // Get L1 alignment requirement
    size_t l1_alignment = hal.get_alignment(tt::tt_metal::HalMemType::L1);

    // Reserve space for both core types with proper alignment
    size_t space_per_risc = l1_size / num_used_riscs_per_tensix_;     // Split between MUX and RELAY
    space_per_risc = (space_per_risc / l1_alignment) * l1_alignment;  // Align down to L1 alignment

    // Get the maximum number of channels per core type based on fabric topology and mode
    auto topology = fabric_context.get_fabric_topology();
    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();

    // Determine num_channels_for_mux based on mode
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        // UDM mode: MUX temporarily has 1 channel (will be changed later based on compute grid cores)
        // RELAY permanently has 1 channel (configured separately in its constructor)
        num_channels_for_mux_ = 1;
    } else {
        // MUX mode: use topology-based channel count
        switch (topology) {
            case tt::tt_fabric::Topology::Linear:
            case tt::tt_fabric::Topology::Ring:
                num_channels_for_mux_ = tt::tt_fabric::builder_config::num_sender_channels_1d_linear;
                break;
            case tt::tt_fabric::Topology::Mesh:
            case tt::tt_fabric::Topology::Torus:
                num_channels_for_mux_ = tt::tt_fabric::builder_config::num_sender_channels_2d_mesh;
                break;
            default: TT_THROW("unknown fabric topology: {}", topology); break;
        }
    }

    // Calculate buffers per channel based on available space and max channels
    size_t space_needed_for_max_channels = num_channels_for_mux_ * buffer_size_bytes_full_size_channel_;
    num_buffers_per_channel_ = std::bit_floor(space_per_risc / space_needed_for_max_channels);

    // Set base addresses for each core type with proper L1 alignment
    for (size_t i = 0; i < num_used_riscs_per_tensix_; ++i) {
        FabricTensixCoreType core_id = static_cast<FabricTensixCoreType>(i);
        base_l1_addresses_[core_id] = l1_base + (i * space_per_risc);
    }
}

std::shared_ptr<FabricTensixDatamoverMuxConfig> FabricTensixDatamoverConfig::create_mux_config(
    FabricTensixCoreType core_id) {
    return std::make_shared<FabricTensixDatamoverMuxConfig>(
        static_cast<uint8_t>(num_channels_for_mux_),     // num_full_size_channels
        0,                                               // num_header_only_channels
        static_cast<uint8_t>(num_buffers_per_channel_),  // num_buffers_full_size_channel
        0,                                               // num_buffers_header_only_channel
        buffer_size_bytes_full_size_channel_,            // buffer_size_bytes_full_size_channel
        base_l1_addresses_[core_id],                     // base_l1_address
        CoreType::WORKER                                 // core_type
    );
}

std::shared_ptr<FabricTensixDatamoverRelayConfig> FabricTensixDatamoverConfig::create_relay_config(
    FabricTensixCoreType core_id) {
    return std::make_shared<FabricTensixDatamoverRelayConfig>(
        static_cast<uint8_t>(num_buffers_per_channel_),  // num_buffers_per_channel
        buffer_size_bytes_full_size_channel_,            // buffer_size_bytes
        base_l1_addresses_[core_id],                     // base_l1_address
        CoreType::WORKER                                 // core_type
    );
}

void FabricTensixDatamoverConfig::create_configs() {
    // Get the fabric tensix config mode
    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();

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
    tt::tt_metal::IDevice* device, uint32_t eth_chan_id) const {
    CoreCoord core = get_core_for_channel(device->id(), eth_chan_id);
    CoreCoord physical_core = device->worker_core_from_logical_core(core);
    return {physical_core.x, physical_core.y};
}

size_t FabricTensixDatamoverConfig::get_channels_base_address(
    FabricTensixCoreType core_id, uint8_t tensix_channel_id) const {
    auto config = get_config(core_id);
    return config->get_channel_base_address(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, tensix_channel_id);
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
    return configs_.find(core_id) != configs_.end();
}

size_t FabricTensixDatamoverConfig::get_local_flow_control_semaphore_address(
    ChipId device_id, uint32_t eth_chan_id, uint32_t channel_id, FabricTensixCoreType core_id) const {
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return config->get_flow_control_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_connection_semaphore_address(
    ChipId device_id, uint32_t eth_chan_id, uint32_t channel_id, FabricTensixCoreType core_id) const {
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return config->get_connection_handshake_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_worker_conn_info_base_address(
    ChipId device_id, uint32_t eth_chan_id, uint32_t channel_id, FabricTensixCoreType core_id) const {
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return config->get_connection_info_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_buffer_index_semaphore_address(
    ChipId device_id, uint32_t eth_chan_id, uint32_t channel_id, FabricTensixCoreType core_id) const {
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return config->get_buffer_index_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_channel_credits_stream_id(
    ChipId device_id, uint32_t eth_chan_id, uint32_t channel_id, FabricTensixCoreType core_id) const {
    auto config = get_config(core_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return config->get_channel_credits_stream_id(channel_type, channel_id);
}

std::pair<uint32_t, uint32_t> FabricTensixDatamoverConfig::get_termination_address_and_signal(
    ChipId device_id, uint32_t eth_chan_id, FabricTensixCoreType core_id) const {
    TT_FATAL(
        is_core_id_active(core_id), "Core type {} is not active in fabric tensix config", static_cast<int>(core_id));

    auto config = get_config(core_id);
    return std::make_pair(
        config->get_termination_signal_address(), tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
}

// FabricTensixDatamoverBuilder implementation

FabricTensixDatamoverBuilder::FabricTensixDatamoverBuilder(
    std::unique_ptr<FabricTensixDatamoverMuxBuilder> mux_builder,
    std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder) :
    mux_builder_(std::move(mux_builder)), relay_builder_(std::move(relay_builder)) {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder cannot be null");
}

FabricTensixDatamoverBuilder FabricTensixDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& /*program*/,
    tt::tt_fabric::FabricNodeId local_fabric_node_id,
    tt::tt_fabric::FabricNodeId remote_fabric_node_id,
    uint32_t ethernet_channel_id,
    eth_chan_directions direction) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& tensix_config = fabric_context.get_tensix_config();
    auto fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();

    // Get core for this ethernet channel
    CoreCoord my_core_logical = tensix_config.get_core_for_channel(device->id(), ethernet_channel_id);

    // Get NOC coordinates
    auto [noc_x, noc_y] = tensix_config.get_noc_xy(device, ethernet_channel_id);

    // Get link index (routing plane ID) from control plane using channel ID
    uint32_t link_idx = control_plane.get_routing_plane_id(local_fabric_node_id, ethernet_channel_id);

    std::unique_ptr<FabricTensixDatamoverMuxBuilder> mux_builder = nullptr;
    std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder = nullptr;

    switch (fabric_tensix_config) {
        case tt::tt_fabric::FabricTensixConfig::MUX: {
            // MUX mode: only create mux builder for MUX core type
            auto mux_config_base = tensix_config.get_config(FabricTensixCoreType::MUX);
            auto mux_config = std::dynamic_pointer_cast<FabricTensixDatamoverMuxConfig>(mux_config_base);
            TT_FATAL(mux_config != nullptr, "Expected mux config for MUX core type");

            mux_builder = std::make_unique<FabricTensixDatamoverMuxBuilder>(
                my_core_logical,
                local_fabric_node_id,
                remote_fabric_node_id,
                ethernet_channel_id,
                link_idx,
                FabricTensixCoreType::MUX,
                noc_x,
                noc_y,
                mux_config,
                direction);
            break;
        }
        case tt::tt_fabric::FabricTensixConfig::UDM: {
            // UDM mode: create both mux builder (MUX core type) and relay builder (RELAY core type)

            // Create mux builder for MUX core type
            auto mux_config_base = tensix_config.get_config(FabricTensixCoreType::MUX);
            auto mux_config = std::dynamic_pointer_cast<FabricTensixDatamoverMuxConfig>(mux_config_base);
            TT_FATAL(mux_config != nullptr, "Expected mux config for MUX core type");

            mux_builder = std::make_unique<FabricTensixDatamoverMuxBuilder>(
                my_core_logical,
                local_fabric_node_id,
                remote_fabric_node_id,
                ethernet_channel_id,
                link_idx,
                FabricTensixCoreType::MUX,
                noc_x,
                noc_y,
                mux_config,
                direction);

            // Create relay builder for RELAY core type
            if (tensix_config.is_core_id_active(FabricTensixCoreType::RELAY)) {
                auto relay_config_base = tensix_config.get_config(FabricTensixCoreType::RELAY);
                auto relay_config = std::dynamic_pointer_cast<FabricTensixDatamoverRelayConfig>(relay_config_base);
                TT_FATAL(relay_config != nullptr, "Expected relay config for RELAY core type");

                relay_builder = std::make_unique<FabricTensixDatamoverRelayBuilder>(
                    my_core_logical,
                    local_fabric_node_id,
                    remote_fabric_node_id,
                    ethernet_channel_id,
                    link_idx,
                    FabricTensixCoreType::RELAY,
                    noc_x,
                    noc_y,
                    relay_config,
                    direction);
            }
            break;
        }
        default: TT_THROW("Unsupported FabricTensixConfig mode: {}", static_cast<int>(fabric_tensix_config));
    }

    return FabricTensixDatamoverBuilder(std::move(mux_builder), std::move(relay_builder));
}

void FabricTensixDatamoverBuilder::create_and_compile(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
    // Always create and compile mux builder
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    mux_builder_->create_and_compile(device, program);

    // In UDM mode, also create and compile relay builder
    if (relay_builder_ != nullptr) {
        relay_builder_->create_and_compile(device, program);
    }
}

tt::tt_fabric::SenderWorkerAdapterSpec FabricTensixDatamoverBuilder::build_connection_to_fabric_channel(
    uint32_t channel_id) const {
    // Delegate to mux builder
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->build_connection_to_fabric_channel(channel_id);
}

const CoreCoord& FabricTensixDatamoverBuilder::get_logical_core() const {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->get_logical_core();
}

tt::tt_fabric::FabricNodeId FabricTensixDatamoverBuilder::get_local_fabric_node_id() const {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->get_local_fabric_node_id();
}

tt::tt_fabric::FabricNodeId FabricTensixDatamoverBuilder::get_remote_fabric_node_id() const {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->get_remote_fabric_node_id();
}

uint32_t FabricTensixDatamoverBuilder::get_ethernet_channel_id() const {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->get_ethernet_channel_id();
}

FabricTensixCoreType FabricTensixDatamoverBuilder::get_core_id() const {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->get_core_id();
}

uint32_t FabricTensixDatamoverBuilder::get_noc_x() const {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->get_noc_x();
}

uint32_t FabricTensixDatamoverBuilder::get_noc_y() const {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->get_noc_y();
}

eth_chan_directions FabricTensixDatamoverBuilder::get_direction() const {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    return mux_builder_->get_direction();
}

void FabricTensixDatamoverBuilder::append_upstream_routers_noc_xy(uint32_t noc_x, uint32_t noc_y) {
    TT_FATAL(mux_builder_ != nullptr, "Mux builder must not be null");
    mux_builder_->append_upstream_routers_noc_xy(noc_x, noc_y);
}

}  // namespace tt::tt_fabric
