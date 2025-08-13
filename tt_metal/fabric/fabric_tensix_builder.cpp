// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_tensix_builder.hpp"

#include <tt-metalium/assert.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include <tt-metalium/core_descriptor.hpp>
#include <tt-metalium/device_pool.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_align.hpp"
#include <bit>
#include <algorithm>

namespace tt::tt_fabric {

// FabricTensixDatamoverConfig implementation

FabricTensixDatamoverConfig::FabricTensixDatamoverConfig() {
    // Initialize channel mappings and configurations
    initialize_channel_mappings();
    calculate_buffer_allocations();
    create_mux_configs();
}

void FabricTensixDatamoverConfig::initialize_channel_mappings() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto num_devices = cluster.number_of_devices();

    // Get logical fabric mux cores from the first available device (same for all devices), except for TG
    const bool is_TG =
        (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::TG);

    const auto& all_active_devices = tt::DevicePool::instance().get_all_active_devices();
    TT_FATAL(!all_active_devices.empty(), "No active devices found in DevicePool");

    auto device_id = all_active_devices.front()->id();
    if (is_TG) {
        // For TG systems, use a non-MMIO device for fabric mux cores
        for (const auto& device : all_active_devices) {
            if (device && !device->is_mmio_capable()) {
                device_id = device->id();
                break;
            }
        }
    }

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
    if (device != nullptr) {
        for (const auto& logical_core : logical_fabric_mux_cores_) {
            CoreCoord translated_core = device->worker_core_from_logical_core(logical_core);
            translated_fabric_or_dispatch_mux_cores_.insert(translated_core);
        }
        for (const auto& logical_core : logical_dispatch_mux_cores_) {
            CoreCoord translated_core = device->worker_core_from_logical_core(logical_core);
            translated_fabric_or_dispatch_mux_cores_.insert(translated_core);
        }
        log_info(
            tt::LogTest,
            "DEBUG: Initialized {} translated fabric mux cores",
            translated_fabric_or_dispatch_mux_cores_.size());
    }

    // Get maximum number of active ethernet channels from control plane across all devices
    size_t max_eth_channels = 0;

    // First pass: find max_eth_channels
    for (const auto& device : all_active_devices) {
        if (!(is_TG && device->is_mmio_capable())) {
            auto dev_id = device->id();
            auto dev_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dev_id);

            // Get all active ethernet channels for this device
            auto active_channels = control_plane.get_active_fabric_eth_channels(dev_fabric_node_id);
            max_eth_channels = std::max(max_eth_channels, active_channels.size());
        }
    }

    TT_FATAL(max_eth_channels > 0, "No active ethernet channels found in the system");
    TT_FATAL(!logical_fabric_mux_cores_.empty(), "logical_fabric_mux_cores_ is empty before division");

    // Calculate number of configs per core and riscs needed BEFORE using them
    num_configs_per_core_ =
        (max_eth_channels + logical_fabric_mux_cores_.size() - 1) / logical_fabric_mux_cores_.size();
    num_used_riscs_per_tensix_ = num_configs_per_core_;

    // Second pass: create per-device channel mappings using real ethernet channel IDs
    for (const auto& device : all_active_devices) {
        if (!(is_TG && device->is_mmio_capable())) {
            auto dev_id = device->id();
            auto dev_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dev_id);

            // Get all active ethernet channels for this device
            auto active_channels = control_plane.get_active_fabric_eth_channels(dev_fabric_node_id);

            // Initialize per-device mappings
            eth_chan_to_core_index_[dev_id] = std::unordered_map<size_t, size_t>();
            eth_chan_to_risc_id_[dev_id] = std::unordered_map<size_t, size_t>();

            // Create round-robin mapping using the actual ethernet channel IDs from active_channels
            size_t channel_index = 0;
            for (auto [eth_chan_id, eth_chan_dir] : active_channels) {
                size_t core_index = channel_index % logical_fabric_mux_cores_.size();
                eth_chan_to_core_index_[dev_id][eth_chan_id] = core_index;

                // Determine RISC ID: round-robin assignment (0 = BRISC, 1 = NCRISC, etc.)
                size_t channels_on_core = (channel_index / logical_fabric_mux_cores_.size());
                size_t risc_id = channels_on_core % num_used_riscs_per_tensix_;
                eth_chan_to_risc_id_[dev_id][eth_chan_id] = risc_id;

                channel_index++;
            }
        }
    }

    // Debug logging
    log_info(tt::LogTest, "DEBUG: num_configs_per_core_ = {}", num_configs_per_core_);
    log_info(tt::LogTest, "DEBUG: num_used_riscs_per_tensix_ = {}", num_used_riscs_per_tensix_);

    // Debug log eth_chan_to_risc_id_
    for (const auto& [dev_id, channel_map] : eth_chan_to_risc_id_) {
        for (const auto& [eth_chan_id, risc_id] : channel_map) {
            log_info(tt::LogTest, "DEBUG: eth_chan_to_risc_id_[{}][{}] = {}", dev_id, eth_chan_id, risc_id);
        }
    }

    // Debug log eth_chan_to_core_index_
    for (const auto& [dev_id, channel_map] : eth_chan_to_core_index_) {
        for (const auto& [eth_chan_id, core_index] : channel_map) {
            log_info(tt::LogTest, "DEBUG: eth_chan_to_core_index_[{}][{}] = {}", dev_id, eth_chan_id, core_index);
        }
    }
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

    // Reserve space for both RISC types with proper alignment
    size_t space_per_risc = l1_size / num_used_riscs_per_tensix_;     // Split between BRISC and NCRISC
    space_per_risc = (space_per_risc / l1_alignment) * l1_alignment;  // Align down to L1 alignment

    // Get the maximum number of channels per RISC based on fabric topology from fabric context
    auto topology = fabric_context.get_fabric_topology();

    switch (topology) {
        case tt::tt_fabric::Topology::Linear:
            num_channels_ = tt::tt_fabric::FabricEriscDatamoverConfig::num_sender_channels_1d_linear;
            break;
        case tt::tt_fabric::Topology::Mesh:
            num_channels_ = tt::tt_fabric::FabricEriscDatamoverConfig::num_sender_channels_2d_mesh;
            break;
        case tt::tt_fabric::Topology::Ring:
            num_channels_ = tt::tt_fabric::FabricEriscDatamoverConfig::num_sender_channels_1d_ring;
            break;
        case tt::tt_fabric::Topology::Torus:
            num_channels_ = tt::tt_fabric::FabricEriscDatamoverConfig::num_sender_channels_2d_torus;
            break;
        default: TT_THROW("unknow fabric topology: {}", topology); break;
    }

    // Calculate buffers per channel based on available space and max channels
    size_t space_needed_for_max_channels = num_channels_ * buffer_size_bytes_full_size_channel_;
    num_buffers_per_channel_ = std::bit_floor(space_per_risc / space_needed_for_max_channels);

    // Set base addresses for each RISC ID with proper L1 alignment
    for (size_t risc_id = 0; risc_id < num_used_riscs_per_tensix_; ++risc_id) {
        base_l1_addresses_[risc_id] = l1_base + (risc_id * space_per_risc);
    }

    // Debug logging
    log_info(tt::LogTest, "DEBUG: num_buffers_per_channel_ = {}", num_buffers_per_channel_);

    // Debug log base_l1_addresses_
    for (const auto& [risc_id, address] : base_l1_addresses_) {
        log_info(tt::LogTest, "DEBUG: base_l1_addresses_[{}] = {}", risc_id, address);
    }
}

void FabricTensixDatamoverConfig::create_mux_configs() {
    // Create mux configs for the number of RISCs we actually need - much cleaner!
    for (size_t risc_id = 0; risc_id < num_used_riscs_per_tensix_; ++risc_id) {
        mux_configs_[risc_id] = std::make_shared<tt::tt_fabric::FabricMuxConfig>(
            static_cast<uint8_t>(num_channels_),             // num_full_size_channels
            0,                                               // num_header_only_channels (set to 0 for now)
            static_cast<uint8_t>(num_buffers_per_channel_),  // num_buffers_full_size_channel
            0,                                               // num_buffers_header_only_channel
            buffer_size_bytes_full_size_channel_,            // buffer_size_bytes_full_size_channel
            base_l1_addresses_[risc_id],                     // base_l1_address
            CoreType::WORKER                                 // core_type (always tensix)
        );
    }
}

// Getter implementations
size_t FabricTensixDatamoverConfig::get_base_l1_address(size_t risc_id) const {
    auto it = base_l1_addresses_.find(risc_id);
    TT_FATAL(it != base_l1_addresses_.end(), "Base L1 address not found for RISC ID {}", risc_id);
    return it->second;
}

std::pair<uint32_t, uint32_t> FabricTensixDatamoverConfig::get_noc_xy(
    tt::tt_metal::IDevice* device, uint32_t eth_chan_id) const {
    CoreCoord core = get_core_for_channel(device->id(), eth_chan_id);
    CoreCoord physical_core = device->worker_core_from_logical_core(core);
    return {physical_core.x, physical_core.y};
}

size_t FabricTensixDatamoverConfig::get_channels_base_address(size_t risc_id, uint8_t tensix_channel_id) const {
    auto mux_config = get_mux_config(risc_id);
    return mux_config->get_channel_base_address(
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, tensix_channel_id);
}

size_t FabricTensixDatamoverConfig::get_risc_id_for_channel(chip_id_t device_id, uint32_t eth_chan_id) const {
    auto device_it = eth_chan_to_risc_id_.find(device_id);
    TT_FATAL(device_it != eth_chan_to_risc_id_.end(), "Device {} not found in risc mapping", device_id);

    auto it = device_it->second.find(eth_chan_id);
    TT_FATAL(
        it != device_it->second.end(),
        "RISC ID not found for ethernet channel {} on device {}",
        eth_chan_id,
        device_id);
    return it->second;
}

CoreCoord FabricTensixDatamoverConfig::get_core_for_channel(chip_id_t device_id, uint32_t eth_chan_id) const {
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

std::shared_ptr<tt::tt_fabric::FabricMuxConfig> FabricTensixDatamoverConfig::get_mux_config(size_t risc_id) const {
    auto it = mux_configs_.find(risc_id);
    TT_FATAL(it != mux_configs_.end(), "Mux config not found for RISC ID {}", risc_id);
    return it->second;
}

bool FabricTensixDatamoverConfig::is_risc_id_active(size_t risc_id) const {
    return mux_configs_.find(risc_id) != mux_configs_.end();
}

size_t FabricTensixDatamoverConfig::get_local_flow_control_semaphore_address(
    chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const {
    auto risc_id = get_risc_id_for_channel(device_id, eth_chan_id);
    auto mux_config = get_mux_config(risc_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return mux_config->get_flow_control_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_connection_semaphore_address(
    chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const {
    auto risc_id = get_risc_id_for_channel(device_id, eth_chan_id);
    auto mux_config = get_mux_config(risc_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return mux_config->get_connection_handshake_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_worker_conn_info_base_address(
    chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const {
    auto risc_id = get_risc_id_for_channel(device_id, eth_chan_id);
    auto mux_config = get_mux_config(risc_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return mux_config->get_connection_info_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_buffer_index_semaphore_address(
    chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const {
    auto risc_id = get_risc_id_for_channel(device_id, eth_chan_id);
    auto mux_config = get_mux_config(risc_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return mux_config->get_buffer_index_address(channel_type, channel_id);
}

size_t FabricTensixDatamoverConfig::get_channel_credits_stream_id(
    chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const {
    auto risc_id = get_risc_id_for_channel(device_id, eth_chan_id);
    auto mux_config = get_mux_config(risc_id);
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    return mux_config->get_channel_credits_stream_id(channel_type, channel_id);
}

// FabricTensixDatamoverBuilder implementation

FabricTensixDatamoverBuilder::FabricTensixDatamoverBuilder(
    const CoreCoord& my_core_logical,
    tt::tt_fabric::FabricNodeId local_fabric_node_id,
    tt::tt_fabric::FabricNodeId remote_fabric_node_id,
    uint32_t ethernet_channel_id,
    uint32_t link_idx,
    size_t risc_id,
    uint32_t noc_x,
    uint32_t noc_y,
    std::shared_ptr<tt::tt_fabric::FabricMuxConfig> fabric_mux_config) :
    my_core_logical_(my_core_logical),
    local_fabric_node_id_(local_fabric_node_id),
    remote_fabric_node_id_(remote_fabric_node_id),
    ethernet_channel_id_(ethernet_channel_id),
    link_idx_(link_idx),
    risc_id_(risc_id),
    noc_x_(noc_x),
    noc_y_(noc_y),
    fabric_mux_config_(fabric_mux_config) {
    TT_FATAL(fabric_mux_config_ != nullptr, "FabricMuxConfig cannot be null");
}

FabricTensixDatamoverBuilder FabricTensixDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    tt::tt_fabric::FabricNodeId local_fabric_node_id,
    tt::tt_fabric::FabricNodeId remote_fabric_node_id,
    uint32_t ethernet_channel_id) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();

    // Get the fabric tensix datamover config from fabric context
    const auto& tensix_config = fabric_context.get_fabric_tensix_config();

    // Get RISC ID for this ethernet channel
    size_t risc_id = tensix_config.get_risc_id_for_channel(device->id(), ethernet_channel_id);

    // Get core for this ethernet channel
    CoreCoord my_core_logical = tensix_config.get_core_for_channel(device->id(), ethernet_channel_id);

    // Get NOC coordinates
    auto [noc_x, noc_y] = tensix_config.get_noc_xy(device, ethernet_channel_id);

    // Get link index (routing plane ID) from control plane using channel ID
    uint32_t link_idx = control_plane.get_routing_plane_id(local_fabric_node_id, ethernet_channel_id);

    // Get mux config for this RISC ID
    auto fabric_mux_config = tensix_config.get_mux_config(risc_id);

    return FabricTensixDatamoverBuilder(
        my_core_logical,
        local_fabric_node_id,
        remote_fabric_node_id,
        ethernet_channel_id,
        link_idx,
        risc_id,
        noc_x,
        noc_y,
        fabric_mux_config);
}

void FabricTensixDatamoverBuilder::create_and_compile(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
    // Select processor and NOC based on RISC ID (much cleaner!)
    tt::tt_metal::DataMovementProcessor processor =
        (risc_id_ == 0) ? tt::tt_metal::DataMovementProcessor::RISCV_0 : tt::tt_metal::DataMovementProcessor::RISCV_1;

    tt::tt_metal::NOC noc = (risc_id_ == 0) ? tt::tt_metal::NOC::RISCV_0_default : tt::tt_metal::NOC::RISCV_1_default;

    // Create the mux kernel using the fabric mux kernel file
    auto mux_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        my_core_logical_,
        tt::tt_metal::DataMovementConfig{
            .processor = processor, .noc = noc, .compile_args = get_compile_time_args(), .defines = {}});

    // Set runtime arguments
    tt::tt_metal::SetRuntimeArgs(program, mux_kernel, my_core_logical_, get_runtime_args(program));
}

tt::tt_fabric::SenderWorkerAdapterSpec FabricTensixDatamoverBuilder::build_connection_to_fabric_channel(
    uint32_t channel_id) const {
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;

    return tt::tt_fabric::SenderWorkerAdapterSpec{
        noc_x_,                                                                  // edm_noc_x
        noc_y_,                                                                  // edm_noc_y
        fabric_mux_config_->get_channel_base_address(channel_type, channel_id),  // edm_buffer_base_addr
        fabric_mux_config_->get_num_buffers(channel_type),                       // num_buffers_per_channel
        fabric_mux_config_->get_flow_control_address(channel_type, channel_id),  // edm_l1_sem_addr
        fabric_mux_config_->get_connection_handshake_address(
            channel_type, channel_id),                                              // edm_connection_handshake_addr
        fabric_mux_config_->get_connection_info_address(channel_type, channel_id),  // edm_worker_location_info_addr
        fabric_mux_config_->get_buffer_size_bytes(channel_type),                    // buffer_size_bytes
        fabric_mux_config_->get_buffer_index_address(channel_type, channel_id),     // buffer_index_semaphore_id
        tt::tt_fabric::eth_chan_directions::EAST                                    // edm_direction
    };
}

std::vector<uint32_t> FabricTensixDatamoverBuilder::get_compile_time_args() const {
    // Get compile time args from the underlying mux config
    return fabric_mux_config_->get_fabric_mux_compile_time_args();
}

std::vector<uint32_t> FabricTensixDatamoverBuilder::get_runtime_args(tt::tt_metal::Program& program) const {
    log_info(tt::LogTest, "FabricTensixDatamoverBuilder::get_runtime_args called");
    log_info(
        tt::LogTest,
        "  local_fabric_node_id: mesh_id={}, chip_id={}",
        local_fabric_node_id_.mesh_id,
        local_fabric_node_id_.chip_id);
    log_info(
        tt::LogTest,
        "  remote_fabric_node_id: mesh_id={}, chip_id={}",
        remote_fabric_node_id_.mesh_id,
        remote_fabric_node_id_.chip_id);
    log_info(tt::LogTest, "  link_idx: {}", link_idx_);
    log_info(tt::LogTest, "  my_core_logical: x={}, y={}", my_core_logical_.x, my_core_logical_.y);
    log_info(tt::LogTest, "  ethernet_channel_id: {}", ethernet_channel_id_);
    log_info(tt::LogTest, "  risc_id: {}", risc_id_);
    log_info(tt::LogTest, "  noc_x: {}, noc_y: {}", noc_x_, noc_y_);

    // Get runtime args from the underlying mux config
    auto args = fabric_mux_config_->get_fabric_mux_run_time_args(
        local_fabric_node_id_, remote_fabric_node_id_, link_idx_, program, {my_core_logical_});

    // log_info(tt::LogTest, "Runtime args breakdown:");

    // if (args.size() > 0) {
    //     size_t idx = 0;

    //     // First arg is num_regions_to_clear
    //     log_info(tt::LogTest, "  num_regions_to_clear: {}", args[idx++]);
    //     uint32_t num_regions = args[0];

    //     // Next are address/size pairs for regions to clear
    //     for (uint32_t i = 0; i < num_regions && idx + 1 < args.size(); ++i) {
    //         log_info(tt::LogTest, "  region[{}]_address: {}", i, args[idx++]);
    //         log_info(tt::LogTest, "  region[{}]_size: {}", i, args[idx++]);
    //     }

    //     // Remaining args are from append_fabric_connection_rt_args
    //     if (idx < args.size()) {
    //         log_info(tt::LogTest, "  fabric_connection_args starting at index {}", idx);

    //         // Based on append_worker_to_fabric_edm_sender_rt_args, these should be:
    //         if (idx < args.size()) {
    //             log_info(tt::LogTest, "  eth_channel: {}", args[idx++]);
    //         }
    //         if (idx < args.size()) {
    //             log_info(tt::LogTest, "  sender_worker_terminate_semaphore_id: {}", args[idx++]);
    //         }
    //         if (idx < args.size()) {
    //             log_info(tt::LogTest, "  sender_worker_buffer_index_semaphore_id: {}", args[idx++]);
    //         }

    //         // Log any remaining args
    //         while (idx < args.size()) {
    //             log_info(tt::LogTest, "  additional_arg[{}]: {}", idx, args[idx]);
    //             idx++;
    //         }
    //     }
    // }

    // log_info(tt::LogTest, "Total runtime args count: {}", args.size());

    return args;
}

}  // namespace tt::tt_fabric
