// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <enchantum/enchantum.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/assert.hpp>
#include <algorithm>
#include <numeric>

namespace tt::tt_fabric {

size_t FabricStaticSizedChannelsAllocator::get_sender_channel_base_address(size_t channel_id) const {
    TT_FATAL(channel_id < sender_channels_base_address.size(), "Sender channel ID {} out of bounds", channel_id);
    return sender_channels_base_address[channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_sender_channel_number_of_slots(size_t channel_id) const {
    TT_FATAL(channel_id < sender_channels_num_buffers.size(), "Sender channel ID {} out of bounds", channel_id);
    return sender_channels_num_buffers[channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_receiver_channel_number_of_slots(size_t channel_id) const {
    TT_FATAL(channel_id < receiver_channels_num_buffers.size(), "Receiver channel ID {} out of bounds", channel_id);
    return receiver_channels_num_buffers[channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_receiver_channel_base_address(size_t channel_id) const {
    TT_FATAL(channel_id < receiver_channels_base_address.size(), "Receiver channel ID {} out of bounds", channel_id);
    return receiver_channels_base_address[channel_id];
}

FabricStaticSizedChannelsAllocator::FabricStaticSizedChannelsAllocator(
    tt::tt_fabric::Topology topology,
    const FabricEriscDatamoverOptions& options,
    size_t num_used_sender_channels,
    size_t num_used_receiver_channels,
    size_t channel_buffer_size_bytes,
    size_t available_channel_buffering_space,
    const std::vector<MemoryRegion>& memory_regions) :
    FabricChannelAllocator(topology, options, memory_regions),
    num_used_sender_channels(num_used_sender_channels),
    num_used_receiver_channels(num_used_receiver_channels),
    channel_buffer_size_bytes(channel_buffer_size_bytes),
    available_channel_buffering_space(available_channel_buffering_space) {
    // Compute buffer region start from memory regions
    TT_FATAL(!memory_regions.empty(), "Memory regions must not be empty");
    this->buffer_region_start = memory_regions[0].start_address;

    // Set max_l1_loading_size from memory regions
    this->max_l1_loading_size = memory_regions[0].get_end_address();
    for (const auto& region : memory_regions) {
        this->max_l1_loading_size = std::max(this->max_l1_loading_size, region.get_end_address());
    }
    std::array<size_t, builder_config::num_sender_channels> num_sender_buffer_slots = {0};
    std::array<size_t, builder_config::num_sender_channels> num_remote_sender_buffer_slots = {0};
    std::array<size_t, builder_config::num_receiver_channels> num_receiver_buffer_slots = {0};
    std::array<size_t, builder_config::num_receiver_channels> num_remote_receiver_buffer_slots = {0};

    bool is_dateline = options.edm_type == FabricEriscDatamoverType::Dateline;
    bool is_dateline_upstream = options.edm_type == FabricEriscDatamoverType::DatelineUpstream;
    bool is_dateline_upstream_adj_dev = options.edm_type == FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
    bool has_tensix_extension = options.fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED;

    configure_buffer_slots_helper(
        topology,
        options,
        num_sender_buffer_slots,
        num_remote_sender_buffer_slots,
        num_receiver_buffer_slots,
        num_remote_receiver_buffer_slots,
        options.direction);

    log_trace(
        tt::LogOp,
        "\tis_dateline {} is_dateline_upstream {} is_dateline_upstream_adj_dev {}",
        is_dateline,
        is_dateline_upstream,
        is_dateline_upstream_adj_dev);
    log_trace(tt::LogOp, "\tnum_sender_buffer_slots: {}", num_sender_buffer_slots);
    log_trace(tt::LogOp, "\tnum_remote_sender_buffer_slots: {}", num_remote_sender_buffer_slots);
    log_trace(tt::LogOp, "\tnum_receiver_buffer_slots: {}", num_receiver_buffer_slots);
    log_trace(tt::LogOp, "\tnum_remote_receiver_buffer_slots: {}", num_remote_receiver_buffer_slots);

    size_t total_sender_slots = std::accumulate(
        num_sender_buffer_slots.begin(), num_sender_buffer_slots.begin() + num_used_sender_channels, size_t{0});
    size_t total_receiver_slots = std::accumulate(
        num_receiver_buffer_slots.begin(), num_receiver_buffer_slots.begin() + num_used_receiver_channels, size_t{0});
    std::size_t total_slot_count = total_sender_slots + total_receiver_slots;
    TT_FATAL(
        total_slot_count * channel_buffer_size_bytes <= available_channel_buffering_space,
        "Total channel size of {} B exceeds available space of {} B",
        total_slot_count * channel_buffer_size_bytes,
        available_channel_buffering_space);

    log_trace(tt::LogOp, "\tAvailable channel buffering space: {}", this->available_channel_buffering_space);

    // set the sender channel sizes and num buffers
    for (uint32_t i = 0; i < num_used_sender_channels; i++) {
        this->sender_channels_size_bytes[i] = channel_buffer_size_bytes * num_sender_buffer_slots[i];
        this->sender_channels_num_buffers[i] = num_sender_buffer_slots[i];
    }
    // set the remote sender channel sizes and num buffers
    for (uint32_t i = 0; i < num_used_sender_channels; i++) {
        this->remote_sender_channels_size_bytes[i] = channel_buffer_size_bytes * num_remote_sender_buffer_slots[i];
        this->remote_sender_channels_num_buffers[i] = num_remote_sender_buffer_slots[i];
    }
    // set the local receiver channel sizes and num buffers
    for (uint32_t i = 0; i < num_used_receiver_channels; i++) {
        this->receiver_channels_size_bytes[i] = channel_buffer_size_bytes * num_receiver_buffer_slots[i];
        this->receiver_channels_num_buffers[i] = num_receiver_buffer_slots[i];
    }
    // set the remote receiver channel sizes and num buffers
    for (uint32_t i = 0; i < num_used_receiver_channels; i++) {
        this->remote_receiver_channels_size_bytes[i] = channel_buffer_size_bytes * num_remote_receiver_buffer_slots[i];
        this->remote_receiver_channels_num_buffers[i] = num_remote_receiver_buffer_slots[i];
    }

    // set the base addresses for the local channels
    uint32_t sender_buffer_addr = buffer_region_start;
    for (uint32_t i = 0; i < num_used_sender_channels; i++) {
        this->sender_channels_base_address[i] = sender_buffer_addr;
        sender_buffer_addr += this->sender_channels_size_bytes[i];
        log_trace(tt::LogOp, "\tSender {} channel_start: {}", i, this->sender_channels_base_address[i]);
    }
    uint32_t receiver_buffer_addr = sender_buffer_addr;
    for (uint32_t i = 0; i < num_used_receiver_channels; i++) {
        this->receiver_channels_base_address[i] = receiver_buffer_addr;
        receiver_buffer_addr += this->receiver_channels_size_bytes[i];
        log_trace(tt::LogOp, "\tReceiver {} channel_start: {}", i, this->receiver_channels_base_address[i]);
    }
    uint32_t buffer_addr_end = receiver_buffer_addr;
    // set the base addresses for the remote channels
    uint32_t remote_sender_buffer_addr = buffer_region_start;
    for (uint32_t i = 0; i < num_used_sender_channels; i++) {
        this->remote_sender_channels_base_address[i] = remote_sender_buffer_addr;
        remote_sender_buffer_addr += this->remote_sender_channels_size_bytes[i];
        log_trace(tt::LogOp, "\tRemote Sender {} channel_start: {}", i, this->remote_sender_channels_base_address[i]);
    }
    uint32_t remote_receiver_buffer_addr = remote_sender_buffer_addr;
    for (uint32_t i = 0; i < num_used_receiver_channels; i++) {
        this->remote_receiver_channels_base_address[i] = remote_receiver_buffer_addr;
        remote_receiver_buffer_addr += this->remote_receiver_channels_size_bytes[i];
        log_trace(
            tt::LogOp, "\tRemote Receiver {} channel_start: {}", i, this->remote_receiver_channels_base_address[i]);
    }

    log_trace(tt::LogOp, "\tAvailable channel buffering space: {}", this->available_channel_buffering_space);

    auto skip_current_sender_channel = [&](uint32_t idx) -> bool {
        // for dateline connection, skip the last sender channel check (2 for 1d, 4 for 2d)
        // for dateline upstream, skip the sender channel 1 check (just for 1d)
        // for dateline upstream adajcent, skip the sender channel 2 check (just for 1d)
        // for fabric with tensix extension, only check the vc1 sender channel and worker channel, other
        // channels are skipped
        bool is_2D_routing = FabricContext::is_2D_topology(topology);
        uint32_t target_channel = get_worker_connected_sender_channel(options.direction, topology);
        uint32_t vc1_target_channel = get_worker_or_vc1_connected_sender_channel(options.direction, topology);
        return (idx == get_dateline_sender_channel_skip_idx(is_2D_routing) && is_dateline) ||
               (idx == this->dateline_upstream_sender_channel_skip_idx && is_dateline_upstream) ||
               (idx == this->dateline_upstream_adjcent_sender_channel_skip_idx && is_dateline_upstream_adj_dev) ||
               (idx != target_channel && idx != vc1_target_channel && has_tensix_extension);
    };

    for (uint32_t i = 0; i < num_used_sender_channels; i++) {
        if (!skip_current_sender_channel(i)) {
            TT_FATAL(
                this->sender_channels_size_bytes[i] > 0,
                "Internal error when computing `sender_channels_size_bytes[{}]` which was computed to be size 0",
                i);
        }
    }

    auto skip_current_receiver_channel = [&](uint32_t idx) -> bool {
        return (idx == this->dateline_receiver_channel_skip_idx && is_dateline) ||
               (idx == this->dateline_upstream_receiver_channel_skip_idx && is_dateline_upstream);
    };

    for (uint32_t i = 0; i < num_used_receiver_channels; i++) {
        if (!skip_current_receiver_channel(i)) {
            TT_FATAL(
                this->receiver_channels_size_bytes[i] > 0,
                "Internal error when computing `receiver_channels_size_bytes[{}]` which was computed to be size 0",
                i);
        }
    }
    TT_FATAL(
        std::accumulate(
            this->sender_channels_size_bytes.begin(),
            this->sender_channels_size_bytes.begin() + num_used_sender_channels,
            0ul) +
                std::accumulate(
                    this->receiver_channels_size_bytes.begin(),
                    this->receiver_channels_size_bytes.begin() + num_used_receiver_channels,
                    0ul) <=
            this->available_channel_buffering_space,
        "Internal error when computing channel sizes. Total channel size exceeds available space");
    TT_FATAL(
        buffer_addr_end < this->max_l1_loading_size,
        "Internal error - channel buffers spilled past the end of usable L1 region.");
}

void FabricStaticSizedChannelsAllocator::configure_buffer_slots_helper(
    Topology topology,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, builder_config::num_sender_channels>& num_sender_buffer_slots,
    std::array<size_t, builder_config::num_sender_channels>& num_remote_sender_buffer_slots,
    std::array<size_t, builder_config::num_receiver_channels>& num_receiver_buffer_slots,
    std::array<size_t, builder_config::num_receiver_channels>& num_remote_receiver_buffer_slots,
    eth_chan_directions direction) {
    // fabric with tensix extension uses different buffer slots options, since only one or two sender channels are
    // used by fabric router, while other sender channels are skipped and have 0 buffer slots.
    static const std::vector<std::vector<std::pair<size_t, size_t>>> default_with_tensix_buffer_slot_options = {
        {{16, 16}, {8, 16}, {8, 8}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
        {{16, 32}, {16, 16}, {8, 16}, {8, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
    };

    static const std::vector<std::vector<std::pair<size_t, size_t>>> ring_buffer_slot_options = {
        {{8, 8}, {4, 8}},
        {{16, 32}, {16, 16}, {8, 16}, {8, 8}, {4, 8}}};

    static const std::vector<std::vector<std::pair<size_t, size_t>>> torus_buffer_slot_options = {
        {{4, 8}, {4, 8}}, {{4, 8}, {4, 8}}};

    static const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> ring_buffer_slot_options_dateline = {
        {{{8, 16}, {8, 8}}, {{16, 16}, {8, 16}, {8, 8}}}, {{{8, 16}, {8, 8}}, {{16, 16}, {8, 16}, {8, 8}}}};

    // TODO: need to investigate why {{8, 8}, {4, 8}} is not working
    static const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> torus_buffer_slot_options_dateline = {
        {{{4, 8}, {8, 8}}, {{4, 8}, {8, 8}}}, {{{4, 8}, {8, 8}}, {{4, 8}, {8, 8}}}};

    static const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>>
        ring_buffer_slot_options_dateline_upstream = {
            {{{8, 16}, {8, 8}}, {{16, 16}, {8, 16}, {8, 8}}}, {{{8, 16}, {8, 8}}, {{16, 16}, {8, 16}, {8, 8}}}};

    static const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>>
        ring_buffer_slot_options_dateline_upstream_adjcent = {
            {{{16, 8}, {8, 8}}, {{16, 8}, {8, 8}}}, {{{16, 8}, {8, 8}}, {{16, 8}, {8, 8}}}};

    auto get_num_buffer_slots = [](Topology topology,
                                   size_t arch_index) -> const std::vector<std::pair<size_t, size_t>>& {
        // Architecture-specific buffer slot configurations
        static const std::vector<std::vector<std::pair<size_t, size_t>>> mesh_buffer_slot_options = {
            {{7, 11}, {4, 8}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
            {{8, 16}, {8, 8}, {4, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
        };
        static const std::vector<std::vector<std::pair<size_t, size_t>>> other_buffer_slot_options = {
            {{8, 16}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
            {{16, 16}, {8, 16}, {8, 8}, {4, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
        };

        static tt::stl::Indestructible<std::vector<std::vector<std::pair<size_t, size_t>>>> mesh_slots(
            mesh_buffer_slot_options);
        static tt::stl::Indestructible<std::vector<std::vector<std::pair<size_t, size_t>>>> other_slots(
            other_buffer_slot_options);

        if (topology == Topology::Mesh) {
            return mesh_slots.get()[arch_index];
        } else {
            return other_slots.get()[arch_index];
        }
    };

    auto get_optimal_num_slots = [this](
                                     auto& buffer_slot_options,
                                     size_t num_sender_channels,
                                     size_t num_receiver_channels,
                                     size_t& num_sender_buffer_slots,
                                     size_t& num_receiver_buffer_slots,
                                     std::optional<size_t> worker_num_sender_buffer_slots = std::nullopt) {
        for (auto& option : buffer_slot_options) {
            num_sender_buffer_slots = option.first;
            num_receiver_buffer_slots = option.second;
            auto num_total_sender_slots = num_sender_channels * num_sender_buffer_slots;
            auto num_total_receiver_slots = num_receiver_channels * num_receiver_buffer_slots;
            if (worker_num_sender_buffer_slots.has_value()) {
                num_total_sender_slots =
                    worker_num_sender_buffer_slots.value() + (num_sender_channels - 1) * num_sender_buffer_slots;
            }
            auto total_num_bytes =
                (num_total_sender_slots + num_total_receiver_slots) * this->channel_buffer_size_bytes;
            if (total_num_bytes <= this->available_channel_buffering_space) {
                break;
            }
        }
    };

    auto fill_sender_buffer_slots = [&](auto& num_buffer_slots,
                                        size_t channel_skip_idx,
                                        uint32_t default_num_buffer_slots,
                                        uint32_t extra_num_buffer_slots) {
        for (size_t i = 0; i < this->num_used_sender_channels; ++i) {
            if (i == channel_skip_idx) {
                num_buffer_slots[i] = 0;
            } else {
                // tensix worker on channel 0, otherwise extra_num_buffer_slots
                num_buffer_slots[i] = (i == 0 ? default_num_buffer_slots : extra_num_buffer_slots);
            }
        }
    };

    auto fill_receiver_buffer_slots =
        [&](auto& num_buffer_slots, size_t channel_skip_idx, uint32_t extra_num_buffer_slots) {
            for (size_t i = 0; i < this->num_used_receiver_channels; ++i) {
                if (i == channel_skip_idx) {
                    num_buffer_slots[i] = 0;
                } else {
                    num_buffer_slots[i] = extra_num_buffer_slots;
                }
            }
        };

    auto axis_index = static_cast<std::size_t>(options.edm_axis);
    auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();
    size_t arch_index;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        arch_index = 0;
    } else if (arch == tt::ARCH::BLACKHOLE) {
        arch_index = 1;
    } else {
        TT_THROW("Unsupported architecture: {}", enchantum::to_string(arch));
    }

    switch (options.fabric_tensix_config) {
        case tt::tt_fabric::FabricTensixConfig::MUX: {
            uint32_t num_sender_channels = this->num_sender_channels_with_tensix_config;
            if (topology == tt::tt_fabric::Topology::Ring || topology == tt::tt_fabric::Topology::Torus) {
                // extra sender channel for vc1
                num_sender_channels = this->num_sender_channels_with_tensix_config_deadlock_avoidance;
            }
            uint32_t target_channel = get_worker_connected_sender_channel(direction, topology);
            uint32_t vc1_target_channel = get_worker_or_vc1_connected_sender_channel(direction, topology);
            size_t default_num_sender_buffer_slots;
            size_t default_num_receiver_buffer_slots;
            // get the default buffer slots
            get_optimal_num_slots(
                default_with_tensix_buffer_slot_options[arch_index],
                num_sender_channels,
                this->num_used_receiver_channels,
                default_num_sender_buffer_slots,
                default_num_receiver_buffer_slots);
            // set default buffer slots.
            num_sender_buffer_slots[target_channel] = default_num_sender_buffer_slots;
            num_sender_buffer_slots[vc1_target_channel] = default_num_sender_buffer_slots;
            num_remote_sender_buffer_slots[target_channel] = default_num_sender_buffer_slots;
            num_remote_sender_buffer_slots[vc1_target_channel] = default_num_sender_buffer_slots;
            num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
            num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
            return;
        }
        default: break;
    }

    if (topology == Topology::Ring) {
        size_t default_num_sender_buffer_slots;
        size_t default_num_receiver_buffer_slots;
        // get the default buffer slots
        get_optimal_num_slots(
            ring_buffer_slot_options[arch_index],
            this->num_used_sender_channels,
            this->num_used_receiver_channels,
            default_num_sender_buffer_slots,
            default_num_receiver_buffer_slots);
        // get the dateline buffer slots
        size_t dateline_num_sender_buffer_slots;
        size_t dateline_num_receiver_buffer_slots;
        get_optimal_num_slots(
            ring_buffer_slot_options_dateline[arch_index][axis_index],
            this->num_used_sender_channels - 1,
            this->num_used_receiver_channels - 1,
            dateline_num_sender_buffer_slots,
            dateline_num_receiver_buffer_slots,
            default_num_sender_buffer_slots);
        // get the dateline upstream buffer slots
        size_t dateline_upstream_num_sender_buffer_slots;
        size_t dateline_upstream_num_receiver_buffer_slots;
        get_optimal_num_slots(
            ring_buffer_slot_options_dateline_upstream[arch_index][axis_index],
            this->num_used_sender_channels - 1,
            this->num_used_receiver_channels - 1,
            dateline_upstream_num_sender_buffer_slots,
            dateline_upstream_num_receiver_buffer_slots,
            default_num_sender_buffer_slots);
        // get the dateline upstream adjacent device buffer slots
        size_t dateline_upstream_adjcent_num_sender_buffer_slots;
        size_t dateline_upstream_adjcent_num_receiver_buffer_slots;
        get_optimal_num_slots(
            ring_buffer_slot_options_dateline_upstream_adjcent[arch_index][axis_index],
            this->num_used_sender_channels - 1,
            this->num_used_receiver_channels,
            dateline_upstream_adjcent_num_sender_buffer_slots,
            dateline_upstream_adjcent_num_receiver_buffer_slots,
            default_num_sender_buffer_slots);
        // set default buffer slots.
        num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
        num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);

        auto buffer_config = options.edm_buffer_config;
        switch (options.edm_type) {
            case FabricEriscDatamoverType::Dateline:
                if (buffer_config.enable_dateline_sender_extra_buffer_slots) {
                    // set num_sender_buffer_slots
                    fill_sender_buffer_slots(
                        num_sender_buffer_slots,
                        this->dateline_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_num_sender_buffer_slots);
                    // set remote sender buffer slots equal to local sender, since remote is also dateline
                    num_remote_sender_buffer_slots = num_sender_buffer_slots;
                }
                if (buffer_config.enable_dateline_receiver_extra_buffer_slots) {
                    // set num_receiver_buffer_slots
                    fill_receiver_buffer_slots(
                        num_receiver_buffer_slots,
                        this->dateline_receiver_channel_skip_idx,
                        dateline_num_receiver_buffer_slots);
                    // set remote receiver buffer slots equal to local receiver, since remote is also dateline
                    num_remote_receiver_buffer_slots = num_receiver_buffer_slots;
                }
                break;
            case FabricEriscDatamoverType::DatelineUpstream:
                if (buffer_config.enable_dateline_upstream_sender_extra_buffer_slots) {
                    // set num_sender_buffer_slots
                    fill_sender_buffer_slots(
                        num_sender_buffer_slots,
                        this->dateline_upstream_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_upstream_num_sender_buffer_slots);
                }
                // set num_receiver_buffer_slots
                if (buffer_config.enable_dateline_upstream_receiver_extra_buffer_slots) {
                    fill_receiver_buffer_slots(
                        num_receiver_buffer_slots,
                        this->dateline_upstream_receiver_channel_skip_idx,
                        dateline_upstream_num_receiver_buffer_slots);
                }
                if (buffer_config.enable_dateline_upstream_adjacent_sender_extra_buffer_slots) {
                    // set remote sender buffer slots equal to dateline upstream dajcent sender buffer slots
                    fill_sender_buffer_slots(
                        num_remote_sender_buffer_slots,
                        this->dateline_upstream_adjcent_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_upstream_adjcent_num_sender_buffer_slots);
                }
                break;
            case FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice:
                if (buffer_config.enable_dateline_upstream_adjacent_sender_extra_buffer_slots) {
                    // set num_sender_buffer_slots
                    fill_sender_buffer_slots(
                        num_sender_buffer_slots,
                        this->dateline_upstream_adjcent_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_upstream_adjcent_num_sender_buffer_slots);
                }
                if (buffer_config.enable_dateline_upstream_sender_extra_buffer_slots) {
                    // set remote sender buffer slots equal to dateline upstream sender buffer slots
                    fill_sender_buffer_slots(
                        num_remote_sender_buffer_slots,
                        this->dateline_upstream_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_upstream_num_sender_buffer_slots);
                }
                if (buffer_config.enable_dateline_upstream_receiver_extra_buffer_slots) {
                    // set remote sender buffer slots equal to dateline upstream sender buffer slots
                    fill_receiver_buffer_slots(
                        num_remote_receiver_buffer_slots,
                        this->dateline_upstream_receiver_channel_skip_idx,
                        dateline_upstream_num_receiver_buffer_slots);
                }
                break;
            default: break;
        }
    } else if (topology == Topology::Torus) {
        // TODO: only handing default and dateline config for now, need to handle other edm types as well
        size_t default_num_sender_buffer_slots;
        size_t default_num_receiver_buffer_slots;
        // get the default buffer slots
        get_optimal_num_slots(
            torus_buffer_slot_options[arch_index],
            this->num_used_sender_channels,
            this->num_used_receiver_channels,
            default_num_sender_buffer_slots,
            default_num_receiver_buffer_slots);

        // get the dateline buffer slots
        size_t dateline_num_sender_buffer_slots;
        size_t dateline_num_receiver_buffer_slots;
        get_optimal_num_slots(
            torus_buffer_slot_options_dateline[arch_index][axis_index],
            this->num_used_sender_channels - 1,
            this->num_used_receiver_channels - 1,
            dateline_num_sender_buffer_slots,
            dateline_num_receiver_buffer_slots,
            default_num_sender_buffer_slots);

        // set default buffer slots.
        num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
        num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);

        auto buffer_config = options.edm_buffer_config;
        if (options.edm_type == FabricEriscDatamoverType::Dateline) {
            if (buffer_config.enable_dateline_sender_extra_buffer_slots) {
                // set num_sender_buffer_slots
                fill_sender_buffer_slots(
                    num_sender_buffer_slots,
                    this->dateline_sender_channel_skip_idx_2d,
                    default_num_sender_buffer_slots,
                    dateline_num_sender_buffer_slots);
                // set remote sender buffer slots equal to local sender, since remote is also dateline
                num_remote_sender_buffer_slots = num_sender_buffer_slots;
            }
            if (buffer_config.enable_dateline_receiver_extra_buffer_slots) {
                // set num_receiver_buffer_slots
                fill_receiver_buffer_slots(
                    num_receiver_buffer_slots,
                    this->dateline_receiver_channel_skip_idx,
                    dateline_num_receiver_buffer_slots);
                // set remote receiver buffer slots equal to local receiver, since remote is also dateline
                num_remote_receiver_buffer_slots = num_receiver_buffer_slots;
            }
        }
    } else {
        size_t default_num_sender_buffer_slots;
        size_t default_num_receiver_buffer_slots;
        get_optimal_num_slots(
            get_num_buffer_slots(topology, arch_index),
            this->num_used_sender_channels,
            this->num_used_receiver_channels,
            default_num_sender_buffer_slots,
            default_num_receiver_buffer_slots);
        // set default buffer slots.
        num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
        num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
    }
}

void FabricStaticSizedChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args, size_t num_fwd_paths, size_t num_used_sender_channels, size_t num_used_receiver_channels) const {
    // NOTE: Special tag 0xabcd1234 is now emitted by MultiPoolChannelAllocator, not here
    // insert the sender channel num buffers - EMIT ALL channels for multi-pool support

    for (size_t i = 0; i < this->num_used_sender_channels; ++i) {
        ct_args.push_back(static_cast<uint32_t>(this->sender_channels_base_address[i]));
        ct_args.push_back(this->sender_channels_num_buffers[i]);
        ct_args.push_back(static_cast<uint32_t>(this->remote_sender_channels_base_address[i]));
        ct_args.push_back(this->remote_sender_channels_num_buffers[i]);
    }
    for (size_t i = 0; i < this->num_used_receiver_channels; ++i) {
        ct_args.push_back(static_cast<uint32_t>(this->receiver_channels_base_address[i]));
        ct_args.push_back(this->receiver_channels_num_buffers[i]);
        ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_base_address[i]));
        ct_args.push_back(this->remote_receiver_channels_num_buffers[i]);
    }
    // ct_args.insert(
    //     ct_args.end(),
    //     this->sender_channels_num_buffers.begin(),
    //     this->sender_channels_num_buffers.end());
    // insert the receiver channel num buffers - EMIT ALL channels for multi-pool support
    // ct_args.insert(
    //     ct_args.end(),
    //     this->receiver_channels_num_buffers.begin(),
    //     this->receiver_channels_num_buffers.end());
    // // insert the remote receiver channel num buffers - EMIT ALL channels for multi-pool support
    // ct_args.insert(
    //     ct_args.end(),
    //     this->remote_receiver_channels_num_buffers.begin(),
    //     this->remote_receiver_channels_num_buffers.end());

    // // Add sender and receiver channel base addresses
    // for (size_t i = 0; i < builder_config::num_sender_channels; ++i) {
    //     if (i < this->sender_channels_base_address.size()) {
    //         ct_args.push_back(static_cast<uint32_t>(this->sender_channels_base_address[i]));
    //     } else {
    //         ct_args.push_back(0);
    //     }
    // }

    // // Add receiver channel base addresses (local and remote interleaved)
    // for (size_t i = 0; i < builder_config::num_receiver_channels; ++i) {
    //     if (i < this->receiver_channels_base_address.size()) {
    //         ct_args.push_back(static_cast<uint32_t>(this->receiver_channels_base_address[i]));
    //     } else {
    //         ct_args.push_back(0);
    //     }
    //     if (i < this->remote_receiver_channels_base_address.size()) {
    //         ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_base_address[i]));
    //     } else {
    //         ct_args.push_back(0);
    //     }
    // }

    // Add remote sender channel base addresses
    // for (size_t i = 0; i < builder_config::num_sender_channels; ++i) {
    //     if (i < this->remote_sender_channels_base_address.size()) {
    //         ct_args.push_back(static_cast<uint32_t>(this->remote_sender_channels_base_address[i]));
    //     } else {
    //         ct_args.push_back(0);
    //     }
    // }
}

};  // namespace tt::tt_fabric
