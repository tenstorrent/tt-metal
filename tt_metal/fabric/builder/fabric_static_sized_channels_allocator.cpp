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
    std::array<size_t, builder_config::num_max_sender_channels> num_sender_buffer_slots = {0};
    std::array<size_t, builder_config::num_max_sender_channels> num_remote_sender_buffer_slots = {0};
    std::array<size_t, builder_config::num_max_receiver_channels> num_receiver_buffer_slots = {0};
    std::array<size_t, builder_config::num_max_receiver_channels> num_remote_receiver_buffer_slots = {0};

    bool has_tensix_extension = options.fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED;

    configure_buffer_slots_helper(
        topology,
        options,
        num_sender_buffer_slots,
        num_remote_sender_buffer_slots,
        num_receiver_buffer_slots,
        num_remote_receiver_buffer_slots);

    log_trace(tt::LogFabric, "num_sender_buffer_slots: {}", num_sender_buffer_slots);
    log_trace(tt::LogFabric, "num_remote_sender_buffer_slots: {}", num_remote_sender_buffer_slots);
    log_trace(tt::LogFabric, "num_receiver_buffer_slots: {}", num_receiver_buffer_slots);
    log_trace(tt::LogFabric, "num_remote_receiver_buffer_slots: {}", num_remote_receiver_buffer_slots);

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

    log_trace(tt::LogFabric, "Available channel buffering space: {}", this->available_channel_buffering_space);

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
        log_trace(tt::LogFabric, "Sender {} channel_start: {}", i, this->sender_channels_base_address[i]);
    }
    uint32_t receiver_buffer_addr = sender_buffer_addr;
    for (uint32_t i = 0; i < num_used_receiver_channels; i++) {
        this->receiver_channels_base_address[i] = receiver_buffer_addr;
        receiver_buffer_addr += this->receiver_channels_size_bytes[i];
        log_trace(tt::LogFabric, "Receiver {} channel_start: {}", i, this->receiver_channels_base_address[i]);
    }
    uint32_t buffer_addr_end = receiver_buffer_addr;
    // set the base addresses for the remote channels
    uint32_t remote_sender_buffer_addr = buffer_region_start;
    for (uint32_t i = 0; i < num_used_sender_channels; i++) {
        this->remote_sender_channels_base_address[i] = remote_sender_buffer_addr;
        remote_sender_buffer_addr += this->remote_sender_channels_size_bytes[i];
        log_trace(tt::LogFabric, "Remote Sender {} channel_start: {}", i, this->remote_sender_channels_base_address[i]);
    }
    uint32_t remote_receiver_buffer_addr = remote_sender_buffer_addr;
    for (uint32_t i = 0; i < num_used_receiver_channels; i++) {
        this->remote_receiver_channels_base_address[i] = remote_receiver_buffer_addr;
        remote_receiver_buffer_addr += this->remote_receiver_channels_size_bytes[i];
        log_trace(tt::LogFabric, "Remote Receiver {} channel_start: {}", i, this->remote_receiver_channels_base_address[i]);
    }

    log_trace(tt::LogFabric, "Available channel buffering space: {}", this->available_channel_buffering_space);

    auto skip_current_sender_channel = [&](uint32_t idx) -> bool {
        // for fabric with tensix extension, only check the vc1 sender channel and worker channel, other
        // channels are skipped
        uint32_t target_channel = get_worker_connected_sender_channel();
        return (idx != target_channel && has_tensix_extension);
    };

    for (uint32_t i = 0; i < num_used_sender_channels; i++) {
        if (!skip_current_sender_channel(i)) {
            TT_FATAL(
                this->sender_channels_size_bytes[i] > 0,
                "Internal error when computing `sender_channels_size_bytes[{}]` which was computed to be size 0",
                i);
        }
    }

    for (uint32_t i = 0; i < num_used_receiver_channels; i++) {
        TT_FATAL(
            this->receiver_channels_size_bytes[i] > 0,
            "Internal error when computing `receiver_channels_size_bytes[{}]` which was computed to be size 0",
            i);
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
    std::array<size_t, builder_config::num_max_sender_channels>& num_sender_buffer_slots,
    std::array<size_t, builder_config::num_max_sender_channels>& num_remote_sender_buffer_slots,
    std::array<size_t, builder_config::num_max_receiver_channels>& num_receiver_buffer_slots,
    std::array<size_t, builder_config::num_max_receiver_channels>& num_remote_receiver_buffer_slots) {
    // fabric with tensix extension uses different buffer slots options, since only one or two sender channels are
    // used by fabric router, while other sender channels are skipped and have 0 buffer slots.
    static const std::vector<std::vector<std::pair<size_t, size_t>>> default_with_tensix_buffer_slot_options = {
        {{16, 16}, {8, 16}, {8, 8}},                     // WORMHOLE_B0: {sender_slots, receiver_slots}
        {{32, 32}, {16, 32}, {16, 16}, {8, 16}, {8, 8}}  // BLACKHOLE: {sender_slots, receiver_slots}
    };

    auto get_num_buffer_slots = [](Topology topology,
                                   size_t arch_index) -> const std::vector<std::pair<size_t, size_t>>& {
        // Architecture-specific buffer slot configurations
        static const std::vector<std::vector<std::pair<size_t, size_t>>> mesh_buffer_slot_options = {
            {{7, 11}, {4, 8}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
            {{8, 16}, {8, 8}, {4, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
        };
        static const std::vector<std::vector<std::pair<size_t, size_t>>> other_buffer_slot_options = {
            {{8, 16}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
            {{32, 32}, {16, 32}, {16, 16}, {8, 16}, {8, 8}, {4, 8}}  // BLACKHOLE: {sender_slots,
        };

        static tt::stl::Indestructible<std::vector<std::vector<std::pair<size_t, size_t>>>> mesh_slots(
            mesh_buffer_slot_options);
        static tt::stl::Indestructible<std::vector<std::vector<std::pair<size_t, size_t>>>> other_slots(
            other_buffer_slot_options);

        if (topology == Topology::Mesh || topology == Topology::Torus) {
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

    // auto axis_index = static_cast<std::size_t>(options.edm_axis);
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
            uint32_t target_channel = get_worker_connected_sender_channel();
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
            num_remote_sender_buffer_slots[target_channel] = default_num_sender_buffer_slots;
            num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
            num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
            return;
        }
        default: break;
    }

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

void FabricStaticSizedChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args) const {
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
}

};  // namespace tt::tt_fabric
