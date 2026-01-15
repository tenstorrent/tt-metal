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

size_t FabricStaticSizedChannelsAllocator::get_sender_channel_base_address(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < sender_channels_base_address[vc_id].size(),
        "Sender channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return sender_channels_base_address[vc_id][channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_sender_channel_number_of_slots(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < sender_channels_num_buffers[vc_id].size(),
        "Sender channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return sender_channels_num_buffers[vc_id][channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_receiver_channel_number_of_slots(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < receiver_channels_num_buffers[vc_id].size(),
        "Receiver channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return receiver_channels_num_buffers[vc_id][channel_id];
}

size_t FabricStaticSizedChannelsAllocator::get_receiver_channel_base_address(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < receiver_channels_base_address[vc_id].size(),
        "Receiver channel ID {} out of bounds for VC{}",
        channel_id,
        vc_id);
    return receiver_channels_base_address[vc_id][channel_id];
}

FabricStaticSizedChannelsAllocator::FabricStaticSizedChannelsAllocator(
    tt::tt_fabric::Topology topology,
    const FabricEriscDatamoverOptions& options,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_sender_channels_per_vc,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& num_used_receiver_channels_per_vc,
    size_t channel_buffer_size_bytes,
    size_t available_channel_buffering_space,
    const std::vector<MemoryRegion>& memory_regions) :
    FabricChannelAllocator(topology, options, memory_regions),
    num_used_sender_channels_per_vc(num_used_sender_channels_per_vc),
    num_used_receiver_channels_per_vc(num_used_receiver_channels_per_vc),
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

    // Per-VC buffer slot arrays
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        num_sender_buffer_slots_per_vc = {};
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        num_remote_sender_buffer_slots_per_vc = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        num_receiver_buffer_slots_per_vc = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        num_remote_receiver_buffer_slots_per_vc = {};

    bool has_tensix_extension = options.fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED;

    configure_buffer_slots_helper(
        topology,
        options,
        num_sender_buffer_slots_per_vc,
        num_remote_sender_buffer_slots_per_vc,
        num_receiver_buffer_slots_per_vc,
        num_remote_receiver_buffer_slots_per_vc);

    // Calculate total slots across all VCs
    size_t total_slot_count = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        size_t vc_sender_slots = std::accumulate(
            num_sender_buffer_slots_per_vc[vc].begin(),
            num_sender_buffer_slots_per_vc[vc].begin() + num_used_sender_channels_per_vc[vc],
            size_t{0});
        size_t vc_receiver_slots = std::accumulate(
            num_receiver_buffer_slots_per_vc[vc].begin(),
            num_receiver_buffer_slots_per_vc[vc].begin() + num_used_receiver_channels_per_vc[vc],
            size_t{0});
        total_slot_count += vc_sender_slots + vc_receiver_slots;

        log_trace(tt::LogFabric, "VC{} num_sender_buffer_slots: {}", vc, num_sender_buffer_slots_per_vc[vc]);
        log_trace(
            tt::LogFabric, "VC{} num_remote_sender_buffer_slots: {}", vc, num_remote_sender_buffer_slots_per_vc[vc]);
        log_trace(tt::LogFabric, "VC{} num_receiver_buffer_slots: {}", vc, num_receiver_buffer_slots_per_vc[vc]);
        log_trace(
            tt::LogFabric,
            "VC{} num_remote_receiver_buffer_slots: {}",
            vc,
            num_remote_receiver_buffer_slots_per_vc[vc]);
    }
    TT_FATAL(
        total_slot_count * channel_buffer_size_bytes <= available_channel_buffering_space,
        "Total channel size of {} B exceeds available space of {} B",
        total_slot_count * channel_buffer_size_bytes,
        available_channel_buffering_space);

    log_trace(tt::LogFabric, "Available channel buffering space: {}", this->available_channel_buffering_space);

    // Set channel sizes and num buffers per VC
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        // set the sender channel sizes and num buffers
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            this->sender_channels_size_bytes[vc][i] = channel_buffer_size_bytes * num_sender_buffer_slots_per_vc[vc][i];
            this->sender_channels_num_buffers[vc][i] = num_sender_buffer_slots_per_vc[vc][i];
        }
        // set the remote sender channel sizes and num buffers
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            this->remote_sender_channels_size_bytes[vc][i] =
                channel_buffer_size_bytes * num_remote_sender_buffer_slots_per_vc[vc][i];
            this->remote_sender_channels_num_buffers[vc][i] = num_remote_sender_buffer_slots_per_vc[vc][i];
        }
        // set the local receiver channel sizes and num buffers
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            this->receiver_channels_size_bytes[vc][i] =
                channel_buffer_size_bytes * num_receiver_buffer_slots_per_vc[vc][i];
            this->receiver_channels_num_buffers[vc][i] = num_receiver_buffer_slots_per_vc[vc][i];
        }
        // set the remote receiver channel sizes and num buffers
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            this->remote_receiver_channels_size_bytes[vc][i] =
                channel_buffer_size_bytes * num_remote_receiver_buffer_slots_per_vc[vc][i];
            this->remote_receiver_channels_num_buffers[vc][i] = num_remote_receiver_buffer_slots_per_vc[vc][i];
        }
    }

    // set the base addresses for the local channels (allocate sequentially: VC0 channels, then VC1 channels)
    uint32_t sender_buffer_addr = buffer_region_start;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            this->sender_channels_base_address[vc][i] = sender_buffer_addr;
            sender_buffer_addr += this->sender_channels_size_bytes[vc][i];
            log_trace(
                tt::LogFabric, "VC{} Sender {} channel_start: {}", vc, i, this->sender_channels_base_address[vc][i]);
        }
    }

    uint32_t receiver_buffer_addr = sender_buffer_addr;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            this->receiver_channels_base_address[vc][i] = receiver_buffer_addr;
            receiver_buffer_addr += this->receiver_channels_size_bytes[vc][i];
            log_trace(
                tt::LogFabric,
                "VC{} Receiver {} channel_start: {}",
                vc,
                i,
                this->receiver_channels_base_address[vc][i]);
        }
    }
    uint32_t buffer_addr_end = receiver_buffer_addr;

    // set the base addresses for the remote channels
    uint32_t remote_sender_buffer_addr = buffer_region_start;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            this->remote_sender_channels_base_address[vc][i] = remote_sender_buffer_addr;
            remote_sender_buffer_addr += this->remote_sender_channels_size_bytes[vc][i];
            log_trace(
                tt::LogFabric,
                "VC{} Remote Sender {} channel_start: {}",
                vc,
                i,
                this->remote_sender_channels_base_address[vc][i]);
        }
    }

    uint32_t remote_receiver_buffer_addr = remote_sender_buffer_addr;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            this->remote_receiver_channels_base_address[vc][i] = remote_receiver_buffer_addr;
            remote_receiver_buffer_addr += this->remote_receiver_channels_size_bytes[vc][i];
            log_trace(
                tt::LogFabric,
                "VC{} Remote Receiver {} channel_start: {}",
                vc,
                i,
                this->remote_receiver_channels_base_address[vc][i]);
        }
    }

    log_trace(tt::LogFabric, "Available channel buffering space: {}", this->available_channel_buffering_space);

    auto skip_current_sender_channel = [&](uint32_t vc_id, uint32_t idx) -> bool {
        // for fabric with tensix extension, only check the worker channel (VC0 channel 0), other channels are skipped
        uint32_t target_channel = get_worker_connected_sender_channel();
        return !((vc_id == 0 && idx == target_channel) || !has_tensix_extension);
    };

    // Validate sender channels per VC
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_sender_channels_per_vc[vc]; i++) {
            if (!skip_current_sender_channel(vc, i)) {
                TT_FATAL(
                    this->sender_channels_size_bytes[vc][i] > 0,
                    "Internal error when computing `sender_channels_size_bytes[VC{}][{}]` which was computed to be "
                    "size 0",
                    vc,
                    i);
            }
        }
    }

    // Validate receiver channels per VC
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t i = 0; i < num_used_receiver_channels_per_vc[vc]; i++) {
            TT_FATAL(
                this->receiver_channels_size_bytes[vc][i] > 0,
                "Internal error when computing `receiver_channels_size_bytes[VC{}][{}]` which was computed to be size "
                "0",
                vc,
                i);
        }
    }

    // Validate total size across all VCs
    size_t total_size = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        total_size += std::accumulate(
            this->sender_channels_size_bytes[vc].begin(),
            this->sender_channels_size_bytes[vc].begin() + num_used_sender_channels_per_vc[vc],
            0ul);
        total_size += std::accumulate(
            this->receiver_channels_size_bytes[vc].begin(),
            this->receiver_channels_size_bytes[vc].begin() + num_used_receiver_channels_per_vc[vc],
            0ul);
    }

    TT_FATAL(
        total_size <= this->available_channel_buffering_space,
        "Internal error when computing channel sizes. Total channel size {} exceeds available space {}",
        total_size,
        this->available_channel_buffering_space);
    TT_FATAL(
        buffer_addr_end < this->max_l1_loading_size,
        "Internal error - channel buffers spilled past the end of usable L1 region.");
}

void FabricStaticSizedChannelsAllocator::configure_buffer_slots_helper(
    Topology topology,
    const FabricEriscDatamoverOptions& options,
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>&
        num_sender_buffer_slots_per_vc,
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>&
        num_remote_sender_buffer_slots_per_vc,
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>&
        num_receiver_buffer_slots_per_vc,
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>&
        num_remote_receiver_buffer_slots_per_vc) {
    // Per-VC buffer slot configuration: {vc0_sender, vc0_receiver, vc1_sender, vc1_receiver}
    struct PerVcBufferSlots {
        size_t vc0_sender_slots;
        size_t vc0_receiver_slots;
        size_t vc1_sender_slots;
        size_t vc1_receiver_slots;
    };

    // fabric with tensix extension uses different buffer slots options, since only one or two sender channels are
    // used by fabric router, while other sender channels are skipped and have 0 buffer slots.
    // Format: {vc0_sender, vc0_receiver, vc1_sender, vc1_receiver}
    static const std::vector<std::vector<PerVcBufferSlots>> default_with_tensix_buffer_slot_options = {
        // WORMHOLE_B0
        {
            {16, 16, 0, 0},  // Option 1
            {8, 16, 0, 0},   // Option 2
            {8, 8, 0, 0},    // Option 3
            {4, 8, 0, 0},    // Option 4
            {4, 4, 0, 0},    // Option 5: VC0 only, smaller
            {2, 4, 0, 0},    // Option 6: VC0 only, smaller
            {2, 2, 0, 0},    // Option 7: VC0 only, smaller
            {1, 2, 0, 0},    // Option 8: VC0 only, smallest
            {1, 1, 0, 0},    // Option 9: VC0 only, smallest
            {4, 8, 4, 4},    // Option 10: supports both VCs
            {4, 8, 2, 2},    // Option 11: supports both VCs
            {4, 4, 2, 2},    // Option 12: supports both VCs
            {2, 2, 2, 2}     // Option 13: supports both VCs
        },
        // BLACKHOLE
        {
            {32, 32, 0, 0},  // Option 1
            {16, 32, 0, 0},  // Option 2
            {16, 16, 0, 0},  // Option 3
            {8, 16, 0, 0},   // Option 4
            {8, 8, 0, 0},    // Option 5
            {4, 8, 0, 0},    // Option 6
            {4, 4, 0, 0},    // Option 7
            {2, 4, 0, 0},    // Option 8
            {2, 2, 0, 0},    // Option 9
            {1, 2, 0, 0},    // Option 10
            {1, 1, 0, 0}     // Option 11
        }};

    auto get_num_buffer_slots = [](Topology topology, size_t arch_index) -> const std::vector<PerVcBufferSlots>& {
        // Architecture-specific buffer slot configurations per VC
        // Format: {vc0_sender, vc0_receiver, vc1_sender, vc1_receiver}
        static const std::vector<std::vector<PerVcBufferSlots>> mesh_buffer_slot_options = {
            // WORMHOLE_B0
            {
                {7, 11, 0, 0},  // Option 1: VC0 only
                {4, 8, 0, 0},   // Option 2: VC0 only, smaller
                {4, 4, 0, 0},   // Option 3: VC0 only, smaller
                {2, 4, 0, 0},   // Option 4: VC0 only, smaller
                {2, 2, 0, 0},   // Option 5: VC0 only, smaller
                {1, 2, 0, 0},   // Option 6: VC0 only, smallest
                {1, 1, 0, 0},   // Option 7: VC0 only, smallest
                {4, 8, 2, 4},   // Option 8: supports both VCs
                {4, 8, 2, 2},   // Option 9: supports both VCs, smaller VC1 receiver
                {2, 4, 2, 2},   // Option 10: supports both VCs, smaller overall
                {2, 4, 1, 1},   // Option 11: supports both VCs, smaller overall
                {2, 2, 1, 1},   // Option 12: supports both VCs, smaller overall
                {1, 1, 1, 1}    // Option 13: supports both VCs, smaller overall
            },
            // BLACKHOLE
            {
                {8, 16, 0, 0},  // Option 1: VC0 only
                {8, 8, 0, 0},   // Option 2: VC0 only
                {4, 8, 0, 0},   // Option 3: VC0 only
                {4, 4, 0, 0},   // Option 4: VC0 only, smaller
                {2, 4, 0, 0},   // Option 5: VC0 only, smaller
                {2, 2, 0, 0},   // Option 6: VC0 only, smaller
                {1, 2, 0, 0},   // Option 7: VC0 only, smallest
                {1, 1, 0, 0},   // Option 8: VC0 only, smallest
                {4, 8, 2, 4},   // Option 9: supports both VCs
                {4, 8, 2, 2},   // Option 10: supports both VCs, smaller VC1 receiver
                {2, 4, 2, 2},   // Option 11: supports both VCs, smaller overall
                {2, 4, 1, 1},   // Option 12: supports both VCs, smaller overall
                {2, 2, 1, 1},   // Option 13: supports both VCs, smaller overall
                {1, 1, 1, 1}    // Option 14: supports both VCs, smaller overall
            }};
        static const std::vector<std::vector<PerVcBufferSlots>> other_buffer_slot_options = {
            // WORMHOLE_B0
            {{16, 16, 0, 0},  // Only VC0 for non-mesh topologies.
             {8, 16, 0, 0},
             {8, 8, 0, 0},
             {4, 8, 0, 0},
             {4, 4, 0, 0},
             {2, 4, 0, 0},
             {2, 2, 0, 0},
             {1, 2, 0, 0},
             {1, 1, 0, 0}},
            // BLACKHOLE
            {{32, 32, 0, 0},  // Only VC0 for non-mesh topologies.
             {16, 32, 0, 0},
             {16, 16, 0, 0},
             {8, 16, 0, 0},
             {8, 8, 0, 0},
             {4, 8, 0, 0},
             {4, 4, 0, 0},
             {2, 4, 0, 0},
             {2, 2, 0, 0},
             {1, 2, 0, 0},
             {1, 1, 0, 0}}};

        static tt::stl::Indestructible<std::vector<std::vector<PerVcBufferSlots>>> mesh_slots(mesh_buffer_slot_options);
        static tt::stl::Indestructible<std::vector<std::vector<PerVcBufferSlots>>> other_slots(
            other_buffer_slot_options);

        if (topology == Topology::Mesh || topology == Topology::Torus) {
            return mesh_slots.get()[arch_index];
        }
        return other_slots.get()[arch_index];
    };

    auto get_optimal_num_slots_per_vc = [this](
                                            auto& buffer_slot_options,
                                            size_t num_vc0_sender_channels,
                                            size_t num_vc0_receiver_channels,
                                            size_t num_vc1_sender_channels,
                                            size_t num_vc1_receiver_channels,
                                            size_t& vc0_sender_buffer_slots,
                                            size_t& vc0_receiver_buffer_slots,
                                            size_t& vc1_sender_buffer_slots,
                                            size_t& vc1_receiver_buffer_slots) {
        bool vc1_needed = (num_vc1_sender_channels > 0) || (num_vc1_receiver_channels > 0);
        bool found_valid_option = false;
        for (auto& option : buffer_slot_options) {
            vc0_sender_buffer_slots = option.vc0_sender_slots;
            vc0_receiver_buffer_slots = option.vc0_receiver_slots;
            vc1_sender_buffer_slots = option.vc1_sender_slots;
            vc1_receiver_buffer_slots = option.vc1_receiver_slots;
            // skip the VC0 only options if VC1 is needed (either sender or receiver channels)
            if (vc1_needed) {
                // Check if we need VC1 sender channels but this option doesn't provide them
                bool skip_due_to_vc1_sender = (num_vc1_sender_channels > 0) && (vc1_sender_buffer_slots == 0);
                // Check if we need VC1 receiver channels but this option doesn't provide them
                bool skip_due_to_vc1_receiver = (num_vc1_receiver_channels > 0) && (vc1_receiver_buffer_slots == 0);

                if (skip_due_to_vc1_sender || skip_due_to_vc1_receiver) {
                    continue;  // Skip this option - VC1 is needed but this option doesn't support it
                }
            }

            // Calculate total slots across both VCs
            auto vc0_total_sender_slots = num_vc0_sender_channels * vc0_sender_buffer_slots;
            auto vc0_total_receiver_slots = num_vc0_receiver_channels * vc0_receiver_buffer_slots;
            auto vc1_total_sender_slots = num_vc1_sender_channels * vc1_sender_buffer_slots;
            auto vc1_total_receiver_slots = num_vc1_receiver_channels * vc1_receiver_buffer_slots;

            auto total_num_bytes = (vc0_total_sender_slots + vc0_total_receiver_slots + vc1_total_sender_slots +
                                    vc1_total_receiver_slots) *
                                   this->channel_buffer_size_bytes;

            if (total_num_bytes <= this->available_channel_buffering_space) {
                found_valid_option = true;
                break;  // Found a configuration that fits
            }
        }

        // Validate that we found a valid option, especially if VC1 is needed
        if (!found_valid_option) {
            TT_THROW(
                "Failed to find suitable buffer slot configuration. VC1 needed: {}, VC0 channels: {} senders/{} "
                "receivers, VC1 channels: {} senders/{} receivers, Available space: {} bytes",
                vc1_needed,
                num_vc0_sender_channels,
                num_vc0_receiver_channels,
                num_vc1_sender_channels,
                num_vc1_receiver_channels,
                this->available_channel_buffering_space);
        }

        // Additional validation: ensure VC1 buffer slots are non-zero if VC1 channels are needed
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
            // MUX mode: Only VC0 channel 0 is used for worker
            uint32_t target_channel = get_worker_connected_sender_channel();
            size_t vc0_sender_buffer_slots, vc0_receiver_buffer_slots;
            size_t vc1_sender_buffer_slots, vc1_receiver_buffer_slots;

            // get the optimal buffer slots for MUX mode (per-VC)
            get_optimal_num_slots_per_vc(
                default_with_tensix_buffer_slot_options[arch_index],
                num_used_sender_channels_per_vc[0],
                num_used_receiver_channels_per_vc[0],
                num_used_sender_channels_per_vc[1],
                num_used_receiver_channels_per_vc[1],
                vc0_sender_buffer_slots,
                vc0_receiver_buffer_slots,
                vc1_sender_buffer_slots,
                vc1_receiver_buffer_slots);

            // set buffer slots for VC0 worker channel only
            num_sender_buffer_slots_per_vc[0][target_channel] = vc0_sender_buffer_slots;
            num_remote_sender_buffer_slots_per_vc[0][target_channel] = vc0_sender_buffer_slots;

            // Fill receiver buffer slots for both VCs
            num_receiver_buffer_slots_per_vc[0].fill(vc0_receiver_buffer_slots);
            num_remote_receiver_buffer_slots_per_vc[0].fill(vc0_receiver_buffer_slots);
            num_receiver_buffer_slots_per_vc[1].fill(vc1_receiver_buffer_slots);
            num_remote_receiver_buffer_slots_per_vc[1].fill(vc1_receiver_buffer_slots);
            return;
        }
        default: break;
    }

    // Default case: Configure buffer slots with per-VC options
    size_t vc0_sender_buffer_slots, vc0_receiver_buffer_slots;
    size_t vc1_sender_buffer_slots, vc1_receiver_buffer_slots;

    // Get optimal buffer slots considering both VCs
    get_optimal_num_slots_per_vc(
        get_num_buffer_slots(topology, arch_index),
        num_used_sender_channels_per_vc[0],
        num_used_receiver_channels_per_vc[0],
        num_used_sender_channels_per_vc[1],
        num_used_receiver_channels_per_vc[1],
        vc0_sender_buffer_slots,
        vc0_receiver_buffer_slots,
        vc1_sender_buffer_slots,
        vc1_receiver_buffer_slots);

    // Apply the buffer slot configuration to each VC
    num_sender_buffer_slots_per_vc[0].fill(vc0_sender_buffer_slots);
    num_remote_sender_buffer_slots_per_vc[0].fill(vc0_sender_buffer_slots);
    num_receiver_buffer_slots_per_vc[0].fill(vc0_receiver_buffer_slots);
    num_remote_receiver_buffer_slots_per_vc[0].fill(vc0_receiver_buffer_slots);

    num_sender_buffer_slots_per_vc[1].fill(vc1_sender_buffer_slots);
    num_remote_sender_buffer_slots_per_vc[1].fill(vc1_sender_buffer_slots);
    num_receiver_buffer_slots_per_vc[1].fill(vc1_receiver_buffer_slots);
    num_remote_receiver_buffer_slots_per_vc[1].fill(vc1_receiver_buffer_slots);
}

void FabricStaticSizedChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args) const {
    // Emit sender channel args for all VCs
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (size_t i = 0; i < this->num_used_sender_channels_per_vc[vc]; ++i) {
            ct_args.push_back(static_cast<uint32_t>(this->sender_channels_base_address[vc][i]));
            ct_args.push_back(this->sender_channels_num_buffers[vc][i]);
            ct_args.push_back(static_cast<uint32_t>(this->remote_sender_channels_base_address[vc][i]));
            ct_args.push_back(this->remote_sender_channels_num_buffers[vc][i]);
        }
    }

    // Emit receiver channel args for all VCs
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (size_t i = 0; i < this->num_used_receiver_channels_per_vc[vc]; ++i) {
            ct_args.push_back(static_cast<uint32_t>(this->receiver_channels_base_address[vc][i]));
            ct_args.push_back(this->receiver_channels_num_buffers[vc][i]);
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_base_address[vc][i]));
            ct_args.push_back(this->remote_receiver_channels_num_buffers[vc][i]);
        }
    }
}

};  // namespace tt::tt_fabric
